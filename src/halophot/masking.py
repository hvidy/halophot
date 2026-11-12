"""Spatial halo masking, sky-dominance guards, and WCS contaminant-star exclusion.

Implements Section 5.2 and Section 6.1 aperture selection rules:

* Automatic exclusion of ``PIXEL_QUALITY`` bits ``STRAP_COL | SAT_BLEED |
  SMEAR_UNRESOLVED | SOURCE_MASK | BKG_USED`` so ``Omega_halo`` and ``Omega_bg``
  are strictly disjoint (``Omega_halo cap Omega_bg = empty``).
* 2D saturation + bleed + LDE-undershoot dilation via ``scipy.ndimage.binary_dilation``.
* Sky-dominance guard ``P_bar_i >= kappa * median_t(B_{i,t})`` (``kappa = 2.0``
  default when ``FLUX_BKG`` is non-zero and ``not legacy_tv``) and pixel
  ``SNR_i >= 5`` preventing TV minimization from diluting astrophysical
  amplitudes on sky-dominated pixels.
* WCS-projected circular contaminant masking from a pre-cached Astropy ``Table``
  (for ``--offline`` HPC execution) or live ``astroquery.mast`` queries.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from ._contract import PixelQuality

__all__ = [
    "DEFAULT_EXCLUDE_QUALITY",
    "HaloMaskResult",
    "_resolve_tpf_wcs",
    "build_halo_masks",
    "mask_background_stars",
    "remove_stars",
]

DEFAULT_EXCLUDE_QUALITY: PixelQuality = (
    PixelQuality.STRAP_COL
    | PixelQuality.SAT_BLEED
    | PixelQuality.SMEAR_UNRESOLVED
    | PixelQuality.SOURCE_MASK
    | PixelQuality.BKG_USED
)


@dataclass(frozen=True)
class HaloMaskResult:
    """Spatial and temporal masks constructed for halo weight optimization."""

    pixel_mask: np.ndarray
    cadence_mask: np.ndarray
    saturated_mask: np.ndarray
    quality_excluded_mask: np.ndarray
    sky_excluded_mask: np.ndarray
    snr_excluded_mask: np.ndarray
    source_excluded_mask: np.ndarray


def _load_catalog_table(catalog_table: Table | str | os.PathLike[str]) -> Table:
    """Load an Astropy ``Table`` from an in-memory instance or local disk path."""
    if isinstance(catalog_table, Table):
        return catalog_table
    path = Path(catalog_table)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return Table.read(path, format="parquet")
    if suffix in (".fits", ".fit", ".fts"):
        return Table.read(path, format="fits")
    if suffix == ".ecsv":
        return Table.read(path, format="ascii.ecsv")
    return Table.read(path)


def _find_column_case_insensitive(table: Table, candidates: tuple[str, ...]) -> str | None:
    lower_map = {col.lower(): col for col in table.colnames}
    for cand in candidates:
        found = lower_map.get(cand.lower())
        if found is not None:
            return found
    return None


def _query_mast_region(
    center: SkyCoord,
    radius_deg: float,
    *,
    catalog: str = "TIC",
) -> Table:
    """Query MAST ``Catalogs.query_region`` when ``offline=False``."""
    from astroquery.mast import Catalogs

    return Catalogs.query_region(center, radius=radius_deg * u.deg, catalog=catalog)


def mask_background_stars(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_exclusion_arcsec: float = 5.0,
    mag_limit: float = 17.5,
    radius_pix: float | None = None,
    catalog: str = "TIC",
) -> np.ndarray:
    """Construct a ``(ny, nx)`` boolean mask of background stars projected via ``wcs``.

    Returns ``True`` on pixels contaminated by field stars (excluding the primary
    target within ``target_exclusion_arcsec`` of ``target_coord`` when provided).
    When ``offline=True``, ``catalog_table`` must be supplied and zero network
    calls are made.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"Invalid stamp shape {shape}; expected positive (ny, nx).")
    if wcs is None or not wcs.has_celestial:
        raise ValueError("A valid celestial astropy.wcs.WCS is required to project star masks.")

    if catalog_table is None:
        if offline:
            raise RuntimeError(
                "Live catalogue queries are disabled when offline=True; "
                "pass a pre-cached Astropy Table via catalog_table."
            )
        corners_x = np.array([-0.5, nx - 0.5, -0.5, nx - 0.5], dtype=np.float64)
        corners_y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=np.float64)
        center_sky = wcs.celestial.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        corner_sky = wcs.celestial.pixel_to_world(corners_x, corners_y)
        sep_deg = float(np.max(center_sky.separation(corner_sky).deg)) * 1.15 + (10.0 / 3600.0)
        tbl = _query_mast_region(center_sky, sep_deg, catalog=catalog)
    else:
        tbl = _load_catalog_table(catalog_table)

    out_mask = np.zeros((ny, nx), dtype=bool)
    if len(tbl) == 0:
        return out_mask

    ra_col = _find_column_case_insensitive(tbl, ("ra", "raj2000", "ra_deg", "ra_icrs"))
    dec_col = _find_column_case_insensitive(tbl, ("dec", "dej2000", "dec_deg", "dec_icrs"))
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Catalogue table with columns {tbl.colnames} lacks recognizable RA/Dec columns."
        )

    ra_vals = np.asarray(tbl[ra_col], dtype=np.float64)
    dec_vals = np.asarray(tbl[dec_col], dtype=np.float64)
    valid = np.isfinite(ra_vals) & np.isfinite(dec_vals)

    mag_col = _find_column_case_insensitive(
        tbl,
        ("tmag", "phot_g_mean_mag", "kepmag", "mag", "vmag", "gmag"),
    )
    mag_vals: np.ndarray | None = None
    if mag_col is not None:
        mag_vals = np.asarray(tbl[mag_col], dtype=np.float64)
        # Retain rows with finite mag <= mag_limit or missing NaN mag.
        valid &= np.isnan(mag_vals) | (mag_vals <= float(mag_limit))

    if not np.any(valid):
        return out_mask

    ra_vals = ra_vals[valid]
    dec_vals = dec_vals[valid]
    if mag_vals is not None:
        mag_vals = mag_vals[valid]

    src_coords = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg, frame="icrs")

    if target_coord is not None and target_exclusion_arcsec > 0.0:
        if isinstance(target_coord, SkyCoord):
            tgt_sky = target_coord.icrs
        else:
            t_ra, t_dec = float(target_coord[0]), float(target_coord[1])
            tgt_sky = (
                SkyCoord(ra=t_ra * u.deg, dec=t_dec * u.deg, frame="icrs")
                if np.isfinite(t_ra) and np.isfinite(t_dec)
                else None
            )
        if tgt_sky is not None:
            not_target = src_coords.separation(tgt_sky).arcsec > float(target_exclusion_arcsec)
            src_coords = src_coords[not_target]
            if mag_vals is not None:
                mag_vals = mag_vals[not_target]

    if len(src_coords) == 0:
        return out_mask

    col_pix, row_pix = wcs.celestial.world_to_pixel(src_coords)
    col_pix = np.atleast_1d(np.asarray(col_pix, dtype=np.float64))
    row_pix = np.atleast_1d(np.asarray(row_pix, dtype=np.float64))

    if radius_pix is not None:
        if radius_pix <= 0.0:
            raise ValueError(f"radius_pix must be > 0, got {radius_pix!r}.")
        radii = np.full(col_pix.shape, float(radius_pix), dtype=np.float64)
    elif mag_vals is not None:
        eff_mag = np.where(np.isfinite(mag_vals), mag_vals, 15.0)
        radii = np.clip(1.5 + 0.35 * np.maximum(15.0 - eff_mag, 0.0), 1.25, 4.5)
    else:
        radii = np.full(col_pix.shape, 1.5, dtype=np.float64)

    max_r = float(np.max(radii))
    in_box = (
        np.isfinite(col_pix)
        & np.isfinite(row_pix)
        & (col_pix >= -max_r)
        & (col_pix <= (nx - 1) + max_r)
        & (row_pix >= -max_r)
        & (row_pix <= (ny - 1) + max_r)
    )
    if not np.any(in_box):
        return out_mask

    col_pix = col_pix[in_box]
    row_pix = row_pix[in_box]
    radii = radii[in_box]

    yy, xx = np.mgrid[0:ny, 0:nx]
    for c_s, r_s, rad_s in zip(col_pix, row_pix, radii, strict=True):
        out_mask |= (xx - c_s) ** 2 + (yy - r_s) ** 2 <= rad_s**2

    return out_mask


def _resolve_tpf_wcs(tpf: Any) -> WCS | None:
    """Extract a celestial ``WCS`` from ``tpf.wcs`` or its ``APERTURE`` extension."""
    cand = getattr(tpf, "wcs", None)
    if cand is not None and cand.has_celestial:
        return cand
    hdul = getattr(tpf, "hdu", None)
    if hdul is not None:
        if "APERTURE" in hdul:
            ap_wcs = WCS(hdul["APERTURE"].header)
            if ap_wcs.has_celestial:
                return ap_wcs
        if len(hdul) > 2:
            hdu2_wcs = WCS(hdul[2].header)
            if hdu2_wcs.has_celestial:
                return hdu2_wcs
    return None


def remove_stars(
    tpf: Any,
    *,
    wcs: WCS | None = None,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_exclusion_arcsec: float = 5.0,
    mag_limit: float = 17.5,
    radius_pix: float | None = None,
    catalog: str = "TIC",
) -> Any:
    """Mask background contaminant stars on a ``TargetPixelFile`` or 3D flux array via WCS.

    * Given a ``lightkurve.TargetPixelFile``, returns a non-mutating copy with
      ``FLUX[:, star_mask] = NaN`` and ``PixelQuality.SOURCE_MASK`` set in
      ``PIXEL_QUALITY``.
    * Given a 3D ``(N_cad, ny, nx)`` ``ndarray``, requires ``wcs=...`` and returns
      a copy with ``arr[:, star_mask] = NaN``.
    """
    if hasattr(tpf, "hdu") and hasattr(tpf, "flux"):
        resolved_wcs = wcs if wcs is not None else _resolve_tpf_wcs(tpf)
        if resolved_wcs is None or not resolved_wcs.has_celestial:
            raise ValueError("TargetPixelFile has no valid celestial WCS for remove_stars.")
        resolved_target = target_coord
        if resolved_target is None:
            ra_t = getattr(tpf, "ra", None)
            dec_t = getattr(tpf, "dec", None)
            if ra_t is not None and dec_t is not None:
                resolved_target = (float(ra_t), float(dec_t))

        ny, nx = int(tpf.flux.shape[1]), int(tpf.flux.shape[2])
        star_mask = mask_background_stars(
            resolved_wcs,
            (ny, nx),
            catalog_table=catalog_table,
            offline=offline,
            target_coord=resolved_target,
            target_exclusion_arcsec=target_exclusion_arcsec,
            mag_limit=mag_limit,
            radius_pix=radius_pix,
            catalog=catalog,
        )

        copied_hdus = []
        for idx, hdu in enumerate(tpf.hdu):
            if idx == 1 and isinstance(hdu, fits.BinTableHDU):
                copied_hdus.append(
                    fits.BinTableHDU(
                        data=hdu.data.copy(),
                        header=hdu.header.copy(),
                        name=hdu.name,
                    )
                )
            else:
                copied_hdus.append(hdu.copy())
        hdul_copy = fits.HDUList(copied_hdus)
        flux_raw = np.array(hdul_copy[1].data["FLUX"], copy=True)
        flux_raw[:, star_mask] = np.nan
        hdul_copy[1].data["FLUX"][:] = flux_raw

        if "PIXEL_QUALITY" in hdul_copy:
            pq = np.asarray(hdul_copy["PIXEL_QUALITY"].data, dtype=np.uint16).copy()
        else:
            pq = np.zeros((ny, nx), dtype=np.uint16)
        pq[star_mask] |= np.uint16(PixelQuality.SOURCE_MASK)
        if "PIXEL_QUALITY" in hdul_copy:
            hdul_copy["PIXEL_QUALITY"].data = pq
        else:
            hdul_copy.append(fits.ImageHDU(data=pq, name="PIXEL_QUALITY"))

        new_tpf = copy.copy(tpf)
        new_tpf.hdu = hdul_copy
        return new_tpf

    arr = np.asarray(tpf)
    if arr.ndim != 3:
        raise TypeError(
            "remove_stars expects a lightkurve.TargetPixelFile or 3D (N_cad, ny, nx) array; "
            f"got {type(tpf).__name__} with ndim={arr.ndim}."
        )
    if wcs is None:
        raise ValueError(
            "remove_stars on a raw 3D flux array requires a celestial astropy.wcs.WCS via wcs=..."
        )

    star_mask = mask_background_stars(
        wcs,
        (arr.shape[1], arr.shape[2]),
        catalog_table=catalog_table,
        offline=offline,
        target_coord=target_coord,
        target_exclusion_arcsec=target_exclusion_arcsec,
        mag_limit=mag_limit,
        radius_pix=radius_pix,
        catalog=catalog,
    )
    out = np.array(arr, dtype=np.result_type(arr.dtype, np.float32), copy=True)
    out[:, star_mask] = np.nan
    return out


def build_halo_masks(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    flux_bkg: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    aperture_mask: np.ndarray | None = None,
    cadence_quality_mask: np.ndarray | None = None,
    pos_corr1: np.ndarray | None = None,
    pos_corr2: np.ndarray | None = None,
    wcs: WCS | None = None,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    remove_contaminants: bool = False,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    target_exclusion_arcsec: float = 5.0,
    mag_limit: float = 17.5,
    star_radius_pix: float | None = None,
    thresh: float = -1.0,
    auto_sat_level: float = 1.5e5,
    bleed_dilation: tuple[int, int] = (3, 3),
    minflux: float = -100.0,
    sky_kappa: float = 2.0,
    min_snr: float = 5.0,
    dt_days: float | None = None,
    exclude_quality: PixelQuality = DEFAULT_EXCLUDE_QUALITY,
    legacy_tv: bool = False,
) -> HaloMaskResult:
    """Build active spatial ``Omega_halo`` and cadence ``G`` masks for v2.0 halo photometry."""
    flux_arr = np.asarray(flux, dtype=np.float64)
    if flux_arr.ndim != 3:
        raise ValueError(f"flux must be 3D (N_cad, ny, nx), got shape {flux_arr.shape}.")
    n_cad, ny, nx = flux_arr.shape
    if n_cad < 2:
        raise ValueError(f"At least 2 cadences are required, got N_cad={n_cad}.")

    cad_mask = (
        np.ones(n_cad, dtype=bool)
        if cadence_quality_mask is None
        else np.asarray(cadence_quality_mask, dtype=bool).copy()
    )
    if cad_mask.shape != (n_cad,):
        raise ValueError(
            f"cadence_quality_mask shape {cad_mask.shape} does not match ({n_cad},)."
        )

    # Exclude gross pointing excursions (> 5 px from median) when >= 80% of pos_corr are finite.
    if pos_corr1 is not None and pos_corr2 is not None and np.any(cad_mask):
        pc1 = np.asarray(pos_corr1, dtype=np.float64)
        pc2 = np.asarray(pos_corr2, dtype=np.float64)
        if pc1.shape == (n_cad,) and pc2.shape == (n_cad,):
            fin_pc = np.isfinite(pc1[cad_mask]) & np.isfinite(pc2[cad_mask])
            if np.sum(fin_pc) >= 0.8 * np.sum(cad_mask):
                xc = float(np.nanmedian(pc1[cad_mask]))
                yc = float(np.nanmedian(pc2[cad_mask]))
                rr = np.hypot(pc1[cad_mask] - xc, pc2[cad_mask] - yc)
                good_pos = fin_pc & (rr < 5.0)
                cad_mask[np.flatnonzero(cad_mask)[~good_pos]] = False

    if not np.any(cad_mask):
        raise ValueError("No valid cadences survived quality and pointing filtering.")

    base_mask = (
        np.ones((ny, nx), dtype=bool)
        if aperture_mask is None
        else np.asarray(aperture_mask, dtype=bool).copy()
    )
    if base_mask.shape != (ny, nx):
        raise ValueError(
            f"aperture_mask shape {base_mask.shape} does not match spatial shape {(ny, nx)}."
        )

    if pixel_quality is not None:
        pq = np.asarray(pixel_quality, dtype=np.uint16)
        if pq.shape != (ny, nx):
            raise ValueError(
                f"pixel_quality shape {pq.shape} does not match spatial shape {(ny, nx)}."
            )
        qual_excl = (pq & np.uint16(exclude_quality)) != 0
    else:
        qual_excl = np.zeros((ny, nx), dtype=bool)

    if remove_contaminants or catalog_table is not None:
        if wcs is None or not wcs.has_celestial:
            raise ValueError(
                "Contaminant star masking requested, but no valid celestial WCS was provided."
            )
        source_excl = mask_background_stars(
            wcs,
            (ny, nx),
            catalog_table=catalog_table,
            offline=offline,
            target_coord=target_coord,
            target_exclusion_arcsec=target_exclusion_arcsec,
            mag_limit=mag_limit,
            radius_pix=star_radius_pix,
        )
    else:
        source_excl = np.zeros((ny, nx), dtype=bool)

    flux_cad = flux_arr[cad_mask]
    any_finite_pix = np.any(np.isfinite(flux_cad), axis=0)
    if not np.any(any_finite_pix & base_mask):
        raise ValueError("No finite pixels exist inside aperture_mask on valid cadences.")

    # 2D saturation + vertical bleed + horizontal LDE undershoot dilation.
    structure_2d = np.ones(bleed_dilation, dtype=bool)
    sat_raw = np.zeros((ny, nx), dtype=bool)
    if thresh is not None and thresh >= 0.0:
        if thresh < 1.0:
            maxflux = float(np.nanmax(flux_cad[:, any_finite_pix]))
            sat_raw[any_finite_pix] = (
                np.nanmax(flux_cad[:, any_finite_pix], axis=0) > float(thresh) * maxflux
            )
            sat_mask = ndimage.binary_dilation(sat_raw, structure=structure_2d)
        else:
            sat_mask = sat_raw
    else:
        # Automatic saturation guard when PIXEL_QUALITY did not already flag SAT_BLEED.
        already_flagged = (
            pixel_quality is not None
            and np.any((np.asarray(pixel_quality, dtype=np.uint16) & np.uint16(PixelQuality.SAT_BLEED)) != 0)
        )
        if not already_flagged and auto_sat_level > 0.0:
            med_img = np.full((ny, nx), np.nan, dtype=np.float64)
            med_img[any_finite_pix] = np.nanmedian(flux_cad[:, any_finite_pix], axis=0)
            sat_raw = np.isfinite(med_img) & (med_img > float(auto_sat_level))
            cand_sat = ndimage.binary_dilation(sat_raw, structure=structure_2d)
            surviving = base_mask & any_finite_pix & ~qual_excl & ~source_excl & ~cand_sat
            sat_mask = cand_sat if int(np.sum(surviving)) >= 4 else np.zeros((ny, nx), dtype=bool)
        else:
            sat_mask = np.zeros((ny, nx), dtype=bool)

    minflux_excl = np.zeros((ny, nx), dtype=bool)
    minflux_excl[any_finite_pix] = np.nanmin(flux_cad[:, any_finite_pix], axis=0) < float(minflux)

    # Require pixels to be finite on the majority of surviving cadences, then drop any cadence
    # where a surviving candidate pixel is NaN.
    n_good_cad = int(np.sum(cad_mask))
    finite_counts = np.sum(np.isfinite(flux_cad), axis=0)
    min_valid_cad = min(60, (n_good_cad - 1) // 2)
    cand_pix = (
        base_mask
        & (finite_counts > min_valid_cad)
        & ~qual_excl
        & ~source_excl
        & ~sat_mask
        & ~minflux_excl
    )
    if not np.any(cand_pix):
        raise ValueError("No valid halo pixels survived spatial quality and saturation masking.")

    all_cand_finite = np.all(np.isfinite(flux_arr[cad_mask][:, cand_pix]), axis=1)
    if not np.any(all_cand_finite):
        raise ValueError("No cadences with all-finite candidate halo pixels survived.")
    cad_mask[np.flatnonzero(cad_mask)[~all_cand_finite]] = False

    flux_final_cad = flux_arr[cad_mask]
    mean_p = np.zeros((ny, nx), dtype=np.float64)
    mean_p[cand_pix] = np.mean(flux_final_cad[:, cand_pix], axis=0)

    sky_excl = np.zeros((ny, nx), dtype=bool)
    snr_excl = np.zeros((ny, nx), dtype=bool)

    if not legacy_tv:
        # Charnes-Cooper coordinates require strictly positive mean pixel flux.
        sky_excl |= cand_pix & (mean_p <= 0.0)

        if flux_bkg is not None and sky_kappa > 0.0:
            bkg_arr = np.asarray(flux_bkg, dtype=np.float64)
            if bkg_arr.shape != (n_cad, ny, nx):
                raise ValueError(
                    f"flux_bkg shape {bkg_arr.shape} does not match {(n_cad, ny, nx)}."
                )
            bkg_sub = bkg_arr[cad_mask]
            fin_bkg = bkg_sub[np.isfinite(bkg_sub)]
            if fin_bkg.size > 0 and np.any(fin_bkg != 0.0):
                med_bkg = np.zeros((ny, nx), dtype=np.float64)
                for r_idx, c_idx in zip(*np.where(cand_pix), strict=True):
                    col_b = bkg_sub[:, r_idx, c_idx]
                    fin_col = col_b[np.isfinite(col_b)]
                    if fin_col.size > 0:
                        med_bkg[r_idx, c_idx] = max(float(np.median(fin_col)), 0.0)
                sky_excl |= cand_pix & (mean_p < float(sky_kappa) * med_bkg)

        if min_snr > 0.0:
            active_for_snr = cand_pix & ~sky_excl
            if np.any(active_for_snr):
                if flux_err is not None:
                    err_arr = np.asarray(flux_err, dtype=np.float64)
                    if err_arr.shape != (n_cad, ny, nx):
                        raise ValueError(
                            f"flux_err shape {err_arr.shape} does not match {(n_cad, ny, nx)}."
                        )
                    err_sub = err_arr[cad_mask]
                    for r_idx, c_idx in zip(*np.where(active_for_snr), strict=True):
                        e_col = err_sub[:, r_idx, c_idx]
                        pos_e = e_col[np.isfinite(e_col) & (e_col > 0.0)]
                        if pos_e.size > 0:
                            med_err = float(np.median(pos_e))
                            if mean_p[r_idx, c_idx] / med_err < float(min_snr):
                                snr_excl[r_idx, c_idx] = True
                elif dt_days is not None and dt_days > 0.0:
                    dt_sec = float(dt_days) * 86400.0
                    shot_err = np.sqrt(np.maximum(mean_p, 0.0) / max(dt_sec, 1.0))
                    snr_val = np.where(shot_err > 0.0, mean_p / shot_err, 0.0)
                    snr_excl |= active_for_snr & (snr_val < float(min_snr))

    final_pix = cand_pix & ~sky_excl & ~snr_excl
    if not np.any(final_pix):
        raise ValueError(
            "No valid halo pixels survived sky-dominance (P_bar >= kappa * B) and SNR filtering."
        )

    return HaloMaskResult(
        pixel_mask=final_pix,
        cadence_mask=cad_mask,
        saturated_mask=sat_mask,
        quality_excluded_mask=qual_excl,
        sky_excluded_mask=sky_excl,
        snr_excluded_mask=snr_excl,
        source_excluded_mask=source_excl,
    )
