"""Spatial halo masking, saturation dilation, and WCS contaminant-star exclusion.

Implements clean aperture selection for halo photometry:

* Excludes ``PIXEL_QUALITY`` bits ``STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED |
  SOURCE_MASK | BKG_USED`` so halo and background pixels stay disjoint.
* 2D saturation + bleed + LDE-undershoot dilation calibrated to physical CCD
  full-well rates (7.5e4 e-/s for *TESS*; 1.661e5 e-/s for *Kepler*/*K2*),
  processed in memory-bounded chunks/strips so 20-s *TESS* TPFs never copy
  full 3D cubes to ``float64``.
* Optional cadence-invariant ``min_snr`` cut (default ``0.0``, disabled; when
  ``> 0``, normalized to 2-min-equivalent exposure time via
  ``sqrt(120 / dt_sec)`` so 20 s, 2 min, and 30 min agree).
* WCS-projected contaminant-star masking with proper-motion propagation and
  Gaussian PRF encircled-energy radii, supporting pre-cached Astropy tables or
  live ``astroquery.mast`` queries without touching cache directories unless a
  live query is actually executed.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from erfa import ErfaWarning
from scipy import ndimage
from scipy.special import erfinv

from ._contract import PixelQuality, resolve_cache_dir

__all__ = [
    "DEFAULT_EXCLUDE_QUALITY",
    "HaloMaskResult",
    "KEPLER_SAT_RATE_E_PER_S",
    "MaskConfig",
    "TESS_SAT_RATE_E_PER_S",
    "_estimate_target_mag",
    "_resolve_tpf_obs_time",
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

TESS_SAT_RATE_E_PER_S: float = 1.5e5 / 2.0  # 7.5e4 e-/s (Vanderspek et al. 2018)
KEPLER_SAT_RATE_E_PER_S: float = 1.0e6 / 6.02  # 1.661e5 e-/s (Van Cleve & Caldwell 2016)

_DETECTOR_PRF_SIGMA_PIX: float = float(0.5 / (math.sqrt(2.0) * erfinv(math.sqrt(0.5))))
_OPTICAL_CORE_SIGMA_ARCSEC: float = 2.0
_ZODIACAL_SKY_MU_MAG_ARCSEC2: float = 22.0
_HALO_WING_CONTRAST_MAG: float = 10.0
_MASK_CHUNK_CADENCES: int = 4096

_MAG_CANDIDATES: tuple[str, ...] = (
    "tmag",
    "phot_g_mean_mag",
    "kepmag",
    "mag",
    "vmag",
    "gmag",
)


@dataclass(frozen=True)
class MaskConfig:
    """Structured spatial and cadence masking options for :func:`extract_halo_lightcurve`."""

    aperture_mask: str | np.ndarray = "all"
    rr: tuple[float, float] | None = None
    sub: int = 1
    thresh: float = -1.0
    auto_sat_level: float | None = None
    bleed_dilation: tuple[int, int] = (3, 3)
    minflux: float = -100.0
    min_snr: float = 0.0
    exclude_quality: PixelQuality = DEFAULT_EXCLUDE_QUALITY
    remove_contaminants: bool = False
    catalog_table: Table | str | os.PathLike[str] | None = None
    catalog: str = "TIC"
    offline: bool = False
    cache_dir: str | os.PathLike[str] | None = None
    strict_hpc: bool = False
    target_mag: float | None = None
    target_exclusion_arcsec: float | None = None
    mag_limit: float | None = None
    star_radius_pix: float | None = None


@dataclass(frozen=True)
class HaloMaskResult:
    """Spatial and temporal masks constructed for halo weight optimization."""

    pixel_mask: np.ndarray
    cadence_mask: np.ndarray
    saturated_mask: np.ndarray
    quality_excluded_mask: np.ndarray
    snr_excluded_mask: np.ndarray
    source_excluded_mask: np.ndarray

    @property
    def sky_excluded_mask(self) -> np.ndarray:
        """All-False compatibility mask (sky-dominance cuts were removed in v2.0.2)."""
        return np.zeros_like(self.pixel_mask, dtype=bool)


def _load_catalog_table(catalog_table: Table | str | os.PathLike[str]) -> Table:
    if isinstance(catalog_table, Table):
        return catalog_table
    path = Path(catalog_table)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            return Table.read(path, format="parquet")
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet star catalogues requires 'pyarrow'. "
                "Install it with `pip install pyarrow` or supply a FITS/ECSV catalogue."
            ) from exc
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


def _coalesce_magnitudes(table: Table) -> tuple[np.ndarray | None, bool, str | None]:
    lower_map = {col.lower(): col for col in table.colnames}
    found_cols = [lower_map[c] for c in _MAG_CANDIDATES if c in lower_map]
    if not found_cols:
        return None, False, None

    coalesced = np.full(len(table), np.nan, dtype=np.float64)
    for col in found_cols:
        col_arr = np.asarray(table[col], dtype=np.float64)
        missing = ~np.isfinite(coalesced) & np.isfinite(col_arr)
        coalesced[missing] = col_arr[missing]
    return coalesced, True, found_cols[0].lower()


def _is_electron_count_unit(unit_obj: Any) -> bool:
    if unit_obj is None or unit_obj == u.dimensionless_unscaled:
        return False
    if unit_obj == u.electron:
        return True
    return str(unit_obj).strip().lower() in ("electron", "electrons", "e-")


def _is_integrated_count_unit(unit_obj: Any) -> bool:
    if _is_electron_count_unit(unit_obj):
        return True
    if unit_obj is None or unit_obj == u.dimensionless_unscaled:
        return False
    if unit_obj == u.count:
        return True
    return str(unit_obj).strip().lower() in ("count", "counts", "ct", "adu")


def _integrated_to_rate_unit(unit_obj: Any) -> u.UnitBase:
    return (u.electron / u.s) if _is_electron_count_unit(unit_obj) else (u.count / u.s)


def _resolve_tpf_column_unit(tpf: Any, col_name: str, attr_unit: Any) -> Any:
    if attr_unit is not None and attr_unit != u.dimensionless_unscaled:
        return attr_unit
    hdul = getattr(tpf, "hdu", None)
    if hdul is not None and len(hdul) > 1 and hasattr(hdul[1], "columns"):
        target_upper = col_name.upper()
        for idx_c, col in enumerate(hdul[1].columns, start=1):
            if col.name.upper() == target_upper:
                hdr_unit = hdul[1].header.get(f"TUNIT{idx_c}") or col.unit
                if isinstance(hdr_unit, str) and hdr_unit.strip():
                    return hdr_unit.strip()
    return attr_unit


def _resolve_hdu_unit(tpf: Any, hdu_name: str, fallback_unit: Any) -> Any:
    hdul = getattr(tpf, "hdu", None)
    if hdul is not None and hdu_name in hdul:
        bunit = hdul[hdu_name].header.get("BUNIT")
        if isinstance(bunit, str) and bunit.strip():
            return bunit.strip()
    return fallback_unit


def _resolve_cadence_exposure_sec(tpf: Any, dt_days: float) -> float:
    """Resolve per-cadence effective integration time in seconds."""
    hdul = getattr(tpf, "hdu", None)
    headers: list[Any] = []
    if hdul is not None:
        if len(hdul) > 1:
            headers.append(hdul[1].header)
        if len(hdul) > 0:
            headers.append(hdul[0].header)
    meta = getattr(tpf, "meta", None)
    if meta is not None:
        headers.append(meta)

    for hdr in headers:
        if not hasattr(hdr, "get"):
            continue
        timedel = hdr.get("TIMEDEL")
        if timedel is not None and not isinstance(timedel, bool):
            try:
                td_val = float(timedel)
            except (TypeError, ValueError):
                td_val = np.nan
            if np.isfinite(td_val) and td_val > 0.0:
                try:
                    deadc_val = float(hdr.get("DEADC", 1.0))
                except (TypeError, ValueError):
                    deadc_val = 1.0
                if not np.isfinite(deadc_val) or deadc_val <= 0.0 or deadc_val > 1.0:
                    deadc_val = 1.0
                return td_val * 86400.0 * deadc_val

        int_time = hdr.get("INT_TIME")
        n_readout = hdr.get("NREADOUT")
        num_frm = hdr.get("NUM_FRM")
        if int_time is not None and (n_readout is not None or num_frm is not None):
            try:
                it_val = float(int_time)
                nf_val = float(n_readout if n_readout is not None else num_frm)
            except (TypeError, ValueError):
                it_val, nf_val = np.nan, np.nan
            if np.isfinite(it_val) and np.isfinite(nf_val) and it_val > 0.0 and nf_val > 0.0:
                return it_val * nf_val

    return max(float(dt_days) * 86400.0, 1.0)


def _query_mast_region(
    center: SkyCoord,
    radius_deg: float,
    *,
    catalog: str = "TIC",
    cache_dir: Path | None = None,
) -> Table:
    try:
        from astroquery.mast import Catalogs
    except ImportError as exc:
        raise ImportError(
            "Live MAST star-catalogue queries require 'astroquery'. "
            "Install it with `pip install astroquery` or pass a pre-cached `catalog_table`."
        ) from exc

    if cache_dir is None:
        return Catalogs.query_region(center, radius=radius_deg * u.deg, catalog=catalog)

    if hasattr(Catalogs, "_cache_location"):
        prev_private = Catalogs._cache_location
        try:
            Catalogs.cache_location = str(cache_dir)
            return Catalogs.query_region(center, radius=radius_deg * u.deg, catalog=catalog)
        finally:
            if prev_private is None:
                reset_fn = getattr(Catalogs, "reset_cache_location", None)
                if callable(reset_fn):
                    reset_fn()
                else:
                    Catalogs._cache_location = None
            else:
                Catalogs.cache_location = prev_private

    missing = object()
    prev_cache = getattr(Catalogs, "cache_location", missing)
    try:
        Catalogs.cache_location = str(cache_dir)
        return Catalogs.query_region(center, radius=radius_deg * u.deg, catalog=catalog)
    finally:
        if prev_cache is missing:
            with contextlib.suppress(AttributeError):
                delattr(Catalogs, "cache_location")
        else:
            Catalogs.cache_location = prev_cache


def _wcs_pixel_scale_arcsec(wcs: WCS) -> float:
    scales_deg = np.asarray(proj_plane_pixel_scales(wcs.celestial), dtype=np.float64)
    pos_scales = scales_deg[np.isfinite(scales_deg) & (scales_deg > 0.0)]
    if pos_scales.size == 0:
        return 21.0
    return float(np.sqrt(np.prod(pos_scales[:2])) * 3600.0)


def _coerce_obs_time(obs_time: Time | str | float | None, wcs: WCS) -> Time | None:
    if obs_time is not None:
        if isinstance(obs_time, Time):
            jd_arr = np.asarray(np.ma.filled(obs_time.tdb.jd, np.nan), dtype=np.float64)
            if jd_arr.ndim > 0:
                fin_jd = jd_arr[np.isfinite(jd_arr)]
                if fin_jd.size == 0:
                    raise ValueError("obs_time Time array contains no finite timestamps.")
                return Time(float(np.median(fin_jd)), format="jd", scale="tdb")
            val_jd = float(jd_arr)
            if not np.isfinite(val_jd):
                raise ValueError("obs_time Time scalar is not finite.")
            return Time(val_jd, format="jd", scale="tdb")
        if isinstance(obs_time, int | float | np.floating):
            val = float(obs_time)
            if np.isfinite(val):
                if 1900.0 <= val <= 2200.0:
                    return Time(val, format="jyear", scale="tdb")
                if val > 2.4e6:
                    return Time(val, format="jd", scale="tdb")
                return Time(val, format="mjd", scale="tdb")
            return None
        if isinstance(obs_time, str) and obs_time.strip():
            return Time(obs_time, scale="utc")

    mjd_obs = getattr(wcs.wcs, "mjdobs", np.nan)
    if np.isfinite(mjd_obs) and float(mjd_obs) > 0.0:
        return Time(float(mjd_obs), format="mjd", scale="tdb")
    date_obs = getattr(wcs.wcs, "dateobs", "")
    if isinstance(date_obs, str) and date_obs.strip():
        try:
            return Time(date_obs.strip(), scale="utc")
        except ValueError:
            return None
    return None


def _propagate_catalogue_coords(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    table_valid: Table,
    *,
    catalog: str,
    obs_time: Time | None,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> tuple[SkyCoord, SkyCoord, SkyCoord]:
    coords_raw = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    pmra_col = _find_column_case_insensitive(table_valid, ("pmra", "pm_ra"))
    pmdec_col = _find_column_case_insensitive(table_valid, ("pmdec", "pm_dec", "pmde"))
    if pmra_col is None or pmdec_col is None:
        return coords_raw, coords_raw, coords_raw

    pmra = np.asarray(table_valid[pmra_col], dtype=np.float64)
    pmdec = np.asarray(table_valid[pmdec_col], dtype=np.float64)
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    if not np.any(has_pm):
        return coords_raw, coords_raw, coords_raw

    ep_col = _find_column_case_insensitive(table_valid, ("ref_epoch", "epoch"))
    lower_names = {c.lower() for c in table_valid.colnames}
    ra_lower = (ra_col or "").lower()
    dec_lower = (dec_col or "").lower()
    has_gaia_cols = bool(
        lower_names & {"source_id", "phot_g_mean_mag", "gmag", "dr3_source_id", "dr2_source_id"}
    )
    has_tic_cols = bool(lower_names & {"tic", "ticid", "posflag"}) or (
        "id" in lower_names and "tmag" in lower_names and "source_id" not in lower_names
    )
    if ra_lower == "raj2000" or dec_lower == "dej2000":
        default_ep = 2000.0
    elif ra_lower == "ra_icrs" or dec_lower in ("de_icrs", "dec_icrs") or "gaia" in catalog.lower():
        default_ep = 2016.0
    elif has_gaia_cols and not has_tic_cols:
        default_ep = 2016.0
    else:
        default_ep = 2000.0

    if ep_col is not None:
        ep_raw = np.asarray(table_valid[ep_col], dtype=np.float64)
        ref_ep = np.where(np.isfinite(ep_raw) & (ep_raw > 1900.0), ep_raw, default_ep)
    else:
        ref_ep = np.full(len(table_valid), default_ep, dtype=np.float64)

    sc_motion = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        pm_ra_cosdec=np.where(has_pm, pmra, 0.0) * (u.mas / u.yr),
        pm_dec=np.where(has_pm, pmdec, 0.0) * (u.mas / u.yr),
        obstime=Time(ref_ep, format="jyear", scale="tdb"),
        frame="icrs",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        sc_j2000_pm = sc_motion.apply_space_motion(
            new_obstime=Time(2000.0, format="jyear", scale="tdb")
        )
        coords_j2000 = SkyCoord(ra=sc_j2000_pm.ra, dec=sc_j2000_pm.dec, frame="icrs")
        if obs_time is not None:
            sc_obs_pm = sc_motion.apply_space_motion(new_obstime=obs_time)
            coords_obs = SkyCoord(ra=sc_obs_pm.ra, dec=sc_obs_pm.dec, frame="icrs")
        else:
            coords_obs = coords_raw

    return coords_raw, coords_j2000, coords_obs


def _estimate_target_mag(
    *,
    tpf: Any | None = None,
    flux: np.ndarray | None = None,
    mission: str | None = None,
    flux_scale: float = 1.0,
) -> float | None:
    """Infer primary target magnitude from TPF headers or median stamp flux without full-cube copies."""
    if tpf is not None:
        header_sources: list[Any] = []
        hdul = getattr(tpf, "hdu", None)
        if hdul is not None and len(hdul) > 0:
            header_sources.append(hdul[0].header)
        meta = getattr(tpf, "meta", None)
        if meta is not None:
            header_sources.append(meta)
        for hdr in header_sources:
            for key in ("TESSMAG", "TMAG", "KEPMAG", "GMAG", "VMAG"):
                val = hdr.get(key) if hasattr(hdr, "get") else None
                if val is not None and not isinstance(val, bool):
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fval):
                        return fval

    if flux is not None:
        flux_arr = np.asarray(flux)
        if flux_arr.ndim == 3 and flux_arr.size > 0:
            n_cad = flux_arr.shape[0]
            stride = max(1, n_cad // 512)
            sample = np.asarray(flux_arr[::stride], dtype=np.float64) * float(flux_scale)
            any_fin = np.any(np.isfinite(sample), axis=0)
            if np.any(any_fin):
                med_img = np.zeros(sample.shape[1:], dtype=np.float64)
                med_img[any_fin] = np.nanmedian(sample[:, any_fin], axis=0)
                sum_pos = float(np.sum(np.maximum(med_img, 0.0)))
                if np.isfinite(sum_pos) and sum_pos > 0.0:
                    eff_mission = mission if mission is not None else getattr(tpf, "mission", None)
                    zp = 25.3 if str(eff_mission or "").upper() in ("KEPLER", "K2") else 20.44
                    return float(zp - 2.5 * math.log10(sum_pos))
    return None


def mask_background_stars(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    strict_hpc: bool = False,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_mag: float | None = None,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    radius_pix: float | None = None,
    obs_time: Time | str | float | None = None,
    catalog: str = "TIC",
) -> np.ndarray:
    """Construct a ``(ny, nx)`` boolean mask of background stars projected via ``wcs``."""
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"Invalid stamp shape {shape}; expected positive (ny, nx).")
    if wcs is None or not wcs.has_celestial:
        raise ValueError("A valid celestial astropy.wcs.WCS is required to project star masks.")

    pix_scale_arcsec = _wcs_pixel_scale_arcsec(wcs)
    sigma_prf_pix = math.hypot(_OPTICAL_CORE_SIGMA_ARCSEC / pix_scale_arcsec, _DETECTOR_PRF_SIGMA_PIX)
    resolved_obs_time = _coerce_obs_time(obs_time, wcs)

    if catalog_table is None:
        if offline:
            raise RuntimeError(
                "Live catalogue queries are disabled when offline=True; "
                "pass a pre-cached Astropy Table via catalog_table."
            )
        resolved_cache = resolve_cache_dir(cache_dir, strict_hpc=strict_hpc, create=True)
        corners_x = np.array([-0.5, nx - 0.5, -0.5, nx - 0.5], dtype=np.float64)
        corners_y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=np.float64)
        center_sky = wcs.celestial.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        corner_sky = wcs.celestial.pixel_to_world(corners_x, corners_y)
        sep_deg = float(np.max(center_sky.separation(corner_sky).deg)) * 1.15 + (
            2.0 * pix_scale_arcsec / 3600.0
        )
        tbl = _query_mast_region(
            center_sky,
            sep_deg,
            catalog=catalog,
            cache_dir=resolved_cache,
        )
    else:
        tbl = _load_catalog_table(catalog_table)

    out_mask = np.zeros((ny, nx), dtype=bool)
    if len(tbl) == 0:
        return out_mask

    lower_tbl_cols = {c.lower() for c in tbl.colnames}
    tbl_is_gaia = ("gaia" in catalog.lower()) or (
        bool(
            lower_tbl_cols
            & {"source_id", "phot_g_mean_mag", "gmag", "dr3_source_id", "dr2_source_id"}
        )
        and not (
            bool(lower_tbl_cols & {"tic", "ticid", "posflag"})
            or ("id" in lower_tbl_cols and "tmag" in lower_tbl_cols and "source_id" not in lower_tbl_cols)
        )
    )
    if tbl_is_gaia:
        ra_cands = ("ra_icrs", "ra", "ra_deg", "raj2000")
        dec_cands = ("de_icrs", "dec_icrs", "dec", "dec_deg", "dej2000")
    else:
        ra_cands = ("ra", "raj2000", "ra_deg", "ra_icrs")
        dec_cands = ("dec", "dej2000", "dec_deg", "dec_icrs", "de_icrs")

    ra_col = _find_column_case_insensitive(tbl, ra_cands)
    dec_col = _find_column_case_insensitive(tbl, dec_cands)
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Catalogue table with columns {tbl.colnames} lacks recognizable RA/Dec columns."
        )

    ra_vals = np.asarray(tbl[ra_col], dtype=np.float64)
    dec_vals = np.asarray(tbl[dec_col], dtype=np.float64)
    valid = np.isfinite(ra_vals) & np.isfinite(dec_vals)

    mag_vals, has_mag_col, primary_mag_col = _coalesce_magnitudes(tbl)
    if has_mag_col and mag_vals is not None:
        valid &= np.isfinite(mag_vals)
    if not np.any(valid):
        return out_mask

    tbl_valid = tbl[valid]
    ra_vals = ra_vals[valid]
    dec_vals = dec_vals[valid]
    if mag_vals is not None:
        mag_vals = mag_vals[valid]

    coords_raw, coords_j2000, coords_obs = _propagate_catalogue_coords(
        ra_vals,
        dec_vals,
        tbl_valid,
        catalog=catalog,
        obs_time=resolved_obs_time,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    inferred_tgt_mag: float | None = (
        float(target_mag)
        if target_mag is not None and np.isfinite(float(target_mag))
        else None
    )
    tgt_sky: SkyCoord | None = None
    primary_sky_obs: SkyCoord | None = None
    if target_coord is not None:
        if isinstance(target_coord, SkyCoord):
            tgt_sky = target_coord.icrs
        else:
            t_ra, t_dec = float(target_coord[0]), float(target_coord[1])
            tgt_sky = (
                SkyCoord(ra=t_ra * u.deg, dec=t_dec * u.deg, frame="icrs")
                if np.isfinite(t_ra) and np.isfinite(t_dec)
                else None
            )
        eff_excl_arcsec = (
            max(1.5 * pix_scale_arcsec, 6.0)
            if target_exclusion_arcsec is None
            else float(target_exclusion_arcsec)
        )
        if tgt_sky is not None:
            primary_sky_obs = tgt_sky
            if eff_excl_arcsec > 0.0:
                min_sep_arcsec = np.minimum(
                    coords_raw.separation(tgt_sky).arcsec,
                    np.minimum(
                        coords_j2000.separation(tgt_sky).arcsec,
                        coords_obs.separation(tgt_sky).arcsec,
                    ),
                )
                in_excl = min_sep_arcsec <= eff_excl_arcsec
                if np.any(in_excl):
                    if mag_vals is not None:
                        cand_idx = np.flatnonzero(in_excl)
                        best_idx = cand_idx[int(np.argmin(mag_vals[cand_idx]))]
                        if primary_mag_col == "vmag":
                            primary_mag_tol = 6.5
                        elif primary_mag_col in ("phot_g_mean_mag", "gmag"):
                            primary_mag_tol = 4.5
                        else:
                            primary_mag_tol = 3.0
                        valid_primary = (
                            inferred_tgt_mag is None
                            or float(mag_vals[best_idx]) <= inferred_tgt_mag + primary_mag_tol
                        )
                        if valid_primary:
                            primary_sky_obs = coords_obs[best_idx]
                            if inferred_tgt_mag is None:
                                inferred_tgt_mag = float(mag_vals[best_idx])
                        if target_exclusion_arcsec is not None:
                            keep = ~in_excl
                            coords_obs = coords_obs[keep]
                            mag_vals = mag_vals[keep]
                        elif valid_primary:
                            is_primary = (
                                coords_obs.separation(coords_obs[best_idx]).arcsec <= 2.0
                            ) & (mag_vals <= mag_vals[best_idx] + 0.5)
                            keep = ~is_primary
                            coords_obs = coords_obs[keep]
                            mag_vals = mag_vals[keep]
                    else:
                        keep = ~in_excl
                        coords_obs = coords_obs[keep]

    if len(coords_obs) == 0:
        return out_mask

    sky_mag_limit = _ZODIACAL_SKY_MU_MAG_ARCSEC2 - 2.5 * math.log10(
        max(pix_scale_arcsec**2, 1e-6) / 0.50
    )
    if mag_limit is not None:
        eff_mag_limit = float(mag_limit)
    elif inferred_tgt_mag is not None and np.isfinite(inferred_tgt_mag):
        eff_mag_limit = min(inferred_tgt_mag + _HALO_WING_CONTRAST_MAG, sky_mag_limit)
    else:
        eff_mag_limit = sky_mag_limit

    if mag_vals is not None:
        bright_enough = mag_vals <= eff_mag_limit
        if not np.any(bright_enough):
            return out_mask
        coords_obs = coords_obs[bright_enough]
        mag_vals = mag_vals[bright_enough]

    col_pix, row_pix = wcs.celestial.world_to_pixel(coords_obs)
    col_pix = np.atleast_1d(np.asarray(col_pix, dtype=np.float64))
    row_pix = np.atleast_1d(np.asarray(row_pix, dtype=np.float64))

    if radius_pix is not None:
        if radius_pix <= 0.0:
            raise ValueError(f"radius_pix must be > 0, got {radius_pix!r}.")
        radii = np.full(col_pix.shape, float(radius_pix), dtype=np.float64)
    elif mag_vals is not None:
        delta_m = np.maximum(eff_mag_limit - mag_vals, 0.0)
        radii = np.sqrt(2.0 * (sigma_prf_pix**2) * (math.log(10.0) / 2.5) * delta_m + 0.5)
    else:
        radii = np.full(col_pix.shape, max(2.0 * sigma_prf_pix, 1.0), dtype=np.float64)

    if target_exclusion_arcsec is None and primary_sky_obs is not None:
        blend_sep_arcsec = max(1.5 * pix_scale_arcsec, 2.0 * sigma_prf_pix * pix_scale_arcsec)
        sep_from_primary = coords_obs.separation(primary_sky_obs).arcsec
        tgt_col_p, tgt_row_p = wcs.celestial.world_to_pixel(primary_sky_obs)
        dist_to_target_pix = np.hypot(col_pix - float(tgt_col_p), row_pix - float(tgt_row_p))
        engulfs_target = (sep_from_primary < blend_sep_arcsec) & (dist_to_target_pix <= radii)
        if np.any(engulfs_target):
            keep_non_blend = ~engulfs_target
            if not np.any(keep_non_blend):
                return out_mask
            col_pix = col_pix[keep_non_blend]
            row_pix = row_pix[keep_non_blend]
            radii = radii[keep_non_blend]

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

    yy, xx = np.mgrid[0:ny, 0:nx]
    for c_s, r_s, rad_s in zip(col_pix[in_box], row_pix[in_box], radii[in_box], strict=True):
        out_mask |= (xx - c_s) ** 2 + (yy - r_s) ** 2 <= rad_s**2

    return out_mask


def _resolve_tpf_wcs(tpf: Any) -> WCS | None:
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


def _resolve_tpf_obs_time(tpf: Any) -> Time | None:
    t_attr = getattr(tpf, "time", None)
    if t_attr is None or len(t_attr) == 0:
        return None
    jd_vals = np.asarray(t_attr.jd, dtype=np.float64)
    fin_jd = jd_vals[np.isfinite(jd_vals)]
    if fin_jd.size == 0:
        return None
    return Time(float(np.median(fin_jd)), format="jd", scale="tdb")


def remove_stars(
    tpf: Any,
    *,
    wcs: WCS | None = None,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    strict_hpc: bool = False,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_mag: float | None = None,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    radius_pix: float | None = None,
    obs_time: Time | str | float | None = None,
    catalog: str = "TIC",
    mission: str | None = None,
) -> Any:
    """Mask background contaminant stars on a ``TargetPixelFile`` or 3D flux array via WCS."""
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
        resolved_obs_time = obs_time if obs_time is not None else _resolve_tpf_obs_time(tpf)
        flux_view = np.asarray(tpf.flux.value if hasattr(tpf.flux, "value") else tpf.flux)
        tpf_flux_unit = _resolve_tpf_column_unit(tpf, "FLUX", getattr(tpf.flux, "unit", None))
        flux_scale = 1.0
        if _is_integrated_count_unit(tpf_flux_unit):
            t_attr = getattr(tpf, "time", None)
            dt_tpf = 1.0 / 48.0
            if t_attr is not None and len(t_attr) > 1:
                tv = np.asarray(t_attr.value, dtype=np.float64)
                d_tv = np.diff(tv[np.isfinite(tv)])
                pos_d = d_tv[d_tv > 0.0]
                if pos_d.size > 0:
                    dt_tpf = float(np.median(pos_d))
            flux_scale = 1.0 / _resolve_cadence_exposure_sec(tpf, dt_tpf)

        eff_mission_tpf = mission if mission is not None else getattr(tpf, "mission", None)
        resolved_target_mag = (
            target_mag
            if target_mag is not None
            else _estimate_target_mag(
                tpf=tpf,
                flux=flux_view,
                mission=eff_mission_tpf,
                flux_scale=flux_scale,
            )
        )

        ny, nx = int(tpf.flux.shape[1]), int(tpf.flux.shape[2])
        star_mask = mask_background_stars(
            resolved_wcs,
            (ny, nx),
            catalog_table=catalog_table,
            offline=offline,
            cache_dir=cache_dir,
            strict_hpc=strict_hpc,
            target_coord=resolved_target,
            target_mag=resolved_target_mag,
            target_exclusion_arcsec=target_exclusion_arcsec,
            mag_limit=mag_limit,
            radius_pix=radius_pix,
            obs_time=resolved_obs_time,
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

    resolved_arr_mag = (
        target_mag if target_mag is not None else _estimate_target_mag(flux=arr, mission=mission)
    )
    star_mask = mask_background_stars(
        wcs,
        (arr.shape[1], arr.shape[2]),
        catalog_table=catalog_table,
        offline=offline,
        cache_dir=cache_dir,
        strict_hpc=strict_hpc,
        target_coord=target_coord,
        target_mag=resolved_arr_mag,
        target_exclusion_arcsec=target_exclusion_arcsec,
        mag_limit=mag_limit,
        radius_pix=radius_pix,
        obs_time=obs_time,
        catalog=catalog,
    )
    out = np.array(arr, dtype=np.result_type(arr.dtype, np.float32), copy=True)
    out[:, star_mask] = np.nan
    return out


def _chunked_flux_stats(
    flux_arr: np.ndarray,
    good_idx: np.ndarray,
    *,
    flux_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(finite_counts, min_img, max_img)`` over ``good_idx`` without copying the 3D cube."""
    _, ny, nx = flux_arr.shape
    fin_counts = np.zeros((ny, nx), dtype=np.int64)
    min_img = np.full((ny, nx), np.inf, dtype=np.float64)
    max_img = np.full((ny, nx), -np.inf, dtype=np.float64)

    for start in range(0, good_idx.size, _MASK_CHUNK_CADENCES):
        idx_chunk = good_idx[start : start + _MASK_CHUNK_CADENCES]
        chunk = np.asarray(flux_arr[idx_chunk], dtype=np.float64)
        if flux_scale != 1.0:
            chunk = chunk * flux_scale
        fin = np.isfinite(chunk)
        fin_counts += np.sum(fin, axis=0)
        min_img = np.minimum(min_img, np.min(np.where(fin, chunk, np.inf), axis=0))
        max_img = np.maximum(max_img, np.max(np.where(fin, chunk, -np.inf), axis=0))

    return fin_counts, min_img, max_img


def _strip_p99_unsub(
    flux_arr: np.ndarray,
    good_idx: np.ndarray,
    *,
    bkg_arr: np.ndarray | None,
    bkgsub: bool,
    flux_scale: float = 1.0,
    bkg_scale: float = 1.0,
) -> np.ndarray:
    """Compute exact 99th-percentile unsubtracted flux row-by-row in bounded host memory."""
    _, ny, nx = flux_arr.shape
    p99 = np.full((ny, nx), np.nan, dtype=np.float64)
    add_bkg = bkgsub and bkg_arr is not None

    for r in range(ny):
        strip = np.asarray(flux_arr[good_idx, r, :], dtype=np.float64)
        if flux_scale != 1.0:
            strip = strip * flux_scale
        if add_bkg:
            assert bkg_arr is not None
            b_strip = np.asarray(bkg_arr[good_idx, r, :], dtype=np.float64)
            if bkg_scale != 1.0:
                b_strip = b_strip * bkg_scale
            strip = np.where(np.isfinite(b_strip), strip + np.maximum(b_strip, 0.0), strip)
        any_fin = np.any(np.isfinite(strip), axis=0)
        if np.any(any_fin):
            p99[r, any_fin] = np.nanpercentile(strip[:, any_fin], 99.0, axis=0)
    return p99


def build_halo_masks(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    flux_bkg: np.ndarray | None = None,
    bkgsub: bool = True,
    pixel_quality: np.ndarray | None = None,
    aperture_mask: np.ndarray | None = None,
    cadence_quality_mask: np.ndarray | None = None,
    pos_corr1: np.ndarray | None = None,
    pos_corr2: np.ndarray | None = None,
    wcs: WCS | None = None,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_mag: float | None = None,
    obs_time: Time | str | float | None = None,
    remove_contaminants: bool = False,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    catalog: str = "TIC",
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    strict_hpc: bool = False,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    star_radius_pix: float | None = None,
    thresh: float = -1.0,
    auto_sat_level: float | None = None,
    mission: str | None = None,
    bleed_dilation: tuple[int, int] = (3, 3),
    minflux: float = -100.0,
    min_snr: float = 0.0,
    dt_days: float | None = None,
    exclude_quality: PixelQuality = DEFAULT_EXCLUDE_QUALITY,
    flux_scale: float = 1.0,
    flux_err_scale: float = 1.0,
    flux_bkg_scale: float = 1.0,
    config: MaskConfig | None = None,
) -> HaloMaskResult:
    """Build active spatial and cadence masks without copying 3D cubes to ``float64``."""
    if config is not None:
        thresh = config.thresh
        auto_sat_level = config.auto_sat_level
        bleed_dilation = config.bleed_dilation
        minflux = config.minflux
        min_snr = config.min_snr
        exclude_quality = config.exclude_quality
        remove_contaminants = config.remove_contaminants
        catalog_table = config.catalog_table
        catalog = config.catalog
        offline = config.offline
        cache_dir = config.cache_dir
        strict_hpc = config.strict_hpc
        target_mag = config.target_mag if target_mag is None else target_mag
        target_exclusion_arcsec = config.target_exclusion_arcsec
        mag_limit = config.mag_limit
        star_radius_pix = config.star_radius_pix

    flux_arr = np.asarray(flux)
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
        raise ValueError(f"cadence_quality_mask shape {cad_mask.shape} does not match ({n_cad},).")

    if pos_corr1 is not None and pos_corr2 is not None and np.any(cad_mask):
        pc1 = np.asarray(pos_corr1, dtype=np.float64)
        pc2 = np.asarray(pos_corr2, dtype=np.float64)
        if pc1.shape == (n_cad,) and pc2.shape == (n_cad,):
            fin_pc = np.isfinite(pc1[cad_mask]) & np.isfinite(pc2[cad_mask])
            if np.sum(fin_pc) >= 0.8 * np.sum(cad_mask):
                xc = float(np.nanmedian(pc1[cad_mask]))
                yc = float(np.nanmedian(pc2[cad_mask]))
                good_pos = fin_pc & (np.hypot(pc1[cad_mask] - xc, pc2[cad_mask] - yc) < 5.0)
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

    good_idx = np.flatnonzero(cad_mask)
    fin_counts, min_img, max_img = _chunked_flux_stats(flux_arr, good_idx, flux_scale=flux_scale)
    any_finite_pix = fin_counts > 0
    if not np.any(any_finite_pix & base_mask):
        raise ValueError("No finite pixels exist inside aperture_mask on valid cadences.")

    if remove_contaminants or catalog_table is not None:
        if wcs is None or not wcs.has_celestial:
            raise ValueError(
                "Contaminant star masking requested, but no valid celestial WCS was provided."
            )
        resolved_tmag = (
            target_mag
            if target_mag is not None
            else _estimate_target_mag(flux=flux_arr, mission=mission, flux_scale=flux_scale)
        )
        source_excl = mask_background_stars(
            wcs,
            (ny, nx),
            catalog_table=catalog_table,
            offline=offline,
            cache_dir=cache_dir,
            strict_hpc=strict_hpc,
            target_coord=target_coord,
            target_mag=resolved_tmag,
            target_exclusion_arcsec=target_exclusion_arcsec,
            mag_limit=mag_limit,
            radius_pix=star_radius_pix,
            obs_time=obs_time,
            catalog=catalog,
        )
    else:
        source_excl = np.zeros((ny, nx), dtype=bool)

    bkg_arr: np.ndarray | None = None
    if flux_bkg is not None:
        bkg_arr = np.asarray(flux_bkg)
        if bkg_arr.shape != (n_cad, ny, nx):
            raise ValueError(f"flux_bkg shape {bkg_arr.shape} does not match {(n_cad, ny, nx)}.")

    structure_2d = np.ones(bleed_dilation, dtype=bool)
    sat_raw = np.zeros((ny, nx), dtype=bool)
    if thresh is not None and thresh >= 0.0:
        if thresh < 1.0:
            maxflux = float(np.max(max_img[any_finite_pix]))
            sat_raw[any_finite_pix] = max_img[any_finite_pix] > float(thresh) * maxflux
            sat_mask = ndimage.binary_dilation(sat_raw, structure=structure_2d)
        else:
            sat_mask = sat_raw
    else:
        already_flagged = pixel_quality is not None and np.any(
            (np.asarray(pixel_quality, dtype=np.uint16) & np.uint16(PixelQuality.SAT_BLEED)) != 0
        )
        if auto_sat_level is not None:
            resolved_sat = float(auto_sat_level)
            apply_auto_sat = resolved_sat > 0.0
        else:
            resolved_sat = (
                KEPLER_SAT_RATE_E_PER_S
                if str(mission or "").upper() in ("KEPLER", "K2")
                else TESS_SAT_RATE_E_PER_S
            )
            apply_auto_sat = (not already_flagged) and (resolved_sat > 0.0)

        if apply_auto_sat:
            p99_img = _strip_p99_unsub(
                flux_arr,
                good_idx,
                bkg_arr=bkg_arr,
                bkgsub=bkgsub,
                flux_scale=flux_scale,
                bkg_scale=flux_bkg_scale,
            )
            sat_raw = np.isfinite(p99_img) & (p99_img >= resolved_sat)
            sat_mask = ndimage.binary_dilation(sat_raw, structure=structure_2d)
        else:
            sat_mask = np.zeros((ny, nx), dtype=bool)

    minflux_excl = any_finite_pix & (min_img < float(minflux))

    min_valid_cad = min(60, (good_idx.size - 1) // 2)
    cand_pix = (
        base_mask
        & (fin_counts > min_valid_cad)
        & ~qual_excl
        & ~source_excl
        & ~sat_mask
        & ~minflux_excl
    )
    if not np.any(cand_pix):
        raise ValueError("No valid halo pixels survived spatial quality and saturation masking.")

    cand_r, cand_c = np.where(cand_pix)
    all_cand_finite = np.ones(good_idx.size, dtype=bool)
    for start in range(0, good_idx.size, _MASK_CHUNK_CADENCES):
        stop = min(start + _MASK_CHUNK_CADENCES, good_idx.size)
        sub = flux_arr[good_idx[start:stop]][:, cand_r, cand_c]
        all_cand_finite[start:stop] = np.all(np.isfinite(sub), axis=1)

    if not np.any(all_cand_finite):
        raise ValueError("No cadences with all-finite candidate halo pixels survived.")
    cad_mask[good_idx[~all_cand_finite]] = False

    snr_excl = np.zeros((ny, nx), dtype=bool)
    if float(min_snr) > 0.0:
        final_idx = np.flatnonzero(cad_mask)
        sum_p = np.zeros(cand_r.size, dtype=np.float64)
        for start in range(0, final_idx.size, _MASK_CHUNK_CADENCES):
            sub = np.asarray(flux_arr[final_idx[start : start + _MASK_CHUNK_CADENCES]][:, cand_r, cand_c], dtype=np.float64)
            if flux_scale != 1.0:
                sub = sub * flux_scale
            sum_p += np.sum(sub, axis=0)
        mean_cand = sum_p / float(final_idx.size)

        eff_dt_days = float(dt_days) if (dt_days is not None and float(dt_days) > 0.0) else (1.0 / 48.0)
        dt_sec = max(eff_dt_days * 86400.0, 1.0)
        # Normalize per-cadence SNR to 2-min (120 s) equivalent exposure time so 20 s, 2 min,
        # and 30 min cadences select identical apertures: SNR_2min = SNR_cad * sqrt(120 / dt_sec).
        cadence_to_2min = math.sqrt(120.0 / dt_sec)

        if flux_err is not None:
            err_arr = np.asarray(flux_err)
            if err_arr.shape != (n_cad, ny, nx):
                raise ValueError(f"flux_err shape {err_arr.shape} does not match {(n_cad, ny, nx)}.")
            err_sub = np.asarray(err_arr[final_idx][:, cand_r, cand_c], dtype=np.float64)
            if flux_err_scale != 1.0:
                err_sub = err_sub * flux_err_scale
            for k in range(cand_r.size):
                pos_e = err_sub[:, k]
                pos_e = pos_e[np.isfinite(pos_e) & (pos_e > 0.0)]
                if pos_e.size > 0:
                    snr_2min = (mean_cand[k] / float(np.median(pos_e))) * cadence_to_2min
                    if snr_2min < float(min_snr):
                        snr_excl[cand_r[k], cand_c[k]] = True
        else:
            bg_shot = np.zeros(cand_r.size, dtype=np.float64)
            if bkg_arr is not None and bkgsub:
                bkg_sub = np.asarray(bkg_arr[final_idx][:, cand_r, cand_c], dtype=np.float64)
                if flux_bkg_scale != 1.0:
                    bkg_sub = bkg_sub * flux_bkg_scale
                for k in range(cand_r.size):
                    fin_b = bkg_sub[:, k]
                    fin_b = fin_b[np.isfinite(fin_b)]
                    if fin_b.size > 0:
                        bg_shot[k] = max(float(np.median(fin_b)), 0.0)
            shot_err_2min = np.sqrt((np.maximum(mean_cand, 0.0) + bg_shot) / 120.0)
            snr_2min = np.where(shot_err_2min > 0.0, mean_cand / shot_err_2min, 0.0)
            snr_excl[cand_r, cand_c] = snr_2min < float(min_snr)

    final_pix = cand_pix & ~snr_excl
    if not np.any(final_pix):
        raise ValueError("No valid halo pixels survived spatial masking and min_snr filtering.")

    return HaloMaskResult(
        pixel_mask=final_pix,
        cadence_mask=cad_mask,
        saturated_mask=sat_mask,
        quality_excluded_mask=qual_excl,
        snr_excluded_mask=snr_excl,
        source_excluded_mask=source_excl,
    )
