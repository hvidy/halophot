"""Spatial halo masking, sky-dominance guards, and WCS contaminant-star exclusion.

Implements Section 5.2 and Section 6.1 aperture selection rules:

* Automatic exclusion of ``PIXEL_QUALITY`` bits ``STRAP_COL | SAT_BLEED |
  SMEAR_UNRESOLVED | SOURCE_MASK | BKG_USED`` so ``Omega_halo`` and ``Omega_bg``
  are strictly disjoint (``Omega_halo cap Omega_bg = empty``).
* 2D saturation + bleed + LDE-undershoot dilation via ``scipy.ndimage.binary_dilation``,
  calibrated to physical single-read CCD full-well rates per frame integration time
  (Vanderspek et al. 2018 for *TESS*; Van Cleve & Caldwell 2016 for *Kepler*/*K2*).
* Sky-dominance guard (``U_bar_i - B_bar_i >= kappa * B_bar_i`` when ``BKGSUB=False``
  and ``P_bar_i >= kappa * sigma_{B,i}`` from ``BKG_CHOL`` / ``FLUX_BKG_ERR`` when
  ``BKGSUB=True``, with ``kappa = 2.0`` default and ``not legacy_tv``) plus pixel
  ``SNR_i >= 5`` preventing TV minimization from diluting astrophysical
  amplitudes on sky-dominated pixels.
* WCS-projected contaminant-star masking with proper-motion propagation, plate-scale
  and flux-contrast scaling from Gaussian PRF encircled energy (Ricker et al. 2015;
  Bryson et al. 2010), supporting pre-cached Astropy ``Table`` inputs for ``--offline``
  HPC execution or live ``astroquery.mast`` queries.
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

# Physical single-frame CCD full-well depth divided by single-frame integration time:
# - TESS CCID-80: 1.5e5 e- full well / 2.0 s frame = 7.5e4 e-/s
#   (Vanderspek et al. 2018, TESS Instrument Handbook, Section 4.2 / Table 4.1).
# - Kepler/K2 e2v CCD90: 1.0e6 e- full well / 6.02 s integration = 1.661e5 e-/s
#   (Van Cleve & Caldwell 2016, Kepler Instrument Handbook, Section 4.2 / Table 13).
TESS_FULL_WELL_ELECTRONS: float = 1.5e5
TESS_FRAME_TIME_SEC: float = 2.0
TESS_SAT_RATE_E_PER_S: float = TESS_FULL_WELL_ELECTRONS / TESS_FRAME_TIME_SEC

KEPLER_FULL_WELL_ELECTRONS: float = 1.0e6
KEPLER_FRAME_TIME_SEC: float = 6.02
KEPLER_SAT_RATE_E_PER_S: float = KEPLER_FULL_WELL_ELECTRONS / KEPLER_FRAME_TIME_SEC

# Pixel-integrated 50% encircled energy in 1x1 pixel (Ricker et al. 2015, JATIS, 1, 014003):
# erf(0.5 / (sqrt(2) * sigma_det))^2 = 0.50 => sigma_det = 0.5 / (sqrt(2) * erfinv(sqrt(0.5))).
_DETECTOR_PRF_SIGMA_PIX: float = float(0.5 / (math.sqrt(2.0) * erfinv(math.sqrt(0.5))))
_OPTICAL_CORE_SIGMA_ARCSEC: float = 2.0
# Zodiacal sky background reference surface brightness (mag / arcsec^2; Ricker et al. 2015)
# and 10^-4 total-flux halo-wing contrast threshold (10.0 mag = -2.5 * log10(1e-4)).
_ZODIACAL_SKY_MU_MAG_ARCSEC2: float = 22.0
_HALO_WING_CONTRAST_MAG: float = 10.0

_MAG_CANDIDATES: tuple[str, ...] = (
    "tmag",
    "phot_g_mean_mag",
    "kepmag",
    "mag",
    "vmag",
    "gmag",
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
    """Coalesce per-row magnitudes across standard optical/NIR bandpass columns.

    Returns ``(mag_array, has_mag_col, primary_col_lower)``. When one or more
    magnitude columns exist, missing ``NaN`` entries in the primary column fall
    back per-row to secondary columns (e.g. ``phot_g_mean_mag`` when ``Tmag``
    is ``NaN``).
    """
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
    """Return True if ``unit_obj`` represents integrated electrons per cadence."""
    if unit_obj is None or unit_obj == u.dimensionless_unscaled:
        return False
    if unit_obj == u.electron:
        return True
    text = str(unit_obj).strip().lower()
    return text in ("electron", "electrons", "e-")


def _is_integrated_count_unit(unit_obj: Any) -> bool:
    """Return True if ``unit_obj`` represents integrated counts/electrons per cadence."""
    if _is_electron_count_unit(unit_obj):
        return True
    if unit_obj is None or unit_obj == u.dimensionless_unscaled:
        return False
    if unit_obj == u.count:
        return True
    text = str(unit_obj).strip().lower()
    return text in ("count", "counts", "ct", "adu")


def _integrated_to_rate_unit(unit_obj: Any) -> u.UnitBase:
    """Return ``e-/s`` for electron count units, or ``count/s`` for ADU/count units."""
    return (u.electron / u.s) if _is_electron_count_unit(unit_obj) else (u.count / u.s)


def _resolve_tpf_column_unit(tpf: Any, col_name: str, attr_unit: Any) -> Any:
    """Resolve column unit from Lightkurve Quantity or raw HDU 1 ``TUNIT*`` header."""
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
    """Resolve ``BUNIT`` from a named FITS extension HDU, falling back to ``fallback_unit``."""
    hdul = getattr(tpf, "hdu", None)
    if hdul is not None and hdu_name in hdul:
        bunit = hdul[hdu_name].header.get("BUNIT")
        if isinstance(bunit, str) and bunit.strip():
            return bunit.strip()
    return fallback_unit


def _resolve_cadence_exposure_sec(tpf: Any, dt_days: float) -> float:
    """Resolve per-cadence effective integration time in seconds for count->rate conversion."""
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
        timedel = hdr.get("TIMEDEL") if hasattr(hdr, "get") else None
        if timedel is not None and not isinstance(timedel, bool):
            try:
                td_val = float(timedel)
            except (TypeError, ValueError):
                td_val = np.nan
            if np.isfinite(td_val) and td_val > 0.0:
                deadc_raw = hdr.get("DEADC", 1.0) if hasattr(hdr, "get") else 1.0
                try:
                    deadc_val = float(deadc_raw)
                except (TypeError, ValueError):
                    deadc_val = 1.0
                if not np.isfinite(deadc_val) or deadc_val <= 0.0 or deadc_val > 1.0:
                    deadc_val = 1.0
                return td_val * 86400.0 * deadc_val

        int_time = hdr.get("INT_TIME") if hasattr(hdr, "get") else None
        n_readout = hdr.get("NREADOUT") if hasattr(hdr, "get") else None
        num_frm = hdr.get("NUM_FRM") if hasattr(hdr, "get") else None
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
    """Query MAST ``Catalogs.query_region`` when ``offline=False``."""
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
    """Compute geometric mean celestial pixel scale in arcsec/pixel from ``wcs``."""
    scales_deg = np.asarray(proj_plane_pixel_scales(wcs.celestial), dtype=np.float64)
    pos_scales = scales_deg[np.isfinite(scales_deg) & (scales_deg > 0.0)]
    if pos_scales.size == 0:
        return 21.0
    return float(np.sqrt(np.prod(pos_scales[:2])) * 3600.0)


def _coerce_obs_time(obs_time: Time | str | float | None, wcs: WCS) -> Time | None:
    """Resolve observation epoch from ``obs_time`` or ``wcs`` header keywords."""
    if obs_time is not None:
        if isinstance(obs_time, Time):
            return Time(float(obs_time.jd), format="jd", scale="tdb")
        if isinstance(obs_time, int | float | np.floating):
            val = float(obs_time)
            if np.isfinite(val):
                # Values < 3000 are Julian years (e.g. 2020.5); > 2.4e6 are JD; else MJD.
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
    """Return ``(coords_raw, coords_j2000, coords_obs)`` propagating proper motions when present."""
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
        lower_names
        & {"source_id", "phot_g_mean_mag", "gmag", "dr3_source_id", "dr2_source_id"}
    )
    has_tic_cols = bool(lower_names & {"tic", "ticid", "posflag"}) or (
        "id" in lower_names and "tmag" in lower_names and "source_id" not in lower_names
    )
    if ra_lower == "raj2000" or dec_lower == "dej2000":
        # CDS/VizieR RAJ2000/DEJ2000 columns are already pre-propagated to J2000.0.
        default_ep = 2000.0
    elif ra_lower == "ra_icrs" or dec_lower in ("de_icrs", "dec_icrs"):
        default_ep = 2016.0
    elif "gaia" in catalog.lower():
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

    pmra_clean = np.where(has_pm, pmra, 0.0)
    pmdec_clean = np.where(has_pm, pmdec, 0.0)
    sc_motion = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        pm_ra_cosdec=pmra_clean * (u.mas / u.yr),
        pm_dec=pmdec_clean * (u.mas / u.yr),
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
) -> float | None:
    """Infer fallback primary target magnitude from TPF header cards or integrated flux."""
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
        flux_arr = np.asarray(flux, dtype=np.float64)
        if flux_arr.ndim == 3 and flux_arr.size > 0:
            any_fin = np.any(np.isfinite(flux_arr), axis=0)
            if np.any(any_fin):
                med_img = np.zeros(flux_arr.shape[1:], dtype=np.float64)
                med_img[any_fin] = np.nanmedian(flux_arr[:, any_fin], axis=0)
                sum_pos = float(np.sum(np.maximum(med_img, 0.0)))
                if np.isfinite(sum_pos) and sum_pos > 0.0:
                    eff_mission = mission
                    if eff_mission is None and tpf is not None:
                        eff_mission = getattr(tpf, "mission", None)
                    mission_upper = str(eff_mission or "").upper()
                    zp = 25.3 if mission_upper in ("KEPLER", "K2") else 20.44
                    return float(zp - 2.5 * math.log10(sum_pos))
    return None


def mask_background_stars(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    strict_hpc: bool = True,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_mag: float | None = None,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    radius_pix: float | None = None,
    obs_time: Time | str | float | None = None,
    catalog: str = "TIC",
) -> np.ndarray:
    """Construct a ``(ny, nx)`` boolean mask of background stars projected via ``wcs``.

    Contaminant exclusion thresholds and pixel radii are derived from WCS plate scale
    and Gaussian PRF encircled energy:

    * **Sky & halo-wing contrast cut**: When ``mag_limit`` is ``None``, stars fainter
      than ``min(m_target + 10.0, mu_sky - 2.5 * log10(s_pix^2 / 0.5))`` (``14.6`` mag
      at *TESS* ``21"/px``, ``17.9`` mag at *Kepler*/*K2* ``3.98"/px``) are below sky
      or ``10^-4`` of the primary target flux and are not masked.
    * **Gaussian PRF radius**: When ``radius_pix`` is ``None``, a contaminant of
      magnitude ``m_s`` is masked out to ``sqrt(2 * sigma_prf^2 * (ln 10 / 2.5) *
      (m_lim - m_s) + 0.5)`` pixels, where ``sigma_prf`` combines the ``2"`` optical
      core with the ``0.678``-pixel 50%-encircled-energy detector core.
    * **Proper-motion & target exclusion**: Propagates ``pmra``/``pmdec`` when present
      and matches ``target_coord`` across both J2000 and observation epochs so nearby
      high-proper-motion halo stars are never self-masked.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"Invalid stamp shape {shape}; expected positive (ny, nx).")
    if wcs is None or not wcs.has_celestial:
        raise ValueError("A valid celestial astropy.wcs.WCS is required to project star masks.")

    pix_scale_arcsec = _wcs_pixel_scale_arcsec(wcs)
    sigma_prf_pix = math.hypot(_OPTICAL_CORE_SIGMA_ARCSEC / pix_scale_arcsec, _DETECTOR_PRF_SIGMA_PIX)
    resolved_obs_time = _coerce_obs_time(obs_time, wcs)

    if catalog_table is None:
        resolved_cache = resolve_cache_dir(
            cache_dir,
            strict_hpc=strict_hpc,
            create=not offline,
        )
        if offline:
            raise RuntimeError(
                "Live catalogue queries are disabled when offline=True; "
                "pass a pre-cached Astropy Table via catalog_table."
            )
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
        resolve_cache_dir(cache_dir, strict_hpc=strict_hpc, create=False)
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
        # Never retain unmeasured NaN-magnitude rows when magnitude columns are present.
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
                        # Bandpass-aware tolerance between inferred T/Kp target_mag and
                        # optical catalogue columns: cool M giants can have G - T ~ 4 mag
                        # and V - T ~ 6 mag, whereas a missing G < 3 naked-eye primary leaves
                        # faint G ~ 12-14 field neighbours > 8 mag fainter.
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
                            # Always honour an explicit caller-supplied exclusion radius.
                            keep = ~in_excl
                            coords_obs = coords_obs[keep]
                            mag_vals = mag_vals[keep]
                        elif valid_primary:
                            # Exclude only the primary star (and duplicate entries within 2")
                            # so resolved fainter neighbours inside 1.5 TESS pixels remain masked.
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

    # Resolve sky- and contrast-calibrated magnitude limit when mag_limit is None.
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
        radii = np.sqrt(
            2.0 * (sigma_prf_pix**2) * (math.log(10.0) / 2.5) * delta_m + 0.5
        )
    else:
        radii = np.full(col_pix.shape, max(2.0 * sigma_prf_pix, 1.0), dtype=np.float64)

    # Do not mask blended visual-binary companions (e.g. alpha Cen A/B, Castor A/B) inside
    # the target core radius whose contaminant disc would engulf the target coordinate itself.
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


def _resolve_tpf_obs_time(tpf: Any) -> Time | None:
    """Extract mid-observation ``astropy.time.Time`` in standard JD (TDB) from a TPF."""
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
    strict_hpc: bool = True,
    target_coord: SkyCoord | tuple[float, float] | None = None,
    target_mag: float | None = None,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    radius_pix: float | None = None,
    obs_time: Time | str | float | None = None,
    catalog: str = "TIC",
    mission: str | None = None,
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
        resolved_obs_time = obs_time if obs_time is not None else _resolve_tpf_obs_time(tpf)
        flux_vals_tpf = np.asarray(
            tpf.flux.value if hasattr(tpf.flux, "value") else tpf.flux,
            dtype=np.float64,
        )
        tpf_flux_unit = _resolve_tpf_column_unit(
            tpf, "FLUX", getattr(tpf.flux, "unit", None)
        )
        if _is_integrated_count_unit(tpf_flux_unit):
            t_attr = getattr(tpf, "time", None)
            dt_tpf = 1.0 / 48.0
            if t_attr is not None and len(t_attr) > 1:
                tv = np.asarray(t_attr.value, dtype=np.float64)
                tv_fin = tv[np.isfinite(tv)]
                d_tv = np.diff(tv_fin)
                pos_d = d_tv[d_tv > 0.0]
                if pos_d.size > 0:
                    dt_tpf = float(np.median(pos_d))
            exp_sec_tpf = _resolve_cadence_exposure_sec(tpf, dt_tpf)
            flux_vals_tpf = flux_vals_tpf / exp_sec_tpf
        eff_mission_tpf = mission if mission is not None else getattr(tpf, "mission", None)
        resolved_target_mag = (
            target_mag
            if target_mag is not None
            else _estimate_target_mag(
                tpf=tpf,
                flux=flux_vals_tpf,
                mission=eff_mission_tpf,
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


def build_halo_masks(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    flux_bkg: np.ndarray | None = None,
    flux_bkg_err: np.ndarray | None = None,
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
    strict_hpc: bool = True,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    star_radius_pix: float | None = None,
    thresh: float = -1.0,
    auto_sat_level: float | None = None,
    mission: str | None = None,
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

    flux_cad = flux_arr[cad_mask]
    any_finite_pix = np.any(np.isfinite(flux_cad), axis=0)
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
            else _estimate_target_mag(flux=flux_cad, mission=mission)
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
        resolve_cache_dir(cache_dir, strict_hpc=strict_hpc, create=False)
        source_excl = np.zeros((ny, nx), dtype=bool)

    bkg_arr: np.ndarray | None = None
    if flux_bkg is not None:
        bkg_arr = np.asarray(flux_bkg, dtype=np.float64)
        if bkg_arr.shape != (n_cad, ny, nx):
            raise ValueError(
                f"flux_bkg shape {bkg_arr.shape} does not match {(n_cad, ny, nx)}."
            )

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
        # Automatic physical full-well saturation guard using 99th-percentile unsubtracted flux
        # to catch partial-sector bleed; always honoured when auto_sat_level is explicitly passed.
        already_flagged = pixel_quality is not None and np.any(
            (np.asarray(pixel_quality, dtype=np.uint16) & np.uint16(PixelQuality.SAT_BLEED)) != 0
        )
        if auto_sat_level is not None:
            resolved_sat = float(auto_sat_level)
            apply_auto_sat = resolved_sat > 0.0
        else:
            mission_upper = str(mission or "").upper()
            resolved_sat = (
                KEPLER_SAT_RATE_E_PER_S
                if mission_upper in ("KEPLER", "K2")
                else TESS_SAT_RATE_E_PER_S
            )
            apply_auto_sat = (not already_flagged) and (resolved_sat > 0.0)

        if apply_auto_sat:
            if bkgsub and bkg_arr is not None:
                bkg_cad = bkg_arr[cad_mask]
                flux_for_sat = np.where(
                    np.isfinite(bkg_cad),
                    flux_cad + np.maximum(bkg_cad, 0.0),
                    flux_cad,
                )
            else:
                flux_for_sat = flux_cad
            p99_img = np.full((ny, nx), np.nan, dtype=np.float64)
            p99_img[any_finite_pix] = np.nanpercentile(
                flux_for_sat[:, any_finite_pix], 99.0, axis=0
            )
            sat_raw = np.isfinite(p99_img) & (p99_img >= resolved_sat)
            sat_mask = ndimage.binary_dilation(sat_raw, structure=structure_2d)
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

        if sky_kappa > 0.0:
            if not bkgsub:
                # Unsubtracted FLUX = P + B: enforce P_bar = U_bar - B_bar >= kappa * B_bar
                # so starlight fraction P / (P + B) >= kappa / (1 + kappa).
                if bkg_arr is not None:
                    bkg_sub = bkg_arr[cad_mask]
                    fin_bkg = bkg_sub[np.isfinite(bkg_sub)]
                    if fin_bkg.size > 0 and np.any(fin_bkg != 0.0):
                        mean_bkg = np.zeros((ny, nx), dtype=np.float64)
                        for r_idx, c_idx in zip(*np.where(cand_pix), strict=True):
                            col_b = bkg_sub[:, r_idx, c_idx]
                            fin_col = col_b[np.isfinite(col_b)]
                            if fin_col.size > 0:
                                mean_bkg[r_idx, c_idx] = max(float(np.mean(fin_col)), 0.0)
                        sky_excl |= cand_pix & ((mean_p - mean_bkg) < float(sky_kappa) * mean_bkg)
            elif flux_bkg_err is not None:
                # Background-subtracted FLUX = P: guard against background-model uncertainty
                # sigma_{B,i} (from FLUX_BKG_ERR or BKG_CHOL + BKG_BASIS) continuously without
                # rejecting clean outer wings under already-subtracted Earthshine.
                fbe_arr = np.asarray(flux_bkg_err, dtype=np.float64)
                if fbe_arr.shape == (ny, nx):
                    med_fbe = np.where(np.isfinite(fbe_arr) & (fbe_arr > 0.0), fbe_arr, 0.0)
                    if np.any(med_fbe[cand_pix] > 0.0):
                        sky_excl |= cand_pix & (mean_p < float(sky_kappa) * med_fbe)
                elif fbe_arr.shape == (n_cad, ny, nx):
                    fbe_sub = fbe_arr[cad_mask]
                    fin_fbe = fbe_sub[np.isfinite(fbe_sub) & (fbe_sub > 0.0)]
                    if fin_fbe.size > 0:
                        med_fbe = np.zeros((ny, nx), dtype=np.float64)
                        for r_idx, c_idx in zip(*np.where(cand_pix), strict=True):
                            fbe_col = fbe_sub[:, r_idx, c_idx]
                            pos_fbe = fbe_col[np.isfinite(fbe_col) & (fbe_col > 0.0)]
                            if pos_fbe.size > 0:
                                med_fbe[r_idx, c_idx] = float(np.median(pos_fbe))
                        sky_excl |= cand_pix & (mean_p < float(sky_kappa) * med_fbe)
                else:
                    raise ValueError(
                        f"flux_bkg_err shape {fbe_arr.shape} does not match "
                        f"{(n_cad, ny, nx)} or {(ny, nx)}."
                    )

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
                else:
                    eff_dt_days = (
                        float(dt_days)
                        if (dt_days is not None and float(dt_days) > 0.0)
                        else (1.0 / 48.0)
                    )
                    dt_sec = eff_dt_days * 86400.0
                    bg_shot = np.zeros((ny, nx), dtype=np.float64)
                    if bkg_arr is not None and bkgsub:
                        bkg_sub_snr = bkg_arr[cad_mask]
                        for r_idx, c_idx in zip(*np.where(active_for_snr), strict=True):
                            b_col = bkg_sub_snr[:, r_idx, c_idx]
                            fin_b = b_col[np.isfinite(b_col)]
                            if fin_b.size > 0:
                                bg_shot[r_idx, c_idx] = max(float(np.median(fin_b)), 0.0)
                    shot_err = np.sqrt(
                        (np.maximum(mean_p, 0.0) + bg_shot) / max(dt_sec, 1.0)
                    )
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
