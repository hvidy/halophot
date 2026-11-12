"""Functional ``lightkurve`` API and unified coherent ``flux_err`` propagation.

Implements Section 5.1 and Section 6.4 of the ``halophot`` v2.0 architecture:

* :func:`column_orthogonal_smear_weights`: projects column weights
  ``g_c = sum_{r} W_{r,c}`` orthogonal to degree-``<= d`` horizontal column
  polynomials already cancelled by ``tessbkgd`` when ``BKGSUB=True``.
* :func:`compute_coherent_flux_err`: exact ``float64`` error propagation combining
  uncorrelated pixel variance, projected column-coherent ``SMEAR_ERR``, and
  ``BKG_CHOL`` / ``BKG_BASIS`` sandwich background covariance without double-counting.
* :func:`extract_halo_lightcurve`: functional ``TargetPixelFile -> LightCurve``
  entry point returning standard ``KeplerLightCurve`` / ``TessLightCurve`` objects
  with genuine ``estimate_centroids()`` centroids, separate ``pos_corr1`` /
  ``pos_corr2`` columns, and non-transposed ``(ny, nx)`` (or ``(n_splits, ny, nx)``)
  ``lc.meta['WEIGHTMAP']``.
"""

from __future__ import annotations

import os
from typing import Any

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from lightkurve.targetpixelfile import (
    KeplerTargetPixelFile,
    TargetPixelFile,
    TessTargetPixelFile,
)

from ._contract import (
    BkgCovariance,
    PixelQuality,
    infer_bkg_poly_order,
    read_bkg_covariance,
    read_bkgsub_state,
    read_pixel_quality,
    read_smear_err,
)
from ._precision import in_float64
from .masking import (
    DEFAULT_EXCLUDE_QUALITY,
    _resolve_tpf_obs_time,
    _resolve_tpf_wcs,
    build_halo_masks,
)
from .optimise import optimise_halo

__all__ = [
    "column_orthogonal_smear_weights",
    "compute_coherent_flux_err",
    "extract_halo_lightcurve",
]

_ERR_CHUNK_CADENCES: int = 8192


def column_orthogonal_smear_weights(
    col_weights: np.ndarray,
    bkg_order: int | None,
) -> np.ndarray:
    """Compute ``g_perp = (I - Pi_{<= d}^{col}) g`` in ``float64``.

    When ``bkg_order`` is ``None`` (``BKGSUB=False``), returns ``g`` unchanged.
    When ``bkg_order = d >= 0``, projects ``g`` orthogonal to all 1D polynomials
    in column index ``c`` of degree ``<= d`` using a Householder QR basis of
    Chebyshev polynomials on ``[-1, 1]``.
    """
    g = np.asarray(col_weights, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError(f"col_weights must be 1D (nx,), got shape {g.shape}.")
    nx = g.shape[0]
    if bkg_order is None or bkg_order < 0 or nx == 0:
        return g.copy()
    if bkg_order >= nx - 1:
        return np.zeros_like(g)

    xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    vander = np.polynomial.chebyshev.chebvander(xi, int(bkg_order))
    q_mat, _ = np.linalg.qr(vander, mode="reduced")
    return g - q_mat @ (q_mat.T @ g)


@in_float64
def compute_coherent_flux_err(
    weightmap_2d: np.ndarray,
    *,
    flux_err: np.ndarray | None,
    smear_err: np.ndarray | None = None,
    bkg_cov: BkgCovariance | None = None,
    flux_bkg_err: np.ndarray | None = None,
    bkgsub: bool = True,
    bkg_order: int | None = None,
    norm_factor: float = 1.0,
) -> np.ndarray:
    """Propagate coherent halo light-curve uncertainties ``sigma_{f,t}`` in ``float64``.

    Evaluates Eq. (6.4.1) without double-counting:

    .. math::

        \\sigma_{f,t}^2 =
            \\sum_{i=1}^M (w_i^\\star)^2 \\sigma_{\\text{pix},i,t}^2
          + \\sum_{c=0}^{n_x-1} g_{\\perp,c}^2 \\sigma_{\\text{smear},c,t}^2
          + \\Bigl\\lVert L_t^\\top \\sum_{i=1}^M w_i^\\star \\phi_i \\Bigr\\rVert_2^2,

    where ``sigma_{pix,i,t}^2 = max(FLUX_ERR_{i,t}^2 - sigma_{smear,c(i),t}^2 - sigma_{B,i,t}^2, 0)``
    and ``g_perp = (I - Pi_{<= d}^{col}) g`` when ``bkgsub=True`` (``g_perp = g``
    when ``bkgsub=False``). Divides the resulting ``sigma_{f,t}`` by ``norm_factor``
    (``P_bar.T @ w_star`` for unit-mean normalized light curves, or ``1.0`` for
    physical flux units).
    """
    w2d = np.asarray(weightmap_2d, dtype=np.float64)
    if w2d.ndim != 2:
        raise ValueError(f"weightmap_2d must be 2D (ny, nx), got shape {w2d.shape}.")
    ny, nx = w2d.shape
    w_clean = np.where(np.isfinite(w2d) & (w2d > 0.0), w2d, 0.0)
    rows_h, cols_h = np.where(w_clean > 0.0)
    w_star = w_clean[rows_h, cols_h]

    # Determine cadence count from whichever per-cadence uncertainty array is supplied.
    n_cad: int | None = None
    for arr_cand in (
        flux_err,
        smear_err,
        None if bkg_cov is None else bkg_cov.chol,
        flux_bkg_err,
    ):
        if arr_cand is not None:
            n_cad = int(np.asarray(arr_cand).shape[0])
            break
    if n_cad is None:
        raise ValueError("At least one per-cadence uncertainty array must be provided.")
    if w_star.size == 0:
        return np.full(n_cad, np.nan, dtype=np.float64)

    if flux_err is None and smear_err is None and bkg_cov is None and flux_bkg_err is None:
        return np.full(n_cad, np.nan, dtype=np.float64)

    fe_arr: np.ndarray | None = None
    if flux_err is not None:
        fe_arr = np.asarray(flux_err, dtype=np.float64)
        if fe_arr.shape != (n_cad, ny, nx):
            raise ValueError(
                f"flux_err shape {fe_arr.shape} does not match expected {(n_cad, ny, nx)}."
            )

    se_arr: np.ndarray | None = None
    if smear_err is not None:
        se_arr = np.asarray(smear_err, dtype=np.float64)
        if se_arr.shape != (n_cad, nx):
            raise ValueError(
                f"smear_err shape {se_arr.shape} does not match expected {(n_cad, nx)}."
            )

    fbe_arr: np.ndarray | None = None
    if flux_bkg_err is not None:
        fbe_arr = np.asarray(flux_bkg_err, dtype=np.float64)
        if fbe_arr.shape != (n_cad, ny, nx):
            raise ValueError(
                f"flux_bkg_err shape {fbe_arr.shape} does not match expected {(n_cad, ny, nx)}."
            )

    chol_arr: np.ndarray | None = None
    phi_halo: np.ndarray | None = None
    if bkg_cov is not None and bkgsub:
        chol_arr = np.asarray(bkg_cov.chol, dtype=np.float64)
        basis_arr = np.asarray(bkg_cov.basis, dtype=np.float64)
        if chol_arr.shape[0] != n_cad or basis_arr.shape[1:] != (ny, nx):
            raise ValueError(
                "bkg_cov shapes are inconsistent with (n_cad, ny, nx) = "
                f"{(n_cad, ny, nx)}: chol={chol_arr.shape}, basis={basis_arr.shape}."
            )
        phi_halo = basis_arr[:, rows_h, cols_h]  # (P, M)

    w_sq = np.square(w_star)
    var_total = np.zeros(n_cad, dtype=np.float64)

    # 1. Uncorrelated pixel-noise term in bounded chunks of _ERR_CHUNK_CADENCES.
    if fe_arr is not None:
        for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
            stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
            fe_sub = fe_arr[start:stop, rows_h, cols_h]  # (chunk, M)
            fe_sq = np.square(fe_sub)

            smear_pix_sq = (
                np.square(se_arr[start:stop][:, cols_h])
                if se_arr is not None
                else 0.0
            )
            if isinstance(smear_pix_sq, np.ndarray):
                smear_pix_sq = np.where(np.isfinite(smear_pix_sq), smear_pix_sq, 0.0)

            if chol_arr is not None and phi_halo is not None:
                chol_sub = chol_arr[start:stop]  # (chunk, P, P)
                # (chunk, P, P)^T @ (P, M) -> (chunk, P, M), squared and summed over P -> (chunk, M)
                lt_phi = np.einsum("tqp,qm->tpm", chol_sub, phi_halo)
                bkg_pix_sq = np.sum(np.square(lt_phi), axis=1)
                # Preserve NaN on cadences where BKG_CHOL is all-NaN; otherwise treat finite.
                nan_chol = np.all(np.isnan(chol_sub), axis=(1, 2))
                bkg_pix_sq = np.where(nan_chol[:, None], np.nan, bkg_pix_sq)
            elif bkgsub and fbe_arr is not None:
                fbe_sub = fbe_arr[start:stop, rows_h, cols_h]
                bkg_pix_sq = np.where(np.isfinite(fbe_sub), np.square(fbe_sub), 0.0)
            else:
                bkg_pix_sq = 0.0

            sigma_pix_sq = np.maximum(fe_sq - smear_pix_sq - bkg_pix_sq, 0.0)
            # Propagate NaN if any active halo pixel has NaN FLUX_ERR or NaN BKG_CHOL on cadence t.
            bad_cad = np.any(~np.isfinite(fe_sub), axis=1)
            if isinstance(bkg_pix_sq, np.ndarray):
                bad_cad |= np.any(~np.isfinite(bkg_pix_sq), axis=1)
            pix_term = sigma_pix_sq @ w_sq
            pix_term[bad_cad] = np.nan
            var_total[start:stop] += pix_term

    # 2. Column-coherent smear term with degree-<= d orthogonal projection when BKGSUB=True.
    if se_arr is not None:
        col_g = np.sum(w_clean, axis=0)  # (nx,)
        eff_order = bkg_order if bkgsub else None
        g_perp = column_orthogonal_smear_weights(col_g, eff_order)
        active_cols = np.abs(g_perp) > 0.0
        if np.any(active_cols):
            se_act = se_arr[:, active_cols]
            g_sq = np.square(g_perp[active_cols])
            smear_term = np.square(se_act) @ g_sq
            smear_term[np.any(~np.isfinite(se_act), axis=1)] = np.nan
            var_total += smear_term

    # 3. Coherent background covariance term ||L_t.T @ sum_i w_i * phi_i||_2^2.
    if chol_arr is not None and phi_halo is not None:
        phi_w = phi_halo @ w_star  # (P,)
        lt_phi_w = np.einsum("tqp,q->tp", chol_arr, phi_w)
        bkg_coh_term = np.sum(np.square(lt_phi_w), axis=1)
        nan_chol_all = np.all(np.isnan(chol_arr), axis=(1, 2))
        bkg_coh_term[nan_chol_all] = np.nan
        var_total += bkg_coh_term
    elif bkgsub and fbe_arr is not None:
        for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
            stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
            fbe_sub = fbe_arr[start:stop, rows_h, cols_h]
            fin = np.isfinite(fbe_sub)
            fbe_clean = np.where(fin, np.maximum(fbe_sub, 0.0), 0.0)
            var_total[start:stop] += np.square(fbe_clean @ w_star)

    scale = abs(float(norm_factor))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"norm_factor must be finite and non-zero, got {norm_factor!r}.")

    return np.sqrt(np.maximum(var_total, 0.0)) / scale


def _strip_quantity(val: Any) -> np.ndarray:
    if hasattr(val, "value"):
        return np.asarray(val.value, dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


def _coerce_tpf(tpf_or_path: Any) -> TargetPixelFile:
    if isinstance(tpf_or_path, TargetPixelFile):
        return tpf_or_path
    if isinstance(tpf_or_path, (str, os.PathLike, fits.HDUList)):
        loaded = lk.read(tpf_or_path)
        if isinstance(loaded, TargetPixelFile):
            return loaded
        raise TypeError(
            f"Input resolved to {type(loaded).__name__}, not a 3D lightkurve.TargetPixelFile."
        )
    raise TypeError(
        f"Expected a lightkurve.TargetPixelFile or TPF FITS path, got {type(tpf_or_path).__name__}."
    )


def _construct_lightcurve_from_tpf(
    tpf: TargetPixelFile,
    *,
    flux: np.ndarray | u.Quantity,
    flux_err: np.ndarray | u.Quantity,
    centroid_col: Any,
    centroid_row: Any,
    extra_meta: dict[str, Any],
) -> lk.LightCurve:
    """Build a ``KeplerLightCurve`` or ``TessLightCurve`` matching ``tpf`` metadata."""
    mission_upper = str(getattr(tpf, "mission", "") or "").upper()
    time_obj = Time(tpf.time.value, format=tpf.time.format, scale=tpf.time.scale)

    base_meta = dict(tpf.meta) if getattr(tpf, "meta", None) is not None else {}
    base_meta.update(extra_meta)

    common_keys: dict[str, Any] = {
        "centroid_col": centroid_col,
        "centroid_row": centroid_row,
        "quality": tpf.quality,
        "cadenceno": tpf.cadenceno,
        "ra": tpf.ra,
        "dec": tpf.dec,
        "label": getattr(tpf, "get_keyword", lambda k: None)("OBJECT"),
        "targetid": tpf.targetid,
        "mission": tpf.mission,
        "meta": base_meta,
    }

    if isinstance(tpf, TessTargetPixelFile) or mission_upper == "TESS":
        tess_keys = {
            **common_keys,
            "sector": getattr(tpf, "sector", None),
            "camera": getattr(tpf, "camera", None),
            "ccd": getattr(tpf, "ccd", None),
        }
        lc: lk.LightCurve = lk.TessLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            **tess_keys,
        )
    elif isinstance(tpf, KeplerTargetPixelFile) or mission_upper in ("KEPLER", "K2"):
        kep_keys = {
            **common_keys,
            "channel": getattr(tpf, "channel", None),
            "campaign": getattr(tpf, "campaign", None),
            "quarter": getattr(tpf, "quarter", None),
        }
        lc = lk.KeplerLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            **kep_keys,
        )
    else:
        lc = lk.LightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            meta=base_meta,
        )

    if hasattr(tpf, "pos_corr1") and tpf.pos_corr1 is not None:
        lc["pos_corr1"] = tpf.pos_corr1
    if hasattr(tpf, "pos_corr2") and tpf.pos_corr2 is not None:
        lc["pos_corr2"] = tpf.pos_corr2

    return lc


@in_float64
def extract_halo_lightcurve(
    tpf: Any,
    *,
    aperture_mask: str | np.ndarray = "all",
    rr: tuple[float, float] | None = None,
    split_times: np.ndarray | list[float] | tuple[float, ...] | None = None,
    sub: int = 1,
    objective: str = "tv",
    lag: int = 1,
    tau: float | None = None,
    epsilon: float = 1e-5,
    legacy_tv: bool = False,
    lambda_phot: float | None = None,
    lambda_z: float | None = None,
    solver: str = "lbfgs",
    maxiter: int = 151,
    sigclip: bool = False,
    window_days: float = 1.0,
    sigma_lower: float = 10.0,
    sigma_upper: float = 3.0,
    thresh: float = -1.0,
    auto_sat_level: float | None = None,
    bleed_dilation: tuple[int, int] = (3, 3),
    minflux: float = -100.0,
    sky_kappa: float = 2.0,
    min_snr: float = 5.0,
    exclude_quality: PixelQuality = DEFAULT_EXCLUDE_QUALITY,
    remove_contaminants: bool = False,
    catalog_table: Table | str | os.PathLike[str] | None = None,
    offline: bool = False,
    target_exclusion_arcsec: float | None = None,
    mag_limit: float | None = None,
    star_radius_pix: float | None = None,
    w_init: np.ndarray | None = None,
    compute_uncertainty: bool = False,
    jackknife_blocks: int = 4,
    normalize: bool = True,
) -> lk.LightCurve:
    """Extract a halo-photometry ``lightkurve.LightCurve`` with coherent ``flux_err``.

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile or path-like
        Input calibrated target pixel file (standard MAST SPOC / Kepler / K2 TPF or
        ``HALOCVER = 1`` ``mundey`` + ``tessbkgd`` TPF).
    aperture_mask : ``'all'``, ``'pipeline'``, ``'threshold'``, or bool ``(ny, nx)``
        Initial candidate spatial mask (defaults to ``'all'`` because saturated halo
        wings routinely extend well outside pipeline core apertures).
    split_times : sequence of float, optional
        Timestamps strictly inside ``(min(tpf.time), max(tpf.time))`` splitting the
        observation into independent halo fits, stitched non-mutatingly via
        ``lk.LightCurveCollection.stitch()`` with stacked ``(n_splits, ny, nx)``
        ``lc.meta['WEIGHTMAP']``.
    legacy_tv : bool, default False
        When ``True``, reproduces unregularized ``w``-simplex total variation
        (``lambda_phot = 0``, ``lambda_z = 0``, sky/SNR guards disabled).
    compute_uncertainty : bool, default False
        When ``True``, computes ``K``-block orbit jackknife pixel weight uncertainties
        stored in ``lc.meta['WEIGHTMAP_ERR']``.
    normalize : bool, default True
        When ``True``, returns unit-mean dimensionless relative ``flux`` and
        ``flux_err`` (Eq. 6.4.1). When ``False``, preserves ``tpf.flux.unit``.
    """
    tpf_obj = _coerce_tpf(tpf)
    if isinstance(sub, bool) or int(sub) < 1:
        raise ValueError(f"sub must be an integer >= 1, got {sub!r}.")
    sub_int = int(sub)

    n_total = len(tpf_obj.time)
    if n_total < 2:
        raise ValueError(f"TargetPixelFile must have at least 2 cadences, got {n_total}.")

    # Multi-segment split workflow: validate both lower and upper split_times bounds,
    # run independent halo fits on non-mutating TPF slices, and stitch via LightCurveCollection.
    if split_times is not None:
        split_arr = np.sort(np.asarray(split_times, dtype=np.float64).ravel())
        if split_arr.size < 1 or not np.all(np.isfinite(split_arr)):
            raise ValueError("split_times must contain at least one finite timestamp.")

        t_all = np.asarray(tpf_obj.time.value, dtype=np.float64)
        fin_t = t_all[np.isfinite(t_all)]
        if fin_t.size < 2:
            raise ValueError("TargetPixelFile contains fewer than 2 finite timestamps.")
        t_min, t_max = float(np.min(fin_t)), float(np.max(fin_t))
        if split_arr[0] <= t_min or split_arr[-1] >= t_max:
            raise ValueError(
                f"All split_times {split_arr.tolist()} must lie strictly inside "
                f"the TPF time span ({t_min}, {t_max})."
            )

        split_indices = [int(np.min(np.where(t_all > st))) for st in split_arr]
        if split_indices[0] <= 0 or split_indices[-1] >= n_total:
            raise ValueError(
                f"split_times map to boundary cadence indices {split_indices} outside (0, {n_total})."
            )
        if len(set(split_indices)) != len(split_indices):
            raise ValueError(
                f"split_times map to duplicate cadence indices {split_indices}."
            )
        if w_init is not None and np.asarray(w_init).ndim == 1:
            raise ValueError(
                "When split_times is provided, w_init must be None or a 2D (ny, nx) weightmap."
            )

        boundaries: list[int | None] = [None, *split_indices, None]
        seg_lcs: list[lk.LightCurve] = []
        seg_wmaps: list[np.ndarray] = []
        seg_werrs: list[np.ndarray] = []

        for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
            tpf_seg = tpf_obj[slice(low, high)]
            lc_seg = extract_halo_lightcurve(
                tpf_seg,
                aperture_mask=aperture_mask,
                rr=rr,
                split_times=None,
                sub=sub_int,
                objective=objective,
                lag=lag,
                tau=tau,
                epsilon=epsilon,
                legacy_tv=legacy_tv,
                lambda_phot=lambda_phot,
                lambda_z=lambda_z,
                solver=solver,
                maxiter=maxiter,
                sigclip=sigclip,
                window_days=window_days,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                thresh=thresh,
                auto_sat_level=auto_sat_level,
                bleed_dilation=bleed_dilation,
                minflux=minflux,
                sky_kappa=sky_kappa,
                min_snr=min_snr,
                exclude_quality=exclude_quality,
                remove_contaminants=remove_contaminants,
                catalog_table=catalog_table,
                offline=offline,
                target_exclusion_arcsec=target_exclusion_arcsec,
                mag_limit=mag_limit,
                star_radius_pix=star_radius_pix,
                w_init=w_init,
                compute_uncertainty=compute_uncertainty,
                jackknife_blocks=jackknife_blocks,
                normalize=normalize,
            )
            seg_wmaps.append(np.asarray(lc_seg.meta["WEIGHTMAP"], dtype=np.float64))
            if compute_uncertainty and "WEIGHTMAP_ERR" in lc_seg.meta:
                seg_werrs.append(np.asarray(lc_seg.meta["WEIGHTMAP_ERR"], dtype=np.float64))
            seg_lcs.append(lc_seg)

        stitched = lk.LightCurveCollection([seg.copy() for seg in seg_lcs]).stitch(
            corrector_func=None
        )
        stitched.meta = dict(stitched.meta)
        stitched.meta["WEIGHTMAP"] = np.stack(seg_wmaps, axis=0)
        stitched.meta["SPLIT_TIMES"] = split_arr.tolist()
        if compute_uncertainty and seg_werrs:
            stitched.meta["WEIGHTMAP_ERR"] = np.stack(seg_werrs, axis=0)
        return stitched

    # Single-segment extraction.
    flux_3d = _strip_quantity(tpf_obj.flux).copy()
    n_cad, ny, nx = flux_3d.shape

    parsed_ap = np.asarray(tpf_obj._parse_aperture_mask(aperture_mask), dtype=bool)
    if rr is not None:
        rmin, rmax = float(rr[0]), float(rr[1])
        xs = np.arange(nx, dtype=np.float64) - nx / 2.0
        ys = np.arange(ny, dtype=np.float64) - ny / 2.0
        xx, yy = np.meshgrid(xs, ys)
        rad = np.hypot(xx, yy)
        parsed_ap = parsed_ap & (rad >= rmin) & (rad <= rmax)

    flux_err_3d: np.ndarray | None = None
    if hasattr(tpf_obj, "flux_err") and tpf_obj.flux_err is not None:
        fe_cand = _strip_quantity(tpf_obj.flux_err)
        if np.any(np.isfinite(fe_cand) & (fe_cand > 0.0)):
            flux_err_3d = fe_cand

    flux_bkg_3d: np.ndarray | None = None
    if hasattr(tpf_obj, "flux_bkg") and tpf_obj.flux_bkg is not None:
        flux_bkg_3d = _strip_quantity(tpf_obj.flux_bkg)

    flux_bkg_err_3d: np.ndarray | None = None
    if hasattr(tpf_obj, "flux_bkg_err") and tpf_obj.flux_bkg_err is not None:
        fbe_cand = _strip_quantity(tpf_obj.flux_bkg_err)
        if np.any(np.isfinite(fbe_cand) & (fbe_cand > 0.0)):
            flux_bkg_err_3d = fbe_cand

    pq_2d = read_pixel_quality(tpf_obj)
    bkgsub = read_bkgsub_state(tpf_obj, warn_legacy_mundey=True)
    bkg_order = infer_bkg_poly_order(tpf_obj)
    smear_err_2d = read_smear_err(tpf_obj, apply_quality_mask=True)
    bkg_cov = read_bkg_covariance(tpf_obj, apply_quality_mask=True)

    times_all = np.asarray(tpf_obj.time.value, dtype=np.float64)
    fin_times = times_all[np.isfinite(times_all)]
    diffs_t = np.diff(fin_times)
    pos_diffs = diffs_t[diffs_t > 0.0]
    dt_days = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)

    pc1 = _strip_quantity(tpf_obj.pos_corr1) if getattr(tpf_obj, "pos_corr1", None) is not None else None
    pc2 = _strip_quantity(tpf_obj.pos_corr2) if getattr(tpf_obj, "pos_corr2", None) is not None else None

    tgt_coord: tuple[float, float] | None = None
    if getattr(tpf_obj, "ra", None) is not None and getattr(tpf_obj, "dec", None) is not None:
        tgt_coord = (float(tpf_obj.ra), float(tpf_obj.dec))

    mask_res = build_halo_masks(
        flux_3d,
        flux_err=flux_err_3d,
        flux_bkg=flux_bkg_3d if bkgsub else None,
        pixel_quality=pq_2d,
        aperture_mask=parsed_ap,
        cadence_quality_mask=np.isfinite(times_all),
        pos_corr1=pc1,
        pos_corr2=pc2,
        wcs=_resolve_tpf_wcs(tpf_obj),
        target_coord=tgt_coord,
        obs_time=_resolve_tpf_obs_time(tpf_obj),
        remove_contaminants=remove_contaminants,
        catalog_table=catalog_table,
        offline=offline,
        target_exclusion_arcsec=target_exclusion_arcsec,
        mag_limit=mag_limit,
        star_radius_pix=star_radius_pix,
        thresh=thresh,
        auto_sat_level=auto_sat_level,
        mission=getattr(tpf_obj, "mission", None),
        bleed_dilation=bleed_dilation,
        minflux=minflux,
        sky_kappa=sky_kappa,
        min_snr=min_snr,
        dt_days=dt_days,
        exclude_quality=exclude_quality,
        legacy_tv=legacy_tv,
    )

    rows_all, cols_all = np.where(mask_res.pixel_mask)
    rows_h = rows_all[::sub_int]
    cols_h = cols_all[::sub_int]
    if rows_h.size < 1:
        raise ValueError("No halo pixels remained after subsampling.")

    active_cad = mask_res.cadence_mask
    pixels_active = flux_3d[active_cad][:, rows_h, cols_h].T  # (M, N_good)
    times_active = times_all[active_cad]

    # Per-pixel independent + column-smear variance for photon-noise regularization lambda_phot.
    pvar_reg: np.ndarray | None = None
    if not legacy_tv and (flux_err_3d is not None or smear_err_2d is not None):
        fe_sq = (
            np.square(flux_err_3d[active_cad][:, rows_h, cols_h].T)
            if flux_err_3d is not None
            else np.zeros_like(pixels_active)
        )
        fe_sq = np.where(np.isfinite(fe_sq), fe_sq, 0.0)
        se_sq = (
            np.square(smear_err_2d[active_cad][:, cols_h].T)
            if smear_err_2d is not None
            else np.zeros_like(pixels_active)
        )
        se_sq = np.where(np.isfinite(se_sq), se_sq, 0.0)

        if bkg_cov is not None and bkgsub:
            chol_act = bkg_cov.chol[active_cad]
            phi_act = bkg_cov.basis[:, rows_h, cols_h]
            fin_c = ~np.all(np.isnan(chol_act), axis=(1, 2))
            bkg_sq_mean = np.zeros(rows_h.size, dtype=np.float64)
            if np.any(fin_c):
                lt_p = np.einsum("tqp,qm->tpm", chol_act[fin_c], phi_act)
                bkg_sq_mean = np.mean(np.sum(np.square(lt_p), axis=1), axis=0)
            pvar_reg = np.maximum(np.mean(fe_sq, axis=1) - bkg_sq_mean, np.mean(se_sq, axis=1))
        elif bkgsub and flux_bkg_err_3d is not None:
            fbe_sq = np.square(flux_bkg_err_3d[active_cad][:, rows_h, cols_h].T)
            fbe_sq = np.where(np.isfinite(fbe_sq), fbe_sq, 0.0)
            pvar_reg = np.maximum(fe_sq - fbe_sq, se_sq)
        else:
            pvar_reg = np.maximum(fe_sq, se_sq)

    w_init_vec: np.ndarray | None = None
    if w_init is not None:
        w_arr = np.asarray(w_init, dtype=np.float64)
        if w_arr.ndim == 2:
            if w_arr.shape != (ny, nx):
                raise ValueError(
                    f"2D w_init shape {w_arr.shape} does not match spatial shape {(ny, nx)}."
                )
            extracted = w_arr[rows_h, cols_h]
            clean_ext = np.where(np.isfinite(extracted) & (extracted > 0.0), extracted, 0.0)
            s_ext = float(np.sum(clean_ext))
            w_init_vec = clean_ext / s_ext if s_ext > 0.0 else None
        elif w_arr.ndim == 1:
            if w_arr.shape != (rows_h.size,):
                raise ValueError(
                    f"1D w_init shape {w_arr.shape} does not match active pixel count ({rows_h.size},)."
                )
            w_init_vec = w_arr
        else:
            raise ValueError(f"w_init must be 1D (M,) or 2D (ny, nx), got shape {w_arr.shape}.")

    fit = optimise_halo(
        pixels_active,
        times_active,
        pixel_var=pvar_reg,
        w_init=w_init_vec,
        objective=objective,
        lag=lag,
        tau=tau,
        epsilon=epsilon,
        legacy_tv=legacy_tv,
        lambda_phot=lambda_phot,
        lambda_z=lambda_z,
        solver=solver,
        maxiter=maxiter,
        sigclip=sigclip,
        window_days=window_days,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        compute_uncertainty=compute_uncertainty,
        jackknife_blocks=jackknife_blocks,
    )

    weightmap_2d = np.zeros((ny, nx), dtype=np.float64)
    weightmap_2d[rows_h, cols_h] = fit.physical_weights

    final_cad_mask = np.zeros(n_cad, dtype=bool)
    final_cad_mask[np.flatnonzero(active_cad)[fit.inlier_mask]] = True

    mean_p_inliers = np.mean(flux_3d[final_cad_mask][:, rows_h, cols_h], axis=0)
    mean_flux_norm = float(np.dot(fit.physical_weights, mean_p_inliers))
    if not np.isfinite(mean_flux_norm) or abs(mean_flux_norm) <= 1e-30:
        mean_flux_norm = 1.0

    norm_scale = mean_flux_norm if normalize else 1.0

    # Evaluate full-cadence light curve on valid cadences, masking clipped/NaN cadences to NaN.
    lc_flux = np.full(n_cad, np.nan, dtype=np.float64)
    lc_flux[final_cad_mask] = (
        flux_3d[final_cad_mask][:, rows_h, cols_h] @ fit.physical_weights
    ) / norm_scale

    has_any_err = (
        flux_err_3d is not None
        or smear_err_2d is not None
        or bkg_cov is not None
        or flux_bkg_err_3d is not None
    )
    if has_any_err:
        lc_flux_err = compute_coherent_flux_err(
            weightmap_2d,
            flux_err=flux_err_3d,
            smear_err=smear_err_2d,
            bkg_cov=bkg_cov,
            flux_bkg_err=flux_bkg_err_3d,
            bkgsub=bkgsub,
            bkg_order=bkg_order,
            norm_factor=norm_scale,
        )
        lc_flux_err[~final_cad_mask] = np.nan
    else:
        lc_flux_err = np.full(n_cad, np.nan, dtype=np.float64)

    # Genuine flux-weighted centroids over active halo aperture.
    halo_ap_2d = np.zeros((ny, nx), dtype=bool)
    halo_ap_2d[rows_h, cols_h] = True
    cent_col, cent_row = tpf_obj.estimate_centroids(aperture_mask=halo_ap_2d)

    extra_meta: dict[str, Any] = {
        "WEIGHTMAP": weightmap_2d,
        "HALO_OBJ": objective,
        "HALO_LAG": int(lag),
        "LEGACYTV": bool(legacy_tv),
        "BKGSUB": bool(bkgsub),
    }
    if bkg_order is not None:
        extra_meta["BKG_ORD"] = int(bkg_order)
    if compute_uncertainty and fit.weight_std is not None:
        werr_2d = (
            np.full((ny, nx), np.nan, dtype=np.float64)
            if np.all(np.isnan(fit.weight_std))
            else np.zeros((ny, nx), dtype=np.float64)
        )
        werr_2d[rows_h, cols_h] = fit.weight_std
        extra_meta["WEIGHTMAP_ERR"] = werr_2d

    flux_unit = getattr(tpf_obj.flux, "unit", None)
    if not normalize and flux_unit is not None and flux_unit != u.dimensionless_unscaled:
        flux_arg: np.ndarray | u.Quantity = u.Quantity(lc_flux, unit=flux_unit)
        err_arg: np.ndarray | u.Quantity = u.Quantity(lc_flux_err, unit=flux_unit)
    else:
        flux_arg = lc_flux
        err_arg = lc_flux_err

    return _construct_lightcurve_from_tpf(
        tpf_obj,
        flux=flux_arg,
        flux_err=err_arg,
        centroid_col=cent_col,
        centroid_row=cent_row,
        extra_meta=extra_meta,
    )
