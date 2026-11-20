"""Functional ``lightkurve`` API and unified coherent ``flux_err`` propagation.

Implements Section 5.1 and Section 6.4 of the ``halophot`` v2.0 architecture:

* :func:`column_orthogonal_smear_weights`: projects column weights
  ``g_c = sum_{r} W_{r,c}`` orthogonal to degree-``<= d`` horizontal column
  polynomials already cancelled by ``tessbkgd`` when ``BKGSUB=True``.
* :func:`compute_coherent_flux_err`: exact ``float64`` error propagation combining
  uncorrelated pixel variance (with a ``float32``-ULP floor), exact per-cadence
  WLS oblique column-smear sensitivity (falling back to unweighted OLS /
  orthogonal projection when ``BKG_BASIS`` or ``BKG_USED`` is absent), and
  ``BKG_CHOL`` / ``BKG_BASIS`` sandwich background covariance without double-counting.
* :func:`extract_halo_lightcurve`: functional ``TargetPixelFile -> LightCurve``
  entry point returning standard ``KeplerLightCurve`` / ``TessLightCurve`` objects
  with positive-flux weightmap-weighted centroids, separate ``pos_corr1`` /
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
    KEPLER_SAT_RATE_E_PER_S,
    TESS_SAT_RATE_E_PER_S,
    _estimate_target_mag,
    _integrated_to_rate_unit,
    _is_integrated_count_unit,
    _resolve_cadence_exposure_sec,
    _resolve_hdu_unit,
    _resolve_tpf_column_unit,
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
_F32_EPS: float = float(np.finfo(np.float32).eps)


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
    bg_mask: np.ndarray | None = None,
    bkgsub: bool = True,
    bkg_order: int | None = None,
    norm_factor: float = 1.0,
) -> np.ndarray:
    """Propagate coherent halo light-curve uncertainties ``sigma_{f,t}`` in ``float64``.

    Evaluates Eq. (6.4.1) without double-counting:

    .. math::

        \\sigma_{f,t}^2 =
            \\sum_{i=1}^M (w_i^\\star)^2 \\sigma_{\\text{pix},i,t}^2
          + \\sum_{c=0}^{n_x-1} \\tilde{g}_{t,c}^2 \\sigma_{\\text{smear},c,t}^2
          + \\Bigl\\lVert L_t^\\top \\sum_{i=1}^M w_i^\\star \\phi_i \\Bigr\\rVert_2^2,

    where ``sigma_{pix,i,t}^2`` applies a ``float32``-ULP-aware floor against FITS
    cancellation in ``FLUX_ERR^2 - SMEAR_ERR^2 - sigma_B^2``. When ``bkgsub=True``
    and both ``bkg_cov`` (``BKG_BASIS`` + ``BKG_CHOL``) and ``bg_mask``
    (``PIXEL_QUALITY & PixelQuality.BKG_USED``) are available, ``g_tilde_{t,c}``
    is the exact per-cadence WLS oblique column-smear sensitivity
    ``g_c - phi_w^T G_t^{-1} sum_{r:(r,c) in Omega_bg} w_{t,(r,c)} phi_{(r,c)}``;
    otherwise falls back to :func:`column_orthogonal_smear_weights`. Divides the
    resulting ``sigma_{f,t}`` by ``norm_factor`` (``P_bar.T @ w_star`` for
    unit-mean normalized light curves, or ``1.0`` for physical flux units).
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

    bg_bool: np.ndarray | None = None
    if bg_mask is not None:
        bg_bool = np.asarray(bg_mask, dtype=bool)
        if bg_bool.shape != (ny, nx):
            raise ValueError(
                f"bg_mask shape {bg_bool.shape} does not match expected {(ny, nx)}."
            )

    chol_arr: np.ndarray | None = None
    basis_arr: np.ndarray | None = None
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

            smear_pix_sq: np.ndarray | float = (
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
                bkg_pix_sq: np.ndarray | float = np.sum(np.square(lt_phi), axis=1)
                # Preserve NaN on cadences where BKG_CHOL is all-NaN; otherwise treat finite.
                nan_chol = np.all(np.isnan(chol_sub), axis=(1, 2))
                bkg_pix_sq = np.where(nan_chol[:, None], np.nan, bkg_pix_sq)
            elif bkgsub and fbe_arr is not None:
                fbe_sub = fbe_arr[start:stop, rows_h, cols_h]
                bkg_pix_sq = np.where(np.isfinite(fbe_sub), np.square(fbe_sub), 0.0)
            else:
                bkg_pix_sq = 0.0

            sigma_sub_sq = np.maximum(fe_sq - bkg_pix_sq, 0.0)
            has_subtracted = (smear_pix_sq > 0.0) | (
                np.isfinite(bkg_pix_sq) & (bkg_pix_sq > 0.0)
            )
            ulp_floor = np.where(
                has_subtracted & np.isfinite(fe_sq) & (fe_sq > 0.0),
                0.5 * _F32_EPS * fe_sq,
                0.0,
            )
            sigma_pix_sq = np.maximum(sigma_sub_sq - smear_pix_sq, ulp_floor)
            # Propagate NaN if any active halo pixel has NaN FLUX_ERR or NaN BKG_CHOL on cadence t.
            bad_cad = np.any(~np.isfinite(fe_sub), axis=1)
            if isinstance(bkg_pix_sq, np.ndarray):
                bad_cad |= np.any(~np.isfinite(bkg_pix_sq), axis=1)
            pix_term = sigma_pix_sq @ w_sq
            pix_term[bad_cad] = np.nan
            var_total[start:stop] += pix_term

    # 2. Column-coherent smear term: exact WLS oblique projection when BKG_BASIS + BKG_USED
    #    are available, falling back to unweighted degree-<= d orthogonal projection otherwise.
    if se_arr is not None:
        col_g = np.sum(w_clean, axis=0)  # (nx,)
        use_oblique = (
            bkgsub
            and chol_arr is not None
            and basis_arr is not None
            and phi_halo is not None
            and bg_bool is not None
            and int(np.count_nonzero(bg_bool)) >= basis_arr.shape[0]
        )
        if use_oblique:
            assert chol_arr is not None
            assert basis_arr is not None
            assert phi_halo is not None
            assert bg_bool is not None
            bg_rows, bg_cols = np.where(bg_bool)
            m_bg = bg_rows.size
            n_basis = basis_arr.shape[0]
            phi_bg = basis_arr[:, bg_rows, bg_cols]  # (P, M_bg)
            phi_w = phi_halo @ w_star  # (P,)
            col_inc = np.zeros((m_bg, nx), dtype=np.float64)
            col_inc[np.arange(m_bg), bg_cols] = 1.0

            # Precompute unweighted OLS oblique projection on bg_bool (or column-orthogonal
            # fallback if even the unweighted design is rank-deficient) for fe_arr=None and
            # for individual cadences where WLS weights lose rank due to NaN FLUX_ERR.
            gram_ols = phi_bg @ phi_bg.T
            gram_ols_sym = 0.5 * (gram_ols + gram_ols.T)
            ols_ev, ols_vec = np.linalg.eigh(gram_ols_sym)
            ols_max = float(ols_ev[-1]) if ols_ev.size > 0 else 0.0
            if ols_max > 0.0 and int(np.sum(ols_ev > 1e-12 * ols_max)) == n_basis:
                u_ols = ols_vec @ ((ols_vec.T @ phi_w) / ols_ev)
                rhs_ols = phi_bg @ col_inc  # (P, nx)
                g_fallback_1d = col_g - u_ols @ rhs_ols
            else:
                g_fallback_1d = column_orthogonal_smear_weights(col_g, bkg_order)

            for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
                stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
                se_chunk = se_arr[start:stop]  # (chunk, nx)
                chol_sub = chol_arr[start:stop]
                nan_chol = np.all(np.isnan(chol_sub), axis=(1, 2))

                if fe_arr is not None:
                    se_bg_sq = np.square(se_chunk[:, bg_cols])
                    se_bg_sq_clean = np.where(np.isfinite(se_bg_sq), se_bg_sq, 0.0)
                    fe_bg = fe_arr[start:stop, bg_rows, bg_cols]
                    fe_bg_sq = np.square(fe_bg)
                    lt_phi_bg = np.einsum("tqp,qm->tpm", chol_sub, phi_bg)
                    bkg_bg_sq = np.sum(np.square(lt_phi_bg), axis=1)
                    bkg_bg_sq_clean = np.where(np.isfinite(bkg_bg_sq), bkg_bg_sq, 0.0)
                    valid_bg = np.isfinite(fe_bg) & (fe_bg > 0.0) & (~nan_chol[:, None])
                    u_var_bg = np.maximum(
                        fe_bg_sq - bkg_bg_sq_clean,
                        np.maximum(se_bg_sq_clean, 1e-12),
                    )
                    w_bg = np.where(valid_bg, 1.0 / u_var_bg, 0.0)

                    gram = np.einsum("pm,tm,qm->tpq", phi_bg, w_bg, phi_bg)
                    gram_sym = 0.5 * (gram + np.swapaxes(gram, -1, -2))
                    evals, evecs = np.linalg.eigh(gram_sym)
                    max_ev = np.max(evals, axis=-1, keepdims=True)
                    cutoff = np.maximum(1e-12 * max_ev, 0.0)
                    safe_ev = np.where(evals > cutoff, evals, 1.0)
                    inv_evals = np.where((max_ev > 0.0) & (evals > cutoff), 1.0 / safe_ev, 0.0)
                    full_rank_wls = (max_ev[:, 0] > 0.0) & (
                        np.sum(evals > cutoff, axis=-1) == n_basis
                    )
                    # Solve G_t u_t = phi_w on full-rank WLS cadences; fall back to OLS
                    # oblique projection on any rank-deficient cadence so polynomial smear
                    # cancellation is preserved without double-counting BKG_CHOL.
                    u_t = np.einsum("tpi,ti,tqi,q->tp", evecs, inv_evals, evecs, phi_w)
                    rhs_cols = np.einsum("pm,tm,mc->tpc", phi_bg, w_bg, col_inc)
                    g_tilde_wls = col_g[None, :] - np.einsum("tp,tpc->tc", u_t, rhs_cols)
                    g_tilde = np.where(
                        full_rank_wls[:, None], g_tilde_wls, g_fallback_1d[None, :]
                    )
                else:
                    g_tilde = np.broadcast_to(g_fallback_1d[None, :], (stop - start, nx))

                active_t = np.abs(g_tilde) > 0.0
                se_sq_safe = np.where(active_t & np.isfinite(se_chunk), np.square(se_chunk), 0.0)
                smear_term = np.sum(np.square(g_tilde) * se_sq_safe, axis=1)
                bad_se = np.any(active_t & ~np.isfinite(se_chunk), axis=1) | nan_chol
                smear_term[bad_se] = np.nan
                var_total[start:stop] += smear_term
        else:
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


def _compute_weighted_centroids(
    tpf_obj: TargetPixelFile,
    flux_3d: np.ndarray,
    rows_h: np.ndarray,
    cols_h: np.ndarray,
    w_star: np.ndarray,
) -> tuple[u.Quantity, u.Quantity]:
    """Compute positive-flux, weightmap-weighted centroids with safe fallback when denom <= 0."""
    col_offset = float(getattr(tpf_obj, "column", 0) or 0)
    row_offset = float(getattr(tpf_obj, "row", 0) or 0)
    col_coords = col_offset + cols_h.astype(np.float64)
    row_coords = row_offset + rows_h.astype(np.float64)

    f_act = flux_3d[:, rows_h, cols_h]  # (n_cad, M)
    fin_act = np.isfinite(f_act)
    any_fin = np.any(fin_act, axis=1)
    f_pos = np.where(fin_act, np.maximum(f_act, 0.0), 0.0)
    pw = f_pos * w_star[None, :]  # (n_cad, M)
    denom = np.sum(pw, axis=1)

    w_sum = float(np.sum(w_star))
    fallback_col = float(np.dot(w_star, col_coords) / w_sum) if w_sum > 0.0 else np.nan
    fallback_row = float(np.dot(w_star, row_coords) / w_sum) if w_sum > 0.0 else np.nan

    pos_denom = denom > 0.0
    safe_denom = np.where(pos_denom, denom, 1.0)
    cent_col = np.where(
        pos_denom,
        (pw @ col_coords) / safe_denom,
        np.where(any_fin, fallback_col, np.nan),
    )
    cent_row = np.where(
        pos_denom,
        (pw @ row_coords) / safe_denom,
        np.where(any_fin, fallback_row, np.nan),
    )
    return u.Quantity(cent_col, unit=u.pixel), u.Quantity(cent_row, unit=u.pixel)


def _coerce_tpf(tpf_or_path: Any) -> TargetPixelFile:
    if isinstance(tpf_or_path, TargetPixelFile):
        return tpf_or_path
    if isinstance(tpf_or_path, str | os.PathLike | fits.HDUList):
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
    base_meta.pop("FILENAME", None)
    base_meta.pop("filename", None)
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
    maxiter: int = 500,
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
    catalog: str = "TIC",
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    strict_hpc: bool = True,
    target_mag: float | None = None,
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
        ``flux_err`` (Eq. 6.4.1). When ``False``, preserves physical rate units
        (converting integrated ``electron``/``count`` columns to ``e-/s`` or
        ``count/s``); for multi-segment ``split_times``, segments are anchored
        to a common full-sector unsaturated reference flux with per-segment
        scale factors recorded in ``lc.meta['SEGMENT_SCALES']`` while
        ``lc.meta['WEIGHTMAP']`` remains unit-sum simplex weights.
    cache_dir : path-like, optional
        Explicit cache directory for live catalogue queries, validated via
        :func:`halophot.resolve_cache_dir`.
    strict_hpc : bool, default True
        When ``True``, rejects ``/fred/`` cache paths (including symlinks) with
        :class:`ValueError` to protect HPC project inode quotas.

    Notes
    -----
    Always serialize returned light curves with :func:`halophot.write_fits`
    (alias for :func:`halophot.write_lightcurve_fits`) and reload pixel weights
    with :func:`halophot.read_weightmap` (plus :func:`halophot.read_weightmap_err`
    and :func:`halophot.read_split_times`). Calling ``lc.to_fits(path)`` directly
    drops ``HALOCVER = 1`` extensions (``WEIGHTMAP``, ``WEIGHTMAP_ERR``,
    ``SPLIT_TIMES``) and provenance cards.
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

        split_indices = [int(np.min(np.where(np.isfinite(t_all) & (t_all > st)))) for st in split_arr]
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
                catalog=catalog,
                offline=offline,
                cache_dir=cache_dir,
                strict_hpc=strict_hpc,
                target_mag=target_mag,
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

        # Anchor normalize=False segments to a common full-sector physical aperture flux scale
        # so shifting simplex weights between bright/faint wings cannot create artificial steps
        # while preserving real astrophysical brightness variations across splits.
        seg_scales = [1.0] * len(seg_lcs)
        if not normalize and len(seg_lcs) > 1:
            flux_full = _strip_quantity(tpf_obj.flux).copy()
            full_flux_unit = _resolve_tpf_column_unit(
                tpf_obj, "FLUX", getattr(tpf_obj.flux, "unit", None)
            )
            convert_full = _is_integrated_count_unit(full_flux_unit)
            if convert_full:
                diffs_full = np.diff(fin_t)
                pos_full = diffs_full[diffs_full > 0.0]
                dt_full = float(np.median(pos_full)) if pos_full.size > 0 else (1.0 / 48.0)
                exp_full = _resolve_cadence_exposure_sec(tpf_obj, dt_full)
                flux_full = flux_full / exp_full
            else:
                exp_full = 1.0

            # Exclude non-finite timestamps and pointing excursions (>= 5 px) from reference profile.
            clean_cad = np.isfinite(t_all)
            pc1_full = (
                _strip_quantity(tpf_obj.pos_corr1)
                if getattr(tpf_obj, "pos_corr1", None) is not None
                else None
            )
            pc2_full = (
                _strip_quantity(tpf_obj.pos_corr2)
                if getattr(tpf_obj, "pos_corr2", None) is not None
                else None
            )
            if pc1_full is not None and pc1_full.shape == (t_all.size,):
                clean_cad &= np.isfinite(pc1_full) & (np.abs(pc1_full) < 5.0)
            if pc2_full is not None and pc2_full.shape == (t_all.size,):
                clean_cad &= np.isfinite(pc2_full) & (np.abs(pc2_full) < 5.0)
            if not np.any(clean_cad):
                clean_cad = np.isfinite(t_all)

            flux_clean = flux_full[clean_cad].copy()

            # Mask saturated pixel-cadences so partial-sector bleed in one segment cannot
            # bias the reference flux of a pixel used unsaturated in another segment.
            unsub_clean = flux_clean.copy()
            bkgsub_full = read_bkgsub_state(tpf_obj, warn_legacy_mundey=False)
            if (
                bkgsub_full
                and hasattr(tpf_obj, "flux_bkg")
                and tpf_obj.flux_bkg is not None
            ):
                fb_full = _strip_quantity(tpf_obj.flux_bkg)
                fb_unit_full = _resolve_tpf_column_unit(
                    tpf_obj, "FLUX_BKG", getattr(tpf_obj.flux_bkg, "unit", full_flux_unit)
                )
                if convert_full and _is_integrated_count_unit(fb_unit_full):
                    fb_full = fb_full / exp_full
                fb_clean = fb_full[clean_cad]
                unsub_clean = np.where(np.isfinite(fb_clean), flux_clean + fb_clean, flux_clean)

            if auto_sat_level is not None:
                sat_cut = float(auto_sat_level)
            elif thresh < 0.0:
                m_up = str(getattr(tpf_obj, "mission", "") or "").upper()
                sat_cut = (
                    KEPLER_SAT_RATE_E_PER_S
                    if m_up in ("KEPLER", "K2")
                    else TESS_SAT_RATE_E_PER_S
                )
            else:
                fin_u = unsub_clean[np.isfinite(unsub_clean)]
                sat_cut = float(thresh) * float(np.max(fin_u)) if fin_u.size > 0 else np.inf
            if np.isfinite(sat_cut):
                flux_clean = np.where(unsub_clean >= sat_cut, np.nan, flux_clean)

            any_fin_full = np.any(np.isfinite(flux_clean), axis=0)
            mean_p_full = np.zeros(flux_full.shape[1:], dtype=np.float64)
            if np.any(any_fin_full):
                # Median over clean unsaturated cadences rejects unclipped cosmic-ray spikes.
                mean_p_full[any_fin_full] = np.nanmedian(flux_clean[:, any_fin_full], axis=0)
            seg_ref_fluxes = [float(np.sum(wm * mean_p_full)) for wm in seg_wmaps]
            valid_refs = [rf for rf in seg_ref_fluxes if np.isfinite(rf) and rf > 1e-30]
            if valid_refs:
                f_ref_common = float(np.mean(valid_refs))
                if np.isfinite(f_ref_common) and f_ref_common > 1e-30:
                    anchored_lcs: list[lk.LightCurve] = []
                    seg_scales = []
                    for lc_s, rf_s in zip(seg_lcs, seg_ref_fluxes, strict=True):
                        lc_copy = lc_s.copy()
                        if np.isfinite(rf_s) and rf_s > 1e-30:
                            scale_s = f_ref_common / rf_s
                            lc_copy.flux = lc_copy.flux * scale_s
                            lc_copy.flux_err = lc_copy.flux_err * scale_s
                            seg_scales.append(float(scale_s))
                        else:
                            seg_scales.append(1.0)
                        anchored_lcs.append(lc_copy)
                    seg_lcs = anchored_lcs

        stitched = lk.LightCurveCollection([seg.copy() for seg in seg_lcs]).stitch(
            corrector_func=None
        )
        stitched.meta = dict(stitched.meta)
        stitched.meta["WEIGHTMAP"] = np.stack(seg_wmaps, axis=0)
        stitched.meta["SPLIT_TIMES"] = split_arr.tolist()
        if not normalize and len(seg_lcs) > 1:
            stitched.meta["SEGMENT_SCALES"] = seg_scales
        if compute_uncertainty and seg_werrs:
            stitched.meta["WEIGHTMAP_ERR"] = np.stack(seg_werrs, axis=0)
        return stitched

    # Single-segment extraction.
    times_all = np.asarray(tpf_obj.time.value, dtype=np.float64)
    fin_times = times_all[np.isfinite(times_all)]
    diffs_t = np.diff(fin_times)
    pos_diffs = diffs_t[diffs_t > 0.0]
    dt_days = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)

    flux_unit_attr = getattr(tpf_obj.flux, "unit", None)
    flux_unit_raw = _resolve_tpf_column_unit(tpf_obj, "FLUX", flux_unit_attr)
    convert_counts = _is_integrated_count_unit(flux_unit_raw)
    exp_sec = _resolve_cadence_exposure_sec(tpf_obj, dt_days) if convert_counts else 1.0
    eff_flux_unit = _integrated_to_rate_unit(flux_unit_raw) if convert_counts else flux_unit_attr

    flux_3d = _strip_quantity(tpf_obj.flux).copy()
    if convert_counts:
        flux_3d = flux_3d / exp_sec
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
        fe_unit_raw = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_ERR", getattr(tpf_obj.flux_err, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fe_unit_raw):
            fe_cand = fe_cand / exp_sec
        if np.any(np.isfinite(fe_cand) & (fe_cand > 0.0)):
            flux_err_3d = fe_cand

    flux_bkg_3d: np.ndarray | None = None
    if hasattr(tpf_obj, "flux_bkg") and tpf_obj.flux_bkg is not None:
        fb_cand = _strip_quantity(tpf_obj.flux_bkg)
        fb_unit_raw = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_BKG", getattr(tpf_obj.flux_bkg, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fb_unit_raw):
            fb_cand = fb_cand / exp_sec
        flux_bkg_3d = fb_cand

    flux_bkg_err_3d: np.ndarray | None = None
    if hasattr(tpf_obj, "flux_bkg_err") and tpf_obj.flux_bkg_err is not None:
        fbe_cand = _strip_quantity(tpf_obj.flux_bkg_err)
        fbe_unit_raw = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_BKG_ERR", getattr(tpf_obj.flux_bkg_err, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fbe_unit_raw):
            fbe_cand = fbe_cand / exp_sec
        if np.any(np.isfinite(fbe_cand) & (fbe_cand > 0.0)):
            flux_bkg_err_3d = fbe_cand

    pq_2d = read_pixel_quality(tpf_obj)
    bkgsub = read_bkgsub_state(tpf_obj, warn_legacy_mundey=True)
    bkg_order = infer_bkg_poly_order(tpf_obj)
    smear_err_2d = read_smear_err(tpf_obj, apply_quality_mask=True)
    if convert_counts and smear_err_2d is not None:
        se_unit_raw = _resolve_tpf_column_unit(tpf_obj, "SMEAR_ERR", flux_unit_raw)
        if _is_integrated_count_unit(se_unit_raw):
            smear_err_2d = smear_err_2d / exp_sec
    bkg_cov = read_bkg_covariance(tpf_obj, apply_quality_mask=True)
    if convert_counts and bkg_cov is not None:
        chol_unit_raw = _resolve_tpf_column_unit(
            tpf_obj, "BKG_CHOL", _resolve_hdu_unit(tpf_obj, "BKG_CHOL", flux_unit_raw)
        )
        if _is_integrated_count_unit(chol_unit_raw):
            bkg_cov = BkgCovariance(chol=bkg_cov.chol / exp_sec, basis=bkg_cov.basis)

    # When BKGSUB=True and FLUX_BKG_ERR is absent/all-zero on HALOCVER=1 TPFs, derive exact
    # per-pixel RMS background-fit uncertainty sigma_{B,i} = sqrt(phi_i^T mean(L_t L_t^T) phi_i)
    # from BKG_CHOL + BKG_BASIS so build_halo_masks guards against background-fit uncertainty.
    mask_bkg_err: np.ndarray | None = flux_bkg_err_3d
    if bkgsub and mask_bkg_err is None and bkg_cov is not None:
        chol_all = np.asarray(bkg_cov.chol, dtype=np.float64)
        basis_all = np.asarray(bkg_cov.basis, dtype=np.float64)
        fin_chol = np.isfinite(times_all) & ~np.any(~np.isfinite(chol_all), axis=(1, 2))
        if np.any(fin_chol):
            c_fin = chol_all[fin_chol]
            mean_cov_beta = np.einsum("tpk,tqk->pq", c_fin, c_fin) / float(c_fin.shape[0])
            bkg_var_2d = np.einsum("prc,pq,qrc->rc", basis_all, mean_cov_beta, basis_all)
            mask_bkg_err = np.sqrt(np.maximum(bkg_var_2d, 0.0))

    pc1 = _strip_quantity(tpf_obj.pos_corr1) if getattr(tpf_obj, "pos_corr1", None) is not None else None
    pc2 = _strip_quantity(tpf_obj.pos_corr2) if getattr(tpf_obj, "pos_corr2", None) is not None else None

    tgt_coord: tuple[float, float] | None = None
    if getattr(tpf_obj, "ra", None) is not None and getattr(tpf_obj, "dec", None) is not None:
        tgt_coord = (float(tpf_obj.ra), float(tpf_obj.dec))

    resolved_target_mag = (
        target_mag
        if target_mag is not None
        else _estimate_target_mag(
            tpf=tpf_obj,
            flux=flux_3d,
            mission=getattr(tpf_obj, "mission", None),
        )
    )

    mask_res = build_halo_masks(
        flux_3d,
        flux_err=flux_err_3d,
        flux_bkg=flux_bkg_3d,
        flux_bkg_err=mask_bkg_err,
        bkgsub=bkgsub,
        pixel_quality=pq_2d,
        aperture_mask=parsed_ap,
        cadence_quality_mask=np.isfinite(times_all),
        pos_corr1=pc1,
        pos_corr2=pc2,
        wcs=_resolve_tpf_wcs(tpf_obj),
        target_coord=tgt_coord,
        target_mag=resolved_target_mag,
        obs_time=_resolve_tpf_obs_time(tpf_obj),
        remove_contaminants=remove_contaminants,
        catalog_table=catalog_table,
        catalog=catalog,
        offline=offline,
        cache_dir=cache_dir,
        strict_hpc=strict_hpc,
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

    # Independent per-pixel variance for photon-noise regularization lambda_phot:
    # average over finite cadences only and subtract both background and coherent smear variance.
    pvar_reg: np.ndarray | None = None
    if not legacy_tv and flux_err_3d is not None:
        fe_act = flux_err_3d[active_cad][:, rows_h, cols_h].T  # (M, N_good)
        fin_fe = np.isfinite(fe_act) & (fe_act >= 0.0)
        fe_count = np.sum(fin_fe, axis=1)
        fe_sq_sum = np.sum(np.where(fin_fe, np.square(fe_act), 0.0), axis=1)
        fe_sq_mean = np.where(fe_count > 0, fe_sq_sum / np.maximum(fe_count, 1), 0.0)

        se_sq_mean = np.zeros(rows_h.size, dtype=np.float64)
        if smear_err_2d is not None:
            se_act = smear_err_2d[active_cad][:, cols_h].T
            fin_se = np.isfinite(se_act) & (se_act >= 0.0)
            se_count = np.sum(fin_se, axis=1)
            se_sq_sum = np.sum(np.where(fin_se, np.square(se_act), 0.0), axis=1)
            se_sq_mean = np.where(se_count > 0, se_sq_sum / np.maximum(se_count, 1), 0.0)

        bkg_sq_mean = np.zeros(rows_h.size, dtype=np.float64)
        if bkg_cov is not None and bkgsub:
            chol_act = bkg_cov.chol[active_cad]
            phi_act = bkg_cov.basis[:, rows_h, cols_h]
            fin_c = ~np.all(np.isnan(chol_act), axis=(1, 2))
            if np.any(fin_c):
                lt_p = np.einsum("tqp,qm->tpm", chol_act[fin_c], phi_act)
                bkg_sq_mean = np.mean(np.sum(np.square(lt_p), axis=1), axis=0)
        elif bkgsub and flux_bkg_err_3d is not None:
            fbe_act = flux_bkg_err_3d[active_cad][:, rows_h, cols_h].T
            fin_fbe = np.isfinite(fbe_act) & (fbe_act >= 0.0)
            fbe_count = np.sum(fin_fbe, axis=1)
            fbe_sq_sum = np.sum(np.where(fin_fbe, np.square(fbe_act), 0.0), axis=1)
            bkg_sq_mean = np.where(fbe_count > 0, fbe_sq_sum / np.maximum(fbe_count, 1), 0.0)

        fe_sub_sq = np.maximum(fe_sq_mean - bkg_sq_mean, 0.0)
        has_sub_reg = (se_sq_mean > 0.0) | (bkg_sq_mean > 0.0)
        ulp_floor_reg = np.where(
            has_sub_reg & (fe_sq_mean > 0.0),
            0.5 * _F32_EPS * fe_sq_mean,
            0.0,
        )
        pvar_reg = np.maximum(fe_sub_sq - se_sq_mean, ulp_floor_reg)

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

    # Outlier clipping (sigclip=True) excludes bad cadences ONLY during weight optimization
    # and baseline normalization; finite cadences (e.g. deep eclipses) are always preserved
    # in the returned light curve.
    inlier_cad_mask = np.zeros(n_cad, dtype=bool)
    inlier_cad_mask[np.flatnonzero(active_cad)[fit.inlier_mask]] = True
    norm_mask = inlier_cad_mask if np.any(inlier_cad_mask) else active_cad

    mean_p_norm = np.mean(flux_3d[norm_mask][:, rows_h, cols_h], axis=0)
    mean_flux_norm = float(np.dot(fit.physical_weights, mean_p_norm))
    if not np.isfinite(mean_flux_norm) or abs(mean_flux_norm) <= 1e-30:
        mean_flux_norm = 1.0

    norm_scale = mean_flux_norm if normalize else 1.0

    lc_flux = np.full(n_cad, np.nan, dtype=np.float64)
    lc_flux[active_cad] = (
        flux_3d[active_cad][:, rows_h, cols_h] @ fit.physical_weights
    ) / norm_scale

    bg_mask_2d = (pq_2d & np.uint16(PixelQuality.BKG_USED)) != 0
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
            bg_mask=bg_mask_2d if np.any(bg_mask_2d) else None,
            bkgsub=bkgsub,
            bkg_order=bkg_order,
            norm_factor=norm_scale,
        )
        lc_flux_err[~active_cad] = np.nan
    else:
        lc_flux_err = np.full(n_cad, np.nan, dtype=np.float64)

    # Positive-flux, weightmap-weighted centroids over active halo aperture.
    cent_col, cent_row = _compute_weighted_centroids(
        tpf_obj, flux_3d, rows_h, cols_h, fit.physical_weights
    )

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

    if not normalize and eff_flux_unit is not None and eff_flux_unit != u.dimensionless_unscaled:
        flux_arg: np.ndarray | u.Quantity = u.Quantity(lc_flux, unit=eff_flux_unit)
        err_arg: np.ndarray | u.Quantity = u.Quantity(lc_flux_err, unit=eff_flux_unit)
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
