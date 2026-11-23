"""Functional ``lightkurve`` API, ``HaloLightCurve``, coherent ``flux_err``, and diagnostics."""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import LombScargle
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
    write_lightcurve_fits,
)
from ._precision import in_float64
from .masking import (
    KEPLER_SAT_RATE_E_PER_S,
    TESS_SAT_RATE_E_PER_S,
    MaskConfig,
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
from .optimise import OptimConfig, optimise_halo

__all__ = [
    "HaloKeplerLightCurve",
    "HaloLightCurve",
    "HaloTessLightCurve",
    "column_orthogonal_smear_weights",
    "compute_coherent_flux_err",
    "extract_halo_lightcurve",
    "plot_diagnostics",
]

_ERR_CHUNK_CADENCES: int = 8192
_F32_EPS: float = float(np.finfo(np.float32).eps)
_DAY_TO_MUHZ: float = 1e6 / 86400.0
_MASK_CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(MaskConfig))
_OPTIM_CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(OptimConfig))


_IN_WRITE_FITS = threading.local()


class HaloLightCurve(lk.LightCurve):
    """``lightkurve.LightCurve`` subclass whose ``.to_fits()`` preserves ``HALOCVER = 1`` extensions.

    Delegates :meth:`to_fits` to :func:`~halophot._contract.write_lightcurve_fits`
    so ``WEIGHTMAP``, ``WEIGHTMAP_ERR``, ``SPLIT_TIMES``, ``SEGMENT_SCALES``,
    ``INLIER_MASK``, ``POS_CORR1``/``POS_CORR2``, and physical rate units
    round-trip losslessly.
    """

    def to_fits(
        self,
        path: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> fits.HDUList:
        """Write or return a ``HALOCVER = 1`` FITS ``HDUList`` preserving halo extensions."""
        if getattr(_IN_WRITE_FITS, "active", False):
            return super().to_fits(path=path, overwrite=overwrite, **kwargs)
        _IN_WRITE_FITS.active = True
        try:
            if path is not None:
                write_lightcurve_fits(self, path, overwrite=overwrite)
                with fits.open(path) as hdul:
                    return fits.HDUList([hdu.copy() for hdu in hdul])
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / "halo_lc.fits"
                write_lightcurve_fits(self, tmp_path, overwrite=True)
                with fits.open(tmp_path) as hdul:
                    return fits.HDUList([hdu.copy() for hdu in hdul])
        finally:
            _IN_WRITE_FITS.active = False


class HaloTessLightCurve(HaloLightCurve, lk.TessLightCurve):
    """``TessLightCurve`` subclass preserving ``HALOCVER = 1`` extensions in ``.to_fits()``."""


class HaloKeplerLightCurve(HaloLightCurve, lk.KeplerLightCurve):
    """``KeplerLightCurve`` subclass preserving ``HALOCVER = 1`` extensions in ``.to_fits()``."""


def column_orthogonal_smear_weights(
    col_weights: np.ndarray,
    bkg_order: int | None,
    *,
    valid_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``g_perp = (I - Pi_{<= d}^{col}) g`` in ``float64``.

    When ``valid_cols`` is provided (boolean ``(nx,)``), restricts the Chebyshev
    polynomial projection strictly to ``valid_cols`` and leaves masked columns
    unchanged (so zero-weight ``SMEAR_ERR = NaN`` columns never acquire spurious
    projection weight).
    """
    g = np.asarray(col_weights, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError(f"col_weights must be 1D (nx,), got shape {g.shape}.")
    nx = g.shape[0]
    if bkg_order is None or bkg_order < 0 or nx == 0:
        return g.copy()

    if valid_cols is None:
        mask = np.ones(nx, dtype=bool)
    else:
        mask = np.asarray(valid_cols, dtype=bool)
        if mask.shape != (nx,):
            raise ValueError(f"valid_cols shape {mask.shape} does not match ({nx},).")

    n_valid = int(np.count_nonzero(mask))
    out = g.copy()
    if n_valid == 0:
        return out
    if bkg_order >= n_valid - 1:
        out[mask] = 0.0
        return out

    xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)[mask]
    vander = np.polynomial.chebyshev.chebvander(xi, int(bkg_order))
    q_mat, _ = np.linalg.qr(vander, mode="reduced")
    g_sub = g[mask]
    out[mask] = g_sub - q_mat @ (q_mat.T @ g_sub)
    return out


def _finite_smear_cols(se_arr: np.ndarray) -> np.ndarray:
    """Identify columns with finite ``SMEAR_ERR`` on valid cadences."""
    finite_rows = np.any(np.isfinite(se_arr), axis=1)
    if np.any(finite_rows):
        all_fin = np.all(np.isfinite(se_arr[finite_rows]), axis=0)
        if np.any(all_fin):
            return all_fin
    return np.any(np.isfinite(se_arr), axis=0)


@in_float64
def compute_coherent_flux_err(
    weightmap_2d: np.ndarray,
    *,
    flux_err: np.ndarray | None,
    smear_err: np.ndarray | None = None,
    bkg_cov: BkgCovariance | None = None,
    flux_bkg_err: np.ndarray | None = None,
    bg_mask: np.ndarray | None = None,
    shot_var: np.ndarray | None = None,
    bkgsub: bool = True,
    bkg_order: int | None = None,
    norm_factor: float = 1.0,
    flux_err_scale: float = 1.0,
    flux_bkg_err_scale: float = 1.0,
) -> np.ndarray:
    """Propagate coherent halo light-curve uncertainties ``sigma_{f,t}`` in ``float64``.

    Combines uncorrelated pixel noise (with a ``float32``-ULP + physical detector
    ``shot_var`` floor), exact per-cadence WLS oblique column-smear sensitivity
    (falling back to finite-``SMEAR_ERR``-restricted Chebyshev orthogonal
    projection), and ``BKG_CHOL`` / ``BKG_BASIS`` background covariance without
    double-counting or copying full 3D error cubes to ``float64``.
    """
    w2d = np.asarray(weightmap_2d, dtype=np.float64)
    if w2d.ndim != 2:
        raise ValueError(f"weightmap_2d must be 2D (ny, nx), got shape {w2d.shape}.")
    ny, nx = w2d.shape
    w_clean = np.where(np.isfinite(w2d) & (w2d > 0.0), w2d, 0.0)
    rows_h, cols_h = np.where(w_clean > 0.0)
    w_star = w_clean[rows_h, cols_h]

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

    fe_view: np.ndarray | None = None
    if flux_err is not None:
        fe_view = np.asarray(flux_err)
        if fe_view.shape != (n_cad, ny, nx):
            raise ValueError(f"flux_err shape {fe_view.shape} does not match {(n_cad, ny, nx)}.")

    se_arr: np.ndarray | None = None
    if smear_err is not None:
        se_arr = np.asarray(smear_err, dtype=np.float64)
        if se_arr.shape != (n_cad, nx):
            raise ValueError(f"smear_err shape {se_arr.shape} does not match {(n_cad, nx)}.")

    fbe_view: np.ndarray | None = None
    if flux_bkg_err is not None:
        fbe_view = np.asarray(flux_bkg_err)
        if fbe_view.shape != (n_cad, ny, nx):
            raise ValueError(f"flux_bkg_err shape {fbe_view.shape} does not match {(n_cad, ny, nx)}.")

    bg_bool: np.ndarray | None = None
    if bg_mask is not None:
        bg_bool = np.asarray(bg_mask, dtype=bool)
        if bg_bool.shape != (ny, nx):
            raise ValueError(f"bg_mask shape {bg_bool.shape} does not match {(ny, nx)}.")

    sv_arr: np.ndarray | None = None
    if shot_var is not None:
        sv_arr = np.asarray(shot_var, dtype=np.float64)
        if sv_arr.shape not in ((ny, nx), (n_cad, ny, nx)):
            raise ValueError(
                f"shot_var shape {sv_arr.shape} does not match {(ny, nx)} or {(n_cad, ny, nx)}."
            )

    chol_arr: np.ndarray | None = None
    basis_arr: np.ndarray | None = None
    phi_halo: np.ndarray | None = None
    if bkg_cov is not None and bkgsub:
        chol_arr = np.asarray(bkg_cov.chol, dtype=np.float64)
        basis_arr = np.asarray(bkg_cov.basis, dtype=np.float64)
        if chol_arr.shape[0] != n_cad or basis_arr.shape[1:] != (ny, nx):
            raise ValueError(
                f"bkg_cov shapes are inconsistent with {(n_cad, ny, nx)}: "
                f"chol={chol_arr.shape}, basis={basis_arr.shape}."
            )
        phi_halo = basis_arr[:, rows_h, cols_h]

    w_sq = np.square(w_star)
    var_total = np.zeros(n_cad, dtype=np.float64)

    # 1. Uncorrelated pixel-noise term in bounded chunks.
    if fe_view is not None:
        for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
            stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
            fe_sub = np.asarray(fe_view[start:stop, rows_h, cols_h], dtype=np.float64)
            if flux_err_scale != 1.0:
                fe_sub = fe_sub * flux_err_scale
            fe_sq = np.square(fe_sub)

            smear_pix_sq: np.ndarray | float = (
                np.square(se_arr[start:stop][:, cols_h]) if se_arr is not None else 0.0
            )
            if isinstance(smear_pix_sq, np.ndarray):
                smear_pix_sq = np.where(np.isfinite(smear_pix_sq), smear_pix_sq, 0.0)

            if chol_arr is not None and phi_halo is not None:
                chol_sub = chol_arr[start:stop]
                lt_phi = np.einsum("tqp,qm->tpm", chol_sub, phi_halo)
                bkg_pix_sq: np.ndarray | float = np.sum(np.square(lt_phi), axis=1)
                nan_chol = np.all(np.isnan(chol_sub), axis=(1, 2))
                bkg_pix_sq = np.where(nan_chol[:, None], np.nan, bkg_pix_sq)
            elif bkgsub and fbe_view is not None:
                fbe_sub = np.asarray(fbe_view[start:stop, rows_h, cols_h], dtype=np.float64)
                if flux_bkg_err_scale != 1.0:
                    fbe_sub = fbe_sub * flux_bkg_err_scale
                bkg_pix_sq = np.where(np.isfinite(fbe_sub), np.square(fbe_sub), 0.0)
            else:
                bkg_pix_sq = 0.0

            sigma_sub_sq = np.maximum(fe_sq - bkg_pix_sq, 0.0)
            has_subtracted = (smear_pix_sq > 0.0) | (np.isfinite(bkg_pix_sq) & (bkg_pix_sq > 0.0))
            ulp_floor = np.where(
                has_subtracted & np.isfinite(fe_sq) & (fe_sq > 0.0),
                0.5 * _F32_EPS * fe_sq,
                0.0,
            )
            raw_pix_sq = sigma_sub_sq - smear_pix_sq
            sigma_pix_sq = np.maximum(raw_pix_sq, ulp_floor)
            if sv_arr is not None:
                sv_sub = (
                    sv_arr[rows_h, cols_h][None, :]
                    if sv_arr.ndim == 2
                    else sv_arr[start:stop, rows_h, cols_h]
                )
                sv_clean = np.where(np.isfinite(sv_sub) & (sv_sub > 0.0), sv_sub, 0.0)
                cancelled = has_subtracted & (raw_pix_sq <= 4.0 * ulp_floor)
                sigma_pix_sq = np.where(cancelled, np.maximum(ulp_floor, sv_clean), sigma_pix_sq)

            bad_cad = np.any(~np.isfinite(fe_sub), axis=1)
            if isinstance(bkg_pix_sq, np.ndarray):
                bad_cad |= np.any(~np.isfinite(bkg_pix_sq), axis=1)
            pix_term = sigma_pix_sq @ w_sq
            pix_term[bad_cad] = np.nan
            var_total[start:stop] += pix_term

    # 2. Column-coherent smear term.
    if se_arr is not None:
        col_g = np.sum(w_clean, axis=0)
        valid_se_cols = _finite_smear_cols(se_arr)
        use_oblique = (
            bkgsub
            and chol_arr is not None
            and basis_arr is not None
            and phi_halo is not None
            and bg_bool is not None
            and int(np.count_nonzero(bg_bool)) >= basis_arr.shape[0]
        )
        if use_oblique:
            assert chol_arr is not None and basis_arr is not None
            assert phi_halo is not None and bg_bool is not None
            bg_rows, bg_cols = np.where(bg_bool)
            m_bg = bg_rows.size
            n_basis = basis_arr.shape[0]
            phi_bg = basis_arr[:, bg_rows, bg_cols]
            phi_w = phi_halo @ w_star
            col_inc = np.zeros((m_bg, nx), dtype=np.float64)
            col_inc[np.arange(m_bg), bg_cols] = 1.0

            gram_ols = phi_bg @ phi_bg.T
            ols_ev, ols_vec = np.linalg.eigh(0.5 * (gram_ols + gram_ols.T))
            ols_max = float(ols_ev[-1]) if ols_ev.size > 0 else 0.0
            if ols_max > 0.0 and int(np.sum(ols_ev > 1e-12 * ols_max)) == n_basis:
                u_ols = ols_vec @ ((ols_vec.T @ phi_w) / ols_ev)
                g_fallback_1d = col_g - u_ols @ (phi_bg @ col_inc)
            else:
                g_fallback_1d = column_orthogonal_smear_weights(
                    col_g, bkg_order, valid_cols=valid_se_cols
                )

            for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
                stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
                se_chunk = se_arr[start:stop]
                chol_sub = chol_arr[start:stop]
                nan_chol = np.all(np.isnan(chol_sub), axis=(1, 2))

                if fe_view is not None:
                    se_bg_sq = np.square(se_chunk[:, bg_cols])
                    se_bg_sq_clean = np.where(np.isfinite(se_bg_sq), se_bg_sq, 0.0)
                    fe_bg = np.asarray(fe_view[start:stop, bg_rows, bg_cols], dtype=np.float64)
                    if flux_err_scale != 1.0:
                        fe_bg = fe_bg * flux_err_scale
                    lt_phi_bg = np.einsum("tqp,qm->tpm", chol_sub, phi_bg)
                    bkg_bg_sq = np.sum(np.square(lt_phi_bg), axis=1)
                    bkg_bg_sq_clean = np.where(np.isfinite(bkg_bg_sq), bkg_bg_sq, 0.0)
                    valid_bg = np.isfinite(fe_bg) & (fe_bg > 0.0) & (~nan_chol[:, None])
                    u_var_bg = np.maximum(
                        np.square(fe_bg) - bkg_bg_sq_clean,
                        np.maximum(se_bg_sq_clean, 1e-12),
                    )
                    w_bg = np.where(valid_bg, 1.0 / u_var_bg, 0.0)

                    gram = np.einsum("pm,tm,qm->tpq", phi_bg, w_bg, phi_bg)
                    evals, evecs = np.linalg.eigh(0.5 * (gram + np.swapaxes(gram, -1, -2)))
                    max_ev = np.max(evals, axis=-1, keepdims=True)
                    cutoff = np.maximum(1e-12 * max_ev, 0.0)
                    safe_ev = np.where(evals > cutoff, evals, 1.0)
                    inv_evals = np.where((max_ev > 0.0) & (evals > cutoff), 1.0 / safe_ev, 0.0)
                    full_rank = (max_ev[:, 0] > 0.0) & (np.sum(evals > cutoff, axis=-1) == n_basis)
                    u_t = np.einsum("tpi,ti,tqi,q->tp", evecs, inv_evals, evecs, phi_w)
                    rhs_cols = np.einsum("pm,tm,mc->tpc", phi_bg, w_bg, col_inc)
                    g_tilde_wls = col_g[None, :] - np.einsum("tp,tpc->tc", u_t, rhs_cols)
                    g_tilde = np.where(full_rank[:, None], g_tilde_wls, g_fallback_1d[None, :])
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
            g_perp = column_orthogonal_smear_weights(col_g, eff_order, valid_cols=valid_se_cols)
            active_cols = np.abs(g_perp) > 0.0
            if np.any(active_cols):
                se_act = se_arr[:, active_cols]
                smear_term = np.square(se_act) @ np.square(g_perp[active_cols])
                smear_term[np.any(~np.isfinite(se_act), axis=1)] = np.nan
                var_total += smear_term

    # 3. Coherent background covariance term.
    if chol_arr is not None and phi_halo is not None:
        phi_w = phi_halo @ w_star
        lt_phi_w = np.einsum("tqp,q->tp", chol_arr, phi_w)
        bkg_coh = np.sum(np.square(lt_phi_w), axis=1)
        bkg_coh[np.all(np.isnan(chol_arr), axis=(1, 2))] = np.nan
        var_total += bkg_coh
    elif bkgsub and fbe_view is not None:
        for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
            stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
            fbe_sub = np.asarray(fbe_view[start:stop, rows_h, cols_h], dtype=np.float64)
            if flux_bkg_err_scale != 1.0:
                fbe_sub = fbe_sub * flux_bkg_err_scale
            fbe_clean = np.where(np.isfinite(fbe_sub), np.maximum(fbe_sub, 0.0), 0.0)
            var_total[start:stop] += np.square(fbe_clean @ w_star)

    scale = abs(float(norm_factor))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"norm_factor must be finite and non-zero, got {norm_factor!r}.")

    return np.sqrt(np.maximum(var_total, 0.0)) / scale


def _raw_view(val: Any) -> np.ndarray:
    """Return a zero-copy native array view (preserving ``float32`` on large TPFs)."""
    return np.asarray(val.value if hasattr(val, "value") else val)


def _is_photon_rate_unit(unit_obj: Any) -> bool:
    if unit_obj is None or unit_obj == u.dimensionless_unscaled:
        return True
    if unit_obj in (u.electron / u.s, u.count / u.s):
        return True
    text = str(unit_obj).strip().lower().replace(" ", "")
    return text in (
        "electron/s",
        "electrons/s",
        "e-/s",
        "count/s",
        "counts/s",
        "ct/s",
        "adu/s",
        "electron*s**-1",
        "count*s**-1",
    )


def _header_get_first(tpf_obj: Any, keys: tuple[str, ...]) -> Any:
    hdul = getattr(tpf_obj, "hdu", None)
    headers: list[Any] = []
    if hdul is not None:
        if len(hdul) > 1:
            headers.append(hdul[1].header)
        if len(hdul) > 0:
            headers.append(hdul[0].header)
    meta = getattr(tpf_obj, "meta", None)
    if meta is not None:
        headers.append(meta)
    for hdr in headers:
        if hasattr(hdr, "get"):
            for k in keys:
                val = hdr.get(k)
                if val is not None:
                    return val
    return None


def _compute_detector_shot_var(
    tpf_obj: Any | None,
    flux_view: np.ndarray,
    flux_bkg_view: np.ndarray | None,
    active_cad: np.ndarray,
    rows_h: np.ndarray,
    cols_h: np.ndarray,
    *,
    bkgsub: bool,
    dt_days: float,
    flux_scale: float = 1.0,
    bkg_scale: float = 1.0,
) -> np.ndarray:
    """Compute physical detector ``shot_var`` floor including sky, smear, read noise, and CRMITEN."""
    ny, nx = flux_view.shape[1], flux_view.shape[2]
    exp_sec = (
        _resolve_cadence_exposure_sec(tpf_obj, dt_days)
        if tpf_obj is not None
        else max(float(dt_days) * 86400.0, 1.0)
    )

    act_idx = np.flatnonzero(active_cad)
    sum_p = np.zeros(rows_h.size, dtype=np.float64)
    sum_b = np.zeros(rows_h.size, dtype=np.float64)
    cnt_b = np.zeros(rows_h.size, dtype=np.int64)
    col_sum_unsub = np.zeros(nx, dtype=np.float64)
    col_cnt = 0

    for start in range(0, act_idx.size, _ERR_CHUNK_CADENCES):
        idx_c = act_idx[start : start + _ERR_CHUNK_CADENCES]
        p_c = np.asarray(flux_view[idx_c][:, rows_h, cols_h], dtype=np.float64) * flux_scale
        sum_p += np.sum(p_c, axis=0)
        if bkgsub and flux_bkg_view is not None:
            b_c = np.asarray(flux_bkg_view[idx_c][:, rows_h, cols_h], dtype=np.float64) * bkg_scale
            fin_b = np.isfinite(b_c)
            sum_b += np.sum(np.where(fin_b, b_c, 0.0), axis=0)
            cnt_b += np.sum(fin_b, axis=0)

        # Estimate column-summed unsubtracted flux on a bounded stride for smear shot noise.
        stride_c = idx_c[:: max(1, idx_c.size // 32)]
        frame_c = np.asarray(flux_view[stride_c], dtype=np.float64) * flux_scale
        if bkgsub and flux_bkg_view is not None:
            bf_c = np.asarray(flux_bkg_view[stride_c], dtype=np.float64) * bkg_scale
            frame_c = np.where(np.isfinite(bf_c), frame_c + np.maximum(bf_c, 0.0), frame_c)
        col_sum_unsub += np.sum(np.where(np.isfinite(frame_c), np.maximum(frame_c, 0.0), 0.0), axis=(0, 1))
        col_cnt += stride_c.size

    mean_p = sum_p / max(float(act_idx.size), 1.0)
    mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), 0.0)
    mean_col_unsub = col_sum_unsub / max(float(col_cnt), 1.0)

    mission_upper = str(getattr(tpf_obj, "mission", "") or "").upper() if tpf_obj is not None else ""
    deadc_raw = _header_get_first(tpf_obj, ("DEADC",)) if tpf_obj is not None else None
    try:
        deadc = float(deadc_raw) if deadc_raw is not None and not isinstance(deadc_raw, bool) else np.nan
    except (TypeError, ValueError):
        deadc = np.nan

    if np.isfinite(deadc) and 0.0 < deadc < 1.0:
        n_rows_ccd = 1024.0 if mission_upper in ("KEPLER", "K2") else 2048.0
        f_transfer = (1.0 - deadc) / (n_rows_ccd * deadc)
    elif mission_upper == "TESS":
        f_transfer = 0.01 / (2048.0 * 0.99)
    else:
        f_transfer = 0.0
    smear_rate_act = f_transfer * mean_col_unsub[cols_h]

    rn_raw = _header_get_first(tpf_obj, ("READNOIS", "READNOISE")) if tpf_obj is not None else None
    try:
        read_noise = float(rn_raw) if rn_raw is not None and not isinstance(rn_raw, bool) else np.nan
    except (TypeError, ValueError):
        read_noise = np.nan
    if not np.isfinite(read_noise) or read_noise < 0.0:
        read_noise = 0.0

    nf_raw = _header_get_first(tpf_obj, ("NUM_FRM", "NREADOUT")) if tpf_obj is not None else None
    try:
        n_frm = float(nf_raw) if nf_raw is not None and not isinstance(nf_raw, bool) else np.nan
    except (TypeError, ValueError):
        n_frm = np.nan
    if not np.isfinite(n_frm) or n_frm < 1.0:
        n_frm = max(round(exp_sec / 2.0), 1.0) if mission_upper == "TESS" else 1.0

    crmiten_raw = _header_get_first(tpf_obj, ("CRMITEN",)) if tpf_obj is not None else None
    crmiten = bool(crmiten_raw) if isinstance(crmiten_raw, bool | np.bool_) else False
    cr_scale = 1.0
    if crmiten:
        blksz_raw = _header_get_first(tpf_obj, ("CRBLKSZ",)) if tpf_obj is not None else None
        try:
            blksz = int(blksz_raw) if blksz_raw is not None and not isinstance(blksz_raw, bool) else 10
        except (TypeError, ValueError):
            blksz = 10
        eff_blk = min(max(blksz, 3), max(int(round(n_frm)), 3))
        if eff_blk > 2:
            cr_scale = float(eff_blk) / float(eff_blk - 2)

    total_rate = np.maximum(mean_p, 0.0) + np.maximum(mean_b, 0.0) + np.maximum(smear_rate_act, 0.0)
    read_var = (n_frm * (read_noise**2)) / max(exp_sec**2, 1.0)
    shot_var_act = cr_scale * (total_rate / max(exp_sec, 1.0) + read_var)

    shot_var_2d = np.zeros((ny, nx), dtype=np.float64)
    shot_var_2d[rows_h, cols_h] = shot_var_act
    return shot_var_2d


def _compute_weighted_centroids(
    flux_view: np.ndarray,
    rows_h: np.ndarray,
    cols_h: np.ndarray,
    w_star: np.ndarray,
    *,
    col_offset: float = 0.0,
    row_offset: float = 0.0,
    flux_scale: float = 1.0,
) -> tuple[u.Quantity, u.Quantity]:
    """Compute positive-flux, weightmap-weighted centroids in bounded chunks."""
    n_cad = flux_view.shape[0]
    col_coords = col_offset + cols_h.astype(np.float64)
    row_coords = row_offset + rows_h.astype(np.float64)
    w_sum = float(np.sum(w_star))
    fallback_col = float(np.dot(w_star, col_coords) / w_sum) if w_sum > 0.0 else np.nan
    fallback_row = float(np.dot(w_star, row_coords) / w_sum) if w_sum > 0.0 else np.nan

    cent_col = np.full(n_cad, np.nan, dtype=np.float64)
    cent_row = np.full(n_cad, np.nan, dtype=np.float64)
    for start in range(0, n_cad, _ERR_CHUNK_CADENCES):
        stop = min(start + _ERR_CHUNK_CADENCES, n_cad)
        f_act = np.asarray(flux_view[start:stop, rows_h, cols_h], dtype=np.float64) * flux_scale
        fin_act = np.isfinite(f_act)
        any_fin = np.any(fin_act, axis=1)
        pw = np.where(fin_act, np.maximum(f_act, 0.0), 0.0) * w_star[None, :]
        denom = np.sum(pw, axis=1)
        pos = denom > 0.0
        safe_denom = np.where(pos, denom, 1.0)
        cent_col[start:stop] = np.where(pos, (pw @ col_coords) / safe_denom, np.where(any_fin, fallback_col, np.nan))
        cent_row[start:stop] = np.where(pos, (pw @ row_coords) / safe_denom, np.where(any_fin, fallback_row, np.nan))

    return u.Quantity(cent_col, unit=u.pixel), u.Quantity(cent_row, unit=u.pixel)


def _construct_halo_lightcurve(
    tpf: TargetPixelFile | None,
    *,
    time_obj: Time,
    flux: np.ndarray | u.Quantity,
    flux_err: np.ndarray | u.Quantity,
    inlier_mask: np.ndarray,
    centroid_col: u.Quantity,
    centroid_row: u.Quantity,
    extra_meta: dict[str, Any],
) -> HaloLightCurve:
    """Build a ``HaloTessLightCurve``, ``HaloKeplerLightCurve``, or ``HaloLightCurve``."""
    base_meta = dict(tpf.meta) if (tpf is not None and getattr(tpf, "meta", None) is not None) else {}
    base_meta.pop("FILENAME", None)
    base_meta.pop("filename", None)
    base_meta.update(extra_meta)

    if tpf is None:
        lc: HaloLightCurve = HaloLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            centroid_col=centroid_col,
            centroid_row=centroid_row,
            meta=base_meta,
        )
        lc["inlier_mask"] = np.asarray(inlier_mask, dtype=bool)
        return lc

    mission_upper = str(getattr(tpf, "mission", "") or "").upper()
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
        lc = HaloTessLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            sector=getattr(tpf, "sector", None),
            camera=getattr(tpf, "camera", None),
            ccd=getattr(tpf, "ccd", None),
            **common_keys,
        )
    elif isinstance(tpf, KeplerTargetPixelFile) or mission_upper in ("KEPLER", "K2"):
        lc = HaloKeplerLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            channel=getattr(tpf, "channel", None),
            campaign=getattr(tpf, "campaign", None),
            quarter=getattr(tpf, "quarter", None),
            **common_keys,
        )
    else:
        lc = HaloLightCurve(
            time=time_obj,
            flux=flux,
            flux_err=flux_err,
            centroid_col=centroid_col,
            centroid_row=centroid_row,
            meta=base_meta,
        )

    if hasattr(tpf, "pos_corr1") and tpf.pos_corr1 is not None:
        lc["pos_corr1"] = tpf.pos_corr1
    if hasattr(tpf, "pos_corr2") and tpf.pos_corr2 is not None:
        lc["pos_corr2"] = tpf.pos_corr2
    lc["inlier_mask"] = np.asarray(inlier_mask, dtype=bool)
    return lc


def _staggered_subsample(
    rows_all: np.ndarray,
    cols_all: np.ndarray,
    sub: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample active pixels on a 2D staggered grid ``(r + c) % sub == 0``."""
    if sub <= 1:
        return rows_all, cols_all
    keep = ((rows_all + cols_all) % sub) == 0
    if not np.any(keep):
        keep = (np.arange(rows_all.size) % sub) == 0
    return rows_all[keep], cols_all[keep]


def _resolve_configs(
    mask_config: MaskConfig | None,
    optim_config: OptimConfig | None,
    *,
    aperture_mask: str | np.ndarray | None,
    objective: str | Callable[..., Any] | None,
    lag: int | None,
    sigclip: bool | None,
    compute_uncertainty: bool | None,
    extra_kwargs: dict[str, Any],
) -> tuple[MaskConfig, OptimConfig]:
    """Merge explicit top-level args and forwarded config overrides into ``(MaskConfig, OptimConfig)``."""
    m_over: dict[str, Any] = {}
    o_over: dict[str, Any] = {}
    for k, v in extra_kwargs.items():
        if k in _MASK_CONFIG_FIELDS:
            m_over[k] = v
        elif k in _OPTIM_CONFIG_FIELDS:
            o_over[k] = v
        else:
            raise TypeError(f"extract_halo_lightcurve() got an unexpected keyword argument {k!r}")

    if aperture_mask is not None:
        m_over["aperture_mask"] = aperture_mask
    if objective is not None:
        o_over["objective"] = objective
    if lag is not None:
        o_over["lag"] = lag
    if sigclip is not None:
        o_over["sigclip"] = sigclip
    if compute_uncertainty is not None:
        o_over["compute_uncertainty"] = compute_uncertainty

    m_cfg = dataclasses.replace(mask_config or MaskConfig(), **m_over)
    o_cfg = dataclasses.replace(optim_config or OptimConfig(), **o_over)
    return m_cfg, o_cfg


def _coerce_input(
    tpf: Any,
    times: np.ndarray | Time | None,
) -> tuple[TargetPixelFile | None, np.ndarray, Time, Any]:
    """Coerce a ``TargetPixelFile``, FITS path/``HDUList``, or raw 3D ``ndarray``."""
    if isinstance(tpf, TargetPixelFile):
        time_obj = Time(tpf.time.value, format=tpf.time.format, scale=tpf.time.scale)
        return tpf, _raw_view(tpf.flux), time_obj, getattr(tpf.flux, "unit", None)
    if isinstance(tpf, str | os.PathLike | fits.HDUList):
        loaded = lk.read(tpf)
        if not isinstance(loaded, TargetPixelFile):
            raise TypeError(
                f"Input resolved to {type(loaded).__name__}, not a lightkurve.TargetPixelFile."
            )
        time_obj = Time(loaded.time.value, format=loaded.time.format, scale=loaded.time.scale)
        return loaded, _raw_view(loaded.flux), time_obj, getattr(loaded.flux, "unit", None)

    flux_unit = getattr(tpf, "unit", None)
    flux_view = _raw_view(tpf)
    if flux_view.ndim != 3:
        raise TypeError(
            "Expected a lightkurve.TargetPixelFile, FITS path, or 3D (N_cad, ny, nx) array; "
            f"got {type(tpf).__name__} with ndim={flux_view.ndim}."
        )
    n_cad = flux_view.shape[0]
    if times is None:
        t_vals = np.arange(n_cad, dtype=np.float64) * (1.0 / 48.0)
        time_obj = Time(t_vals, format="btjd", scale="tdb")
    elif isinstance(times, Time):
        time_obj = times
    else:
        t_vals = np.asarray(times, dtype=np.float64)
        if t_vals.shape != (n_cad,):
            raise ValueError(f"times shape {t_vals.shape} does not match ({n_cad},).")
        fmt = "jd" if (np.any(np.isfinite(t_vals)) and float(np.nanmedian(t_vals)) > 2.4e6) else "btjd"
        time_obj = Time(t_vals, format=fmt, scale="tdb")
    return None, flux_view, time_obj, flux_unit


def _compute_segment_ref_fluxes(
    tpf_obj: TargetPixelFile | None,
    flux_view: np.ndarray,
    times_all: np.ndarray,
    seg_wmaps: list[np.ndarray],
    *,
    mask_cfg: MaskConfig,
    flux_scale: float,
    flux_bkg_view: np.ndarray | None,
    bkg_scale: float,
    bkgsub: bool,
) -> list[float]:
    """Compute unsaturated full-sector median reference fluxes row-by-row for split anchoring."""
    n_cad, ny, nx = flux_view.shape
    clean_cad = np.isfinite(times_all)
    if tpf_obj is not None:
        pc1 = _raw_view(tpf_obj.pos_corr1) if getattr(tpf_obj, "pos_corr1", None) is not None else None
        pc2 = _raw_view(tpf_obj.pos_corr2) if getattr(tpf_obj, "pos_corr2", None) is not None else None
        if pc1 is not None and pc2 is not None and pc1.shape == (n_cad,) and pc2.shape == (n_cad,):
            fin_pc = np.isfinite(pc1[clean_cad]) & np.isfinite(pc2[clean_cad])
            if np.sum(fin_pc) >= 0.8 * np.sum(clean_cad):
                xc = float(np.nanmedian(pc1[clean_cad]))
                yc = float(np.nanmedian(pc2[clean_cad]))
                good_p = fin_pc & (np.hypot(pc1[clean_cad] - xc, pc2[clean_cad] - yc) < 5.0)
                clean_cad[np.flatnonzero(clean_cad)[~good_p]] = False
    if not np.any(clean_cad):
        clean_cad = np.isfinite(times_all)
    clean_idx = np.flatnonzero(clean_cad)

    if mask_cfg.auto_sat_level is not None:
        sat_cut = float(mask_cfg.auto_sat_level)
    elif mask_cfg.thresh < 0.0:
        m_up = str(getattr(tpf_obj, "mission", "") or "").upper() if tpf_obj is not None else ""
        sat_cut = KEPLER_SAT_RATE_E_PER_S if m_up in ("KEPLER", "K2") else TESS_SAT_RATE_E_PER_S
    else:
        max_val = -np.inf
        for start in range(0, clean_idx.size, _ERR_CHUNK_CADENCES):
            ch = np.asarray(flux_view[clean_idx[start : start + _ERR_CHUNK_CADENCES]], dtype=np.float64) * flux_scale
            fin = ch[np.isfinite(ch)]
            if fin.size > 0:
                max_val = max(max_val, float(np.max(fin)))
        sat_cut = float(mask_cfg.thresh) * max_val if np.isfinite(max_val) else np.inf

    med_img = np.zeros((ny, nx), dtype=np.float64)
    for r in range(ny):
        strip = np.asarray(flux_view[clean_idx, r, :], dtype=np.float64) * flux_scale
        if bkgsub and flux_bkg_view is not None:
            b_strip = np.asarray(flux_bkg_view[clean_idx, r, :], dtype=np.float64) * bkg_scale
            unsub = np.where(np.isfinite(b_strip), strip + b_strip, strip)
        else:
            unsub = strip
        if np.isfinite(sat_cut):
            strip = np.where(unsub >= sat_cut, np.nan, strip)
        any_f = np.any(np.isfinite(strip), axis=0)
        if np.any(any_f):
            med_img[r, any_f] = np.nanmedian(strip[:, any_f], axis=0)

    return [float(np.sum(wm * med_img)) for wm in seg_wmaps]


def _stitch_halo_segments(
    seg_lcs: list[HaloLightCurve],
    seg_wmaps: list[np.ndarray],
    seg_werrs: list[np.ndarray],
    split_arr: np.ndarray,
    seg_scales: list[float],
    *,
    normalize: bool,
    compute_uncertainty: bool,
) -> HaloLightCurve:
    """Concatenate per-segment ``HaloLightCurve`` objects preserving class, ``inlier_mask``, and meta."""
    lc_cls = type(seg_lcs[0])
    inliers_cat = np.concatenate([np.asarray(seg["inlier_mask"], dtype=bool) for seg in seg_lcs])
    stitched_base = lk.LightCurveCollection([seg.copy() for seg in seg_lcs]).stitch(corrector_func=None)
    stitched_base.meta = dict(stitched_base.meta)
    stitched_base.meta["WEIGHTMAP"] = np.stack(seg_wmaps, axis=0)
    stitched_base.meta["SPLIT_TIMES"] = split_arr.tolist()
    if not normalize and len(seg_lcs) > 1:
        stitched_base.meta["SEGMENT_SCALES"] = seg_scales
    if compute_uncertainty and seg_werrs:
        stitched_base.meta["WEIGHTMAP_ERR"] = np.stack(seg_werrs, axis=0)
    stitched_base["inlier_mask"] = inliers_cat
    if not isinstance(stitched_base, HaloLightCurve):
        stitched_base.__class__ = lc_cls
    return stitched_base  # type: ignore[return-value]


def _extract_split_lightcurve(
    tpf_obj: TargetPixelFile | None,
    flux_view: np.ndarray,
    time_obj: Time,
    times_all: np.ndarray,
    fin_times: np.ndarray,
    split_times: np.ndarray | list[float] | tuple[float, ...],
    *,
    flux_err: np.ndarray | None,
    w_init: np.ndarray | None,
    normalize: bool,
    mask_cfg: MaskConfig,
    optim_cfg: OptimConfig,
    flux_unit_raw: Any,
    convert_counts: bool,
    exp_sec: float,
    flux_scale: float,
) -> HaloLightCurve:
    """Split a TPF or 3D cube at ``split_times``, fit each segment, anchor scales, and stitch."""
    n_cad = flux_view.shape[0]
    split_arr = np.sort(np.asarray(split_times, dtype=np.float64).ravel())
    if split_arr.size < 1 or not np.all(np.isfinite(split_arr)):
        raise ValueError("split_times must contain at least one finite timestamp.")
    t_min, t_max = float(np.min(fin_times)), float(np.max(fin_times))
    if split_arr[0] <= t_min or split_arr[-1] >= t_max:
        raise ValueError(
            f"All split_times {split_arr.tolist()} must lie strictly inside ({t_min}, {t_max})."
        )
    split_indices = [int(np.min(np.where(np.isfinite(times_all) & (times_all > st)))) for st in split_arr]
    if split_indices[0] <= 0 or split_indices[-1] >= n_cad:
        raise ValueError(f"split_times map to boundary indices {split_indices} outside (0, {n_cad}).")
    if len(set(split_indices)) != len(split_indices):
        raise ValueError(f"split_times map to duplicate cadence indices {split_indices}.")
    if w_init is not None and np.asarray(w_init).ndim == 1:
        raise ValueError("When split_times is provided, w_init must be None or 2D (ny, nx).")

    boundaries: list[int | None] = [None, *split_indices, None]
    seg_lcs: list[HaloLightCurve] = []
    seg_wmaps: list[np.ndarray] = []
    seg_werrs: list[np.ndarray] = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        sl = slice(low, high)
        if tpf_obj is not None:
            seg_in: Any = tpf_obj[sl]
            seg_times = None
            seg_fe = None
        else:
            seg_in = flux_view[sl]
            seg_times = time_obj[sl]
            seg_fe = None if flux_err is None else _raw_view(flux_err)[sl]
        lc_seg = extract_halo_lightcurve(
            seg_in,
            split_times=None,
            normalize=normalize,
            w_init=w_init,
            times=seg_times,
            flux_err=seg_fe,
            mask_config=mask_cfg,
            optim_config=optim_cfg,
        )
        seg_wmaps.append(np.asarray(lc_seg.meta["WEIGHTMAP"], dtype=np.float64))
        if optim_cfg.compute_uncertainty and "WEIGHTMAP_ERR" in lc_seg.meta:
            seg_werrs.append(np.asarray(lc_seg.meta["WEIGHTMAP_ERR"], dtype=np.float64))
        seg_lcs.append(lc_seg)

    seg_scales = [1.0] * len(seg_lcs)
    if not normalize and len(seg_lcs) > 1:
        bkgsub_full = read_bkgsub_state(tpf_obj, warn_legacy_mundey=False) if tpf_obj is not None else False
        fb_view: np.ndarray | None = None
        fb_scale = 1.0
        if tpf_obj is not None and getattr(tpf_obj, "flux_bkg", None) is not None:
            fb_view = _raw_view(tpf_obj.flux_bkg)
            fb_unit = _resolve_tpf_column_unit(
                tpf_obj, "FLUX_BKG", getattr(tpf_obj.flux_bkg, "unit", flux_unit_raw)
            )
            if convert_counts and _is_integrated_count_unit(fb_unit):
                fb_scale = 1.0 / exp_sec
        seg_refs = _compute_segment_ref_fluxes(
            tpf_obj,
            flux_view,
            times_all,
            seg_wmaps,
            mask_cfg=mask_cfg,
            flux_scale=flux_scale,
            flux_bkg_view=fb_view,
            bkg_scale=fb_scale,
            bkgsub=bkgsub_full,
        )
        valid_refs = [rf for rf in seg_refs if np.isfinite(rf) and rf > 1e-30]
        if valid_refs:
            f_ref = float(np.mean(valid_refs))
            if np.isfinite(f_ref) and f_ref > 1e-30:
                anchored: list[HaloLightCurve] = []
                seg_scales = []
                for lc_s, rf_s in zip(seg_lcs, seg_refs, strict=True):
                    lc_c = lc_s.copy()
                    if np.isfinite(rf_s) and rf_s > 1e-30:
                        scale_s = f_ref / rf_s
                        lc_c.flux = lc_c.flux * scale_s
                        lc_c.flux_err = lc_c.flux_err * scale_s
                        seg_scales.append(float(scale_s))
                    else:
                        seg_scales.append(1.0)
                    anchored.append(lc_c)
                seg_lcs = anchored

    return _stitch_halo_segments(
        seg_lcs,
        seg_wmaps,
        seg_werrs,
        split_arr,
        seg_scales,
        normalize=normalize,
        compute_uncertainty=optim_cfg.compute_uncertainty,
    )


def _load_tpf_auxiliary_views(
    tpf_obj: TargetPixelFile | None,
    flux_view: np.ndarray,
    *,
    flux_err: np.ndarray | None,
    flux_unit_raw: Any,
    convert_counts: bool,
    exp_sec: float,
    flux_scale: float,
    mask_cfg: MaskConfig,
) -> dict[str, Any]:
    """Read native float32 auxiliary views and ``HALOCVER=1`` metadata without full-cube copies."""
    _, ny, nx = flux_view.shape
    flux_err_view: np.ndarray | None = None
    flux_err_scale = 1.0
    if tpf_obj is not None and getattr(tpf_obj, "flux_err", None) is not None:
        fe_cand = _raw_view(tpf_obj.flux_err)
        fe_unit = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_ERR", getattr(tpf_obj.flux_err, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fe_unit):
            flux_err_scale = 1.0 / exp_sec
        if np.any(np.isfinite(fe_cand) & (fe_cand > 0.0)):
            flux_err_view = fe_cand
    elif flux_err is not None:
        fe_cand = _raw_view(flux_err)
        if np.any(np.isfinite(fe_cand) & (fe_cand > 0.0)):
            flux_err_view = fe_cand

    flux_bkg_view: np.ndarray | None = None
    flux_bkg_scale = 1.0
    if tpf_obj is not None and getattr(tpf_obj, "flux_bkg", None) is not None:
        flux_bkg_view = _raw_view(tpf_obj.flux_bkg)
        fb_unit = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_BKG", getattr(tpf_obj.flux_bkg, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fb_unit):
            flux_bkg_scale = 1.0 / exp_sec

    flux_bkg_err_view: np.ndarray | None = None
    flux_bkg_err_scale = 1.0
    if tpf_obj is not None and getattr(tpf_obj, "flux_bkg_err", None) is not None:
        fbe_cand = _raw_view(tpf_obj.flux_bkg_err)
        fbe_unit = _resolve_tpf_column_unit(
            tpf_obj, "FLUX_BKG_ERR", getattr(tpf_obj.flux_bkg_err, "unit", flux_unit_raw)
        )
        if convert_counts and _is_integrated_count_unit(fbe_unit):
            flux_bkg_err_scale = 1.0 / exp_sec
        if np.any(np.isfinite(fbe_cand) & (fbe_cand > 0.0)):
            flux_bkg_err_view = fbe_cand

    if tpf_obj is None:
        return {
            "flux_err_view": flux_err_view,
            "flux_err_scale": flux_err_scale,
            "flux_bkg_view": flux_bkg_view,
            "flux_bkg_scale": flux_bkg_scale,
            "flux_bkg_err_view": flux_bkg_err_view,
            "flux_bkg_err_scale": flux_bkg_err_scale,
            "pq_2d": np.zeros((ny, nx), dtype=np.uint16),
            "bkgsub": False,
            "bkg_order": None,
            "smear_err_2d": None,
            "bkg_cov": None,
            "pc1": None,
            "pc2": None,
            "tgt_coord": None,
            "resolved_tmag": mask_cfg.target_mag,
            "wcs_obj": None,
            "obs_time_obj": None,
            "mission_val": None,
        }

    pq_2d = read_pixel_quality(tpf_obj)
    bkgsub = read_bkgsub_state(tpf_obj, warn_legacy_mundey=True)
    bkg_order = infer_bkg_poly_order(tpf_obj)
    smear_err_2d = read_smear_err(tpf_obj, quality_mask=True)
    if convert_counts and smear_err_2d is not None:
        se_unit = _resolve_tpf_column_unit(tpf_obj, "SMEAR_ERR", flux_unit_raw)
        if _is_integrated_count_unit(se_unit):
            smear_err_2d = smear_err_2d / exp_sec
    bkg_cov = read_bkg_covariance(tpf_obj, quality_mask=True)
    if convert_counts and bkg_cov is not None:
        chol_unit = _resolve_tpf_column_unit(
            tpf_obj, "BKG_CHOL", _resolve_hdu_unit(tpf_obj, "BKG_CHOL", flux_unit_raw)
        )
        if _is_integrated_count_unit(chol_unit):
            bkg_cov = BkgCovariance(chol=bkg_cov.chol / exp_sec, basis=bkg_cov.basis)

    pc1 = _raw_view(tpf_obj.pos_corr1) if getattr(tpf_obj, "pos_corr1", None) is not None else None
    pc2 = _raw_view(tpf_obj.pos_corr2) if getattr(tpf_obj, "pos_corr2", None) is not None else None
    tgt_coord = (
        (float(tpf_obj.ra), float(tpf_obj.dec))
        if getattr(tpf_obj, "ra", None) is not None and getattr(tpf_obj, "dec", None) is not None
        else None
    )
    resolved_tmag = (
        mask_cfg.target_mag
        if mask_cfg.target_mag is not None
        else _estimate_target_mag(
            tpf=tpf_obj,
            flux=flux_view,
            mission=getattr(tpf_obj, "mission", None),
            flux_scale=flux_scale,
        )
    )
    return {
        "flux_err_view": flux_err_view,
        "flux_err_scale": flux_err_scale,
        "flux_bkg_view": flux_bkg_view,
        "flux_bkg_scale": flux_bkg_scale,
        "flux_bkg_err_view": flux_bkg_err_view,
        "flux_bkg_err_scale": flux_bkg_err_scale,
        "pq_2d": pq_2d,
        "bkgsub": bkgsub,
        "bkg_order": bkg_order,
        "smear_err_2d": smear_err_2d,
        "bkg_cov": bkg_cov,
        "pc1": pc1,
        "pc2": pc2,
        "tgt_coord": tgt_coord,
        "resolved_tmag": resolved_tmag,
        "wcs_obj": _resolve_tpf_wcs(tpf_obj),
        "obs_time_obj": _resolve_tpf_obs_time(tpf_obj),
        "mission_val": getattr(tpf_obj, "mission", None),
    }


def _parse_aperture_mask_and_annulus(
    tpf_obj: TargetPixelFile | None,
    ny: int,
    nx: int,
    mask_cfg: MaskConfig,
) -> np.ndarray:
    """Resolve ``aperture_mask`` (string or 2D array) and optional radial annulus ``rr``."""
    if tpf_obj is not None:
        parsed_ap = np.asarray(tpf_obj._parse_aperture_mask(mask_cfg.aperture_mask), dtype=bool)
    elif isinstance(mask_cfg.aperture_mask, str):
        if mask_cfg.aperture_mask != "all":
            raise ValueError(
                f"String aperture_mask={mask_cfg.aperture_mask!r} requires a TargetPixelFile; "
                "pass 'all' or a 2D boolean mask for raw 3D arrays."
            )
        parsed_ap = np.ones((ny, nx), dtype=bool)
    else:
        parsed_ap = np.asarray(mask_cfg.aperture_mask, dtype=bool)

    if mask_cfg.rr is not None:
        rmin, rmax = float(mask_cfg.rr[0]), float(mask_cfg.rr[1])
        xx, yy = np.meshgrid(np.arange(nx, dtype=np.float64) - nx / 2.0, np.arange(ny, dtype=np.float64) - ny / 2.0)
        rad = np.hypot(xx, yy)
        parsed_ap = parsed_ap & (rad >= rmin) & (rad <= rmax)
    return parsed_ap


def _resolve_w_init_vector(
    w_init: np.ndarray | None,
    rows_h: np.ndarray,
    cols_h: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray | None:
    """Validate and extract 1D ``(M,)`` initial simplex weights from 1D or 2D ``w_init``."""
    if w_init is None:
        return None
    w_arr = np.asarray(w_init, dtype=np.float64)
    if w_arr.ndim == 2:
        if w_arr.shape != (ny, nx):
            raise ValueError(f"2D w_init shape {w_arr.shape} does not match {(ny, nx)}.")
        ext = np.where(
            np.isfinite(w_arr[rows_h, cols_h]) & (w_arr[rows_h, cols_h] > 0.0),
            w_arr[rows_h, cols_h],
            0.0,
        )
        s_ext = float(np.sum(ext))
        return ext / s_ext if s_ext > 0.0 else None
    if w_arr.ndim == 1:
        if w_arr.shape != (rows_h.size,):
            raise ValueError(f"1D w_init shape {w_arr.shape} does not match ({rows_h.size},).")
        return w_arr
    raise ValueError(f"w_init must be 1D (M,) or 2D (ny, nx), got shape {w_arr.shape}.")


@in_float64
def extract_halo_lightcurve(
    tpf: Any,
    *,
    aperture_mask: str | np.ndarray | None = None,
    split_times: np.ndarray | list[float] | tuple[float, ...] | None = None,
    objective: str | Callable[..., Any] | None = None,
    lag: int | None = None,
    sigclip: bool | None = None,
    normalize: bool = True,
    compute_uncertainty: bool | None = None,
    w_init: np.ndarray | None = None,
    times: np.ndarray | Time | None = None,
    flux_err: np.ndarray | None = None,
    mask_config: MaskConfig | None = None,
    optim_config: OptimConfig | None = None,
    legacy_tv: bool | None = None,
    **kwargs: Any,
) -> HaloLightCurve:
    """Extract a halo-photometry :class:`HaloLightCurve` with coherent ``flux_err``.

    Accepts a ``lightkurve.TargetPixelFile``, a TPF FITS path / ``HDUList``, or
    a raw 3D ``(N_cad, ny, nx)`` NumPy array. Pass any registered name in
    :data:`~halophot.OBJECTIVES` or a custom JAX callable as ``objective=...``.
    Additional masking or solver fields from :class:`MaskConfig` and
    :class:`OptimConfig` may be passed either via ``mask_config`` /
    ``optim_config`` or directly as keyword arguments.
    """
    if legacy_tv is not None:
        warnings.warn(
            "legacy_tv is deprecated and has no effect; halophot now optimizes "
            "physical simplex weights directly.",
            DeprecationWarning,
            stacklevel=3,
        )

    mask_cfg, optim_cfg = _resolve_configs(
        mask_config,
        optim_config,
        aperture_mask=aperture_mask,
        objective=objective,
        lag=lag,
        sigclip=sigclip,
        compute_uncertainty=compute_uncertainty,
        extra_kwargs=kwargs,
    )
    if isinstance(mask_cfg.sub, bool) or int(mask_cfg.sub) < 1:
        raise ValueError(f"sub must be an integer >= 1, got {mask_cfg.sub!r}.")
    sub_int = int(mask_cfg.sub)

    tpf_obj, flux_view, time_obj, flux_unit_attr = _coerce_input(tpf, times)
    n_cad, ny, nx = flux_view.shape
    if n_cad < 2:
        raise ValueError(f"Input must have at least 2 cadences, got {n_cad}.")

    times_all = np.asarray(time_obj.value, dtype=np.float64)
    fin_times = times_all[np.isfinite(times_all)]
    if fin_times.size < 2:
        raise ValueError("Input contains fewer than 2 finite timestamps.")
    pos_diffs = np.diff(fin_times)
    pos_diffs = pos_diffs[pos_diffs > 0.0]
    dt_days = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)

    flux_unit_raw = (
        _resolve_tpf_column_unit(tpf_obj, "FLUX", flux_unit_attr)
        if tpf_obj is not None
        else flux_unit_attr
    )
    convert_counts = _is_integrated_count_unit(flux_unit_raw)
    exp_sec = _resolve_cadence_exposure_sec(tpf_obj, dt_days) if (convert_counts and tpf_obj is not None) else 1.0
    flux_scale = (1.0 / exp_sec) if convert_counts else 1.0
    eff_flux_unit = _integrated_to_rate_unit(flux_unit_raw) if convert_counts else flux_unit_attr

    if split_times is not None:
        return _extract_split_lightcurve(
            tpf_obj,
            flux_view,
            time_obj,
            times_all,
            fin_times,
            split_times,
            flux_err=flux_err,
            w_init=w_init,
            normalize=normalize,
            mask_cfg=mask_cfg,
            optim_cfg=optim_cfg,
            flux_unit_raw=flux_unit_raw,
            convert_counts=convert_counts,
            exp_sec=exp_sec,
            flux_scale=flux_scale,
        )

    parsed_ap = _parse_aperture_mask_and_annulus(tpf_obj, ny, nx, mask_cfg)
    aux = _load_tpf_auxiliary_views(
        tpf_obj,
        flux_view,
        flux_err=flux_err,
        flux_unit_raw=flux_unit_raw,
        convert_counts=convert_counts,
        exp_sec=exp_sec,
        flux_scale=flux_scale,
        mask_cfg=mask_cfg,
    )

    mask_res = build_halo_masks(
        flux_view,
        flux_err=aux["flux_err_view"],
        flux_bkg=aux["flux_bkg_view"],
        bkgsub=aux["bkgsub"],
        pixel_quality=aux["pq_2d"],
        aperture_mask=parsed_ap,
        cadence_quality_mask=np.isfinite(times_all),
        pos_corr1=aux["pc1"],
        pos_corr2=aux["pc2"],
        wcs=aux["wcs_obj"],
        target_coord=aux["tgt_coord"],
        target_mag=aux["resolved_tmag"],
        obs_time=aux["obs_time_obj"],
        mission=aux["mission_val"],
        dt_days=dt_days,
        flux_scale=flux_scale,
        flux_err_scale=aux["flux_err_scale"],
        flux_bkg_scale=aux["flux_bkg_scale"],
        config=mask_cfg,
    )

    rows_all, cols_all = np.where(mask_res.pixel_mask)
    rows_h, cols_h = _staggered_subsample(rows_all, cols_all, sub_int)
    if rows_h.size < 1:
        raise ValueError("No halo pixels remained after subsampling.")

    # Extract ONLY active halo pixels (M, N_good) into float64!
    active_cad = mask_res.cadence_mask
    act_idx = np.flatnonzero(active_cad)
    pixels_active = np.empty((rows_h.size, act_idx.size), dtype=np.float64)
    for start in range(0, act_idx.size, _ERR_CHUNK_CADENCES):
        stop = min(start + _ERR_CHUNK_CADENCES, act_idx.size)
        ch = np.asarray(flux_view[act_idx[start:stop]][:, rows_h, cols_h], dtype=np.float64)
        if flux_scale != 1.0:
            ch = ch * flux_scale
        pixels_active[:, start:stop] = ch.T
    times_active = times_all[act_idx]

    w_init_vec = _resolve_w_init_vector(w_init, rows_h, cols_h, ny, nx)
    fit = optimise_halo(pixels_active, times_active, w_init=w_init_vec, config=optim_cfg)

    weightmap_2d = np.zeros((ny, nx), dtype=np.float64)
    weightmap_2d[rows_h, cols_h] = fit.weights

    # Exclude !inlier_mask ONLY during optimization & baseline normalization; preserve all finite cadences.
    inlier_cad_mask = np.zeros(n_cad, dtype=bool)
    inlier_cad_mask[act_idx[fit.inlier_mask]] = True
    norm_idx = fit.inlier_mask if np.any(fit.inlier_mask) else np.ones(act_idx.size, dtype=bool)

    mean_flux_norm = float(np.mean(fit.raw_flux[norm_idx]))
    if not np.isfinite(mean_flux_norm) or abs(mean_flux_norm) <= 1e-30:
        mean_flux_norm = 1.0
    norm_scale = mean_flux_norm if normalize else 1.0

    lc_flux = np.full(n_cad, np.nan, dtype=np.float64)
    lc_flux[act_idx] = fit.raw_flux / norm_scale

    shot_var_2d: np.ndarray | None = None
    if _is_photon_rate_unit(eff_flux_unit):
        shot_var_2d = _compute_detector_shot_var(
            tpf_obj,
            flux_view,
            aux["flux_bkg_view"],
            active_cad,
            rows_h,
            cols_h,
            bkgsub=aux["bkgsub"],
            dt_days=dt_days,
            flux_scale=flux_scale,
            bkg_scale=aux["flux_bkg_scale"],
        )

    bg_mask_2d = (aux["pq_2d"] & np.uint16(PixelQuality.BKG_USED)) != 0
    has_any_err = (
        aux["flux_err_view"] is not None
        or aux["smear_err_2d"] is not None
        or aux["bkg_cov"] is not None
        or aux["flux_bkg_err_view"] is not None
    )
    if has_any_err:
        lc_flux_err = compute_coherent_flux_err(
            weightmap_2d,
            flux_err=aux["flux_err_view"],
            smear_err=aux["smear_err_2d"],
            bkg_cov=aux["bkg_cov"],
            flux_bkg_err=aux["flux_bkg_err_view"],
            bg_mask=bg_mask_2d if np.any(bg_mask_2d) else None,
            shot_var=shot_var_2d,
            bkgsub=aux["bkgsub"],
            bkg_order=aux["bkg_order"],
            norm_factor=norm_scale,
            flux_err_scale=aux["flux_err_scale"],
            flux_bkg_err_scale=aux["flux_bkg_err_scale"],
        )
        lc_flux_err[~active_cad] = np.nan
    else:
        lc_flux_err = np.full(n_cad, np.nan, dtype=np.float64)

    cent_col, cent_row = _compute_weighted_centroids(
        flux_view,
        rows_h,
        cols_h,
        fit.weights,
        col_offset=float(getattr(tpf_obj, "column", 0) or 0) if tpf_obj is not None else 0.0,
        row_offset=float(getattr(tpf_obj, "row", 0) or 0) if tpf_obj is not None else 0.0,
        flux_scale=flux_scale,
    )

    obj_label = (
        optim_cfg.objective
        if isinstance(optim_cfg.objective, str)
        else getattr(optim_cfg.objective, "__name__", "custom")
    )
    extra_meta: dict[str, Any] = {
        "WEIGHTMAP": weightmap_2d,
        "HALO_OBJ": str(obj_label)[:32],
        "HALO_LAG": int(optim_cfg.lag),
        "BKGSUB": bool(aux["bkgsub"]),
    }
    if aux["bkg_order"] is not None:
        extra_meta["BKG_ORD"] = int(aux["bkg_order"])
    if optim_cfg.compute_uncertainty and fit.weight_std is not None:
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

    return _construct_halo_lightcurve(
        tpf_obj,
        time_obj=time_obj,
        flux=flux_arg,
        flux_err=err_arg,
        inlier_mask=inlier_cad_mask,
        centroid_col=cent_col,
        centroid_row=cent_row,
        extra_meta=extra_meta,
    )


def plot_diagnostics(
    lc: lk.LightCurve,
    tpf: Any | None = None,
    *,
    save_path: str | os.PathLike[str] | None = None,
):
    """Plot a multi-panel diagnostic figure for an extracted halo light curve.

    Displays the median flux stamp (when ``tpf`` is supplied), the native
    ``(ny, nx)`` halo ``WEIGHTMAP`` with colour bar, the extracted light curve
    (highlighting ``sigclip`` outliers from ``lc['inlier_mask']``), and the
    Lomb-Scargle amplitude power spectrum.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    t_vals = np.asarray(lc.time.value, dtype=np.float64)
    f_raw = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    f_vals = np.asarray(f_raw, dtype=np.float64)
    f_unit = getattr(lc.flux, "unit", None)
    unit_str = (
        f" ({f_unit})"
        if f_unit is not None and f_unit != u.dimensionless_unscaled
        else " (relative)"
    )

    wmap_raw = lc.meta.get("WEIGHTMAP")
    wmap_2d: np.ndarray | None = None
    n_splits = 1
    if wmap_raw is not None:
        wm_arr = np.asarray(wmap_raw, dtype=np.float64)
        if wm_arr.ndim == 3:
            n_splits = wm_arr.shape[0]
            wmap_2d = np.nanmean(wm_arr, axis=0)
        elif wm_arr.ndim == 2:
            wmap_2d = wm_arr

    has_tpf = tpf is not None
    if has_tpf:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
        ax_flux_map, ax_wmap, ax_lc, ax_pg = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
        ax_flux_map = None
        ax_wmap, ax_lc, ax_pg = axes[0], axes[1], axes[2]

    if ax_flux_map is not None:
        tpf_flux = _raw_view(tpf.flux) if hasattr(tpf, "flux") else _raw_view(tpf)
        stride = max(1, tpf_flux.shape[0] // 256)
        med_img = np.nanmedian(np.asarray(tpf_flux[::stride], dtype=np.float64), axis=0)
        pos = np.isfinite(med_img) & (med_img > 0.0)
        log_img = np.where(pos, np.log10(np.where(pos, med_img, 1.0)), np.nan)
        cmap_f = mpl.colormaps["inferno"].with_extremes(bad="0.15")
        im_f = ax_flux_map.imshow(log_img, origin="lower", cmap=cmap_f, aspect="auto")
        fig.colorbar(im_f, ax=ax_flux_map, label=r"$\log_{10}$ Median Flux (e$^-$/s)")
        ax_flux_map.set_xlabel("Column (pix)")
        ax_flux_map.set_ylabel("Row (pix)")
        ax_flux_map.set_title("Median Stamp Flux")

    if wmap_2d is not None:
        pos_w = np.isfinite(wmap_2d) & (wmap_2d > 0.0)
        log_w = np.where(pos_w, np.log10(np.where(pos_w, wmap_2d, 1.0)), np.nan)
        cmap_w = mpl.colormaps["viridis"].with_extremes(bad="0.15")
        im_w = ax_wmap.imshow(log_w, origin="lower", cmap=cmap_w, aspect="auto")
        fig.colorbar(im_w, ax=ax_wmap, label=r"$\log_{10}$ Weight")
        title_w = "Halo Weightmap" if n_splits == 1 else f"Mean Halo Weightmap ({n_splits} splits)"
        ax_wmap.set_title(title_w)
    else:
        ax_wmap.text(0.5, 0.5, "No WEIGHTMAP", ha="center", va="center", transform=ax_wmap.transAxes)
        ax_wmap.set_title("Halo Weightmap")
    ax_wmap.set_xlabel("Column (pix)")
    ax_wmap.set_ylabel("Row (pix)")

    fin = np.isfinite(t_vals) & np.isfinite(f_vals)
    if "inlier_mask" in lc.colnames:
        inl = np.asarray(lc["inlier_mask"], dtype=bool)
        good = fin & inl
        clipped = fin & (~inl)
    else:
        good = fin
        clipped = np.zeros_like(fin, dtype=bool)

    if np.any(good):
        ax_lc.plot(t_vals[good], f_vals[good], ".", ms=2.5, alpha=0.8, label="Inliers")
    if np.any(clipped):
        ax_lc.plot(t_vals[clipped], f_vals[clipped], "x", color="tab:red", ms=4.5, label="Clipped (opt)")
        ax_lc.legend(loc="best", fontsize=8)
    ax_lc.set_xlabel(f"Time ({lc.time.format.upper()}, d)")
    ax_lc.set_ylabel(f"Flux{unit_str}")
    ax_lc.set_title(f"Halo Light Curve ({lc.meta.get('HALO_OBJ', 'tv')})")

    if np.sum(good) >= 8:
        t_g = t_vals[good]
        f_g = f_vals[good]
        med_g = float(np.median(f_g))
        rel_g = (f_g / med_g - 1.0) if abs(med_g) > 1e-30 else (f_g - med_g)
        span = max(float(t_g[-1] - t_g[0]), 1e-3)
        dt_g = float(np.median(np.diff(t_g)))
        f_nyq = 0.5 / max(dt_g, 1e-5)
        ls = LombScargle(t_g, rel_g, normalization="psd")
        freq, power = ls.autopower(
            minimum_frequency=1.0 / span,
            maximum_frequency=min(f_nyq, 24.0),
            samples_per_peak=5,
        )
        if freq.size > 1 and np.sum(power) > 0.0:
            df_muhz = float(np.mean(np.diff(freq * _DAY_TO_MUHZ)))
            psd = power * (float(np.std(rel_g * 1e6)) ** 2 / (float(np.sum(power)) * max(df_muhz, 1e-30)))
            ax_pg.plot(freq * _DAY_TO_MUHZ, psd, lw=0.8, color="0.3")
            ax_pg.set_yscale("log")
    ax_pg.set_xlabel(r"Frequency ($\mu$Hz)")
    ax_pg.set_ylabel(r"PSD (ppm$^2/\mu$Hz)")
    ax_pg.set_title("Lomb-Scargle Power Spectrum")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
