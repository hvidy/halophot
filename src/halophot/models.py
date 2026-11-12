"""Zodiax v0.5.0 PyTree models for halo photometry optimization.

``HaloWeights(zdx.Base)`` isolates ``z_ilr: jax.Array`` as its sole differentiable
array leaf so ``eqx.filter_value_and_grad`` and ``optimistix`` differentiate
strictly with respect to the unconstrained ILR coordinates ``z in R^{M-1}``.
``HaloObservation(zdx.Base)`` holds static calibration arrays padded to bucketed
``N_pad`` cadences for XLA compilation reuse across ``sigclip`` refits, orbit
block-jackknife folds, and multi-sector surveys.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zodiax as zdx

from ._precision import in_float64
from .simplex import (
    flux_share_to_weights,
    ilr_to_simplex,
    uniform_weights_ilr,
    weights_to_ilr,
)

__all__ = [
    "HaloObservation",
    "HaloWeights",
    "bucket_pad_length",
    "compute_gap_diff_mask",
]

# Bucket ladder covering short K2 splits up to full 20-second TESS sectors (~130k cadences).
_PAD_BUCKETS: tuple[int, ...] = (
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
)


def bucket_pad_length(n_cad: int) -> int:
    """Return a bucketed time-axis padding length ``N_pad >= n_cad`` for XLA cache reuse."""
    if isinstance(n_cad, bool) or int(n_cad) < 1:
        raise ValueError(f"n_cad must be an integer >= 1, got {n_cad!r}.")
    n_int = int(n_cad)
    for bucket in _PAD_BUCKETS:
        if n_int <= bucket:
            return bucket
    # Above 262,144 cadences, round up to the next multiple of 65,536.
    return ((n_int + 65535) // 65536) * 65536


def compute_gap_diff_mask(
    times_padded: jax.Array,
    valid_cadence_mask: jax.Array,
    lag: int,
    dt_nom: jax.Array | float,
    *,
    gap_tol: float = 0.25,
) -> jax.Array:
    """Construct the lag-``lag`` gap-valid index mask ``V_lag`` of shape ``(N_pad,)``.

    Entry ``k`` (for ``0 <= k < N_pad - lag``) is ``True`` iff both cadences
    ``k`` and ``k + lag`` are active and
    ``|(t_{k+lag} - t_k) - lag * dt_nom| <= gap_tol * lag * dt_nom``, ensuring
    finite differences never bridge spacecraft downlink or safe-mode gaps.
    """
    n_pad = times_padded.shape[0]
    if lag < 1 or lag >= n_pad:
        return jnp.zeros(n_pad, dtype=jnp.bool_)

    dt_pair = times_padded[lag:] - times_padded[:-lag]
    expected_dt = float(lag) * jnp.asarray(dt_nom, dtype=times_padded.dtype)
    pair_ok = (
        valid_cadence_mask[:-lag]
        & valid_cadence_mask[lag:]
        & (jnp.abs(dt_pair - expected_dt) <= gap_tol * expected_dt)
    )
    tail_false = jnp.zeros(lag, dtype=jnp.bool_)
    return jnp.concatenate([pair_ok, tail_false], axis=0)


class HaloWeights(zdx.Base):
    """Differentiable halo weight PyTree with ``z_ilr`` as its sole array leaf."""

    z_ilr: jax.Array
    legacy_tv: bool = eqx.field(static=True, default=False)

    @classmethod
    @in_float64
    def from_uniform(
        cls,
        mean_pixel_flux: jax.Array | np.ndarray,
        *,
        legacy_tv: bool = False,
    ) -> HaloWeights:
        """Initialize at uniform physical pixel weights ``w_i = 1 / M``."""
        z0 = uniform_weights_ilr(jnp.asarray(mean_pixel_flux, dtype=jnp.float64), legacy_tv=legacy_tv)
        return cls(z_ilr=z0, legacy_tv=bool(legacy_tv))

    @classmethod
    @in_float64
    def from_physical_weights(
        cls,
        w: jax.Array | np.ndarray,
        mean_pixel_flux: jax.Array | np.ndarray,
        *,
        legacy_tv: bool = False,
    ) -> HaloWeights:
        """Initialize from an explicit physical simplex weight vector ``w in Delta^{M-1}``."""
        z0 = weights_to_ilr(
            jnp.asarray(w, dtype=jnp.float64),
            jnp.asarray(mean_pixel_flux, dtype=jnp.float64),
            legacy_tv=legacy_tv,
        )
        return cls(z_ilr=z0, legacy_tv=bool(legacy_tv))

    def flux_shares(self) -> jax.Array:
        """Return simplex coordinates ``theta(z_ilr) in Delta^{M-1}``."""
        return ilr_to_simplex(self.z_ilr)

    def physical_weights(self, mean_pixel_flux: jax.Array) -> jax.Array:
        """Return physical aperture weights ``w in Delta^{M-1}``."""
        theta = ilr_to_simplex(self.z_ilr)
        if self.legacy_tv:
            return theta
        return flux_share_to_weights(theta, mean_pixel_flux)

    def evaluate_lightcurve(self, obs: HaloObservation) -> jax.Array:
        """Evaluate the unit-mean padded light curve ``f_tilde`` of shape ``(N_pad,)``."""
        if not self.legacy_tv:
            theta = ilr_to_simplex(self.z_ilr)
            return jnp.dot(theta, obs.norm_pixels)

        w = ilr_to_simplex(self.z_ilr)
        weighted_p = jnp.dot(w * obs.mean_pixel_flux, obs.norm_pixels)
        mean_f = jnp.dot(w, obs.mean_pixel_flux)
        safe_denom = jnp.where(
            jnp.abs(mean_f) > jnp.finfo(weighted_p.dtype).tiny,
            mean_f,
            jnp.ones_like(mean_f),
        )
        return weighted_p / safe_denom


class HaloObservation(zdx.Base):
    """Static calibration and cadence-mask PyTree passed to solvers via ``args=obs``."""

    norm_pixels: jax.Array
    mean_pixel_flux: jax.Array
    mean_frac_var: jax.Array
    valid_diff_mask: jax.Array
    z_prior_centre: jax.Array
    times_padded: jax.Array
    valid_cadence_mask: jax.Array
    dt_nom: jax.Array
    b_hat: jax.Array
    lambda_phot: jax.Array
    lambda_z: jax.Array
    n_valid: jax.Array
    N_pad: int = eqx.field(static=True)
    lag: int = eqx.field(static=True, default=1)
    gap_tol: float = eqx.field(static=True, default=0.25)

    @classmethod
    @in_float64
    def from_arrays(
        cls,
        pixels: np.ndarray | jax.Array,
        times: np.ndarray | jax.Array | None = None,
        *,
        pixel_var: np.ndarray | jax.Array | None = None,
        lag: int = 1,
        legacy_tv: bool = False,
        lambda_phot: float | None = None,
        lambda_z: float | None = None,
        n_pad: int | None = None,
        gap_tol: float = 0.25,
    ) -> HaloObservation:
        """Construct a padded ``HaloObservation`` in 64-bit precision.

        Parameters
        ----------
        pixels : array-like, shape ``(M, N)``
            Active halo pixel time series strictly over valid cadences ``G``.
        times : array-like, shape ``(N,)``, optional
            Monotonically increasing timestamps in days. Defaults to uniform
            cadences ``0, 1, ..., N - 1`` (30-min equivalent shot-noise fallback).
        pixel_var : array-like, shape ``(M, N)`` or ``(M,)``, optional
            Per-pixel independent variance ``sigma_P^2`` in physical flux^2 units.
            When omitted and ``legacy_tv=False``, defaults to Poisson shot-noise
            scaling ``P_bar_i / dt_sec``.
        lag : int, default 1
            Finite-difference lag ``ell >= 1``.
        legacy_tv : bool, default False
            When ``True``, sets ``lambda_phot = 0``, ``lambda_z = 0``, and
            ``z_prior_centre = 0`` for exact *K2* / ``halo-tess`` DR1 reproduction.
        lambda_phot : float, optional
            Photon-noise regularization strength. When ``None`` and ``not legacy_tv``,
            auto-calibrated to ``0.1 * |V_lag| / max(b_hat^2, median(mean_frac_var))``.
        lambda_z : float, optional
            ILR logit-damping strength around uniform-``w`` ``z_init``. Defaults
            to ``1e-3`` when ``not legacy_tv`` and ``0.0`` when ``legacy_tv=True``.
        n_pad : int, optional
            Explicit padded time length ``>= N``. Defaults to ``bucket_pad_length(N)``.
        gap_tol : float, default 0.25
            Fractional tolerance around ``lag * dt_nom`` for gap-aware differencing.
        """
        if isinstance(lag, bool) or int(lag) < 1:
            raise ValueError(f"lag must be an integer >= 1, got {lag!r}.")
        lag_int = int(lag)

        pix_np = np.asarray(pixels, dtype=np.float64)
        if pix_np.ndim != 2 or pix_np.shape[0] < 1 or pix_np.shape[1] < 2:
            raise ValueError(
                f"pixels must be 2D (M, N) with M >= 1 and N >= 2, got shape {pix_np.shape}."
            )
        if not np.all(np.isfinite(pix_np)):
            raise ValueError("pixels contains non-finite values.")
        m_pix, n_cad = pix_np.shape

        if times is None:
            times_np = np.arange(n_cad, dtype=np.float64) * (1.0 / 48.0)
        else:
            times_np = np.asarray(times, dtype=np.float64)
            if times_np.shape != (n_cad,):
                raise ValueError(
                    f"times shape {times_np.shape} does not match cadence count ({n_cad},)."
                )
            if not np.all(np.isfinite(times_np)):
                raise ValueError("times contains non-finite timestamps.")

        diffs_t = np.diff(times_np)
        pos_diffs = diffs_t[diffs_t > 0.0]
        dt_nom_val = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)

        resolved_pad = bucket_pad_length(n_cad) if n_pad is None else int(n_pad)
        if resolved_pad < n_cad:
            raise ValueError(f"n_pad={resolved_pad} must be >= n_cad={n_cad}.")

        mean_p = np.mean(pix_np, axis=1)
        if not legacy_tv and np.any(mean_p <= 0.0):
            raise ValueError(
                "Charnes-Cooper coordinates require strictly positive mean pixel flux "
                "on all active halo pixels (mean_pixel_flux > 0)."
            )

        safe_scale = np.where(np.abs(mean_p) > 1e-30, mean_p, 1.0)
        norm_p = pix_np / safe_scale[:, np.newaxis]

        if pixel_var is not None:
            pvar_np = np.asarray(pixel_var, dtype=np.float64)
            if pvar_np.shape == (m_pix, n_cad):
                mean_var_raw = np.nanmean(np.maximum(pvar_np, 0.0), axis=1)
            elif pvar_np.shape == (m_pix,):
                mean_var_raw = np.maximum(pvar_np, 0.0)
            else:
                raise ValueError(
                    f"pixel_var must have shape ({m_pix}, {n_cad}) or ({m_pix},), "
                    f"got {pvar_np.shape}."
                )
            mean_var_raw = np.where(np.isfinite(mean_var_raw), mean_var_raw, 0.0)
            frac_var = mean_var_raw / np.maximum(safe_scale**2, 1e-30)
        else:
            dt_sec = max(dt_nom_val * 86400.0, 1.0)
            pos_mean = np.maximum(np.abs(mean_p), 1.0)
            frac_var = 1.0 / (pos_mean * dt_sec)

        frac_var = np.maximum(frac_var, 0.0)

        norm_padded = np.ones((m_pix, resolved_pad), dtype=np.float64)
        norm_padded[:, :n_cad] = norm_p

        valid_cad = np.zeros(resolved_pad, dtype=bool)
        valid_cad[:n_cad] = True

        times_padded_np = np.empty(resolved_pad, dtype=np.float64)
        times_padded_np[:n_cad] = times_np
        if resolved_pad > n_cad:
            pad_offsets = np.arange(1, resolved_pad - n_cad + 1, dtype=np.float64)
            times_padded_np[n_cad:] = times_np[-1] + 1e6 * dt_nom_val * pad_offsets

        times_j = jnp.asarray(times_padded_np, dtype=jnp.float64)
        valid_cad_j = jnp.asarray(valid_cad, dtype=jnp.bool_)
        dt_nom_j = jnp.asarray(dt_nom_val, dtype=jnp.float64)
        diff_mask_j = compute_gap_diff_mask(
            times_j,
            valid_cad_j,
            lag_int,
            dt_nom_j,
            gap_tol=gap_tol,
        )
        # Fallback if timestamps are irregular everywhere (e.g. non-uniform synthetic grid):
        # retain consecutive active pairs within [0, n_cad - lag) if gap filter left 0 pairs.
        if n_cad > lag_int and int(jnp.sum(diff_mask_j)) == 0:
            fallback_mask = np.zeros(resolved_pad, dtype=bool)
            fallback_mask[: n_cad - lag_int] = True
            diff_mask_j = jnp.asarray(fallback_mask, dtype=jnp.bool_)

        mean_p_j = jnp.asarray(mean_p, dtype=jnp.float64)
        z_init_j = uniform_weights_ilr(mean_p_j, legacy_tv=legacy_tv)

        if legacy_tv:
            b_hat_val = 1.0
            lam_phot_val = 0.0
            lam_z_val = 0.0
        else:
            theta_init = ilr_to_simplex(z_init_j)
            f_init = np.asarray(jnp.dot(theta_init, jnp.asarray(norm_p, dtype=jnp.float64)))
            valid_idx = np.flatnonzero(np.asarray(diff_mask_j)[:n_cad])
            if valid_idx.size > 0:
                abs_diffs = np.abs(f_init[valid_idx + lag_int] - f_init[valid_idx])
                med_abs = float(np.median(abs_diffs)) / np.log(2.0)
                mean_abs = float(np.mean(abs_diffs))
                b_hat_val = max(med_abs if med_abs > 0.0 else mean_abs, 1e-6)
            else:
                b_hat_val = max(float(np.std(f_init)), 1e-6)

            n_v = max(int(valid_idx.size), 1)
            if lambda_phot is None:
                med_fv = float(np.median(frac_var)) if frac_var.size > 0 else 0.0
                ref_var = max(b_hat_val**2, med_fv, 1e-12)
                lam_phot_val = 0.1 * float(n_v) / ref_var
            else:
                lam_phot_val = float(lambda_phot)
                if lam_phot_val < 0.0:
                    raise ValueError(f"lambda_phot must be >= 0, got {lambda_phot!r}.")

            if lambda_z is None:
                lam_z_val = 1e-3
            else:
                lam_z_val = float(lambda_z)
                if lam_z_val < 0.0:
                    raise ValueError(f"lambda_z must be >= 0, got {lambda_z!r}.")

        return cls(
            norm_pixels=jnp.asarray(norm_padded, dtype=jnp.float64),
            mean_pixel_flux=mean_p_j,
            mean_frac_var=jnp.asarray(frac_var, dtype=jnp.float64),
            valid_diff_mask=diff_mask_j,
            z_prior_centre=z_init_j,
            times_padded=times_j,
            valid_cadence_mask=valid_cad_j,
            dt_nom=dt_nom_j,
            b_hat=jnp.asarray(b_hat_val, dtype=jnp.float64),
            lambda_phot=jnp.asarray(lam_phot_val, dtype=jnp.float64),
            lambda_z=jnp.asarray(lam_z_val, dtype=jnp.float64),
            n_valid=jnp.asarray(n_cad, dtype=jnp.int32),
            N_pad=resolved_pad,
            lag=lag_int,
            gap_tol=float(gap_tol),
        )

    @in_float64
    def with_cadence_subset(
        self,
        keep_mask: np.ndarray | jax.Array,
        *,
        legacy_tv: bool = False,
    ) -> HaloObservation:
        """Build a leave-out ``HaloObservation`` keeping ``N_pad`` static for zero recompilation.

        ``keep_mask`` is a boolean mask of length ``n_valid`` (or ``N_pad``) selecting
        retained cadences (e.g. for ``sigclip`` refits or orbit block-jackknife folds).
        Pixel time-means ``mean_pixel_flux`` and ``norm_pixels`` are re-centred strictly
        over the retained subset while keeping ``(M, N_pad)`` unchanged.
        """
        n_orig = int(self.n_valid)
        mask_np = np.asarray(keep_mask, dtype=bool)
        if mask_np.shape == (n_orig,):
            full_keep = np.zeros(self.N_pad, dtype=bool)
            full_keep[:n_orig] = mask_np & np.asarray(self.valid_cadence_mask[:n_orig])
        elif mask_np.shape == (self.N_pad,):
            full_keep = mask_np & np.asarray(self.valid_cadence_mask)
        else:
            raise ValueError(
                f"keep_mask shape {mask_np.shape} must match ({n_orig},) or ({self.N_pad},)."
            )

        idx_keep = np.flatnonzero(full_keep)
        if idx_keep.size < 2:
            raise ValueError("At least 2 active cadences are required in with_cadence_subset.")

        raw_sub = (
            np.asarray(self.mean_pixel_flux)[:, np.newaxis]
            * np.asarray(self.norm_pixels)[:, idx_keep]
        )
        times_sub = np.asarray(self.times_padded)[idx_keep]

        old_scale = np.where(np.abs(np.asarray(self.mean_pixel_flux)) > 1e-30, np.asarray(self.mean_pixel_flux), 1.0)
        pixel_var_est = np.asarray(self.mean_frac_var) * (old_scale**2)

        lam_phot_arg: float | None = 0.0 if float(self.lambda_phot) == 0.0 else None
        lam_z_arg: float | None = float(self.lambda_z)

        return HaloObservation.from_arrays(
            raw_sub,
            times_sub,
            pixel_var=pixel_var_est,
            lag=self.lag,
            legacy_tv=legacy_tv,
            lambda_phot=lam_phot_arg,
            lambda_z=lam_z_arg,
            n_pad=self.N_pad,
            gap_tol=self.gap_tol,
        )
