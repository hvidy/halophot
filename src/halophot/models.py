"""Zodiax v0.5.0 PyTree models for halo photometry optimization.

``HaloWeights(zdx.Base)`` isolates ``z_ilr: jax.Array`` as its sole differentiable
array leaf so ``eqx.filter_value_and_grad`` and ``optimistix`` differentiate
strictly with respect to the unconstrained ILR coordinates ``z in R^{M-1}``.
``HaloObservation(zdx.Base)`` holds static calibration arrays padded to bucketed
``N_pad`` cadences for XLA compilation reuse across ``sigclip`` refits, orbit
block-jackknife folds, and multi-sector surveys.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zodiax as zdx

from ._precision import in_float64
from .simplex import (
    _flux_share_to_weights,
    _ilr_to_simplex,
    _simplex_to_ilr,
    _uniform_weights_ilr,
    _weights_to_ilr,
)

__all__ = [
    "HaloObservation",
    "HaloWeights",
    "bucket_pad_length",
    "compute_effective_constraints",
    "compute_gap_diff_mask",
]

# Baseline photon-noise regularization weight at r -> 0 and Marchenko-Pastur linear
# discrepancy slope bounding white-noise shrinkage alpha* = 2 * sqrt(2 / pi) * r / c(r)
# below alpha_tol = c_0 / 2 = 0.05 for all r = (M - 1) / n_eff > 0.
_PHOT_BASE_COEFF: float = 0.1
_PHOT_ALPHA_TOL: float = 0.05
_PHOT_MP_SLOPE: float = 2.0 * math.sqrt(2.0 / math.pi) / _PHOT_ALPHA_TOL

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


def _compute_gap_diff_mask(
    times_padded: jax.Array,
    valid_cadence_mask: jax.Array,
    lag: int,
    dt_nom: jax.Array | float,
    *,
    gap_tol: float = 0.25,
) -> jax.Array:
    """Undecorated lag-``lag`` gap-valid mask kernel."""
    n_pad = times_padded.shape[0]
    if lag < 1 or lag >= n_pad:
        return jnp.zeros(n_pad, dtype=jnp.bool_)

    dt_pair = times_padded[lag:] - times_padded[:-lag]
    dt_nom_arr = jnp.asarray(dt_nom, dtype=times_padded.dtype)
    expected_dt = float(lag) * dt_nom_arr
    tol_ok = (
        jnp.ones_like(dt_pair, dtype=jnp.bool_)
        if math.isinf(float(gap_tol))
        else (jnp.abs(dt_pair - expected_dt) <= float(gap_tol) * dt_nom_arr)
    )
    pair_ok = valid_cadence_mask[:-lag] & valid_cadence_mask[lag:] & tol_ok
    tail_false = jnp.zeros(lag, dtype=jnp.bool_)
    return jnp.concatenate([pair_ok, tail_false], axis=0)


@in_float64
def compute_gap_diff_mask(
    times_padded: jax.Array | np.ndarray,
    valid_cadence_mask: jax.Array | np.ndarray,
    lag: int,
    dt_nom: jax.Array | float,
    *,
    gap_tol: float = 0.25,
) -> jax.Array:
    """Construct the lag-``lag`` gap-valid index mask ``V_lag`` of shape ``(N_pad,)``.

    Entry ``k`` (for ``0 <= k < N_pad - lag``) is ``True`` iff both cadences
    ``k`` and ``k + lag`` are active and
    ``|(t_{k+lag} - t_k) - lag * dt_nom| <= gap_tol * dt_nom``, ensuring
    finite differences never bridge missing cadences or spacecraft gaps at any lag.
    """
    return _compute_gap_diff_mask(
        jnp.asarray(times_padded, dtype=jnp.float64),
        jnp.asarray(valid_cadence_mask, dtype=jnp.bool_),
        lag,
        dt_nom,
        gap_tol=gap_tol,
    )


def compute_effective_constraints(
    valid_diff_mask: np.ndarray | jax.Array,
    *,
    n_valid: int,
    n_pad: int,
    lag: int = 1,
    objective: str = "tv",
    tau: float | None = None,
    times_padded: np.ndarray | jax.Array | None = None,
    valid_cadence_mask: np.ndarray | jax.Array | None = None,
    dt_nom: float | None = None,
    gap_tol: float = 0.25,
) -> int:
    """Return the active scalar constraint count ``n_eff`` for ``objective``."""
    vdm = np.asarray(valid_diff_mask, dtype=bool)
    if objective == "owl":
        return max(int(n_valid) - 1, 0)
    if objective == "tv_o2":
        if 2 * lag >= n_pad:
            return 0
        return int(np.sum(vdm[: n_pad - 2 * lag] & vdm[lag : n_pad - lag]))
    if (
        objective == "tv_exp"
        and times_padded is not None
        and valid_cadence_mask is not None
        and dt_nom is not None
    ):
        tau_val = float(lag) if tau is None else float(tau)
        scan_len = min(max(int(math.ceil(4.0 * max(tau_val, 1e-6))), 1), max(n_pad - 1, 1))
        t_np = np.asarray(times_padded, dtype=np.float64)
        vc_np = np.asarray(valid_cadence_mask, dtype=bool)
        dt_val = float(dt_nom)
        best_count = 0
        for ell in range(1, min(scan_len, max(int(n_valid) - 1, 0)) + 1):
            dt_p = t_np[ell:] - t_np[:-ell]
            exp_dt = float(ell) * dt_val
            tol_ok = (
                np.ones_like(dt_p, dtype=bool)
                if math.isinf(float(gap_tol))
                else (np.abs(dt_p - exp_dt) <= float(gap_tol) * dt_val)
            )
            cnt = int(np.sum(vc_np[:-ell] & vc_np[ell:] & tol_ok))
            if cnt > best_count:
                best_count = cnt
        return best_count
    if lag >= n_pad:
        return 0
    return int(np.sum(vdm[: n_pad - lag]))


def _objective_noise_calibration(objective: str) -> tuple[float, float]:
    """Return ``(noise_mult, obj_slope)`` matching the operator variance and L_p white-noise slope."""
    noise_mult = 1.0 if objective == "owl" else (6.0 if objective == "tv_o2" else 2.0)
    if objective == "l2v":
        obj_slope = math.sqrt(2.0 * math.pi)
    elif objective == "l3v":
        obj_slope = 6.0
    elif objective == "owl":
        obj_slope = math.sqrt(math.pi / 2.0)
    else:
        obj_slope = 1.0
    return noise_mult, obj_slope


def _safe_minvar_frac_var(
    mean_p_np: np.ndarray,
    frac_var: np.ndarray,
) -> np.ndarray:
    """Apply a gain-covariant sub-Poisson floor to ``frac_var`` before harmonic aggregation.

    Computes ``eta_i = frac_var_i * P_bar_i`` (proportional to ``sigma_i^2 / P_bar_i``),
    which scales linearly with flux gain ``a`` under ``(P -> a P, var -> a^2 var)``, so
    ``0.1 * median(eta[eta > 0]) / P_bar_i`` is strictly homogeneous of degree 0 under
    arbitrary unit-gain rescaling while preventing a single float32-cancelled pixel from
    collapsing the harmonic floor ``1 / sum(1 / frac_var)``.
    """
    if (
        frac_var.size > 1
        and np.all(np.isfinite(mean_p_np))
        and np.all(mean_p_np > 0.0)
        and np.all(np.isfinite(frac_var))
        and np.any(frac_var > 0.0)
    ):
        eta = frac_var * mean_p_np
        pos_eta = eta[eta > 0.0]
        if pos_eta.size > 0:
            med_eta = float(np.median(pos_eta))
            return np.maximum(frac_var, 0.1 * med_eta / mean_p_np)
    return frac_var


def _minvar_aperture_floor_and_centre(
    mean_p_j: jax.Array,
    frac_var: np.ndarray,
) -> tuple[float, jax.Array]:
    """Compute harmonic simplex minimum-variance floor and matching ILR prior centre.

    When all pixels have positive finite fractional variance ``frac_var_i > 0``, the
    exact simplex minimum of ``sum_i theta_i^2 * safe_fv_i`` is attained at
    ``theta_mv_i = (1 / safe_fv_i) / sum_j (1 / safe_fv_j)`` with value
    ``ap_var_floor = 1 / sum_j (1 / safe_fv_j)`` (with ``safe_fv`` guarded by
    :func:`_safe_minvar_frac_var`). This coincides identically with uniform physical
    weights ``w_i = 1 / M`` on Poisson-limited and homoscedastic stamps, while remaining
    invariant to both high-variance smear pixels and sub-Poisson cancelled outliers.
    """
    mean_p_np = np.asarray(mean_p_j, dtype=np.float64)
    safe_fv = _safe_minvar_frac_var(mean_p_np, frac_var)
    if safe_fv.size >= 1 and np.all(np.isfinite(safe_fv)) and np.all(safe_fv > 0.0):
        inv_fv = 1.0 / safe_fv
        inv_sum = float(np.sum(inv_fv))
        ap_var_floor = 1.0 / inv_sum
        theta_mv = inv_fv / inv_sum
        if safe_fv.size == 1:
            return ap_var_floor, jnp.zeros(0, dtype=jnp.float64)
        interior = np.maximum(theta_mv, 1e-15)
        interior = interior / np.sum(interior)
        z_centre = _simplex_to_ilr(jnp.asarray(interior, dtype=jnp.float64))
        return ap_var_floor, z_centre

    z_unif = _uniform_weights_ilr(mean_p_j, legacy_tv=False)
    theta_init_np = np.asarray(_ilr_to_simplex(z_unif), dtype=np.float64)
    ap_var_floor = float(np.sum(np.square(theta_init_np) * frac_var))
    return ap_var_floor, z_unif


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
        z0 = _uniform_weights_ilr(
            jnp.asarray(mean_pixel_flux, dtype=jnp.float64),
            legacy_tv=legacy_tv,
        )
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
        z0 = _weights_to_ilr(
            jnp.asarray(w, dtype=jnp.float64),
            jnp.asarray(mean_pixel_flux, dtype=jnp.float64),
            legacy_tv=legacy_tv,
        )
        return cls(z_ilr=z0, legacy_tv=bool(legacy_tv))

    def _flux_shares_raw(self) -> jax.Array:
        return _ilr_to_simplex(self.z_ilr)

    @in_float64
    def flux_shares(self) -> jax.Array:
        """Return simplex coordinates ``theta(z_ilr) in Delta^{M-1}``."""
        return self._flux_shares_raw()

    def _physical_weights_raw(self, mean_pixel_flux: jax.Array) -> jax.Array:
        theta = _ilr_to_simplex(self.z_ilr)
        if self.legacy_tv:
            return theta
        return _flux_share_to_weights(theta, mean_pixel_flux)

    @in_float64
    def physical_weights(self, mean_pixel_flux: jax.Array | np.ndarray) -> jax.Array:
        """Return physical aperture weights ``w in Delta^{M-1}``."""
        return self._physical_weights_raw(jnp.asarray(mean_pixel_flux, dtype=jnp.float64))

    def _evaluate_lightcurve_raw(self, obs: HaloObservation) -> jax.Array:
        if not self.legacy_tv:
            theta = _ilr_to_simplex(self.z_ilr)
            lc_scaled = jnp.dot(theta, obs.norm_pixels)
            mean_ratio = jnp.dot(theta, obs.norm_mean_ratio)
            safe_ratio = jnp.where(
                jnp.abs(mean_ratio) > 1e-30,
                mean_ratio,
                jnp.ones_like(mean_ratio),
            )
            return lc_scaled / safe_ratio

        w = _ilr_to_simplex(self.z_ilr)
        safe_scale = jnp.where(jnp.abs(obs.mean_pixel_flux) > 1e-30, obs.mean_pixel_flux, 1.0)
        weighted_p = jnp.dot(w * safe_scale, obs.norm_pixels)
        mean_f = jnp.dot(w, obs.mean_pixel_flux)
        safe_denom = jnp.where(
            jnp.abs(mean_f) > 1e-30,
            mean_f,
            jnp.ones_like(mean_f),
        )
        return weighted_p / safe_denom

    @in_float64
    def evaluate_lightcurve(self, obs: HaloObservation) -> jax.Array:
        """Evaluate the unit-mean padded light curve ``f_tilde`` of shape ``(N_pad,)``."""
        return self._evaluate_lightcurve_raw(obs)


class HaloObservation(zdx.Base):
    """Static calibration and cadence-mask PyTree passed to solvers via ``args=obs``."""

    norm_pixels: jax.Array
    mean_pixel_flux: jax.Array
    norm_mean_ratio: jax.Array
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
    n_eff: jax.Array
    N_pad: int = eqx.field(static=True)
    lag: int = eqx.field(static=True, default=1)
    gap_tol: float = eqx.field(static=True, default=0.25)
    objective: str = eqx.field(static=True, default="tv")
    tau: float | None = eqx.field(static=True, default=None)
    legacy_tv: bool = eqx.field(static=True, default=False)
    user_lambda_phot: float | None = eqx.field(static=True, default=None)
    user_lambda_z: float | None = eqx.field(static=True, default=None)

    @in_float64
    def effective_constraint_count(
        self,
        *,
        objective: str | None = None,
        tau: float | None = None,
    ) -> int:
        """Return the active constraint count matching ``objective``."""
        eff_obj = self.objective if objective is None else objective
        eff_tau = self.tau if tau is None else tau
        return compute_effective_constraints(
            self.valid_diff_mask,
            n_valid=int(self.n_valid),
            n_pad=self.N_pad,
            lag=self.lag,
            objective=eff_obj,
            tau=eff_tau,
            times_padded=self.times_padded,
            valid_cadence_mask=self.valid_cadence_mask,
            dt_nom=float(self.dt_nom),
            gap_tol=self.gap_tol,
        )

    @classmethod
    @in_float64
    def from_arrays(
        cls,
        pixels: np.ndarray | jax.Array,
        times: np.ndarray | jax.Array | None = None,
        *,
        pixel_var: np.ndarray | jax.Array | None = None,
        lag: int = 1,
        objective: str = "tv",
        tau: float | None = None,
        legacy_tv: bool = False,
        lambda_phot: float | None = None,
        lambda_z: float | None = None,
        n_pad: int | None = None,
        gap_tol: float = 0.25,
        dt_nom: float | None = None,
    ) -> HaloObservation:
        """Construct a padded ``HaloObservation`` in 64-bit precision.

        Parameters
        ----------
        pixels : array-like, shape ``(M, N)``
            Active halo pixel time series strictly over valid cadences ``G``.
            When ``pixel_var=None`` and ``legacy_tv=False``, ``pixels`` is assumed
            to be in physical ``e-/s`` units for the Poisson shot-noise fallback
            ``mean_frac_var_i = 1 / (max(|P_bar_i|, 1) * dt_sec)``; pass explicit
            ``pixel_var`` whenever ``pixels`` is in rescaled or arbitrary units.
        times : array-like, shape ``(N,)``, optional
            Strictly increasing timestamps in days. Defaults to uniform
            cadences ``0, 1/48, ..., (N - 1)/48`` (30-min cadence).
        pixel_var : array-like, shape ``(M, N)`` or ``(M,)``, optional
            Per-pixel independent variance ``sigma_P^2`` in physical flux^2 units.
            When omitted and ``legacy_tv=False``, defaults to Poisson shot-noise
            scaling ``P_bar_i / dt_sec`` (assuming ``pixels`` in ``e-/s``).
        lag : int, default 1
            Finite-difference lag ``ell >= 1``.
        objective : str, default ``'tv'``
            Target objective name used to count active constraints ``n_eff``
            (1st-order pairs for ``'tv'``/``'l2v'``/``'l3v'``, 2nd-order triples
            for ``'tv_o2'``, multi-lag pairs for ``'tv_exp'``, or ``N - 1``
            centred cadences for ``'owl'``).
        tau : float, optional
            Exponential lag scale for ``objective='tv_exp'``.
        legacy_tv : bool, default False
            When ``True``, sets ``lambda_phot = 0``, ``lambda_z = 0``, and
            ``b_hat = 1`` for exact *K2* / ``halo-tess`` DR1 reproduction.
        lambda_phot : float, optional
            Photon-noise regularization strength. When ``None`` and ``not legacy_tv``,
            auto-calibrated from the harmonic minimum-variance aperture photon-noise
            floor ``ap_var_floor = 1 / sum_i (1 / mean_frac_var_i)`` so pointing
            jitter, common-mode pulsations, and high-variance smear pixels never
            suppress photon-noise regularization.
        lambda_z : float, optional
            Bounded ILR gauge-damping strength around ``z_prior_centre``.
            Defaults to ``1e-3`` when ``not legacy_tv`` and ``0.0`` when
            ``legacy_tv=True``.
        n_pad : int, optional
            Explicit padded time length ``>= N``. Defaults to ``bucket_pad_length(N)``.
        gap_tol : float, default 0.25
            Fixed-cadence tolerance around ``lag * dt_nom``
            (``|dt_pair - lag * dt_nom| <= gap_tol * dt_nom``) for gap-aware differencing.
        dt_nom : float, optional
            Explicit nominal cadence spacing in days. Defaults to ``median(diff(times))``.
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
        if np.any(diffs_t <= 0.0):
            raise ValueError(
                "times must be strictly increasing (all consecutive differences > 0)."
            )
        if dt_nom is None:
            dt_nom_val = float(np.median(diffs_t))
        else:
            dt_nom_val = float(dt_nom)
            if not np.isfinite(dt_nom_val) or dt_nom_val <= 0.0:
                raise ValueError(f"dt_nom must be finite and > 0, got {dt_nom!r}.")

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
        diff_mask_j = _compute_gap_diff_mask(
            times_j,
            valid_cad_j,
            lag_int,
            dt_nom_j,
            gap_tol=gap_tol,
        )

        mean_p_j = jnp.asarray(mean_p, dtype=jnp.float64)

        n_constraints = compute_effective_constraints(
            diff_mask_j,
            n_valid=n_cad,
            n_pad=resolved_pad,
            lag=lag_int,
            objective=objective,
            tau=tau,
            times_padded=times_padded_np,
            valid_cadence_mask=valid_cad,
            dt_nom=dt_nom_val,
            gap_tol=gap_tol,
        )
        n_eff = max(n_constraints, 1)

        if legacy_tv:
            z_centre_j = _uniform_weights_ilr(mean_p_j, legacy_tv=True)
            b_hat_val = 1.0
            lam_phot_val = 0.0
            lam_z_val = 0.0
            user_lam_phot_val: float | None = 0.0
            user_lam_z_val: float | None = 0.0
        else:
            ap_var_floor, z_centre_j = _minvar_aperture_floor_and_centre(mean_p_j, frac_var)
            noise_mult, obj_slope = _objective_noise_calibration(objective)

            # Anchor both the data-term denominator b_hat and lambda_phot strictly to the
            # operator-matched minimum-variance aperture photon-noise floor so pointing
            # jitter, common-mode pulsations, and noisy smear pixels never crush lambda_phot.
            ref_var = max(noise_mult * ap_var_floor, 1e-12)
            b_hat_val = math.sqrt(ref_var)

            dof_ratio = float(m_pix - 1) / float(n_eff) if m_pix > 1 else 0.0
            c_phot = _PHOT_BASE_COEFF + _PHOT_MP_SLOPE * dof_ratio
            if lambda_phot is None:
                phot_ref_var = max(2.0 * ap_var_floor, 1e-12)
                lam_phot_val = obj_slope * c_phot * float(n_eff) / phot_ref_var
                user_lam_phot_val = None
            else:
                lam_phot_val = float(lambda_phot)
                if lam_phot_val < 0.0:
                    raise ValueError(f"lambda_phot must be >= 0, got {lambda_phot!r}.")
                user_lam_phot_val = lam_phot_val

            if lambda_z is None:
                lam_z_val = 1e-3
                user_lam_z_val = None
            else:
                lam_z_val = float(lambda_z)
                if lam_z_val < 0.0:
                    raise ValueError(f"lambda_z must be >= 0, got {lambda_z!r}.")
                user_lam_z_val = lam_z_val

        return cls(
            norm_pixels=jnp.asarray(norm_padded, dtype=jnp.float64),
            mean_pixel_flux=mean_p_j,
            norm_mean_ratio=jnp.ones(m_pix, dtype=jnp.float64),
            mean_frac_var=jnp.asarray(frac_var, dtype=jnp.float64),
            valid_diff_mask=diff_mask_j,
            z_prior_centre=z_centre_j,
            times_padded=times_j,
            valid_cadence_mask=valid_cad_j,
            dt_nom=dt_nom_j,
            b_hat=jnp.asarray(b_hat_val, dtype=jnp.float64),
            lambda_phot=jnp.asarray(lam_phot_val, dtype=jnp.float64),
            lambda_z=jnp.asarray(lam_z_val, dtype=jnp.float64),
            n_valid=jnp.asarray(n_cad, dtype=jnp.int32),
            n_eff=jnp.asarray(n_eff, dtype=jnp.float64),
            N_pad=resolved_pad,
            lag=lag_int,
            gap_tol=float(gap_tol),
            objective=str(objective),
            tau=None if tau is None else float(tau),
            legacy_tv=bool(legacy_tv),
            user_lambda_phot=user_lam_phot_val,
            user_lambda_z=user_lam_z_val,
        )

    @in_float64
    def with_cadence_subset(
        self,
        keep_mask: np.ndarray | jax.Array,
        *,
        legacy_tv: bool | None = None,
        preserve_regularization: bool = True,
        objective: str | None = None,
        tau: float | None = None,
    ) -> HaloObservation:
        """Build a leave-out ``HaloObservation`` keeping ``N_pad`` static for zero recompilation.

        ``keep_mask`` is a boolean mask of length ``n_valid`` (or ``N_pad``) selecting
        retained cadences (e.g. for ``sigclip`` refits or orbit block-jackknife folds).
        ``self.dt_nom`` is always preserved so removing cadences never distorts gap masks.
        Active pixel fluxes are always re-centred on the retained subset means ``mean_sub``;
        if any faint wing pixel dips ``mean_sub_i <= 0`` under ``not legacy_tv``, only that
        pixel's positive Charnes-Cooper chart scale falls back to ``self.mean_pixel_flux[i]``
        while ``norm_mean_ratio = mean_sub / chart_scale`` keeps ``evaluate_lightcurve``
        strictly unit-mean on the subset. When ``preserve_regularization=True`` (used by
        ``orbit_block_jackknife``), ``self.b_hat`` is preserved and auto-calibrated
        ``lambda_phot`` / ``lambda_z`` scale with ``n_eff_sub / n_eff_full``.
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
        n_sub = int(idx_keep.size)
        if n_sub < 2:
            raise ValueError("At least 2 active cadences are required in with_cadence_subset.")

        eff_legacy = self.legacy_tv if legacy_tv is None else bool(legacy_tv)
        eff_obj = self.objective if objective is None else objective
        eff_tau = self.tau if tau is None else tau
        dt_nom_val = float(self.dt_nom)

        mean_orig = np.asarray(self.mean_pixel_flux, dtype=np.float64)
        old_scale = np.where(np.abs(mean_orig) > 1e-30, mean_orig, 1.0)
        raw_sub = old_scale[:, np.newaxis] * np.asarray(self.norm_pixels, dtype=np.float64)[:, idx_keep]
        times_sub = np.asarray(self.times_padded, dtype=np.float64)[idx_keep]
        pixel_var_est = np.asarray(self.mean_frac_var, dtype=np.float64) * (old_scale**2)
        mean_sub = np.mean(raw_sub, axis=1)

        if eff_legacy or np.all(mean_sub > 0.0):
            lam_phot_arg: float | None = 0.0 if eff_legacy else self.user_lambda_phot
            lam_z_arg: float | None = 0.0 if eff_legacy else self.user_lambda_z
            sub_obs = HaloObservation.from_arrays(
                raw_sub,
                times_sub,
                pixel_var=pixel_var_est,
                lag=self.lag,
                objective=eff_obj,
                tau=eff_tau,
                legacy_tv=eff_legacy,
                lambda_phot=lam_phot_arg,
                lambda_z=lam_z_arg,
                n_pad=self.N_pad,
                gap_tol=self.gap_tol,
                dt_nom=dt_nom_val,
            )
            if eff_legacy or not preserve_regularization:
                return sub_obs

            n_eff_full = max(self.effective_constraint_count(objective=eff_obj, tau=eff_tau), 1)
            n_eff_sub = max(int(sub_obs.n_eff), 1)
            count_ratio = float(n_eff_sub) / float(n_eff_full)
            new_lam_phot = (
                sub_obs.lambda_phot
                if self.user_lambda_phot is not None
                else jnp.asarray(float(self.lambda_phot) * count_ratio, dtype=jnp.float64)
            )
            new_lam_z = (
                sub_obs.lambda_z
                if self.user_lambda_z is not None
                else jnp.asarray(float(self.lambda_z) * count_ratio, dtype=jnp.float64)
            )
            new_b_hat = sub_obs.b_hat if self.user_lambda_phot is not None else self.b_hat
            return eqx.tree_at(
                lambda o: (o.lambda_phot, o.lambda_z, o.b_hat),
                sub_obs,
                (new_lam_phot, new_lam_z, new_b_hat),
            )

        # At least one faint wing pixel dipped mean_sub <= 0 under not legacy_tv:
        # keep strictly positive chart_scale_i = mean_sub_i when > 0 else mean_orig_i (> 0),
        # while norm_mean_ratio = mean_sub / chart_scale guarantees exact unit-mean light curves.
        if not np.all(mean_orig > 0.0):
            raise ValueError(
                "Charnes-Cooper coordinates require strictly positive mean pixel flux "
                "on all active halo pixels (mean_pixel_flux > 0)."
            )
        m_pix = mean_sub.shape[0]
        chart_scale = np.where(mean_sub > 0.0, mean_sub, mean_orig)
        norm_sub = raw_sub / chart_scale[:, np.newaxis]
        norm_ratio = mean_sub / chart_scale
        frac_var_sub = np.maximum(pixel_var_est / np.maximum(chart_scale**2, 1e-30), 0.0)

        norm_padded = np.ones((m_pix, self.N_pad), dtype=np.float64)
        norm_padded[:, :n_sub] = norm_sub
        valid_cad = np.zeros(self.N_pad, dtype=bool)
        valid_cad[:n_sub] = True
        times_padded_np = np.empty(self.N_pad, dtype=np.float64)
        times_padded_np[:n_sub] = times_sub
        if self.N_pad > n_sub:
            pad_offsets = np.arange(1, self.N_pad - n_sub + 1, dtype=np.float64)
            times_padded_np[n_sub:] = times_sub[-1] + 1e6 * dt_nom_val * pad_offsets

        times_j = jnp.asarray(times_padded_np, dtype=jnp.float64)
        valid_cad_j = jnp.asarray(valid_cad, dtype=jnp.bool_)
        dt_nom_j = jnp.asarray(dt_nom_val, dtype=jnp.float64)
        diff_mask_j = _compute_gap_diff_mask(
            times_j,
            valid_cad_j,
            self.lag,
            dt_nom_j,
            gap_tol=self.gap_tol,
        )
        chart_scale_j = jnp.asarray(chart_scale, dtype=jnp.float64)
        ap_var_floor_sub, z_centre_sub = _minvar_aperture_floor_and_centre(
            chart_scale_j, frac_var_sub
        )
        n_eff_sub = max(
            compute_effective_constraints(
                diff_mask_j,
                n_valid=n_sub,
                n_pad=self.N_pad,
                lag=self.lag,
                objective=eff_obj,
                tau=eff_tau,
                times_padded=times_padded_np,
                valid_cadence_mask=valid_cad,
                dt_nom=dt_nom_val,
                gap_tol=self.gap_tol,
            ),
            1,
        )

        if preserve_regularization:
            n_eff_full = max(self.effective_constraint_count(objective=eff_obj, tau=eff_tau), 1)
            count_ratio = float(n_eff_sub) / float(n_eff_full)
            new_lam_phot = (
                jnp.asarray(float(self.user_lambda_phot), dtype=jnp.float64)
                if self.user_lambda_phot is not None
                else jnp.asarray(float(self.lambda_phot) * count_ratio, dtype=jnp.float64)
            )
            new_lam_z = (
                jnp.asarray(float(self.user_lambda_z), dtype=jnp.float64)
                if self.user_lambda_z is not None
                else jnp.asarray(float(self.lambda_z) * count_ratio, dtype=jnp.float64)
            )
            new_b_hat = self.b_hat
        else:
            noise_mult, obj_slope = _objective_noise_calibration(eff_obj)
            ref_var = max(noise_mult * ap_var_floor_sub, 1e-12)
            new_b_hat = jnp.asarray(math.sqrt(ref_var), dtype=jnp.float64)
            dof_ratio = float(m_pix - 1) / float(n_eff_sub) if m_pix > 1 else 0.0
            c_phot = _PHOT_BASE_COEFF + _PHOT_MP_SLOPE * dof_ratio
            new_lam_phot = (
                jnp.asarray(float(self.user_lambda_phot), dtype=jnp.float64)
                if self.user_lambda_phot is not None
                else jnp.asarray(
                    obj_slope * c_phot * float(n_eff_sub) / max(2.0 * ap_var_floor_sub, 1e-12),
                    dtype=jnp.float64,
                )
            )
            new_lam_z = (
                jnp.asarray(float(self.user_lambda_z), dtype=jnp.float64)
                if self.user_lambda_z is not None
                else jnp.asarray(1e-3, dtype=jnp.float64)
            )

        return HaloObservation(
            norm_pixels=jnp.asarray(norm_padded, dtype=jnp.float64),
            mean_pixel_flux=chart_scale_j,
            norm_mean_ratio=jnp.asarray(norm_ratio, dtype=jnp.float64),
            mean_frac_var=jnp.asarray(frac_var_sub, dtype=jnp.float64),
            valid_diff_mask=diff_mask_j,
            z_prior_centre=z_centre_sub,
            times_padded=times_j,
            valid_cadence_mask=valid_cad_j,
            dt_nom=dt_nom_j,
            b_hat=new_b_hat,
            lambda_phot=new_lam_phot,
            lambda_z=new_lam_z,
            n_valid=jnp.asarray(n_sub, dtype=jnp.int32),
            n_eff=jnp.asarray(n_eff_sub, dtype=jnp.float64),
            N_pad=self.N_pad,
            lag=self.lag,
            gap_tol=self.gap_tol,
            objective=str(eff_obj),
            tau=None if eff_tau is None else float(eff_tau),
            legacy_tv=False,
            user_lambda_phot=self.user_lambda_phot,
            user_lambda_z=self.user_lambda_z,
        )
