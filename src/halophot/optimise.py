"""Zodiax / Optax halo weight solvers, narrow local spike clipping, and block jackknife.

Provides:
* Cached ``eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))`` evaluation on
  :class:`~halophot.models.HaloWeights` PyTrees (scoped under ``@in_float64``).
* Tier-1 two-tier preconditioned ``optax.lbfgs`` + zoom line search solver
  (Fisher-preconditioned for ``M - 1 <= 256``, O(M) diagonal log-simplex
  curvature scaled for ``M - 1 > 256`` or ``precondition=False``) and Tier-2
  ``zodiax.optimisation`` (``optax.adam`` + ``safe_eigen_projection``), both
  normalized internally by ``max(n_eff, 1.0)`` so loss and gradients remain O(1).
* Gap-aware narrow local-median / Theil-Sen cosmic-ray and thruster spike
  detector that excludes outlier cadences strictly during weight optimization
  without overwriting finite astrophysical cadences in returned light curves.
* Warm-started orbit-gap-aware ``K``-block jackknife uncertainty estimation
  with Busing-Meijer-van der Leeden delete-``m_k`` weighting and static ``N_pad``
  XLA reuse.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import zodiax as zdx
from astropy.stats import mad_std, sigma_clip

from ._precision import in_float64
from .models import HaloObservation, HaloWeights, _safe_minvar_frac_var
from .objectives import (
    AVAILABLE_OBJECTIVES,
    _halo_objective,
    _l2v_gap_masked,
    _l3v_gap_masked,
    _owl_masked,
    _photon_noise_penalty,
    _tv_charbonnier_o1,
    _tv_charbonnier_o2,
    _tv_exp_scan,
)
from .simplex import _ilr_to_logits, _ilr_to_simplex, _logits_to_ilr

__all__ = [
    "HaloFitResult",
    "build_value_and_grad",
    "cadence_scaled_savgol_window",
    "cadence_scaled_sigclip_mask",
    "optimise_halo",
    "orbit_block_jackknife",
    "safe_eigen_projection",
]


@dataclass(frozen=True)
class HaloFitResult:
    """Result of ``optimise_halo`` on active halo pixels ``Omega_halo``.

    ``converged`` indicates stationarity or 2-step stabilized weights and objective
    within ``(rtol, atol)`` for ``solver='lbfgs'``, or finite objective with
    preconditioned max-norm gradient within ``atol + rtol * max(|loss / n_eff|, 1.0)``
    for fixed-schedule first-order solvers (``'zodiax'`` / ``'adam'``).
    """

    weights_tree: HaloWeights
    physical_weights: np.ndarray
    flux_shares: np.ndarray
    norm_flux: np.ndarray
    raw_flux: np.ndarray
    inlier_mask: np.ndarray
    loss_init: float
    loss_final: float
    weight_std: np.ndarray | None = None
    jackknife_weights: np.ndarray | None = None
    converged: bool = True
    n_steps: int = 0


@functools.lru_cache(maxsize=32)
def build_value_and_grad(
    objective: str = "tv",
    epsilon: float = 1e-5,
    tau: float | None = None,
):
    """Return a cached ``@in_float64``-scoped ``eqx.filter_jit(eqx.filter_value_and_grad)`` callable."""
    if objective not in AVAILABLE_OBJECTIVES:
        valid = ", ".join(AVAILABLE_OBJECTIVES)
        raise ValueError(f"Unknown objective {objective!r}; expected one of: {valid}.")

    def _loss(weights: HaloWeights, obs: HaloObservation) -> jax.Array:
        return _halo_objective(weights, obs, objective=objective, epsilon=epsilon, tau=tau)

    jitted_vg = eqx.filter_jit(eqx.filter_value_and_grad(_loss))

    @in_float64
    def _scoped_vg(weights: HaloWeights, obs: HaloObservation):
        return jitted_vg(weights, obs)

    return _scoped_vg


@in_float64
def safe_eigen_projection(
    fmat: jax.Array | np.ndarray | None = None,
    cov: jax.Array | np.ndarray | None = None,
    *,
    eps: float = 1e-8,
    rcond: float = 1e-6,
) -> jax.Array:
    """Compute a rank-deficiency-guarded Hessian / covariance preconditioning matrix.

    Wraps ``zodiax.optimisation.decompose`` while symmetrizing the input matrix and
    clamping only non-positive diagonal entries (leaving all positive curvature scales
    intact) and small/negative normalized eigenvalues strictly above zero so flat or
    rank-deficient directions never emit ``0/0`` or ``1/sqrt(0)`` ``NaN``s.
    """
    if (fmat is None) == (cov is None):
        raise ValueError("Specify exactly one of fmat or cov in safe_eigen_projection.")
    if eps <= 0.0 or rcond < 0.0:
        raise ValueError("eps must be > 0 and rcond must be >= 0.")

    raw = jnp.asarray(fmat if fmat is not None else cov, dtype=jnp.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Expected a 2D square matrix, got shape {raw.shape}.")
    dim = raw.shape[0]
    if dim == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    mat = 0.5 * (raw + raw.T)
    diag = jnp.diag(mat)
    tiny = jnp.finfo(raw.dtype).tiny
    pos_mask = jnp.isfinite(diag) & (diag > tiny)
    has_pos = jnp.any(pos_mask)
    if fmat is not None:
        max_pos = jnp.max(jnp.where(pos_mask, diag, 0.0))
        fallback_diag = jnp.where(has_pos, max_pos, 1.0)
        safe_diag = jnp.where(pos_mask, diag, fallback_diag)
    else:
        min_pos = jnp.min(jnp.where(pos_mask, diag, jnp.inf))
        cov_floor = jnp.where(has_pos, jnp.maximum(eps * min_pos, tiny), eps)
        safe_diag = jnp.where(pos_mask, diag, cov_floor)

    inv_sqrt_diag = 1.0 / jnp.sqrt(safe_diag)
    norm_mat = mat * jnp.outer(inv_sqrt_diag, inv_sqrt_diag)
    norm_mat = 0.5 * (norm_mat + norm_mat.T)

    vals, vecs = zdx.optimisation.decompose(norm_mat, normalise=False)
    top_val = jnp.maximum(vals[0], eps)
    floor = jnp.maximum(rcond * top_val, eps)
    safe_vals = jnp.maximum(vals, floor)

    if fmat is not None:
        diag_scale = inv_sqrt_diag
        eig_scale = 1.0 / jnp.sqrt(safe_vals)
    else:
        diag_scale = jnp.sqrt(safe_diag)
        eig_scale = jnp.sqrt(safe_vals)

    return (diag_scale[:, None] * vecs.T) * eig_scale[None, :]


@functools.lru_cache(maxsize=16)
def _get_fisher_fn(
    objective: str,
    tau: float | None,
    epsilon: float = 1e-5,
):
    """Build and cache a JIT-compiled Gauss-Newton Fisher curvature evaluator wrt ``z_ilr``.

    Evaluates the convex simplex Hessian ``H_theta`` of the ``1 / max(n_eff, 1)``-normalized
    objective at a noise-matched smoothing scale ``eps_gn`` (``max(b_hat, eps_eff)`` in
    regularized mode, or the active difference scale at ``theta0`` floored at ``epsilon``
    under ``legacy_tv=True``) and pulls it back through ``J_theta = d(theta)/d(z_ilr)``.
    """

    @eqx.filter_jit
    def _fisher(w0: HaloWeights, obs: HaloObservation) -> jax.Array:
        z0 = w0.z_ilr
        theta0 = _ilr_to_simplex(z0)
        jac_theta = jax.jacrev(_ilr_to_simplex)(z0)
        inv_n = 1.0 / jnp.maximum(obs.n_eff, 1.0)
        tiny = jnp.finfo(z0.dtype).tiny
        b_scale0 = jnp.maximum(obs.b_hat, tiny)
        eps_req = jnp.asarray(epsilon, dtype=z0.dtype)
        if not w0.legacy_tv:
            eps_eff = jnp.maximum(jnp.minimum(eps_req, 0.1 * b_scale0), tiny)
            eps_gn = jnp.maximum(b_scale0, eps_eff)
        else:
            f0 = w0._evaluate_lightcurve_raw(obs)
            lag = obs.lag
            if objective == "owl":
                mask_c = obs.valid_cadence_mask.astype(z0.dtype)
                cnt_c = jnp.maximum(jnp.sum(mask_c), 1.0)
                mu_c = jnp.sum(mask_c * f0) / cnt_c
                var_c = (
                    jnp.sum(jnp.where(obs.valid_cadence_mask, jnp.square(f0 - mu_c), 0.0)) / cnt_c
                )
                scale0 = jnp.sqrt(jnp.maximum(var_c, 0.0))
            elif objective == "tv_o2" and 2 * lag < obs.N_pad:
                m2 = obs.valid_diff_mask[: -2 * lag] & obs.valid_diff_mask[lag:-lag]
                d2 = jnp.where(
                    m2, jnp.abs(2.0 * f0[lag:-lag] - f0[2 * lag :] - f0[: -2 * lag]), 0.0
                )
                scale0 = jnp.sum(d2) / jnp.maximum(jnp.sum(m2.astype(z0.dtype)), 1.0)
            elif lag < obs.N_pad:
                m1 = obs.valid_diff_mask[:-lag]
                d1 = jnp.where(m1, jnp.abs(f0[lag:] - f0[:-lag]), 0.0)
                scale0 = jnp.sum(d1) / jnp.maximum(jnp.sum(m1.astype(z0.dtype)), 1.0)
            else:
                scale0 = eps_req
            eps_gn = jnp.maximum(scale0, eps_req)

        def _simplex_loss(theta: jax.Array) -> jax.Array:
            if not w0.legacy_tv:
                lc_scaled = jnp.dot(theta, obs.norm_pixels)
                mean_ratio = jnp.dot(theta, obs.norm_mean_ratio)
                safe_ratio = jnp.where(
                    jnp.abs(mean_ratio) > 1e-30,
                    mean_ratio,
                    jnp.ones_like(mean_ratio),
                )
                f_tilde = lc_scaled / safe_ratio
            else:
                safe_scale = jnp.where(
                    jnp.abs(obs.mean_pixel_flux) > 1e-30,
                    obs.mean_pixel_flux,
                    1.0,
                )
                weighted_p = jnp.dot(theta * safe_scale, obs.norm_pixels)
                mean_f = jnp.dot(theta, obs.mean_pixel_flux)
                safe_denom = jnp.where(
                    jnp.abs(mean_f) > 1e-30,
                    mean_f,
                    jnp.ones_like(mean_f),
                )
                f_tilde = weighted_p / safe_denom

            b_scale = jnp.maximum(obs.b_hat, jnp.finfo(f_tilde.dtype).tiny)
            if objective == "tv":
                mean_val, count = _tv_charbonnier_o1(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "tv_o2":
                mean_val, count = _tv_charbonnier_o2(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "tv_exp":
                mean_val, count = _tv_exp_scan(f_tilde, obs, tau=tau, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "l2v":
                mean_val, count = _l2v_gap_masked(f_tilde, obs)
                data_term = (count / jnp.square(b_scale)) * mean_val
            elif objective == "l3v":
                mean_val, count = _l3v_gap_masked(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / (b_scale * jnp.square(b_scale))) * mean_val
            else:
                std_val, count = _owl_masked(f_tilde, obs, epsilon=eps_gn)
                data_term = std_val if w0.legacy_tv else (count / b_scale) * std_val

            total = data_term if w0.legacy_tv else (data_term + _photon_noise_penalty(theta, obs))
            return total * inv_n

        h_theta = jax.hessian(_simplex_loss)(theta0)
        damping = jnp.maximum(obs.lambda_z * inv_n, 1e-6)
        eye = jnp.eye(z0.shape[0], dtype=z0.dtype)
        return jac_theta.T @ h_theta @ jac_theta + damping * eye

    return _fisher


def _compute_diag_logit_scale(
    w0: HaloWeights,
    obs: HaloObservation,
    objective: str,
) -> jax.Array:
    """Compute O(M) diagonal log-simplex curvature scaling ``d in [1e-3, 1e3]^M``."""
    theta0 = _ilr_to_simplex(w0.z_ilr)
    dtype = theta0.dtype
    n_scale = jnp.maximum(obs.n_eff, 1.0)
    f0 = w0._evaluate_lightcurve_raw(obs)

    if not w0.legacy_tv:
        mean_ratio0 = jnp.dot(theta0, obs.norm_mean_ratio)
        safe_ratio0 = jnp.where(jnp.abs(mean_ratio0) > 1e-30, mean_ratio0, 1.0)
        p_eff = obs.norm_pixels / safe_ratio0
    else:
        safe_scale = jnp.where(jnp.abs(obs.mean_pixel_flux) > 1e-30, obs.mean_pixel_flux, 1.0)
        mean_f0 = jnp.dot(theta0, obs.mean_pixel_flux)
        safe_denom0 = jnp.where(jnp.abs(mean_f0) > 1e-30, mean_f0, 1.0)
        p_eff = (safe_scale[:, None] * obs.norm_pixels) / safe_denom0

    lag = obs.lag
    if objective == "owl":
        mask = obs.valid_cadence_mask
        cnt = jnp.maximum(jnp.sum(mask.astype(dtype)), 1.0)
        mu_f0 = jnp.sum(jnp.where(mask, f0, 0.0)) / cnt
        df0 = jnp.where(mask, f0 - mu_f0, 0.0)
        mu_p = jnp.sum(jnp.where(mask[None, :], p_eff, 0.0), axis=1, keepdims=True) / cnt
        dp_c = jnp.where(mask[None, :], (p_eff - mu_p) - df0[None, :], 0.0)
    elif objective == "tv_o2" and 2 * lag < obs.N_pad:
        mask = obs.valid_diff_mask[: -2 * lag] & obs.valid_diff_mask[lag:-lag]
        cnt = jnp.maximum(jnp.sum(mask.astype(dtype)), 1.0)
        df0 = jnp.where(mask, 2.0 * f0[lag:-lag] - f0[2 * lag :] - f0[: -2 * lag], 0.0)
        dp_raw = 2.0 * p_eff[:, lag:-lag] - p_eff[:, 2 * lag :] - p_eff[:, : -2 * lag]
        dp_c = jnp.where(mask[None, :], dp_raw - df0[None, :], 0.0)
    else:
        if lag < obs.N_pad:
            mask_lag = obs.valid_diff_mask[:-lag]
            has_lag = jnp.any(mask_lag)
        else:
            mask_lag = jnp.zeros(0, dtype=jnp.bool_)
            has_lag = jnp.asarray(False)

        # Fallback to lag-1 active cadence pairs when obs.lag exceeds segment lengths (e.g. tv_exp).
        mask_1 = obs.valid_cadence_mask[:-1] & obs.valid_cadence_mask[1:]
        cnt_1 = jnp.maximum(jnp.sum(mask_1.astype(dtype)), 1.0)
        df0_1 = jnp.where(mask_1, f0[1:] - f0[:-1], 0.0)
        dp_c_1 = jnp.where(mask_1[None, :], (p_eff[:, 1:] - p_eff[:, :-1]) - df0_1[None, :], 0.0)
        mean_dp_sq_1 = jnp.sum(jnp.square(dp_c_1), axis=1) / cnt_1
        rms_df0_1 = jnp.sqrt(jnp.sum(jnp.square(df0_1)) / cnt_1)

        if lag < obs.N_pad:
            cnt_l = jnp.maximum(jnp.sum(mask_lag.astype(dtype)), 1.0)
            df0_l = jnp.where(mask_lag, f0[lag:] - f0[:-lag], 0.0)
            dp_c_l = jnp.where(
                mask_lag[None, :], (p_eff[:, lag:] - p_eff[:, :-lag]) - df0_l[None, :], 0.0
            )
            mean_dp_sq_l = jnp.sum(jnp.square(dp_c_l), axis=1) / cnt_l
            rms_df0_l = jnp.sqrt(jnp.sum(jnp.square(df0_l)) / cnt_l)
            mean_dp_sq = jnp.where(has_lag, mean_dp_sq_l, mean_dp_sq_1)
            rms_df0 = jnp.where(has_lag, rms_df0_l, rms_df0_1)
        else:
            mean_dp_sq = mean_dp_sq_1
            rms_df0 = rms_df0_1

    if objective in ("owl", "tv_o2"):
        mean_dp_sq = jnp.sum(jnp.square(dp_c), axis=1) / cnt
        rms_df0 = jnp.sqrt(jnp.sum(jnp.square(df0)) / cnt)

    rms_safe = jnp.maximum(rms_df0, 1e-12)
    if w0.legacy_tv:
        if objective == "l2v":
            curv_data = jnp.square(theta0) * mean_dp_sq
        elif objective == "l3v":
            curv_data = jnp.square(theta0) * mean_dp_sq * rms_safe
        else:
            curv_data = jnp.square(theta0) * mean_dp_sq / rms_safe
        curv_phot = jnp.zeros_like(theta0)
        curv_z = jnp.asarray(1e-6, dtype=dtype)
    else:
        b_scale = jnp.maximum(obs.b_hat, 1e-12)
        if objective == "l2v":
            curv_data = jnp.square(theta0) * mean_dp_sq / jnp.square(b_scale)
        elif objective == "l3v":
            curv_data = jnp.square(theta0) * mean_dp_sq * rms_safe / (b_scale * jnp.square(b_scale))
        else:
            curv_data = jnp.square(theta0) * mean_dp_sq / (b_scale * rms_safe)
        curv_phot = (obs.lambda_phot / n_scale) * jnp.square(theta0) * obs.mean_frac_var
        curv_z = jnp.maximum(obs.lambda_z / n_scale, 1e-6)

    d_raw = 1.0 / jnp.sqrt(jnp.maximum(curv_data + curv_phot + curv_z, 1e-30))
    d_med = jnp.maximum(jnp.median(d_raw), 1e-30)
    return jnp.clip(d_raw / d_med, 1e-3, 1e3)


def _run_optax_lbfgs_loop(
    val_fn,
    weights_from_xi,
    xi0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run JIT-compatible ``optax.lbfgs`` + zoom linesearch in ``jax.lax.while_loop``."""
    opt = optax.lbfgs(
        memory_size=20,
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20, verbose=False),
    )
    vg_fn = jax.value_and_grad(val_fn)
    v0, g0 = vg_fn(xi0)
    opt_state0 = opt.init(xi0)
    w0_phys = weights_from_xi(xi0)

    f_scale0 = jnp.maximum(jnp.abs(v0), 1.0)
    finite0 = jnp.isfinite(v0) & jnp.all(jnp.isfinite(g0))
    conv0 = finite0 & (jnp.max(jnp.abs(g0)) <= atol + rtol * f_scale0)
    failed0 = ~finite0

    init_carry = (
        xi0,
        opt_state0,
        v0,
        g0,
        w0_phys,
        xi0,
        v0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        conv0,
        failed0,
    )

    def _cond(carry):
        _, _, _, _, _, _, _, _, step, conv, failed = carry
        return (~conv) & (~failed) & (step < max_steps)

    def _body(carry):
        xi, opt_state, v_curr, g_curr, w_curr, best_xi, best_v, consec, step, _, _ = carry
        updates, next_state = opt.update(
            g_curr,
            opt_state,
            xi,
            value=v_curr,
            grad=g_curr,
            value_fn=val_fn,
        )
        xi_cand = optax.apply_updates(xi, updates)
        v_cand, g_cand = vg_fn(xi_cand)
        finite_step = (
            jnp.all(jnp.isfinite(xi_cand))
            & jnp.isfinite(v_cand)
            & jnp.all(jnp.isfinite(g_cand))
        )

        xi_next = jnp.where(finite_step, xi_cand, xi)
        v_next = jnp.where(finite_step, v_cand, v_curr)
        g_next = jnp.where(finite_step, g_cand, g_curr)
        state_next = jax.tree.map(
            lambda a, b: jnp.where(finite_step, a, b),
            next_state,
            opt_state,
        )

        improved = finite_step & (v_next < best_v)
        next_best_xi = jnp.where(improved, xi_next, best_xi)
        next_best_v = jnp.where(improved, v_next, best_v)

        w_next = weights_from_xi(xi_next)
        f_scale = jnp.maximum(jnp.abs(v_next), 1.0)
        grad_ok = finite_step & (jnp.max(jnp.abs(g_next)) <= atol + rtol * f_scale)
        dw_max = jnp.max(jnp.abs(w_next - w_curr))
        dv_rel = jnp.abs(v_curr - v_next) / f_scale
        step_small = (
            finite_step
            & (v_next <= v_curr + 1e-12 * f_scale)
            & (dw_max <= atol + rtol)
            & (dv_rel <= rtol)
        )
        consec_next = jnp.where(step_small, consec + 1, jnp.zeros_like(consec))
        conv_next = grad_ok | (consec_next >= 2)
        failed_next = ~finite_step
        return (
            xi_next,
            state_next,
            v_next,
            g_next,
            w_next,
            next_best_xi,
            next_best_v,
            consec_next,
            step + 1,
            conv_next,
            failed_next,
        )

    final_carry = jax.lax.while_loop(_cond, _body, init_carry)
    _, _, _, _, _, best_xi, best_v, _, n_steps, converged, _ = final_carry
    use_best = jnp.isfinite(best_v) & (best_v <= v0)
    final_xi = jnp.where(use_best, best_xi, xi0)
    final_v = jnp.where(use_best, best_v, v0)
    return final_xi, v0, final_v, converged & jnp.isfinite(final_v), n_steps


@functools.lru_cache(maxsize=32)
def _get_lbfgs_proj_runner(
    objective: str,
    epsilon: float,
    tau: float | None,
    rtol: float,
    atol: float,
):
    """Build and cache a Tier-1a Fisher-preconditioned ``optax.lbfgs`` zoom-LS runner."""

    @eqx.filter_jit
    def _run(
        w0: HaloWeights,
        obs: HaloObservation,
        proj: jax.Array,
        max_steps: int,
    ) -> tuple[HaloWeights, jax.Array, jax.Array, jax.Array, jax.Array]:
        z0 = w0.z_ilr
        legacy_tv = w0.legacy_tv
        n_scale = jnp.maximum(obs.n_eff, 1.0)

        def _val_fn(xi: jax.Array) -> jax.Array:
            w = HaloWeights(z_ilr=z0 + jnp.dot(proj, xi), legacy_tv=legacy_tv)
            return _halo_objective(w, obs, objective=objective, epsilon=epsilon, tau=tau) / n_scale

        def _w_phys(xi: jax.Array) -> jax.Array:
            w = HaloWeights(z_ilr=z0 + jnp.dot(proj, xi), legacy_tv=legacy_tv)
            return w._physical_weights_raw(obs.mean_pixel_flux)

        xi0 = jnp.zeros_like(z0)
        best_xi, v0, best_v, conv, n_steps = _run_optax_lbfgs_loop(
            _val_fn, _w_phys, xi0, max_steps, rtol, atol
        )
        z_best = z0 + jnp.dot(proj, best_xi)
        return (
            HaloWeights(z_ilr=z_best, legacy_tv=legacy_tv),
            v0 * n_scale,
            best_v * n_scale,
            conv,
            n_steps,
        )

    return _run


@functools.lru_cache(maxsize=32)
def _get_lbfgs_diag_runner(
    objective: str,
    epsilon: float,
    tau: float | None,
    rtol: float,
    atol: float,
):
    """Build and cache a Tier-1b O(M) diagonally scaled ``optax.lbfgs`` zoom-LS runner."""

    @eqx.filter_jit
    def _run(
        w0: HaloWeights,
        obs: HaloObservation,
        max_steps: int,
    ) -> tuple[HaloWeights, jax.Array, jax.Array, jax.Array, jax.Array]:
        z0 = w0.z_ilr
        legacy_tv = w0.legacy_tv
        n_scale = jnp.maximum(obs.n_eff, 1.0)
        u0 = _ilr_to_logits(z0)
        d_scale = _compute_diag_logit_scale(w0, obs, objective)

        def _z_from_xi(xi: jax.Array) -> jax.Array:
            u = u0 + d_scale * _ilr_to_logits(xi)
            return _logits_to_ilr(u)

        def _val_fn(xi: jax.Array) -> jax.Array:
            w = HaloWeights(z_ilr=_z_from_xi(xi), legacy_tv=legacy_tv)
            return _halo_objective(w, obs, objective=objective, epsilon=epsilon, tau=tau) / n_scale

        def _w_phys(xi: jax.Array) -> jax.Array:
            w = HaloWeights(z_ilr=_z_from_xi(xi), legacy_tv=legacy_tv)
            return w._physical_weights_raw(obs.mean_pixel_flux)

        xi0 = jnp.zeros_like(z0)
        best_xi, v0, best_v, conv, n_steps = _run_optax_lbfgs_loop(
            _val_fn, _w_phys, xi0, max_steps, rtol, atol
        )
        z_best = _z_from_xi(best_xi)
        return (
            HaloWeights(z_ilr=z_best, legacy_tv=legacy_tv),
            v0 * n_scale,
            best_v * n_scale,
            conv,
            n_steps,
        )

    return _run


@functools.lru_cache(maxsize=16)
def _get_zodiax_runner(
    objective: str,
    epsilon: float,
    tau: float | None,
    learning_rate: float,
    max_steps: int,
    rtol: float = 5e-5,
    atol: float = 5e-5,
):
    """Build and cache a Tier-2 ``zodiax.optimisation`` + ``optax.adam`` runner."""

    def _loss_z_norm(z_ilr: jax.Array, legacy_tv: bool, obs: HaloObservation) -> jax.Array:
        w = HaloWeights(z_ilr=z_ilr, legacy_tv=legacy_tv)
        n_scale = jnp.maximum(obs.n_eff, 1.0)
        return _halo_objective(w, obs, objective=objective, epsilon=epsilon, tau=tau) / n_scale

    @eqx.filter_jit
    def _run(
        w0: HaloWeights,
        obs: HaloObservation,
        proj: jax.Array,
    ) -> tuple[HaloWeights, jax.Array, jax.Array, jax.Array, jax.Array]:
        z0 = w0.z_ilr
        legacy_tv = w0.legacy_tv
        n_scale = jnp.maximum(obs.n_eff, 1.0)
        loss0_norm = _loss_z_norm(z0, legacy_tv, obs)

        params0 = {"xi": jnp.zeros_like(z0)}
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=max(max_steps, 1),
            alpha=0.005,
        )
        optim, opt_state0 = zdx.optimisation.map_optimisers(
            params0,
            {"xi": optax.adam(schedule)},
        )

        def _xi_loss(p_dict: dict[str, jax.Array]) -> jax.Array:
            z_curr = z0 + jnp.dot(proj, p_dict["xi"])
            return _loss_z_norm(z_curr, legacy_tv, obs)

        vg_fn = jax.value_and_grad(_xi_loss)

        def _step(carry, _):
            p_curr, st_curr, best_xi, best_val = carry
            val, grads = vg_fn(p_curr)
            grads = zdx.optimisation.zero_nan_check(grads)
            updates, st_next = optim.update(grads, st_curr, p_curr)
            p_next = optax.apply_updates(p_curr, updates)
            improved = jnp.isfinite(val) & (val < best_val)
            next_best_xi = jnp.where(improved, p_curr["xi"], best_xi)
            next_best_val = jnp.where(improved, val, best_val)
            return (p_next, st_next, next_best_xi, next_best_val), None

        (p_end, _, best_xi, best_val), _ = jax.lax.scan(
            _step,
            (params0, opt_state0, params0["xi"], loss0_norm),
            None,
            length=max_steps,
        )
        end_val = _xi_loss(p_end)
        use_end = jnp.isfinite(end_val) & (end_val < best_val)
        final_xi = jnp.where(use_end, p_end["xi"], best_xi)
        final_val_norm, final_grad = vg_fn({"xi": final_xi})
        grad_norm = jnp.max(jnp.abs(final_grad["xi"]))
        tol_thresh = atol + rtol * jnp.maximum(jnp.abs(final_val_norm), 1.0)
        z_best = z0 + jnp.dot(proj, final_xi)
        converged = (
            jnp.isfinite(final_val_norm) & jnp.isfinite(grad_norm) & (grad_norm <= tol_thresh)
        )
        n_steps = jnp.asarray(max_steps, dtype=jnp.int64)
        return (
            HaloWeights(z_ilr=z_best, legacy_tv=legacy_tv),
            loss0_norm * n_scale,
            final_val_norm * n_scale,
            converged,
            n_steps,
        )

    return _run


def cadence_scaled_savgol_window(dt_days: float, window_days: float = 1.0) -> int:
    """Compute an odd Savitzky-Golay window ``max(5, int(round(window_days / dt_days)) | 1)``."""
    if not np.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError(f"dt_days must be finite and > 0, got {dt_days!r}.")
    if not np.isfinite(window_days) or window_days <= 0.0:
        raise ValueError(f"window_days must be finite and > 0, got {window_days!r}.")
    return max(5, int(round(window_days / dt_days)) | 1)


def cadence_scaled_sigclip_mask(
    flux: np.ndarray,
    times: np.ndarray | None = None,
    *,
    dt_days: float | None = None,
    window_days: float = 1.0,
    sigma_lower: float = 10.0,
    sigma_upper: float = 3.0,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Return a boolean inlier mask using a gap-aware narrow local spike detector.

    Filters finite ``(flux, times)`` cadences before segmenting across gaps so
    ``NaN``-represented downlink blocks are never interpolated across, then predicts
    each cadence from a fixed narrow 5-point (``hw = 2``) leave-one-out local median
    (with robust 7-point Theil-Sen linear prediction at segment boundaries). This
    detects single-cadence cosmic-ray and thruster spikes while preserving
    astrophysical eclipses, transits, and multi-periodic pulsations.

    Notes
    -----
    ``window_days`` is retained for API compatibility and input validation (checked
    via :func:`cadence_scaled_savgol_window`), while the local spike baseline uses
    the fixed narrow 5-point window rather than a multi-hour polynomial baseline.
    """
    y = np.asarray(flux, dtype=np.float64)
    n_cad = y.shape[0]
    if times is not None:
        t_arr: np.ndarray | None = np.asarray(times, dtype=np.float64)
        fin_mask = np.isfinite(y) & np.isfinite(t_arr)
    else:
        t_arr = None
        fin_mask = np.isfinite(y)

    if n_cad < 3:
        return fin_mask.copy()

    fin_idx = np.flatnonzero(fin_mask)
    if fin_idx.size == 0:
        return np.zeros(n_cad, dtype=bool)

    if dt_days is None:
        if t_arr is not None:
            diffs = np.diff(t_arr[fin_idx])
            pos_diffs = diffs[diffs > 0.0]
            dt_val = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)
        else:
            dt_val = 1.0 / 48.0
    else:
        dt_val = float(dt_days)

    cadence_scaled_savgol_window(dt_val, window_days=window_days)

    if t_arr is not None:
        breaks = np.flatnonzero(np.diff(t_arr[fin_idx]) > gap_factor * dt_val) + 1
    else:
        breaks = np.flatnonzero(np.diff(fin_idx) > 1) + 1
    segments = np.split(fin_idx, breaks)

    hw = 2
    nbr_offsets = np.array([-2, -1, 1, 2], dtype=np.intp)
    baseline = np.full(n_cad, np.nan, dtype=np.float64)

    for seg in segments:
        seg_len = seg.size
        if seg_len == 0:
            continue
        ys = y[seg]
        if seg_len < 5:
            baseline[seg] = float(np.median(ys))
            continue
        if seg_len < 7:
            b_seg = np.empty(seg_len, dtype=np.float64)
            for j in range(seg_len):
                others = np.delete(ys, j)
                b_seg[j] = float(np.median(others))
            baseline[seg] = b_seg
            continue

        b_seg = np.empty(seg_len, dtype=np.float64)
        centres = np.arange(hw, seg_len - hw, dtype=np.intp)
        nbr_vals = ys[centres[:, None] + nbr_offsets[None, :]]
        b_seg[centres] = np.median(nbr_vals, axis=1)

        ts = t_arr[seg] if t_arr is not None else np.arange(seg_len, dtype=np.float64)
        edge_win = min(seg_len, 7)
        for j in (0, 1, seg_len - 2, seg_len - 1):
            lo = max(0, min(j - edge_win // 2, seg_len - edge_win))
            hi = lo + edge_win
            others = [k for k in range(lo, hi) if k != j]
            to = ts[others]
            yo = ys[others]
            slopes = []
            for a in range(len(others)):
                for b in range(a + 1, len(others)):
                    dt_ab = to[b] - to[a]
                    if abs(dt_ab) > 0.0:
                        slopes.append((yo[b] - yo[a]) / dt_ab)
            slope = float(np.median(slopes)) if slopes else 0.0
            b_seg[j] = float(np.median(yo + slope * (ts[j] - to)))

        baseline[seg] = b_seg

    inliers = np.zeros(n_cad, dtype=bool)
    fin_all = fin_mask & np.isfinite(baseline)
    if np.any(fin_all):
        clipped = sigma_clip(
            y[fin_all] - baseline[fin_all],
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            stdfunc=mad_std,
        )
        inliers[fin_all] = ~np.ma.getmaskarray(clipped)
    return inliers


@in_float64
def _solve_single_pass(
    w0: HaloWeights,
    obs: HaloObservation,
    *,
    objective: str,
    epsilon: float,
    tau: float | None,
    solver: str,
    maxiter: int,
    rtol: float,
    atol: float,
    learning_rate: float,
    precondition: bool,
) -> tuple[HaloWeights, float, float, bool, int]:
    """Run one optimization pass on ``obs`` starting from ``w0``."""
    m_minus_1 = w0.z_ilr.shape[0]
    if m_minus_1 == 0:
        loss_val = float(_halo_objective(w0, obs, objective=objective, epsilon=epsilon, tau=tau))
        return w0, loss_val, loss_val, True, 0

    if solver == "lbfgs":
        if precondition and m_minus_1 <= 256:
            fisher_fn = _get_fisher_fn(objective, tau, epsilon)
            f_mat = fisher_fn(w0, obs)
            proj = safe_eigen_projection(fmat=f_mat, rcond=1e-8)
            runner_proj = _get_lbfgs_proj_runner(objective, epsilon, tau, rtol, atol)
            w_best, loss0, loss_end, conv, n_steps = runner_proj(w0, obs, proj, maxiter)
            return w_best, float(loss0), float(loss_end), bool(conv), int(n_steps)

        runner_diag = _get_lbfgs_diag_runner(objective, epsilon, tau, rtol, atol)
        w_best, loss0, loss_end, conv, n_steps = runner_diag(w0, obs, maxiter)
        return w_best, float(loss0), float(loss_end), bool(conv), int(n_steps)

    if solver in ("zodiax", "adam"):
        if precondition and m_minus_1 <= 256:
            fisher_fn = _get_fisher_fn(objective, tau, epsilon)
            f_mat = fisher_fn(w0, obs)
            proj = safe_eigen_projection(fmat=f_mat, rcond=1e-4)
            eff_lr = float(learning_rate) / math.sqrt(max(m_minus_1, 1))
        else:
            proj = jnp.eye(m_minus_1, dtype=jnp.float64)
            eff_lr = float(learning_rate)

        runner_zdx = _get_zodiax_runner(objective, epsilon, tau, eff_lr, maxiter, rtol, atol)
        w_best, loss0, loss_end, conv, n_steps = runner_zdx(w0, obs, proj)
        return w_best, float(loss0), float(loss_end), bool(conv), int(n_steps)

    raise ValueError(f"Unknown solver {solver!r}; expected 'lbfgs', 'zodiax', or 'adam'.")


def _partition_orbit_blocks(
    times_valid: np.ndarray,
    dt_nom: float,
    k_eff: int,
    min_req: int,
) -> list[np.ndarray]:
    """Split active cadence indices into ``k_eff`` blocks respecting major orbit/downlink gaps."""
    n_valid = times_valid.shape[0]
    gap_thresh = max(10.0 * dt_nom, 0.25)
    breaks = np.flatnonzero(np.diff(times_valid) > gap_thresh) + 1
    if breaks.size == 0:
        return [blk for blk in np.array_split(np.arange(n_valid), k_eff) if blk.size > 0]

    segs = [s for s in np.split(np.arange(n_valid), breaks) if s.size > 0]
    if len(segs) <= k_eff:
        counts = np.ones(len(segs), dtype=int)
        rem = k_eff - len(segs)
        if rem > 0:
            lengths = np.array([s.size for s in segs], dtype=np.float64)
            raw_extra = rem * (lengths / np.sum(lengths))
            extra = np.floor(raw_extra).astype(int)
            leftover = rem - int(np.sum(extra))
            if leftover > 0:
                order = np.argsort(-(raw_extra - extra))
                extra[order[:leftover]] += 1
            counts += extra
        blocks: list[np.ndarray] = []
        for seg, cnt in zip(segs, counts, strict=True):
            c_use = min(int(cnt), seg.size)
            for sub_blk in np.array_split(seg, c_use):
                if sub_blk.size > 0:
                    blocks.append(sub_blk)
        if len(blocks) == k_eff and all((n_valid - b.size) >= min_req for b in blocks):
            return blocks

    return [blk for blk in np.array_split(np.arange(n_valid), k_eff) if blk.size > 0]


@in_float64
def orbit_block_jackknife(
    obs: HaloObservation,
    w_hat: np.ndarray | jax.Array,
    *,
    n_blocks: int = 4,
    legacy_tv: bool = False,
    objective: str = "tv",
    epsilon: float = 1e-5,
    tau: float | None = None,
    solver: str = "lbfgs",
    maxiter: int = 500,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    learning_rate: float = 0.2,
    precondition: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate physical weight uncertainties ``sigma_w`` via ``K``-block orbit jackknife.

    Splits active cadences along major orbit/downlink gaps when present, leaves out
    each block ``B_k`` in turn while retaining ``obs.N_pad`` unchanged for zero XLA
    recompilation, and combines leave-out folds using Busing-Meijer-van der Leeden
    delete-``m_k`` constraint weighting. Returns ``(weight_std, jackknife_weights)``
    where ``jackknife_weights`` has shape ``(k_eff, M)`` whenever ``k_eff >= 2`` folds
    succeed (or ``(n_blocks, M)`` of ``NaN`` when fewer than 2 valid folds exist).
    """
    if isinstance(n_blocks, bool) or int(n_blocks) < 2:
        raise ValueError(f"n_blocks must be an integer >= 2, got {n_blocks!r}.")
    k_req = int(n_blocks)

    w_full = np.asarray(w_hat, dtype=np.float64)
    m_pix = w_full.shape[0]
    n_valid = int(obs.n_valid)
    min_req = 2 * obs.lag + 1 if objective == "tv_o2" else (2 if objective == "owl" else obs.lag + 1)

    if m_pix <= 1:
        return np.zeros(m_pix, dtype=np.float64), np.tile(w_full, (k_req, 1))
    if n_valid < max(4, min_req + 2):
        return (
            np.full(m_pix, np.nan, dtype=np.float64),
            np.full((k_req, m_pix), np.nan, dtype=np.float64),
        )

    k_eff = min(k_req, n_valid // 2)
    t_valid = np.asarray(obs.times_padded[:n_valid], dtype=np.float64)
    blocks = _partition_orbit_blocks(t_valid, float(obs.dt_nom), k_eff, min_req)
    n_eff_full = max(obs.effective_constraint_count(objective=objective, tau=tau), 1)

    jk_weights: list[np.ndarray] = []
    dropped_counts: list[float] = []
    n_unconverged = 0
    for blk in blocks:
        keep = np.ones(n_valid, dtype=bool)
        keep[blk] = False
        if int(np.sum(keep)) < min_req:
            continue
        obs_k = obs.with_cadence_subset(
            keep,
            legacy_tv=legacy_tv,
            preserve_regularization=True,
            objective=objective,
            tau=tau,
        )
        w0_k = HaloWeights.from_physical_weights(
            w_full,
            obs_k.mean_pixel_flux,
            legacy_tv=legacy_tv,
        )
        w_tree_k, _, _, conv_k, _ = _solve_single_pass(
            w0_k,
            obs_k,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
            solver=solver,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            learning_rate=learning_rate,
            precondition=precondition,
        )
        if solver == "lbfgs" and not conv_k:
            n_unconverged += 1
        jk_weights.append(
            np.asarray(w_tree_k._physical_weights_raw(obs_k.mean_pixel_flux), dtype=np.float64)
        )
        n_eff_k = obs_k.effective_constraint_count(objective=objective, tau=tau)
        dropped_counts.append(float(max(n_eff_full - n_eff_k, 1)))

    if len(jk_weights) < 2:
        return (
            np.full(m_pix, np.nan, dtype=np.float64),
            np.full((k_req, m_pix), np.nan, dtype=np.float64),
        )

    if n_unconverged > 0:
        warnings.warn(
            f"orbit_block_jackknife: {n_unconverged}/{len(jk_weights)} L-BFGS folds did not "
            f"converge within maxiter={maxiter} steps; WEIGHTMAP_ERR may be underestimated.",
            RuntimeWarning,
            stacklevel=3,
        )

    w_jk = np.vstack(jk_weights)
    k_used = w_jk.shape[0]
    m_k = np.asarray(dropped_counts, dtype=np.float64)
    m_tot = float(np.sum(m_k))
    comp_weights = np.maximum(m_tot - m_k, 0.0)
    comp_sum = float(np.sum(comp_weights))
    h_k = np.maximum(m_tot / m_k, 1.0)
    if comp_sum > 0.0:
        w_mean = np.sum(comp_weights[:, None] * w_jk, axis=0, keepdims=True) / comp_sum
        var_jk = np.sum((h_k - 1.0)[:, None] * np.square(w_jk - w_mean), axis=0) / float(k_used)
    else:
        w_mean = np.mean(w_jk, axis=0, keepdims=True)
        var_jk = ((k_used - 1.0) / float(k_used)) * np.sum(np.square(w_jk - w_mean), axis=0)
    return np.sqrt(np.maximum(var_jk, 0.0)), w_jk


def _default_initial_weights(obs: HaloObservation, *, legacy_tv: bool) -> HaloWeights:
    """Build default starting weights (gain-covariant inverse-variance start when heteroscedastic)."""
    if legacy_tv:
        return HaloWeights.from_uniform(obs.mean_pixel_flux, legacy_tv=True)
    fv = np.asarray(obs.mean_frac_var, dtype=np.float64)
    p_bar = np.asarray(obs.mean_pixel_flux, dtype=np.float64)
    safe_fv = _safe_minvar_frac_var(p_bar, fv)
    if safe_fv.size > 1 and np.all(np.isfinite(safe_fv)) and np.all(safe_fv > 0.0) and np.all(p_bar > 0.0):
        inv_fv = 1.0 / safe_fv
        th_mv = inv_fv / np.sum(inv_fv)
        w_mv = th_mv / p_bar
        w_mv_sum = float(np.sum(w_mv))
        if np.isfinite(w_mv_sum) and w_mv_sum > 0.0:
            return HaloWeights.from_physical_weights(
                w_mv / w_mv_sum,
                obs.mean_pixel_flux,
                legacy_tv=False,
            )
    return HaloWeights.from_uniform(obs.mean_pixel_flux, legacy_tv=False)


@in_float64
def optimise_halo(
    pixels: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array | None = None,
    *,
    pixel_var: np.ndarray | jax.Array | None = None,
    w_init: np.ndarray | jax.Array | None = None,
    objective: str = "tv",
    lag: int = 1,
    tau: float | None = None,
    epsilon: float = 1e-5,
    legacy_tv: bool = False,
    lambda_phot: float | None = None,
    lambda_z: float | None = None,
    solver: str = "lbfgs",
    maxiter: int = 500,
    rtol: float = 5e-5,
    atol: float = 5e-5,
    learning_rate: float = 0.2,
    precondition: bool = True,
    sigclip: bool = False,
    window_days: float = 1.0,
    sigma_lower: float = 10.0,
    sigma_upper: float = 3.0,
    compute_uncertainty: bool = False,
    jackknife_blocks: int = 4,
    n_pad: int | None = None,
    gap_tol: float = 0.25,
) -> HaloFitResult:
    """Optimize halo aperture weights in locally scoped float64 precision.

    Notes
    -----
    * When ``pixel_var=None`` and ``legacy_tv=False``, the automatic Poisson
      shot-noise floor assumes ``pixels`` is in physical ``e-/s`` units. Supply
      ``pixel_var`` explicitly for exact unit-gain invariance on rescaled fluxes.
    * When ``sigclip=True``, narrow local spike clipping excludes outlier cadences
      strictly during weight optimization; ``raw_flux`` and ``norm_flux`` are always
      evaluated across all finite input cadences at the optimal weights ``w_opt``.
    """
    if isinstance(maxiter, bool) or int(maxiter) < 1:
        raise ValueError(f"maxiter must be an integer >= 1, got {maxiter!r}.")
    maxiter_int = int(maxiter)

    pix_np = np.asarray(pixels, dtype=np.float64)
    obs = HaloObservation.from_arrays(
        pix_np,
        times,
        pixel_var=pixel_var,
        lag=lag,
        objective=objective,
        tau=tau,
        legacy_tv=legacy_tv,
        lambda_phot=lambda_phot,
        lambda_z=lambda_z,
        n_pad=n_pad,
        gap_tol=gap_tol,
    )

    n_valid = int(obs.n_valid)
    min_req = 2 * obs.lag + 1 if objective == "tv_o2" else (2 if objective == "owl" else obs.lag + 1)
    if n_valid < min_req:
        raise ValueError(
            f"Objective {objective!r} with lag={obs.lag} requires at least {min_req} cadences, "
            f"got {n_valid}."
        )

    m_pix = pix_np.shape[0]
    n_valid_diffs = obs.effective_constraint_count(objective=objective, tau=tau)
    if m_pix > 1 and m_pix >= 0.5 * n_valid_diffs:
        warnings.warn(
            f"Active halo pixel count M={m_pix} approaches or exceeds valid cadence "
            f"constraints N_valid={n_valid_diffs} (M >= 0.5 * N_valid); simplex weights "
            "have elevated degrees of freedom relative to cadence constraints.",
            UserWarning,
            stacklevel=3,
        )

    if w_init is None:
        w0 = _default_initial_weights(obs, legacy_tv=legacy_tv)
    else:
        w_arr = np.asarray(w_init, dtype=np.float64)
        if w_arr.shape != (pix_np.shape[0],):
            raise ValueError(
                f"w_init shape {w_arr.shape} does not match (M,) = ({pix_np.shape[0]},)."
            )
        if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0) or float(np.sum(w_arr)) <= 0.0:
            raise ValueError("w_init simplex weights must be finite, non-negative, and sum > 0.")
        w0 = HaloWeights.from_physical_weights(w_arr, obs.mean_pixel_flux, legacy_tv=legacy_tv)

    w_tree, loss0, loss_final, converged, n_steps = _solve_single_pass(
        w0,
        obs,
        objective=objective,
        epsilon=epsilon,
        tau=tau,
        solver=solver,
        maxiter=maxiter_int,
        rtol=rtol,
        atol=atol,
        learning_rate=learning_rate,
        precondition=precondition,
    )

    w_phys = np.asarray(w_tree._physical_weights_raw(obs.mean_pixel_flux), dtype=np.float64)
    raw_lc = w_phys @ pix_np
    inliers = np.ones(n_valid, dtype=bool)
    active_obs = obs

    if sigclip:
        t_valid = np.asarray(obs.times_padded[:n_valid], dtype=np.float64)
        inliers = cadence_scaled_sigclip_mask(
            raw_lc,
            t_valid,
            dt_days=float(obs.dt_nom),
            window_days=window_days,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
        )
        if np.any(~inliers) and int(np.sum(inliers)) >= min_req:
            obs_sub = obs.with_cadence_subset(
                inliers,
                legacy_tv=legacy_tv,
                preserve_regularization=False,
                objective=objective,
                tau=tau,
            )
            w0_init_sub = (
                _default_initial_weights(obs_sub, legacy_tv=legacy_tv)
                if w_init is None
                else HaloWeights.from_physical_weights(
                    np.asarray(w_init, dtype=np.float64),
                    obs_sub.mean_pixel_flux,
                    legacy_tv=legacy_tv,
                )
            )
            loss0_sub = float(
                _halo_objective(
                    w0_init_sub,
                    obs_sub,
                    objective=objective,
                    epsilon=epsilon,
                    tau=tau,
                )
            )
            w0_refit = HaloWeights.from_physical_weights(
                w_phys,
                obs_sub.mean_pixel_flux,
                legacy_tv=legacy_tv,
            )
            w_tree, _, loss_final, converged, n_steps = _solve_single_pass(
                w0_refit,
                obs_sub,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
                solver=solver,
                maxiter=maxiter_int,
                rtol=rtol,
                atol=atol,
                learning_rate=learning_rate,
                precondition=precondition,
            )
            if not np.isfinite(loss_final) or loss_final > loss0_sub:
                w_tree, _, loss_final, converged, n_steps = _solve_single_pass(
                    w0_init_sub,
                    obs_sub,
                    objective=objective,
                    epsilon=epsilon,
                    tau=tau,
                    solver=solver,
                    maxiter=maxiter_int,
                    rtol=rtol,
                    atol=atol,
                    learning_rate=learning_rate,
                    precondition=precondition,
                )
            loss0 = loss0_sub
            w_phys = np.asarray(
                w_tree._physical_weights_raw(obs_sub.mean_pixel_flux),
                dtype=np.float64,
            )
            raw_lc = w_phys @ pix_np
            active_obs = obs_sub
        elif np.any(~inliers):
            inliers = np.ones(n_valid, dtype=bool)

    if solver == "lbfgs" and m_pix > 1 and not converged:
        warnings.warn(
            f"L-BFGS solver did not converge within maxiter={maxiter_int} steps "
            f"(completed {n_steps} steps); consider increasing maxiter.",
            RuntimeWarning,
            stacklevel=3,
        )

    flux_shares = np.asarray(w_tree._flux_shares_raw(), dtype=np.float64)
    denom = float(np.dot(w_phys, np.mean(pix_np[:, inliers], axis=1)))
    safe_denom = denom if abs(denom) > 1e-30 else 1.0
    norm_lc = raw_lc / safe_denom

    weight_std: np.ndarray | None = None
    jk_weights: np.ndarray | None = None
    if compute_uncertainty:
        jk_rtol = min(float(rtol), 1e-7)
        jk_atol = min(float(atol), 1e-7)
        weight_std, jk_weights = orbit_block_jackknife(
            active_obs,
            w_phys,
            n_blocks=jackknife_blocks,
            legacy_tv=legacy_tv,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
            solver=solver,
            maxiter=maxiter_int,
            rtol=jk_rtol,
            atol=jk_atol,
            learning_rate=learning_rate,
            precondition=precondition,
        )

    return HaloFitResult(
        weights_tree=w_tree,
        physical_weights=w_phys,
        flux_shares=flux_shares,
        norm_flux=norm_lc,
        raw_flux=raw_lc,
        inlier_mask=inliers,
        loss_init=loss0,
        loss_final=loss_final,
        weight_std=weight_std,
        jackknife_weights=jk_weights,
        converged=converged,
        n_steps=n_steps,
    )
