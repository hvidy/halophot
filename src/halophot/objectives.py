"""Gap-aware Charbonnier total-variation objectives and O(N) scan ``tv_exp``.

Implements the convex flux-share objective family from Sections 6.3 and 6.4:

* Cancellation-free Charbonnier total variation
  ``phi_eps(r) = r^2 / (sqrt(r^2 + eps^2) + eps)`` strictly over gap-masked
  lag-``ell`` index sets ``V_ell`` so spacecraft downlink / safe-mode gaps are
  never bridged by finite differences.
* O(N) memory ``jax.lax.scan`` multi-lag ``tv_exp`` (``< 10 MB`` RAM at 20-s
  TESS cadence vs 115.2 GB for an ``N x N`` pairwise difference matrix).
* Photon-noise regularization ``(lambda_phot / 2) * sum_i theta_i^2 * sigma_tilde_i^2``
  and ILR logit damping ``(lambda_z / 2) * ||z - z_init||_2^2``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .models import HaloObservation, HaloWeights

__all__ = [
    "AVAILABLE_OBJECTIVES",
    "charbonnier_penalty",
    "halo_objective",
    "l2v_gap_masked",
    "l3v_gap_masked",
    "logit_damping_penalty",
    "owl_masked",
    "photon_noise_penalty",
    "tv_charbonnier_o1",
    "tv_charbonnier_o2",
    "tv_exp_scan",
]

AVAILABLE_OBJECTIVES: tuple[str, ...] = ("tv", "tv_o2", "tv_exp", "l2v", "l3v", "owl")


def charbonnier_penalty(r: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    """Evaluate ``phi_eps(r) = r^2 / (sqrt(r^2 + eps^2) + eps)`` without cancellation.

    Algebraically identical to ``sqrt(r^2 + eps^2) - eps`` in exact arithmetic,
    this rationalized form avoids catastrophic floating-point cancellation when
    ``|r| << eps`` and has finite derivatives of all orders at ``r = 0``.
    """
    r_arr = jnp.asarray(r)
    eps = jnp.maximum(jnp.asarray(epsilon, dtype=r_arr.dtype), jnp.finfo(r_arr.dtype).tiny)
    r2 = jnp.square(r_arr)
    return r2 / (jnp.sqrt(r2 + eps * eps) + eps)


def tv_charbonnier_o1(
    f_tilde: jax.Array,
    obs: HaloObservation,
    *,
    epsilon: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mean_tv_o1, n_valid_pairs)`` for 1st-order gap-masked Charbonnier TV."""
    lag = obs.lag
    if lag >= obs.N_pad:
        zero = jnp.zeros((), dtype=f_tilde.dtype)
        return zero, zero

    mask = obs.valid_diff_mask[:-lag]
    diffs = jnp.where(mask, f_tilde[lag:] - f_tilde[:-lag], 0.0)
    penalties = jnp.where(mask, charbonnier_penalty(diffs, epsilon=epsilon), 0.0)
    count = jnp.sum(mask.astype(f_tilde.dtype))
    mean_tv = jnp.where(count > 0.0, jnp.sum(penalties) / jnp.maximum(count, 1.0), 0.0)
    return mean_tv, count


def tv_charbonnier_o2(
    f_tilde: jax.Array,
    obs: HaloObservation,
    *,
    epsilon: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mean_tv_o2, n_valid_triples)`` for 2nd-order gap-masked Charbonnier TV."""
    lag = obs.lag
    if 2 * lag >= obs.N_pad:
        zero = jnp.zeros((), dtype=f_tilde.dtype)
        return zero, zero

    mask_o2 = obs.valid_diff_mask[: -2 * lag] & obs.valid_diff_mask[lag:-lag]
    fallback_o2 = (
        obs.valid_cadence_mask[: -2 * lag]
        & obs.valid_cadence_mask[lag:-lag]
        & obs.valid_cadence_mask[2 * lag :]
    )
    mask = jnp.where(jnp.any(mask_o2), mask_o2, fallback_o2)
    diffs2 = jnp.where(
        mask,
        2.0 * f_tilde[lag:-lag] - f_tilde[2 * lag :] - f_tilde[: -2 * lag],
        0.0,
    )
    penalties = jnp.where(mask, charbonnier_penalty(diffs2, epsilon=epsilon), 0.0)
    count = jnp.sum(mask.astype(f_tilde.dtype))
    mean_tv = jnp.where(count > 0.0, jnp.sum(penalties) / jnp.maximum(count, 1.0), 0.0)
    return mean_tv, count


def tv_exp_scan(
    f_tilde: jax.Array,
    obs: HaloObservation,
    *,
    tau: float | None = None,
    epsilon: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate multi-lag exponential TV via ``jax.lax.scan`` in O(N) memory.

    Accumulates ``sum_{ell=1}^{L_max} exp(-ell / tau) * L_TV^(1)(ell)`` normalized
    by ``sum_{ell=1}^{L_max} exp(-ell / tau)`` over ``L_max = min(ceil(4 * tau), N - 1)``
    with per-step gradient checkpointing so reverse-mode AD never materializes an
    ``(L_max, N)`` or ``(N, N)`` tensor.
    """
    tau_val = float(obs.lag) if tau is None else float(tau)
    if tau_val <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau_val!r}.")

    n_pad = obs.N_pad
    scan_len = min(max(int(math.ceil(4.0 * tau_val)), 1), n_pad - 1)
    if scan_len < 1:
        zero = jnp.zeros((), dtype=f_tilde.dtype)
        return zero, zero

    idx = jnp.arange(n_pad, dtype=jnp.int32)
    tau_arr = jnp.asarray(tau_val, dtype=f_tilde.dtype)
    gap_tol = obs.gap_tol

    @jax.checkpoint
    def _scan_step(
        carry: tuple[jax.Array, jax.Array],
        ell: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        num_acc, den_acc = carry
        in_bounds = idx < (n_pad - ell)
        both_active = in_bounds & obs.valid_cadence_mask & jnp.roll(obs.valid_cadence_mask, -ell)

        dt_pair = jnp.roll(obs.times_padded, -ell) - obs.times_padded
        expected_dt = ell.astype(obs.times_padded.dtype) * obs.dt_nom
        gap_ok = both_active & (jnp.abs(dt_pair - expected_dt) <= gap_tol * expected_dt)
        mask_ell = jnp.where(jnp.any(gap_ok), gap_ok, both_active)

        diffs = jnp.where(mask_ell, jnp.roll(f_tilde, -ell) - f_tilde, 0.0)
        phi = jnp.where(mask_ell, charbonnier_penalty(diffs, epsilon=epsilon), 0.0)
        count = jnp.sum(mask_ell.astype(f_tilde.dtype))
        l_tv_ell = jnp.where(count > 0.0, jnp.sum(phi) / jnp.maximum(count, 1.0), 0.0)

        active_lag = (ell < obs.n_valid) & (count > 0.0)
        w_ell = jnp.where(active_lag, jnp.exp(-ell.astype(f_tilde.dtype) / tau_arr), 0.0)
        return (num_acc + w_ell * l_tv_ell, den_acc + w_ell), None

    lags = jnp.arange(1, scan_len + 1, dtype=jnp.int32)
    zero = jnp.zeros((), dtype=f_tilde.dtype)
    (num_sum, den_sum), _ = jax.lax.scan(_scan_step, (zero, zero), lags)
    l_tv_exp = jnp.where(den_sum > 0.0, num_sum / jnp.maximum(den_sum, 1e-30), 0.0)

    n_eff = jnp.sum(obs.valid_diff_mask.astype(f_tilde.dtype))
    return l_tv_exp, n_eff


def l2v_gap_masked(
    f_tilde: jax.Array,
    obs: HaloObservation,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mean_l2v, n_valid_pairs)`` for gap-masked squared 1st differences."""
    lag = obs.lag
    if lag >= obs.N_pad:
        zero = jnp.zeros((), dtype=f_tilde.dtype)
        return zero, zero

    mask = obs.valid_diff_mask[:-lag]
    diffs = jnp.where(mask, f_tilde[lag:] - f_tilde[:-lag], 0.0)
    sq = jnp.where(mask, jnp.square(diffs), 0.0)
    count = jnp.sum(mask.astype(f_tilde.dtype))
    mean_l2 = jnp.where(count > 0.0, jnp.sum(sq) / jnp.maximum(count, 1.0), 0.0)
    return mean_l2, count


def l3v_gap_masked(
    f_tilde: jax.Array,
    obs: HaloObservation,
    *,
    epsilon: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(mean_l3v, n_valid_pairs)`` using strictly convex smooth ``r^2 * phi_eps(r)``."""
    lag = obs.lag
    if lag >= obs.N_pad:
        zero = jnp.zeros((), dtype=f_tilde.dtype)
        return zero, zero

    mask = obs.valid_diff_mask[:-lag]
    diffs = jnp.where(mask, f_tilde[lag:] - f_tilde[:-lag], 0.0)
    cubic_smooth = jnp.square(diffs) * charbonnier_penalty(diffs, epsilon=epsilon)
    vals = jnp.where(mask, cubic_smooth, 0.0)
    count = jnp.sum(mask.astype(f_tilde.dtype))
    mean_l3 = jnp.where(count > 0.0, jnp.sum(vals) / jnp.maximum(count, 1.0), 0.0)
    return mean_l3, count


def owl_masked(
    f_tilde: jax.Array,
    obs: HaloObservation,
    *,
    epsilon: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(smooth_std, n_active_cadences)`` over active cadences."""
    mask = obs.valid_cadence_mask.astype(f_tilde.dtype)
    count = jnp.sum(mask)
    safe_count = jnp.maximum(count, 1.0)
    mean_f = jnp.sum(mask * f_tilde) / safe_count
    centered = jnp.where(obs.valid_cadence_mask, f_tilde - mean_f, 0.0)
    var_f = jnp.sum(jnp.square(centered)) / safe_count
    eps = jnp.maximum(jnp.asarray(epsilon, dtype=f_tilde.dtype), jnp.finfo(f_tilde.dtype).tiny)
    smooth_std = var_f / (jnp.sqrt(var_f + eps * eps) + eps)
    return smooth_std, count


def photon_noise_penalty(theta: jax.Array, obs: HaloObservation) -> jax.Array:
    """Evaluate ``(lambda_phot / 2) * sum_i theta_i^2 * mean_frac_var_i``."""
    return 0.5 * obs.lambda_phot * jnp.sum(jnp.square(theta) * obs.mean_frac_var)


def logit_damping_penalty(z_ilr: jax.Array, obs: HaloObservation) -> jax.Array:
    """Evaluate ``(lambda_z / 2) * ||z_ilr - z_prior_centre||_2^2``."""
    if z_ilr.shape[0] == 0:
        return jnp.zeros((), dtype=z_ilr.dtype)
    diff = z_ilr - obs.z_prior_centre
    return 0.5 * obs.lambda_z * jnp.sum(jnp.square(diff))


def halo_objective(
    weights: HaloWeights,
    obs: HaloObservation,
    *,
    objective: str = "tv",
    epsilon: float = 1e-5,
    tau: float | None = None,
) -> jax.Array:
    """Evaluate the regularized v2.0 halo photometry objective ``J(z)`` (Eq. 6.4)."""
    if objective not in AVAILABLE_OBJECTIVES:
        valid = ", ".join(AVAILABLE_OBJECTIVES)
        raise ValueError(f"Unknown objective {objective!r}; expected one of: {valid}.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon!r}.")

    f_tilde = weights.evaluate_lightcurve(obs)
    b_scale = jnp.maximum(obs.b_hat, jnp.finfo(f_tilde.dtype).tiny)

    if objective == "tv":
        mean_val, count = tv_charbonnier_o1(f_tilde, obs, epsilon=epsilon)
        data_term = (count / b_scale) * mean_val
    elif objective == "tv_o2":
        mean_val, count = tv_charbonnier_o2(f_tilde, obs, epsilon=epsilon)
        data_term = (count / b_scale) * mean_val
    elif objective == "tv_exp":
        mean_val, count = tv_exp_scan(f_tilde, obs, tau=tau, epsilon=epsilon)
        data_term = (count / b_scale) * mean_val
    elif objective == "l2v":
        mean_val, count = l2v_gap_masked(f_tilde, obs)
        data_term = (count / jnp.square(b_scale)) * mean_val
    elif objective == "l3v":
        mean_val, count = l3v_gap_masked(f_tilde, obs, epsilon=epsilon)
        data_term = (count / (b_scale * jnp.square(b_scale))) * mean_val
    else:  # objective == "owl"
        std_val, count = owl_masked(f_tilde, obs, epsilon=epsilon)
        data_term = std_val if weights.legacy_tv else (count / b_scale) * std_val

    if weights.legacy_tv:
        return data_term

    theta = weights.flux_shares()
    return (
        data_term
        + photon_noise_penalty(theta, obs)
        + logit_damping_penalty(weights.z_ilr, obs)
    )
