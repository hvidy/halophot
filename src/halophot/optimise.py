"""Zodiax / Optimistix halo weight solvers, cadence-scaled sigclip, and block jackknife.

Provides:
* Cached ``eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))`` evaluation on
  :class:`~halophot.models.HaloWeights` PyTrees.
* Tier-1 ``optx.BestSoFarMinimiser(optx.LBFGS(...))`` quasi-Newton solver and
  Tier-2 ``zodiax.optimisation`` (``optax.adam`` + ``safe_eigen_projection``)
  first-order preconditioned solver, both scoped under ``@in_float64``.
* Gap-aware, cadence-scaled Savitzky-Golay outlier clipping
  (``window = max(5, int(round(window_days / dt_days)) | 1)``).
* Warm-started ``K = 4`` orbit block-jackknife uncertainty estimation reusing
  static ``N_pad`` padding for zero XLA recompilation.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
import zodiax as zdx
from astropy.stats import mad_std, sigma_clip
from scipy.signal import savgol_filter

from ._precision import in_float64
from .models import HaloObservation, HaloWeights
from .objectives import (
    AVAILABLE_OBJECTIVES,
    halo_objective,
    l2v_gap_masked,
    l3v_gap_masked,
    owl_masked,
    photon_noise_penalty,
    tv_charbonnier_o1,
    tv_charbonnier_o2,
    tv_exp_scan,
)
from .simplex import ilr_to_simplex

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
    """Result of ``optimise_halo`` on active halo pixels ``Omega_halo``."""

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


@functools.lru_cache(maxsize=32)
def build_value_and_grad(
    objective: str = "tv",
    epsilon: float = 1e-5,
    tau: float | None = None,
):
    """Return a cached ``eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))`` callable."""
    if objective not in AVAILABLE_OBJECTIVES:
        valid = ", ".join(AVAILABLE_OBJECTIVES)
        raise ValueError(f"Unknown objective {objective!r}; expected one of: {valid}.")

    def _loss(weights: HaloWeights, obs: HaloObservation) -> jax.Array:
        return halo_objective(weights, obs, objective=objective, epsilon=epsilon, tau=tau)

    return eqx.filter_jit(eqx.filter_value_and_grad(_loss))


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
    clamping both diagonal entries and small/negative eigenvalues strictly above zero
    so flat or rank-deficient curvature directions never emit ``0/0`` or ``1/sqrt(0)``
    ``NaN``s.
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
    max_diag = jnp.maximum(jnp.max(jnp.abs(diag)), 1.0)
    pos_floor = eps * max_diag
    if fmat is not None:
        # Non-positive Fisher diagonals carry zero local curvature information: damp to max_diag
        # rather than amplifying flat directions by 1 / sqrt(eps).
        safe_diag = jnp.where(diag > pos_floor, diag, max_diag)
    else:
        safe_diag = jnp.maximum(diag, pos_floor)

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
):
    """Build and cache a JIT-compiled Gauss-Newton Fisher curvature evaluator wrt ``z_ilr``.

    Evaluates the convex simplex Hessian ``H_theta`` at scale ``eps_gn = max(obs.b_hat, 1e-3)``
    and pulls it back through ``J_theta = d(theta)/d(z_ilr)`` so ``F_z`` is strictly symmetric
    positive definite without indefinite softmax second-derivative terms.
    """

    @eqx.filter_jit
    def _fisher(w0: HaloWeights, obs: HaloObservation) -> jax.Array:
        z0 = w0.z_ilr
        theta0 = ilr_to_simplex(z0)
        jac_theta = jax.jacrev(ilr_to_simplex)(z0)
        eps_gn = jnp.maximum(obs.b_hat, 1e-3)

        def _simplex_loss(theta: jax.Array) -> jax.Array:
            if not w0.legacy_tv:
                f_tilde = jnp.dot(theta, obs.norm_pixels)
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
                mean_val, count = tv_charbonnier_o1(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "tv_o2":
                mean_val, count = tv_charbonnier_o2(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "tv_exp":
                mean_val, count = tv_exp_scan(f_tilde, obs, tau=tau, epsilon=eps_gn)
                data_term = (count / b_scale) * mean_val
            elif objective == "l2v":
                mean_val, count = l2v_gap_masked(f_tilde, obs)
                data_term = (count / jnp.square(b_scale)) * mean_val
            elif objective == "l3v":
                mean_val, count = l3v_gap_masked(f_tilde, obs, epsilon=eps_gn)
                data_term = (count / (b_scale * jnp.square(b_scale))) * mean_val
            else:
                std_val, count = owl_masked(f_tilde, obs, epsilon=eps_gn)
                data_term = std_val if w0.legacy_tv else (count / b_scale) * std_val

            if w0.legacy_tv:
                return data_term
            return data_term + photon_noise_penalty(theta, obs)

        h_theta = jax.hessian(_simplex_loss)(theta0)
        damping = jnp.maximum(obs.lambda_z, 1e-3)
        eye = jnp.eye(z0.shape[0], dtype=z0.dtype)
        return jac_theta.T @ h_theta @ jac_theta + damping * eye

    return _fisher


@functools.lru_cache(maxsize=32)
def _get_lbfgs_runner(
    objective: str,
    epsilon: float,
    tau: float | None,
    rtol: float,
    atol: float,
):
    """Build and cache a JIT-compiled Optimistix BestSoFar L-BFGS solver."""
    solver = optx.BestSoFarMinimiser(optx.LBFGS(rtol=rtol, atol=atol))

    def _loss(weights: HaloWeights, obs: HaloObservation) -> jax.Array:
        return halo_objective(weights, obs, objective=objective, epsilon=epsilon, tau=tau)

    @eqx.filter_jit
    def _run(
        w0: HaloWeights,
        obs: HaloObservation,
        max_steps: int,
    ) -> tuple[HaloWeights, jax.Array, jax.Array]:
        loss0 = _loss(w0, obs)
        sol = optx.minimise(
            _loss,
            solver,
            w0,
            args=obs,
            max_steps=max_steps,
            throw=False,
        )
        loss_final = _loss(sol.value, obs)
        best_weights = jax.tree.map(
            lambda a, b: jnp.where(loss_final <= loss0, a, b),
            sol.value,
            w0,
        )
        best_loss = jnp.minimum(loss_final, loss0)
        return best_weights, loss0, best_loss

    return _run


@functools.lru_cache(maxsize=16)
def _get_zodiax_runner(
    objective: str,
    epsilon: float,
    tau: float | None,
    learning_rate: float,
    max_steps: int,
):
    """Build and cache a Tier-2 ``zodiax.optimisation`` + ``optax.adam`` runner."""

    def _loss_z(z_ilr: jax.Array, legacy_tv: bool, obs: HaloObservation) -> jax.Array:
        w = HaloWeights(z_ilr=z_ilr, legacy_tv=legacy_tv)
        return halo_objective(w, obs, objective=objective, epsilon=epsilon, tau=tau)

    @eqx.filter_jit
    def _run(
        w0: HaloWeights,
        obs: HaloObservation,
        proj: jax.Array,
    ) -> tuple[HaloWeights, jax.Array, jax.Array]:
        z0 = w0.z_ilr
        legacy_tv = w0.legacy_tv
        loss0 = _loss_z(z0, legacy_tv, obs)

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
            return _loss_z(z_curr, legacy_tv, obs)

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
            (params0, opt_state0, params0["xi"], loss0),
            None,
            length=max_steps,
        )
        end_val = _xi_loss(p_end)
        use_end = jnp.isfinite(end_val) & (end_val < best_val)
        final_xi = jnp.where(use_end, p_end["xi"], best_xi)
        final_val = jnp.where(use_end, end_val, best_val)
        z_best = z0 + jnp.dot(proj, final_xi)
        return HaloWeights(z_ilr=z_best, legacy_tv=legacy_tv), loss0, final_val

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
    """Return a boolean inlier mask using gap-segmented, cadence-scaled Savitzky-Golay detrending."""
    y = np.asarray(flux, dtype=np.float64)
    n_cad = y.shape[0]
    if n_cad < 3:
        return np.isfinite(y)

    if dt_days is None:
        if times is not None:
            t_arr = np.asarray(times, dtype=np.float64)
            diffs = np.diff(t_arr[np.isfinite(t_arr)])
            pos_diffs = diffs[diffs > 0.0]
            dt_val = float(np.median(pos_diffs)) if pos_diffs.size > 0 else (1.0 / 48.0)
        else:
            dt_val = 1.0 / 48.0
    else:
        dt_val = float(dt_days)

    target_win = cadence_scaled_savgol_window(dt_val, window_days=window_days)

    # Split Savitzky-Golay smoothing across downlink / safe-mode gaps so baseline polynomials
    # never bridge spacecraft interruptions.
    if times is not None:
        t_arr = np.asarray(times, dtype=np.float64)
        breaks = np.flatnonzero(np.diff(t_arr) > gap_factor * dt_val) + 1
        segments = np.split(np.arange(n_cad), breaks)
    else:
        segments = [np.arange(n_cad)]

    baseline = np.full_like(y, np.nan)
    for seg in segments:
        if seg.size == 0:
            continue
        fin_seg = seg[np.isfinite(y[seg])]
        if fin_seg.size >= 9:
            max_odd = fin_seg.size if fin_seg.size % 2 == 1 else fin_seg.size - 1
            win_seg = max(5, min(target_win, max_odd))
            poly_ord = 1 if win_seg < 15 else 3
            baseline[fin_seg] = savgol_filter(y[fin_seg], win_seg, poly_ord)
        elif fin_seg.size >= 1:
            baseline[fin_seg] = float(np.median(y[fin_seg]))

    inliers = np.zeros(n_cad, dtype=bool)
    fin_all = np.isfinite(y) & np.isfinite(baseline)
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
) -> tuple[HaloWeights, float, float]:
    """Run one optimization pass on ``obs`` starting from ``w0``."""
    m_minus_1 = w0.z_ilr.shape[0]
    if m_minus_1 == 0:
        loss_val = float(halo_objective(w0, obs, objective=objective, epsilon=epsilon, tau=tau))
        return w0, loss_val, loss_val

    if solver == "lbfgs":
        runner = _get_lbfgs_runner(objective, epsilon, tau, rtol, atol)
        w_best, loss0, loss_end = runner(w0, obs, maxiter)
        return w_best, float(loss0), float(loss_end)

    if solver in ("zodiax", "adam"):
        if precondition and m_minus_1 <= 256:
            fisher_fn = _get_fisher_fn(objective, tau)
            f_mat = fisher_fn(w0, obs)
            proj = safe_eigen_projection(fmat=f_mat, rcond=1e-4)
        else:
            proj = jnp.eye(m_minus_1, dtype=jnp.float64)

        runner_zdx = _get_zodiax_runner(objective, epsilon, tau, learning_rate, maxiter)
        w_best, loss0, loss_end = runner_zdx(w0, obs, proj)
        return w_best, float(loss0), float(loss_end)

    raise ValueError(f"Unknown solver {solver!r}; expected 'lbfgs', 'zodiax', or 'adam'.")


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
    maxiter: int = 151,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    learning_rate: float = 0.2,
    precondition: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate physical weight uncertainties ``sigma_w`` via ``K``-block orbit jackknife.

    Leaves out each contiguous cadence block ``B_k`` in turn while retaining
    ``obs.N_pad`` unchanged so all ``K`` warm-started solves share the
    already-compiled XLA executable with zero recompilation.
    """
    if isinstance(n_blocks, bool) or int(n_blocks) < 2:
        raise ValueError(f"n_blocks must be an integer >= 2, got {n_blocks!r}.")
    k_req = int(n_blocks)

    w_full = np.asarray(w_hat, dtype=np.float64)
    m_pix = w_full.shape[0]
    n_valid = int(obs.n_valid)
    min_req = 2 * obs.lag + 1 if objective == "tv_o2" else obs.lag + 1

    if m_pix <= 1:
        return np.zeros(m_pix, dtype=np.float64), np.tile(w_full, (k_req, 1))
    if n_valid < max(4, min_req + 2):
        return (
            np.full(m_pix, np.nan, dtype=np.float64),
            np.full((k_req, m_pix), np.nan, dtype=np.float64),
        )

    k_eff = min(k_req, n_valid // 2)
    blocks = np.array_split(np.arange(n_valid), k_eff)

    jk_weights: list[np.ndarray] = []
    for blk in blocks:
        keep = np.ones(n_valid, dtype=bool)
        keep[blk] = False
        if int(np.sum(keep)) < min_req:
            continue
        obs_k = obs.with_cadence_subset(keep, legacy_tv=legacy_tv)
        w0_k = HaloWeights.from_physical_weights(
            w_full,
            obs_k.mean_pixel_flux,
            legacy_tv=legacy_tv,
        )
        w_tree_k, _, _ = _solve_single_pass(
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
        jk_weights.append(
            np.asarray(w_tree_k.physical_weights(obs_k.mean_pixel_flux), dtype=np.float64)
        )

    if len(jk_weights) < 2:
        return (
            np.full(m_pix, np.nan, dtype=np.float64),
            np.full((k_req, m_pix), np.nan, dtype=np.float64),
        )

    w_jk = np.vstack(jk_weights)
    k_used = w_jk.shape[0]
    w_mean = np.mean(w_jk, axis=0, keepdims=True)
    var_jk = ((k_used - 1.0) / float(k_used)) * np.sum(np.square(w_jk - w_mean), axis=0)
    return np.sqrt(np.maximum(var_jk, 0.0)), w_jk


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
    maxiter: int = 151,
    rtol: float = 1e-6,
    atol: float = 1e-6,
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
    """Optimize halo aperture weights in locally scoped float64 precision."""
    if isinstance(maxiter, bool) or int(maxiter) < 1:
        raise ValueError(f"maxiter must be an integer >= 1, got {maxiter!r}.")
    maxiter_int = int(maxiter)

    pix_np = np.asarray(pixels, dtype=np.float64)
    obs = HaloObservation.from_arrays(
        pix_np,
        times,
        pixel_var=pixel_var,
        lag=lag,
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

    if w_init is None:
        w0 = HaloWeights.from_uniform(obs.mean_pixel_flux, legacy_tv=legacy_tv)
    else:
        w_arr = np.asarray(w_init, dtype=np.float64)
        if w_arr.shape != (pix_np.shape[0],):
            raise ValueError(
                f"w_init shape {w_arr.shape} does not match (M,) = ({pix_np.shape[0]},)."
            )
        if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0) or float(np.sum(w_arr)) <= 0.0:
            raise ValueError("w_init simplex weights must be finite, non-negative, and sum > 0.")
        w0 = HaloWeights.from_physical_weights(w_arr, obs.mean_pixel_flux, legacy_tv=legacy_tv)

    w_tree, loss0, loss_final = _solve_single_pass(
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

    w_phys = np.asarray(w_tree.physical_weights(obs.mean_pixel_flux), dtype=np.float64)
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
            mean_sub = np.mean(pix_np[:, inliers], axis=1)
            if legacy_tv or np.all(mean_sub > 0.0):
                obs_sub = obs.with_cadence_subset(inliers, legacy_tv=legacy_tv)
                w0_init_sub = (
                    HaloWeights.from_uniform(obs_sub.mean_pixel_flux, legacy_tv=legacy_tv)
                    if w_init is None
                    else HaloWeights.from_physical_weights(
                        np.asarray(w_init, dtype=np.float64),
                        obs_sub.mean_pixel_flux,
                        legacy_tv=legacy_tv,
                    )
                )
                loss0_sub = float(
                    halo_objective(
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
                w_tree, _, loss_final = _solve_single_pass(
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
                if loss_final > loss0_sub:
                    w_tree, _, loss_final = _solve_single_pass(
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
                    w_tree.physical_weights(obs_sub.mean_pixel_flux),
                    dtype=np.float64,
                )
                raw_lc = w_phys @ pix_np
                active_obs = obs_sub
            else:
                inliers = np.ones(n_valid, dtype=bool)
        elif np.any(~inliers):
            inliers = np.ones(n_valid, dtype=bool)

    flux_shares = np.asarray(w_tree.flux_shares(), dtype=np.float64)
    denom = float(np.dot(w_phys, np.mean(pix_np[:, inliers], axis=1)))
    safe_denom = denom if abs(denom) > 1e-30 else 1.0
    norm_lc = raw_lc / safe_denom

    weight_std: np.ndarray | None = None
    jk_weights: np.ndarray | None = None
    if compute_uncertainty:
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
            rtol=rtol,
            atol=atol,
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
    )
