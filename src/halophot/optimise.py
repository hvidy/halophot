"""Halo weight optimization via ``optax.lbfgs``, spike clipping, and orbit block jackknife."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from astropy.stats import mad_std, sigma_clip

from ._precision import in_float64
from .objectives import compute_step_mask, normalized_flux, resolve_objective
from .simplex import initial_logits, logits_to_weights

__all__ = [
    "HaloFitResult",
    "OptimConfig",
    "cadence_scaled_savgol_window",
    "cadence_scaled_sigclip_mask",
    "optimise_halo",
    "orbit_block_jackknife",
]


@dataclass(frozen=True)
class OptimConfig:
    """Configuration for :func:`optimise_halo`."""

    objective: str | Callable[..., Any] = "tv"
    lag: int = 1
    tau: float | None = None
    epsilon: float = 1e-2
    maxiter: int = 500
    rtol: float = 1e-7
    atol: float = 1e-12
    precondition: bool = True
    sigclip: bool = False
    window_days: float = 0.02
    sigma_lower: float = 10.0
    sigma_upper: float = 3.0
    compute_uncertainty: bool = False
    jackknife_blocks: int = 4
    gap_tol: float = 0.25


@dataclass(frozen=True)
class HaloFitResult:
    """Result of :func:`optimise_halo` on active halo pixels."""

    weights: np.ndarray
    norm_flux: np.ndarray
    raw_flux: np.ndarray
    inlier_mask: np.ndarray
    loss_init: float
    loss_final: float
    weight_std: np.ndarray | None = None
    jackknife_weights: np.ndarray | None = None
    converged: bool = True
    n_steps: int = 0

    @property
    def physical_weights(self) -> np.ndarray:
        """Alias for :attr:`weights` (v2.0 compatibility)."""
        return self.weights


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
    window_days: float = 0.02,
    sigma_lower: float = 10.0,
    sigma_upper: float = 3.0,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Return a boolean inlier mask using a gap-aware local leave-one-out median detector.

    Scales the local half-width to ``window_days`` via
    ``half_width = clamp(round(0.5 * window_days / dt_days), 3, 15)`` so a
    default ``window_days = 0.02`` (~29 min) yields ``half_width = 15`` at 20 s
    (catching 5-cadence 100 s bursts cleanly) and ``half_width = 7`` at 2 min,
    while preserving multi-hour eclipses and never bridging downlink gaps.
    """
    y = np.asarray(flux, dtype=np.float64)
    n_cad = y.shape[0]
    t_arr = np.asarray(times, dtype=np.float64) if times is not None else None
    fin_mask = np.isfinite(y) if t_arr is None else (np.isfinite(y) & np.isfinite(t_arr))

    if n_cad < 3:
        return fin_mask.copy()

    fin_idx = np.flatnonzero(fin_mask)
    if fin_idx.size == 0:
        return np.zeros(n_cad, dtype=bool)

    if dt_days is None:
        if t_arr is not None:
            diffs = np.diff(t_arr[fin_idx])
            pos = diffs[diffs > 0.0]
            dt_val = float(np.median(pos)) if pos.size > 0 else (1.0 / 48.0)
        else:
            dt_val = 1.0 / 48.0
    else:
        dt_val = float(dt_days)

    if not np.isfinite(dt_val) or dt_val <= 0.0:
        raise ValueError(f"dt_days must be finite and > 0, got {dt_val!r}.")
    if not np.isfinite(window_days) or window_days <= 0.0:
        raise ValueError(f"window_days must be finite and > 0, got {window_days!r}.")

    hw = max(3, min(15, int(round(0.5 * float(window_days) / dt_val))))
    nbr_offsets = np.concatenate([np.arange(-hw, 0, dtype=np.intp), np.arange(1, hw + 1, dtype=np.intp)])

    if t_arr is not None:
        breaks = np.flatnonzero(np.diff(t_arr[fin_idx]) > gap_factor * dt_val) + 1
    else:
        breaks = np.flatnonzero(np.diff(fin_idx) > 1) + 1
    segments = np.split(fin_idx, breaks)

    baseline = np.full(n_cad, np.nan, dtype=np.float64)
    for seg in segments:
        seg_len = seg.size
        if seg_len == 0:
            continue
        ys = y[seg]
        if seg_len < 5:
            baseline[seg] = float(np.median(ys))
            continue

        b_seg = np.empty(seg_len, dtype=np.float64)
        hw_seg = min(hw, (seg_len - 1) // 2)
        offsets_seg = (
            nbr_offsets
            if hw_seg == hw
            else np.concatenate([np.arange(-hw_seg, 0, dtype=np.intp), np.arange(1, hw_seg + 1, dtype=np.intp)])
        )
        centres = np.arange(hw_seg, seg_len - hw_seg, dtype=np.intp)
        if centres.size > 0:
            b_seg[centres] = np.median(ys[centres[:, None] + offsets_seg[None, :]], axis=1)

        # Robust local Theil-Sen linear prediction near segment boundaries.
        ts = t_arr[seg] if t_arr is not None else np.arange(seg_len, dtype=np.float64)
        edge_win = min(seg_len, max(7, 2 * hw_seg + 1))
        edge_indices = list(range(hw_seg)) + list(range(seg_len - hw_seg, seg_len))
        for j in edge_indices:
            lo = max(0, min(j - edge_win // 2, seg_len - edge_win))
            hi = lo + edge_win
            others = np.array([k for k in range(lo, hi) if k != j], dtype=np.intp)
            to = ts[others]
            yo = ys[others]
            dt_mat = to[:, None] - to[None, :]
            dy_mat = yo[:, None] - yo[None, :]
            upper = np.triu(np.abs(dt_mat) > 0.0, k=1)
            slope = float(np.median(dy_mat[upper] / dt_mat[upper])) if np.any(upper) else 0.0
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


def _estimate_initial_scale(
    f0: np.ndarray,
    step_mask: np.ndarray,
    *,
    obj_name: str,
    lag: int,
) -> float:
    """Compute characteristic scale ``b_hat = max(median(|df_init|), 1e-30)`` with no floor bias."""
    if obj_name == "owl":
        val = float(np.std(f0))
    elif obj_name == "tv_o2" and 2 * lag < f0.shape[0]:
        mask2 = step_mask[:-lag] & step_mask[lag:]
        diffs2 = np.abs(2.0 * f0[lag:-lag] - f0[2 * lag :] - f0[: -2 * lag])
        sel = diffs2[mask2] if np.any(mask2) else diffs2
        val = float(np.median(sel)) if sel.size > 0 else 0.0
    elif lag < f0.shape[0]:
        diffs = np.abs(f0[lag:] - f0[:-lag])
        sel = diffs[step_mask] if np.any(step_mask) else diffs
        val = float(np.median(sel)) if sel.size > 0 else 0.0
    else:
        val = 0.0
    return max(val, 1e-30) if np.isfinite(val) else 1e-30


def _compute_diag_scale(
    w0: np.ndarray,
    pixels: np.ndarray,
    step_mask: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Compute O(M) diagonal pixel-scale preconditioning weights on simplex logits."""
    raw0 = w0 @ pixels
    mean0 = float(np.mean(raw0))
    safe0 = mean0 if abs(mean0) > 1e-30 else 1.0
    f0 = raw0 / safe0
    p_eff = pixels / safe0

    if lag < pixels.shape[1] and np.any(step_mask):
        df0 = f0[lag:] - f0[:-lag]
        dp = (p_eff[:, lag:] - p_eff[:, :-lag]) - df0[None, :]
        dp_valid = dp[:, step_mask]
    else:
        df0 = f0[1:] - f0[:-1]
        dp_valid = (p_eff[:, 1:] - p_eff[:, :-1]) - df0[None, :]

    mean_dp_sq = np.mean(np.square(dp_valid), axis=1)
    curv = np.square(w0) * mean_dp_sq
    med_curv = max(float(np.median(curv)), 1e-30)
    d_raw = 1.0 / np.sqrt(np.maximum(curv, 1e-6 * med_curv))
    d_med = max(float(np.median(d_raw)), 1e-30)
    return np.clip(d_raw / d_med, 0.1, 10.0)


def _run_optax_lbfgs_loop(
    val_fn: Callable[[jax.Array], jax.Array],
    weights_from_xi: Callable[[jax.Array], jax.Array],
    xi0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run ``optax.lbfgs`` + zoom linesearch inside ``jax.lax.while_loop``.

    Scales the gradient convergence threshold strictly to ``max(|v0|, ||g0||_inf)``
    (never floored at ``1.0``) so small-amplitude light curves never trigger
    step-0 false convergence.
    """
    opt = optax.lbfgs(
        memory_size=20,
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20, verbose=False),
    )
    vg_fn = jax.value_and_grad(val_fn)
    v0, g0 = vg_fn(xi0)
    opt_state0 = opt.init(xi0)
    w0 = weights_from_xi(xi0)

    g0_inf = jnp.max(jnp.abs(g0))
    scale0 = jnp.maximum(jnp.abs(v0), g0_inf)
    grad_tol = atol + rtol * scale0
    finite0 = jnp.isfinite(v0) & jnp.all(jnp.isfinite(g0))
    conv0 = finite0 & (g0_inf <= grad_tol)
    failed0 = ~finite0

    init_carry = (
        xi0,
        opt_state0,
        v0,
        g0,
        w0,
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
        g_inf = jnp.max(jnp.abs(g_next))
        v_scale = jnp.maximum(jnp.maximum(jnp.abs(v_curr), 1e-3 * jnp.abs(v0)), 1e-30)
        grad_ok = (
            finite_step
            & (v_next <= next_best_v + 1e-12 * v_scale)
            & (g_inf <= grad_tol)
        )

        dv_rel = jnp.abs(v_curr - v_next) / v_scale
        ftol = jnp.maximum(500.0 * rtol, 1e-15)
        step_small = (
            finite_step
            & (v_next <= v_curr + 1e-12 * v_scale)
            & (dv_rel <= ftol)
        )
        consec_next = jnp.where(step_small, consec + 1, jnp.zeros_like(consec))
        conv_next = grad_ok | ((step >= 15) & (consec_next >= 5))
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


def _validate_inputs(
    pixels: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array | None,
    lag: int,
    obj_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    n_cad = pix_np.shape[1]

    min_req = 2 * lag_int + 1 if obj_name == "tv_o2" else (2 if obj_name == "owl" else lag_int + 1)
    if n_cad < min_req:
        raise ValueError(
            f"Objective {obj_name!r} with lag={lag_int} requires at least {min_req} cadences, "
            f"got {n_cad}."
        )

    if times is None:
        t_np = np.arange(n_cad, dtype=np.float64) * (1.0 / 48.0)
    else:
        t_np = np.asarray(times, dtype=np.float64)
        if t_np.shape != (n_cad,):
            raise ValueError(f"times shape {t_np.shape} does not match ({n_cad},).")
        if not np.all(np.isfinite(t_np)):
            raise ValueError("times contains non-finite timestamps.")

    diffs = np.diff(t_np)
    if np.any(diffs <= 0.0):
        raise ValueError("times must be strictly increasing.")
    dt_nom = float(np.median(diffs))
    return pix_np, t_np, dt_nom


def _solve_single_pass(
    pix_np: np.ndarray,
    t_np: np.ndarray,
    dt_nom: float,
    *,
    w_init: np.ndarray | None,
    obj_name: str,
    obj_fn: Callable[..., jax.Array],
    lag: int,
    tau: float | None,
    epsilon: float,
    maxiter: int,
    rtol: float,
    atol: float,
    precondition: bool,
    gap_tol: float,
) -> tuple[np.ndarray, float, float, bool, int]:
    """Run one ``optax.lbfgs`` pass in locally scoped float64 precision."""
    m_pix = pix_np.shape[0]
    step_mask_np = compute_step_mask(t_np, lag=lag, dt_nom=dt_nom, gap_tol=gap_tol)
    u0 = initial_logits(np.mean(pix_np, axis=1), w_init=w_init)
    w0_np = np.asarray(logits_to_weights(u0), dtype=np.float64)

    raw0 = w0_np @ pix_np
    mean0 = float(np.mean(raw0))
    f0_np = raw0 / (mean0 if abs(mean0) > 1e-30 else 1.0)
    b_hat = _estimate_initial_scale(f0_np, step_mask_np, obj_name=obj_name, lag=lag)
    eps_eff = max(float(epsilon) * b_hat, float(np.finfo(np.float64).tiny))

    pix_j = jnp.asarray(pix_np, dtype=jnp.float64)
    t_j = jnp.asarray(t_np, dtype=jnp.float64)
    mask_j = jnp.asarray(step_mask_np, dtype=jnp.bool_)
    dt_j = jnp.asarray(dt_nom, dtype=jnp.float64)
    eps_j = jnp.asarray(eps_eff, dtype=jnp.float64)

    def _eval_weights(w: jax.Array) -> jax.Array:
        flux = normalized_flux(w, pix_j)
        return obj_fn(
            flux=flux,
            step_mask=mask_j,
            weights=w,
            pixels=pix_j,
            times=t_j,
            dt_nom=dt_j,
            lag=lag,
            eps=eps_j,
            tau=tau,
            gap_tol=gap_tol,
        )

    if m_pix == 1:
        w_one = jnp.ones(1, dtype=jnp.float64)
        loss_val = float(_eval_weights(w_one))
        return np.ones(1, dtype=np.float64), loss_val, loss_val, True, 0

    d_scale = (
        jnp.asarray(_compute_diag_scale(w0_np, pix_np, step_mask_np, lag), dtype=jnp.float64)
        if precondition
        else jnp.ones(m_pix, dtype=jnp.float64)
    )

    @jax.jit
    def _run_jit(u_start: jax.Array, scale: jax.Array):
        def _w_from_xi(xi: jax.Array) -> jax.Array:
            return logits_to_weights(u_start + scale * xi)

        def _val_fn(xi: jax.Array) -> jax.Array:
            return _eval_weights(_w_from_xi(xi))

        xi0 = jnp.zeros_like(u_start)
        best_xi, v0, best_v, conv, n_steps = _run_optax_lbfgs_loop(
            _val_fn, _w_from_xi, xi0, maxiter, rtol, atol
        )
        return _w_from_xi(best_xi), v0, best_v, conv, n_steps

    w_opt, v0, v_final, conv, n_steps = _run_jit(u0, d_scale)
    return (
        np.asarray(w_opt, dtype=np.float64),
        float(v0),
        float(v_final),
        bool(conv),
        int(n_steps),
    )


def _partition_orbit_blocks(
    times_valid: np.ndarray,
    dt_nom: float,
    k_eff: int,
    min_req: int,
) -> list[np.ndarray]:
    """Split active cadence indices into ``k_eff`` blocks respecting major orbit gaps."""
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
                extra[np.argsort(-(raw_extra - extra))[:leftover]] += 1
            counts += extra
        blocks = [
            sub_blk
            for seg, cnt in zip(segs, counts, strict=True)
            for sub_blk in np.array_split(seg, min(int(cnt), seg.size))
            if sub_blk.size > 0
        ]
        if len(blocks) == k_eff and all((n_valid - b.size) >= min_req for b in blocks):
            return blocks

    return [blk for blk in np.array_split(np.arange(n_valid), k_eff) if blk.size > 0]


@in_float64
def orbit_block_jackknife(
    pixels: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array | None,
    w_hat: np.ndarray | jax.Array,
    *,
    n_blocks: int = 4,
    objective: str | Callable[..., Any] = "tv",
    lag: int = 1,
    tau: float | None = None,
    epsilon: float = 1e-2,
    maxiter: int = 500,
    rtol: float = 1e-7,
    atol: float = 1e-12,
    precondition: bool = True,
    gap_tol: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate physical weight uncertainties via delete-``m_k`` orbit block jackknife.

    Uses Busing-Meijer-van der Leeden delete-``m_k`` variance weighting on
    warm-started leave-one-block-out fits respecting major orbit/downlink gaps.
    """
    if isinstance(n_blocks, bool) or int(n_blocks) < 2:
        raise ValueError(f"n_blocks must be an integer >= 2, got {n_blocks!r}.")
    k_req = int(n_blocks)

    obj_name, obj_fn = resolve_objective(objective)
    pix_np, t_np, dt_nom = _validate_inputs(pixels, times, lag, obj_name)
    w_full = np.asarray(w_hat, dtype=np.float64)
    m_pix, n_valid = pix_np.shape
    min_req = 2 * lag + 1 if obj_name == "tv_o2" else (2 if obj_name == "owl" else lag + 1)

    if m_pix <= 1:
        return np.zeros(m_pix, dtype=np.float64), np.tile(w_full, (k_req, 1))
    if n_valid < max(4, min_req + 2):
        return (
            np.full(m_pix, np.nan, dtype=np.float64),
            np.full((k_req, m_pix), np.nan, dtype=np.float64),
        )

    k_eff = min(k_req, n_valid // 2)
    blocks = _partition_orbit_blocks(t_np, dt_nom, k_eff, min_req)
    full_steps = max(int(np.sum(compute_step_mask(t_np, lag=lag, dt_nom=dt_nom, gap_tol=gap_tol))), 1)

    jk_weights: list[np.ndarray] = []
    dropped_counts: list[float] = []
    n_unconverged = 0
    for blk in blocks:
        keep = np.ones(n_valid, dtype=bool)
        keep[blk] = False
        if int(np.sum(keep)) < min_req:
            continue
        t_k = t_np[keep]
        w_k, _, _, conv_k, _ = _solve_single_pass(
            pix_np[:, keep],
            t_k,
            dt_nom,
            w_init=w_full,
            obj_name=obj_name,
            obj_fn=obj_fn,
            lag=lag,
            tau=tau,
            epsilon=epsilon,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            precondition=precondition,
            gap_tol=gap_tol,
        )
        if not conv_k:
            n_unconverged += 1
        jk_weights.append(w_k)
        steps_k = int(np.sum(compute_step_mask(t_k, lag=lag, dt_nom=dt_nom, gap_tol=gap_tol)))
        dropped_counts.append(float(max(full_steps - steps_k, 1)))

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


@in_float64
def optimise_halo(
    pixels: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array | None = None,
    *,
    w_init: np.ndarray | jax.Array | None = None,
    config: OptimConfig | None = None,
    objective: str | Callable[..., Any] | None = None,
    lag: int | None = None,
    tau: float | None = None,
    epsilon: float | None = None,
    maxiter: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    precondition: bool | None = None,
    sigclip: bool | None = None,
    window_days: float | None = None,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    compute_uncertainty: bool | None = None,
    jackknife_blocks: int | None = None,
    gap_tol: float | None = None,
    legacy_tv: bool | None = None,
) -> HaloFitResult:
    """Optimize physical simplex halo weights in locally scoped float64 precision.

    Parameters
    ----------
    pixels : array-like, shape ``(M, N)``
        Active halo pixel fluxes over valid cadences.
    times : array-like, shape ``(N,)``, optional
        Strictly increasing timestamps in days. Defaults to uniform 30-min cadence.
    w_init : array-like, shape ``(M,)``, optional
        Starting physical simplex weights (default uniform ``1 / M``).
    config : OptimConfig, optional
        Structured optimization settings; explicit keyword arguments override fields.
    objective : str or callable, default ``'tv'``
        Registered objective name in :data:`~halophot.OBJECTIVES` or any custom
        JAX callable (``def fn(flux): ...``, ``def fn(flux, step_mask): ...``, or
        keyword-aware ``(flux, *, step_mask, weights, pixels, times, dt_nom, lag)``).
    sigclip : bool, default False
        When ``True``, excludes local flux outliers strictly during weight
        optimization and baseline normalization while evaluating ``raw_flux`` and
        ``norm_flux`` on all input cadences and recording ``inlier_mask``.
    legacy_tv : bool, optional
        Deprecated compatibility no-op (physical simplex weights are always used).
    """
    if legacy_tv is not None:
        warnings.warn(
            "legacy_tv is deprecated and has no effect; halophot now optimizes "
            "physical simplex weights directly.",
            DeprecationWarning,
            stacklevel=3,
        )

    cfg = config if config is not None else OptimConfig()
    eff_obj = cfg.objective if objective is None else objective
    eff_lag = cfg.lag if lag is None else int(lag)
    eff_tau = cfg.tau if tau is None else tau
    eff_eps = cfg.epsilon if epsilon is None else float(epsilon)
    eff_maxiter = cfg.maxiter if maxiter is None else int(maxiter)
    eff_rtol = cfg.rtol if rtol is None else float(rtol)
    eff_atol = cfg.atol if atol is None else float(atol)
    eff_precond = cfg.precondition if precondition is None else bool(precondition)
    eff_sigclip = cfg.sigclip if sigclip is None else bool(sigclip)
    eff_win = cfg.window_days if window_days is None else float(window_days)
    eff_sig_lo = cfg.sigma_lower if sigma_lower is None else float(sigma_lower)
    eff_sig_hi = cfg.sigma_upper if sigma_upper is None else float(sigma_upper)
    eff_unc = cfg.compute_uncertainty if compute_uncertainty is None else bool(compute_uncertainty)
    eff_jk = cfg.jackknife_blocks if jackknife_blocks is None else int(jackknife_blocks)
    eff_gap = cfg.gap_tol if gap_tol is None else float(gap_tol)

    if isinstance(eff_maxiter, bool) or eff_maxiter < 1:
        raise ValueError(f"maxiter must be an integer >= 1, got {eff_maxiter!r}.")
    if eff_eps <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {eff_eps!r}.")

    obj_name, obj_fn = resolve_objective(eff_obj)
    pix_np, t_np, dt_nom = _validate_inputs(pixels, times, eff_lag, obj_name)
    m_pix, n_valid = pix_np.shape
    min_req = 2 * eff_lag + 1 if obj_name == "tv_o2" else (2 if obj_name == "owl" else eff_lag + 1)

    n_valid_steps = int(np.sum(compute_step_mask(t_np, lag=eff_lag, dt_nom=dt_nom, gap_tol=eff_gap)))
    if m_pix > 1 and n_valid_steps > 0 and m_pix >= 0.5 * n_valid_steps:
        warnings.warn(
            f"Active halo pixel count M={m_pix} approaches or exceeds valid cadence "
            f"constraints N_valid={n_valid_steps} (M >= 0.5 * N_valid).",
            UserWarning,
            stacklevel=3,
        )

    w_init_np = None if w_init is None else np.asarray(w_init, dtype=np.float64)
    w_opt, loss0, loss_final, converged, n_steps = _solve_single_pass(
        pix_np,
        t_np,
        dt_nom,
        w_init=w_init_np,
        obj_name=obj_name,
        obj_fn=obj_fn,
        lag=eff_lag,
        tau=eff_tau,
        epsilon=eff_eps,
        maxiter=eff_maxiter,
        rtol=eff_rtol,
        atol=eff_atol,
        precondition=eff_precond,
        gap_tol=eff_gap,
    )

    raw_lc = w_opt @ pix_np
    inliers = np.ones(n_valid, dtype=bool)
    fit_pix = pix_np
    fit_times = t_np

    if eff_sigclip:
        inliers = cadence_scaled_sigclip_mask(
            raw_lc,
            t_np,
            dt_days=dt_nom,
            window_days=eff_win,
            sigma_lower=eff_sig_lo,
            sigma_upper=eff_sig_hi,
        )
        if np.any(~inliers) and int(np.sum(inliers)) >= min_req:
            fit_pix = pix_np[:, inliers]
            fit_times = t_np[inliers]
            w_opt, loss0, loss_final, converged, n_steps = _solve_single_pass(
                fit_pix,
                fit_times,
                dt_nom,
                w_init=w_opt,
                obj_name=obj_name,
                obj_fn=obj_fn,
                lag=eff_lag,
                tau=eff_tau,
                epsilon=eff_eps,
                maxiter=eff_maxiter,
                rtol=eff_rtol,
                atol=eff_atol,
                precondition=eff_precond,
                gap_tol=eff_gap,
            )
            raw_lc = w_opt @ pix_np
        elif np.any(~inliers):
            inliers = np.ones(n_valid, dtype=bool)

    if m_pix > 1 and not converged:
        warnings.warn(
            f"L-BFGS solver did not converge within maxiter={eff_maxiter} steps "
            f"(completed {n_steps} steps); consider increasing maxiter.",
            RuntimeWarning,
            stacklevel=3,
        )

    denom = float(np.mean(raw_lc[inliers]))
    safe_denom = denom if abs(denom) > 1e-30 else 1.0
    norm_lc = raw_lc / safe_denom

    weight_std: np.ndarray | None = None
    jk_weights: np.ndarray | None = None
    if eff_unc:
        weight_std, jk_weights = orbit_block_jackknife(
            fit_pix,
            fit_times,
            w_opt,
            n_blocks=eff_jk,
            objective=eff_obj,
            lag=eff_lag,
            tau=eff_tau,
            epsilon=eff_eps,
            maxiter=eff_maxiter,
            rtol=eff_rtol,
            atol=eff_atol,
            precondition=eff_precond,
            gap_tol=eff_gap,
        )

    return HaloFitResult(
        weights=w_opt,
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
