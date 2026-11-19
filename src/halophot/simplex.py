"""Singularity-free O(M) isometric log-ratio (ILR) and Charnes-Cooper simplex transforms.

This module parameterizes the open probability simplex ``int(Delta^{M-1})``
without ``softmax`` translation gauge degeneracy via unconstrained coordinates
``z in R^{M-1}`` and an implicit O(M) Helmert contrast operator ``u = V z``
(``V.T @ V = I_{M-1}``, ``V.T @ 1_M = 0``), paired with the Charnes-Cooper
bijection between physical pixel weights ``w`` and flux-share weights ``theta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "flux_share_to_weights",
    "ilr_to_simplex",
    "simplex_to_ilr",
    "uniform_weights_ilr",
    "weights_to_flux_share",
    "weights_to_ilr",
]


def _ensure_floating(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return arr
    return arr.astype(jnp.result_type(arr, float))


def ilr_to_simplex(z: jax.Array) -> jax.Array:
    """Map unconstrained ILR coordinates ``z in R^{..., M-1}`` to ``theta in Delta^{M-1}``.

    Uses zero-padded suffix sums ``c_i - d_i`` with ``s_k = z_k / sqrt(k(k+1))``
    (``k = 1, ..., M-1``), which requires O(M) operations and incurs zero ``0/0``
    singularities in value, gradient, or Hessian at ``z = 0``.
    """
    z_arr = _ensure_floating(z)
    if z_arr.ndim < 1:
        raise ValueError(f"z must be at least 1D (..., M - 1), got shape {z_arr.shape}.")

    m_minus_1 = z_arr.shape[-1]
    if m_minus_1 == 0:
        return jnp.ones((*z_arr.shape[:-1], 1), dtype=z_arr.dtype)

    k = jnp.arange(1, m_minus_1 + 1, dtype=z_arr.dtype)
    s = z_arr / jnp.sqrt(k * (k + 1.0))

    suffix = jnp.cumsum(s[..., ::-1], axis=-1)[..., ::-1]
    zero_pad = jnp.zeros((*z_arr.shape[:-1], 1), dtype=z_arr.dtype)
    c = jnp.concatenate([suffix, zero_pad], axis=-1)
    d = jnp.concatenate([zero_pad, k * s], axis=-1)
    u = c - d
    return jax.nn.softmax(u, axis=-1)


def simplex_to_ilr(theta: jax.Array) -> jax.Array:
    """Map interior simplex weights ``theta in int(Delta^{M-1})`` to ``z in R^{..., M-1}``."""
    t_arr = _ensure_floating(theta)
    if t_arr.ndim < 1 or t_arr.shape[-1] < 1:
        raise ValueError(
            f"theta must have shape (..., M) with M >= 1, got shape {t_arr.shape}."
        )

    m_dim = t_arr.shape[-1]
    if m_dim == 1:
        return jnp.zeros((*t_arr.shape[:-1], 0), dtype=t_arr.dtype)

    tiny = jnp.finfo(t_arr.dtype).tiny
    log_t = jnp.log(jnp.maximum(t_arr, tiny))
    k = jnp.arange(1, m_dim, dtype=t_arr.dtype)
    prefix_sum = jnp.cumsum(log_t[..., :-1], axis=-1)
    return (prefix_sum - k * log_t[..., 1:]) / jnp.sqrt(k * (k + 1.0))


def weights_to_flux_share(w: jax.Array, mean_pixel_flux: jax.Array) -> jax.Array:
    """Charnes-Cooper forward bijection ``theta_i = w_i P_bar_i / sum_j (w_j P_bar_j)``."""
    w_arr = _ensure_floating(w)
    p_bar = jnp.asarray(mean_pixel_flux, dtype=w_arr.dtype)
    if w_arr.shape[-1:] != p_bar.shape:
        raise ValueError(
            f"Last dimension of w {w_arr.shape} must match mean_pixel_flux shape {p_bar.shape}."
        )
    tiny = jnp.finfo(w_arr.dtype).tiny
    num = jnp.maximum(w_arr, 0.0) * p_bar
    denom = jnp.maximum(jnp.sum(num, axis=-1, keepdims=True), tiny)
    return num / denom


def flux_share_to_weights(theta: jax.Array, mean_pixel_flux: jax.Array) -> jax.Array:
    """Charnes-Cooper inverse bijection ``w_i = (theta_i / P_bar_i) / sum_j (theta_j / P_bar_j)``."""
    t_arr = _ensure_floating(theta)
    p_bar = jnp.asarray(mean_pixel_flux, dtype=t_arr.dtype)
    if t_arr.shape[-1:] != p_bar.shape:
        raise ValueError(
            f"Last dimension of theta {t_arr.shape} must match mean_pixel_flux shape {p_bar.shape}."
        )
    tiny = jnp.finfo(t_arr.dtype).tiny
    safe_p = jnp.maximum(p_bar, tiny)
    unnorm = jnp.maximum(t_arr, 0.0) / safe_p
    denom = jnp.maximum(jnp.sum(unnorm, axis=-1, keepdims=True), tiny)
    return unnorm / denom


def uniform_weights_ilr(
    mean_pixel_flux: jax.Array,
    *,
    legacy_tv: bool = False,
) -> jax.Array:
    """Return ``z_init`` corresponding to uniform physical weights ``w_i = 1 / M``.

    When ``legacy_tv=False`` (default Charnes-Cooper coordinates), uniform ``w``
    corresponds to ``theta_init = P_bar / sum(P_bar)``, so ``z_init = simplex_to_ilr(theta_init)``
    avoids the ``z = 0 => w_i propto 1 / P_bar_i`` inverse-flux bias. When
    ``legacy_tv=True`` (direct ``w``-simplex parameterization), ``theta = w`` and
    ``z_init = 0``.
    """
    p_bar = _ensure_floating(mean_pixel_flux)
    if p_bar.ndim != 1 or p_bar.shape[0] < 1:
        raise ValueError(
            f"mean_pixel_flux must be 1D (M,) with M >= 1, got shape {p_bar.shape}."
        )
    m_dim = p_bar.shape[0]
    if m_dim == 1 or legacy_tv:
        return jnp.zeros(m_dim - 1, dtype=p_bar.dtype)

    tiny = jnp.finfo(p_bar.dtype).tiny
    safe_p = jnp.maximum(p_bar, tiny)
    theta_init = safe_p / jnp.sum(safe_p)
    return simplex_to_ilr(theta_init)


def weights_to_ilr(
    w: jax.Array,
    mean_pixel_flux: jax.Array,
    *,
    legacy_tv: bool = False,
) -> jax.Array:
    """Convert physical simplex weights ``w in Delta^{M-1}`` to ILR coordinates ``z``."""
    w_arr = _ensure_floating(w)
    p_bar = jnp.asarray(mean_pixel_flux, dtype=w_arr.dtype)
    if w_arr.ndim != 1 or w_arr.shape != p_bar.shape or w_arr.shape[0] < 1:
        raise ValueError(
            f"w shape {w_arr.shape} and mean_pixel_flux shape {p_bar.shape} must match (M,)."
        )
    if w_arr.shape[0] == 1:
        return jnp.zeros(0, dtype=w_arr.dtype)

    w_norm = jnp.maximum(w_arr, 0.0)
    w_norm = w_norm / jnp.maximum(jnp.sum(w_norm), jnp.finfo(w_arr.dtype).tiny)
    target_simplex = w_norm if legacy_tv else weights_to_flux_share(w_norm, p_bar)
    # Guard exact boundary zeros to machine epsilon while preserving sparse warm starts.
    floor = jnp.asarray(1e-15, dtype=w_arr.dtype)
    interior = jnp.maximum(target_simplex, floor)
    interior = interior / jnp.sum(interior)
    return simplex_to_ilr(interior)
