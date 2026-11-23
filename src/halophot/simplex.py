"""Direct symmetric softmax parameterization for physical simplex weights.

Physical aperture weights ``w in Delta^{M-1}`` (``w_i >= 0``, ``sum_i w_i = 1``)
are parameterized directly via zero-mean logits::

    u = logits - mean(logits)
    w = jax.nn.softmax(u)

This removes the softmax translation gauge symmetrically, starts uniformly at
``logits = 0``, and works seamlessly even when individual outer-halo pixels
have negative background-subtracted mean flux.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._precision import in_float64

__all__ = [
    "initial_logits",
    "logits_to_weights",
    "weights_to_logits",
]


def logits_to_weights(logits: jax.Array | np.ndarray) -> jax.Array:
    """Map unconstrained logits ``(..., M)`` to physical simplex weights ``w in Delta^{M-1}``."""
    x = jnp.asarray(logits)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError(f"logits must have shape (..., M) with M >= 1, got {x.shape}.")
    u = x - jnp.mean(x, axis=-1, keepdims=True)
    return jax.nn.softmax(u, axis=-1)


def weights_to_logits(
    weights: jax.Array | np.ndarray,
    *,
    floor: float = 1e-15,
) -> jax.Array:
    """Map physical simplex weights ``w in Delta^{M-1}`` to zero-mean logits."""
    w = jnp.asarray(weights)
    if w.ndim < 1 or w.shape[-1] < 1:
        raise ValueError(f"weights must have shape (..., M) with M >= 1, got {w.shape}.")
    w_pos = jnp.maximum(w, 0.0)
    w_norm = w_pos / jnp.maximum(jnp.sum(w_pos, axis=-1, keepdims=True), jnp.finfo(w.dtype).tiny)
    log_w = jnp.log(jnp.maximum(w_norm, floor))
    return log_w - jnp.mean(log_w, axis=-1, keepdims=True)


@in_float64
def initial_logits(
    mean_pixel_flux: jax.Array | np.ndarray,
    w_init: jax.Array | np.ndarray | None = None,
) -> jax.Array:
    """Return starting zero-mean logits (uniform ``0`` unless ``w_init`` is supplied).

    When ``w_init`` is omitted and mixed-sign sky pixels pull the uniform mean
    aperture flux ``<= 0``, initializes uniformly across positive-mean pixels so
    the starting light curve has strictly positive mean.
    """
    p_bar = np.asarray(mean_pixel_flux, dtype=np.float64)
    if p_bar.ndim != 1 or p_bar.shape[0] < 1:
        raise ValueError(f"mean_pixel_flux must be 1D (M,) with M >= 1, got {p_bar.shape}.")
    m_pix = p_bar.shape[0]

    if w_init is not None:
        w_arr = np.asarray(w_init, dtype=np.float64)
        if w_arr.shape != (m_pix,):
            raise ValueError(f"w_init shape {w_arr.shape} does not match (M,) = ({m_pix},).")
        if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0) or float(np.sum(w_arr)) <= 0.0:
            raise ValueError("w_init simplex weights must be finite, non-negative, and sum > 0.")
        return weights_to_logits(jnp.asarray(w_arr, dtype=jnp.float64))

    if float(np.mean(p_bar)) <= 0.0 and bool(np.any(p_bar > 0.0)):
        pos = p_bar > 0.0
        w0 = np.where(pos, 1.0 / float(np.sum(pos)), 1e-12)
        return weights_to_logits(jnp.asarray(w0, dtype=jnp.float64))

    return jnp.zeros(m_pix, dtype=jnp.float64)
