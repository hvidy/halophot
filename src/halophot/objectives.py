"""Built-in and custom JAX objective functions for halo photometry.

All built-in objectives operate directly on the unit-mean 1D light curve
``flux = (w @ pixels) / mean(w @ pixels)`` over valid cadences, with gap-safe
finite-difference masks so spacecraft downlink and safe-mode gaps are never
bridged.

Writing a custom objective
--------------------------
Pass any JAX function directly as ``objective=my_fn`` to
:func:`~halophot.optimise_halo` or :func:`~halophot.extract_halo_lightcurve`,
or register it globally with ``@halophot.register_objective("my_tv")``.
Supported signatures are inspected automatically before JIT compilation:

* ``def my_fn(flux): ...`` — 1D unit-mean light curve on valid cadences.
* ``def my_fn(flux, step_mask): ...`` — light curve plus boolean gap-safe
  difference mask of shape ``(N - lag,)``.
* Keyword-aware functions accepting any subset of
  ``(flux, *, step_mask, weights, pixels, times, dt_nom, lag)``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "OBJECTIVES",
    "charbonnier_abs",
    "compute_step_mask",
    "l2v",
    "l3v",
    "normalized_flux",
    "owl",
    "register_objective",
    "resolve_objective",
    "tv_exp",
    "tv_o1",
    "tv_o2",
]

OBJECTIVES: dict[str, Callable[..., jax.Array]] = {}

_CONTEXT_KEYS: tuple[str, ...] = (
    "flux",
    "step_mask",
    "weights",
    "pixels",
    "times",
    "dt_nom",
    "lag",
    "eps",
    "tau",
    "gap_tol",
)


def register_objective(
    name: str,
) -> Callable[[Callable[..., jax.Array]], Callable[..., jax.Array]]:
    """Register a JAX objective callable under ``name`` in :data:`OBJECTIVES`."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Objective name must be a non-empty string, got {name!r}.")

    def _decorator(fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
        if not callable(fn):
            raise TypeError(f"Registered objective {name!r} must be callable.")
        OBJECTIVES[name] = fn
        return fn

    return _decorator


def normalized_flux(weights: jax.Array, pixels: jax.Array) -> jax.Array:
    """Compute the unit-mean weighted light curve ``(w @ pixels) / mean(w @ pixels)``."""
    raw_flux = jnp.dot(weights, pixels)
    mean_f = jnp.mean(raw_flux)
    safe_mean = jnp.where(jnp.abs(mean_f) > 1e-30, mean_f, 1.0)
    return raw_flux / safe_mean


def charbonnier_abs(x: jax.Array, eps: float | jax.Array) -> jax.Array:
    """Cancellation-free smooth L1 penalty ``x^2 / (sqrt(x^2 + eps^2) + eps)``.

    Algebraically equal to ``sqrt(x^2 + eps^2) - eps`` in exact arithmetic, but
    avoids floating-point cancellation for ``|x| << eps`` and has finite
    derivatives of all orders at ``x = 0``.
    """
    x_arr = jnp.asarray(x)
    eps_safe = jnp.maximum(jnp.asarray(eps, dtype=x_arr.dtype), jnp.finfo(x_arr.dtype).tiny)
    x2 = jnp.square(x_arr)
    return x2 / (jnp.sqrt(x2 + eps_safe * eps_safe) + eps_safe)


def compute_step_mask(
    times: np.ndarray,
    lag: int = 1,
    dt_nom: float | None = None,
    gap_tol: float = 0.25,
) -> np.ndarray:
    """Return boolean gap-safe mask of shape ``(N - lag,)`` for lag-``lag`` differences."""
    t_arr = np.asarray(times, dtype=np.float64)
    n_cad = t_arr.shape[0]
    if lag < 1 or lag >= n_cad:
        return np.zeros(0, dtype=bool)
    if math.isinf(float(gap_tol)):
        return np.ones(n_cad - lag, dtype=bool)
    if dt_nom is None:
        diffs = np.diff(t_arr)
        pos = diffs[diffs > 0.0]
        dt_val = float(np.median(pos)) if pos.size > 0 else (1.0 / 48.0)
    else:
        dt_val = float(dt_nom)
    dt_pair = t_arr[lag:] - t_arr[:-lag]
    return np.abs(dt_pair - float(lag) * dt_val) <= float(gap_tol) * dt_val


def _masked_mean(values: jax.Array, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return jnp.mean(values) if values.shape[0] > 0 else jnp.zeros((), dtype=values.dtype)
    count = jnp.sum(mask.astype(values.dtype))
    total = jnp.sum(jnp.where(mask, values, 0.0))
    return jnp.where(count > 0.0, total / jnp.maximum(count, 1.0), 0.0)


@register_objective("tv")
def tv_o1(
    flux: jax.Array,
    step_mask: jax.Array | None = None,
    *,
    lag: int = 1,
    eps: float | jax.Array = 1e-5,
) -> jax.Array:
    """1st-order gap-safe Charbonnier total variation."""
    if lag >= flux.shape[0]:
        return jnp.zeros((), dtype=flux.dtype)
    diffs = flux[lag:] - flux[:-lag]
    return _masked_mean(charbonnier_abs(diffs, eps), step_mask)


@register_objective("tv_o2")
def tv_o2(
    flux: jax.Array,
    step_mask: jax.Array | None = None,
    *,
    lag: int = 1,
    eps: float | jax.Array = 1e-5,
) -> jax.Array:
    """2nd-order gap-safe Charbonnier total variation."""
    if 2 * lag >= flux.shape[0]:
        return jnp.zeros((), dtype=flux.dtype)
    diffs2 = 2.0 * flux[lag:-lag] - flux[2 * lag :] - flux[: -2 * lag]
    mask2 = None if step_mask is None else (step_mask[:-lag] & step_mask[lag:])
    return _masked_mean(charbonnier_abs(diffs2, eps), mask2)


@register_objective("tv_exp")
def tv_exp(
    flux: jax.Array,
    *,
    times: jax.Array | None = None,
    dt_nom: float | jax.Array | None = None,
    lag: int = 1,
    tau: float | None = None,
    eps: float | jax.Array = 1e-5,
    gap_tol: float = 0.25,
) -> jax.Array:
    """Multi-lag exponentially weighted TV evaluated in O(N) memory via ``jax.lax.scan``."""
    tau_val = float(lag) if tau is None else float(tau)
    if tau_val <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau_val!r}.")
    n_cad = flux.shape[0]
    scan_len = min(max(int(math.ceil(4.0 * tau_val)), 1), n_cad - 1)
    if scan_len < 1:
        return jnp.zeros((), dtype=flux.dtype)

    idx = jnp.arange(n_cad, dtype=jnp.int32)
    tau_arr = jnp.asarray(tau_val, dtype=flux.dtype)
    check_gaps = times is not None and dt_nom is not None and not math.isinf(float(gap_tol))

    @jax.checkpoint
    def _scan_step(carry: tuple[jax.Array, jax.Array], ell: jax.Array):
        num_acc, den_acc = carry
        in_bounds = idx < (n_cad - ell)
        if check_gaps:
            assert times is not None and dt_nom is not None
            dt_pair = jnp.roll(times, -ell) - times
            expected_dt = ell.astype(times.dtype) * dt_nom
            mask = in_bounds & (jnp.abs(dt_pair - expected_dt) <= float(gap_tol) * dt_nom)
        else:
            mask = in_bounds
        diffs = jnp.where(mask, jnp.roll(flux, -ell) - flux, 0.0)
        phi = jnp.where(mask, charbonnier_abs(diffs, eps), 0.0)
        count = jnp.sum(mask.astype(flux.dtype))
        mean_ell = jnp.where(count > 0.0, jnp.sum(phi) / jnp.maximum(count, 1.0), 0.0)
        w_ell = jnp.where(count > 0.0, jnp.exp(-ell.astype(flux.dtype) / tau_arr), 0.0)
        return (num_acc + w_ell * mean_ell, den_acc + w_ell), None

    zero = jnp.zeros((), dtype=flux.dtype)
    (num_sum, den_sum), _ = jax.lax.scan(
        _scan_step, (zero, zero), jnp.arange(1, scan_len + 1, dtype=jnp.int32)
    )
    return jnp.where(den_sum > 0.0, num_sum / jnp.maximum(den_sum, 1e-30), 0.0)


@register_objective("l2v")
def l2v(
    flux: jax.Array,
    step_mask: jax.Array | None = None,
    *,
    lag: int = 1,
) -> jax.Array:
    """Gap-safe mean squared 1st difference."""
    if lag >= flux.shape[0]:
        return jnp.zeros((), dtype=flux.dtype)
    diffs = flux[lag:] - flux[:-lag]
    return _masked_mean(jnp.square(diffs), step_mask)


@register_objective("l3v")
def l3v(
    flux: jax.Array,
    step_mask: jax.Array | None = None,
    *,
    lag: int = 1,
    eps: float | jax.Array = 1e-5,
) -> jax.Array:
    """Gap-safe smooth cubic 1st difference ``r^2 * charbonnier_abs(r, eps)``."""
    if lag >= flux.shape[0]:
        return jnp.zeros((), dtype=flux.dtype)
    diffs = flux[lag:] - flux[:-lag]
    return _masked_mean(jnp.square(diffs) * charbonnier_abs(diffs, eps), step_mask)


@register_objective("owl")
def owl(
    flux: jax.Array,
    *,
    eps: float | jax.Array = 1e-5,
) -> jax.Array:
    """Smooth standard deviation of the light curve."""
    centered = flux - jnp.mean(flux)
    var_f = jnp.mean(jnp.square(centered))
    eps_safe = jnp.maximum(jnp.asarray(eps, dtype=flux.dtype), jnp.finfo(flux.dtype).tiny)
    return var_f / (jnp.sqrt(var_f + eps_safe * eps_safe) + eps_safe)


def resolve_objective(
    objective: str | Callable[..., Any],
) -> tuple[str, Callable[..., jax.Array]]:
    """Resolve an objective name or custom JAX callable into a uniform evaluator.

    Inspects ``inspect.signature`` once before JIT so custom student objectives
    with signatures ``(flux)``, ``(flux, step_mask)``, or any keyword subset of
    ``(flux, *, step_mask, weights, pixels, times, dt_nom, lag)`` all work
    without editing ``halophot`` internals.
    """
    if isinstance(objective, str):
        if objective not in OBJECTIVES:
            valid = ", ".join(sorted(OBJECTIVES))
            raise ValueError(f"Unknown objective {objective!r}; expected one of: {valid}.")
        name = objective
        fn = OBJECTIVES[objective]
    elif callable(objective):
        name = getattr(objective, "__name__", "custom")
        fn = objective
    else:
        raise TypeError(
            f"objective must be a registered objective name or callable, got {type(objective).__name__}."
        )

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    kw_only_params = [p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    all_pos_named = all(
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name in _CONTEXT_KEYS
        for p in pos_params
    )

    pos_keys: list[str] = []
    kw_keys: list[str] = []
    if all_pos_named:
        kw_keys.extend(p.name for p in pos_params)
    else:
        fallbacks = ["flux", "step_mask"]
        for p in pos_params:
            if p.name in _CONTEXT_KEYS:
                pos_keys.append(p.name)
                if p.name in fallbacks:
                    fallbacks.remove(p.name)
            elif fallbacks:
                pos_keys.append(fallbacks.pop(0))
            elif p.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Custom objective {name!r} has unsupported required positional parameter "
                    f"{p.name!r}; supported names: {_CONTEXT_KEYS[:7]}."
                )

    for p in kw_only_params:
        if p.name in _CONTEXT_KEYS:
            kw_keys.append(p.name)
        elif p.default is inspect.Parameter.empty:
            raise TypeError(
                f"Custom objective {name!r} has unsupported required keyword parameter "
                f"{p.name!r}; supported names: {_CONTEXT_KEYS[:7]}."
            )

    bound = set(pos_keys) | set(kw_keys)

    def _wrapped(**ctx: Any) -> jax.Array:
        args = [ctx[k] for k in pos_keys]
        kwargs = {k: ctx[k] for k in kw_keys}
        if has_var_kw:
            for k in _CONTEXT_KEYS:
                if k not in bound:
                    kwargs[k] = ctx[k]
        return fn(*args, **kwargs)

    return name, _wrapped
