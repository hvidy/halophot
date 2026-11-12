"""Tests for locally scoped 64-bit JAX execution (``halophot._precision``)."""

from __future__ import annotations

import importlib
import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as _plt  # noqa: F401
import numpy as np
import pytest

import halophot
import halophot._contract
import halophot._precision
import halophot.cli
import halophot.halo_objectives
import halophot.halo_tools
import halophot.lightcurve
import halophot.masking
import halophot.models
import halophot.objectives
import halophot.optimise
import halophot.simplex
from halophot._precision import cast_tree, in_float64, run_in


def test_import_has_zero_global_side_effects() -> None:
    """Importing ``halophot`` and all v2.0 submodules must not mutate JAX, Matplotlib, or warnings."""
    assert jax.config.jax_enable_x64 is False
    rc_before = dict(mpl.rcParams)
    filters_before = list(warnings.filters)

    for mod in (
        halophot._precision,
        halophot._contract,
        halophot.simplex,
        halophot.models,
        halophot.objectives,
        halophot.optimise,
        halophot.masking,
        halophot.lightcurve,
        halophot.halo_objectives,
        halophot.halo_tools,
        halophot.cli,
        halophot,
    ):
        importlib.reload(mod)

    assert jax.config.jax_enable_x64 is False
    assert dict(mpl.rcParams) == rc_before
    assert list(warnings.filters) == filters_before


def test_in_float64_scopes_x64_and_casts_pytrees() -> None:
    assert jax.config.jax_enable_x64 is False

    @in_float64
    def inspect_fn(tree: dict[str, object], scale: float, *, extra: np.ndarray) -> tuple[object, ...]:
        assert jax.config.jax_enable_x64 is True
        internal = jnp.ones(2)
        assert internal.dtype == jnp.float64
        return tree, scale, extra

    payload = {
        "jax_f32": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "np_f32": np.array([3.0, 4.0], dtype=np.float32),
        "cplx64": jnp.array([1.0 + 2.0j], dtype=jnp.complex64),
        "ints": np.array([7, 8], dtype=np.int32),
        "flag": True,
        "label": "halo",
    }
    extra_in = np.float32(0.25) * np.ones(3, dtype=np.float32)

    out_tree, out_scale, out_extra = inspect_fn(payload, 2.5, extra=extra_in)

    # Global x64 state must be restored immediately on exit.
    assert jax.config.jax_enable_x64 is False

    assert out_tree["jax_f32"].dtype == jnp.float64  # type: ignore[index]
    assert out_tree["np_f32"].dtype == jnp.float64  # type: ignore[index]
    assert out_tree["cplx64"].dtype == jnp.complex128  # type: ignore[index]
    assert out_tree["ints"].dtype == np.int32  # type: ignore[index]
    assert out_tree["flag"] is True  # type: ignore[index]
    assert out_tree["label"] == "halo"  # type: ignore[index]
    assert out_scale == 2.5
    assert out_extra.dtype == jnp.float64


def test_in_float64_restores_state_on_exception_and_nesting() -> None:
    @in_float64
    def boom(x: jax.Array) -> jax.Array:
        assert jax.config.jax_enable_x64 is True
        assert x.dtype == jnp.float64
        raise RuntimeError("expected test error")

    assert jax.config.jax_enable_x64 is False
    with pytest.raises(RuntimeError, match="expected test error"):
        boom(jnp.ones(4, dtype=jnp.float32))
    assert jax.config.jax_enable_x64 is False

    # When called inside an already-enabled x64 context, True must be preserved on exit.
    with jax.enable_x64(True):
        with pytest.raises(RuntimeError, match="expected test error"):
            boom(jnp.ones(4, dtype=jnp.float32))
        assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_enable_x64 is False


def test_cast_tree_float64_while_global_x64_disabled() -> None:
    """Public ``halophot.cast_tree`` must allocate float64/complex128 without global x64."""
    assert jax.config.jax_enable_x64 is False
    tree_in = {
        "jax_f32": jnp.array([1.25, -2.5], dtype=jnp.float32),
        "np_f32": np.array([3.5, 4.75], dtype=np.float32),
        "cplx64": jnp.array([1.0 - 3.0j], dtype=jnp.complex64),
        "ints": np.array([5, 6], dtype=np.int32),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree_out = halophot.cast_tree(tree_in, "float64")

    assert jax.config.jax_enable_x64 is False
    assert len(caught) == 0
    assert tree_out["jax_f32"].dtype == jnp.float64  # type: ignore[index]
    assert tree_out["np_f32"].dtype == jnp.float64  # type: ignore[index]
    assert tree_out["cplx64"].dtype == jnp.complex128  # type: ignore[index]
    assert tree_out["ints"].dtype == np.int32  # type: ignore[index]


def test_iterators_pass_through_and_invalid_dtype_raises() -> None:
    @in_float64
    def consume_first(it):
        return next(it)

    gen = iter([10, 20, 30])
    assert consume_first(gen) == 10
    assert next(gen) == 20

    with run_in("float64"):
        f32_tree = cast_tree({"x": jnp.ones(2, dtype=jnp.float64)}, "float32")
        assert f32_tree["x"].dtype == jnp.float32  # type: ignore[index]

    with pytest.raises(ValueError, match="dtype must be"):
        with run_in("float16"):
            pass

    with pytest.raises(ValueError, match="dtype must be"):
        cast_tree(jnp.ones(1), "bfloat16")
