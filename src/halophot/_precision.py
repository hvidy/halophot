"""Locally scoped 64-bit JAX precision helpers (canonical v2 contract).

None of ``mundey``, ``tessbkgd``, or ``halophot`` mutates ``jax_enable_x64``
globally on import. Entry points run inside :func:`run_in` / :func:`in_float64`
and restore the caller's prior JAX precision state on exit.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

_enable_x64 = getattr(jax, "enable_x64", None)
if _enable_x64 is None:  # pragma: no cover - JAX < 0.8 fallback
    from jax.experimental import enable_x64 as _enable_x64

_DTYPES: dict[str, tuple[Any, Any, Any, Any]] = {
    "float32": (jnp.float32, jnp.complex64, np.float32, np.complex64),
    "float64": (jnp.float64, jnp.complex128, np.float64, np.complex128),
}


def _check_dtype(dtype: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be 'float32' or 'float64', not {dtype!r}.")


@contextlib.contextmanager
def run_in(dtype: str = "float64") -> Iterator[None]:
    """Execute the enclosed block with JAX 64-bit mode enabled or disabled."""
    _check_dtype(dtype)
    with _enable_x64(dtype == "float64"):
        yield


def cast_tree(tree: Any, dtype: str = "float64") -> Any:
    """Cast floating/complex leaves in ``tree`` to ``dtype``, keeping Quantity units."""
    _check_dtype(dtype)
    j_real, j_cplx, np_real, np_cplx = _DTYPES[dtype]

    def _cast(leaf: Any) -> Any:
        if isinstance(leaf, np.ndarray) and hasattr(leaf, "unit"):
            k = leaf.dtype.kind
            if k in ("f", "c"):
                return leaf.astype(np_cplx if k == "c" else np_real, copy=False)
            return leaf
        if not isinstance(leaf, jax.Array | np.ndarray | np.number):
            return leaf
        if leaf.dtype.kind in ("f", "c"):
            target = j_cplx if leaf.dtype.kind == "c" else j_real
            if isinstance(leaf, jax.Array) and leaf.dtype == target:
                return leaf
            return jnp.asarray(leaf, dtype=target)
        return leaf

    with run_in(dtype):
        return jax.tree_util.tree_map(_cast, tree)


def in_float64(fn: F) -> F:
    """Run ``fn`` inside ``run_in("float64")`` with floating/complex inputs cast."""

    def _cast_arg(arg: Any) -> Any:
        return arg if isinstance(arg, Iterator) else cast_tree(arg, "float64")

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with run_in("float64"):
            return fn(
                *(_cast_arg(a) for a in args),
                **{k: _cast_arg(v) for k, v in kwargs.items()},
            )

    return wrapped  # type: ignore[return-value]
