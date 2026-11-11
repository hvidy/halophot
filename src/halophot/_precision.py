"""Local numerical precision helpers for JAX optimization entry points.

``halophot`` never mutates ``jax_enable_x64`` globally on import. Numerical
entry points execute inside :func:`in_float64`, which enters a locally scoped
``jax.enable_x64(True)`` context, casts floating-point inputs to 64-bit
precision, and restores the caller's global JAX x64 state on exit.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

_F = TypeVar("_F", bound=Callable[..., object])

_DTYPES: dict[str, tuple[jnp.dtype, jnp.dtype]] = {
    "float32": (jnp.float32, jnp.complex64),
    "float64": (jnp.float64, jnp.complex128),
}


def _check_dtype(dtype: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be 'float32' or 'float64', not {dtype!r}.")


@contextlib.contextmanager
def run_in(dtype: str) -> Iterator[None]:
    """Run the enclosed block with JAX 64-bit precision enabled or disabled."""
    _check_dtype(dtype)
    with jax.enable_x64(dtype == "float64"):
        yield


def cast_tree(tree: object, dtype: str = "float64") -> object:
    """Cast floating and complex JAX/NumPy leaves in ``tree`` to ``dtype``."""
    _check_dtype(dtype)
    real_dtype, complex_dtype = _DTYPES[dtype]

    def _cast_leaf(leaf: object) -> object:
        if not isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
            return leaf
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
            return jnp.asarray(leaf, dtype=complex_dtype)
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return jnp.asarray(leaf, dtype=real_dtype)
        return leaf

    return jax.tree_util.tree_map(_cast_leaf, tree)


def in_float64(fn: _F) -> _F:
    """Execute ``fn`` inside ``jax.enable_x64(True)`` with float64 array inputs.

    Floating-point JAX and NumPy arrays (and PyTrees thereof) are cast to
    ``float64`` (and complex arrays to ``complex128``) on entry; non-array
    arguments and iterators pass through unchanged. Global ``jax_enable_x64``
    state is restored on exit even if ``fn`` raises.
    """

    def _cast_arg(arg: object) -> object:
        return arg if isinstance(arg, Iterator) else cast_tree(arg, "float64")

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        with run_in("float64"):
            cast_args = [_cast_arg(a) for a in args]
            cast_kwargs = {k: _cast_arg(v) for k, v in kwargs.items()}
            return fn(*cast_args, **cast_kwargs)

    return wrapped  # type: ignore[return-value]
