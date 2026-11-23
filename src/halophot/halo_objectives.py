"""Backwards-compatible v1 objective wrappers delegating to :mod:`halophot.objectives`."""

from __future__ import annotations

import jax.numpy as jnp

from .objectives import l2v as _l2v
from .objectives import l3v as _l3v
from .objectives import normalized_flux
from .objectives import owl as _owl
from .objectives import tv_exp as _tv_exp
from .objectives import tv_o1 as _tv_o1
from .objectives import tv_o2 as _tv_o2
from .simplex import logits_to_weights


def softmax(x):
    """Map logits to physical simplex weights."""
    return logits_to_weights(x)


def make_flux(weights, pixelvector):
    """Evaluate unit-mean weighted flux from unconstrained logits."""
    return normalized_flux(logits_to_weights(weights), pixelvector)


def tv(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    return float(max(flux.shape[0] - lag, 1)) * _tv_o1(flux, lag=lag, eps=1e-12)


def tv_exp(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    return float(max(flux.shape[0] - 1, 1)) * _tv_exp(flux, lag=lag, tau=float(lag), eps=1e-12)


def tv_o2(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    return float(max(flux.shape[0] - 2 * lag, 1)) * _tv_o2(flux, lag=lag, eps=1e-12)


def l2v(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    return float(max(flux.shape[0] - lag, 1)) * _l2v(flux, lag=lag)


def l3v(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    return float(max(flux.shape[0] - lag, 1)) * _l3v(flux, lag=lag, eps=1e-12)


def owl(weights, pixelvector):
    flux = make_flux(weights, pixelvector)
    return _owl(flux, eps=jnp.finfo(flux.dtype).tiny)


available = [tv, tv_o2, l2v, l3v, owl, tv_exp]
mapping = {avail.__name__: avail for avail in available}
