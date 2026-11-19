import math

import jax
from jax import numpy as np


def softmax(x):
    """From https://gist.github.com/stober/1946926"""
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def make_flux(weights, pixelvector):
    flux = np.dot(softmax(weights).T, pixelvector)
    return flux / np.mean(flux)


def tv(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    diff = np.sum(np.abs(flux[lag:] - flux[:(-lag)]))
    return diff


def tv_exp(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    n_cad = flux.shape[0]
    tau = float(lag)
    scan_len = min(max(int(math.ceil(4.0 * tau)), 1), n_cad - 1)
    if scan_len < 1:
        return np.zeros((), dtype=flux.dtype)

    idx = np.arange(n_cad, dtype=np.int32)
    tau_arr = np.asarray(tau, dtype=flux.dtype)

    @jax.checkpoint
    def _step(carry, ell):
        num_acc, den_acc = carry
        in_bounds = idx < (n_cad - ell)
        diffs = np.where(in_bounds, np.abs(np.roll(flux, -ell) - flux), 0.0)
        count = np.maximum(np.sum(in_bounds.astype(flux.dtype)), 1.0)
        w_ell = np.exp(-ell.astype(flux.dtype) / tau_arr)
        return (num_acc + w_ell * (np.sum(diffs) / count), den_acc + w_ell), None

    zero = np.zeros((), dtype=flux.dtype)
    (num_sum, den_sum), _ = jax.lax.scan(_step, (zero, zero), np.arange(1, scan_len + 1, dtype=np.int32))
    return float(n_cad - 1) * num_sum / np.maximum(den_sum, 1e-30)


def tv_o2(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    diff = np.sum(np.abs(2.0 * flux[lag:(-lag)] - flux[(2 * lag) :] - flux[: (-2 * lag)]))
    return diff / np.mean(flux)


def l2v(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    diff = np.sum((flux[lag:] - flux[:(-lag)]) ** 2)
    return diff


def l3v(weights, lag, pixelvector):
    flux = make_flux(weights, pixelvector)
    diff = np.sum(np.abs(flux[lag:] - flux[:(-lag)]) ** 3)
    return diff


def owl(weights, pixelvector):
    # for halophot this is strictly worse than all other options
    flux = make_flux(weights, pixelvector)
    var = np.var(flux)
    safe_var = np.where(var > 0.0, var, 1.0)
    diff = np.where(var > 0.0, np.sqrt(safe_var), 0.0)
    return diff / np.mean(flux)


available = [tv, tv_o2, l2v, l3v, owl, tv_exp]
mapping = {avail.__name__: avail for avail in available}
