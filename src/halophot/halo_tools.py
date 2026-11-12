import functools
from typing import NamedTuple

import astropy.table
import equinox as eqx
import jax
import jax.numpy as jnp
import lightkurve
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import LombScargle
from astropy.units import Quantity
from lightkurve.utils import KeplerQualityFlags, TessQualityFlags
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d as gaussfilt
from scipy.signal import savgol_filter
from tqdm import tqdm

from . import halo_objectives as objectives
from ._contract import (
    infer_bkg_poly_order,
    read_bkg_covariance,
    read_bkgsub_state,
    read_smear_err,
)
from ._precision import in_float64
from .lightcurve import compute_coherent_flux_err
from .masking import remove_stars as remove_stars

"""-----------------------------------------------------------------
halo_tools.py

In this package we include all the functions that are necessary for
halo photometry in Python.

-----------------------------------------------------------------"""

# Exact conversion factor from cycles per day (d^-1) to microhertz (uHz).
DAY_TO_MUHZ: float = 1e6 / 86400.0

simbadgreek = [
    "alf",
    "bet",
    "gam",
    "del",
    "eps",
    "zet",
    "eta",
    "tet",
    "iot",
    "kap",
    "lam",
    "mu",
    "nu",
    "ksi",
    "omi",
    "pi",
    "rho",
    "sig",
    "tau",
    "ups",
    "phi",
    "khi",
    "psi",
    "ome",
]
latexgreek = [
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\epsilon",
    "\\zeta",
    "\\eta",
    "\\theta",
    "\\iota",
    "\\kappa",
    "\\lambda",
    "\\mu",
    "\\nu",
    "\\xi",
    "o",
    "\\pi",
    "\\rho",
    "\\sigma",
    "\\tau",
    "\\upsilon",
    "\\phi",
    "\\chi",
    "\\psi",
    "\\omega",
]


def translate_greek(word):
    if word[0].isupper():
        return word
    else:
        for j, letter in enumerate(simbadgreek):
            if letter in word:
                word = word.replace(letter, f"${latexgreek[j]}$")
                return word
        return word


# =========================================================================
# =========================================================================


def softmax(x):
    """From https://gist.github.com/stober/1946926"""
    x_arr = np.asarray(x, dtype=np.float64)
    e_x = np.exp(x_arr - np.max(x_arr))
    out = e_x / e_x.sum()
    return out


# =========================================================================
# =========================================================================


def get_pgram(time, flux, min_p=1.0 / 24.0, max_p=20.0):
    finite = np.isfinite(flux)
    ls = LombScargle(time[finite], flux[finite], normalization="psd")

    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_p, maximum_frequency=1.0 / min_p, samples_per_peak=5
    )

    norm = np.nanstd(flux * 1e6) ** 2 / np.sum(
        power
    )  # normalize according to Parseval's theorem - same form used by Oliver Hall in lightkurve
    fs = np.mean(np.diff(frequency * DAY_TO_MUHZ))  # ppm^2/muHz

    power *= norm / fs

    spower = gaussfilt(power, 15)

    return frequency, power, spower


# =========================================================================
# =========================================================================


def print_time(t):
    if t > 3600:
        hours = int(np.floor(t / 3600))
        mins = int(np.floor(np.mod(t, 3600) / 60))
        secs = float(np.mod(t, 60))
        print(f"Time taken: {hours:d} h {mins:d} m {secs:3f} s")
    elif t > 60:
        mins = int(np.floor(np.mod(t, 3600) / 60))
        secs = float(np.mod(t, 60))
        print(f"Time taken: {mins:d} m {secs:3f} s")
    else:
        print(f"Time taken: {t:3f} s")


# =========================================================================
# =========================================================================


def read_tpf(fname):
    with fits.open(fname) as target_fits:
        data = target_fits[1].data
        tpf = np.array(data["FLUX"][:])
        t = np.array(data["TIME"][:])
        x = np.array(data["POS_CORR1"][:])
        y = np.array(data["POS_CORR2"][:])
        cad = np.array(data["CADENCENO"][:])
        quality = np.array(data["QUALITY"][:], dtype="int32")

    ts = Table({"time": t, "cadence": cad, "x": x, "y": y, "quality": quality})
    return tpf, ts


# =========================================================================
# =========================================================================


def censor_tpf(
    tpf,
    ts,
    thresh=-1,
    minflux=-100.0,
    do_quality=True,
    verbose=True,
    sub=1,
    lag=1,
    objective="tv",
    mission="kepler",
    bitmask="default",
):
    """Throw away bad pixels and bad cadences"""

    if not np.any(np.isfinite(tpf)):
        raise ValueError("No valid pixels survived censoring in censor_tpf.")
    dummy = tpf.copy()
    tsd = ts.copy()
    maxflux = np.nanmax(tpf)

    # find bad pixels

    if do_quality:
        if mission == "kepler":
            qf = KeplerQualityFlags()
            m = np.asarray(qf.create_quality_mask(ts["quality"], bitmask=bitmask), dtype=bool)
        elif mission == "tess":
            qf = TessQualityFlags()
            m = np.asarray(qf.create_quality_mask(ts["quality"], bitmask=bitmask), dtype=bool)
        else:
            m = np.asarray(ts["quality"] == 0, dtype=bool)  # get bad quality
    else:
        m = np.ones(len(ts["quality"]), dtype=bool)

    # Preserve negative background-subtracted noise excursions above minflux.
    if not np.any(m):
        raise ValueError("No valid cadences survived quality filtering in censor_tpf.")

    if thresh >= 0:
        valid_sat = np.any(np.isfinite(dummy[m, :, :]), axis=0)
        sat_raw = np.zeros(dummy.shape[1:], dtype=bool)
        if np.any(valid_sat):
            sat_raw[valid_sat] = np.nanmax(dummy[m][:, valid_sat], axis=0) > (thresh * maxflux)
        saturated = ndimage.binary_dilation(
            sat_raw,
            structure=np.ones((3, 1), dtype=bool),
        )
        dummy[:, saturated] = np.nan
        n_sat = int(np.sum(saturated))
        if verbose:
            print(f"{n_sat:d} saturated pixels")

    # automatic saturation threshold
    # TODO: does not have binary dilation mask implemented
    if thresh < 0:
        nstart = max(0, np.sum(np.nanmedian(dummy[m, :, :], axis=0) > 7e4) - 20)
        nfinish = np.sum(np.nanmedian(dummy[m, :, :], axis=0) > 5e4)
        if verbose:
            print(
                f"Searching for number of saturated pixels to cut between {int(nstart):d} and {int(nfinish):d}"
            )
        stds = []
        threshs = np.arange(nstart, nfinish)
        if len(threshs) == 0:
            pix = 0
        else:
            for thr in tqdm(threshs, desc="thresholds"):
                med_img = np.nanmedian(dummy[m, :, :], axis=0)
                saturated = np.unravel_index((-med_img).argsort(axis=None)[:thr], med_img.shape)
                dummy2 = dummy.copy()
                dummy2[:, saturated[0], saturated[1]] = np.nan
                pf, ts_iter, weights, weightmap, pixels_sub = do_lc(
                    dummy2,
                    tsd,
                    (None, None),
                    sub,
                    maxiter=101,
                    w_init=None,
                    random_init=False,
                    thresh=1.0,
                    minflux=-100,
                    analytic=True,
                    sigclip=False,
                    verbose=False,
                    lag=lag,
                    objective=objective,
                    mission=mission,
                    bitmask=bitmask,
                )
                fl = ts_iter["corr_flux"]
                fs = fl[~np.isnan(fl)] / np.nanmedian(fl)
                sfs = savgol_filter(fs, int(np.floor(len(fs) / 8) * 2 - 1), 3)
                stds.append(np.std(fs / sfs))
            stds = np.asarray(stds)
            d1 = np.r_[0, stds[1:] - stds[:-1]]
            d2 = np.r_[0, stds[2:] - 2 * stds[1:-1] + stds[:-2], 0]
            i1 = []
            for cut in [3, 2, 1, 0]:  # no while loop!!
                ind = (
                    (stds > cut * stds[0])
                    & (d1 > 0)
                    & np.r_[True, d2[1:] < d2[:-1]]
                    & np.r_[d2[:-1] < d2[1:], True]
                    & (d2 < 0)
                )
                i1 = np.arange(len(ind))[ind]
                if len(i1) != 0:
                    break

            if len(i1) == 0:
                pix = int(threshs[0])
            else:
                if len(i1) > 3:
                    i1 = i1[i1.argsort(axis=None)][0:3]

                if threshs[i1[np.argmax(d1[i1])]] == threshs[i1[np.argmin(d2[i1])]]:
                    pix = threshs[i1[np.argmax(d1[i1])]]
                else:
                    p1 = i1[np.argmax(d1[i1])]
                    p2 = i1[np.argmin(d2[i1])]
                    if abs(d1[p1] - d1[p2]) > abs(d2[p1] - d2[p2]):
                        pix = threshs[p1]
                    else:
                        pix = threshs[p2]
        if verbose:
            print(f"Finished optimization: {int(pix):d} saturated pixels")
        max_img = np.nanmax(dummy[m, :, :], axis=0)
        saturated = np.unravel_index((-max_img).argsort(axis=None)[:pix], max_img.shape)
        dummy[:, saturated[0], saturated[1]] = np.nan
        n_sat = int(len(saturated[0]))

    valid_pix = np.any(np.isfinite(dummy[m, :, :]), axis=0)
    no_flux = np.zeros(dummy.shape[1:], dtype=bool)
    if np.any(valid_pix):
        no_flux[valid_pix] = np.nanmin(dummy[m][:, valid_pix], axis=0) < minflux
    dummy[:, no_flux] = np.nan

    xc, yc = np.nanmedian(ts["x"][m]), np.nanmedian(ts["y"][m])

    if np.sum(np.isfinite(ts["x"])) >= 0.8 * tsd["x"][m].shape[0]:
        rr = np.sqrt((tsd["x"][m] - xc) ** 2 + (tsd["y"][m] - yc) ** 2)
        goodpos = (rr < 5) * np.isfinite(tsd["x"][m]) * np.isfinite(tsd["y"][m])
        m[np.flatnonzero(m)[~goodpos]] = False
        if np.sum(~goodpos) > 0:
            if verbose:
                print(f"Throwing out {int(np.sum(~goodpos)):d} bad cadences")

    # then pick only pixels which are mostly good

    n_good_cad = dummy[m, :, :].shape[0]
    if n_good_cad == 0:
        raise ValueError("No valid cadences survived quality and centroid filtering in censor_tpf.")
    pixels = np.reshape(dummy[m, :, :].T, (tpf.shape[1] * tpf.shape[2], n_good_cad))
    indic = np.sum(np.isfinite(pixels), axis=1)
    min_valid = min(60, (n_good_cad - 1) // 2)
    keep_pix = indic > min_valid
    if not np.any(keep_pix):
        raise ValueError("No valid pixels survived censoring in censor_tpf.")
    pixels = pixels[keep_pix, :]

    good_cols = np.all(np.isfinite(pixels), axis=0)
    if not np.any(good_cols):
        raise ValueError("No cadences with all-finite surviving pixels remained in censor_tpf.")
    m[np.flatnonzero(m)[~good_cols]] = False
    tsd = ts[m]
    pixels = pixels[:, good_cols]

    # this should get all the nans but if not just set them to 0

    pixels[~np.isfinite(pixels)] = 0

    return pixels, tsd, m, np.where(keep_pix), n_sat


# =========================================================================
# =========================================================================


def get_slice(tpf, ts, start, stop):
    return tpf[start:stop, :, :], ts[start:stop]


# =========================================================================
# =========================================================================


def get_annulus(tpf, rmin, rmax):
    xs, ys = np.arange(tpf.shape[2]) - tpf.shape[2] / 2.0, np.arange(tpf.shape[1]) - tpf.shape[1] / 2.0
    xx, yy = np.meshgrid(xs, ys)
    rr = np.sqrt(xx**2 + yy**2)
    mask = (rr > rmax) + (rr < rmin)
    tpf[:, mask] = np.nan
    return tpf


# =========================================================================
# =========================================================================


def stitch(tslist):
    newts = tslist[0].copy()
    newts["corr_flux"] /= np.nanmedian(tslist[0]["corr_flux"])
    for tsj in tslist[1:]:
        tsj_copy = tsj.copy()
        tsj_copy["corr_flux"] /= np.nanmedian(tsj_copy["corr_flux"])
        newts = astropy.table.vstack([newts, tsj_copy])
    return newts


"""-----------------------------------------------------------------
In this section we include the actual detrending code.
-----------------------------------------------------------------"""

## it seems to be faster than using np.diff?


def diff_1(z):
    return np.sum(np.abs(z[1:] - z[:-1]))


def diff_2(z):
    return np.sum(np.abs(2.0 * z[1:-1] - z[2:] - z[:-2]))


# =========================================================================
# =========================================================================


class OptimResult(NamedTuple):
    """Optimization result pairing parameter logits with the Optimistix solution."""

    params: jax.Array
    solution: optx.Solution


def _min_cadences_for_objective(objective: str, lag: int) -> int:
    """Return the minimum number of cadences required to evaluate ``objective`` cleanly."""
    if objective == "owl":
        return 2
    if objective == "tv_o2":
        return 2 * lag + 1
    return lag + 1


@functools.lru_cache(maxsize=32)
def _get_cached_minimiser(objective: str, lag: int):
    """Build and cache a JIT-compiled Optimistix L-BFGS runner for ``(objective, lag)``."""
    obj_callable = objectives.mapping[objective]
    solver = optx.BestSoFarMinimiser(optx.LBFGS(rtol=1e-6, atol=1e-6))

    if objective == "owl":

        def loss_fn(weights, args):
            (pvec,) = args
            return obj_callable(weights, pvec)

    else:

        def loss_fn(weights, args):
            (pvec,) = args
            return obj_callable(weights, lag, pvec)

    @eqx.filter_jit
    def _run(w0: jax.Array, pvec: jax.Array, max_steps: int) -> tuple[optx.Solution, jax.Array]:
        sol = optx.minimise(
            loss_fn,
            solver,
            w0,
            args=(pvec,),
            max_steps=max_steps,
            throw=False,
        )
        return sol, loss_fn(sol.value, (pvec,))

    return _run


@in_float64
def jax_func(pixelvector_jax, w_init, objective, maxiter, lag):
    """Solve a halo photometry objective function using Optimistix L-BFGS in float64."""
    if objective not in objectives.mapping:
        valid = ", ".join(sorted(objectives.mapping))
        raise ValueError(f"Unknown objective {objective!r}; expected one of: {valid}")

    if isinstance(maxiter, bool) or int(maxiter) < 1:
        raise ValueError(f"maxiter must be an integer >= 1, got {maxiter!r}.")
    maxiter_int = int(maxiter)

    if isinstance(lag, bool) or int(lag) < 1:
        raise ValueError(f"lag must be an integer >= 1, got {lag!r}.")
    lag_int = int(lag)

    pvec = jnp.asarray(pixelvector_jax, dtype=jnp.float64)
    if pvec.ndim != 2:
        raise ValueError(f"pixelvector_jax must be 2D (npix, n_cad), got shape {pvec.shape}.")
    npix, n_cad = pvec.shape
    if npix < 1:
        raise ValueError(f"pixelvector_jax must contain at least 1 pixel, got npix={npix}.")
    min_cad = _min_cadences_for_objective(objective, lag_int)
    if n_cad < min_cad:
        raise ValueError(
            f"Objective {objective!r} with lag={lag_int} requires at least {min_cad} cadences, "
            f"got n_cad={n_cad}."
        )
    if not bool(jnp.all(jnp.isfinite(pvec))):
        raise ValueError("pixelvector_jax contains non-finite values.")

    w0 = jnp.asarray(w_init, dtype=jnp.float64)
    if w0.shape != (npix,):
        raise ValueError(f"w_init shape {w0.shape} does not match (npix,) = ({npix},).")
    if not bool(jnp.all(jnp.isfinite(w0))):
        raise ValueError("w_init contains non-finite values.")

    runner = _get_cached_minimiser(objective, lag_int)
    sol, final_loss = runner(w0, pvec, maxiter_int)
    if not bool(jnp.all(jnp.isfinite(sol.value))) or not bool(jnp.isfinite(final_loss)):
        raise ValueError(
            f"Optimization for objective={objective!r} produced a non-finite solution or loss."
        )
    return OptimResult(params=sol.value, solution=sol)


@in_float64
def tv_tpf(
    pixelvector,
    w_init=None,
    maxiter=101,
    analytic=False,
    sigclip=False,
    verbose=True,
    lag=1,
    objective="tv",
):
    """
    This is the main function here - once you have loaded the data, pass it to this
    to do a TV-min light curve. Implemented in JAX.

    Keywords

    maxiter: int
        Number of iterations to optimize. 101 is default & usually sufficient.
    w_init: None or array-like.
        Initialize weights with a particular simplex weight vector - useful if you have
        already run TV-min and want to update, but otherwise set to None
        and it will have default uniform initialization.
    random_init: Boolean
        If False, and w_init is None, it will initialize with uniform weights; if True, it
        will initialize with random weights. False is usually better.
    thresh: float
        A float greater than 0. Pixels less than this fraction of the maximum
        flux at any pixel will be masked out - this is to deal with saturation.
        Because halo is usually intended for saturated stars, the default is 0.8,
        to deal with saturated pixels. If your star is not saturated, set this
        greater than 1.0.
    analytic: Boolean
        If True, it will optimize the TV with JAX analytic derivatives, which is
        several orders of magnitude faster than with numerical derivatives. This is
        by default True but you can run it numerically with False if you prefer.
    sigclip: Boolean
        If True, it will iteratively run the TV-min algorithm clipping outliers.
        Use this for data with a lot of outliers, but by default it is set False.
    """
    del analytic  # JAX reverse-mode AD is always used.

    pixelvector_np = np.asarray(pixelvector, dtype=np.float64)
    if pixelvector_np.ndim != 2 or pixelvector_np.shape[0] < 1:
        raise ValueError(
            f"pixelvector must be 2D (npix, n_cad) with npix >= 1, got shape {pixelvector_np.shape}."
        )
    npix = pixelvector_np.shape[0]

    if w_init is None:
        logits_init = jnp.zeros(npix, dtype=jnp.float64)
    else:
        w_arr = jnp.asarray(w_init, dtype=jnp.float64)
        if w_arr.shape != (npix,):
            raise ValueError(f"w_init shape {w_arr.shape} does not match (npix,) = ({npix},).")
        if not bool(jnp.all(jnp.isfinite(w_arr))) or bool(jnp.any(w_arr < 0.0)):
            raise ValueError("w_init simplex weights must be finite and non-negative.")
        w_sum = jnp.sum(w_arr)
        if not bool(w_sum > 0.0):
            raise ValueError("w_init simplex weights must have strictly positive sum.")
        w_norm = w_arr / w_sum
        logits_init = jnp.log(jnp.maximum(w_norm, jnp.finfo(jnp.float64).tiny))

    if verbose:
        print("Using Analytic Derivatives")

    pixelvector_jax = jnp.asarray(pixelvector_np, dtype=jnp.float64)

    res = jax_func(pixelvector_jax, logits_init, objective, maxiter, lag)
    obj_params, _ = res

    w_best = np.asarray(objectives.softmax(obj_params), dtype=np.float64)

    lc_first_try = np.dot(w_best.T, pixelvector_np)

    if sigclip:
        if verbose:
            print("Sigma clipping")

        if len(lc_first_try) >= 3:
            win = min(
                51, len(lc_first_try) if len(lc_first_try) % 2 == 1 else len(lc_first_try) - 1
            )
            win = max(3, win)
            baseline = savgol_filter(lc_first_try, win, 1)
        else:
            baseline = np.nanmedian(lc_first_try)
        good = ~sigma_clip(
            lc_first_try - baseline,
            sigma_lower=10.0,
            sigma_upper=3.0,
        ).mask

        min_cad = _min_cadences_for_objective(objective, int(lag))
        if np.sum(~good) > 0 and int(np.sum(good)) >= min_cad:
            if verbose:
                print(f"Clipping {int(np.sum(~good)):d} bad points")

            pixels_masked = jnp.asarray(pixelvector_np[:, good], dtype=jnp.float64)

            res = jax_func(pixels_masked, logits_init, objective, maxiter, lag)
            tv_params, _ = res

            w_best = np.asarray(objectives.softmax(tv_params), dtype=np.float64)

        elif np.sum(~good) > 0:
            if verbose:
                print("Too few cadences remain after clipping for refit; retaining initial fit")
            good = np.isfinite(lc_first_try)
        else:
            if verbose:
                print("No outliers found, continuing")
    else:
        good = np.isfinite(lc_first_try)

    lc_opt = np.asarray(np.dot(w_best.T, pixelvector_np), dtype=np.float64).copy()
    lc_opt[~good] = np.nan
    return w_best, lc_opt, res


# =========================================================================
# =========================================================================


def print_flex(splits):
    s = "Taking cadences from: beginning to "
    for split in splits:
        s += f"{split:.1f}; to "
    s += "end"
    print(s)


# =========================================================================
# =========================================================================


def do_lc(
    tpf,
    ts,
    splits,
    sub,
    maxiter=101,
    split_times=None,
    w_init=None,
    random_init=False,
    thresh=-1.0,
    minflux=-100.0,
    analytic=False,
    sigclip=False,
    verbose=True,
    lag=1,
    objective="tv",
    mission="kepler",
    bitmask="default",
):
    ### get a slice corresponding to the splits you want
    if isinstance(sub, bool) or int(sub) < 1:
        raise ValueError(f"sub must be an integer >= 1, got {sub!r}.")
    sub = int(sub)

    if split_times is not None:
        split_arr = np.sort(np.asarray(split_times, dtype=np.float64).ravel())
        if split_arr.size < 1 or not np.all(np.isfinite(split_arr)):
            raise ValueError("split_times must contain at least one finite timestamp.")
        if split_arr[0] <= np.nanmin(ts["time"]) or split_arr[-1] >= np.nanmax(ts["time"]):
            raise ValueError("All split_times must lie strictly inside the campaign time span.")
        splits = [int(np.min(np.where(ts["time"] > split))) for split in split_arr]
        if len(set(splits)) != len(splits):
            raise ValueError(f"split_times map to duplicate cadence indices {splits}.")
        if w_init is not None and np.asarray(w_init).ndim == 1:
            raise ValueError(
                "When split_times is provided, w_init must be None or a 2D (nx, ny) weightmap."
            )
        all_splits = [None, *splits, None]
        tss = []
        cad1 = []
        cad2 = []
        sat = []
        weightmap = []

        for j, low in enumerate(all_splits[:-1]):
            high = all_splits[j + 1]
            pff, tsj, weights, pmap, pixels_sub = do_lc(
                tpf,
                ts,
                (low, high),
                sub,
                maxiter=maxiter,
                split_times=None,
                w_init=w_init,
                random_init=random_init,
                thresh=thresh,
                minflux=minflux,
                analytic=analytic,
                sigclip=sigclip,
                verbose=verbose,
                lag=lag,
                objective=objective,
                mission=mission,
                bitmask=bitmask,
            )
            tss.append(tsj)
            if low is None:
                cad1.append(ts["cadence"][0])
            else:
                cad1.append(ts["cadence"][low])
            if high is None:
                cad2.append(ts["cadence"][-1])
            else:
                cad2.append(ts["cadence"][high])
            sat.append(pmap["sat_pixels"])
            weightmap.append(pmap["weightmap"])
        wmap = {
            "initial_cadence": cad1,
            "final_cadence": cad2,
            "sat_pixels": sat,
            "weightmap": weightmap[0],
            "weightmaps": np.stack(weightmap, axis=0),
        }
        ts = stitch(tss)

    else:
        if splits[0] is None and splits[1] is not None:
            c1 = ts["cadence"][0]
            c2 = ts["cadence"][splits[1]]
        elif splits[0] is not None and splits[1] is None:
            c1 = ts["cadence"][splits[0]]
            c2 = ts["cadence"][-1]
        elif splits[0] is None and splits[1] is None:
            c1 = ts["cadence"][0]
            c2 = ts["cadence"][-1]
        else:
            c1 = ts["cadence"][splits[0]]
            c2 = ts["cadence"][splits[1]]

        if verbose:
            if splits[0] is None and splits[1] is not None:
                print("Taking cadences from beginning to", splits[1])
            elif splits[0] is not None and splits[1] is None:
                print("Taking cadences from", splits[0], "to end")
            elif splits[0] is None and splits[1] is None:
                print("Taking cadences from beginning to end")
            else:
                print("Taking cadences from", splits[0], "to", splits[1])

        tpf, ts = get_slice(tpf, ts, splits[0], splits[1])

        ### now throw away saturated columns, nan pixels and nan cadences

        pixels, tsd, goodcad, mapping, sat = censor_tpf(
            tpf,
            ts,
            thresh=thresh,
            minflux=minflux,
            verbose=verbose,
            sub=sub,
            lag=lag,
            objective=objective,
            mission=mission,
            bitmask=bitmask,
        )
        pixelmap = np.zeros((tpf.shape[2], tpf.shape[1]))
        if verbose:
            print("Censored TPF")

        ### subsample

        pixels_sub = pixels[::sub, :]
        if pixels_sub.shape[0] < 1:
            raise ValueError("No pixels remained after subsampling.")
        if verbose:
            print(f"Subsampling by a factor of {sub:d}")

        ### now calculate the halo

        if verbose:
            print("Calculating weights")

        w_seg_init = w_init
        if w_init is not None:
            w_init_arr = np.asarray(w_init, dtype=np.float64)
            if w_init_arr.ndim == 2:
                expected_2d = (tpf.shape[2], tpf.shape[1])
                if w_init_arr.shape != expected_2d:
                    raise ValueError(
                        f"2D w_init shape {w_init_arr.shape} does not match weightmap shape "
                        f"{expected_2d}."
                    )
                active_idx = mapping[0] if sub == 1 else mapping[0][::sub]
                raw_w = w_init_arr.ravel()[active_idx]
                extracted = np.where(np.isfinite(raw_w) & (raw_w > 0.0), raw_w, 0.0)
                w_sum = float(np.sum(extracted))
                w_seg_init = extracted / w_sum if w_sum > 0.0 else None

        if random_init:
            rng = np.random.default_rng()
            u_draws = np.clip(
                rng.uniform(size=pixels_sub.shape[0]),
                np.finfo(np.float64).tiny,
                1.0 - np.finfo(np.float64).eps,
            )
            exp_draws = -np.log(u_draws)
            w_seg_init = exp_draws / np.sum(exp_draws)

        weights, opt_lc, _ = tv_tpf(
            pixels_sub,
            maxiter=maxiter,
            w_init=w_seg_init,
            analytic=analytic,
            verbose=verbose,
            sigclip=sigclip,
            lag=lag,
            objective=objective,
        )
        if verbose:
            print("Calculated weights!")

        ts["corr_flux"] = np.nan * np.ones_like(ts["x"], dtype=np.float64)
        ts["corr_flux"][goodcad] = opt_lc / np.nanmedian(opt_lc)

        if sub == 1:
            pixelmap.ravel()[mapping] = weights
        else:
            pixelmap.ravel()[mapping[0][::sub]] = weights

        wmap = {
            "initial_cadence": c1,
            "final_cadence": c2,
            "sat_pixels": sat,
            "weightmap": pixelmap,
        }

    return tpf, ts, weights, wmap, pixels_sub


# =========================================================================
# Make Plots
# =========================================================================


def plot_lc(ax1, time, lc, name, trends=None, title=False):
    m = (lc > 0.0) * (np.isfinite(lc))

    ax1.plot(time[m], lc[m] / np.nanmedian(lc[m]), ".")
    dt = np.nanmedian(time[m][1:] - time[m][:-1])
    ax1.set_xlim(time[m].min() - dt, time[m].max() + dt)
    if trends is not None:
        colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        for j, trend in enumerate(trends):
            ax1.plot(
                time[m],
                trend[m] / np.nanmedian(trend[m]),
                "-",
                color=colours[(2 - j) % len(colours)],
            )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Relative Flux")
    if title:
        plt.title(f"{name}")


def plot_fluxmap(ax1, image, name, title=False):
    cmap = mpl.colormaps["hot"].with_extremes(bad="k")
    img_arr = np.asarray(image, dtype=np.float64)
    pos = np.isfinite(img_arr) & (img_arr > 0.0)
    im = np.where(pos, np.log10(np.where(pos, img_arr, 1.0)), np.nan)
    pic = ax1.imshow(im, cmap=cmap, vmax=np.nanmax(im), interpolation="nearest", origin="lower")
    if title:
        plt.title(f"{name} Flux Map")

    aspect = 20
    pad_fraction = 0.5

    divider = make_axes_locatable(ax1)
    width = axes_size.AxesY(ax1, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("left", size=width, pad=pad)
    plt.colorbar(pic, cax=cax)

    ax1.yaxis.set_ticks_position("right")

    cax.yaxis.set_ticks_position("left")


def plot_weightmap(ax1, weightmap, name, title=False, aspect=None):
    norm = np.size(weightmap)

    cmap = mpl.colormaps["seismic"].with_extremes(bad="k")

    wm_scaled = np.ma.filled(weightmap.T * norm, np.nan).astype(np.float64)
    pos = np.isfinite(wm_scaled) & (wm_scaled > 0.0)
    im = np.where(pos, np.log10(np.where(pos, wm_scaled, 1.0)), np.nan)
    pic = ax1.imshow(
        im,
        cmap=cmap,
        vmin=-2 * np.nanmax(im),
        vmax=2 * np.nanmax(im),
        interpolation=None,
        origin="lower",
        aspect=aspect,
    )
    if title:
        plt.title(f"TV-min Weightmap: {name}")

    if aspect is not None:
        aspect_val = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(ax1)
        width = axes_size.AxesY(ax1, aspect=1.0 / aspect_val)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=0.1 * width, pad=0.1 * pad)
    else:
        aspect_val = 20
        pad_fraction = 0.5

        divider = make_axes_locatable(ax1)
        width = axes_size.AxesY(ax1, aspect=1.0 / aspect_val)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)

    plt.colorbar(pic, cax=cax, label="log Weight")

    ax1.yaxis.set_ticks_position("left")


def plot_pgram(
    ax1,
    frequency,
    power,
    spower,
    name,
    min_p=1.0 / 24.0,
    max_p=20.0,
    title=False,
    fontsize=14,
    alias=1,
):
    ax1.plot(frequency * DAY_TO_MUHZ, power, "0.8", label="Raw")
    ax1.plot(frequency * DAY_TO_MUHZ, spower, linewidth=3.0, label="Smoothed")
    ax1.set_xlim(1.0 / max_p * DAY_TO_MUHZ, 1.0 / min_p * DAY_TO_MUHZ)
    ax1.set_xlabel(r"Frequency ($\mu$Hz)")
    ax1.set_ylabel(r"Power (ppm$^2/{\mu}Hz)$")

    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", direction="in", pad=-20)

    ax2.set_xlim(1.0 / max_p, 1.0 / min_p)
    ax2.set_xlabel("c/d", labelpad=-20)
    plt.ylim(0, np.max(power))
    if title:
        plt.title(f"{name} Power Spectrum", y=1.01)
    ax1.legend()
    peak_period = 1.0 / frequency[np.argmax(power)] * alias
    ax1.text(
        frequency[np.argmax(power)] * DAY_TO_MUHZ * 1.2,
        np.nanmax(power) * 0.7,
        f"P = {peak_period:.3f} d",
        fontsize=fontsize,
    )


def plot_log_pgram(ax1, frequency, power, spower, name, min_p=1.0 / 24.0, max_p=20.0, title=False):
    ax1.plot(frequency * DAY_TO_MUHZ, power, "0.8", label="Raw")
    ax1.plot(frequency * DAY_TO_MUHZ, spower, linewidth=3.0, label="Smoothed")
    ax1.set_xlim(1.0 / max_p * DAY_TO_MUHZ, 1.0 / min_p * DAY_TO_MUHZ)
    ax1.set_xlabel(r"Frequency ($\mu$Hz)")
    ax1.set_ylabel(r"Power (ppm$^2/{\mu}Hz)$")

    ax2 = ax1.twiny()
    ax2.tick_params(axis="x", direction="in", pad=-20)

    ax2.set_xlim(1.0 / max_p, 1.0 / min_p)
    ax2.set_xlabel("c/d", labelpad=-35)

    plt.ylim(np.percentile(spower, 5), np.max(power))
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    plt.yscale("log")
    if title:
        plt.title(f"{name} Power Spectrum", y=1.01)
    ax1.legend()


def plot_all(ts, image, weightmap, save_file=None, formal_name="test", title=True):
    min_p, max_p = 1.0 / 24.0, 20.0

    PW, PH = 8.27, 11.69

    frequency, power, spower = get_pgram(ts["time"], ts["whitened"], min_p=min_p, max_p=max_p)

    local_rc = {
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "font.size": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1,
    }
    with mpl.rc_context(local_rc):
        fig = plt.figure(figsize=(PW, PH))
        gs1 = GridSpec(2, 2)
        gs1.update(top=0.95, bottom=2 / 3.0 * 1.05, hspace=0.0, left=0.09, right=0.96)
        gs2 = GridSpec(1, 2)
        gs2.update(top=2 / 3.0 * 0.97, bottom=1 / 3.0 * 1.07, hspace=0.35, left=0.09, right=0.96)
        gs3 = GridSpec(2, 2)
        gs3.update(top=1 / 3.0 * 0.96, bottom=0.04, hspace=0.07, left=0.09, right=0.96)

        ax_lctime = subplot(gs1[0, :])
        ax_lcwhite = subplot(gs1[1, :], sharex=ax_lctime)
        ax_fluxmap = subplot(gs2[0, 0])
        ax_weightmap = subplot(gs2[0, 1])
        ax_periodogram = subplot(gs3[0, :])
        ax_logpgram = subplot(gs3[1, :])

        plot_lc(ax_lctime, ts["time"], ts["corr_flux"], formal_name, trends=[ts["trend"]])
        plot_lc(ax_lcwhite, ts["time"], ts["whitened"], formal_name + ": Whitened")
        plot_weightmap(ax_weightmap, weightmap, formal_name)
        plot_fluxmap(ax_fluxmap, image, formal_name)
        plot_pgram(ax_periodogram, frequency, power, spower, formal_name)
        plot_log_pgram(ax_logpgram, frequency, power, spower, formal_name)

        if title:
            fig.suptitle(formal_name, y=0.99, fontsize=20)
        ax_periodogram.set_title("Periodograms")
        ax_fluxmap.set_title("Flux Map")
        ax_weightmap.set_title("TV-Min Weight Map")

        if save_file is not None:
            plt.savefig(save_file)
    return fig


# =========================================================================
# Coherent flux_err helper for halo_tpf / halo_tpf_tess shims
# =========================================================================


def _compute_legacy_tpf_flux_err(
    tpf_obj,
    weightmap_dict: dict[str, object],
    corr_flux: np.ndarray,
    flux_arr: np.ndarray,
) -> np.ndarray:
    """Propagate coherent normalized ``flux_err`` on ``halo_tpf`` / ``halo_tpf_tess``."""
    fe_obj = getattr(tpf_obj, "flux_err", None)
    fe_arr = (
        np.asarray(fe_obj.value if hasattr(fe_obj, "value") else fe_obj, dtype=np.float64)
        if fe_obj is not None
        else None
    )
    if fe_arr is None or not np.any(np.isfinite(fe_arr) & (fe_arr > 0.0)):
        return np.nan * np.asarray(corr_flux, dtype=np.float64)

    fbe_obj = getattr(tpf_obj, "flux_bkg_err", None)
    fbe_arr = (
        np.asarray(fbe_obj.value if hasattr(fbe_obj, "value") else fbe_obj, dtype=np.float64)
        if fbe_obj is not None
        else None
    )
    if fbe_arr is not None and not np.any(np.isfinite(fbe_arr) & (fbe_arr > 0.0)):
        fbe_arr = None

    w_ny_nx = np.asarray(weightmap_dict["weightmap"], dtype=np.float64).T
    rows_h, cols_h = np.where(np.isfinite(w_ny_nx) & (w_ny_nx > 0.0))
    if rows_h.size == 0:
        return np.nan * np.asarray(corr_flux, dtype=np.float64)

    valid_cad = np.isfinite(corr_flux)
    if not np.any(valid_cad):
        return np.nan * np.asarray(corr_flux, dtype=np.float64)

    raw_lc = flux_arr[valid_cad][:, rows_h, cols_h] @ w_ny_nx[rows_h, cols_h]
    med_raw = float(np.nanmedian(raw_lc))
    if not np.isfinite(med_raw) or abs(med_raw) <= 1e-30:
        return np.nan * np.asarray(corr_flux, dtype=np.float64)

    err_out = compute_coherent_flux_err(
        w_ny_nx,
        flux_err=fe_arr,
        smear_err=read_smear_err(tpf_obj, apply_quality_mask=True),
        bkg_cov=read_bkg_covariance(tpf_obj, apply_quality_mask=True),
        flux_bkg_err=fbe_arr,
        bkgsub=read_bkgsub_state(tpf_obj, warn_legacy_mundey=False),
        bkg_order=infer_bkg_poly_order(tpf_obj),
        norm_factor=med_raw,
    )
    err_out[~valid_cad] = np.nan
    return err_out


"""-----------------------------------------------------------------
The cuts for Campaign 4 are

0:550
550:2200
2200:
-----------------------------------------------------------------"""


class halo_tpf(lightkurve.KeplerTargetPixelFile):
    def halo(
        self,
        aperture_mask="pipeline",
        split_times=None,
        sub=1,
        maxiter=101,
        w_init=None,
        random_init=False,
        thresh=-1,
        minflux=-100.0,
        objective="tv",
        rr=None,
        analytic=True,
        sigclip=False,
        mask=None,
        verbose=True,
        lag=1,
        bitmask="default",
    ):
        """Performs 'halo' TV-min weighted-aperture photometry.
         Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the Kepler pipeline mask.
         splits : tuple, (None, None) or (2152,2175) etc.
            A tuple including two times at which to split the light curve and run halo
            separately outside these splits.
         sub : int
            Do you want to subsample every nth pixel in your light curve? Not advised,
            but can come in handy for very large TPFs.
         maxiter: int
            Number of iterations to optimize. 101 is default & usually sufficient.
         w_init: None or array-like.
            Initialize weights with a particular weight vector - useful if you have
            already run TV-min and want to update, but otherwise set to None
            and it will have default initialization.
         random_init: Boolean
            If False, and w_init is None, it will initialize with uniform weights; if True, it
            will initialize with random weights. False is usually better.
         thresh: float
            A float greater than 0. Pixels less than this fraction of the maximum
            flux at any pixel will be masked out - this is to deal with saturation.
            Because halo is usually intended for saturated stars, the default is 0.8,
            to deal with saturated pixels. If your star is not saturated, set this
            greater than 1.0.
         analytic: Boolean
            If True, it will optimize the TV with JAX analytic derivatives, which is
            several orders of magnitude faster than with numerical derivatives. This is
            by default True but you can run it numerically with False if you prefer.
         sigclip: Boolean
            If True, it will iteratively run the TV-min algorithm clipping outliers.
            Use this for data with a lot of outliers, but by default it is set False.
         Returns
        -------
        lc : KeplerLightCurve object
            Array containing the TV-min flux within the aperture for each
            cadence.
        """

        if mask is None:
            aperture_mask = self._parse_aperture_mask(aperture_mask)
        else:
            aperture_mask = mask

        if self.mission == "TESS":
            mission = "tess"
        else:
            mission = "kepler"

        cent_col, cent_row = self.estimate_centroids(aperture_mask=aperture_mask)

        x = self.hdu[1].data["POS_CORR1"][self.quality_mask]
        y = self.hdu[1].data["POS_CORR2"][self.quality_mask]
        quality = self.quality
        ts = Table(
            {
                "time": self.time.value,
                "cadence": self.cadenceno,
                "x": x,
                "y": y,
                "quality": quality,
            }
        )

        flux = np.copy(self.flux)

        flux[:, ~aperture_mask] = np.nan
        if rr is not None:
            rmin, rmax = rr
            if verbose:
                print("Getting annulus from", rmin, "to", rmax)
            flux = get_annulus(flux, rmin, rmax)
            if verbose:
                print("Using", np.sum(np.isfinite(flux[0, :, :])), "pixels")

        flux_arr = flux.value if hasattr(flux, "value") else np.asarray(flux)
        pf, ts, weights, weightmap, pixels_sub = do_lc(
            flux_arr,
            ts,
            (None, None),
            sub,
            maxiter=maxiter,
            split_times=split_times,
            w_init=w_init,
            random_init=random_init,
            thresh=thresh,
            minflux=minflux,
            analytic=analytic,
            sigclip=sigclip,
            verbose=verbose,
            lag=lag,
            objective=objective,
            mission=mission,
            bitmask=bitmask,
        )

        keys = {
            "centroid_col": cent_col,
            "centroid_row": cent_row,
            "quality": ts["quality"],
            "channel": self.channel,
            "campaign": self.campaign,
            "targetid": self.targetid,
            "mission": self.mission,
            "quarter": self.quarter,
            "ra": self.ra,
            "dec": self.dec,
            "cadenceno": ts["cadence"],
        }

        flux_err_lc = _compute_legacy_tpf_flux_err(self, weightmap, ts["corr_flux"], flux_arr)
        lc_out = lightkurve.KeplerLightCurve(
            flux=ts["corr_flux"],
            time=Time(ts["time"], format=self.time.format, scale=self.time.scale),
            flux_err=flux_err_lc,
            **keys,
        )
        lc_out["pos_corr1"] = self.pos_corr1
        lc_out["pos_corr2"] = self.pos_corr2
        lc_out.meta = dict(lc_out.meta)
        hdu = lc_out.to_fits()
        object.__setattr__(lc_out, "primary_header", hdu[0].header)
        object.__setattr__(lc_out, "data_header", hdu[1].header)
        if "weightmaps" in weightmap:
            lc_out.meta["WEIGHTMAP"] = np.swapaxes(
                np.asarray(weightmap["weightmaps"], dtype=np.float64), -1, -2
            )
        else:
            lc_out.meta["WEIGHTMAP"] = np.asarray(weightmap["weightmap"], dtype=np.float64).T
        return weightmap, lc_out


class halo_tpf_tess(lightkurve.TessTargetPixelFile):
    def halo(
        self,
        aperture_mask="pipeline",
        split_times=None,
        sub=1,
        maxiter=101,
        w_init=None,
        random_init=False,
        thresh=-1,
        minflux=-100.0,
        objective="tv",
        rr=None,
        analytic=True,
        sigclip=False,
        mask=None,
        verbose=True,
        lag=1,
        bitmask="default",
    ):
        """Performs 'halo' TV-min weighted-aperture photometry.
         Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the Kepler pipeline mask.
         splits : tuple, (None, None) or (2152,2175) etc.
            A tuple including two times at which to split the light curve and run halo
            separately outside these splits.
         sub : int
            Do you want to subsample every nth pixel in your light curve? Not advised,
            but can come in handy for very large TPFs.
         maxiter: int
            Number of iterations to optimize. 101 is default & usually sufficient.
         w_init: None or array-like.
            Initialize weights with a particular weight vector - useful if you have
            already run TV-min and want to update, but otherwise set to None
            and it will have default initialization.
         random_init: Boolean
            If False, and w_init is None, it will initialize with uniform weights; if True, it
            will initialize with random weights. False is usually better.
         thresh: float
            A float greater than 0. Pixels less than this fraction of the maximum
            flux at any pixel will be masked out - this is to deal with saturation.
            Because halo is usually intended for saturated stars, the default is 0.8,
            to deal with saturated pixels. If your star is not saturated, set this
            greater than 1.0.
         analytic: Boolean
            If True, it will optimize the TV with JAX analytic derivatives, which is
            several orders of magnitude faster than with numerical derivatives. This is
            by default True but you can run it numerically with False if you prefer.
         sigclip: Boolean
            If True, it will iteratively run the TV-min algorithm clipping outliers.
            Use this for data with a lot of outliers, but by default it is set False.
         Returns
        -------
        lc : TessLightCurve object
            Array containing the TV-min flux within the aperture for each
            cadence.
        """

        if mask is None:
            aperture_mask = self._parse_aperture_mask(aperture_mask)
        else:
            aperture_mask = mask

        if self.mission == "TESS":
            mission = "tess"
        else:
            mission = "kepler"

        cent_col, cent_row = self.estimate_centroids(aperture_mask=aperture_mask)

        x = self.hdu[1].data["POS_CORR1"][self.quality_mask]
        y = self.hdu[1].data["POS_CORR2"][self.quality_mask]
        quality = self.quality
        ts = Table(
            {
                "time": self.time.value,
                "cadence": self.cadenceno,
                "x": x,
                "y": y,
                "quality": quality,
            }
        )

        flux = np.copy(self.flux)

        flux[:, ~aperture_mask] = np.nan
        if rr is not None:
            rmin, rmax = rr
            if verbose:
                print("Getting annulus from", rmin, "to", rmax)
            flux = get_annulus(flux, rmin, rmax)
            if verbose:
                print("Using", np.sum(np.isfinite(flux[0, :, :])), "pixels")

        flux_arr = flux.value if hasattr(flux, "value") else np.asarray(flux)
        pf, ts, weights, weightmap, pixels_sub = do_lc(
            flux_arr,
            ts,
            (None, None),
            sub,
            maxiter=maxiter,
            split_times=split_times,
            w_init=w_init,
            random_init=random_init,
            thresh=thresh,
            minflux=minflux,
            analytic=analytic,
            sigclip=sigclip,
            verbose=verbose,
            lag=lag,
            objective=objective,
            mission=mission,
            bitmask=bitmask,
        )

        keys = {
            "centroid_col": cent_col,
            "centroid_row": cent_row,
            "quality": ts["quality"],
            "targetid": self.targetid,
            "ra": self.ra,
            "dec": self.dec,
            "sector": self.sector,
            "camera": self.camera,
            "ccd": self.ccd,
            "mission": self.mission,
            "cadenceno": ts["cadence"],
        }
        flux_err_lc = _compute_legacy_tpf_flux_err(self, weightmap, ts["corr_flux"], flux_arr)
        lc_out = lightkurve.TessLightCurve(
            flux=ts["corr_flux"],
            time=Time(ts["time"], format=self.time.format, scale=self.time.scale),
            flux_err=flux_err_lc,
            **keys,
        )
        lc_out["pos_corr1"] = self.pos_corr1
        lc_out["pos_corr2"] = self.pos_corr2
        lc_out.meta = dict(lc_out.meta)
        if "weightmaps" in weightmap:
            lc_out.meta["WEIGHTMAP"] = np.swapaxes(
                np.asarray(weightmap["weightmaps"], dtype=np.float64), -1, -2
            )
        else:
            lc_out.meta["WEIGHTMAP"] = np.asarray(weightmap["weightmap"], dtype=np.float64).T
        object.__setattr__(lc_out, "primary_header", self.hdu[0].header)
        object.__setattr__(lc_out, "data_header", self.hdu[1].header)
        return weightmap, lc_out

    @property
    def flux(self) -> Quantity:
        """Returns the flux for all good-quality cadences."""
        unit = None
        if self.get_header(1).get("TUNIT5") == "e-/s":
            unit = "electron/s"
        return Quantity(self.hdu[1].data["FLUX"][self.quality_mask], unit=unit)

    @flux.setter
    def flux(self, value):
        raw = value.value if hasattr(value, "value") else value
        self.hdu[1].data["FLUX"][self.quality_mask] = raw

    @property
    def flux_bkg(self) -> Quantity:
        """Returns the background flux for all good-quality cadences."""
        return Quantity(self.hdu[1].data["FLUX_BKG"][self.quality_mask], unit="electron/s")

    @flux_bkg.setter
    def flux_bkg(self, value):
        raw = value.value if hasattr(value, "value") else value
        self.hdu[1].data["FLUX_BKG"][self.quality_mask] = raw

    @property
    def raw_cnts(self):
        """Returns the raw counts for all good-quality cadences."""
        return self.hdu[1].data["RAW_CNTS"][self.quality_mask]

    @raw_cnts.setter
    def raw_cnts(self, value):
        self.hdu[1].data["RAW_CNTS"][self.quality_mask] = value
