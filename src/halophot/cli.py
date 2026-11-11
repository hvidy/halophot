"""Command-line entry point for ``halo``."""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sequence
from os.path import exists, join
from time import time as clock

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from .halo_tools import (
    diff_1,
    do_lc,
    get_annulus,
    plot_all,
    print_time,
    read_tpf,
    remove_stars,
    stitch,
    translate_greek,
)


def _parse_float_array(text: str) -> np.ndarray:
    return np.fromstring(text.strip("[]"), sep=",")


def build_parser() -> ArgumentParser:
    """Construct the ``halo`` CLI argument parser."""
    ap = ArgumentParser(description="halophot: K2 and TESS halo photometry with total variation.")
    ap.add_argument("fname", type=str, help="Input target pixel file name.")
    ap.add_argument("--data-dir", default="", type=str)
    ap.add_argument("--name", default="test", type=str, help="Target name")
    ap.add_argument("-c", "--campaign", metavar="C", default=4, type=int, help="Campaign number")
    ap.add_argument(
        "--objective",
        default="tv",
        type=str,
        help="Objective function: can be tv, tv_o2, l2v, l3v, owl, or tv_exp",
    )
    ap.add_argument(
        "--bitmask",
        default="default",
        type=str,
        help="Quality bitmask: can be default, hard, or hardest",
    )
    ap.add_argument("-sub", type=int, default=1, help="Subsampling parameter")
    ap.add_argument("--lag", type=int, default=1, help="Lag parameter")
    ap.add_argument("-maxiter", type=int, default=151, help="Maximum # iterations")
    ap.add_argument(
        "--splits",
        default=None,
        type=_parse_float_array,
        help="List of time values for kernel splits",
    )
    ap.add_argument(
        "--rr",
        default=None,
        type=_parse_float_array,
        help="rmin, rmax (pix)",
    )
    ap.add_argument("--quiet", action="store_true", default=False, help="suppress messages")
    ap.add_argument("--save-dir", default=".", help="The directory to save the output file in")
    ap.add_argument("--do-plot", action="store_true", default=False, help="produce plots")
    ap.add_argument("--do-split", action="store_true", default=False, help="split campaign segments")
    ap.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="initialize search with random seed",
    )
    ap.add_argument("--minflux", type=float, default=100.0, help="Minimum flux to include")
    ap.add_argument(
        "--thresh", type=float, default=0.8, help="What fraction of saturation to throw away"
    )
    ap.add_argument(
        "--analytic",
        action="store_true",
        default=True,
        help="use analytic derivatives; orders of magnitude faster",
    )
    ap.add_argument(
        "--sigclip",
        action=BooleanOptionalAction,
        default=True,
        help="sigma-clip the final light curve",
    )
    ap.add_argument(
        "--deathstar", action="store_true", default=False, help="remove background star pixels"
    )
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``halo`` command-line workflow."""
    ap = build_parser()
    args = ap.parse_args(argv)

    csplits: dict[int, list[int] | None] = {j: None for j in range(16)}
    csplits[4] = [550, 2200]

    if args.splits is None:
        splits = csplits.get(args.campaign)
    else:
        splits = args.splits

    if not exists(args.save_dir):
        raise FileNotFoundError(f"Save directory {args.save_dir!r} does not exist.")

    fname = join(args.data_dir, args.fname) if args.data_dir else args.fname
    tpf, ts = read_tpf(fname)

    if args.campaign == 13:
        m1 = np.logical_or(ts["time"] < 2988.2553329814764, ts["time"] > 2988.494)
        m2 = np.logical_and(m1, ts["time"] < 3064.016165412577)
        m3 = np.logical_or(m2, ts["time"] > 3064.75)
        m4 = np.logical_and(m3, ts["time"] < 3065.9776255118923)
        m = np.logical_or(m4, ts["time"] > 3066.2225)
        tpf, ts = tpf[m, :, :], ts[m]

    if args.campaign == 10:
        m = ts["time"] > 2760
        tpf, ts = tpf[m, :, :], ts[m]

    if args.campaign == 7:
        m = ts["time"] > 2470
        tpf, ts = tpf[m, :, :], ts[m]

    if args.campaign == 12:
        m = (ts["time"] > 2937.75) * (ts["time"] < 2938.8)
        tpf, ts = tpf[~m, :, :], ts[~m]

    if not args.quiet:
        print(f"Loaded {args.fname}!")

    start = clock()

    if args.rr is not None:
        rmin, rmax = args.rr
        if not args.quiet:
            print("Getting annulus from", rmin, "to", rmax)
        tpf = get_annulus(tpf, rmin, rmax)
        if not args.quiet:
            print("Using", np.sum(np.isfinite(tpf[0, :, :])), "pixels")

    if args.deathstar:
        if not args.quiet:
            print("Removing background stars")
        tpf = remove_stars(tpf)

    if args.do_split:
        if splits is None or len(splits) < 2:
            raise ValueError("--do-split requires at least two split indices via --splits or -c 4.")
        if not args.quiet:
            print("First doing one run to establish weights")

        tpf, newts, weights, _weightmap, pixelvector = do_lc(
            tpf,
            ts,
            (None, None),
            args.sub,
            maxiter=args.maxiter,
            random_init=args.random_init,
            thresh=args.thresh,
            minflux=args.minflux,
            analytic=args.analytic,
            sigclip=args.sigclip,
            verbose=not args.quiet,
            objective=args.objective,
            lag=args.lag,
            bitmask=args.bitmask,
        )

        if not args.quiet:
            print("Splitting at", splits)
        _tpf1, ts1, _w1, _wm1, _pv1 = do_lc(
            tpf,
            ts,
            (None, int(splits[0])),
            args.sub,
            maxiter=args.maxiter,
            w_init=weights,
            random_init=args.random_init,
            thresh=args.thresh,
            minflux=args.minflux,
            analytic=args.analytic,
            sigclip=args.sigclip,
            verbose=not args.quiet,
            objective=args.objective,
            lag=args.lag,
            bitmask=args.bitmask,
        )

        _tpf2, ts2, _w2, _wm2, _pv2 = do_lc(
            tpf,
            ts,
            (int(splits[0]), int(splits[1])),
            args.sub,
            maxiter=args.maxiter,
            w_init=weights,
            random_init=args.random_init,
            thresh=args.thresh,
            minflux=args.minflux,
            sigclip=args.sigclip,
            verbose=not args.quiet,
            objective=args.objective,
            lag=args.lag,
            bitmask=args.bitmask,
        )

        tpf, ts3, _w3, wmap, pixelvector = do_lc(
            tpf,
            ts,
            (int(splits[1]), None),
            args.sub,
            maxiter=args.maxiter,
            w_init=weights,
            random_init=args.random_init,
            thresh=args.thresh,
            minflux=args.minflux,
            analytic=args.analytic,
            lag=args.lag,
            sigclip=args.sigclip,
            verbose=not args.quiet,
            objective=args.objective,
            bitmask=args.bitmask,
        )

        newts = stitch([ts1, ts2, ts3])
    else:
        if not args.quiet:
            print("Not splitting")
        tpf, newts, weights, wmap, pixelvector = do_lc(
            tpf,
            ts,
            (None, None),
            args.sub,
            maxiter=args.maxiter,
            random_init=args.random_init,
            thresh=args.thresh,
            minflux=args.minflux,
            analytic=args.analytic,
            sigclip=args.sigclip,
            verbose=not args.quiet,
            objective=args.objective,
            lag=args.lag,
            bitmask=args.bitmask,
        )

    weightmap = wmap["weightmap"]
    if not args.quiet:
        print_time(clock() - start)

    opt_lc = newts["corr_flux"][:]
    finite_lc = opt_lc[np.isfinite(opt_lc)]
    tv1 = diff_1(finite_lc / np.nanmedian(opt_lc)) / float(np.size(finite_lc))
    if not args.quiet:
        print(f"Total variation per point (first order): {tv1:f} ")

    finite = np.isfinite(newts["corr_flux"])
    t_fin = np.asarray(newts["time"][finite], dtype=np.float64)
    f_fin = np.asarray(newts["corr_flux"][finite], dtype=np.float64)
    t_span = float(np.ptp(t_fin)) if t_fin.size > 0 else 1.0
    t_norm = (t_fin - float(np.mean(t_fin))) / (t_span if t_span > 0.0 else 1.0)
    poly_deg = min(15, max(1, int(np.sum(finite)) - 1))
    poly = np.poly1d(np.polyfit(t_norm, f_fin, poly_deg))
    scf = poly(t_norm)

    smooth = newts["corr_flux"][finite] - scf + np.nanmedian(scf)
    smooth /= np.nanmedian(smooth)

    newts["trend"] = np.nan * np.zeros_like(newts["corr_flux"])
    newts["trend"][finite] = scf

    newts["whitened"] = np.nan * np.zeros_like(newts["corr_flux"])
    newts["whitened"][finite] = smooth

    hdu = fits.PrimaryHDU(weightmap.T)
    cols = [fits.Column(name=key, format="D", array=newts[key]) for key in newts.keys()]
    tab = fits.BinTableHDU.from_columns(cols)

    hdul = fits.HDUList([hdu, tab])
    out_fits = join(args.save_dir, f"{args.name}_halo_lc_{args.objective}.fits")
    hdul.writeto(out_fits, overwrite=True)
    if not args.quiet:
        print(f"Saved halo-corrected light curve to {out_fits}")

    masked_weightmap = np.ma.array(weightmap, mask=np.isnan(weightmap))

    if args.do_plot:
        formal_name = translate_greek(args.name).replace("_", " ")
        image = np.nansum(tpf, axis=0)
        out_png = join(args.save_dir, f"{args.name}_all_{args.objective}.png")
        fig = plot_all(
            newts,
            image,
            masked_weightmap,
            formal_name=f"{formal_name} - C{args.campaign:02d}",
            save_file=out_png,
        )
        plt.close(fig)
        if not args.quiet:
            print(f"Everything saved to {out_png}")


if __name__ == "__main__":
    main()
