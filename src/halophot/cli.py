"""Command-line entry point for ``halo``."""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sequence
from os.path import exists, join
from time import time as clock

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from lightkurve.targetpixelfile import TargetPixelFile

from ._contract import resolve_cache_dir, write_lightcurve_fits
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
from .lightcurve import extract_halo_lightcurve


def _parse_float_array(text: str) -> np.ndarray:
    stripped = text.strip().strip("[]").strip()
    if not stripped:
        raise ValueError("Expected one or more comma-separated float values.")
    return np.asarray([float(tok.strip()) for tok in stripped.split(",")], dtype=np.float64)


def _parse_int_array(text: str) -> list[int]:
    stripped = text.strip().strip("[]").strip()
    if not stripped:
        raise ValueError("Expected one or more comma-separated integer cadence indices.")
    return [int(tok.strip()) for tok in stripped.split(",")]


def _resolve_split_indices(
    ts: Table,
    *,
    split_times: np.ndarray | None,
    split_cadences: list[int] | None,
    campaign: int,
) -> list[int]:
    """Resolve time splits or cadence-index splits to strictly increasing 0-based slice indices."""
    if split_times is not None and split_cadences is not None:
        raise ValueError(
            "Pass at most one of --splits (time values) or --split-cadences (cadence indices)."
        )

    n_cad = len(ts)
    if n_cad < 4:
        raise ValueError(f"Cannot split TPF with only {n_cad} cadences.")

    if split_times is not None:
        times_arr = np.asarray(split_times, dtype=np.float64).ravel()
        if times_arr.size < 1 or not np.all(np.isfinite(times_arr)):
            raise ValueError("--splits must provide at least one finite timestamp.")
        t_col = np.asarray(ts["time"], dtype=np.float64)
        finite_t = t_col[np.isfinite(t_col)]
        if finite_t.size == 0:
            raise ValueError("TPF time column contains no finite timestamps.")
        t_min = float(np.min(finite_t))
        t_max = float(np.max(finite_t))
        sorted_times = np.sort(times_arr)
        if sorted_times[0] <= t_min or sorted_times[-1] >= t_max:
            raise ValueError(
                f"--splits timestamps {sorted_times.tolist()} must lie strictly inside "
                f"the observation time span ({t_min}, {t_max})."
            )
        indices = [int(np.min(np.where(t_col > t_val))) for t_val in sorted_times]
    elif split_cadences is not None:
        if len(split_cadences) < 1:
            raise ValueError("--split-cadences must provide at least one cadence index.")
        indices = sorted(int(idx) for idx in split_cadences)
    else:
        csplits: dict[int, list[int] | None] = {j: None for j in range(16)}
        csplits[4] = [550, 2200]
        default_splits = csplits.get(campaign)
        if default_splits is None:
            raise ValueError(
                "--do-split requires split points via --splits, --split-cadences, or -c 4."
            )
        indices = list(default_splits)

    if indices[0] <= 0 or indices[-1] >= n_cad:
        raise ValueError(
            f"Split cadence indices {indices} must lie strictly inside (0, {n_cad})."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Split points map to duplicate cadence indices {indices}.")
    return indices


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
        help="Comma-separated time values (matching TPF TIME column) for kernel splits",
    )
    ap.add_argument(
        "--split-cadences",
        default=None,
        type=_parse_int_array,
        help="Comma-separated 0-based cadence indices for kernel splits",
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
    ap.add_argument(
        "--catalog-table",
        default=None,
        type=str,
        help="Path to pre-cached Gaia/TIC catalogue table for --deathstar / --offline",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Disable live network catalogue queries (for HPC compute nodes)",
    )
    ap.add_argument(
        "--strict-hpc",
        action="store_true",
        default=False,
        help="Reject cache paths under /fred/ to protect HPC inode quotas",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="Explicit cache directory (defaults to $SCRATCH/.cache/tess_bright)",
    )
    ap.add_argument(
        "--legacy-tv",
        action="store_true",
        default=False,
        help="Use unregularized w-simplex TV in --hlsp-fits v2.0 mode",
    )
    ap.add_argument(
        "--hlsp-fits",
        action="store_true",
        default=False,
        help="Run v2.0 extract_halo_lightcurve and write HALOCVER=1 HLSP FITS with WEIGHTMAP",
    )
    ap.add_argument(
        "--uncertainty",
        action="store_true",
        default=False,
        help="Compute orbit block-jackknife pixel weight uncertainties in --hlsp-fits mode",
    )
    return ap


def _read_tpf_wcs_and_target(fname: str) -> tuple[WCS | None, tuple[float, float] | None]:
    """Extract celestial WCS and (RA_OBJ, DEC_OBJ) from a TPF FITS file."""
    with fits.open(fname) as hdul:
        wcs_obj: WCS | None = None
        if "APERTURE" in hdul:
            cand = WCS(hdul["APERTURE"].header)
            if cand.has_celestial:
                wcs_obj = cand
        if wcs_obj is None and len(hdul) > 2:
            cand = WCS(hdul[2].header)
            if cand.has_celestial:
                wcs_obj = cand
        ra = hdul[0].header.get("RA_OBJ")
        dec = hdul[0].header.get("DEC_OBJ")
        tgt = (float(ra), float(dec)) if ra is not None and dec is not None else None
        return wcs_obj, tgt


def _attach_trend_and_whitened(ts: Table) -> None:
    """Populate polynomial ``trend`` and ``whitened`` columns on ``ts`` in-place."""
    finite = np.isfinite(ts["time"]) & np.isfinite(ts["corr_flux"])
    if not np.any(finite):
        raise ValueError("No finite timestamps and fluxes survived in detrended light curve.")
    t_fin = np.asarray(ts["time"][finite], dtype=np.float64)
    f_fin = np.asarray(ts["corr_flux"][finite], dtype=np.float64)
    t_span = float(np.ptp(t_fin)) if t_fin.size > 0 else 1.0
    t_norm = (t_fin - float(np.mean(t_fin))) / (t_span if t_span > 0.0 else 1.0)
    poly_deg = min(15, max(0, int(np.sum(finite)) - 1))
    poly = np.poly1d(np.polyfit(t_norm, f_fin, poly_deg))
    scf = poly(t_norm)

    smooth = ts["corr_flux"][finite] - scf + np.nanmedian(scf)
    smooth /= np.nanmedian(smooth)

    ts["trend"] = np.nan * np.zeros_like(ts["corr_flux"])
    ts["trend"][finite] = scf

    ts["whitened"] = np.nan * np.zeros_like(ts["corr_flux"])
    ts["whitened"][finite] = smooth


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``halo`` command-line workflow."""
    ap = build_parser()
    args = ap.parse_args(argv)

    if not exists(args.save_dir):
        raise FileNotFoundError(f"Save directory {args.save_dir!r} does not exist.")

    if args.strict_hpc or args.cache_dir is not None:
        resolve_cache_dir(args.cache_dir, strict_hpc=args.strict_hpc)

    fname = join(args.data_dir, args.fname) if args.data_dir else args.fname
    if args.hlsp_fits:
        tpf_lk = lk.read(fname, quality_bitmask=args.bitmask)
        if not isinstance(tpf_lk, TargetPixelFile):
            raise TypeError(
                f"Expected a 3D lightkurve.TargetPixelFile from {fname!r}, got {type(tpf_lk).__name__}."
            )

        t_lk = np.asarray(tpf_lk.time.value, dtype=np.float64)
        if args.campaign == 13:
            m1 = np.logical_or(t_lk < 2988.2553329814764, t_lk > 2988.494)
            m2 = np.logical_and(m1, t_lk < 3064.016165412577)
            m3 = np.logical_or(m2, t_lk > 3064.75)
            m4 = np.logical_and(m3, t_lk < 3065.9776255118923)
            m_camp = np.logical_or(m4, t_lk > 3066.2225)
            tpf_lk = tpf_lk[m_camp]
        elif args.campaign == 10:
            tpf_lk = tpf_lk[t_lk > 2760]
        elif args.campaign == 7:
            tpf_lk = tpf_lk[t_lk > 2470]
        elif args.campaign == 12:
            m_camp = (t_lk > 2937.75) & (t_lk < 2938.8)
            tpf_lk = tpf_lk[~m_camp]

        hlsp_splits: np.ndarray | None = None
        if args.do_split or args.splits is not None or args.split_cadences is not None:
            t_col = np.asarray(tpf_lk.time.value, dtype=np.float64)
            ts_hlsp = Table({"time": t_col})
            split_indices = _resolve_split_indices(
                ts_hlsp,
                split_times=args.splits,
                split_cadences=args.split_cadences,
                campaign=args.campaign,
            )
            if args.splits is not None and args.split_cadences is None:
                hlsp_splits = np.sort(np.asarray(args.splits, dtype=np.float64).ravel())
            else:
                split_ts: list[float] = []
                for idx_s in split_indices:
                    left_fin = t_col[:idx_s][np.isfinite(t_col[:idx_s])]
                    right_fin = t_col[idx_s:][np.isfinite(t_col[idx_s:])]
                    if left_fin.size == 0 or right_fin.size == 0:
                        raise ValueError(
                            f"Split cadence index {idx_s} lacks finite flanking timestamps."
                        )
                    split_ts.append(0.5 * (float(left_fin[-1]) + float(right_fin[0])))
                hlsp_splits = np.asarray(split_ts, dtype=np.float64)

        rr_tuple = (float(args.rr[0]), float(args.rr[1])) if args.rr is not None else None
        lc = extract_halo_lightcurve(
            tpf_lk,
            rr=rr_tuple,
            split_times=hlsp_splits,
            sub=args.sub,
            objective=args.objective,
            lag=args.lag,
            legacy_tv=args.legacy_tv,
            maxiter=args.maxiter,
            sigclip=args.sigclip,
            thresh=args.thresh,
            minflux=args.minflux,
            remove_contaminants=args.deathstar,
            catalog_table=args.catalog_table,
            offline=args.offline,
            compute_uncertainty=args.uncertainty,
        )
        out_fits = join(args.save_dir, f"{args.name}_halo_lc_{args.objective}.fits")
        write_lightcurve_fits(lc, out_fits, overwrite=True)
        if not args.quiet:
            print(f"Saved HALOCVER=1 halo light curve to {out_fits}")

        if args.do_plot:
            flux_vals = np.asarray(
                lc.flux.value if hasattr(lc.flux, "value") else lc.flux,
                dtype=np.float64,
            )
            plot_ts = Table(
                {
                    "time": np.asarray(lc.time.value, dtype=np.float64),
                    "corr_flux": flux_vals,
                }
            )
            _attach_trend_and_whitened(plot_ts)
            tpf_flux = np.asarray(
                tpf_lk.flux.value if hasattr(tpf_lk.flux, "value") else tpf_lk.flux,
                dtype=np.float64,
            )
            image = np.nansum(tpf_flux, axis=0)
            wmap_meta = np.asarray(lc.meta["WEIGHTMAP"], dtype=np.float64)
            wmap_2d = np.mean(wmap_meta, axis=0) if wmap_meta.ndim == 3 else wmap_meta
            wmap_legacy = wmap_2d.T
            masked_wmap = np.ma.array(
                wmap_legacy,
                mask=~np.isfinite(wmap_legacy) | (wmap_legacy <= 0.0),
            )
            formal_name = translate_greek(args.name).replace("_", " ")
            out_png = join(args.save_dir, f"{args.name}_all_{args.objective}.png")
            fig = plot_all(
                plot_ts,
                image,
                masked_wmap,
                formal_name=f"{formal_name} - C{args.campaign:02d}",
                save_file=out_png,
            )
            plt.close(fig)
            if not args.quiet:
                print(f"Everything saved to {out_png}")
        return

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
        wcs_obj, tgt_coord = _read_tpf_wcs_and_target(fname)
        tpf = remove_stars(
            tpf,
            wcs=wcs_obj,
            target_coord=tgt_coord,
            catalog_table=args.catalog_table,
            offline=args.offline,
        )

    if args.do_split:
        split_indices = _resolve_split_indices(
            ts,
            split_times=args.splits,
            split_cadences=args.split_cadences,
            campaign=args.campaign,
        )
        if not args.quiet:
            print("First doing one run to establish weights")

        _full_tpf, _full_ts, _full_weights, full_wmap, pixelvector = do_lc(
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
            print("Splitting at cadence indices", split_indices)

        boundaries: list[int | None] = [None, *split_indices, None]
        seg_tables: list[Table] = []
        seg_wmaps: list[dict[str, object]] = []
        warm_weightmap = None if args.random_init else full_wmap["weightmap"]

        for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
            _seg_tpf, seg_ts, _seg_w, seg_wmap, _seg_pv = do_lc(
                tpf,
                ts,
                (low, high),
                args.sub,
                maxiter=args.maxiter,
                w_init=warm_weightmap,
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
            seg_tables.append(seg_ts)
            seg_wmaps.append(seg_wmap)

        newts = stitch(seg_tables)
        wmap = {
            "initial_cadence": [wm["initial_cadence"] for wm in seg_wmaps],
            "final_cadence": [wm["final_cadence"] for wm in seg_wmaps],
            "sat_pixels": [wm["sat_pixels"] for wm in seg_wmaps],
            "weightmap": np.mean([wm["weightmap"] for wm in seg_wmaps], axis=0),
        }
    else:
        if not args.quiet:
            print("Not splitting")
        tpf, newts, _weights, wmap, pixelvector = do_lc(
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

    _attach_trend_and_whitened(newts)

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
