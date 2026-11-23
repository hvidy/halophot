# halophot

## Contributors

[Tim White](https://github.com/hvidy), [Benjamin Pope](https://github.com/benjaminpope), Jonatan Rudrasingam

## Installation

Simply run

```bash
pip install halophot
```

or clone this git repository, enter the directory, and run

```bash
pip install -e .
```

## K2 and TESS Halo Photometry

This package implements Total Variation (TV) halo photometry for *Kepler*/*K2* and *TESS* observations of very bright, saturated stars in JAX.

We minimize a smoothness objective (by default 1st-order smooth Charbonnier total variation) of a normalized light curve created as the non-negative weighted sum of unsaturated halo pixels, $w \in \Delta^{M-1}$, parameterized directly via symmetric zero-mean softmax logits. This is effective for *Kepler* targets brighter than $K_p \sim 6$ and saturated *TESS* targets brighter than $T \sim 4$.

## Basic usage

### Functional `lightkurve` API

`extract_halo_lightcurve` accepts a `lightkurve.TargetPixelFile`, a TPF FITS path, or a raw 3D `(N_cad, ny, nx)` NumPy array, and returns a `HaloLightCurve` (`KeplerLightCurve` / `TessLightCurve` subclass) whose `.to_fits()` delegates directly to `write_lightcurve_fits`:

```python
import lightkurve as lk
import halophot

tpf = lk.read("tess...-s_tp.fits")
lc = halophot.extract_halo_lightcurve(
    tpf,
    sigclip=True,
    compute_uncertainty=True,
)

# Visual diagnostic summary (stamp, weightmap, light curve, Lomb-Scargle PSD)
fig = halophot.plot_diagnostics(lc, tpf)

# Save and reload with lossless WEIGHTMAP, WEIGHTMAP_ERR, SPLIT_TIMES, SEGMENT_SCALES, POS_CORR1/2
lc.to_fits("target_halo_lc.fits", overwrite=True)
lc_reloaded = lk.read("target_halo_lc.fits")
wmap = halophot.read_weightmap("target_halo_lc.fits")
```

Returned `HaloLightCurve` objects include:

- `flux`, unified coherent `flux_err` (uncorrelated pixel noise with `float32`-ULP and detector shot-noise floor, WLS oblique `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` covariance), and `lc["inlier_mask"]` marking cadences kept during `sigclip` weight fitting;
- positive-flux weightmap-weighted `centroid_col` / `centroid_row` alongside preserved `pos_corr1` / `pos_corr2`;
- native `(ny, nx)` `lc.meta["WEIGHTMAP"]` (or `(n_splits, ny, nx)` with `split_times`), preserved automatically by both `lc.to_fits(path)` and `halophot.write_fits(lc, path)`.

### Writing custom JAX objective functions

All built-in objectives (`"tv"`, `"tv_o2"`, `"tv_exp"`, `"l2v"`, `"l3v"`, `"owl"`) live in `halophot.objectives.OBJECTIVES`. You can test your own JAX objective function directly without modifying `halophot`:

```python
import jax.numpy as jnp
import halophot

# 1-argument objective on unit-mean 1D flux
def my_tv(flux):
    diffs = jnp.diff(flux)
    return jnp.mean(jnp.sqrt(diffs**2 + 1e-8))

lc = halophot.extract_halo_lightcurve(tpf, objective=my_tv)

# Gap-safe 2-argument objective receiving boolean step_mask of shape (N - lag,)
def gap_safe_tv(flux, step_mask):
    diffs = jnp.diff(flux)
    return jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-10), 0.0)) / jnp.sum(step_mask)

fit = halophot.optimise_halo(pixels, times, objective=gap_safe_tv)

# Keyword-aware objective requesting any subset of
# (flux, *, step_mask, weights, pixels, times, dt_nom, lag)
def weighted_custom_obj(flux, *, step_mask, weights):
    diffs = jnp.diff(flux)
    tv = jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-10), 0.0)) / jnp.sum(step_mask)
    return tv + 1e-3 * jnp.sum(weights**2)

fit = halophot.optimise_halo(pixels, times, objective=weighted_custom_obj)

# Register by name in halophot.OBJECTIVES
@halophot.register_objective("l2_entropy")
def l2_entropy(flux, *, step_mask, weights):
    diffs = jnp.diff(flux)
    l2 = jnp.sum(jnp.where(step_mask, diffs**2, 0.0)) / jnp.sum(step_mask)
    return l2 + 1e-4 * jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-12)))

lc = halophot.extract_halo_lightcurve(tpf, objective="l2_entropy")
```

Structured options can be grouped into `halophot.MaskConfig(...)` and `halophot.OptimConfig(...)` or passed directly as keyword arguments. Legacy `halo_tpf` / `halo_tpf_tess` shims remain available (emitting `DeprecationWarning`).

### Command-line usage

Call `halo` from the command line as, e.g.

```bash
halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot
```

Key CLI flags:

- `--hlsp-fits`: run `extract_halo_lightcurve` and write a `HALOCVER = 1` FITS file with lossless `WEIGHTMAP` (plus `SPLIT_TIMES`, `SEGMENT_SCALES`, and `WEIGHTMAP_ERR` when requested);
- `--uncertainty`: compute orbit block-jackknife pixel weight uncertainties (`WEIGHTMAP_ERR`) in `--hlsp-fits` mode;
- `--sigclip` / `--no-sigclip`: exclude local flux outliers during weight optimization and record `inlier_mask`;
- `--deathstar --catalog-table <path>`: mask contaminant stars via WCS using a pre-cached FITS/Parquet/ECSV catalogue table (pair with `--offline` on compute nodes);
- `--cache-dir <path>` and `--strict-hpc` / `--no-strict-hpc`: guard HPC `/fred/` inode quotas when performing live catalogue downloads.

## Licence

We invite anyone interested to use and modify this code under a GPL v3 licence.

## Citation

We request that anyone using this code for photometry of stars observed under the bright star GO programmes in *K2* include White and Pope as coauthors on any publications.

If you use our code, please cite

    White et al. (2017), MNRAS, 471, 2882-2901, arXiv:1708.07462

    Pope et al. (2019), ApJS, 245, 8, arXiv:1908.06981

and

    Rudrasingam et al. (2026), MNRAS, 547, stag413, arXiv:2602.22472

Or use these BibTeX entries:

    @ARTICLE{White2017,
       author = {{White}, T.~R. and {Pope}, B.~J.~S. and {Antoci}, V. and {P{\'a}pics}, P.~I. and
      {Aerts}, C. and {Gies}, D.~R. and {Gordon}, K. and {Huber}, D. and
      {Schaefer}, G.~H. and {Aigrain}, S. and {Albrecht}, S. and {Barclay}, T. and
      {Barentsen}, G. and {Beck}, P.~G. and {Bedding}, T.~R. and {Fredslund Andersen}, M. and
      {Grundahl}, F. and {Howell}, S.~B. and {Ireland}, M.~J. and
      {Murphy}, S.~J. and {Nielsen}, M.~B. and {Silva Aguirre}, V. and
      {Tuthill}, P.~G.},
        title = "{Beyond the Kepler/K2 bright limit: variability in the seven brightest members of the Pleiades}",
      journal = {\mnras},
    archivePrefix = "arXiv",
       eprint = {1708.07462},
     primaryClass = "astro-ph.SR",
     keywords = {asteroseismology, techniques: photometric, stars: early type, stars: variables: general, open clusters and associations: individual: Pleiades},
         year = 2017,
        month = nov,
       volume = 471,
        pages = {2882-2901},
          doi = {10.1093/mnras/stx1050},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.2882W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{Pope2019,
       author = {{Pope}, Benjamin J.~S. and {White}, Timothy R. and {Farr}, Will M. and {Yu}, Jie and {Greklek-McKeon}, Michael and {Huber}, Daniel and {Aerts}, Conny and {Aigrain}, Suzanne and {Bedding}, Timothy R. and {Boyajian}, Tabetha and {Creevey}, Orlagh L. and {Hogg}, David W.},
        title = "{The K2 Bright Star Survey. I. Methodology and Data Release}",
      journal = {\apjs},
     keywords = {Asteroseismology, Photometry, Astronomical techniques, Eclipsing binary stars, Red giant stars, Variable stars, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = nov,
       volume = {245},
       number = {1},
          eid = {8},
        pages = {8},
          doi = {10.3847/1538-4365/ab3d29},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJS..245....8P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{Rudrasingam2026,
       author = {{Rudrasingam}, Jonatan and {Bedding}, Timothy R. and {Pope}, Benjamin J.~S. and {Pedersen}, May Gade and {Lund}, Mikkel N. and {White}, Timothy R. and {Hey}, Daniel},
        title = "{Halo Photometry and Asteroseismology for 98 of the Brightest Stars Observed by TESS}",
      journal = {\mnras},
    archivePrefix = "arXiv",
       eprint = {2602.22472},
     primaryClass = "astro-ph.SR",
         year = 2026,
       volume = {547},
          eid = {stag413},
        pages = {stag413},
          doi = {10.1093/mnras/stag413},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026MNRAS.547..413R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
