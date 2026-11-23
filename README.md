# halophot
[![CI](https://github.com/hvidy/halophot/actions/workflows/github-workflow.yml/badge.svg)](https://github.com/hvidy/halophot/actions/workflows/github-workflow.yml)
[![Licence](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](https://www.gnu.org/licenses/gpl-3.0.html)
[![arXiv](https://img.shields.io/badge/arXiv-1708.07462-blue.svg?style=flat)](https://arxiv.org/abs/1708.07462)
[![arXiv](https://img.shields.io/badge/arXiv-1908.06981-blue.svg?style=flat)](https://arxiv.org/abs/1908.06981)
[![arXiv](https://img.shields.io/badge/arXiv-2602.22472-blue.svg?style=flat)](https://arxiv.org/abs/2602.22472)
[![PyPI version](https://badge.fury.io/py/halophot.svg)](https://badge.fury.io/py/halophot)

## Contributors

Tim White, Benjamin Pope, Jonatan Rudrasingam

## Installation

Simply run

```bash
pip install halophot
```

or clone this repository, enter the directory, and run

```bash
pip install -e .
```

## K2 and TESS Halo Photometry

This package implements Total Variation (TV) halo photometry for *Kepler*/*K2* and *TESS* observations of very bright, saturated stars in JAX.

We minimize an objective (by default 1st-order smooth Charbonnier total variation) of a normalized light curve formed as a non-negative weighted sum of unsaturated halo pixels, $w \in \Delta^{M-1}$ ($w_i \ge 0$, $\sum_i w_i = 1$), parameterized directly via symmetric zero-mean softmax logits. This removes pointing-jitter and scattered-light systematics cleanly for *Kepler* targets brighter than $K_p \sim 6$ and saturated *TESS* targets brighter than $T \sim 4$.

## Basic usage

### Extracting a halo light curve

`halophot.extract_halo_lightcurve` accepts a `lightkurve.TargetPixelFile`, a FITS path, or a raw 3D `(N_cad, ny, nx)` NumPy array, and returns a `HaloLightCurve` (`KeplerLightCurve` / `TessLightCurve` subclass) whose `.to_fits()` preserves all `HALOCVER = 1` extensions (`WEIGHTMAP`, `WEIGHTMAP_ERR`, `SPLIT_TIMES`, `SEGMENT_SCALES`, `POS_CORR1`/`POS_CORR2`):

```python
import lightkurve as lk
import halophot

tpf = lk.read("tess...-s_tp.fits")
lc = halophot.extract_halo_lightcurve(
    tpf,
    sigclip=True,
    compute_uncertainty=True,
)

# Inspect stamp, weightmap, light curve (with sigclip outliers), and periodogram
fig = halophot.plot_diagnostics(lc, tpf)

# Lossless HALOCVER=1 FITS serialization (both lc.to_fits and halophot.write_fits work)
lc.to_fits("target_halo_lc.fits", overwrite=True)

lc_reloaded = lk.read("target_halo_lc.fits")
wmap = halophot.read_weightmap("target_halo_lc.fits")
```

For array-only workflows (without `lightkurve` or `matplotlib`), call `halophot.optimise_halo(pixels, times)` directly on a 2D `(M, N)` pixel array.

### Writing and testing custom JAX objective functions

Every built-in objective lives in [`src/halophot/objectives.py`](src/halophot/objectives.py) (`OBJECTIVES`: `"tv"`, `"tv_o2"`, `"tv_exp"`, `"l2v"`, `"l3v"`, `"owl"`). You can pass **any JAX function** directly to `optimise_halo(..., objective=my_fn)` or `extract_halo_lightcurve(..., objective=my_fn)` with zero edits to `halophot` internals. Parameter names are inspected once before JIT so all three styles work out of the box:

```python
import jax.numpy as jnp
import halophot

# 1. Minimal 1-argument objective: unit-mean 1D light curve on valid cadences
def huber_tv(flux):
    diffs = jnp.diff(flux)
    return jnp.mean(jnp.sqrt(diffs**2 + 1e-8))

lc = halophot.extract_halo_lightcurve(tpf, objective=huber_tv)

# 2. Gap-safe 2-argument objective: receives boolean step_mask of shape (N - lag,)
def gap_safe_l2(flux, step_mask):
    diffs = jnp.diff(flux)
    return jnp.sum(jnp.where(step_mask, diffs**2, 0.0)) / jnp.sum(step_mask)

fit = halophot.optimise_halo(pixels, times, objective=gap_safe_l2)

# 3. Keyword-aware objective: request any subset of
#    (flux, *, step_mask, weights, pixels, times, dt_nom, lag)
def entropy_regularized_tv(flux, *, step_mask, weights):
    diffs = jnp.diff(flux)
    tv = jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-10), 0.0)) / jnp.sum(step_mask)
    entropy_pen = 1e-4 * jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-12)))
    return tv + entropy_pen

fit = halophot.optimise_halo(pixels, times, objective=entropy_regularized_tv)

# 4. Register globally by name so it can be reused across scripts
@halophot.register_objective("my_tv")
def my_tv(flux, step_mask):
    diffs = jnp.diff(flux)
    return jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-10), 0.0)) / jnp.sum(step_mask)

lc = halophot.extract_halo_lightcurve(tpf, objective="my_tv")
```

### Key features

- **Direct Physical Simplex Core**: Zero-mean softmax logits (`w = softmax(u - mean(u))`) optimized in locally scoped `float64` via `optax.lbfgs` + zoom line search without mutating global `jax_enable_x64`.
- **Scale-Invariant Smooth L1 & $\mathcal{O}(N)$ `tv_exp`**: Cancellation-free Charbonnier smoothing scales dynamically to `eps_eff = epsilon * median(|df_init|)` across `0.1 ppm` to `10%` targets, strictly masks spacecraft downlink gaps, and evaluates multi-lag `tv_exp` via `jax.lax.scan` in `< 10 MB` RAM.
- **Unified Coherent `flux_err` & Low-RAM 20-s Processing**: Keeps 3D TPF arrays as `float32` views (`< 2 GiB` host RAM on 20-s *TESS* sectors), excludes `PIXEL_QUALITY` bad pixels, flags `sigclip` outliers in `lc["inlier_mask"]` without discarding astrophysical eclipses, and propagates uncorrelated pixel noise, exact WLS oblique `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` covariance without double-counting.
- **HPC Templates**: Root-mapped Apptainer (`halo.sif` via `hpc/virgil_halophot.def` and `hpc/build_halo_sif.sbatch`) and CPU/A100 Slurm templates (`hpc/halophot_survey_cpu.sbatch`, `hpc/halophot_survey_gpu.sbatch`) for Swinburne Ngarrgu Tindebeek (`tooarrana` / `trevor`) are provided in `hpc/`.

### Command-line usage

Call `halo` from the command line as, e.g.

```bash
halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot
```

Add `--hlsp-fits` (and `--offline --cache-dir "$SCRATCH/.cache/tess_bright"` on HPC clusters; `--strict-hpc` is enabled by default in the CLI and can be relaxed with `--no-strict-hpc`) to emit `HALOCVER = 1` FITS files with `WEIGHTMAP` extensions.

### Swinburne NT / OzSTAR HPC usage

1. **Build the single-inode `halo.sif` container on `trevor`** (internet access required):
   ```bash
   mkdir -p "$SCRATCH/logs"
   sbatch hpc/build_halo_sif.sbatch
   # For pre-tag branch or commit builds:
   # MUNDEY_REF=dev TESSBKGD_REF=dev HALOPHOT_REF=dev sbatch hpc/build_halo_sif.sbatch
   ```
2. **Run CPU array or A100 GPU survey jobs on `tooarrana`**:
   ```bash
   mkdir -p "$SCRATCH/logs"
   sbatch hpc/halophot_survey_cpu.sbatch
   sbatch hpc/halophot_survey_gpu.sbatch
   # Full 3-package pass (mundey -> tessbkgd -> halophot) after Stage-1 `mundey prefetch`:
   sbatch --export=ALL,PIPELINE_MODE=full_e2e hpc/halophot_survey_cpu.sbatch
   sbatch --export=ALL,PIPELINE_MODE=full_e2e hpc/halophot_survey_gpu.sbatch
   ```

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
