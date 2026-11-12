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

`pip install halophot`

or else clone this git repository, enter the directory, and run

`pip install -e .`

## K2 and TESS Halo Photometry

This is code that implements a Total Variation (TV) based regularization for *Kepler*/*K2* and *TESS* photometry of very bright stars.

We minimize nth order TV - i.e. the sum of the absolute values of the nth differences of a light curve - of a light curve created as the weighted sum of pixels, with weights in (0,1). This appears remarkably effective at removing pointing-based and scattered-light systematics from *K2* and *TESS* light curves where it is impractical to do photometry otherwise and apply more standard detrending methods.

We believe this is of practical use for *Kepler* targets brighter than Kp ~ 6 and saturated *TESS* targets brighter than T ~ 4.

## Basic usage

### Functional `lightkurve` API (`halophot` v2.0)

`halophot` v2.0 operates directly on standard `lightkurve.TargetPixelFile` objects (including `HALOCVER = 1` products from `mundey` and `tessbkgd`) via convex Charnes–Cooper flux-share coordinates parameterized on the simplex with singularity-free isometric log-ratio (ILR) transforms in JAX and Zodiax:

```python
import lightkurve as lk
import halophot

tpf = lk.read("tess...-s_tp.fits")
lc = halophot.extract_halo_lightcurve(
    tpf,
    sigclip=True,
    compute_uncertainty=True,
)
# Save HLSP FITS with lossless WEIGHTMAP ImageHDU and coherent flux_err
halophot.write_fits(lc, "target_halo_lc.fits")
```

Key v2.0 features:

- **Convex Charnes–Cooper + ILR Simplex Core**: `HaloWeights(zdx.Base)` isolates `z_ilr` as its sole differentiable leaf, optimized in locally scoped `float64` precision via `optimistix.LBFGS` (or `zodiax.optimisation` with rank-guarded eigen-projection) without mutating global `jax_enable_x64`.
- **Gap-Aware Charbonnier TV & $\mathcal{O}(N)$ `tv_exp`**: Smooth Charbonnier total variation strictly masks spacecraft downlink and safe-mode gaps, while `jax.lax.scan` evaluates multi-lag `tv_exp` in `< 10 MB` RAM even on 20-second *TESS* sectors.
- **Sky-Dilution Guard & Unified Coherent `flux_err`**: Automatically excludes `PIXEL_QUALITY` strap, bleed, unresolved-smear, background-star, and `BKG_USED` pixels ($\Omega_{\text{halo}} \cap \Omega_{\text{bg}} = \emptyset$), enforces $\bar{P}_i \ge 2\,\operatorname{median}_t(B_{i,t})$ and $\operatorname{SNR}_i \ge 5$, and propagates pixel, column-orthogonalized `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` uncertainties without double-counting. Pass `legacy_tv=True` to reproduce unregularized *K2* / `halo-tess` DR1 weights.
- **HPC Templates**: Slurm array and Apptainer recipes for Swinburne Ngarrgu Tindebeek (`tooarrana`) are provided in `hpc/`.

### Command-line usage

Call `halo` from the command line as, e.g.

`halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot`

where `--name` is the star name you would like to save the outputs under, and `-c` is the campaign. Add `--hlsp-fits` (and `--offline --strict-hpc` on HPC clusters) to emit `HALOCVER = 1` `lightkurve` FITS files with `WEIGHTMAP` extensions.

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
