# halophot

## Contributors

[Tim White](https://github.com/hvidy), [Benjamin Pope](https://github.com/benjaminpope), Jonatan Rudrasingam

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

```python
import lightkurve as lk
import halophot

tpf = lk.read("tess...-s_tp.fits")
lc = halophot.extract_halo_lightcurve(
    tpf,
    sigclip=True,
    compute_uncertainty=True,
)
halophot.write_fits(lc, "target_halo_lc.fits")
```

`extract_halo_lightcurve` returns a standard `lightkurve.LightCurve` (`KeplerLightCurve` or `TessLightCurve`) with:

- `flux` and unified coherent `flux_err` propagated in `float64` across uncorrelated pixel noise, degree-$\le d$ column-projected `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` background covariance;
- flux-weighted aperture centroids in `centroid_col` / `centroid_row` (keeping `pos_corr1` / `pos_corr2` separate);
- non-transposed `(ny, nx)` `lc.meta['WEIGHTMAP']` (or `(n_splits, ny, nx)` when `split_times` is supplied), preserved losslessly on disk via `halophot.write_fits(lc, path)`.

Legacy `halo_tpf` / `halo_tpf_tess` shims, `do_lc`, and `legacy_tv=True` remain available for *K2* and `halo-tess` DR1 compatibility. Slurm and Apptainer templates for HPC survey execution are provided in `hpc/`.

### Command-line usage

Call `halo` from the command line as, e.g.

`halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot`

where `--name` is the star name you would like to save the outputs under, and `-c` is the campaign. Key v2.0 CLI flags include:

- `--hlsp-fits`: emit a `HALOCVER = 1` `lightkurve` FITS file with lossless `WEIGHTMAP` (and `SPLIT_TIMES` / `WEIGHTMAP_ERR` when requested), while honouring `--bitmask`, `--do-split` / `--splits` / `--split-cadences` / `-c`, and `--do-plot`;
- `--uncertainty`: compute orbit block-jackknife pixel weight uncertainties (`WEIGHTMAP_ERR`) in `--hlsp-fits` mode;
- `--legacy-tv`: reproduce unregularized $w$-simplex total variation for *K2* and `halo-tess` DR1 compatibility;
- `--deathstar --catalog-table <path>`: mask contaminant stars via WCS using a pre-cached FITS/Parquet/ECSV catalogue table (pair with `--offline` on compute nodes without internet access);
- `--strict-hpc --cache-dir <path>`: enforce zero-`/fred` cache paths on HPC clusters.

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
