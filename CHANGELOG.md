# Changelog

All notable changes to `halophot` are documented in this file.

## [2.0.0] - 2026-11-12

### Added
- **Zodiax v0.5.0 Convex Simplex Core (`simplex.py`, `models.py`)**:
  - Singularity-free $\mathcal{O}(M)$ Helmert isometric log-ratio (`ilr_to_simplex`, `simplex_to_ilr`) transforms and Charnes–Cooper flux-share bijection ($\theta_i = w_i \bar{P}_i / \sum_j w_j \bar{P}_j$).
  - `HaloWeights(zdx.Base)` isolating `z_ilr` as its sole differentiable leaf, paired with `HaloObservation(zdx.Base)` using bucketed `N_pad` time padding for XLA compilation reuse across targets.
- **Gap-Aware Objectives & Regularization (`objectives.py`)**:
  - Cancellation-free Charbonnier total variation (`tv`, `tv_o2`) strictly evaluated over intra-orbit lag-$\ell$ cadence pairs without cross-gap fallbacks across spacecraft downlinks or safe modes.
  - $\mathcal{O}(N)$ `jax.lax.scan` multi-lag `tv_exp` (`< 10 MB` memory footprint on 20-second *TESS* sectors), plus `l2v`, `l3v`, `owl`, photon-noise regularization ($\lambda_{\text{phot}}$), logit damping ($\lambda_z$), and `legacy_tv=True` *K2* / `halo-tess` DR1 compatibility.
- **Solvers, Uncertainties & Outlier Clipping (`optimise.py`)**:
  - Locally scoped `@in_float64` optimization via `optimistix.BestSoFarMinimiser(optimistix.LBFGS(...))` and optional `zodiax.optimisation` (`optax`) with rank-guarded Gauss–Newton Fisher preconditioning (`safe_eigen_projection`).
  - Cadence-scaled Savitzky–Golay outlier clipping (`mad_std`, finite-cadence smoothing) and $K$-block warm-started orbit jackknife pixel weight uncertainties (`orbit_block_jackknife`).
- **Spatial Masking & WCS Contaminant Exclusion (`masking.py`)**:
  - Automatic `PIXEL_QUALITY` exclusion (`STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK | BKG_USED` ensuring $\Omega_{\text{halo}} \cap \Omega_{\text{bg}} = \emptyset$), physical CCD full-well saturation rate thresholds (*TESS* and *Kepler*/*K2*) with 2D bleed dilation, and sky-dominance ($\bar{P}_i \ge 2 B_i$) / pixel $\operatorname{SNR}_i \ge 5$ guards.
  - WCS-projected background-star circular masking (`mask_background_stars`, `remove_stars`) calibrated to plate scale and PRF encircled energy, with proper-motion epoch propagation and `--offline` pre-cached catalogue support.
- **Functional `lightkurve` API & Coherent `flux_err` (`lightcurve.py`, `_contract.py`)**:
  - `extract_halo_lightcurve(tpf, ...)` returning standard `KeplerLightCurve` / `TessLightCurve` objects with genuine `estimate_centroids()` centroids, separate `pos_corr1` / `pos_corr2` columns, non-transposed `(ny, nx)` (or `(n_splits, ny, nx)`) `lc.meta['WEIGHTMAP']`, and lossless FITS serialization (`write_lightcurve_fits`, `SPLIT_TIMES`, `WEIGHTMAP_ERR`).
  - Unified analytical `float64` error propagation (`compute_coherent_flux_err`) combining uncorrelated pixel variance, degree-$\le d$ column-orthogonal projected `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` background covariance without double-counting.
- **CLI & HPC Templates (`cli.py`, `hpc/`)**:
  - Added `--hlsp-fits`, `--uncertainty`, `--legacy-tv`, `--catalog-table`, `--offline`, `--strict-hpc`, and `--cache-dir` CLI options, plus Swinburne Ngarrgu Tindebeek (`tooarrana`) CPU/A100 Slurm templates and pinned Apptainer recipe in `hpc/`.

## [2.0.0a1] - 2026-11-11

### Changed
- Modernized package layout (`src/halophot/`), `pyproject.toml` metadata, JAX / Equinox / Optimistix transition from Autograd / TensorFlow / PyTorch, locally scoped `@in_float64` precision context, and initial `HALOCVER = 1` FITS contract helpers.

## [1.0.2] - 2021-02-01

### Added
- Legacy *Kepler*/*K2* and *TESS* total-variation halo photometry (`halo_tpf`, `halo_tpf_tess`, `do_lc`) and command-line workflow (`halo`).
