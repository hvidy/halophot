# Changelog

All notable changes to `halophot` are documented in this file.

## [2.0.0] - 2026-11-14

### Added
- **Zodiax v0.5.0 Convex Simplex Core (`simplex.py`, `models.py`)**:
  - Singularity-free $\mathcal{O}(M)$ Helmert isometric log-ratio (`ilr_to_simplex`, `simplex_to_ilr`) transforms and Charnes–Cooper flux-share bijection ($\theta_i = w_i \bar{P}_i / \sum_j w_j \bar{P}_j$).
  - `HaloWeights(zdx.Base)` isolating `z_ilr` as its sole differentiable leaf, paired with `HaloObservation(zdx.Base)` using bucketed `N_pad` time padding for XLA compilation reuse across targets.
- **Gap-Aware Objectives & Regularization (`objectives.py`, `models.py`, `optimise.py`)**:
  - Cancellation-free Charbonnier total variation (`tv`, `tv_o2`) strictly evaluated over intra-orbit lag-$\ell$ cadence pairs without cross-gap fallbacks across spacecraft downlinks or safe modes.
  - $\mathcal{O}(N)$ `jax.lax.scan` multi-lag `tv_exp` (`< 10 MB` memory footprint on 20-second *TESS* sectors), plus `l2v`, `l3v`, `owl`, $M$-invariant aperture photon-noise regularization ($\lambda_{\text{phot}}$ calibrated against the combined aperture lag-difference variance floor `2 * ap_var_floor` with degrees-of-freedom scaling when $M \gtrsim N$ and a `UserWarning` at $M \ge 0.5\,|V_\ell|$), logit damping ($\lambda_z$), and `legacy_tv=True` *K2* / `halo-tess` DR1 compatibility.
- **Solvers, Uncertainties & Outlier Clipping (`optimise.py`)**:
  - Locally scoped `@in_float64` optimization via `optimistix.BestSoFarMinimiser(optimistix.LBFGS(...))` and optional `zodiax.optimisation` (`optax`) with rank-guarded Gauss–Newton Fisher preconditioning (`safe_eigen_projection`).
  - Cadence-scaled Savitzky–Golay outlier clipping (`mad_std`, finite-cadence smoothing) and $K$-block warm-started orbit jackknife pixel weight uncertainties (`orbit_block_jackknife`).
- **Spatial Masking & WCS Contaminant Exclusion (`masking.py`)**:
  - Automatic `PIXEL_QUALITY` exclusion (`STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK | BKG_USED` ensuring $\Omega_{\text{halo}} \cap \Omega_{\text{bg}} = \emptyset$), physical CCD full-well saturation rate thresholds (*TESS* and *Kepler*/*K2*) with 2D bleed dilation, and sky-dominance ($\bar{P}_i \ge 2 B_i$) / pixel $\operatorname{SNR}_i \ge 5$ guards.
  - WCS-projected background-star circular masking (`mask_background_stars`, `remove_stars`) calibrated to plate scale and PRF encircled energy, with proper-motion epoch propagation and `--offline` pre-cached catalogue support.
- **Functional `lightkurve` API & Coherent `flux_err` (`lightcurve.py`, `_contract.py`)**:
  - `extract_halo_lightcurve(tpf, ...)` returning standard `KeplerLightCurve` / `TessLightCurve` objects with genuine `estimate_centroids()` centroids, separate `pos_corr1` / `pos_corr2` columns, non-transposed `(ny, nx)` (or `(n_splits, ny, nx)`) `lc.meta['WEIGHTMAP']`, and lossless `HALOCVER = 1` FITS serialization (`write_fits` / `write_lightcurve_fits`, `read_weightmap`, `read_split_times`, `WEIGHTMAP_ERR`, provenance header cards `BKGSUB`, `BKG_ORD`, `HALO_OBJ`, `HALO_LAG`, `LEGACYTV`, `INSTRUME = 'TESS Photometer'`, and physical `e-/s` `TUNIT` round-tripping when `normalize=False`).
  - Unified analytical `float64` error propagation (`compute_coherent_flux_err`) combining uncorrelated pixel variance, degree-$\le d$ column-orthogonal projected `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` background covariance without double-counting.
  - Standardized `resolve_cache_dir` supporting `$HALOPHOT_CACHE_DIR` -> `$TESS_BRIGHT_CACHE_DIR` -> `$SCRATCH/.cache/tess_bright` -> `$XDG_CACHE_HOME/tess_bright` -> `~/.cache/tess_bright` with `strict_hpc=True` default `/fred/` inode protection and `create=True` support.
- **CLI, Packaging & Swinburne NT / OzSTAR HPC Templates (`cli.py`, `hpc/`, `pyproject.toml`, `LICENSE`)**:
  - Added `--hlsp-fits` (using v2.0 `minflux=-100.0`, `thresh=-1.0` defaults, mission-gated *K2* BJD cuts, and *TESS* `SXX` plot titles), `--uncertainty`, `--legacy-tv`, `--catalog-table`, `--offline`, `--strict-hpc`, and `--cache-dir` CLI options.
  - Added root-mapped `python:3.12.7-bookworm` Apptainer recipe (`hpc/virgil_halophot.def`), `trevor` builder (`hpc/build_halo_sif.sbatch`), and `tooarrana` CPU/A100 survey Slurm templates (`hpc/halophot_survey_cpu.sbatch`, `hpc/halophot_survey_gpu.sbatch`) with `PIPELINE_MODE=full_e2e` support, plus `[project.optional-dependencies] hpc` and GNU GPL v3.0 `LICENSE`.

### Deprecated
- Legacy `halo_tpf` and `halo_tpf_tess` classes and `.halo()` shims now emit `DeprecationWarning` in favour of `halophot.extract_halo_lightcurve`.

## [2.0.0a1] - 2026-11-11

### Changed
- Modernized package layout (`src/halophot/`), `pyproject.toml` metadata, JAX / Equinox / Optimistix transition from Autograd / TensorFlow / PyTorch, locally scoped `@in_float64` precision context, and initial `HALOCVER = 1` FITS contract helpers.

## [1.0.2] - 2021-02-01

### Added
- Legacy *Kepler*/*K2* and *TESS* total-variation halo photometry (`halo_tpf`, `halo_tpf_tess`, `do_lc`) and command-line workflow (`halo`).
