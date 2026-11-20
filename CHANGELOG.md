# Changelog

All notable changes to `halophot` are documented in this file.

## [2.0.1] - 2026-11-20

### Fixed
- **Detector Error Propagation & Coherent `flux_err` (`lightcurve.py`, `halo_tools.py`)**:
  - Replaced unweighted Euclidean column projection with exact per-cadence WLS oblique column-smear sensitivity ($\tilde{g}_{t,c} = g_c - \phi_w^\top s_{t,c}$) reconstructed from `BKG_BASIS`, `BKG_CHOL`, `PIXEL_QUALITY & PixelQuality.BKG_USED`, and unsubtracted pixel inverse variances (with rank-safe pseudoinverse protection and fallback to `column_orthogonal_smear_weights` only when `BKG_BASIS` or `BKG_USED` is absent).
  - Added a `float32`-ULP-aware variance floor in `compute_coherent_flux_err` to prevent FITS rounding cancellation in `FLUX_ERR^2 - SMEAR_ERR^2` from collapsing independent pixel noise to `0.0`.
  - Fixed `pvar_reg` in `extract_halo_lightcurve` to average over finite cadences only (`nanmean` instead of zero-filled `NaN`s) and subtract coherent `SMEAR_ERR^2` alongside background variance.
- **Outlier Clipping, Split Continuity & Centroids (`lightcurve.py`)**:
  - Preserved 100% of finite cadences (including deep eclipses such as Algol) in `extract_halo_lightcurve(..., sigclip=True)` so outlier clipping excludes bad cadences only during weight optimization and baseline normalization.
  - Anchored `normalize=False` multi-segment `split_times` light curves to a common full-sector reference aperture flux so shifting simplex weights between bright and faint halo wings never creates artificial step discontinuities.
  - Replaced unweighted `estimate_centroids` calls with positive-flux, weightmap-weighted centroids (with safe fallback to unweighted weight-centroids when `denom <= 0`) so negative background-subtracted pixels cannot cause centroid blow-ups.
- **`HALOCVER = 1` FITS Contract (`_contract.py`)**:
  - Preserved `POS_CORR1` and `POS_CORR2` columns in HDU 1 of `write_lightcurve_fits` so `lk.read` round-trips pointing corrections cleanly.
  - Sliced axis `0` of 3D `WEIGHTMAP` and `WEIGHTMAP_ERR` to the surviving contiguous segment range (squeezing to 2D when 1 segment survives) when re-saving time-sliced split light curves so `NSPLITS + 1` always matches the `WEIGHTMAP` segment count.
- **Spatial Masking & WCS Contaminant Exclusion (`masking.py`, `lightcurve.py`, `cli.py`)**:
  - Added fallback `target_mag` estimation from TPF headers (`TESSMAG`/`TMAG`/`KEPMAG`) or integrated stamp flux so missing $G < 3$ primaries in Gaia never mistake a faint neighbour for the primary, and prevented blended visual-binary companions ($\alpha$ Cen A/B, Castor A/B) from masking the target core.
  - Exposed `catalog` in `build_halo_masks` and `extract_halo_lightcurve`, inferred Gaia J2016.0 epoch automatically on Gaia+`Tmag`/`Gmag` tables unless TIC columns indicate J2000, suppressed `ErfaWarning` in `apply_space_motion`, and passed `obs_time` in `halo --deathstar`.
  - Enabled sky-dominance filtering ($\bar{U}_i - \bar{B}_i \ge \kappa \bar{B}_i$) when `BKGSUB=False`, and switched `BKGSUB=True` sky filtering to dark-orbit 10th-percentile background plus `FLUX_BKG_ERR` uncertainty guards so high-SNR wings under subtracted Earthshine are preserved.
  - Upgraded saturation detection to 99th-percentile unsubtracted flux (catching partial-sector bleed), always honoured explicit `auto_sat_level` alongside `SAT_BLEED`, converted integrated `electron`/`count` TPFs to `e-/s`, and defaulted `dt_days` for `min_snr` when `flux_err=None`.
- **Math & Optimization Core (`models.py`, `objectives.py`, `optimise.py`, `simplex.py`)**:
  - Fixed `lambda_phot` jitter/pulsator `ref_var` calibration, `tv_o2`/`owl` matched `b_hat`, gap-tolerant multi-lag `tv_exp_scan` pair counts, fixed-cadence `gap_tol` for `lag >= 4`, `n_eff`-normalized L-BFGS/jackknife convergence tolerances, leave-one-out negative-mean jackknife stability, and `@in_float64` coverage on public helpers.
- **CLI, Legacy Tools, Docs & HPC Templates (`cli.py`, `halo_tools.py`, `mkdocs.yml`, `hpc/`)**:
  - Resolved integer `--split-cadences` and K2 `-c 4` presets on raw unfiltered HDU 1 cadence indices before `quality_bitmask` filtering, warned on mode-mismatched CLI flags, and replaced degree-15 monomial trend fitting with gap-aware Savitzky–Golay detrending.
  - Fixed inclusive `final_cadence = high - 1` and `(0, len(ts))` boundary handling in `do_lc`.
  - Hardened Swinburne NT HPC survey scripts with sector-disambiguated GPU output stems, `-sub` catalogue fallback, configurable `MAX_FAIL_FRAC` non-zero exit codes, atomic `.tar.part` -> `mv` Lustre archiving with `$SCRATCH/stage` cleanup, array-size manifest warnings, and `v2.0.1` default Apptainer build refs.
  - Modernized `mkdocs.yml`, added `[project.optional-dependencies] docs`, and refreshed `docs/notebooks/aldebaran.ipynb` onto the v2.0.1 API.

## [2.0.0] - 2026-11-14

### Added
- **Zodiax v0.5.0 Convex Simplex Core (`simplex.py`, `models.py`)**:
  - Singularity-free $\mathcal{O}(M)$ Helmert isometric log-ratio (`ilr_to_simplex`, `simplex_to_ilr`) transforms and Charnes–Cooper flux-share bijection ($\theta_i = w_i \bar{P}_i / \sum_j w_j \bar{P}_j$).
  - `HaloWeights(zdx.Base)` isolating `z_ilr` as its sole differentiable leaf, paired with `HaloObservation(zdx.Base)` using bucketed `N_pad` time padding for XLA compilation reuse across targets.
- **Gap-Aware Objectives & Regularization (`objectives.py`, `models.py`, `optimise.py`)**:
  - Cancellation-free Charbonnier total variation (`tv`, `tv_o2`) strictly evaluated over intra-orbit lag-$\ell$ cadence pairs without cross-gap fallbacks across spacecraft downlinks or safe modes.
  - $\mathcal{O}(N)$ `jax.lax.scan` multi-lag `tv_exp` (`< 10 MB` memory footprint on 20-second *TESS* sectors), plus `l2v`, `l3v`, `owl`, $M$-invariant aperture photon-noise regularization ($\lambda_{\text{phot}}$ calibrated against the combined aperture variance floor `ap_var_floor` via the linear Marchenko–Pastur tangent-space discrepancy balance `(0.1 + 40 * sqrt(2 / pi) * r) * n_eff / ref_var` for $r = (M - 1) / n_{\text{eff}}$ with objective-aware constraint counts and a `UserWarning` at $M \ge 0.5\,n_{\text{eff}}$), bounded ILR gauge damping ($\lambda_z = 10^{-3}$), and `legacy_tv=True` *K2* / `halo-tess` DR1 compatibility.
- **Solvers, Uncertainties & Outlier Clipping (`optimise.py`)**:
  - Locally scoped `@in_float64` optimization via `optimistix.BestSoFarMinimiser(optimistix.LBFGS(...))` and optional `zodiax.optimisation` (`optax`) with rank-guarded Gauss–Newton Fisher preconditioning (`safe_eigen_projection`).
  - Cadence-scaled Savitzky–Golay outlier clipping (`mad_std`, finite-cadence smoothing) and $K$-block warm-started orbit jackknife pixel weight uncertainties (`orbit_block_jackknife`).
- **Spatial Masking & WCS Contaminant Exclusion (`masking.py`)**:
  - Automatic `PIXEL_QUALITY` exclusion (`STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK | BKG_USED` ensuring $\Omega_{\text{halo}} \cap \Omega_{\text{bg}} = \emptyset$), physical CCD full-well saturation rate thresholds (*TESS* and *Kepler*/*K2*) with 2D bleed dilation, and sky-dominance ($\bar{P}_i \ge 2 B_i$) / pixel $\operatorname{SNR}_i \ge 5$ guards.
  - WCS-projected background-star circular masking (`mask_background_stars`, `remove_stars`) calibrated to plate scale and PRF encircled energy, with proper-motion epoch propagation and `--offline` pre-cached catalogue support.
- **Functional `lightkurve` API & Coherent `flux_err` (`lightcurve.py`, `_contract.py`)**:
  - `extract_halo_lightcurve(tpf, ...)` returning standard `KeplerLightCurve` / `TessLightCurve` objects with genuine `estimate_centroids()` centroids, separate `pos_corr1` / `pos_corr2` columns, non-transposed `(ny, nx)` (or `(n_splits, ny, nx)`) `lc.meta['WEIGHTMAP']`, and lossless `HALOCVER = 1` FITS serialization (`write_fits` / `write_lightcurve_fits`, `read_weightmap`, `read_weightmap_err`, `read_split_times`, provenance header cards `BKGSUB`, `BKG_ORD`, `HALO_OBJ`, `HALO_LAG`, `LEGACYTV`, `INSTRUME = 'TESS Photometer'`, and physical `e-/s` `TUNIT` round-tripping when `normalize=False`).
  - Unified analytical `float64` error propagation (`compute_coherent_flux_err`) combining uncorrelated pixel variance, degree-$\le d$ column-orthogonal projected `SMEAR_ERR`, and `BKG_CHOL` / `BKG_BASIS` background covariance without double-counting.
  - Standardized `resolve_cache_dir` supporting `$HALOPHOT_CACHE_DIR` -> `$TESS_BRIGHT_CACHE_DIR` -> `$SCRATCH/.cache/tess_bright` -> `$XDG_CACHE_HOME/tess_bright` -> `~/.cache/tess_bright` with `strict_hpc=True` default `/fred/` inode protection and `create=True` support, wired through `mask_background_stars`, `remove_stars`, `build_halo_masks`, `extract_halo_lightcurve`, and `halo`.
- **CLI, Packaging & Swinburne NT / OzSTAR HPC Templates (`cli.py`, `hpc/`, `pyproject.toml`, `LICENSE`)**:
  - Added `--hlsp-fits` (using v2.0 `minflux=-100.0`, `thresh=-1.0` defaults, mission-gated *K2* BJD cuts, and *TESS* `SXX` / *Kepler* `QXX` plot titles), `--uncertainty`, `--legacy-tv`, `--catalog-table`, `--offline`, `--strict-hpc` / `--no-strict-hpc` (default `True`), and `--cache-dir` CLI options.
  - Added root-mapped `python:3.12.7-bookworm` Apptainer recipe (`hpc/virgil_halophot.def`), `trevor` builder (`hpc/build_halo_sif.sbatch`), and `tooarrana` CPU/A100 survey Slurm templates (`hpc/halophot_survey_cpu.sbatch`, `hpc/halophot_survey_gpu.sbatch`) with `PIPELINE_MODE=full_e2e` support, plus `[project.optional-dependencies] hpc` and GNU GPL v3.0 `LICENSE`.

### Deprecated
- Legacy `halo_tpf` and `halo_tpf_tess` classes and `.halo()` shims now emit `DeprecationWarning` in favour of `halophot.extract_halo_lightcurve`.

## [2.0.0a1] - 2026-11-11

### Changed
- Modernized package layout (`src/halophot/`), `pyproject.toml` metadata, JAX / Equinox / Optimistix transition from Autograd / TensorFlow / PyTorch, locally scoped `@in_float64` precision context, and initial `HALOCVER = 1` FITS contract helpers.

## [1.0.2] - 2021-02-01

### Added
- Legacy *Kepler*/*K2* and *TESS* total-variation halo photometry (`halo_tpf`, `halo_tpf_tess`, `do_lc`) and command-line workflow (`halo`).
