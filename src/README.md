# Source Code

Package sources live under `src/halophot/`:

- `__init__.py`: package metadata and public exports.
- `_precision.py`: locally scoped `@in_float64` JAX precision helpers.
- `_contract.py`: versioned `HALOCVER = 1` FITS contract readers, writers (`WEIGHTMAP`, `WEIGHTMAP_ERR`, `SPLIT_TIMES`), and HPC cache path utilities.
- `simplex.py`: singularity-free $\mathcal{O}(M)$ Helmert isometric log-ratio (ILR) transforms and Charnes–Cooper flux-share bijection.
- `models.py`: Zodiax v0.5.0 PyTrees (`HaloWeights`, `HaloObservation`) with bucketed `N_pad` compilation reuse.
- `objectives.py`: gap-masked Charbonnier total variation (`tv`, `tv_o2`), $\mathcal{O}(N)$ `jax.lax.scan` `tv_exp`, `l2v`, `l3v`, `owl`, photon-noise regularization, and `legacy_tv=True` compatibility.
- `optimise.py`: `optimistix.LBFGS` and `zodiax.optimisation` solvers, rank-guarded Gauss–Newton Fisher preconditioning, cadence-scaled Savitzky–Golay clipping, and orbit block-jackknife weight uncertainties.
- `masking.py`: `PIXEL_QUALITY` exclusion, calibrated CCD full-well saturation + 2D bleed dilation, sky-dominance / SNR guards, and WCS-projected contaminant-star masking (`remove_stars`).
- `lightcurve.py`: functional `lightkurve` API (`extract_halo_lightcurve`) and unified coherent `flux_err` propagation (`compute_coherent_flux_err`).
- `halo_objectives.py`: legacy softmax total-variation objective functions in JAX.
- `halo_tools.py`: legacy halo photometry extraction (`do_lc`, `halo_tpf`, `halo_tpf_tess`) and diagnostic plotting routines.
- `cli.py`: `halo` command-line interface (classic and `--hlsp-fits` HLSP modes).
