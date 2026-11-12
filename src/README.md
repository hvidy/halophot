# Source Code

Package sources live under `src/halophot/`:

- `__init__.py`: package metadata and public exports.
- `_precision.py`: locally scoped `@in_float64` JAX precision helpers.
- `_contract.py`: versioned `HALOCVER = 1` FITS contract readers, writers, and HPC cache path utilities.
- `halo_objectives.py`: total-variation and smoothness objective functions in JAX.
- `halo_tools.py`: halo photometry extraction, masking, and plotting routines.
- `cli.py`: `halo` command-line interface.
