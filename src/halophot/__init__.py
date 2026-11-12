"""halophot: Kepler/K2 and TESS halo photometry in JAX."""

from . import halo_objectives, halo_tools
from ._contract import (
    HALOCVER,
    BkgCovariance,
    PixelQuality,
    pack_bkg_chol,
    read_bkg_covariance,
    read_bkgsub_state,
    read_pixel_quality,
    read_smear_err,
    resolve_cache_dir,
    write_lightcurve_fits,
)
from ._precision import cast_tree, in_float64, run_in

name = "halophot"
__version__ = "2.0.0a1"

write_fits = write_lightcurve_fits

__all__ = [
    "HALOCVER",
    "BkgCovariance",
    "PixelQuality",
    "__version__",
    "cast_tree",
    "halo_objectives",
    "halo_tools",
    "in_float64",
    "name",
    "pack_bkg_chol",
    "read_bkg_covariance",
    "read_bkgsub_state",
    "read_pixel_quality",
    "read_smear_err",
    "resolve_cache_dir",
    "run_in",
    "write_fits",
    "write_lightcurve_fits",
]
