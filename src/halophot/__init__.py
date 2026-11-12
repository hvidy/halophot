"""halophot: Kepler/K2 and TESS halo photometry in JAX and Zodiax."""

from . import (
    halo_objectives,
    halo_tools,
    lightcurve,
    masking,
    models,
    objectives,
    optimise,
    simplex,
)
from ._contract import (
    HALOCVER,
    BkgCovariance,
    PixelQuality,
    infer_bkg_poly_order,
    pack_bkg_chol,
    read_bkg_covariance,
    read_bkgsub_state,
    read_pixel_quality,
    read_smear_err,
    resolve_cache_dir,
    write_lightcurve_fits,
)
from ._precision import cast_tree, in_float64, run_in
from .halo_tools import halo_tpf, halo_tpf_tess
from .lightcurve import (
    column_orthogonal_smear_weights,
    compute_coherent_flux_err,
    extract_halo_lightcurve,
)
from .masking import (
    DEFAULT_EXCLUDE_QUALITY,
    HaloMaskResult,
    build_halo_masks,
    mask_background_stars,
    remove_stars,
)
from .models import HaloObservation, HaloWeights, bucket_pad_length, compute_gap_diff_mask
from .objectives import (
    AVAILABLE_OBJECTIVES,
    charbonnier_penalty,
    halo_objective,
    tv_charbonnier_o1,
    tv_charbonnier_o2,
    tv_exp_scan,
)
from .optimise import (
    HaloFitResult,
    build_value_and_grad,
    cadence_scaled_savgol_window,
    cadence_scaled_sigclip_mask,
    optimise_halo,
    orbit_block_jackknife,
    safe_eigen_projection,
)
from .simplex import (
    flux_share_to_weights,
    ilr_to_simplex,
    simplex_to_ilr,
    uniform_weights_ilr,
    weights_to_flux_share,
    weights_to_ilr,
)

name = "halophot"
__version__ = "2.0.0"

write_fits = write_lightcurve_fits

__all__ = [
    "AVAILABLE_OBJECTIVES",
    "BkgCovariance",
    "DEFAULT_EXCLUDE_QUALITY",
    "HALOCVER",
    "HaloFitResult",
    "HaloMaskResult",
    "HaloObservation",
    "HaloWeights",
    "PixelQuality",
    "__version__",
    "bucket_pad_length",
    "build_halo_masks",
    "build_value_and_grad",
    "cadence_scaled_savgol_window",
    "cadence_scaled_sigclip_mask",
    "cast_tree",
    "charbonnier_penalty",
    "column_orthogonal_smear_weights",
    "compute_coherent_flux_err",
    "compute_gap_diff_mask",
    "extract_halo_lightcurve",
    "flux_share_to_weights",
    "halo_objective",
    "halo_objectives",
    "halo_tools",
    "halo_tpf",
    "halo_tpf_tess",
    "ilr_to_simplex",
    "in_float64",
    "infer_bkg_poly_order",
    "lightcurve",
    "mask_background_stars",
    "masking",
    "models",
    "name",
    "objectives",
    "optimise",
    "optimise_halo",
    "orbit_block_jackknife",
    "pack_bkg_chol",
    "read_bkg_covariance",
    "read_bkgsub_state",
    "read_pixel_quality",
    "read_smear_err",
    "remove_stars",
    "resolve_cache_dir",
    "run_in",
    "safe_eigen_projection",
    "simplex",
    "simplex_to_ilr",
    "tv_charbonnier_o1",
    "tv_charbonnier_o2",
    "tv_exp_scan",
    "uniform_weights_ilr",
    "weights_to_flux_share",
    "weights_to_ilr",
    "write_fits",
    "write_lightcurve_fits",
]
