"""halophot: Kepler/K2 and TESS halo photometry in JAX."""

from __future__ import annotations

import importlib
from typing import Any

name = "halophot"
__version__ = "2.0.1"

__all__ = [
    "HaloFitResult",
    "HaloLightCurve",
    "MaskConfig",
    "OBJECTIVES",
    "OptimConfig",
    "PixelQuality",
    "__version__",
    "extract_halo_lightcurve",
    "halo_tpf",
    "halo_tpf_tess",
    "optimise_halo",
    "plot_diagnostics",
    "read_weightmap",
    "read_weightmap_err",
    "register_objective",
    "write_fits",
]

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        "_contract",
        "_precision",
        "cli",
        "halo_objectives",
        "halo_tools",
        "lightcurve",
        "masking",
        "objectives",
        "optimise",
        "simplex",
    }
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Public __all__ symbols
    "HaloFitResult": (".optimise", "HaloFitResult"),
    "HaloLightCurve": (".lightcurve", "HaloLightCurve"),
    "MaskConfig": (".masking", "MaskConfig"),
    "OBJECTIVES": (".objectives", "OBJECTIVES"),
    "OptimConfig": (".optimise", "OptimConfig"),
    "PixelQuality": ("._contract", "PixelQuality"),
    "extract_halo_lightcurve": (".lightcurve", "extract_halo_lightcurve"),
    "halo_tpf": (".halo_tools", "halo_tpf"),
    "halo_tpf_tess": (".halo_tools", "halo_tpf_tess"),
    "optimise_halo": (".optimise", "optimise_halo"),
    "plot_diagnostics": (".lightcurve", "plot_diagnostics"),
    "read_weightmap": ("._contract", "read_weightmap"),
    "read_weightmap_err": ("._contract", "read_weightmap_err"),
    "register_objective": (".objectives", "register_objective"),
    "write_fits": ("._contract", "write_lightcurve_fits"),
    # Backwards-compatible contract / precision / helper attributes (not in __all__)
    "BkgCovariance": ("._contract", "BkgCovariance"),
    "HALOCVER": ("._contract", "HALOCVER"),
    "HaloMaskResult": (".masking", "HaloMaskResult"),
    "build_halo_masks": (".masking", "build_halo_masks"),
    "cadence_scaled_savgol_window": (".optimise", "cadence_scaled_savgol_window"),
    "cadence_scaled_sigclip_mask": (".optimise", "cadence_scaled_sigclip_mask"),
    "cast_tree": ("._precision", "cast_tree"),
    "charbonnier_abs": (".objectives", "charbonnier_abs"),
    "column_orthogonal_smear_weights": (".lightcurve", "column_orthogonal_smear_weights"),
    "compute_coherent_flux_err": (".lightcurve", "compute_coherent_flux_err"),
    "in_float64": ("._precision", "in_float64"),
    "infer_bkg_poly_order": ("._contract", "infer_bkg_poly_order"),
    "mask_background_stars": (".masking", "mask_background_stars"),
    "orbit_block_jackknife": (".optimise", "orbit_block_jackknife"),
    "pack_bkg_chol": ("._contract", "pack_bkg_chol"),
    "read_bkg_covariance": ("._contract", "read_bkg_covariance"),
    "read_bkgsub_state": ("._contract", "read_bkgsub_state"),
    "read_pixel_quality": ("._contract", "read_pixel_quality"),
    "read_segment_scales": ("._contract", "read_segment_scales"),
    "read_smear_err": ("._contract", "read_smear_err"),
    "read_split_times": ("._contract", "read_split_times"),
    "remove_stars": (".masking", "remove_stars"),
    "resolve_cache_dir": ("._contract", "resolve_cache_dir"),
    "run_in": ("._precision", "run_in"),
    "write_lightcurve_fits": ("._contract", "write_lightcurve_fits"),
}


def __getattr__(attr: str) -> Any:
    if attr in _LAZY_ATTRS:
        mod_name, obj_name = _LAZY_ATTRS[attr]
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, obj_name)
    if attr in _LAZY_SUBMODULES:
        return importlib.import_module(f".{attr}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
