"""Versioned FITS contract (``HALOCVER = 1``) and HPC cache path utilities.

This module defines the zero-import-coupling schema shared between ``mundey``,
``tessbkgd``, and ``halophot`` via standard ``lightkurve`` / FITS products:

* ``HALOCVER = 1`` provenance and ``BKGSUB`` background-subtraction state in
  ``hdu[0].header`` / ``hdu[1].header``.
* ``PIXEL_QUALITY`` spatial ``uint16`` ``ImageHDU`` bitmask (:class:`PixelQuality`).
* Cadence-slicing-safe ``hdu[1]`` (``PIXELS``) columns ``SMEAR_ERR`` and row-major
  lower-triangular ``BKG_CHOL``, paired with static ``BKG_BASIS`` ``ImageHDU``.
* Lossless ``WEIGHTMAP`` ``ImageHDU`` serialization for ``lightkurve.LightCurve``.
* ``/fred`` inode-safe cache resolution for Swinburne Ngarrgu Tindebeek / OzSTAR.
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Iterator
from enum import IntFlag
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from astropy.io import fits

HALOCVER: int = 1


class PixelQuality(IntFlag):
    """Spatial pixel-quality bitmask stored in ``PIXEL_QUALITY`` (``uint16``)."""

    NONE = 0
    STRAP_COL = 1 << 0
    SAT_BLEED = 1 << 1
    SMEAR_UNRESOLVED = 1 << 2
    SOURCE_MASK = 1 << 3
    BKG_USED = 1 << 4


class BkgCovariance(NamedTuple):
    """Unpacked per-cadence background Cholesky factor and static spatial basis.

    Attributes
    ----------
    chol : np.ndarray
        Lower-triangular Cholesky factors ``L_t`` of ``Sigma_{beta,t} = L_t @ L_t.T``
        with shape ``(N_cad, P, P)``.
    basis : np.ndarray
        Static 2D polynomial basis ``Phi_full`` evaluated on the TPF stamp with
        shape ``(P, ny, nx)``.
    """

    chol: np.ndarray
    basis: np.ndarray


@contextlib.contextmanager
def _open_hdul_and_mask(
    tpf_or_hdul: Any,
    *,
    apply_quality_mask: bool = True,
) -> Iterator[tuple[fits.HDUList, np.ndarray | None]]:
    """Yield ``(hdul, quality_mask)`` from a ``TargetPixelFile``, ``HDUList``, or path."""
    if isinstance(tpf_or_hdul, (str, os.PathLike)):
        with fits.open(tpf_or_hdul) as hdul:
            yield hdul, None
        return

    if isinstance(tpf_or_hdul, fits.HDUList):
        yield tpf_or_hdul, None
        return

    if hasattr(tpf_or_hdul, "hdu") and isinstance(tpf_or_hdul.hdu, fits.HDUList):
        hdul = tpf_or_hdul.hdu
        qmask: np.ndarray | None = None
        if apply_quality_mask and hasattr(tpf_or_hdul, "quality_mask"):
            raw_mask = tpf_or_hdul.quality_mask
            if raw_mask is not None:
                qmask = np.asarray(raw_mask, dtype=bool)
                if len(hdul) > 1 and hdul[1].data is not None:
                    n_rows = len(hdul[1].data)
                    if qmask.shape != (n_rows,):
                        raise ValueError(
                            f"TargetPixelFile quality_mask shape {qmask.shape} does not "
                            f"match HDU 1 row count ({n_rows},)."
                        )
        yield hdul, qmask
        return

    raise TypeError(
        "Expected a lightkurve.TargetPixelFile, astropy.io.fits.HDUList, or FITS path; "
        f"got {type(tpf_or_hdul).__name__}."
    )


def _validate_halocver(hdul: fits.HDUList) -> None:
    """Validate ``HALOCVER`` header cards when present."""
    v0 = hdul[0].header.get("HALOCVER") if len(hdul) > 0 else None
    v1 = hdul[1].header.get("HALOCVER") if len(hdul) > 1 else None
    if v0 is not None and v1 is not None and v0 != v1:
        raise ValueError(
            f"Conflicting HALOCVER header cards: HDU 0 has {v0!r}, HDU 1 has {v1!r}."
        )
    ver = v0 if v0 is not None else v1
    if ver is not None and (isinstance(ver, bool) or not isinstance(ver, int) or ver != HALOCVER):
        raise ValueError(f"Unsupported HALOCVER={ver!r}; expected {HALOCVER}.")


def _tpf_shape(hdul: fits.HDUList) -> tuple[int, int, int] | None:
    """Return ``(n_raw, ny, nx)`` from ``hdu[1].data['FLUX']`` if present."""
    if len(hdul) <= 1 or hdul[1].data is None or hdul[1].data.names is None:
        return None
    if "FLUX" not in hdul[1].data.names:
        return None
    shape = hdul[1].data["FLUX"].shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D FLUX column (n_cad, ny, nx), got shape {shape}.")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _parse_bool_card(value: object, card_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"Header card {card_name!r} must be a FITS logical boolean, got {value!r}."
    )


def read_bkgsub_state(tpf_or_hdul: Any, *, warn_legacy_mundey: bool = True) -> bool:
    """Determine whether ``FLUX`` in ``tpf_or_hdul`` is already background-subtracted.

    Checks ``BKGSUB`` in ``hdu[0].header`` and ``hdu[1].header`` first. When
    ``BKGSUB`` is absent (standard SPOC / Kepler / TessCut FITS files), returns
    ``False`` if ``FLUX_BKG`` is missing, all-zero, or all-NaN, and ``True``
    otherwise (emitting a defensive ``UserWarning`` if ``warn_legacy_mundey=True``
    detects unsubtracted dark-cadence ``FLUX`` paired with stale non-zero
    ``FLUX_BKG``).
    """
    with _open_hdul_and_mask(tpf_or_hdul, apply_quality_mask=False) as (hdul, _):
        _validate_halocver(hdul)

        has_b0 = len(hdul) > 0 and "BKGSUB" in hdul[0].header
        has_b1 = len(hdul) > 1 and "BKGSUB" in hdul[1].header
        b0 = _parse_bool_card(hdul[0].header["BKGSUB"], "BKGSUB") if has_b0 else None
        b1 = _parse_bool_card(hdul[1].header["BKGSUB"], "BKGSUB") if has_b1 else None

        if b0 is not None and b1 is not None and b0 != b1:
            raise ValueError(
                f"Conflicting BKGSUB header cards: HDU 0 has {b0!r}, HDU 1 has {b1!r}."
            )
        if b0 is not None:
            return b0
        if b1 is not None:
            return b1

        if len(hdul) <= 1 or hdul[1].data is None or hdul[1].data.names is None:
            return False
        if "FLUX_BKG" not in hdul[1].data.names:
            return False

        flux_bkg = np.asarray(hdul[1].data["FLUX_BKG"], dtype=np.float64)
        finite_bkg = flux_bkg[np.isfinite(flux_bkg)]
        if finite_bkg.size == 0 or np.all(finite_bkg == 0.0):
            return False

        if warn_legacy_mundey and "FLUX" in hdul[1].data.names:
            med_bkg = float(np.median(finite_bkg))
            if med_bkg > 0.0:
                flux = np.asarray(hdul[1].data["FLUX"], dtype=np.float64)
                valid_cad = np.any(np.isfinite(flux), axis=(1, 2))
                if np.any(valid_cad):
                    q10_per_cad = np.nanpercentile(flux[valid_cad], 10.0, axis=(1, 2))
                    dark_q10 = float(np.nanpercentile(q10_per_cad, 25.0))
                    if dark_q10 > 0.5 * med_bkg:
                        warnings.warn(
                            "BKGSUB header card is absent while FLUX_BKG is non-zero and "
                            "dark-cadence Q10(FLUX) exceeds 0.5 * median(FLUX_BKG); "
                            "TPF may contain unsubtracted FLUX with stale SPOC FLUX_BKG "
                            "(e.g. legacy mundey v0.1).",
                            UserWarning,
                            stacklevel=2,
                        )

        return True


def read_pixel_quality(tpf_or_hdul: Any) -> np.ndarray:
    """Read the ``(ny, nx)`` ``uint16`` ``PIXEL_QUALITY`` bitmask extension.

    If ``PIXEL_QUALITY`` is absent, returns an all-zero ``uint16`` array matching
    the spatial dimensions of ``FLUX``.
    """
    with _open_hdul_and_mask(tpf_or_hdul, apply_quality_mask=False) as (hdul, _):
        _validate_halocver(hdul)
        tpf_dims = _tpf_shape(hdul)

        if "PIXEL_QUALITY" not in hdul:
            if tpf_dims is None:
                raise ValueError(
                    "Cannot infer spatial dimensions when both PIXEL_QUALITY and FLUX are absent."
                )
            _, ny, nx = tpf_dims
            return np.zeros((ny, nx), dtype=np.uint16)

        raw = hdul["PIXEL_QUALITY"].data
        if raw is None or raw.ndim != 2:
            shape = None if raw is None else raw.shape
            raise ValueError(
                f"PIXEL_QUALITY ImageHDU must be a 2D array (ny, nx), got shape {shape}."
            )
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(
                f"PIXEL_QUALITY ImageHDU must have an integer dtype, got {raw.dtype}."
            )
        if np.any(raw < 0) or np.any(raw > np.iinfo(np.uint16).max):
            raise ValueError("PIXEL_QUALITY values must fit within uint16 [0, 65535].")

        if tpf_dims is not None:
            _, ny, nx = tpf_dims
            if raw.shape != (ny, nx):
                raise ValueError(
                    f"PIXEL_QUALITY shape {raw.shape} does not match FLUX spatial shape {(ny, nx)}."
                )

        return np.asarray(raw, dtype=np.uint16)


def read_smear_err(
    tpf_or_hdul: Any,
    *,
    apply_quality_mask: bool = True,
) -> np.ndarray | None:
    """Read per-cadence column smear uncertainty ``SMEAR_ERR`` ``(N_cad, nx)`` from HDU 1.

    Returns ``None`` if the ``SMEAR_ERR`` column is not present. When reading from
    a ``lightkurve.TargetPixelFile`` with ``apply_quality_mask=True`` (the default),
    slices rows by ``tpf.quality_mask`` so ``SMEAR_ERR`` stays in 1-to-1 lockstep
    with ``tpf.time``, ``tpf.flux``, and ``tpf[mask]``.
    """
    with _open_hdul_and_mask(tpf_or_hdul, apply_quality_mask=apply_quality_mask) as (
        hdul,
        qmask,
    ):
        _validate_halocver(hdul)
        if len(hdul) <= 1 or hdul[1].data is None or hdul[1].data.names is None:
            return None
        if "SMEAR_ERR" not in hdul[1].data.names:
            return None

        tpf_dims = _tpf_shape(hdul)
        raw = np.asarray(hdul[1].data["SMEAR_ERR"], dtype=np.float64)
        if raw.ndim == 1 and tpf_dims is not None and tpf_dims[2] == 1:
            raw = raw[:, np.newaxis]

        if raw.ndim != 2:
            raise ValueError(f"SMEAR_ERR column must be 2D (N_raw, nx), got shape {raw.shape}.")
        if tpf_dims is not None:
            n_raw, _, nx = tpf_dims
            if raw.shape != (n_raw, nx):
                raise ValueError(
                    f"SMEAR_ERR shape {raw.shape} does not match expected {(n_raw, nx)}."
                )

        if qmask is not None:
            raw = raw[qmask]
        return raw


def pack_bkg_chol(chol: np.ndarray) -> np.ndarray:
    """Pack lower-triangular ``(N_cad, P, P)`` Cholesky factors in row-major ``tril`` order.

    All-``NaN`` cadences (emitted by ``tessbkgd`` when fewer than ``P`` valid
    background pixels survive on a cadence) are permitted and pack to all-``NaN``
    rows; partial ``NaN``s, infinities, negative diagonal entries, and non-zero
    strict upper-triangular entries raise ``ValueError``.
    """
    arr = np.asarray(chol, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2] or arr.shape[1] < 1:
        raise ValueError(
            f"Expected 3D square Cholesky array (N_cad, P, P) with P >= 1, got shape {arr.shape}."
        )
    p_dim = arr.shape[1]
    active = ~np.all(np.isnan(arr), axis=(1, 2))
    if np.any(active):
        arr_active = arr[active]
        if not np.all(np.isfinite(arr_active)):
            raise ValueError(
                "Cholesky factors must contain only finite entries (or all-NaN cadences)."
            )
        if p_dim > 1:
            triu_r, triu_c = np.triu_indices(p_dim, k=1)
            if np.any(arr_active[:, triu_r, triu_c] != 0.0):
                raise ValueError(
                    "Cholesky array must be lower-triangular "
                    "(found non-zero strict upper-triangular entries)."
                )
        diag = np.diagonal(arr_active, axis1=1, axis2=2)
        if np.any(diag < 0.0):
            raise ValueError("Cholesky factors must have non-negative diagonal entries.")
    tril_r, tril_c = np.tril_indices(p_dim)
    return np.asarray(arr[:, tril_r, tril_c], dtype=np.float32)


def _poly_degree_from_basis_count(p_dim: int) -> int | None:
    """Solve ``P = (d + 1) * (d + 2) // 2`` for integer ``d >= 0``, or return ``None``."""
    if p_dim < 1:
        return None
    disc = 1 + 8 * p_dim
    root = int(math.isqrt(disc))
    if root * root == disc and root >= 3 and (root - 3) % 2 == 0:
        return (root - 3) // 2
    return None


def infer_bkg_poly_order(tpf_or_hdul: Any) -> int | None:
    """Infer the 2D background polynomial degree ``d`` when ``BKGSUB=True``.

    Checks ``BKG_ORD`` in ``BKG_BASIS``, ``hdu[0]``, and ``hdu[1]`` headers first,
    then falls back to solving ``P = (d + 1)(d + 2) / 2`` from ``BKG_BASIS`` shape
    ``(P, ny, nx)``. Returns ``None`` when ``BKGSUB`` is ``False`` or no polynomial
    background metadata are present.
    """
    if not read_bkgsub_state(tpf_or_hdul, warn_legacy_mundey=False):
        return None

    with _open_hdul_and_mask(tpf_or_hdul, apply_quality_mask=False) as (hdul, _):
        _validate_halocver(hdul)

        candidate_headers: list[fits.Header] = []
        if "BKG_BASIS" in hdul:
            candidate_headers.append(hdul["BKG_BASIS"].header)
        if len(hdul) > 0:
            candidate_headers.append(hdul[0].header)
        if len(hdul) > 1:
            candidate_headers.append(hdul[1].header)

        orders_found: list[int] = []
        for hdr in candidate_headers:
            if "BKG_ORD" in hdr:
                raw_ord = hdr["BKG_ORD"]
                if isinstance(raw_ord, bool) or not isinstance(raw_ord, (int, np.integer)) or int(raw_ord) < 0:
                    raise ValueError(
                        f"BKG_ORD header card must be a non-negative integer, got {raw_ord!r}."
                    )
                orders_found.append(int(raw_ord))

        if orders_found:
            if len(set(orders_found)) > 1:
                raise ValueError(f"Conflicting BKG_ORD header cards across HDUs: {orders_found}.")
            return orders_found[0]

        if "BKG_BASIS" in hdul:
            basis_raw = hdul["BKG_BASIS"].data
            if basis_raw is None or basis_raw.ndim != 3 or basis_raw.shape[0] < 1:
                shape = None if basis_raw is None else basis_raw.shape
                raise ValueError(
                    f"BKG_BASIS ImageHDU must be 3D (P, ny, nx) with P >= 1, got shape {shape}."
                )
            inferred = _poly_degree_from_basis_count(int(basis_raw.shape[0]))
            if inferred is None:
                raise ValueError(
                    f"Cannot infer 2D polynomial degree d from BKG_BASIS term count "
                    f"P={basis_raw.shape[0]} without an explicit BKG_ORD header card."
                )
            return inferred

        return None


def read_bkg_covariance(
    tpf_or_hdul: Any,
    *,
    apply_quality_mask: bool = True,
) -> BkgCovariance | None:
    """Unpack row-major ``np.tril_indices(P)`` ``BKG_CHOL`` + ``BKG_BASIS`` from a TPF.

    Returns ``None`` if neither ``BKG_CHOL`` (in ``hdu[1]``) nor ``BKG_BASIS``
    (``ImageHDU``) is present, and raises ``ValueError`` if only one of the two
    is present or if dimensions are inconsistent. All-``NaN`` cadences in
    ``BKG_CHOL`` unpack to all-``NaN`` ``(P, P)`` slices.
    """
    with _open_hdul_and_mask(tpf_or_hdul, apply_quality_mask=apply_quality_mask) as (
        hdul,
        qmask,
    ):
        _validate_halocver(hdul)

        has_chol = (
            len(hdul) > 1
            and hdul[1].data is not None
            and hdul[1].data.names is not None
            and "BKG_CHOL" in hdul[1].data.names
        )
        has_basis = "BKG_BASIS" in hdul
        if not has_chol and not has_basis:
            return None
        if has_chol != has_basis:
            raise ValueError(
                "Incomplete HALOCVER=1 background covariance: BKG_CHOL column in HDU 1 "
                "and BKG_BASIS ImageHDU must both be present."
            )

        basis_raw = hdul["BKG_BASIS"].data
        if basis_raw is None or basis_raw.ndim != 3:
            shape = None if basis_raw is None else basis_raw.shape
            raise ValueError(
                f"BKG_BASIS ImageHDU must be 3D (P, ny, nx), got shape {shape}."
            )
        basis = np.asarray(basis_raw, dtype=np.float64)
        p_dim, ny, nx = basis.shape
        if p_dim < 1:
            raise ValueError(f"BKG_BASIS basis count P must be >= 1, got {p_dim}.")

        tpf_dims = _tpf_shape(hdul)
        if tpf_dims is not None:
            n_raw, flux_ny, flux_nx = tpf_dims
            if (ny, nx) != (flux_ny, flux_nx):
                raise ValueError(
                    f"BKG_BASIS spatial shape {(ny, nx)} does not match FLUX shape {(flux_ny, flux_nx)}."
                )
        else:
            n_raw = len(hdul[1].data)

        packed = np.asarray(hdul[1].data["BKG_CHOL"], dtype=np.float64)
        if packed.ndim == 1 and p_dim == 1:
            packed = packed[:, np.newaxis]

        n_tril = p_dim * (p_dim + 1) // 2
        if packed.ndim != 2 or packed.shape != (n_raw, n_tril):
            raise ValueError(
                f"BKG_CHOL shape {packed.shape} does not match expected {(n_raw, n_tril)} "
                f"for P={p_dim}."
            )

        if qmask is not None:
            packed = packed[qmask]

        all_nan_cad = np.all(np.isnan(packed), axis=1)
        active = ~all_nan_cad
        if np.any(active) and not np.all(np.isfinite(packed[active])):
            raise ValueError("BKG_CHOL contains non-finite entries in partially valid cadences.")

        chol = np.zeros((packed.shape[0], p_dim, p_dim), dtype=np.float64)
        tril_r, tril_c = np.tril_indices(p_dim)
        chol[:, tril_r, tril_c] = packed
        if np.any(all_nan_cad):
            chol[all_nan_cad, :, :] = np.nan
        if np.any(active) and np.any(np.diagonal(chol[active], axis1=1, axis2=2) < 0.0):
            raise ValueError("BKG_CHOL diagonal entries must be non-negative.")

        return BkgCovariance(chol=chol, basis=basis)


def write_lightcurve_fits(
    lc: Any,
    path: str | os.PathLike[str],
    weightmap: np.ndarray | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Serialize a ``lightkurve.LightCurve`` to FITS with a lossless ``WEIGHTMAP`` ``ImageHDU``.

    ``LightCurve.to_fits()`` drops 2D/3D ``ndarray`` entries from ``lc.meta``.
    This writer preserves ``lc.meta['WEIGHTMAP']`` (or an explicit ``weightmap``
    argument) in an appended ``ImageHDU(name='WEIGHTMAP')`` while leaving the
    in-memory ``lc.meta`` dictionary untouched.
    """
    out_path = Path(path)

    meta = getattr(lc, "meta", None)
    resolved_wmap = weightmap
    resolved_werr = None
    resolved_splits = None
    if isinstance(meta, dict):
        if resolved_wmap is None:
            if "WEIGHTMAP" in meta:
                resolved_wmap = meta["WEIGHTMAP"]
            elif "weightmap" in meta:
                resolved_wmap = meta["weightmap"]
        if "WEIGHTMAP_ERR" in meta:
            resolved_werr = meta["WEIGHTMAP_ERR"]
        elif "weightmap_err" in meta:
            resolved_werr = meta["weightmap_err"]
        if "SPLIT_TIMES" in meta:
            resolved_splits = meta["SPLIT_TIMES"]
        elif "split_times" in meta:
            resolved_splits = meta["split_times"]

    lc_export = lc.copy() if hasattr(lc, "copy") else lc
    export_meta = getattr(lc_export, "meta", None)
    if isinstance(export_meta, dict):
        if export_meta is meta:
            export_meta = dict(meta)
            lc_export.meta = export_meta
        for key in (
            "WEIGHTMAP",
            "weightmap",
            "WEIGHTMAP_ERR",
            "weightmap_err",
            "SPLIT_TIMES",
            "split_times",
        ):
            export_meta.pop(key, None)
        for attr, meta_key in (
            ("targetid", "TARGETID"),
            ("channel", "CHANNEL"),
            ("mission", "MISSION"),
            ("ra", "RA"),
            ("dec", "DEC"),
        ):
            if (
                meta_key not in export_meta
                and attr not in export_meta
                and not hasattr(lc_export, attr)
            ):
                export_meta[meta_key] = "" if attr == "targetid" else None
    if not hasattr(lc_export, "quality") and hasattr(lc_export, "__setitem__"):
        lc_export["quality"] = np.zeros(len(lc_export), dtype=np.int32)

    hdul = lc_export.to_fits()

    hdul[0].header["HALOCVER"] = (HALOCVER, "Halophot pipeline contract version")

    splits_arr: np.ndarray | None = None
    if resolved_splits is not None:
        cand_splits = np.asarray(resolved_splits, dtype=np.float64).ravel()
        if cand_splits.size > 0:
            if not np.all(np.isfinite(cand_splits)):
                raise ValueError("SPLIT_TIMES must contain only finite timestamps.")
            splits_arr = cand_splits
            hdul[0].header["NSPLITS"] = (
                int(splits_arr.size),
                "Number of segment split timestamps",
            )

    if resolved_wmap is not None:
        if isinstance(resolved_wmap, np.ma.MaskedArray):
            resolved_wmap = resolved_wmap.filled(np.nan)
        w_arr = np.asarray(resolved_wmap, dtype=np.float64)
        if w_arr.ndim < 2:
            raise ValueError(
                f"WEIGHTMAP must have at least 2 dimensions (ny, nx), got shape {w_arr.shape}."
            )
        if "WEIGHTMAP" in hdul:
            del hdul["WEIGHTMAP"]
        wmap_hdu = fits.ImageHDU(data=w_arr, name="WEIGHTMAP")
        if splits_arr is not None:
            wmap_hdu.header["NSPLITS"] = (
                int(splits_arr.size),
                "Number of segment split timestamps",
            )
            if splits_arr.size <= 999:
                for idx_s, st_val in enumerate(splits_arr, start=1):
                    wmap_hdu.header[f"SPLIT{idx_s:03d}"] = (
                        float(st_val),
                        f"Segment split timestamp {idx_s}",
                    )
        hdul.append(wmap_hdu)

    if resolved_werr is not None:
        werr_arr = np.asarray(resolved_werr, dtype=np.float64)
        if werr_arr.ndim >= 2:
            if "WEIGHTMAP_ERR" in hdul:
                del hdul["WEIGHTMAP_ERR"]
            hdul.append(fits.ImageHDU(data=werr_arr, name="WEIGHTMAP_ERR"))

    if splits_arr is not None:
        if "SPLIT_TIMES" in hdul:
            del hdul["SPLIT_TIMES"]
        split_col = fits.Column(name="SPLIT_TIME", format="D", array=splits_arr)
        hdul.append(fits.BinTableHDU.from_columns([split_col], name="SPLIT_TIMES"))

    hdul.writeto(out_path, overwrite=overwrite)
    return out_path


def read_split_times(lc_or_hdul_or_path: Any) -> np.ndarray | None:
    """Read segment ``SPLIT_TIMES`` from a light curve, FITS ``HDUList``, or path.

    Returns ``None`` when the observation was extracted without segment splits.
    """
    meta = getattr(lc_or_hdul_or_path, "meta", None)
    if isinstance(meta, dict):
        for key in ("SPLIT_TIMES", "split_times"):
            if key in meta and meta[key] is not None:
                arr = np.asarray(meta[key], dtype=np.float64).ravel()
                return arr if arr.size > 0 else None

    with _open_hdul_and_mask(lc_or_hdul_or_path, apply_quality_mask=False) as (hdul, _):
        if "SPLIT_TIMES" in hdul and hdul["SPLIT_TIMES"].data is not None:
            names = [n.upper() for n in (hdul["SPLIT_TIMES"].data.names or ())]
            if "SPLIT_TIME" in names:
                arr = np.asarray(hdul["SPLIT_TIMES"].data["SPLIT_TIME"], dtype=np.float64).ravel()
                return arr if arr.size > 0 else None
        if "WEIGHTMAP" in hdul and "NSPLITS" in hdul["WEIGHTMAP"].header:
            n_splits = int(hdul["WEIGHTMAP"].header["NSPLITS"])
            if n_splits > 0:
                cards = [
                    float(hdul["WEIGHTMAP"].header[f"SPLIT{idx_s:03d}"])
                    for idx_s in range(1, n_splits + 1)
                    if f"SPLIT{idx_s:03d}" in hdul["WEIGHTMAP"].header
                ]
                if len(cards) == n_splits:
                    return np.asarray(cards, dtype=np.float64)
        return None


def _is_under_fred(path: Path) -> bool:
    """Return True if ``path`` (lexically or after symlink resolution) lies under ``/fred``."""
    fred = Path("/fred")
    lexical = Path(os.path.abspath(path))
    if lexical == fred or fred in lexical.parents:
        return True
    resolved = path.resolve(strict=False)
    return resolved == fred or fred in resolved.parents


def resolve_cache_dir(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    strict_hpc: bool = False,
) -> Path:
    """Resolve the package cache directory, guarding ``/fred`` inode quotas in HPC mode.

    Precedence when ``cache_dir`` is ``None``:
    1. ``$SCRATCH/.cache/tess_bright`` if ``SCRATCH`` is set and non-empty.
    2. ``$XDG_CACHE_HOME/tess_bright`` if ``XDG_CACHE_HOME`` is set and non-empty.
    3. ``~/.cache/tess_bright``.

    When ``strict_hpc=True``, raises ``ValueError`` if the resolved path points
    to or inside ``/fred/`` (protecting Swinburne NT / OzSTAR project inode caps).
    """
    if cache_dir is not None:
        resolved = Path(cache_dir).expanduser()
    elif os.environ.get("SCRATCH"):
        resolved = Path(os.environ["SCRATCH"]).expanduser() / ".cache" / "tess_bright"
    elif os.environ.get("XDG_CACHE_HOME"):
        resolved = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "tess_bright"
    else:
        resolved = Path.home() / ".cache" / "tess_bright"

    if strict_hpc and _is_under_fred(resolved):
        raise ValueError(
            f"Cache directory {resolved} lies under /fred/, which is forbidden in "
            "strict_hpc=True mode to protect project inode quotas; use $SCRATCH or $JOBFS."
        )

    return resolved
