"""Canonical HALOCVER=1 FITS contract & HPC helpers (pure numpy + astropy)."""

from __future__ import annotations

import contextlib
import errno
import math
import os
import shutil
import socket
import uuid
import warnings
from collections.abc import Iterator, Mapping, MutableMapping
from enum import IntFlag
from pathlib import Path
from typing import Any, NamedTuple

import astropy.units as u
import numpy as np
from astropy.io import fits

HALOCVER: int = 1
PIXEL_QUALITY_EXTNAME: str = "PIXEL_QUALITY"
SMEAR_ERR_COLNAME: str = "SMEAR_ERR"
_LC_DROP = {"WEIGHTMAP", "WEIGHTMAP_ERR", "SPLIT_TIMES", "SEGMENT_SCALES", "FILENAME"}
_LC_META = (
    ("BKGSUB", bool), ("BKG_ORD", int), ("HALO_OBJ", str),
    ("HALO_LAG", int), ("LEGACYTV", bool),
)
_LC_COLS = (
    ("pos_corr1", "POS_CORR1", "E", np.float32),
    ("pos_corr2", "POS_CORR2", "E", np.float32),
    ("inlier_mask", "INLIER_MASK", "L", bool),
)

class PixelQuality(IntFlag):
    """Spatial pixel-quality bitmask stored in ``PIXEL_QUALITY`` (``uint16``)."""
    NONE = 0
    STRAP_COL = 1 << 0
    SAT_BLEED = 1 << 1
    SMEAR_UNRESOLVED = 1 << 2
    SOURCE_MASK = 1 << 3
    BKG_USED = 1 << 4
class BkgCovariance(NamedTuple):
    """Unpacked per-cadence background Cholesky factor ``(N, P, P)`` and basis."""
    chol: np.ndarray
    basis: np.ndarray

def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)
def _is_int(v: Any) -> bool:
    return not isinstance(v, bool | np.bool_) and isinstance(v, int | np.integer)
def _tri_root(n: int) -> int | None:
    r = math.isqrt(1 + 8 * n) if n >= 1 else 0
    return (r - 1) // 2 if n >= 1 and r * r == 1 + 8 * n and r % 2 == 1 else None
def _crows(n_pix: int) -> int:
    return min(4096, max(1, (8 * 1024 * 1024) // max(int(n_pix) * 8, 1)))
def _cmap(hdul: fits.HDUList) -> dict[str, str]:
    d = hdul[1].data if len(hdul) > 1 else None
    return {n.upper(): n for n in d.names} if (d is not None and d.names) else {}
def _pq_data(hdul: fits.HDUList) -> np.ndarray | None:
    return hdul[PIXEL_QUALITY_EXTNAME].data if PIXEL_QUALITY_EXTNAME in hdul else None
def _set_hdu(hdul: fits.HDUList, hdu: Any) -> None:
    if hdu.name in hdul:
        del hdul[hdu.name]
    hdul.append(hdu)
def _unwrap_hdulist(target: Any) -> fits.HDUList:
    if isinstance(target, fits.HDUList):
        return target
    if isinstance(getattr(target, "hdu", None), fits.HDUList):
        return target.hdu
    raise TypeError(f"Expected HDUList or TPF, got {type(target).__name__}.")
@contextlib.contextmanager
def _open_hdul(target: Any, *, use_qmask: bool = True, check_ver: bool = True):
    owned = isinstance(target, str | os.PathLike)
    hdul = fits.open(Path(target).expanduser()) if owned else _unwrap_hdulist(target)
    try:
        if check_ver:
            _validate_halocver(hdul)
        q = None if (owned or not use_qmask) else getattr(target, "quality_mask", None)
        qm = np.asarray(q, bool) if q is not None else None
        if qm is not None and len(hdul) > 1 and hdul[1].data is not None:
            _req(qm.shape == (len(hdul[1].data),), f"quality_mask {qm.shape} mismatch.")
        yield hdul, qm
    finally:
        if owned:
            hdul.close()
def _cutout_shape(hdul: fits.HDUList) -> tuple[int, int, int] | None:
    for col in ("FLUX", "RAW_CNTS", "FLUX_ERR"):
        if col in (cm := _cmap(hdul)):
            sh = hdul[1].data[cm[col]].shape
            if len(sh) == 3:
                return int(sh[0]), int(sh[1]), int(sh[2])
            _req(col != "FLUX", f"Expected 3D FLUX, got {sh}.")
    return None
def _sync_tpf_meta(target: Any, hdul: fits.HDUList) -> None:
    if hasattr(target, "meta") and hasattr(target, "hdu"):
        m = target.meta
        if isinstance(m, MutableMapping):
            m.update(hdul[0].header)
        elif type(m).__name__ == "HduToMetaMapping":
            target.meta = type(m)(hdul[0])
        elif isinstance(getattr(m, "_dict", None), dict):
            m._dict.update(hdul[0].header)
def _flux_column_unit_str(hdu1: fits.BinTableHDU) -> str | None:
    for i, c in enumerate(hdu1.columns or (), 1):
        if c.name.upper() == "FLUX":
            return str(v) if (v := c.unit or hdu1.header.get(f"TUNIT{i}")) else None
    return None
def _put_col(hdul: fits.HDUList, name: str, arr: np.ndarray | None, unit=None) -> None:
    key = name.upper()
    if arr is not None:
        cur = {c.name.upper(): c for c in (hdul[1].columns or ())}.get(key)
        if cur is not None and (str(cur.unit) if cur.unit else None) == unit:
            sh, (nr, w) = hdul[1].data[cur.name].shape, arr.shape
            if sh == (nr, w) or (w == 1 and sh == (nr,)):
                hdul[1].data[cur.name] = arr[:, 0] if w == 1 and len(sh) == 1 else arr
                return
    kept = [c for c in hdul[1].columns if c.name.upper() != key]
    if arr is not None:
        w = arr.shape[1]
        kept.append(fits.Column(name, f"{w}E", unit=unit, dim=f"({w})", array=arr))
    h1 = hdul[1]
    hdul[1] = fits.BinTableHDU.from_columns(
        fits.ColDefs(kept), h1.header.copy(), name=h1.name or "PIXELS"
    )
def _validate_halocver(hdul: fits.HDUList) -> int | None:
    vals = [h.header["HALOCVER"] for h in hdul[:2] if "HALOCVER" in h.header]
    ok = all(_is_int(c) and int(c) == HALOCVER for c in vals)
    _req(ok, f"Unsupported HALOCVER={vals!r}.")
    return int(vals[0]) if vals else None
def _bkgsub_card(hdul: fits.HDUList) -> bool | None:
    vals = [h.header["BKGSUB"] for h in hdul[:2] if "BKGSUB" in h.header]
    _req(all(isinstance(v, bool | np.bool_) for v in vals), f"Bad BKGSUB={vals!r}")
    _req(len({bool(v) for v in vals}) <= 1, f"Conflicting BKGSUB cards: {vals}.")
    return bool(vals[0]) if vals else None
def _set_bkgsub_header(target: Any, bkgsub: bool) -> None:
    hdul, flag = _unwrap_hdulist(target), bool(bkgsub)
    for h in hdul[:2]:
        h.header["BKGSUB"] = (flag, "Whether sky background is subtracted")
        if "BACKAPP" in h.header:
            h.header["BACKAPP"] = (flag, "Background subtraction applied")
    _sync_tpf_meta(target, hdul)
def _check_pq(raw: np.ndarray, *, allow_bool: bool = False) -> np.ndarray:
    is_int = np.issubdtype(raw.dtype, np.integer)
    ok_dt = raw.ndim == 2 and (is_int or (allow_bool and raw.dtype == np.bool_))
    _req(ok_dt, f"PIXEL_QUALITY must be 2D int, got {raw.shape}/{raw.dtype}.")
    _req(not is_int or not (np.any(raw < 0) or np.any(raw > 65535)), "uint16 overflow.")
    return np.array(raw, dtype=np.uint16, copy=True)

def write_mundey_provenance(
    target: Any, *, bkgsub: bool = False, halocver: int = HALOCVER, smear_mode=None
) -> fits.HDUList:
    """Write ``HALOCVER``, ``BKGSUB``, and optional ``SMEARMOD`` to HDU 0/1."""
    hdul = _unwrap_hdulist(target)
    _req(len(hdul) >= 2 and _is_int(halocver) and int(halocver) == HALOCVER, "Bad arg.")
    for h in hdul[:2]:
        h.header["HALOCVER"] = (HALOCVER, "Halo FITS contract version")
    _set_bkgsub_header(target, bkgsub)
    if smear_mode is not None:
        hdul[0].header["SMEARMOD"] = (str(smear_mode), "Mundey smear mode")
        _sync_tpf_meta(target, hdul)
    return hdul
def read_mundey_provenance(target: Any) -> dict[str, int | bool | None]:
    """Read and validate ``HALOCVER`` and ``BKGSUB`` from HDU 0 and HDU 1."""
    with _open_hdul(target, use_qmask=False, check_ver=False) as (hdul, _):
        return {"halocver": _validate_halocver(hdul), "bkgsub": _bkgsub_card(hdul)}
def write_pixel_quality(target: Any, mask: np.ndarray) -> fits.HDUList:
    """Write or update ``PIXEL_QUALITY`` ``ImageHDU`` (``uint16``)."""
    hdul, arr = _unwrap_hdulist(target), _check_pq(np.asarray(mask), allow_bool=True)
    dims = _cutout_shape(hdul)
    _req(dims is None or arr.shape == dims[1:], f"PIXEL_QUALITY shape {arr.shape}.")
    img = fits.ImageHDU(data=arr, name=PIXEL_QUALITY_EXTNAME)
    img.header["HALOCVER"] = (HALOCVER, "Halo FITS contract version")
    _set_hdu(hdul, img)
    return hdul
def read_pixel_quality(target: Any) -> np.ndarray:
    """Read ``PIXEL_QUALITY`` as ``uint16 (ny, nx)``, returning zeros if absent."""
    with _open_hdul(target, use_qmask=False) as (hdul, _):
        dims = _cutout_shape(hdul)
        if PIXEL_QUALITY_EXTNAME not in hdul:
            _req(dims is not None, "Cannot infer shape without PIXEL_QUALITY or FLUX.")
            return np.zeros(dims[1:], dtype=np.uint16)
        raw = _pq_data(hdul)
        _req(raw is not None, "PIXEL_QUALITY ImageHDU has no data.")
        arr = _check_pq(np.asarray(raw))
        _req(dims is None or arr.shape == dims[1:], f"PIXEL_QUALITY shape {arr.shape}.")
        return arr
def append_smear_err_column(target: Any, smear_err: Any) -> fits.HDUList:
    """Append or update ``SMEAR_ERR`` (``float32 (N_raw, nx)``) in HDU 1."""
    hdul = _unwrap_hdulist(target)
    ok_h1 = len(hdul) >= 2 and isinstance(hdul[1], fits.BinTableHDU)
    _req(ok_h1 and hdul[1].data is not None, "HDU[1] must be populated BinTableHDU.")
    raw = smear_err
    if isinstance(raw, u.Quantity):
        dimless = raw.unit == u.dimensionless_unscaled
        raw = raw.value if dimless else raw.to_value(u.electron / u.s)
    arr, nr, dims = np.asarray(raw, np.float32), len(hdul[1].data), _cutout_shape(hdul)
    nx = dims[2] if dims is not None else (arr.shape[1] if arr.ndim == 2 else -1)
    ok_s = arr.ndim == 2 and arr.shape == (nr, nx) and not np.any(arr < 0.0)
    _req(ok_s, f"Invalid non-negative SMEAR_ERR shape {arr.shape}.")
    _put_col(hdul, SMEAR_ERR_COLNAME, arr, "e-/s")
    return hdul
def read_smear_err(target: Any, *, quality_mask: bool = True,
                   apply_quality_mask=None, quality_masked=None) -> np.ndarray | None:
    """Read ``SMEAR_ERR`` as ``float64 (N_cad, nx)``, or ``None`` if absent."""
    use_q = quality_mask if apply_quality_mask is None else apply_quality_mask
    use_q = use_q if quality_masked is None else quality_masked
    with _open_hdul(target, use_qmask=bool(use_q)) as (hdul, qm):
        if SMEAR_ERR_COLNAME not in (cm := _cmap(hdul)):
            return None
        dims = _cutout_shape(hdul)
        raw = np.array(hdul[1].data[cm[SMEAR_ERR_COLNAME]], float)
        raw = raw[:, None] if raw.ndim == 1 and (dims is None or dims[2] == 1) else raw
        ok_sh = raw.ndim == 2 and (dims is None or raw.shape == (dims[0], dims[2]))
        _req(ok_sh and not np.any(raw < 0.0), "Invalid non-negative SMEAR_ERR.")
        return raw[qm].copy() if qm is not None else raw
read_smear_err_column = read_smear_err

def pack_bkg_chol(bkg_chol: np.ndarray) -> np.ndarray:
    """Pack lower-triangular ``(N, P, P)`` or ``(N, P*(P+1)//2)`` to ``float32``."""
    arr = np.asarray(bkg_chol, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 1 and arr.shape[1] == arr.shape[2]:
        p = arr.shape[1]
        if np.any(a3 := ~np.all(np.isnan(arr), (1, 2))) and p > 1:
            _req(not np.any(arr[a3][:, *np.triu_indices(p, 1)] != 0), "Not lower-tri.")
        r, c = np.tril_indices(p)
        pk = arr[:, r, c]
    elif arr.ndim == 2 and (p := _tri_root(arr.shape[1])) is not None:
        r, c, pk = *np.tril_indices(p), arr
    else:
        raise ValueError(f"Invalid bkg_chol shape {arr.shape}.")
    if np.any(act_m := ~np.all(np.isnan(pk), axis=1)):
        act = pk[act_m]
        _req(bool(np.all(np.isfinite(act))), "Active bkg_chol must be finite.")
        _req(not np.any(act[:, np.flatnonzero(r == c)] < 0.0), "Negative chol diag.")
        fmax = float(np.finfo(np.float32).max)
        _req(not np.any(np.abs(act) > fmax), "bkg_chol overflowed float32 range.")
    return np.ascontiguousarray(pk, dtype=np.float32)
def unpack_bkg_chol(bkg_chol_packed: np.ndarray, n_basis: int | None = None):
    """Unpack ``(N, P*(P+1)//2)`` lower-triangular factors into ``(N, P, P)``."""
    arr = np.asarray(bkg_chol_packed)
    arr = arr[:, None] if arr.ndim == 1 else arr
    p = _tri_root(arr.shape[1]) if (arr.ndim == 2 and n_basis is None) else n_basis
    ok = arr.ndim == 2 and p is not None and int(p) >= 1
    _req(ok and int(p) * (int(p) + 1) // 2 == arr.shape[1], f"Bad shape {arr.shape}.")
    p_i = int(p)
    out = np.zeros((arr.shape[0], p_i, p_i), dtype=arr.dtype)
    out[:, *np.tril_indices(p_i)] = arr
    if np.issubdtype(out.dtype, np.floating):
        out[np.all(np.isnan(arr), axis=1)] = np.nan
    return out
def read_bkg_covariance(target: Any, *, quality_mask: bool = True,
                        apply_quality_mask=None) -> BkgCovariance | None:
    """Read and unpack ``(BKG_CHOL, BKG_BASIS)`` as ``float64``."""
    use_q = quality_mask if apply_quality_mask is None else apply_quality_mask
    with _open_hdul(target, use_qmask=bool(use_q)) as (hdul, qm):
        cm = _cmap(hdul)
        has_c, has_b = "BKG_CHOL" in cm, "BKG_BASIS" in hdul
        if not (has_c or has_b):
            return None
        br = hdul["BKG_BASIS"].data if has_b else None
        _req(has_c == has_b and br is not None and br.ndim == 3 and br.shape[0] > 0,
             "Incomplete or invalid (BKG_CHOL, BKG_BASIS) covariance.")
        basis, dims = np.array(br, float), _cutout_shape(hdul)
        p, ny, nx = basis.shape
        _req(dims is None or (ny, nx) == dims[1:], f"BKG_BASIS {(ny, nx)} mismatch.")
        nr = dims[0] if dims is not None else len(hdul[1].data)
        pk = np.array(hdul[1].data[cm["BKG_CHOL"]], float)
        pk = pk[:, None] if pk.ndim == 1 and p == 1 else pk
        _req(pk.ndim == 2 and pk.shape == (nr, p * (p + 1) // 2), "Bad BKG_CHOL shape.")
        pk = pk[qm] if qm is not None else pk
        chol, act = unpack_bkg_chol(pk, n_basis=p), ~np.all(np.isnan(pk), axis=1)
        bad_d = np.any(np.diagonal(chol[act], axis1=1, axis2=2) < 0.0)
        _req(not np.any(act) or (bool(np.all(np.isfinite(pk[act]))) and not bad_d),
             "Active BKG_CHOL must be finite with diagonal >= 0.")
        return BkgCovariance(chol=chol, basis=basis)
def _strip_bkg_covariance(target: Any, *, has_nonzero_bkg: bool = True) -> bool:
    hdul = _unwrap_hdulist(target)
    cm, pq_d = _cmap(hdul), _pq_data(hdul)
    has_c, has_b = "BKG_CHOL" in cm, "BKG_BASIS" in hdul
    has_u = pq_d is not None and bool(np.any(_check_pq(np.asarray(pq_d)) & 16))
    if not (has_c or has_b or has_u):
        return False
    br = hdul["BKG_BASIS"].data if (has_c and has_b and "FLUX_ERR" in cm) else None
    if br is not None and br.ndim == 3 and br.shape[0] > 0:
        fe, pd = hdul[1].data[cm["FLUX_ERR"]], hdul[1].data[cm["BKG_CHOL"]]
        p, ny, nx = br.shape
        nr, nt = fe.shape[0], p * (p + 1) // 2
        ok_p = pd.shape == (nr, nt) or (p == 1 and pd.shape == (nr,))
        if fe.shape[1:] == (ny, nx) and ok_p:
            phi, step = np.ascontiguousarray(br.reshape(p, -1), float), _crows(ny * nx)
            for s in range(0, nr, step):
                cc = unpack_bkg_chol(np.asarray(pd[s : s + step], float), p)
                bad = ~np.all(np.isfinite(cc), (1, 2))
                cok = np.where(bad[:, None, None], 0.0, cc)
                bsq = sum(np.square(cok[:, :, j] @ phi) for j in range(p))
                b3 = np.where(bad[:, None], np.nan, bsq).reshape(-1, ny, nx)
                fc = np.asarray(fe[s : s + step], float)
                ok = np.isfinite(fc) & (fc > 0) & np.isfinite(b3) & (b3 >= 0)
                fe[s : s + step] = np.where(
                    ok, np.sqrt(np.maximum(fc * fc - b3, 0.0)), fc
                ).astype(np.float32)
    if "FLUX_BKG_ERR" in cm:
        fbe = hdul[1].data[cm["FLUX_BKG_ERR"]]
        fill = np.float32(np.nan if has_nonzero_bkg else 0.0)
        step = _crows(int(np.prod(fbe.shape[1:])) if fbe.ndim > 1 else 1)
        for s in range(0, len(fbe), step):
            fbe[s : s + step] = fill
    if has_u:
        hdul[PIXEL_QUALITY_EXTNAME].data = np.array(pq_d, "u2") & ~np.uint16(16)
    for h in hdul[:2]:
        for k in ("BKG_ORD", "BKG_ORDX", "BKG_ORDY"):
            h.header.pop(k, None)
    if has_c:
        _put_col(hdul, "BKG_CHOL", None)
    while "BKG_BASIS" in hdul:
        del hdul["BKG_BASIS"]
    _sync_tpf_meta(target, hdul)
    return True
def write_tessbkgd_provenance(
    hdu: Any, bkg_basis: np.ndarray, bkg_chol: np.ndarray, bkg_used_mask: np.ndarray,
    *, bkg_order: int | tuple[int, int] | None = None, source_mask=None,
) -> fits.HDUList:
    """Write ``HALOCVER = 1`` provenance, ``BKG_BASIS``, and ``BKG_CHOL``."""
    hdul = _unwrap_hdulist(hdu)
    _validate_halocver(hdul)
    _req(len(hdul) >= 2 and hdul[1].data is not None, "Populated HDU[1] required.")
    used = np.asarray(bkg_used_mask, bool)
    src = None if source_mask is None else np.asarray(source_mask, bool)
    b32 = np.ascontiguousarray(bkg_basis, np.float32)
    _req(used.ndim == 2 and (src is None or src.shape == used.shape), "Bad mask shape.")
    ny, nx = used.shape
    _req(b32.ndim == 3 and b32.shape[1:] == (ny, nx) and b32.shape[0] > 0, "Bad basis.")
    p, oy, ox = b32.shape[0], None, None
    if bkg_order is None:
        res_ord = (r - 1) if (r := _tri_root(p)) is not None else None
    else:
        pair = (bkg_order, bkg_order) if _is_int(bkg_order) else bkg_order
        ok_p = isinstance(pair, tuple | list) and len(pair) == 2
        _req(ok_p and all(_is_int(v) and int(v) >= 0 for v in pair), "Bad bkg_order.")
        oy, ox = int(pair[0]), int(pair[1])
        ry, rx = range(oy + 1), range(ox + 1)
        _req(sum(q + c <= max(oy, ox) for q in ry for c in rx) == p, "Bad P.")
        res_ord = ox
    chol_p = pack_bkg_chol(bkg_chol)
    _req(chol_p.shape == (len(hdul[1].data), p * (p + 1) // 2), "Bad bkg_chol shape.")
    for h in hdul[:2]:
        h.header["HALOCVER"] = (HALOCVER, "Halo FITS contract version")
    _set_bkgsub_header(hdu, True)
    pq_d = _pq_data(hdul)
    pq = _check_pq(np.asarray(pq_d)) if pq_d is not None else np.zeros((ny, nx), "u2")
    _req(pq.shape == (ny, nx), f"PIXEL_QUALITY {pq.shape} != {(ny, nx)}.")
    pq &= ~np.uint16(PixelQuality.BKG_USED)
    if src is not None:
        pq[src] |= np.uint16(PixelQuality.SOURCE_MASK)
    pq[used] |= np.uint16(PixelQuality.BKG_USED)
    write_pixel_quality(hdul, pq)
    img = fits.ImageHDU(data=b32, name="BKG_BASIS")
    for k, val in (("BKG_ORD", res_ord), ("BKG_ORDY", oy), ("BKG_ORDX", ox)):
        if val is not None:
            img.header[k] = (val, "2D background polynomial degree")
    _set_hdu(hdul, img)
    _put_col(hdul, "BKG_CHOL", chol_p, _flux_column_unit_str(hdul[1]))
    _sync_tpf_meta(hdu, hdul)
    return hdul

def read_bkgsub_state(target: Any, *, warn_legacy_mundey: bool = True) -> bool:
    """Return whether ``FLUX`` is background-subtracted (chunked, <100 MB RAM)."""
    with _open_hdul(target, use_qmask=True) as (hdul, qm):
        if (explicit := _bkgsub_card(hdul)) is not None:
            return explicit
        cm = _cmap(hdul)
        if "FLUX_BKG" not in cm or len(hdul[1].data) == 0:
            return False
        rbkg, nr = hdul[1].data[cm["FLUX_BKG"]], len(hdul[1].data)
        use_q = qm is not None and len(qm) == nr and bool(np.any(qm))
        ridx = np.flatnonzero(qm) if use_q else np.arange(nr)
        n_pix = int(np.prod(rbkg.shape[1:])) if rbkg.ndim > 1 else 1
        n_rows, step = ridx.size, _crows(n_pix)
        is_k2 = any(
            str(h.header.get(k, "")).strip().upper() in ("KEPLER", "K2")
            for h in hdul[:2] for k in ("MISSION", "TELESCOP")
        )
        if not (warn_legacy_mundey and "FLUX" in cm and not is_k2):
            for s in range(0, n_rows, step):
                b = np.asarray(rbkg[ridx[s : s + step]], float)
                if np.any(np.isfinite(b) & (b != 0.0)):
                    return True
            return False
        rflux, gpx = hdul[1].data[cm["FLUX"]], np.ones(n_pix, bool)
        if (pq_d := _pq_data(hdul)) is not None and pq_d.shape == rflux.shape[1:]:
            bad = ((_check_pq(np.asarray(pq_d)) & np.uint16(15)) != 0).ravel()
            gpx = ~bad if not np.all(bad) else gpx
        has_nz, cbkg, cq10 = False, np.full(n_rows, np.nan), np.full(n_rows, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            for s in range(0, n_rows, step):
                sl = ridx[s : s + step]
                b = np.asarray(rbkg[sl], float).reshape(-1, n_pix)
                f = np.asarray(rflux[sl], float).reshape(-1, n_pix)
                has_nz = has_nz or bool(np.any(np.isfinite(b) & (b != 0.0)))
                cbkg[s : s + step] = np.nanmedian(b, 1)
                cq10[s : s + step] = np.nanpercentile(f[:, gpx], 10.0, 1)
        if not has_nz:
            return False
        ok = np.isfinite(cbkg) & np.isfinite(cq10)
        if not np.any(ok):
            return True
        mbkg, dq10 = float(np.nanmedian(cbkg)), float(np.percentile(cq10[ok], 25.0))
        if not (np.isfinite(mbkg) and mbkg > 0 and dq10 > 0.5 * mbkg):
            return True
        bc, cov = cbkg[ok] - np.mean(cbkg[ok]), True
        if bc.size >= 3 and float(np.std(bc)) > 1e-3 * mbkg:
            cov = float(np.dot(cq10[ok] - np.mean(cq10[ok]), bc) / np.dot(bc, bc)) > 0.5
        if cov:
            warnings.warn(
                f"BKGSUB absent with non-zero FLUX_BKG ({mbkg:.3g}) and co-varying "
                f"dark FLUX ({dq10:.3g}); possible legacy mundey < 2.0.",
                UserWarning, stacklevel=2,
            )
        return True
def infer_bkg_poly_order(target: Any) -> int | None:
    """Infer horizontal column polynomial degree ``d`` when ``BKGSUB=True``."""
    if not read_bkgsub_state(target, warn_legacy_mundey=False):
        return None
    with _open_hdul(target, use_qmask=False) as (hdul, _):
        hdrs = ([hdul["BKG_BASIS"].header] if "BKG_BASIS" in hdul else []) + [
            h.header for h in hdul[:2]
        ]
        for key in ("BKG_ORDX", "BKG_ORD"):
            if vals := [h[key] for h in hdrs if key in h]:
                _req(all(_is_int(v) and int(v) >= 0 for v in vals), f"Bad {key}.")
                _req(len(ints := {int(v) for v in vals}) == 1, f"Conflicting {key}.")
                return next(iter(ints))
        if "BKG_BASIS" in hdul:
            b = hdul["BKG_BASIS"].data
            r = _tri_root(int(b.shape[0])) if (b is not None and b.ndim == 3) else None
            _req(r is not None, "Cannot infer polynomial degree from BKG_BASIS.")
            return r - 1
        return None

def _cards_io(hdr: fits.Header, ckey: str, pfx: str, arr=None) -> np.ndarray | None:
    if arr is not None:
        hdr[ckey] = int(arr.size)
        for i, val in enumerate(arr[:999], 1):
            hdr[f"{pfx}{i:03d}"] = float(val)
        return None
    if ckey not in hdr or (n := int(hdr[ckey])) <= 0:
        return None
    ks = [f"{pfx}{i:03d}" for i in range(1, n + 1) if f"{pfx}{i:03d}" in hdr]
    return np.array([float(hdr[k]) for k in ks], float) if len(ks) == n else None
def _is_seg_card(k: Any) -> bool:
    uk = str(k).upper()
    return uk in ("NSPLITS", "NSEGSC") or (
        uk[:5] in ("SEGSC", "SPLIT") and uk[5:].isdigit())
def _read_lc_ext(obj: Any, ext: str, col: str | None = None, cards=()):
    meta = getattr(obj, "meta", None)
    if isinstance(meta, Mapping):
        for k in (ext, ext.lower()):
            if (v := meta.get(k)) is not None:
                v = v.filled(np.nan) if isinstance(v, np.ma.MaskedArray) else v
                a = np.array(v, float)
                return (a.ravel() if a.size else None) if col else a
    cands: list[Any] = [obj, getattr(obj, "_hdu", None), getattr(obj, "filename", None)]
    if not isinstance(getattr(type(obj), "hdu", None), property):
        cands.insert(2, getattr(obj, "hdu", None))
    if isinstance(meta, Mapping):
        cands.extend((meta.get("FILENAME"), meta.get("filename")))
    tgt = next((c if isinstance(c, fits.HDUList) else Path(c).expanduser()
                for c in cands if isinstance(c, fits.HDUList) or (
                    isinstance(c, str | os.PathLike) and str(c).strip()
                    and (c is obj or Path(c).expanduser().is_file()))), None)
    if tgt is None:
        if isinstance(meta, Mapping) or hasattr(obj, "flux"):
            return None
        raise TypeError(f"Expected LightCurve/HDUList/path, got {type(obj).__name__}.")
    with _open_hdul(tgt, use_qmask=False, check_ver=False) as (hdul, _):
        if ext in hdul and (d := hdul[ext].data) is not None:
            if col is None:
                return np.array(d, float)
            if cn := {n.upper(): n for n in (d.names or ())}.get(col):
                return v if (v := np.array(d[cn], float).ravel()).size else None
        for href, ckey, pfx in cards:
            if (isinstance(href, int) and len(hdul) > href) or href in hdul:
                if (found := _cards_io(hdul[href].header, ckey, pfx)) is not None:
                    return found
        return None
def read_weightmap(obj: Any) -> np.ndarray | None:
    """Read 2D ``(ny, nx)`` or 3D ``(n_seg, ny, nx)`` ``WEIGHTMAP``."""
    return _read_lc_ext(obj, "WEIGHTMAP")
def read_weightmap_err(obj: Any) -> np.ndarray | None:
    """Read 2D or 3D ``WEIGHTMAP_ERR`` from a light curve or FITS file."""
    return _read_lc_ext(obj, "WEIGHTMAP_ERR")
def read_split_times(obj: Any) -> np.ndarray | None:
    """Read 1D ``SPLIT_TIMES`` from a light curve, ``HDUList``, or FITS path."""
    c = (("WEIGHTMAP", "NSPLITS", "SPLIT"),)
    return _read_lc_ext(obj, "SPLIT_TIMES", "SPLIT_TIME", c)
def read_segment_scales(obj: Any) -> np.ndarray | None:
    """Read 1D ``SEGMENT_SCALES`` from a light curve, ``HDUList``, or FITS path."""
    c = tuple((h, "NSEGSC", "SEGSC") for h in ("SPLIT_TIMES", "WEIGHTMAP", 0))
    return _read_lc_ext(obj, "SEGMENT_SCALES", "SEGMENT_SCALE", c)
def write_lightcurve_fits(
    lc: Any, path: str | os.PathLike[str], weightmap=None, *, overwrite: bool = True
) -> Path:
    """Serialize a light curve to FITS with lossless ``HALOCVER = 1`` extensions."""
    out_path, meta = Path(path), getattr(lc, "meta", None)
    wmap = read_weightmap(lc) if weightmap is None else weightmap
    werr = read_weightmap_err(lc)
    splits, scales = read_split_times(lc), read_segment_scales(lc)
    m_s = isinstance(meta, Mapping) and ("SPLIT_TIMES" in meta or "split_times" in meta)
    from_disk = splits is not None and not m_s
    lc_e = lc.copy() if hasattr(lc, "copy") else lc
    if isinstance(em := getattr(lc_e, "meta", None), Mapping):
        lc_e.meta = em = {k: v for k, v in em.items()
                          if str(k).upper() not in _LC_DROP and not _is_seg_card(k)}
        cn, ups = type(lc_e).__name__, {str(k).upper() for k in em}
        if "TARGETID" not in ups and getattr(lc_e, "targetid", None) is None:
            em["TARGETID"] = ""
        if "MISSION" not in ups and getattr(lc_e, "mission", None) is None:
            if m := ("TESS" if "Tess" in cn else ("Kepler" if "Kepler" in cn else "")):
                em["MISSION"] = m
        if "Kepler" in cn:
            for a in ("channel", "ra", "dec"):
                if a.upper() not in ups and not hasattr(lc_e, a):
                    em[a.upper()] = None
    if not hasattr(lc_e, "quality") and hasattr(lc_e, "__setitem__"):
        lc_e["quality"] = np.zeros(len(lc_e), dtype=np.int32)
    base_fn = next((c.to_fits for c in type(lc_e).__mro__ if "to_fits" in c.__dict__
                    and c.__module__.startswith("lightkurve.")), None)
    hdul = base_fn(lc_e) if base_fn is not None else lc_e.to_fits()
    if len(hdul) > 1 and isinstance(h1 := hdul[1], fits.BinTableHDU):
        ex = {c.name.upper() for c in h1.columns}
        lcm = {str(c).lower(): str(c) for c in (getattr(lc, "colnames", ()) or ())}
        add = []
        for lk_k, fk, fmt, dt in _LC_COLS:
            if fk in ex or lk_k not in lcm:
                continue
            v = getattr(col := lc[lcm[lk_k]], "value", col)
            v = v.filled(np.nan) if isinstance(v, np.ma.MaskedArray) else v
            if (a := np.asarray(v, dt).ravel()).size == len(h1.data):
                cu = getattr(col, "unit", None)
                pix = cu in (None, u.dimensionless_unscaled, u.pixel)
                us = None if fmt == "L" else ("pixel" if pix else cu.to_string("fits"))
                add.append(fits.Column(fk, fmt, unit=us, array=a))
        if add:
            cols = h1.columns + fits.ColDefs(add)
            hdul[1] = fits.BinTableHDU.from_columns(cols, h1.header, name=h1.name)
    h0 = hdul[0].header
    for k in [k for k in h0 if _is_seg_card(k)]:
        del h0[k]
    if str(h0.get("OBJECT", "")).strip() == "None":
        h0["OBJECT"] = ""
    if str(h0.get("INSTRUME", "")).strip() == "None":
        del h0["INSTRUME"]
    tel = str(h0.get("TELESCOP") or getattr(lc, "mission", "") or "").strip().upper()
    if tel == "TESS":
        if not str(h0.get("TELESCOP") or "").strip():
            h0["TELESCOP"] = ("TESS", "Telescope")
        if str(h0.get("INSTRUME") or "").strip() in ("", "None"):
            h0["INSTRUME"] = ("TESS Photometer", "Instrument name")
    h0["HALOCVER"] = (HALOCVER, "Halo FITS contract version")
    if isinstance(meta, Mapping):
        h0.update({k: fn(meta[k]) for k, fn in _LC_META if meta.get(k) is not None})
    fu = getattr(getattr(lc, "flux", None), "unit", None)
    if fu not in (None, u.dimensionless_unscaled) and len(hdul) > 1:
        rates = ("electron / s", "electron/s", "e-/s")
        is_r = fu == u.electron / u.s or str(fu).strip() in rates
        us = "e-/s" if is_r else fu.to_string("fits")
        for i, col in enumerate(getattr(hdul[1], "columns", ()) or (), 1):
            if col.name.upper() in ("FLUX", "FLUX_ERR"):
                col.unit, hdul[1].header[f"TUNIT{i}"] = us, us
    splits_a, seg_r, scales_a = None, None, None
    if splits is not None:
        cand = np.asarray(splits, float).ravel()
        _req(not cand.size or bool(np.all(np.isfinite(cand))), "Bad SPLIT_TIMES.")
        orig = np.sort(cand)
        if orig.size and getattr(lc, "time", None) is not None:
            t_raw = np.asarray(getattr(lc.time, "value", lc.time), float)
            if (tf := t_raw[np.isfinite(t_raw)]).size:
                s0 = int(np.searchsorted(orig, float(np.min(tf)), side="left"))
                s1 = int(np.searchsorted(orig, float(np.max(tf)), side="left"))
                seg_r, cand = (int(orig.size) + 1, s0, s1), orig[s0:s1]
            elif from_disk:
                cand = np.empty(0, float)
        if cand.size:
            splits_a, h0["NSPLITS"] = cand, int(cand.size)
    if scales is not None:
        sc = np.asarray(scales, float).ravel()
        _req(not sc.size or bool(np.all(np.isfinite(sc))), "Bad SEGMENT_SCALES.")
        sc = sc[seg_r[1] : seg_r[2] + 1] if (seg_r and sc.size == seg_r[0]) else sc
        if sc.size:
            _cards_io(h0, "NSEGSC", "SEGSC", scales_a := sc)
    for ext, raw in (("WEIGHTMAP", wmap), ("WEIGHTMAP_ERR", werr)):
        if raw is not None:
            a = np.asarray(raw, float)
            if seg_r is not None and a.ndim == 3 and a.shape[0] == seg_r[0]:
                sl = a[seg_r[1] : seg_r[2] + 1]
                a = sl[0] if sl.shape[0] == 1 else sl
            _req(a.ndim >= 2, f"{ext} must be >= 2D, got {a.shape}.")
            ihdu = fits.ImageHDU(data=a, name=ext)
            if ext == "WEIGHTMAP" and splits_a is not None:
                _cards_io(ihdu.header, "NSPLITS", "SPLIT", splits_a)
            _set_hdu(hdul, ihdu)
    for ext, col, arr, cards in (
        ("SPLIT_TIMES", "SPLIT_TIME", splits_a, scales_a),
        ("SEGMENT_SCALES", "SEGMENT_SCALE", scales_a, None),
    ):
        if arr is None:
            if ext in hdul:
                del hdul[ext]
        else:
            cols = [fits.Column(col, "D", array=arr)]
            thdu = fits.BinTableHDU.from_columns(cols, name=ext)
            if cards is not None:
                _cards_io(thdu.header, "NSEGSC", "SEGSC", cards)
            _set_hdu(hdul, thdu)
    hdul.writeto(out_path, overwrite=overwrite)
    return out_path
def _is_fred_path(path: Path) -> bool:
    return ("/", "fred") in (
        Path(os.path.abspath(os.fspath(path))).parts[:2], path.resolve(False).parts[:2])
def resolve_cache_dir(
    cache_dir=None, *, env_var: str = "TESS_BRIGHT_CACHE_DIR",
    strict_hpc: bool = False, create: bool = False,
) -> Path:
    """Resolve cache directory with ``/fred/`` guard (warns by default)."""
    raw = Path(os.fspath(cache_dir)).expanduser() if cache_dir is not None else None
    for k, s in ((env_var, ""), ("TESS_BRIGHT_CACHE_DIR", ""),
                 ("SCRATCH", ".cache/tess_bright"), ("XDG_CACHE_HOME", "tess_bright")):
        if raw is None and k and (v := os.environ.get(k, "").strip()):
            raw = Path(v).expanduser() / s
    raw = raw or (Path.home() / ".cache" / "tess_bright")
    resolved = raw.resolve(strict=False)
    if _is_fred_path(raw) or resolved.parts[:2] == ("/", "fred"):
        msg = f"Cache dir {resolved} lies under /fred/ (inode quota); use $SCRATCH."
        _req(not strict_hpc, msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved
def make_atomic_tmp_path(dst: str | os.PathLike[str]) -> Path:
    """Build collision-free sibling temp path ``{dst}.{host}.{pid}.{uuid}.tmp``."""
    p, host = Path(dst), socket.gethostname().split(".")[0] or "host"
    return p.parent / f"{p.name}.{host}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
def _fsync(path: Path, *, is_dir: bool = False) -> None:
    if is_dir and not hasattr(os, "O_DIRECTORY"):
        return
    with contextlib.suppress(OSError):
        fd = os.open(os.fspath(path), os.O_RDONLY | (os.O_DIRECTORY if is_dir else 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
def atomic_publish(src_tmp: str | os.PathLike[str], dst: str | os.PathLike[str]):
    """Atomically publish ``src_tmp`` to ``dst`` via ``fsync`` + ``os.replace``."""
    src, dst_p = Path(src_tmp), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Temporary source file does not exist: {src}")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if os.path.abspath(src.parent) == os.path.abspath(dst_p.parent):
        _fsync(src)
        try:
            os.replace(src, dst_p)
            _fsync(dst_p.parent, is_dir=True)
            return dst_p
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    stage = make_atomic_tmp_path(dst_p)
    try:
        with src.open("rb") as sf, stage.open("wb") as df:
            shutil.copyfileobj(sf, df, 1024 * 1024)
            df.flush()
            os.fsync(df.fileno())
        os.replace(stage, dst_p)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    src.unlink(missing_ok=True)
    _fsync(dst_p.parent, is_dir=True)
    return dst_p
@contextlib.contextmanager
def atomic_temp_path(dst: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a sibling temp path and atomically publish it to ``dst`` on exit."""
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = make_atomic_tmp_path(dst_p)
    try:
        yield tmp
        if tmp.exists():
            atomic_publish(tmp, dst_p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
