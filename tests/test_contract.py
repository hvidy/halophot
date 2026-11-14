"""Tests for the ``HALOCVER = 1`` FITS contract and HPC ``/fred`` cache guard."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
import types
import warnings
from pathlib import Path

import astropy.units as u
import jax.numpy as jnp
import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from lightkurve.targetpixelfile import TargetPixelFile

import halophot
from halophot._contract import (
    HALOCVER,
    PixelQuality,
    infer_bkg_poly_order,
    pack_bkg_chol,
    read_bkg_covariance,
    read_bkgsub_state,
    read_pixel_quality,
    read_smear_err,
    read_weightmap,
    resolve_cache_dir,
    write_lightcurve_fits,
)


def _make_contract_tpf_hdulist(
    n_raw: int = 20,
    ny: int = 4,
    nx: int = 5,
    p_dim: int = 3,
    *,
    bkgsub: bool | None = True,
    include_contract_exts: bool = True,
) -> tuple[fits.HDUList, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct an in-memory SPOC-style TESS TPF HDUList populated with HALOCVER=1 data."""
    rng = np.random.default_rng(42)

    time = 1500.0 + np.arange(n_raw, dtype=np.float64) * (2.0 / 1440.0)
    cadenceno = np.arange(1000, 1000 + n_raw, dtype=np.int32)
    quality = np.zeros(n_raw, dtype=np.int32)
    quality[2] = 128  # Flagged bad cadence so quality_mask is non-trivial
    quality[11] = 128

    flux = rng.uniform(50.0, 200.0, size=(n_raw, ny, nx)).astype(np.float32)
    flux_err = np.sqrt(flux).astype(np.float32)
    flux_bkg = np.full((n_raw, ny, nx), 30.0, dtype=np.float32)
    flux_bkg_err = np.full((n_raw, ny, nx), 1.5, dtype=np.float32)
    raw_cnts = np.full((n_raw, ny, nx), 1000, dtype=np.int32)
    pos_corr1 = np.zeros(n_raw, dtype=np.float32)
    pos_corr2 = np.zeros(n_raw, dtype=np.float32)

    smear_err = rng.uniform(0.5, 2.5, size=(n_raw, nx)).astype(np.float32)

    # Lower-triangular Cholesky factors L_t with positive diagonal entries
    chol_true = np.zeros((n_raw, p_dim, p_dim), dtype=np.float32)
    for r in range(p_dim):
        for c in range(r + 1):
            vals = rng.uniform(0.2, 1.2, size=n_raw).astype(np.float32)
            if r == c:
                vals += 1.0
            chol_true[:, r, c] = vals
    chol_packed = pack_bkg_chol(chol_true)

    basis_true = rng.normal(0.0, 1.0, size=(p_dim, ny, nx)).astype(np.float32)
    pq_true = np.zeros((ny, nx), dtype=np.uint16)
    pq_true[:, 0] = int(PixelQuality.STRAP_COL)
    pq_true[1:3, 2] = int(PixelQuality.SAT_BLEED | PixelQuality.SMEAR_UNRESOLVED)
    pq_true[0, -1] = int(PixelQuality.BKG_USED)
    pq_true[-1, -1] = int(PixelQuality.SOURCE_MASK)

    prim = fits.PrimaryHDU()
    prim.header["MISSION"] = "TESS"
    prim.header["TELESCOP"] = "TESS"
    prim.header["INSTRUME"] = "TESS Photometer"
    prim.header["ORIGIN"] = "NASA/Ames"
    prim.header["CREATOR"] = "1234 TargetPixelExporterPipelineModule"
    prim.header["OBJECT"] = "TEST STAR"
    prim.header["TICID"] = 123456789
    prim.header["SECTOR"] = 6
    prim.header["CAMERA"] = 1
    prim.header["CCD"] = 2
    prim.header["RA_OBJ"] = 101.287
    prim.header["DEC_OBJ"] = -16.716
    if include_contract_exts:
        prim.header["HALOCVER"] = HALOCVER
    if bkgsub is not None:
        prim.header["BKGSUB"] = bkgsub

    dim_str = f"({nx},{ny})"
    cols = [
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=time),
        fits.Column(name="TIMECORR", format="E", unit="d", array=np.zeros(n_raw, dtype=np.float32)),
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(name="RAW_CNTS", format=f"{ny * nx}J", dim=dim_str, unit="count", array=raw_cnts),
        fits.Column(name="FLUX", format=f"{ny * nx}E", dim=dim_str, unit="e-/s", array=flux),
        fits.Column(name="FLUX_ERR", format=f"{ny * nx}E", dim=dim_str, unit="e-/s", array=flux_err),
        fits.Column(name="FLUX_BKG", format=f"{ny * nx}E", dim=dim_str, unit="e-/s", array=flux_bkg),
        fits.Column(
            name="FLUX_BKG_ERR", format=f"{ny * nx}E", dim=dim_str, unit="e-/s", array=flux_bkg_err
        ),
        fits.Column(name="QUALITY", format="J", array=quality),
        fits.Column(name="POS_CORR1", format="E", unit="pixel", array=pos_corr1),
        fits.Column(name="POS_CORR2", format="E", unit="pixel", array=pos_corr2),
    ]
    if include_contract_exts:
        n_tril = p_dim * (p_dim + 1) // 2
        cols.extend(
            [
                fits.Column(name="SMEAR_ERR", format=f"{nx}E", unit="e-/s", array=smear_err),
                fits.Column(name="BKG_CHOL", format=f"{n_tril}E", array=chol_packed),
            ]
        )

    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    pixels_hdu.header["INSTRUME"] = "TESS Photometer"
    if include_contract_exts:
        pixels_hdu.header["HALOCVER"] = HALOCVER
    if bkgsub is not None:
        pixels_hdu.header["BKGSUB"] = bkgsub

    aperture = np.full((ny, nx), 3, dtype=np.int32)
    ap_hdu = fits.ImageHDU(data=aperture, name="APERTURE")

    hdus: list[fits.PrimaryHDU | fits.BinTableHDU | fits.ImageHDU] = [prim, pixels_hdu, ap_hdu]
    if include_contract_exts:
        hdus.append(fits.ImageHDU(data=pq_true, name="PIXEL_QUALITY"))
        hdus.append(fits.ImageHDU(data=basis_true, name="BKG_BASIS"))

    return fits.HDUList(hdus), smear_err, chol_true, basis_true, pq_true


def test_pixel_quality_flags() -> None:
    assert HALOCVER == 1
    assert int(PixelQuality.STRAP_COL) == 1
    assert int(PixelQuality.SAT_BLEED) == 2
    assert int(PixelQuality.SMEAR_UNRESOLVED) == 4
    assert int(PixelQuality.SOURCE_MASK) == 8
    assert int(PixelQuality.BKG_USED) == 16
    combined = PixelQuality.STRAP_COL | PixelQuality.BKG_USED
    assert PixelQuality.STRAP_COL in combined
    assert PixelQuality.SAT_BLEED not in combined


def test_bkgsub_state_and_legacy_mundey_warning() -> None:
    hdul_true, *_ = _make_contract_tpf_hdulist(bkgsub=True)
    assert read_bkgsub_state(hdul_true) is True

    hdul_false, *_ = _make_contract_tpf_hdulist(bkgsub=False)
    assert read_bkgsub_state(hdul_false) is False

    # Conflicting BKGSUB cards must raise ValueError
    hdul_conflict, *_ = _make_contract_tpf_hdulist(bkgsub=True)
    hdul_conflict[1].header["BKGSUB"] = False
    with pytest.raises(ValueError, match="Conflicting BKGSUB"):
        read_bkgsub_state(hdul_conflict)

    # Unsupported HALOCVER must raise ValueError
    hdul_badver, *_ = _make_contract_tpf_hdulist(bkgsub=True)
    hdul_badver[0].header["HALOCVER"] = 99
    hdul_badver[1].header["HALOCVER"] = 99
    with pytest.raises(ValueError, match="Unsupported HALOCVER"):
        read_bkgsub_state(hdul_badver)

    # TessCut-like file: no BKGSUB header and all-zero FLUX_BKG -> False
    hdul_tesscut, *_ = _make_contract_tpf_hdulist(bkgsub=None, include_contract_exts=False)
    hdul_tesscut[1].data["FLUX_BKG"][:] = 0.0
    assert read_bkgsub_state(hdul_tesscut) is False

    # Legacy mundey v0.1 detector: no BKGSUB header, FLUX_BKG = 100, dark FLUX > 50 -> warns
    hdul_legacy, *_ = _make_contract_tpf_hdulist(bkgsub=None, include_contract_exts=False)
    hdul_legacy[1].data["FLUX_BKG"][:] = 100.0
    hdul_legacy[1].data["FLUX"][:] = 140.0
    with pytest.warns(UserWarning, match="legacy mundey"):
        assert read_bkgsub_state(hdul_legacy, warn_legacy_mundey=True) is True

    # Standard SPOC subtracted file: dark FLUX near 0 with FLUX_BKG = 100 -> True without warning
    hdul_spoc, *_ = _make_contract_tpf_hdulist(bkgsub=None, include_contract_exts=False)
    hdul_spoc[1].data["FLUX_BKG"][:] = 100.0
    hdul_spoc[1].data["FLUX"][:] = 5.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert read_bkgsub_state(hdul_spoc, warn_legacy_mundey=True) is True
    assert len(caught) == 0


def test_tpf_cadence_slicing_lockstep_and_covariance_unpack(tmp_path: Path) -> None:
    hdul, smear_true, chol_true, basis_true, pq_true = _make_contract_tpf_hdulist(
        n_raw=20, ny=4, nx=5, p_dim=3
    )
    tpf_path = tmp_path / "contract_tpf.fits"
    hdul.writeto(tpf_path, overwrite=True)

    tpf = lk.read(str(tpf_path), quality_bitmask="default")
    qmask = np.asarray(tpf.quality_mask, dtype=bool)
    assert int(np.sum(qmask)) == 18
    assert len(tpf.time) == 18

    # Spatial PIXEL_QUALITY round-trip
    pq = read_pixel_quality(tpf)
    assert pq.dtype == np.uint16
    np.testing.assert_array_equal(pq, pq_true)

    # Quality-masked SMEAR_ERR and BKG_CHOL match active cadences
    smear = read_smear_err(tpf)
    assert smear is not None
    assert smear.shape == (18, 5)
    np.testing.assert_allclose(smear, smear_true[qmask], rtol=1e-6)

    cov = read_bkg_covariance(tpf)
    assert cov is not None
    assert cov.chol.shape == (18, 3, 3)
    assert cov.basis.shape == (3, 4, 5)
    np.testing.assert_allclose(cov.chol, np.tril(cov.chol))
    np.testing.assert_allclose(cov.chol, chol_true[qmask], rtol=1e-6)
    np.testing.assert_allclose(cov.basis, basis_true, rtol=1e-6)

    # Reconstructed Sigma_beta = L @ L^T matches ground truth
    sigma_rec = cov.chol @ np.swapaxes(cov.chol, -1, -2)
    sigma_ref = chol_true[qmask].astype(np.float64) @ np.swapaxes(
        chol_true[qmask].astype(np.float64), -1, -2
    )
    np.testing.assert_allclose(sigma_rec, sigma_ref, rtol=1e-6)

    # Boolean mask slicing AND integer range slicing on TargetPixelFile stay in 1-to-1 lockstep
    bmask = np.zeros(len(tpf.time), dtype=bool)
    bmask[[0, 3, 4, 9, 15, 17]] = True
    sub_bool = tpf[bmask]
    assert len(sub_bool.time) == 6
    np.testing.assert_allclose(read_smear_err(sub_bool), smear[bmask], rtol=1e-6)  # type: ignore[arg-type]
    sub_cov_bool = read_bkg_covariance(sub_bool)
    assert sub_cov_bool is not None
    np.testing.assert_allclose(sub_cov_bool.chol, cov.chol[bmask], rtol=1e-6)
    np.testing.assert_allclose(sub_cov_bool.basis, basis_true, rtol=1e-6)
    np.testing.assert_array_equal(read_pixel_quality(sub_bool), pq_true)

    # FITS serialization of a sliced TPF preserves lockstep
    sliced_path = tmp_path / "sliced_tpf.fits"
    sub_bool.to_fits(str(sliced_path), overwrite=True)
    reloaded = lk.read(str(sliced_path), quality_bitmask="default")
    np.testing.assert_allclose(read_smear_err(reloaded), smear[bmask], rtol=1e-6)  # type: ignore[arg-type]
    reloaded_cov = read_bkg_covariance(reloaded)
    assert reloaded_cov is not None
    np.testing.assert_allclose(reloaded_cov.chol, cov.chol[bmask], rtol=1e-6)

    sub_slice = tpf[4:11]
    np.testing.assert_allclose(read_smear_err(sub_slice), smear[4:11], rtol=1e-6)  # type: ignore[arg-type]
    sub_cov_slice = read_bkg_covariance(sub_slice)
    assert sub_cov_slice is not None
    np.testing.assert_allclose(sub_cov_slice.chol, cov.chol[4:11], rtol=1e-6)


def test_contract_missing_and_malformed_extensions() -> None:
    hdul_plain, *_ = _make_contract_tpf_hdulist(include_contract_exts=False)
    pq_zeros = read_pixel_quality(hdul_plain)
    assert pq_zeros.shape == (4, 5)
    assert pq_zeros.dtype == np.uint16
    assert np.all(pq_zeros == 0)
    assert read_smear_err(hdul_plain) is None
    assert read_bkg_covariance(hdul_plain) is None

    # Orphan BKG_BASIS without BKG_CHOL must fail validation
    hdul_orphan, *_ = _make_contract_tpf_hdulist(include_contract_exts=False)
    hdul_orphan.append(fits.ImageHDU(data=np.ones((3, 4, 5), dtype=np.float32), name="BKG_BASIS"))
    with pytest.raises(ValueError, match="Incomplete HALOCVER=1"):
        read_bkg_covariance(hdul_orphan)

    # Wrong PIXEL_QUALITY shape must fail validation
    hdul_bad_pq, *_ = _make_contract_tpf_hdulist()
    hdul_bad_pq["PIXEL_QUALITY"].data = np.zeros((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="PIXEL_QUALITY shape"):
        read_pixel_quality(hdul_bad_pq)

    # pack_bkg_chol must reject upper-triangular arrays, negative diagonals, P=0, and NaNs
    upper = np.zeros((4, 3, 3), dtype=np.float64)
    upper[:, 0, 0] = 1.0
    upper[:, 1, 1] = 1.0
    upper[:, 2, 2] = 1.0
    upper[:, 0, 1] = 0.5  # Strict upper-triangular entry (e.g. scipy cholesky lower=False)
    with pytest.raises(ValueError, match="lower-triangular"):
        pack_bkg_chol(upper)

    neg_diag = np.zeros((4, 2, 2), dtype=np.float64)
    neg_diag[:, 0, 0] = 1.0
    neg_diag[:, 1, 1] = 1.0
    neg_diag[2, 1, 1] = -0.25
    with pytest.raises(ValueError, match="non-negative diagonal"):
        pack_bkg_chol(neg_diag)

    with pytest.raises(ValueError, match="P >= 1"):
        pack_bkg_chol(np.zeros((4, 0, 0), dtype=np.float64))

    # Partial NaNs in a cadence must raise ValueError, while all-NaN cadences round-trip seamlessly
    partial_nan = np.zeros((4, 2, 2), dtype=np.float64)
    partial_nan[:, 0, 0] = 1.0
    partial_nan[:, 1, 1] = 1.0
    partial_nan[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="all-NaN"):
        pack_bkg_chol(partial_nan)

    hdul_nan, _, chol_true, _, _ = _make_contract_tpf_hdulist(n_raw=14, ny=4, nx=5, p_dim=3)
    chol_with_nan = chol_true.copy()
    chol_with_nan[4, :, :] = np.nan
    packed_nan = pack_bkg_chol(chol_with_nan)
    assert np.all(np.isnan(packed_nan[4]))
    hdul_nan[1].data["BKG_CHOL"][:] = packed_nan
    unpacked_nan = read_bkg_covariance(hdul_nan)
    assert unpacked_nan is not None
    assert np.all(np.isnan(unpacked_nan.chol[4]))
    np.testing.assert_allclose(unpacked_nan.chol[:4], chol_with_nan[:4], rtol=1e-6)


def test_infer_bkg_poly_order_headers_and_basis_count() -> None:
    for d_expected, p_dim in ((0, 1), (1, 3), (2, 6), (3, 10)):
        hdul, *_ = _make_contract_tpf_hdulist(p_dim=p_dim, bkgsub=True)
        assert infer_bkg_poly_order(hdul) == d_expected

    # Explicit BKG_ORD header card overrides basis count
    hdul_ord, *_ = _make_contract_tpf_hdulist(p_dim=3, bkgsub=True)
    hdul_ord["BKG_BASIS"].header["BKG_ORD"] = 2
    assert infer_bkg_poly_order(hdul_ord) == 2

    # BKGSUB=False returns None
    hdul_unsub, *_ = _make_contract_tpf_hdulist(p_dim=3, bkgsub=False)
    assert infer_bkg_poly_order(hdul_unsub) is None

    # Conflicting BKG_ORD cards raise ValueError
    hdul_conf, *_ = _make_contract_tpf_hdulist(p_dim=3, bkgsub=True)
    hdul_conf[0].header["BKG_ORD"] = 1
    hdul_conf["BKG_BASIS"].header["BKG_ORD"] = 2
    with pytest.raises(ValueError, match="Conflicting BKG_ORD"):
        infer_bkg_poly_order(hdul_conf)


def test_write_lightcurve_fits_weightmap_round_trip(tmp_path: Path) -> None:
    time = Time(1500.0 + np.arange(30) * 0.02, format="btjd", scale="tdb")
    flux = 1.0 + 1e-3 * np.sin(2.0 * np.pi * np.arange(30) / 10.0)
    flux_err = np.full_like(flux, 2e-4)
    lc = lk.TessLightCurve(time=time, flux=flux, flux_err=flux_err)

    wmap = np.arange(20, dtype=np.float64).reshape(4, 5)
    wmap /= wmap.sum()
    wmap[0, 0] = np.nan
    lc.meta["WEIGHTMAP"] = wmap

    out_fits = tmp_path / "halo_lc.fits"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = write_lightcurve_fits(lc, out_fits)
    assert written == out_fits
    assert not any("WEIGHTMAP" in str(w.message) for w in caught)

    # Caller's in-memory lc.meta['WEIGHTMAP'] must remain intact
    assert "WEIGHTMAP" in lc.meta
    np.testing.assert_allclose(lc.meta["WEIGHTMAP"], wmap, equal_nan=True)

    with fits.open(out_fits) as hdul:
        assert hdul[0].header["HALOCVER"] == HALOCVER
        assert "WEIGHTMAP" in hdul
        np.testing.assert_allclose(hdul["WEIGHTMAP"].data, wmap, equal_nan=True)

    # Also verify jax.Array weightmap in lc.meta is stripped cleanly without FITS card warnings
    lc_jax = lk.TessLightCurve(time=time, flux=flux, flux_err=flux_err)
    jax_wmap = jnp.asarray(wmap, dtype=jnp.float32)
    lc_jax.meta["WEIGHTMAP"] = jax_wmap
    out_jax_fits = tmp_path / "halo_lc_jax.fits"
    with warnings.catch_warnings(record=True) as caught_jax:
        warnings.simplefilter("always")
        write_lightcurve_fits(lc_jax, out_jax_fits)
    assert not any("WEIGHTMAP" in str(w.message) for w in caught_jax)
    assert isinstance(lc_jax.meta["WEIGHTMAP"], type(jax_wmap))
    with fits.open(out_jax_fits) as hdul_jax:
        assert "WEIGHTMAP" in hdul_jax
        np.testing.assert_allclose(hdul_jax["WEIGHTMAP"].data, wmap, rtol=1e-6, equal_nan=True)


def test_resolve_cache_dir_fred_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_key in ("HALOPHOT_CACHE_DIR", "TESS_BRIGHT_CACHE_DIR", "SCRATCH", "XDG_CACHE_HOME"):
        monkeypatch.delenv(env_key, raising=False)

    # Verify precedence: HALOPHOT_CACHE_DIR > TESS_BRIGHT_CACHE_DIR > SCRATCH > XDG_CACHE_HOME
    xdg_dir = tmp_path / "xdg"
    scratch_root = tmp_path / "aphid_scratch"
    tbc_dir = tmp_path / "tess_bright_env"
    halo_dir = tmp_path / "halophot_env"

    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_dir))
    assert resolve_cache_dir() == (xdg_dir / "tess_bright").resolve(strict=False)

    monkeypatch.setenv("SCRATCH", str(scratch_root))
    assert resolve_cache_dir() == (scratch_root / ".cache" / "tess_bright").resolve(strict=False)

    monkeypatch.setenv("TESS_BRIGHT_CACHE_DIR", str(tbc_dir))
    assert resolve_cache_dir() == tbc_dir.resolve(strict=False)

    monkeypatch.setenv("HALOPHOT_CACHE_DIR", str(halo_dir))
    created = resolve_cache_dir(create=True)
    assert created == halo_dir.resolve(strict=False)
    assert created.is_dir()

    monkeypatch.delenv("HALOPHOT_CACHE_DIR")
    monkeypatch.delenv("TESS_BRIGHT_CACHE_DIR")

    # strict_hpc=True is the default: explicit /fred path must raise ValueError without passing strict_hpc!
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred/oz440/bpo/.cache")

    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred")

    # Allowed when strict_hpc=False, but MUST emit UserWarning
    with pytest.warns(UserWarning, match="/fred/"):
        allowed = resolve_cache_dir("/fred/oz440/bpo/.cache", strict_hpc=False)
    assert allowed == Path("/fred/oz440/bpo/.cache")

    # $SCRATCH pointing into /fred must also be caught by default (strict_hpc=True)
    monkeypatch.setenv("SCRATCH", "/fred/oz440/bpo/scratch")
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir()

    # Symlink into /fred must be caught
    fred_link = tmp_path / "sneaky_link"
    fred_link.symlink_to("/fred/oz440/bpo/cache")
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir(fred_link)

    # Prefix lookalike (/frederick) must NOT false-positive
    assert resolve_cache_dir("/frederick/cache") == Path("/frederick/cache")


def _extract_python_heredocs(script_text: str) -> list[str]:
    """Extract all ``<<'PY' ... PY`` Python heredoc bodies from a shell script."""
    pattern = re.compile(r"<<'PY'\n(.*?)\nPY(?:\n|$)", re.DOTALL)
    return [m.group(1) for m in pattern.finditer(script_text)]


def test_hpc_sbatch_and_apptainer_nt_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hpc_dir = repo_root / "hpc"

    sbatch_paths = sorted(hpc_dir.glob("*.sbatch"))
    expected_sbatch = {
        "build_halo_sif.sbatch",
        "halophot_survey_cpu.sbatch",
        "halophot_survey_gpu.sbatch",
    }
    assert expected_sbatch.issubset({p.name for p in sbatch_paths})

    bash_bin = shutil.which("bash")

    for sbatch_path in sbatch_paths:
        text = sbatch_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        active_lines = [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.lstrip().startswith("#")
        ]

        # 1. Shebang & Lmod-safe shell strict mode (never -u, which fails on unset PS1)
        assert lines[0] == "#!/bin/bash", f"{sbatch_path.name} must start with #!/bin/bash"
        assert "set -euo pipefail" not in text, f"{sbatch_path.name} must not set -u"
        assert any(ln.startswith("set -eo pipefail") for ln in active_lines), (
            f"{sbatch_path.name} must set -eo pipefail"
        )

        # 2. Slurm log paths under /aphid/scratch-3month/%u/logs/ & <= 15-char job name
        assert re.search(
            r"^#SBATCH\s+--output=/aphid/scratch-3month/%u/logs/\S+\.out$",
            text,
            re.MULTILINE,
        ), f"{sbatch_path.name} missing scratch --output directive"
        assert re.search(
            r"^#SBATCH\s+--error=/aphid/scratch-3month/%u/logs/\S+\.err$",
            text,
            re.MULTILINE,
        ), f"{sbatch_path.name} missing scratch --error directive"
        job_match = re.search(r"^#SBATCH\s+--job-name=(\S+)$", text, re.MULTILINE)
        assert job_match is not None and len(job_match.group(1)) <= 15

        # 3. Module load apptainer appears BEFORE any apptainer invocation
        mod_indices = [
            i for i, ln in enumerate(active_lines) if ln == "module -q load apptainer"
        ]
        app_indices = [
            i for i, ln in enumerate(active_lines) if ln.startswith("apptainer ") or " apptainer " in ln
        ]
        assert mod_indices, f"{sbatch_path.name} must call `module -q load apptainer`"
        assert app_indices, f"{sbatch_path.name} must invoke apptainer"
        assert mod_indices[0] < app_indices[0], (
            f"{sbatch_path.name} must load apptainer module before invoking apptainer"
        )

        # 4. Apptainer cache/tmp redirected off /home onto scratch/jobfs, and halo.sif default
        cachedir_lines = [ln for ln in active_lines if "APPTAINER_CACHEDIR=" in ln]
        tmpdir_lines = [ln for ln in active_lines if "APPTAINER_TMPDIR=" in ln]
        assert cachedir_lines and all("/home" not in ln and "~" not in ln for ln in cachedir_lines)
        assert tmpdir_lines and all("/home" not in ln and "~" not in ln for ln in tmpdir_lines)
        assert any(
            any(tok in ln for tok in ("$SCRATCH", "$BUILD_DIR", "${SCRATCH"))
            for ln in cachedir_lines
        )
        assert any(
            any(tok in ln for tok in ("$SCRATCH", "$BUILD_DIR", "$JOBFS", "${JOBFS"))
            for ln in tmpdir_lines
        )
        assert "halo.sif" in text, f"{sbatch_path.name} must default/link to halo.sif"
        assert "virgil.sif" not in text, f"{sbatch_path.name} must not default to virgil.sif"

        # 5. Embedded Python heredocs parse cleanly & bash -n syntax check succeeds
        for idx, py_src in enumerate(_extract_python_heredocs(text)):
            ast.parse(py_src, filename=f"{sbatch_path.name}[heredoc_{idx}]")
        if bash_bin is not None:
            subprocess.run([bash_bin, "-n", str(sbatch_path)], check=True)

    # Verify root-mapped Apptainer recipe & builder specifics
    def_text = (hpc_dir / "virgil_halophot.def").read_text(encoding="utf-8")
    assert "Bootstrap: docker" in def_text
    assert "From: python:3.12.7-bookworm" in def_text
    assert "apt-get" not in def_text
    assert "mundey_ref=dev" in def_text
    assert "tessbkgd_ref=dev" in def_text
    assert "halophot_ref=dev" in def_text
    assert "--build-arg mundey_ref=v2.0.0" in def_text
    assert "UV_SYSTEM_PYTHON=1" in def_text
    assert 'uv pip install -U "jax[cuda12]"' in def_text
    assert "/etc/halo-sif-freeze.txt" in def_text
    assert "rm -rf /opt/.build" in def_text

    build_text = (hpc_dir / "build_halo_sif.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --time=00:45:00" in build_text
    assert "#SBATCH --mem=8G" in build_text
    assert 'MUNDEY_REF="${MUNDEY_REF:-dev}"' in build_text
    assert 'TESSBKGD_REF="${TESSBKGD_REF:-dev}"' in build_text
    assert 'HALOPHOT_REF="${HALOPHOT_REF:-dev}"' in build_text
    assert "--fakeroot" in build_text
    assert "--ignore-fakeroot-command" in build_text
    assert '--bind "$APPTAINER_TMPDIR:/tmp"' in build_text
    assert 'ln -sfn "$SIF" "$MYFRED/containers/halo.sif"' in build_text
    assert ".sha256" in build_text


def test_hpc_survey_embedded_full_e2e_and_halophot_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hpc_dir = repo_root / "hpc"

    cpu_text = (hpc_dir / "halophot_survey_cpu.sbatch").read_text(encoding="utf-8")
    gpu_text = (hpc_dir / "halophot_survey_gpu.sbatch").read_text(encoding="utf-8")
    for text in (cpu_text, gpu_text):
        assert 'PIPELINE_MODE="${PIPELINE_MODE:-halophot_only}"' in text
        assert 'APPTAINER_BIND="/fred,/aphid/scratch-3month${JOBFS:+,$JOBFS}"' in text

    cpu_heredocs = _extract_python_heredocs(cpu_text)
    gpu_heredocs = _extract_python_heredocs(gpu_text)
    assert len(cpu_heredocs) == 1
    assert len(gpu_heredocs) == 1

    hdul, *_ = _make_contract_tpf_hdulist(n_raw=20, ny=4, nx=5, p_dim=3, bkgsub=True)
    tpf_path = tmp_path / "tess2019001-s0006-0000000123456789-0126-s_tp.fits"
    hdul.writeto(tpf_path, overwrite=True)

    cache_dir = tmp_path / "scratch_cache" / "tess_bright"
    cache_dir.mkdir(parents=True)
    cat_dir = tmp_path / "catalogs"
    cat_dir.mkdir(parents=True)

    calls: list[tuple[str, dict[str, object]]] = []

    fake_mundey = types.ModuleType("mundey")

    def _fake_recalibrate_tpf(path: object, **kwargs: object) -> TargetPixelFile:
        calls.append(("recalibrate_tpf", {"path": Path(str(path)), **kwargs}))
        loaded = lk.read(str(path))
        assert isinstance(loaded, TargetPixelFile)
        return loaded

    fake_mundey.recalibrate_tpf = _fake_recalibrate_tpf  # type: ignore[attr-defined]

    fake_tessbkgd = types.ModuleType("tessbkgd")

    def _fake_subtract_background(tpf_in: TargetPixelFile) -> TargetPixelFile:
        calls.append(("subtract_background", {"tpf_type": type(tpf_in).__name__}))
        return tpf_in

    fake_tessbkgd.subtract_background = _fake_subtract_background  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mundey", fake_mundey)
    monkeypatch.setitem(sys.modules, "tessbkgd", fake_tessbkgd)
    monkeypatch.setenv("MUNDEY_SMEAR", "alternate")

    # 1. Execute CPU survey full_e2e Python heredoc end-to-end
    cpu_out = tmp_path / "cpu_out"
    cpu_out.mkdir()
    stem = tpf_path.name.removesuffix(".fits")
    monkeypatch.setattr(
        sys,
        "argv",
        ["-", str(tpf_path), stem, str(cpu_out), str(cache_dir), str(cat_dir / f"{stem}_cat.parquet")],
    )
    exec(compile(cpu_heredocs[0], "halophot_survey_cpu.sbatch[py]", "exec"), {"__name__": "__main__"})

    cpu_fits = cpu_out / f"{stem}_halo_lc_tv.fits"
    assert cpu_fits.is_file()
    assert len(calls) == 2
    assert calls[0][0] == "recalibrate_tpf"
    assert calls[0][1]["smear"] == "alternate"
    assert calls[0][1]["offline"] is True
    assert calls[0][1]["strict_hpc"] is True
    assert calls[1][0] == "subtract_background"

    # 2. Execute GPU survey Python heredoc in both halophot_only and full_e2e modes
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"# comment\n{tpf_path}\n", encoding="utf-8")

    for mode, expected_new_calls in (("halophot_only", 0), ("full_e2e", 2)):
        calls.clear()
        gpu_out = tmp_path / f"gpu_{mode}"
        gpu_out.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["-", str(manifest), str(gpu_out), str(cat_dir), str(cache_dir), mode],
        )
        exec(
            compile(gpu_heredocs[0], "halophot_survey_gpu.sbatch[py]", "exec"),
            {"__name__": "__main__"},
        )
        assert len(calls) == expected_new_calls
        out_fits = gpu_out / f"{stem}_halo_lc.fits"
        assert out_fits.is_file()
        summary = Table.read(gpu_out / "summary.parquet", format="parquet")
        assert len(summary) == 1
        assert str(summary["status"][0]) == "ok"
        assert str(summary["pipeline_mode"][0]) == mode


def test_license_and_pyproject_metadata() -> None:
    """Verify [HALO-02] and [HALO-06]: GPLv3 LICENSE, pyproject.toml extras/excludes, and no .pyc."""
    repo_root = Path(__file__).resolve().parents[1]
    lic_path = repo_root / "LICENSE"
    assert lic_path.is_file()
    lic_text = lic_path.read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in lic_text
    assert "Version 3, 29 June 2007" in lic_text

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
    hpc_deps = " ".join(pyproject["project"]["optional-dependencies"]["hpc"])
    assert "pyarrow" in hpc_deps
    assert "fsspec" in hpc_deps
    assert "s3fs" in hpc_deps
    ruff_excl = set(pyproject["tool"]["ruff"]["extend-exclude"])
    assert {"notebooks", "docs", "tv_halo.ipynb", "halo_atlas.ipynb", "experiments"}.issubset(
        ruff_excl
    )
    assert not (repo_root / "experiments" / "psf_sim.pyc").exists()


def test_write_fits_headers_tunit_and_read_weightmap(tmp_path: Path) -> None:
    """Verify [HALO-04]: INSTRUME='TESS Photometer', provenance cards, e-/s TUNIT, and read_weightmap."""
    t = Time(1900.0 + np.arange(16, dtype=np.float64) * (2.0 / 1440.0), format="btjd", scale="tdb")
    flux_q = u.Quantity(np.linspace(800.0, 820.0, 16, dtype=np.float64), u.electron / u.s)
    ferr_q = u.Quantity(np.full(16, 2.5, dtype=np.float64), u.electron / u.s)
    lc = lk.TessLightCurve(time=t, flux=flux_q, flux_err=ferr_q)
    wmap = np.array([[0.25, 0.25, 0.0], [0.25, 0.25, np.nan]], dtype=np.float64)
    lc.meta.update(
        {
            "WEIGHTMAP": wmap,
            "BKGSUB": True,
            "BKG_ORD": 2,
            "HALO_OBJ": "tv_exp",
            "HALO_LAG": 2,
            "LEGACYTV": False,
        }
    )

    out_path = tmp_path / "tess_raw_halo_lc.fits"
    halophot.write_fits(lc, out_path)

    with fits.open(out_path) as hdul:
        hdr0 = hdul[0].header
        assert hdr0["TELESCOP"] == "TESS"
        assert hdr0["INSTRUME"] == "TESS Photometer"
        assert hdr0.get("OBJECT") != "None"
        assert hdr0["HALOCVER"] == HALOCVER
        assert hdr0["BKGSUB"] is True
        assert hdr0["BKG_ORD"] == 2
        assert hdr0["HALO_OBJ"] == "tv_exp"
        assert hdr0["HALO_LAG"] == 2
        assert hdr0["LEGACYTV"] is False

    lc_back = lk.read(str(out_path))
    assert lc_back.flux.unit == u.electron / u.s
    assert lc_back.flux_err.unit == u.electron / u.s
    np.testing.assert_allclose(lc_back.flux.value, flux_q.value, rtol=1e-6)
    np.testing.assert_allclose(lc_back.flux_err.value, ferr_q.value, rtol=1e-6)

    w_mem = halophot.read_weightmap(lc)
    w_disk = read_weightmap(out_path)
    assert w_mem is not None and w_disk is not None
    np.testing.assert_allclose(w_mem, wmap, equal_nan=True)
    np.testing.assert_allclose(w_disk, wmap, equal_nan=True)

