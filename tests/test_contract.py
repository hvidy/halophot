"""Tests for the ``HALOCVER = 1`` FITS contract and HPC ``/fred`` cache guard."""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from halophot._contract import (
    HALOCVER,
    PixelQuality,
    pack_bkg_chol,
    read_bkg_covariance,
    read_bkgsub_state,
    read_pixel_quality,
    read_smear_err,
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
    scratch_root = tmp_path / "aphid_scratch"
    monkeypatch.setenv("SCRATCH", str(scratch_root))
    assert resolve_cache_dir(strict_hpc=True) == scratch_root / ".cache" / "tess_bright"

    # Explicit /fred path must raise in strict_hpc=True
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred/oz440/bpo/.cache", strict_hpc=True)

    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred", strict_hpc=True)

    # Allowed when strict_hpc=False
    assert resolve_cache_dir("/fred/oz440/bpo/.cache", strict_hpc=False) == Path(
        "/fred/oz440/bpo/.cache"
    )

    # $SCRATCH pointing into /fred must also be caught in strict_hpc=True
    monkeypatch.setenv("SCRATCH", "/fred/oz440/bpo/scratch")
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir(strict_hpc=True)

    # Symlink into /fred must be caught
    fred_link = tmp_path / "sneaky_link"
    fred_link.symlink_to("/fred/oz440/bpo/cache")
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir(fred_link, strict_hpc=True)

    # Prefix lookalike (/frederick) must NOT false-positive
    assert resolve_cache_dir("/frederick/cache", strict_hpc=True) == Path("/frederick/cache")
