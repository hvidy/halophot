"""Tests for ``halophot.masking``, coherent ``flux_err`` Monte Carlo, and ``extract_halo_lightcurve``."""

from __future__ import annotations

import types
import warnings
from pathlib import Path

import astropy.units as u
import jax
import lightkurve as lk
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

import halophot
from halophot import cli
from halophot._contract import (
    HALOCVER,
    BkgCovariance,
    PixelQuality,
    pack_bkg_chol,
    read_pixel_quality,
    read_split_times,
    write_lightcurve_fits,
)
from halophot.lightcurve import (
    column_orthogonal_smear_weights,
    compute_coherent_flux_err,
    extract_halo_lightcurve,
)
from halophot.masking import (
    KEPLER_SAT_RATE_E_PER_S,
    TESS_SAT_RATE_E_PER_S,
    build_halo_masks,
    mask_background_stars,
    remove_stars,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EPIC_FITS = REPO_ROOT / "EPIC_211309989_mast.fits"
K2_TPF_FITS = REPO_ROOT / "data" / "ktwo205897543-c03_lpd-targ.fits.gz"


def _make_celestial_wcs(ny: int, nx: int, ra0: float = 101.287, dec0: float = -16.716) -> WCS:
    """Create a standard 21 arcsec/pix TESS TAN celestial WCS centred on (ra0, dec0)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = np.array([-21.0 / 3600.0, 21.0 / 3600.0], dtype=np.float64)
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _build_synthetic_halocver_tpf_file(
    path: Path,
    *,
    flux: np.ndarray,
    flux_err: np.ndarray,
    flux_bkg: np.ndarray,
    flux_bkg_err: np.ndarray | None = None,
    smear_err: np.ndarray | None = None,
    chol: np.ndarray | None = None,
    basis: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    bkgsub: bool = True,
    bkg_ord: int | None = None,
    pos_corr1: np.ndarray | None = None,
    pos_corr2: np.ndarray | None = None,
    ra0: float = 101.287,
    dec0: float = -16.716,
) -> lk.TessTargetPixelFile:
    """Write a complete HALOCVER=1 TESS TargetPixelFile to ``path`` and load via ``lk.read``."""
    n_cad, ny, nx = flux.shape
    btjd = 2000.0 + np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)
    cadenceno = np.arange(1, n_cad + 1, dtype=np.int32)
    quality = np.zeros(n_cad, dtype=np.int32)

    if flux_bkg_err is None:
        flux_bkg_err = np.zeros_like(flux, dtype=np.float32)
    if pos_corr1 is None:
        pos_corr1 = np.zeros(n_cad, dtype=np.float32)
    if pos_corr2 is None:
        pos_corr2 = np.zeros(n_cad, dtype=np.float32)

    wcs_2d = _make_celestial_wcs(ny, nx, ra0=ra0, dec0=dec0)

    prim = fits.PrimaryHDU()
    prim.header["MISSION"] = "TESS"
    prim.header["TELESCOP"] = "TESS"
    prim.header["INSTRUME"] = "TESS Photometer"
    prim.header["ORIGIN"] = "NASA/Ames"
    prim.header["CREATOR"] = "1234 TargetPixelExporterPipelineModule"
    prim.header["OBJECT"] = "HALO V2 SYNTH"
    prim.header["TICID"] = 555666777
    prim.header["SECTOR"] = 18
    prim.header["CAMERA"] = 1
    prim.header["CCD"] = 2
    prim.header["RA_OBJ"] = ra0
    prim.header["DEC_OBJ"] = dec0
    prim.header["HALOCVER"] = HALOCVER
    prim.header["BKGSUB"] = bkgsub

    dim_str = f"({nx},{ny})"
    cols = [
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=btjd),
        fits.Column(name="TIMECORR", format="E", unit="d", array=np.zeros(n_cad, dtype=np.float32)),
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(
            name="RAW_CNTS",
            format=f"{ny * nx}J",
            dim=dim_str,
            unit="count",
            array=np.full((n_cad, ny, nx), 1000, dtype=np.int32),
        ),
        fits.Column(
            name="FLUX",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=flux.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_ERR",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=flux_err.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_BKG",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=flux_bkg.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_BKG_ERR",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=flux_bkg_err.astype(np.float32),
        ),
        fits.Column(name="QUALITY", format="J", array=quality),
        fits.Column(
            name="POS_CORR1",
            format="E",
            unit="pixel",
            array=np.asarray(pos_corr1, dtype=np.float32),
        ),
        fits.Column(
            name="POS_CORR2",
            format="E",
            unit="pixel",
            array=np.asarray(pos_corr2, dtype=np.float32),
        ),
    ]
    if smear_err is not None:
        cols.append(
            fits.Column(
                name="SMEAR_ERR",
                format=f"{nx}E",
                unit="e-/s",
                array=smear_err.astype(np.float32),
            )
        )
    if chol is not None:
        packed = pack_bkg_chol(chol)
        cols.append(fits.Column(name="BKG_CHOL", format=f"{packed.shape[1]}E", array=packed))

    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    pixels_hdu.header["INSTRUME"] = "TESS Photometer"
    pixels_hdu.header["HALOCVER"] = HALOCVER
    pixels_hdu.header["BKGSUB"] = bkgsub

    ap_hdu = fits.ImageHDU(
        data=np.full((ny, nx), 3, dtype=np.int32),
        header=wcs_2d.to_header(),
        name="APERTURE",
    )
    hdus: list[fits.PrimaryHDU | fits.BinTableHDU | fits.ImageHDU] = [prim, pixels_hdu, ap_hdu]

    if pixel_quality is not None:
        hdus.append(
            fits.ImageHDU(data=np.asarray(pixel_quality, dtype=np.uint16), name="PIXEL_QUALITY")
        )
    if basis is not None:
        basis_hdu = fits.ImageHDU(data=basis.astype(np.float32), name="BKG_BASIS")
        if bkg_ord is not None:
            basis_hdu.header["BKG_ORD"] = int(bkg_ord)
        hdus.append(basis_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=True)
    loaded = lk.read(str(path))
    assert isinstance(loaded, lk.TessTargetPixelFile)
    return loaded


def test_sky_dilution_guard_preserves_pulsation_and_transit_amplitude(tmp_path: Path) -> None:
    """Prove v2.0 sky-dominance guard preserves astrophysical amplitude under positive sky residual."""
    n_cad, ny, nx = 96, 4, 4
    t = np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)

    # 3% pulsation + 4% box transit on cadences 40..52
    transit = np.where((np.arange(n_cad) >= 40) & (np.arange(n_cad) < 52), -0.04, 0.0)
    stellar = 1.0 + 0.03 * np.sin(2.0 * np.pi * t / (30.0 * 2.0 / 1440.0)) + transit
    jitter = 0.08 * ((np.arange(n_cad) % 9) / 8.0 - 0.5)

    flux = np.zeros((n_cad, ny, nx), dtype=np.float64)
    flux_bkg = np.full((n_cad, ny, nx), 120.0, dtype=np.float64)

    # Rows 0..1: bright halo wing pixels (mean ~ 1200 >= 2 * 120) with complementary jitter
    for r in range(2):
        for c in range(nx):
            sens = -1.0 if (c % 2 == 0) else 1.0
            flux[:, r, c] = 1200.0 * stellar * (1.0 + sens * jitter)

    # Rows 2..3: sky-dominated outer pixels with +60 e-/s unsubtracted sky pedestal residual
    # and almost zero starlight (mean ~ 62 < 2 * 120 = 240): normalized curve is nearly flat!
    for r in range(2, ny):
        for c in range(nx):
            flux[:, r, c] = 2.0 * stellar + 60.0

    flux_err = np.sqrt(flux + flux_bkg)
    tpf_path = tmp_path / "sky_guard_tpf.fits"
    tpf = _build_synthetic_halocver_tpf_file(
        tpf_path,
        flux=flux,
        flux_err=flux_err,
        flux_bkg=flux_bkg,
        bkgsub=True,
    )

    # Default v2.0 (legacy_tv=False) excludes rows 2..3 via P_bar >= 2 * B_i and recovers full signal
    lc_v2 = extract_halo_lightcurve(tpf, maxiter=151)
    wmap_v2 = lc_v2.meta["WEIGHTMAP"]
    assert np.all(wmap_v2[2:, :] == 0.0)
    assert np.isclose(np.sum(wmap_v2[:2, :]), 1.0, atol=1e-6)
    expected_norm = stellar / np.mean(stellar)
    np.testing.assert_allclose(lc_v2.flux.value, expected_norm, atol=2e-3)

    # Under legacy_tv=True (unregularized w-simplex without sky guard), TV shifts weight onto the
    # flat sky-residual pixels and severely dilutes the pulsation + transit amplitude!
    lc_legacy = extract_halo_lightcurve(tpf, legacy_tv=True, maxiter=151)
    amp_true = float(np.ptp(expected_norm))
    amp_v2 = float(np.ptp(lc_v2.flux.value))
    amp_legacy = float(np.ptp(lc_legacy.flux.value))
    assert amp_v2 > 0.95 * amp_true
    assert amp_legacy < 0.20 * amp_true


def test_2d_bleed_dilation_and_pixel_quality_disjointness() -> None:
    """Verify 2D (3, 3) saturation+undershoot dilation and PIXEL_QUALITY bit exclusion."""
    n_cad, ny, nx = 30, 7, 7
    flux = np.full((n_cad, ny, nx), 500.0, dtype=np.float64)
    flux[:, 3, 3] = 50000.0  # Single saturated pixel at stamp centre
    flux_err = np.full_like(flux, 5.0)

    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 0] = int(PixelQuality.STRAP_COL)
    pq[0, :] |= int(PixelQuality.BKG_USED)
    pq[6, 6] = int(PixelQuality.SOURCE_MASK)
    pq[6, 1] = int(PixelQuality.SMEAR_UNRESOLVED)
    # Low-SNR pixel at (5, 5): mean = 500, err = 200 -> SNR = 2.5 < 5.0
    flux_err[:, 5, 5] = 200.0

    res = build_halo_masks(
        flux,
        flux_err=flux_err,
        pixel_quality=pq,
        thresh=0.8,
        bleed_dilation=(3, 3),
        min_snr=5.0,
    )
    # Full 3x3 block around (3, 3) must be masked (both vertical bleed AND horizontal undershoot!)
    assert int(np.sum(res.saturated_mask)) == 9
    assert np.all(res.saturated_mask[2:5, 2:5])
    assert not np.any(res.pixel_mask[2:5, 2:5])

    # Strict disjointness with BKG_USED and all PIXEL_QUALITY flags
    bkg_used_pixels = (pq & int(PixelQuality.BKG_USED)) != 0
    assert not np.any(res.pixel_mask & bkg_used_pixels)
    assert not np.any(res.pixel_mask[:, 0])
    assert not res.pixel_mask[6, 6]
    assert not res.pixel_mask[6, 1]
    assert not res.pixel_mask[5, 5]

    # Verify physical TESS full-well saturation rate (7.5e4 e-/s < 9.0e4 < 1.5e5) when thresh < 0
    assert TESS_SAT_RATE_E_PER_S == pytest.approx(7.5e4)
    assert KEPLER_SAT_RATE_E_PER_S == pytest.approx(1.0e6 / 6.02)
    flux_tess_sat = np.full((n_cad, ny, nx), 800.0, dtype=np.float64)
    flux_tess_sat[:, 3, 3] = 9.0e4
    res_tess = build_halo_masks(
        flux_tess_sat,
        flux_err=np.sqrt(flux_tess_sat),
        thresh=-1.0,
        mission="TESS",
        bleed_dilation=(3, 3),
    )
    assert int(np.sum(res_tess.saturated_mask)) == 9
    assert np.all(res_tess.saturated_mask[2:5, 2:5])

    # Small (3, 3) stamp where saturation + (3, 3) bleed dilation leaves 0 pixels MUST raise
    # ValueError rather than silently clearing saturated_mask to False!
    flux_small_bleed = np.full((n_cad, 3, 3), 800.0, dtype=np.float64)
    flux_small_bleed[:, 1, 1] = 9.0e4
    with pytest.raises(ValueError, match="No valid halo pixels survived"):
        build_halo_masks(
            flux_small_bleed,
            flux_err=np.sqrt(flux_small_bleed),
            thresh=-1.0,
            mission="TESS",
            bleed_dilation=(3, 3),
        )


@pytest.mark.parametrize("bkg_order", [1, 2])
def test_monte_carlo_unified_coherent_flux_err_with_smear_projection(bkg_order: int) -> None:
    """Verify Eq. (6.4.1) coherent flux_err against a 6,000-trial generative Monte Carlo."""
    rng = np.random.default_rng(202604 + bkg_order)
    n_mc = 6000
    ny, nx = 7, 7

    # Construct 2D Chebyshev polynomial basis of total degree <= bkg_order on [-1, 1]^2
    xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    eta = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xi, eta)
    terms: list[np.ndarray] = []
    for p in range(bkg_order + 1):
        for q in range(bkg_order + 1 - p):
            terms.append(
                np.polynomial.chebyshev.chebval(xx, [0.0] * p + [1.0])
                * np.polynomial.chebyshev.chebval(yy, [0.0] * q + [1.0])
            )
    phi_raw = np.stack(terms, axis=0)  # (P, ny, nx)
    assert phi_raw.shape[0] == (bkg_order + 1) * (bkg_order + 2) // 2

    # Disjoint background rows {0, 3, 6} x {0..nx-1} vs halo rows {1, 2, 4, 5} x {0..nx-1}:
    # balanced across columns and 3 distinct y coordinates so degree <= 2 in (x, y) is full-rank.
    bg_mask = np.zeros((ny, nx), dtype=bool)
    bg_mask[[0, 3, 6], :] = True

    halo_mask = ~bg_mask
    assert not np.any(halo_mask & bg_mask)

    # QR-equilibrate basis on Omega_bg just as tessbkgd v2.0 does
    _, r_bg = np.linalg.qr(phi_raw[:, bg_mask].T, mode="reduced")  # (P, P)
    phi_full = np.einsum("ij,jrc->irc", np.linalg.inv(r_bg), phi_raw)  # (P, ny, nx)
    phi_bg = phi_full[:, bg_mask].T  # (M_bg, P), orthonormal columns!

    # Heteroscedastic pixel shot/read noise and strong column-coherent smear noise
    sigma_pix = rng.uniform(1.2, 3.0, size=(ny, nx))
    sigma_smear = rng.uniform(2.5, 5.5, size=nx)

    # Non-uniform physical halo weights w* summing to 1 on Omega_halo
    w_2d = np.zeros((ny, nx), dtype=np.float64)
    raw_w = rng.uniform(0.1, 2.0, size=int(np.sum(halo_mask)))
    w_2d[halo_mask] = raw_w / np.sum(raw_w)
    mean_p = np.full((ny, nx), 800.0, dtype=np.float64)
    norm_factor = float(np.sum(w_2d * mean_p))  # 800.0

    # OLS background fit beta_hat = a_mat @ u_bg on Omega_bg and sandwich covariance L L^T:
    # degree <= bkg_order smear modes are reproduced identically by hat_B and cancelled everywhere,
    # while degree > bkg_order smear modes eps_perp lie in ker(a_mat), so sandwich residuals on
    # Omega_bg carry purely heteroscedastic pixel variance!
    a_mat = np.linalg.solve(phi_bg.T @ phi_bg, phi_bg.T)  # (P, M_bg)
    bg_rows, bg_cols = np.where(bg_mask)
    cov_u_bg = np.diag(np.square(sigma_pix[bg_rows, bg_cols]))
    cov_beta = a_mat @ cov_u_bg @ a_mat.T
    chol_beta = np.linalg.cholesky(cov_beta)

    eps_pix = rng.normal(0.0, 1.0, size=(n_mc, ny, nx)) * sigma_pix[None, :, :]
    eps_smear = rng.normal(0.0, 1.0, size=(n_mc, nx)) * sigma_smear[None, :]
    u_mc = mean_p[None, :, :] + eps_pix + eps_smear[:, None, :]

    # Case A: BKGSUB = False (unsubtracted)
    f_unsub_mc = np.einsum("trc,rc->t", u_mc, w_2d) / norm_factor
    emp_std_unsub = float(np.std(f_unsub_mc, ddof=1))

    fe_unsub = np.sqrt(np.square(sigma_pix)[None, :, :] + np.square(sigma_smear)[None, None, :])
    analytic_unsub = compute_coherent_flux_err(
        w_2d,
        flux_err=fe_unsub,
        smear_err=sigma_smear[None, :],
        bkg_cov=None,
        bkgsub=False,
        bkg_order=None,
        norm_factor=norm_factor,
    )[0]
    assert analytic_unsub == pytest.approx(emp_std_unsub, rel=0.03)

    # Case B: BKGSUB = True (subtract polynomial fit hat_B = beta_hat @ phi_full on every trial!)
    u_bg_mc = u_mc[:, bg_rows, bg_cols]  # (n_mc, M_bg)
    beta_mc = u_bg_mc @ a_mat.T  # (n_mc, P)
    b_hat_mc = np.einsum("tp,prc->trc", beta_mc, phi_full)
    p_sub_mc = u_mc - b_hat_mc
    f_sub_mc = np.einsum("trc,rc->t", p_sub_mc, w_2d) / norm_factor
    emp_std_sub = float(np.std(f_sub_mc, ddof=1))

    # Per-pixel background variance sigma_B^2 = ||L^T phi_{r,c}||_2^2 included in FLUX_ERR^2
    sigma_b_sq = np.sum(np.square(np.einsum("qp,qrc->prc", chol_beta, phi_full)), axis=0)
    fe_sub = np.sqrt(
        np.square(sigma_pix) + np.square(sigma_smear)[None, :] + sigma_b_sq
    )[None, :, :]

    # Include a second cadence with all-NaN Cholesky to verify NaN propagation simultaneously
    chol_2cad = np.stack([chol_beta, np.full_like(chol_beta, np.nan)], axis=0)
    fe_2cad = np.repeat(fe_sub, 2, axis=0)
    se_2cad = np.repeat(sigma_smear[None, :], 2, axis=0)

    analytic_sub_arr = compute_coherent_flux_err(
        w_2d,
        flux_err=fe_2cad,
        smear_err=se_2cad,
        bkg_cov=BkgCovariance(chol=chol_2cad, basis=phi_full),
        bkgsub=True,
        bkg_order=bkg_order,
        norm_factor=norm_factor,
    )
    assert np.isnan(analytic_sub_arr[1])
    assert analytic_sub_arr[0] == pytest.approx(emp_std_sub, rel=0.03)

    # Verify g_perp is strictly orthogonal to degree <= bkg_order column polynomials
    col_g = np.sum(w_2d, axis=0)
    g_perp = column_orthogonal_smear_weights(col_g, bkg_order)
    for deg in range(bkg_order + 1):
        assert abs(float(np.dot(g_perp, xi**deg))) < 1e-14

    # Naive pixel-only quadrature error severely underestimates coherent smear+background scatter
    naive_quad = float(np.sqrt(np.sum((w_2d**2) * fe_sub[0] ** 2)) / norm_factor)
    assert abs(naive_quad - emp_std_sub) / emp_std_sub > 0.15


def test_wcs_star_masking_offline_and_remove_stars(tmp_path: Path) -> None:
    """Verify WCS contaminant masking with pre-cached catalog_table, target exclusion, and --offline."""
    n_cad, ny, nx = 24, 8, 9
    ra0, dec0 = 101.287, -16.716
    wcs_2d = _make_celestial_wcs(ny, nx, ra0=ra0, dec0=dec0)

    # Offline without catalog_table must raise RuntimeError with zero network access
    with pytest.raises(RuntimeError, match="offline=True"):
        mask_background_stars(wcs_2d, (ny, nx), catalog_table=None, offline=True)

    # Place one bright contaminant star at pixel (col=6, row=2) and include the primary target itself
    contam_sky = wcs_2d.pixel_to_world(6.0, 2.0)
    tgt_sky = wcs_2d.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    cat = Table(
        {
            "ra": [tgt_sky.ra.deg, contam_sky.ra.deg],
            "dec": [tgt_sky.dec.deg, contam_sky.dec.deg],
            "Tmag": [1.5, 11.0],
        }
    )

    star_mask = mask_background_stars(
        wcs_2d,
        (ny, nx),
        catalog_table=cat,
        offline=True,
        target_coord=(ra0, dec0),
        target_exclusion_arcsec=8.0,
        radius_pix=1.5,
    )
    assert star_mask.shape == (ny, nx)
    assert bool(star_mask[2, 6]) is True
    # Primary target at stamp centre (row=3.5, col=4.0) must NOT be masked!
    assert bool(star_mask[ny // 2, nx // 2]) is False

    # Verify remove_stars on a TargetPixelFile sets PIXEL_QUALITY SOURCE_MASK non-mutatingly
    flux = np.full((n_cad, ny, nx), 600.0, dtype=np.float64)
    tpf_path = tmp_path / "wcs_tpf.fits"
    tpf = _build_synthetic_halocver_tpf_file(
        tpf_path,
        flux=flux,
        flux_err=np.sqrt(flux),
        flux_bkg=np.full_like(flux, 20.0),
        ra0=ra0,
        dec0=dec0,
    )
    cleaned_tpf = remove_stars(
        tpf,
        catalog_table=cat,
        offline=True,
        radius_pix=1.5,
    )
    assert np.all(np.isfinite(tpf.flux.value[:, 2, 6]))  # Input TPF untouched
    assert np.all(np.isnan(cleaned_tpf.flux.value[:, 2, 6]))
    pq_clean = read_pixel_quality(cleaned_tpf)
    assert (pq_clean[2, 6] & int(PixelQuality.SOURCE_MASK)) != 0

    # Verify TESS plate-scale contrast cut, NaN-magnitude handling/coalescing, and high-PM target guard:
    # Primary star at J2000 (ra0, dec0) has pmra=+600 mas/yr, pmdec=-500 mas/yr -> moves ~12.5" by J2016!
    pm_ra_mas = 600.0
    pm_dec_mas = -500.0
    dt_yr = 16.0
    cos_dec = np.cos(np.deg2rad(dec0))
    ra_2016 = ra0 + (pm_ra_mas * 1e-3 * dt_yr) / (3600.0 * cos_dec)
    dec_2016 = dec0 + (pm_dec_mas * 1e-3 * dt_yr) / 3600.0

    faint_t = wcs_2d.pixel_to_world(1.0, 1.0)
    faint_gaia_nan_t = wcs_2d.pixel_to_world(1.0, 6.0)
    all_nan_mag = wcs_2d.pixel_to_world(7.0, 6.0)
    bright_nb = wcs_2d.pixel_to_world(6.0, 2.0)

    cat_pm = Table(
        {
            "ra": [
                ra_2016,
                faint_t.ra.deg,
                faint_gaia_nan_t.ra.deg,
                all_nan_mag.ra.deg,
                bright_nb.ra.deg,
            ],
            "dec": [
                dec_2016,
                faint_t.dec.deg,
                faint_gaia_nan_t.dec.deg,
                all_nan_mag.dec.deg,
                bright_nb.dec.deg,
            ],
            "Tmag": [4.0, 16.8, np.nan, np.nan, 9.5],
            "phot_g_mean_mag": [4.2, 17.1, 17.2, np.nan, 9.8],
            "pmRA": [pm_ra_mas, 0.0, 0.0, 0.0, 0.0],
            "pmDEC": [pm_dec_mas, 0.0, 0.0, 0.0, 0.0],
            "ref_epoch": [2016.0, 2016.0, 2016.0, 2016.0, 2016.0],
        }
    )
    pm_mask = mask_background_stars(
        wcs_2d,
        (ny, nx),
        catalog_table=cat_pm,
        offline=True,
        target_coord=(ra0, dec0),
        obs_time=Time(2019.0, format="jyear", scale="tdb"),
        catalog="Gaia",
    )
    # High-PM primary star must NOT self-mask, and faint/NaN stars must NOT wipe out TESS pixels!
    assert bool(pm_mask[ny // 2, nx // 2]) is False
    assert bool(pm_mask[1, 1]) is False
    assert bool(pm_mask[6, 1]) is False
    assert bool(pm_mask[6, 7]) is False
    # Bright T=9.5 neighbour at (row=2, col=6) IS masked with a compact TESS PRF disc
    assert bool(pm_mask[2, 6]) is True
    assert int(np.sum(pm_mask)) <= 16


def test_rectangular_weightmap_splits_centroids_and_bounds(tmp_path: Path) -> None:
    """Verify non-transposed (ny, nx) and (n_splits, ny, nx) WEIGHTMAP, centroids, and split bounds."""
    rng = np.random.default_rng(5150)
    n_cad, ny, nx = 198, 2, 4  # Explicitly rectangular ny != nx (M=8 reuses compiled XLA kernel!)
    t = np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)
    stellar = 1.0 + 0.015 * np.sin(2.0 * np.pi * t / 0.04)
    jitter = 0.10 * ((np.arange(n_cad) % 7) / 6.0 - 0.5)

    flux = np.zeros((n_cad, ny, nx), dtype=np.float64)
    for r in range(ny):
        for c in range(nx):
            sens = (c - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
            flux[:, r, c] = 900.0 * stellar * (1.0 + sens * jitter) + rng.normal(0.0, 0.3, n_cad)

    # Set distinct pos_corr1 / pos_corr2 so we can verify centroid_col != pos_corr1
    pc1 = np.linspace(-0.25, 0.25, n_cad, dtype=np.float32)
    pc2 = np.linspace(0.15, -0.15, n_cad, dtype=np.float32)

    tpf_path = tmp_path / "rect_tpf.fits"
    tpf = _build_synthetic_halocver_tpf_file(
        tpf_path,
        flux=flux,
        flux_err=np.sqrt(flux),
        flux_bkg=np.full_like(flux, 25.0),
        pos_corr1=pc1,
        pos_corr2=pc2,
    )
    flux_orig = tpf.flux.value.copy()

    lc_single = extract_halo_lightcurve(tpf[:120], maxiter=151, compute_uncertainty=True)
    assert jax.config.jax_enable_x64 is False
    assert lc_single.meta["WEIGHTMAP"].shape == (ny, nx)
    assert lc_single.meta["WEIGHTMAP_ERR"].shape == (ny, nx)
    assert np.all(np.isfinite(lc_single.flux_err.value))
    np.testing.assert_allclose(lc_single["pos_corr1"].value, pc1[:120], rtol=1e-6)
    np.testing.assert_allclose(lc_single["pos_corr2"].value, pc2[:120], rtol=1e-6)
    # Centroids come from flux-weighted estimate_centroids (~col 1.5), NOT pos_corr1 (~0)!
    assert float(np.nanmedian(lc_single.centroid_col.value)) > 1.0
    assert not np.allclose(lc_single.centroid_col.value, pc1[:120])

    # Validate BOTH lower and upper split_times bounds
    t_vals = tpf.time.value
    with pytest.raises(ValueError, match="strictly inside"):
        extract_halo_lightcurve(tpf, split_times=[float(t_vals[0]), float(t_vals[65])])
    with pytest.raises(ValueError, match="strictly inside"):
        extract_halo_lightcurve(tpf, split_times=[float(t_vals[65]), float(t_vals[-1])])

    # Valid 2-split -> 3 segments of 66 cadences (N_pad=128) -> (3, ny, nx) WEIGHTMAP
    splits = [float(t_vals[65]), float(t_vals[131])]
    lc_split = extract_halo_lightcurve(
        tpf,
        split_times=splits,
        maxiter=151,
        compute_uncertainty=True,
    )
    np.testing.assert_array_equal(tpf.flux.value, flux_orig)
    assert len(lc_split) == n_cad
    assert lc_split.meta["WEIGHTMAP"].shape == (3, ny, nx)
    assert lc_split.meta["WEIGHTMAP_ERR"].shape == (3, ny, nx)
    np.testing.assert_allclose(np.sum(lc_split.meta["WEIGHTMAP"], axis=(1, 2)), 1.0, atol=1e-6)
    # Each stitched segment must preserve exact unit mean (NOT median re-normalization!)
    assert float(np.nanmean(lc_split.flux.value[:66])) == pytest.approx(1.0, abs=1e-11)
    assert float(np.nanmean(lc_split.flux.value[66:132])) == pytest.approx(1.0, abs=1e-11)
    assert float(np.nanmean(lc_split.flux.value[132:])) == pytest.approx(1.0, abs=1e-11)

    # Verify normalize=False with split_times preserves physical tpf.flux.unit (electron / s)
    # Use maxiter=151 so all 3 segments reuse the already-compiled (N_pad=128, M=8, maxiter=151) kernel!
    lc_split_unnorm = extract_halo_lightcurve(
        tpf,
        split_times=splits,
        maxiter=151,
        normalize=False,
    )
    assert lc_split_unnorm.flux.unit == u.electron / u.s
    assert lc_split_unnorm.flux_err.unit == u.electron / u.s
    assert float(np.nanmean(lc_split_unnorm.flux.value)) == pytest.approx(900.0, rel=0.05)

    # Verify skipped jackknife (< 2 folds when n_cad = 3 < 4) sets WEIGHTMAP_ERR to NaN (never 0.0!)
    # bucket_pad_length(3) == 128, so this ALSO reuses the already-compiled (N_pad=128, M=8, maxiter=151) kernel!
    lc_short = extract_halo_lightcurve(tpf[:3], maxiter=151, compute_uncertainty=True)
    assert np.all(np.isnan(lc_short.meta["WEIGHTMAP_ERR"]))

    # Round-trip 3D split WEIGHTMAP, WEIGHTMAP_ERR, and SPLIT_TIMES through write_lightcurve_fits
    split_fits = tmp_path / "split_halo_lc.fits"
    write_lightcurve_fits(lc_split, split_fits)
    with fits.open(split_fits) as hdul:
        assert hdul["WEIGHTMAP"].data.shape == (3, ny, nx)
        assert hdul["WEIGHTMAP_ERR"].data.shape == (3, ny, nx)
        assert "SPLIT_TIMES" in hdul
        assert hdul[0].header["NSPLITS"] == 2
        assert hdul["WEIGHTMAP"].header["NSPLITS"] == 2
        np.testing.assert_allclose(hdul["WEIGHTMAP"].data, lc_split.meta["WEIGHTMAP"])
    np.testing.assert_allclose(read_split_times(split_fits), splits, rtol=0.0, atol=1e-12)


def test_end_to_end_real_k2_tpf_and_epic_mast_fits(tmp_path: Path) -> None:
    """Test extract_halo_lightcurve + write_lightcurve_fits + CLI --hlsp-fits on real FITS data."""
    # 1. EPIC_211309989_mast.fits is a 1D K2SC light curve:
    #    - Passing it to extract_halo_lightcurve must raise a clear TypeError.
    #    - Loading its K2SC columns into a KeplerLightCurve and writing via write_lightcurve_fits
    #      must round-trip cleanly with HALOCVER = 1 and WEIGHTMAP ImageHDU.
    assert EPIC_FITS.exists()
    with pytest.raises((TypeError, lk.LightkurveError)):
        extract_halo_lightcurve(EPIC_FITS)

    with fits.open(EPIC_FITS) as epic_hdul:
        k2sc = epic_hdul["K2SC"].data[:120]
        fin = np.isfinite(k2sc["time"]) & np.isfinite(k2sc["flux"])
        epic_lc = lk.KeplerLightCurve(
            time=Time(np.asarray(k2sc["time"][fin], dtype=np.float64), format="bkjd", scale="tdb"),
            flux=np.asarray(k2sc["flux"][fin], dtype=np.float64),
            flux_err=np.asarray(k2sc["error"][fin], dtype=np.float64),
            targetid=211309989,
        )
    epic_wmap = np.full((4, 5), 1.0 / 20.0, dtype=np.float64)
    epic_lc.meta["WEIGHTMAP"] = epic_wmap
    epic_out = tmp_path / "epic_211309989_roundtrip.fits"
    halophot.write_fits(epic_lc, epic_out)
    with fits.open(epic_out) as hdul_e:
        assert hdul_e[0].header["HALOCVER"] == HALOCVER
        assert hdul_e["WEIGHTMAP"].data.shape == (4, 5)
        assert len(hdul_e[1].data) == len(epic_lc)

    # 2. End-to-end v2.0 extract_halo_lightcurve + CLI --hlsp-fits on real K2 TPF fixture
    assert K2_TPF_FITS.exists()
    mini_k2 = tmp_path / "mini_k2_v2.fits"
    with fits.open(K2_TPF_FITS) as hdul:
        hdul[1].data = hdul[1].data[:80].copy()
        hdul.writeto(mini_k2, overwrite=True)

    k2_tpf = lk.read(str(mini_k2))
    ny_k2, nx_k2 = k2_tpf.flux.shape[1:]
    lc_k2 = extract_halo_lightcurve(
        k2_tpf,
        rr=(1.0, 4.5),
        thresh=0.8,
        minflux=50.0,
        maxiter=151,
        sigclip=True,
    )
    assert isinstance(lc_k2, lk.KeplerLightCurve)
    assert lc_k2.meta["WEIGHTMAP"].shape == (ny_k2, nx_k2)
    assert np.sum(np.isfinite(lc_k2.flux.value)) >= 65
    assert np.sum(np.isfinite(lc_k2.flux_err.value)) >= 65

    k2_lc_fits = tmp_path / "k2_v2_lc.fits"
    write_lightcurve_fits(lc_k2, k2_lc_fits)
    with fits.open(k2_lc_fits) as hdul_k2:
        assert hdul_k2[0].header["HALOCVER"] == HALOCVER
        assert hdul_k2["WEIGHTMAP"].data.shape == (ny_k2, nx_k2)

    # Verify CLI --hlsp-fits honours --bitmask, --do-split / --split-cadences, and --do-plot
    cache_dir = tmp_path / "scratch_cache"
    out_cli = tmp_path / "hpc_cli_out"
    out_cli.mkdir()
    cli.main(
        [
            str(mini_k2),
            "--name",
            "k2_hlsp",
            "-c",
            "3",
            "--bitmask",
            "hard",
            "-maxiter",
            "151",
            "--rr",
            "[1.0,4.5]",
            "--thresh",
            "0.8",
            "--minflux",
            "50.0",
            "--do-split",
            "--split-cadences",
            "[40]",
            "--do-plot",
            "--hlsp-fits",
            "--offline",
            "--strict-hpc",
            "--cache-dir",
            str(cache_dir),
            "--quiet",
            "--save-dir",
            str(out_cli),
        ]
    )
    cli_fits = out_cli / "k2_hlsp_halo_lc_tv.fits"
    cli_png = out_cli / "k2_hlsp_all_tv.png"
    assert cli_fits.exists()
    assert cli_png.exists()
    with fits.open(cli_fits) as hdul_cli:
        assert hdul_cli[0].header["HALOCVER"] == HALOCVER
        assert hdul_cli["WEIGHTMAP"].data.shape == (2, ny_k2, nx_k2)
        assert "SPLIT_TIMES" in hdul_cli
        assert len(hdul_cli["SPLIT_TIMES"].data) == 1


def test_cli_hlsp_fits_faint_tess_wings_campaign_guard_and_sector_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify [HALO-01]: halo --hlsp-fits preserves <100 e-/s TESS wings, ignores K2 -c cuts on TESS, and labels S07."""
    rng = np.random.default_rng(70707)
    n_cad, ny, nx = 64, 5, 5
    t_rel = np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)
    stellar = 1.0 + 0.012 * np.sin(2.0 * np.pi * t_rel / 0.04)
    jitter = 0.06 * ((np.arange(n_cad) % 11) / 10.0 - 0.5)

    flux = np.zeros((n_cad, ny, nx), dtype=np.float64)
    for r in range(ny):
        for c in range(nx):
            sens = (c - 2.0) / 2.0
            wing_level = 45.0 + 8.0 * r + 3.0 * c  # 45..89 e-/s (< 100 e-/s legacy K2 cutoff!)
            flux[:, r, c] = wing_level * stellar * (1.0 + sens * jitter) + rng.normal(
                0.0, 0.25, size=n_cad
            )
    # Single saturated core pixel (> TESS_SAT_RATE_E_PER_S) so saturated-column masking triggers cleanly
    flux[:, 2, 2] = 2.0e5 * stellar
    flux_err = np.full_like(flux, 0.25)
    flux_bkg = np.full_like(flux, 30.0)
    pq = np.zeros((ny, nx), dtype=np.int32)
    pq[:, 2] = int(PixelQuality.SAT_BLEED)

    tpf_path = tmp_path / "tess_s07_faint_wings_tpf.fits"
    _build_synthetic_halocver_tpf_file(
        tpf_path,
        flux=flux,
        flux_err=flux_err,
        flux_bkg=flux_bkg,
        pixel_quality=pq,
        bkgsub=True,
    )
    # Set SECTOR = 7; note TIME starts at BTJD 2000.0 (< 2470.0), which K2 Campaign 7 cut would wipe out!
    with fits.open(tpf_path, mode="update") as hdul_upd:
        hdul_upd[0].header["SECTOR"] = 7
        hdul_upd.flush()

    tpf_lk = lk.read(str(tpf_path))
    lc_py = extract_halo_lightcurve(tpf_lk, maxiter=151, offline=True)

    recorded_titles: list[str] = []
    orig_plot_all = cli.plot_all

    def _spy_plot_all(*args: object, **kwargs: object) -> object:
        recorded_titles.append(str(kwargs.get("formal_name", "")))
        return orig_plot_all(*args, **kwargs)

    monkeypatch.setattr(cli, "plot_all", _spy_plot_all)

    cache_dir = tmp_path / "cli_s07_cache"
    out_dir = tmp_path / "cli_s07_out"
    out_dir.mkdir()
    cli.main(
        [
            str(tpf_path),
            "--hlsp-fits",
            "-c",
            "7",
            "-maxiter",
            "151",
            "--do-plot",
            "--name",
            "alf_CMa",
            "--offline",
            "--strict-hpc",
            "--cache-dir",
            str(cache_dir),
            "--quiet",
            "--save-dir",
            str(out_dir),
        ]
    )

    cli_fits = out_dir / "alf_CMa_halo_lc_tv.fits"
    assert cli_fits.exists()
    lc_cli = lk.read(str(cli_fits))
    assert len(lc_cli) == n_cad == len(lc_py)
    np.testing.assert_allclose(lc_cli.flux.value, lc_py.flux.value, rtol=1e-5, atol=1e-6)

    wmap_cli = halophot.read_weightmap(cli_fits)
    wmap_py = halophot.read_weightmap(lc_py)
    assert wmap_cli is not None and wmap_py is not None
    assert int(np.sum(wmap_cli > 0.0)) >= 12  # Faint <100 e-/s halo wings preserved!
    assert int(np.sum(wmap_cli > 0.0)) == int(np.sum(wmap_py > 0.0))
    np.testing.assert_allclose(wmap_cli, wmap_py, rtol=1e-5, atol=1e-6)
    assert len(recorded_titles) == 1
    assert recorded_titles[0].endswith(" - S07")


def test_mask_background_stars_cache_dir_forwarding_and_fred_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /fred symlink guard, Catalogs.cache_location forwarding, and clean None restoration."""
    import astroquery.mast as aq_mast

    for env_key in ("HALOPHOT_CACHE_DIR", "TESS_BRIGHT_CACHE_DIR", "SCRATCH", "XDG_CACHE_HOME"):
        monkeypatch.delenv(env_key, raising=False)

    n_cad, ny, nx = 20, 6, 6
    ra0, dec0 = 101.287, -16.716
    wcs_2d = _make_celestial_wcs(ny, nx, ra0=ra0, dec0=dec0)
    contam_sky = wcs_2d.pixel_to_world(4.0, 1.0)
    cat = Table({"ra": [contam_sky.ra.deg], "dec": [contam_sky.dec.deg], "Tmag": [10.5]})

    fred_link = tmp_path / "fred_cache_symlink"
    fred_link.symlink_to("/fred/oz440/bpo/.cache/tess_bright")

    # 1. /fred symlink rejected BEFORE any live MAST query (and when catalog_table is supplied)
    def _must_not_query(*args: object, **kwargs: object) -> Table:
        raise AssertionError("Catalogs.query_region must never execute when /fred guard rejects cache_dir!")

    stub_guard = types.SimpleNamespace(cache_location=None, query_region=_must_not_query)
    monkeypatch.setattr(aq_mast, "Catalogs", stub_guard)

    with pytest.raises(ValueError, match="/fred/"):
        mask_background_stars(wcs_2d, (ny, nx), catalog_table=None, offline=False, cache_dir=fred_link)
    with pytest.raises(ValueError, match="/fred/"):
        mask_background_stars(wcs_2d, (ny, nx), catalog_table=cat, offline=True, cache_dir=fred_link)

    flux = np.full((n_cad, ny, nx), 600.0, dtype=np.float64)
    tpf = _build_synthetic_halocver_tpf_file(
        tmp_path / "fred_guard_tpf.fits",
        flux=flux,
        flux_err=np.sqrt(flux),
        flux_bkg=np.full_like(flux, 20.0),
        ra0=ra0,
        dec0=dec0,
    )
    with pytest.raises(ValueError, match="/fred/"):
        remove_stars(tpf, catalog_table=cat, offline=True, cache_dir=fred_link)
    with pytest.raises(ValueError, match="/fred/"):
        build_halo_masks(flux, flux_err=np.sqrt(flux), cache_dir=fred_link)
    with pytest.raises(ValueError, match="/fred/"):
        extract_halo_lightcurve(tpf, cache_dir=fred_link, maxiter=15)

    # strict_hpc=False downgrades /fred rejection to UserWarning
    with pytest.warns(UserWarning, match="/fred/"):
        mask_relaxed = mask_background_stars(
            wcs_2d, (ny, nx), catalog_table=cat, offline=True, cache_dir=fred_link, strict_hpc=False
        )
    assert bool(mask_relaxed[1, 4]) is True

    # 2. Real CatalogsClass instance (with _cache_location initially None): forwards cache_dir
    #    and restores _cache_location = None cleanly on both success and exception!
    real_cat = aq_mast.CatalogsClass()
    assert real_cat._cache_location is None
    observed_caches: list[Path] = []

    def _fake_real_query(center: object, *, radius: object, catalog: str = "TIC") -> Table:
        observed_caches.append(Path(real_cat.cache_location))
        return cat

    monkeypatch.setattr(real_cat, "query_region", _fake_real_query)
    monkeypatch.setattr(aq_mast, "Catalogs", real_cat)

    custom_cache = tmp_path / "aphid_scratch" / "mast_cache"
    live_mask = mask_background_stars(
        wcs_2d,
        (ny, nx),
        catalog_table=None,
        offline=False,
        cache_dir=custom_cache,
        radius_pix=1.2,
    )
    assert custom_cache.is_dir()
    assert observed_caches == [custom_cache.resolve()]
    assert real_cat._cache_location is None
    assert bool(live_mask[1, 4]) is True

    def _boom_query(*args: object, **kwargs: object) -> Table:
        assert Path(real_cat.cache_location) == custom_cache.resolve()
        raise RuntimeError("simulated MAST outage")

    monkeypatch.setattr(real_cat, "query_region", _boom_query)
    with pytest.raises(RuntimeError, match="simulated MAST outage"):
        mask_background_stars(wcs_2d, (ny, nx), catalog_table=None, offline=False, cache_dir=custom_cache)
    assert real_cat._cache_location is None

    # 3. Plain stub Catalogs without _cache_location whose cache_location starts as None:
    #    must ALSO restore cache_location to None!
    seen_stub_locs: list[object] = []

    def _stub_query(*args: object, **kwargs: object) -> Table:
        seen_stub_locs.append(stub_none.cache_location)
        return cat

    stub_none = types.SimpleNamespace(cache_location=None, query_region=_stub_query)
    monkeypatch.setattr(aq_mast, "Catalogs", stub_none)
    mask_stub = mask_background_stars(
        wcs_2d, (ny, nx), catalog_table=None, offline=False, cache_dir=custom_cache, radius_pix=1.2
    )
    assert seen_stub_locs == [str(custom_cache.resolve())]
    assert stub_none.cache_location is None
    assert bool(mask_stub[1, 4]) is True


def test_cli_strict_hpc_boolean_optional_and_k2_campaign_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI --strict-hpc (default True) / --no-strict-hpc and implicit K2 campaign cuts."""
    parser = cli.build_parser()
    assert parser.parse_args(["foo.fits"]).strict_hpc is True
    assert parser.parse_args(["foo.fits", "--strict-hpc"]).strict_hpc is True
    assert parser.parse_args(["foo.fits", "--no-strict-hpc"]).strict_hpc is False

    # Build a K2 Campaign 12 TPF from the real K2 fixture spanning the (2937.75, 2938.8) safe-mode window
    assert K2_TPF_FITS.exists()
    n_cad = 40
    bkjd = np.linspace(2937.0, 2940.0, n_cad, dtype=np.float64)
    in_window = (bkjd > 2937.75) & (bkjd < 2938.8)
    assert 4 <= int(np.sum(in_window)) <= 20

    k2_c12_path = tmp_path / "ktwo205897543-c12_lpd-targ.fits"
    with fits.open(K2_TPF_FITS) as hdul_src:
        hdul_c12 = fits.HDUList([h.copy() for h in hdul_src])
        sub_tab = hdul_c12[1].data[:n_cad].copy()
        sub_tab["TIME"][:] = bkjd
        sub_tab["QUALITY"][:] = 0
        # Inject a step inside the C12 safe-mode window that must be excised automatically!
        sub_tab["FLUX"][in_window, :, :] *= 3.0
        hdul_c12[1].data = sub_tab
        hdul_c12[0].header["CAMPAIGN"] = 12
        hdul_c12.writeto(k2_c12_path, overwrite=True)

    out_dir = tmp_path / "k2_c12_out"
    out_dir.mkdir()

    # Default strict_hpc=True rejects /fred cache dir even without --strict-hpc on CLI!
    with pytest.raises(ValueError, match="/fred/"):
        cli.main(
            [
                str(k2_c12_path),
                "--hlsp-fits",
                "--cache-dir",
                "/fred/oz440/bpo/.cache",
                "--save-dir",
                str(out_dir),
            ]
        )

    recorded_titles: list[str] = []
    orig_plot_all = cli.plot_all

    def _spy_plot_all(*args: object, **kwargs: object) -> object:
        recorded_titles.append(str(kwargs.get("formal_name", "")))
        return orig_plot_all(*args, **kwargs)

    monkeypatch.setattr(cli, "plot_all", _spy_plot_all)

    # --no-strict-hpc downgrades /fred rejection to UserWarning, and omitted -c infers C12 from TPF!
    with pytest.warns(UserWarning, match="/fred/"):
        cli.main(
            [
                str(k2_c12_path),
                "--hlsp-fits",
                "--no-strict-hpc",
                "--cache-dir",
                "/fred/oz440/bpo/.cache",
                "--rr",
                "[1.0,4.5]",
                "--thresh",
                "0.8",
                "--minflux",
                "50.0",
                "-maxiter",
                "25",
                "--do-plot",
                "--name",
                "k2_c12",
                "--quiet",
                "--save-dir",
                str(out_dir),
            ]
        )

    c12_fits = out_dir / "k2_c12_halo_lc_tv.fits"
    assert c12_fits.is_file()
    lc_c12 = lk.read(str(c12_fits))
    assert len(lc_c12) == n_cad - int(np.sum(in_window))
    assert not np.any((lc_c12.time.value > 2937.75) & (lc_c12.time.value < 2938.8))
    assert recorded_titles == ["k2 c12 - C12"]

    # Kepler prime-mission Quarter 7 TPF (BKJD ~ 700 < 1600) passed with `-c 7` must NOT trigger
    # K2 Campaign 7 `t > 2470` excision, and `--do-plot` must label `- Q07` in both modes!
    kep_q7_path = tmp_path / "kplr001234567-2010355_lpd-targ.fits"
    with fits.open(K2_TPF_FITS) as hdul_src:
        hdul_q7 = fits.HDUList([h.copy() for h in hdul_src])
        q7_tab = hdul_q7[1].data[:36].copy()
        q7_tab["TIME"][:] = np.linspace(700.0, 701.0, 36, dtype=np.float64)
        q7_tab["QUALITY"][:] = 0
        hdul_q7[1].data = q7_tab
        hdul_q7[0].header["MISSION"] = "Kepler"
        hdul_q7[0].header["QUARTER"] = 7
        if "CAMPAIGN" in hdul_q7[0].header:
            del hdul_q7[0].header["CAMPAIGN"]
        hdul_q7.writeto(kep_q7_path, overwrite=True)

    recorded_titles.clear()
    cli.main(
        [
            str(kep_q7_path),
            "--hlsp-fits",
            "-c",
            "7",
            "--rr",
            "[1.0,4.5]",
            "--thresh",
            "0.8",
            "--minflux",
            "50.0",
            "-maxiter",
            "15",
            "--do-plot",
            "--name",
            "kep_q7_hlsp",
            "--quiet",
            "--save-dir",
            str(out_dir),
        ]
    )
    lc_q7 = lk.read(str(out_dir / "kep_q7_hlsp_halo_lc_tv.fits"))
    assert len(lc_q7) == 36
    assert recorded_titles == ["kep q7 hlsp - Q07"]

    recorded_titles.clear()
    cli.main(
        [
            str(kep_q7_path),
            "-c",
            "7",
            "--rr",
            "[1.0,4.5]",
            "--thresh",
            "0.8",
            "--minflux",
            "50.0",
            "-maxiter",
            "10",
            "--do-plot",
            "--name",
            "kep_q7_legacy",
            "--quiet",
            "--save-dir",
            str(out_dir),
        ]
    )
    assert recorded_titles == ["kep q7 legacy - Q07"]


def _equilibrated_2d_chebyshev(
    ny: int, nx: int, bkg_order: int, bg_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Construct QR-equilibrated 2D Chebyshev basis ``(phi_full, phi_bg)`` on ``bg_mask``."""
    xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    eta = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xi, eta)
    terms: list[np.ndarray] = []
    for p in range(bkg_order + 1):
        for q in range(bkg_order + 1 - p):
            terms.append(
                np.polynomial.chebyshev.chebval(xx, [0.0] * p + [1.0])
                * np.polynomial.chebyshev.chebval(yy, [0.0] * q + [1.0])
            )
    phi_raw = np.stack(terms, axis=0)
    _, r_bg = np.linalg.qr(phi_raw[:, bg_mask].T, mode="reduced")
    phi_full = np.einsum("ij,jrc->irc", np.linalg.inv(r_bg), phi_raw)
    phi_bg = phi_full[:, bg_mask].T
    return phi_full, phi_bg


@pytest.mark.parametrize("case", ["core_masked", "full_col_wls", "asymmetric_one_sided"])
def test_oblique_wls_smear_projection_matches_monte_carlo(case: str) -> None:
    """Verify [HALO-IO-01]: exact WLS oblique column-smear sensitivity matches MC across masks."""
    rng = np.random.default_rng(20261120)
    n_mc = 6000
    bkg_order = 2

    if case == "core_masked":
        ny, nx = 9, 13
        bg_mask = np.zeros((ny, nx), dtype=bool)
        bg_mask[:, :4] = True
        bg_mask[:, 9:] = True
        halo_mask = np.zeros((ny, nx), dtype=bool)
        halo_mask[[1, 2, 6, 7], 4:9] = True
        sigma_pix = np.full((ny, nx), 2.0, dtype=np.float64)
        sigma_pix[halo_mask] = 8.0
        sigma_smear = np.full(nx, 1.5, dtype=np.float64)
        sigma_smear[4:9] = 25.0
    elif case == "full_col_wls":
        ny, nx = 7, 7
        bg_mask = np.zeros((ny, nx), dtype=bool)
        bg_mask[[0, 3, 6], :] = True
        halo_mask = np.zeros((ny, nx), dtype=bool)
        halo_mask[:, 2:5] = True
        halo_mask &= ~bg_mask
        sigma_pix = np.full((ny, nx), 2.0, dtype=np.float64)
        sigma_smear = np.array([1.5, 1.5, 30.0, 40.0, 30.0, 1.5, 1.5], dtype=np.float64)
    else:
        ny, nx = 8, 10
        bg_mask = np.zeros((ny, nx), dtype=bool)
        bg_mask[:, :5] = True
        halo_mask = np.zeros((ny, nx), dtype=bool)
        halo_mask[2:6, 5:10] = True
        sigma_pix = np.full((ny, nx), 2.5, dtype=np.float64)
        sigma_smear = np.array(
            [1.5, 1.5, 1.5, 2.0, 2.0, 18.0, 22.0, 25.0, 20.0, 15.0], dtype=np.float64
        )

    phi_full, phi_bg = _equilibrated_2d_chebyshev(ny, nx, bkg_order, bg_mask)
    w_2d = np.zeros((ny, nx), dtype=np.float64)
    w_2d[halo_mask] = 1.0 / np.sum(halo_mask)
    norm_factor = 1000.0

    bg_rows, bg_cols = np.where(bg_mask)
    w_wls = 1.0 / (sigma_pix[bg_rows, bg_cols] ** 2 + sigma_smear[bg_cols] ** 2)
    a_wls = np.linalg.solve((phi_bg.T * w_wls) @ phi_bg, phi_bg.T * w_wls)
    cov_beta_indep = (a_wls * (sigma_pix[bg_rows, bg_cols] ** 2)) @ a_wls.T
    chol_beta_indep = np.linalg.cholesky(cov_beta_indep)

    eps_pix = rng.normal(0.0, 1.0, size=(n_mc, ny, nx)) * sigma_pix[None, :, :]
    eps_smear = rng.normal(0.0, 1.0, size=(n_mc, nx)) * sigma_smear[None, :]
    u_mc = 1000.0 + eps_pix + eps_smear[:, None, :]
    beta_mc = u_mc[:, bg_rows, bg_cols] @ a_wls.T
    p_sub_mc = u_mc - np.einsum("tp,prc->trc", beta_mc, phi_full)
    emp_std = float(np.std(np.einsum("trc,rc->t", p_sub_mc, w_2d) / norm_factor, ddof=1))

    be_indep_sq = np.sum(np.square(np.einsum("qp,qrc->prc", chol_beta_indep, phi_full)), axis=0)
    fe_sub = np.sqrt(sigma_pix**2 + sigma_smear[None, :] ** 2 + be_indep_sq)

    # Cadence 0: full rank; Cadence 1: rank-deficient (all bg pixels NaN -> no crash, finite fallback)
    fe_2cad = np.stack([fe_sub, fe_sub.copy()], axis=0)
    fe_2cad[1, bg_mask] = np.nan
    se_2cad = np.repeat(sigma_smear[None, :], 2, axis=0)
    chol_2cad = np.repeat(chol_beta_indep[None, :, :], 2, axis=0)

    got = compute_coherent_flux_err(
        w_2d,
        flux_err=fe_2cad,
        smear_err=se_2cad,
        bkg_cov=BkgCovariance(chol=chol_2cad, basis=phi_full),
        bg_mask=bg_mask,
        bkgsub=True,
        bkg_order=bkg_order,
        norm_factor=norm_factor,
    )
    assert got[0] == pytest.approx(emp_std, rel=0.03)
    assert np.isfinite(got[1]) and got[1] > 0.0


def test_fp32_cancellation_floor_and_algol_eclipse_sigclip_preservation(tmp_path: Path) -> None:
    """Verify [HALO-IO-09] FP32 ULP floor and [MATH-02] Algol eclipse preservation under sigclip=True."""
    # 1. FP32 cancellation in FLUX_ERR^2 - SMEAR_ERR^2 must never collapse photon noise to 0.0
    sigma_indep = np.array([[[2.0, 2.0]]], dtype=np.float64)
    sm_f64 = np.array([[6000.1, 6000.1]], dtype=np.float64)
    fe_f32 = np.sqrt(sigma_indep**2 + sm_f64[:, None, :] ** 2).astype(np.float32)
    err_f32 = compute_coherent_flux_err(
        np.array([[0.5, 0.5]]),
        flux_err=fe_f32,
        smear_err=sm_f64.astype(np.float32),
        bkg_cov=BkgCovariance(chol=np.zeros((1, 1, 1)), basis=np.ones((1, 1, 2))),
        bkgsub=True,
        bkg_order=0,
        norm_factor=1.0,
    )[0]
    assert np.isfinite(err_f32) and err_f32 > 0.5

    # 2. Algol-like 30% deep primary eclipse + isolated spike: sigclip=True excludes spikes only
    #    during weight optimization and NEVER overwrites finite eclipse cadences with NaN!
    rng = np.random.default_rng(4242)
    n_cad, ny, nx = 240, 4, 4
    # Deep 30% eclipse spanning 18 cadences (~0.6 h) + single cosmic-ray spike
    in_eclipse = (np.arange(n_cad) >= 100) & (np.arange(n_cad) < 118)
    stellar = np.ones(n_cad, dtype=np.float64)
    stellar[in_eclipse] = 0.70
    jitter = 0.04 * ((np.arange(n_cad) % 7) / 6.0 - 0.5)

    flux = np.zeros((n_cad, ny, nx), dtype=np.float64)
    for r in range(ny):
        for c in range(nx):
            sens = -1.0 if (c % 2 == 0) else 1.0
            flux[:, r, c] = 1000.0 * stellar * (1.0 + sens * jitter) + rng.normal(0.0, 0.5, n_cad)
    flux[40, :, :] *= 1.4  # Cosmic-ray spike

    tpf_algol = _build_synthetic_halocver_tpf_file(
        tmp_path / "algol_tpf.fits",
        flux=flux,
        flux_err=np.sqrt(flux),
        flux_bkg=np.full_like(flux, 15.0),
        bkgsub=True,
    )
    lc_algol = extract_halo_lightcurve(tpf_algol, sigclip=True, maxiter=151)
    assert np.all(np.isfinite(lc_algol.flux.value))
    assert np.all(np.isfinite(lc_algol.flux_err.value))
    assert np.all(np.isfinite(lc_algol.flux.value[in_eclipse]))
    assert float(np.mean(lc_algol.flux.value[in_eclipse])) == pytest.approx(0.70, abs=0.02)


def test_wcs_epoch_inference_visual_binaries_and_missing_primary(tmp_path: Path) -> None:
    """Verify [HALO-IO-03], [HALO-IO-04], [HALO-IO-14] on Gaia+Tmag PM, visual binaries, and missing G<3."""
    from halophot.masking import _propagate_catalogue_coords

    ra_acen, dec_acen = 219.8738, -60.8322
    obs_2023 = Time(2023.3, format="jyear", scale="tdb")
    sc_true_2016 = SkyCoord(
        ra=ra_acen * u.deg,
        dec=dec_acen * u.deg,
        pm_ra_cosdec=-3679.0 * (u.mas / u.yr),
        pm_dec=474.0 * (u.mas / u.yr),
        obstime=Time(2016.0, format="jyear", scale="tdb"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_true = sc_true_2016.apply_space_motion(new_obstime=obs_2023)

    # Un-hinted Gaia+Tmag and VizieR Gmag tables without ref_epoch infer J2016.0 with 0 ErfaWarnings
    for tbl in (
        Table(
            {
                "ra": [ra_acen],
                "dec": [dec_acen],
                "phot_g_mean_mag": [8.5],
                "Tmag": [7.9],
                "pmRA": [-3679.0],
                "pmDEC": [474.0],
            }
        ),
        Table(
            {
                "ra": [ra_acen],
                "dec": [dec_acen],
                "Gmag": [8.5],
                "pmRA": [-3679.0],
                "pmDEC": [474.0],
            }
        ),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, c_obs = _propagate_catalogue_coords(
                np.array([ra_acen]),
                np.array([dec_acen]),
                tbl,
                catalog="TIC",
                obs_time=obs_2023,
            )
        assert float(c_obs[0].separation(c_true).arcsec) < 1e-3
        assert len(caught) == 0

    # TIC table with ID + Tmag + posflag (no Gaia source_id) infers J2000.0
    tbl_tic = Table(
        {
            "ID": [123456],
            "ra": [ra_acen],
            "dec": [dec_acen],
            "Tmag": [7.9],
            "posflag": ["tmgaia2"],
            "pmRA": [-3679.0],
            "pmDEC": [474.0],
        }
    )
    _, _, c_tic_obs = _propagate_catalogue_coords(
        np.array([ra_acen]),
        np.array([dec_acen]),
        tbl_tic,
        catalog="TIC",
        obs_time=obs_2023,
    )
    assert float(c_tic_obs[0].separation(c_true).arcsec) > 50.0  # J2000 -> 2023.3 (+16 yr PM diff)

    # Visual binaries (alpha Cen A/B at 8" and Castor A/B at 3.5") must NOT self-mask target core
    ny, nx = 15, 15
    wcs_bin = _make_celestial_wcs(ny, nx, ra0=219.9021, dec0=-60.8340)
    star_a = wcs_bin.pixel_to_world(7.0, 7.0)
    for sep_arcsec, mags in ((8.0, [-0.7, 0.6]), (3.5, [1.9, 2.0])):
        star_b = wcs_bin.pixel_to_world(7.0 + sep_arcsec / 21.0, 7.0)
        cat_bin = Table(
            {
                "ra": [star_a.ra.deg, star_b.ra.deg],
                "dec": [star_a.dec.deg, star_b.dec.deg],
                "Tmag": mags,
            }
        )
        m_bin = mask_background_stars(
            wcs_bin,
            (ny, nx),
            catalog_table=cat_bin,
            offline=True,
            target_coord=(star_a.ra.deg, star_a.dec.deg),
        )
        assert not np.any(m_bin)

    # Missing G < 3 naked-eye primary in Gaia must not mistake a G=13.5 neighbour for the primary
    # when target_mag is inferred from TPF header (TESSMAG) or integrated stamp flux!
    comp_22 = wcs_bin.pixel_to_world(7.0 + 22.0 / 21.0, 7.0)
    field_1 = wcs_bin.pixel_to_world(2.0, 2.0)
    field_2 = wcs_bin.pixel_to_world(12.0, 11.0)
    gaia_no_prim = Table(
        {
            "ra": [comp_22.ra.deg, field_1.ra.deg, field_2.ra.deg],
            "dec": [comp_22.dec.deg, field_1.dec.deg, field_2.dec.deg],
            "phot_g_mean_mag": [13.5, 12.5, 13.8],
        }
    )
    # Integrated flux ~ 2.7e7 e-/s -> estimated Tmag ~ 1.86 -> eff_mag_limit ~ 11.86 (< 12.5, 13.5, 13.8)
    flux_sir = np.full((24, ny, nx), 1.2e5, dtype=np.float64)
    sir_path = tmp_path / "sirius_missing_gaia.fits"
    tpf_sir = _build_synthetic_halocver_tpf_file(
        sir_path,
        flux=flux_sir,
        flux_err=np.sqrt(flux_sir),
        flux_bkg=np.full_like(flux_sir, 20.0),
        ra0=star_a.ra.deg,
        dec0=star_a.dec.deg,
    )
    tpf_cleaned = remove_stars(tpf_sir, catalog_table=gaia_no_prim, offline=True)
    # Neither faint field star (G=12.5/13.8) should be falsely masked against bright primary!
    assert np.all(np.isfinite(tpf_cleaned.flux.value[:, 2, 2]))
    assert np.all(np.isfinite(tpf_cleaned.flux.value[:, 11, 12]))
    # Also verify explicit TESSMAG=-1.4 header card fallback even on fainter wing-only stamps
    with fits.open(sir_path, mode="update") as h_sir:
        h_sir[0].header["TESSMAG"] = -1.4
        h_sir[1].data["FLUX"][:] = 500.0
        h_sir.flush()
    tpf_sir_hdr = lk.read(str(sir_path))
    tpf_hdr_cleaned = remove_stars(tpf_sir_hdr, catalog_table=gaia_no_prim, offline=True)
    assert np.all(np.isfinite(tpf_hdr_cleaned.flux.value[:, 2, 2]))
    assert np.all(np.isfinite(tpf_hdr_cleaned.flux.value[:, 11, 12]))


def test_bkgsub_false_earthshine_saturation_units_and_split_centroids(tmp_path: Path) -> None:
    """Verify [HALO-IO-05], [HALO-IO-06], [HALO-IO-07], [HALO-IO-08], [HALO-IO-15]."""
    # 1. BKGSUB=False sky-dominance guard prevents sky dilution
    t_arr = np.arange(96, dtype=np.float64) * (2.0 / 1440.0)
    stellar = 1.0 + 0.04 * np.sin(2.0 * np.pi * t_arr / 0.04)
    jitter = 0.08 * ((np.arange(96) % 9) / 8.0 - 0.5)
    f_unsub = np.zeros((96, 4, 4), dtype=np.float64)
    f_bkg = np.full((96, 4, 4), 200.0, dtype=np.float64)
    for r in range(2):
        for c in range(4):
            sens = -1.0 if (c % 2 == 0) else 1.0
            f_unsub[:, r, c] = 1200.0 * stellar * (1.0 + sens * jitter) + 200.0
    for r in range(2, 4):
        for c in range(4):
            f_unsub[:, r, c] = 5.0 * stellar + 200.0

    tpf_unsub = _build_synthetic_halocver_tpf_file(
        tmp_path / "unsub_sky_guard.fits",
        flux=f_unsub,
        flux_err=np.sqrt(f_unsub),
        flux_bkg=f_bkg,
        bkgsub=False,
    )
    lc_unsub = extract_halo_lightcurve(tpf_unsub, maxiter=151)
    assert float(np.sum(lc_unsub.meta["WEIGHTMAP"][2:, :])) == 0.0
    assert float(np.ptp(lc_unsub.flux.value)) > 0.065

    # 2. BKGSUB=True clean SNR=148 halo wings under 300 e-/s subtracted Earthshine survive
    flux_earth = np.full((30, 7, 7), 400.0, dtype=np.float64)
    flux_earth[:, 3, 3] = 1.0e5
    bkg_earth = np.full((30, 7, 7), 300.0, dtype=np.float64)
    err_earth = np.sqrt(flux_earth + bkg_earth) / np.sqrt(95.0)
    with pytest.warns(UserWarning, match="sky_kappa"):
        res_earth = build_halo_masks(
            flux_earth, flux_err=err_earth, flux_bkg=bkg_earth, thresh=-1.0, mission="TESS"
        )
    assert int(np.sum(res_earth.pixel_mask)) == 40

    # 3. Partial-cadence saturation (45% at 95k), explicit auto_sat_level with SAT_BLEED, and min_snr
    flux_psat = np.full((40, 7, 7), 1000.0, dtype=np.float64)
    flux_psat[:22, 3, 3] = 65000.0
    flux_psat[22:, 3, 3] = 95000.0
    res_psat = build_halo_masks(flux_psat, flux_err=np.sqrt(flux_psat), thresh=-1.0, mission="TESS")
    assert bool(res_psat.saturated_mask[3, 3]) is True

    pq_one = np.zeros((7, 7), dtype=np.uint16)
    pq_one[3, 3] = int(PixelQuality.SAT_BLEED)
    flux_as = np.full((40, 7, 7), 1000.0, dtype=np.float64)
    flux_as[:, 3, 3] = 1.0e5
    flux_as[:, 2, 3] = 55000.0
    res_as = build_halo_masks(
        flux_as,
        flux_err=np.sqrt(flux_as),
        pixel_quality=pq_one,
        thresh=-1.0,
        auto_sat_level=40000.0,
        mission="TESS",
    )
    assert bool(res_as.saturated_mask[2, 3]) is True

    with pytest.raises(ValueError, match="SNR filtering"):
        build_halo_masks(np.full((20, 4, 4), 1e-4, dtype=np.float64), min_snr=5.0)

    # 4. Integrated 'electron' TPF converts to e-/s without false saturation
    tpf_cnt_path = tmp_path / "counts_electron_tpf.fits"
    _build_synthetic_halocver_tpf_file(
        tpf_cnt_path,
        flux=np.full((24, 4, 4), 800.0 * 120.0, dtype=np.float64),
        flux_err=np.full((24, 4, 4), np.sqrt(800.0 * 120.0), dtype=np.float64),
        flux_bkg=np.zeros((24, 4, 4), dtype=np.float64),
        bkgsub=False,
    )
    with fits.open(tpf_cnt_path, mode="update") as h_cnt:
        for idx_c, col in enumerate(h_cnt[1].columns, start=1):
            if col.name.upper() in ("FLUX", "FLUX_ERR", "FLUX_BKG", "FLUX_BKG_ERR"):
                h_cnt[1].header[f"TUNIT{idx_c}"] = "electron"
        h_cnt.flush()
    tpf_cnt = lk.read(str(tpf_cnt_path))
    lc_cnt = extract_halo_lightcurve(tpf_cnt, normalize=False)
    assert lc_cnt.flux.unit == u.electron / u.s
    assert float(np.nanmean(lc_cnt.flux.value)) == pytest.approx(800.0, rel=0.05)

    # 5. normalize=False + split_times continuity on constant star with 5x wing flux contrast
    n_c = 120
    jit = 0.08 * ((-1.0) ** np.arange(n_c))
    flux_sp = np.zeros((n_c, 4, 4), dtype=np.float64)
    flux_sp[:60, :2, :] = 2000.0
    flux_sp[60:, :2, 0::2] = (2000.0 * (1.0 + jit[60:]))[:, None, None]
    flux_sp[60:, :2, 1::2] = (2000.0 * (1.0 + 0.8 * jit[60:]))[:, None, None]
    flux_sp[:60, 2:, 0::2] = (400.0 * (1.0 + jit[:60]))[:, None, None]
    flux_sp[:60, 2:, 1::2] = (400.0 * (1.0 + 0.8 * jit[:60]))[:, None, None]
    flux_sp[60:, 2:, :] = 400.0
    tpf_sp = _build_synthetic_halocver_tpf_file(
        tmp_path / "split_unnorm_cont.fits",
        flux=flux_sp,
        flux_err=np.sqrt(flux_sp),
        flux_bkg=np.full_like(flux_sp, 20.0),
        bkgsub=True,
    )
    t_mid = 0.5 * float(tpf_sp.time.value[59] + tpf_sp.time.value[60])
    lc_sp_unnorm = extract_halo_lightcurve(tpf_sp, split_times=[t_mid], normalize=False)
    ratio = float(np.mean(lc_sp_unnorm.flux.value[:60]) / np.mean(lc_sp_unnorm.flux.value[60:]))
    assert ratio == pytest.approx(1.0, rel=0.02)

    # 6. Negative background-subtracted halo pixels must not explode centroids
    flux_neg = np.full((30, 4, 4), 10.0, dtype=np.float64)
    flux_neg[:, 0, :] = -9.95
    flux_neg[5, 0, :] = -30.001
    tpf_neg = _build_synthetic_halocver_tpf_file(
        tmp_path / "neg_centroids.fits",
        flux=flux_neg,
        flux_err=np.full_like(flux_neg, 1.0),
        flux_bkg=np.full_like(flux_neg, 10.0),
        bkgsub=True,
    )
    lc_neg = extract_halo_lightcurve(tpf_neg, legacy_tv=True)
    assert 0.0 <= float(lc_neg.centroid_row.value[5]) <= 3.0
    assert 0.0 <= float(lc_neg.centroid_col.value[5]) <= 3.0

