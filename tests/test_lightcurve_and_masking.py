"""Tests for ``halophot.masking``, coherent ``flux_err`` Monte Carlo, and ``extract_halo_lightcurve``."""

from __future__ import annotations

from pathlib import Path

import jax
import lightkurve as lk
import numpy as np
import pytest
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
    write_lightcurve_fits,
)
from halophot.lightcurve import (
    column_orthogonal_smear_weights,
    compute_coherent_flux_err,
    extract_halo_lightcurve,
)
from halophot.masking import build_halo_masks, mask_background_stars, remove_stars

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

    # Round-trip 3D split WEIGHTMAP and WEIGHTMAP_ERR through write_lightcurve_fits
    split_fits = tmp_path / "split_halo_lc.fits"
    write_lightcurve_fits(lc_split, split_fits)
    with fits.open(split_fits) as hdul:
        assert hdul["WEIGHTMAP"].data.shape == (3, ny, nx)
        assert hdul["WEIGHTMAP_ERR"].data.shape == (3, ny, nx)
        np.testing.assert_allclose(hdul["WEIGHTMAP"].data, lc_split.meta["WEIGHTMAP"])


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

    # Verify CLI --hlsp-fits --offline --strict-hpc path (matching hpc/halophot_survey_cpu.sbatch)
    cache_dir = tmp_path / "scratch_cache"
    out_cli = tmp_path / "hpc_cli_out"
    out_cli.mkdir()
    cli.main(
        [
            str(mini_k2),
            "--name",
            "k2_hlsp",
            "-maxiter",
            "151",
            "--rr",
            "[1.0,4.5]",
            "--thresh",
            "0.8",
            "--minflux",
            "50.0",
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
    assert cli_fits.exists()
    with fits.open(cli_fits) as hdul_cli:
        assert hdul_cli[0].header["HALOCVER"] == HALOCVER
        assert "WEIGHTMAP" in hdul_cli
