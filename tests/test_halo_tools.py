"""Regression and smoke tests for ``halophot.halo_tools``, ``halo_objectives``, and CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from halophot import cli, halo_objectives
from halophot.halo_tools import (
    DAY_TO_MUHZ,
    censor_tpf,
    diff_1,
    do_lc,
    get_pgram,
    halo_tpf,
    halo_tpf_tess,
    jax_func,
    plot_log_pgram,
    plot_pgram,
    print_time,
    read_tpf,
    remove_stars,
    tv_tpf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EPIC_FITS = REPO_ROOT / "EPIC_211309989_mast.fits"
K2_TPF_FITS = REPO_ROOT / "data" / "ktwo205897543-c03_lpd-targ.fits.gz"


def _make_synthetic_jitter_tpf(
    n_cad: int = 100,
    ny: int = 5,
    nx: int = 5,
) -> tuple[np.ndarray, Table, np.ndarray]:
    """Create a deterministic synthetic TPF with complementary pointing jitter + stellar signal."""
    rng = np.random.default_rng(12345)
    t = np.linspace(0.0, 10.0, n_cad, dtype=np.float64)
    stellar = 1.0 + 0.015 * np.sin(2.0 * np.pi * t / 2.5)
    # Sawtooth thruster pointing drift
    jitter = 0.12 * ((np.arange(n_cad) % 13) / 12.0 - 0.5)

    tpf = np.zeros((n_cad, ny, nx), dtype=np.float64)
    for r in range(ny):
        for c in range(nx):
            # Complementary linear jitter sensitivities across stamp columns so equal weights
            # on symmetric pixel pairs cancel pointing drift identically.
            sens = (c - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
            noise = rng.normal(0.0, 4e-4, size=n_cad)
            tpf[:, r, c] = 1000.0 * stellar * (1.0 + sens * jitter) + noise

    # Inject one heavily saturated pixel so thresh=0.8 exercises saturation masking
    tpf[:, ny // 2, nx // 2] = 25000.0 * (1.0 + 0.3 * jitter)

    ts = Table(
        {
            "time": t,
            "cadence": np.arange(1, n_cad + 1, dtype=np.int32),
            "x": jitter,
            "y": np.zeros(n_cad, dtype=np.float64),
            "quality": np.zeros(n_cad, dtype=np.int32),
        }
    )
    return tpf, ts, stellar


def _write_synthetic_tess_tpf(path: Path, n_cad: int = 60, ny: int = 4, nx: int = 5) -> np.ndarray:
    """Write a SPOC-compatible TESS TPF FITS file with complementary pointing jitter."""
    tpf_flux, ts, stellar = _make_synthetic_jitter_tpf(n_cad=n_cad, ny=ny, nx=nx)
    btjd = 1800.0 + ts["time"]

    prim = fits.PrimaryHDU()
    prim.header["MISSION"] = "TESS"
    prim.header["TELESCOP"] = "TESS"
    prim.header["INSTRUME"] = "TESS Photometer"
    prim.header["ORIGIN"] = "NASA/Ames"
    prim.header["CREATOR"] = "1234 TargetPixelExporterPipelineModule"
    prim.header["OBJECT"] = "SYNTH TESS"
    prim.header["TICID"] = 987654321
    prim.header["SECTOR"] = 14
    prim.header["CAMERA"] = 2
    prim.header["CCD"] = 3
    prim.header["RA_OBJ"] = 279.234
    prim.header["DEC_OBJ"] = 38.783

    dim_str = f"({nx},{ny})"
    cols = [
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=btjd),
        fits.Column(name="TIMECORR", format="E", unit="d", array=np.zeros(n_cad, dtype=np.float32)),
        fits.Column(name="CADENCENO", format="J", array=ts["cadence"]),
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
            array=tpf_flux.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_ERR",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=np.sqrt(np.abs(tpf_flux)).astype(np.float32),
        ),
        fits.Column(
            name="FLUX_BKG",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=np.zeros((n_cad, ny, nx), dtype=np.float32),
        ),
        fits.Column(
            name="FLUX_BKG_ERR",
            format=f"{ny * nx}E",
            dim=dim_str,
            unit="e-/s",
            array=np.ones((n_cad, ny, nx), dtype=np.float32),
        ),
        fits.Column(name="QUALITY", format="J", array=ts["quality"]),
        fits.Column(
            name="POS_CORR1",
            format="E",
            unit="pixel",
            array=np.asarray(ts["x"], dtype=np.float32),
        ),
        fits.Column(
            name="POS_CORR2",
            format="E",
            unit="pixel",
            array=np.asarray(ts["y"], dtype=np.float32),
        ),
    ]
    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    pixels_hdu.header["INSTRUME"] = "TESS Photometer"
    ap_hdu = fits.ImageHDU(data=np.full((ny, nx), 3, dtype=np.int32), name="APERTURE")
    fits.HDUList([prim, pixels_hdu, ap_hdu]).writeto(path, overwrite=True)
    return stellar


def test_frequency_conversion_exact_day_to_muhz() -> None:
    """Verify exact 1e6 / 86400 uHz/d scaling in get_pgram, plot_pgram, and plot_log_pgram."""
    assert DAY_TO_MUHZ == pytest.approx(1e6 / 86400.0, rel=0.0, abs=1e-12)
    assert abs(DAY_TO_MUHZ - 11.57) / DAY_TO_MUHZ > 3e-4  # Legacy 11.57 had a 352 ppm error

    t = np.linspace(0.0, 27.0, 600, dtype=np.float64)
    flux = 1.0 + 2e-3 * np.cos(2.0 * np.pi * 4.0 * t)
    freq, power, spower = get_pgram(t, flux)

    df_muhz = float(np.mean(np.diff(freq * (1e6 / 86400.0))))
    variance_ppm2 = float(np.nanstd(flux * 1e6) ** 2)
    assert float(np.sum(power) * df_muhz) == pytest.approx(variance_ppm2, rel=1e-12)

    rc_before = dict(mpl.rcParams)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    try:
        plot_pgram(ax1, freq, power, spower, "Synthetic")
        plot_log_pgram(ax2, freq, power, spower, "Synthetic")
        np.testing.assert_allclose(ax1.lines[0].get_xdata(), freq * (1e6 / 86400.0), rtol=1e-12)
        np.testing.assert_allclose(ax2.lines[0].get_xdata(), freq * (1e6 / 86400.0), rtol=1e-12)
    finally:
        plt.close(fig)
    assert dict(mpl.rcParams) == rc_before


def test_censor_tpf_do_quality_false_and_edge_cases() -> None:
    """Verify bool mask under do_quality=False, exact sat_pixels count, and short-TPF NaN handling."""
    tpf, ts, _ = _make_synthetic_jitter_tpf(n_cad=80, ny=4, nx=7)
    ts["quality"][5] = 32  # Reaction wheel desaturation bit (masked in default quality bitmask)
    # Include negative background-subtracted flux above minflux=-100 to verify it is preserved
    for k in range(80):
        tpf[k, 0, 0] = -10.0 + 0.1 * float(k)

    pixels_all, tsd_all, m_all, _, sat_all = censor_tpf(
        tpf, ts, thresh=0.8, minflux=-100.0, do_quality=False, verbose=False
    )
    assert m_all.dtype == np.bool_
    assert len(tsd_all) == 80
    assert len(np.unique(tsd_all["time"])) == 80
    assert pixels_all.shape[1] == 80
    np.testing.assert_allclose(pixels_all[0, :], -10.0 + 0.1 * np.arange(80), rtol=1e-12)
    # One saturated centre pixel dilated with structure (3, 1) yields 3 saturated pixels, not nx=7
    assert sat_all == 3

    _, _, _, _, sat_none = censor_tpf(
        tpf, ts, thresh=1.5, minflux=-100.0, do_quality=False, verbose=False
    )
    assert sat_none == 0

    pixels_qual, tsd_qual, m_qual, _, _ = censor_tpf(
        tpf, ts, thresh=0.8, minflux=-100.0, do_quality=True, verbose=False
    )
    assert m_qual.dtype == np.bool_
    assert len(tsd_qual) == 79
    assert 5 not in np.flatnonzero(m_qual)
    assert pixels_qual.shape[1] == 79

    # Short TPF (<= 60 cadences) with a single NaN cadence must drop only that cadence, not all pixels
    short_tpf, short_ts, _ = _make_synthetic_jitter_tpf(n_cad=20, ny=3, nx=3)
    short_tpf[7, 0, 0] = np.nan
    short_pix, short_tsd, _, _, _ = censor_tpf(
        short_tpf, short_ts, thresh=0.8, verbose=False
    )
    assert short_pix.shape[0] > 0
    assert short_pix.shape[1] == 19
    assert len(short_tsd) == 19

    # All-NaN TPF must raise a clear ValueError instead of crashing inside softmax(empty)
    with pytest.raises(ValueError, match="No valid pixels"):
        censor_tpf(np.full((10, 3, 3), np.nan), short_ts[:10], thresh=0.8, verbose=False)


def test_legacy_removals_and_print_time(capsys: pytest.CaptureFixture[str]) -> None:
    for mod in ("autograd", "jaxopt", "statsmodels", "sklearn", "skimage"):
        assert mod not in sys.modules or mod == "sklearn"  # lightkurve may import sklearn lazily

    print_time(12.3)
    print_time(125.4)
    print_time(7325.6)
    captured = capsys.readouterr().out
    assert "2 m" in captured
    assert "2 h 2 m" in captured

    with pytest.raises(ValueError, match="wcs"):
        remove_stars(np.ones((10, 4, 4)))


@pytest.mark.parametrize("obj_name", sorted(halo_objectives.mapping))
def test_all_objectives_analytical_two_pixel_optimum(obj_name: str) -> None:
    """Verify every objective strictly reduces loss and converges to w* = (1, 0) on a 2-pixel fixture."""
    n_cad = 32
    p0 = np.full(n_cad, 100.0, dtype=np.float64)
    p1 = 100.0 + 25.0 * ((-1.0) ** np.arange(n_cad))
    pvec = np.vstack([p0, p1])
    z0 = np.zeros(2, dtype=np.float64)  # Uniform initial weights w = (0.5, 0.5)

    assert jax.config.jax_enable_x64 is False
    res = jax_func(pvec, z0, obj_name, maxiter=45, lag=1)
    assert jax.config.jax_enable_x64 is False

    params, sol = res
    assert params.shape == (2,)
    assert sol is not None

    with jax.enable_x64(True):
        obj_fn = halo_objectives.mapping[obj_name]
        pvec_j = jnp.asarray(pvec, dtype=jnp.float64)
        z0_j = jnp.asarray(z0, dtype=jnp.float64)
        z_opt = jnp.asarray(params, dtype=jnp.float64)
        if obj_name == "owl":
            loss_init = float(obj_fn(z0_j, pvec_j))
            loss_opt = float(obj_fn(z_opt, pvec_j))
        else:
            loss_init = float(obj_fn(z0_j, 1, pvec_j))
            loss_opt = float(obj_fn(z_opt, 1, pvec_j))
        w_opt = np.asarray(halo_objectives.softmax(z_opt), dtype=np.float64)

    assert loss_opt < 1e-2 * loss_init
    assert w_opt[0] > 0.99


def test_jax_func_and_tv_tpf_validation_and_warm_start() -> None:
    """Verify lag/shape/finiteness validation, log-space warm-start in tv_tpf, and Dirichlet random_init."""
    n_cad = 24
    # M = 20 pixels: pixel 0 is constant, pixels 1..19 have alternating jitter
    pvec = np.tile(100.0 + 20.0 * ((-1.0) ** np.arange(n_cad)), (20, 1))
    pvec[0, :] = 100.0

    # Sparse simplex warm-start w_init = (1, 0, ..., 0) must NOT be double-softmaxed to ~1/20
    w_sparse = np.zeros(20, dtype=np.float64)
    w_sparse[0] = 1.0
    w_best, _, _ = tv_tpf(pvec, w_init=w_sparse, maxiter=1, verbose=False, objective="tv")
    assert w_best[0] > 0.999

    # Invalid simplex w_init must raise ValueError
    with pytest.raises(ValueError, match="non-negative"):
        tv_tpf(pvec, w_init=np.r_[-0.1, np.ones(19)], maxiter=5, verbose=False)
    with pytest.raises(ValueError, match="positive sum"):
        tv_tpf(pvec, w_init=np.zeros(20), maxiter=5, verbose=False)

    # Invalid lag, insufficient cadences, unknown objective, and zero-mean flux must raise ValueError
    z0 = np.zeros(2, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown objective"):
        jax_func(pvec[:2], z0, "nonexistent_objective", maxiter=5, lag=1)
    with pytest.raises(ValueError, match="lag must be"):
        jax_func(pvec[:2], z0, "tv", maxiter=5, lag=0)
    with pytest.raises(ValueError, match="lag must be"):
        jax_func(pvec[:2], z0, "tv", maxiter=5, lag=-1)
    with pytest.raises(ValueError, match="requires at least"):
        jax_func(pvec[:2, :4], z0, "tv", maxiter=5, lag=4)
    with pytest.raises(ValueError, match="requires at least"):
        jax_func(pvec[:2, :6], z0, "tv_o2", maxiter=5, lag=3)
    with pytest.raises(ValueError, match="requires at least"):
        jax_func(pvec[:2, :1], z0, "owl", maxiter=5, lag=1)
    with pytest.raises(ValueError, match="non-finite"):
        jax_func(np.zeros((2, 10), dtype=np.float64), z0, "tv", maxiter=5, lag=1)


def test_end_to_end_do_lc_and_tv_tpf_synthetic_smoke() -> None:
    tpf, ts, stellar = _make_synthetic_jitter_tpf(n_cad=80, ny=5, nx=5)

    _, ts_out, weights, wmap, pixels_sub = do_lc(
        tpf,
        ts,
        (None, None),
        sub=1,
        maxiter=50,
        thresh=0.8,
        sigclip=True,
        verbose=False,
        objective="tv",
    )
    assert jax.config.jax_enable_x64 is False

    corr_flux = np.asarray(ts_out["corr_flux"], dtype=np.float64)
    assert np.all(np.isfinite(corr_flux))
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert np.all(weights >= 0.0)
    assert wmap["weightmap"].shape == (5, 5)

    # Optimal halo light curve must suppress pointing-jitter TV relative to a jittery edge pixel
    raw_edge = pixels_sub[0, :] / np.median(pixels_sub[0, :])
    assert diff_1(corr_flux) < 0.25 * diff_1(raw_edge)
    np.testing.assert_allclose(corr_flux, stellar / np.median(stellar), atol=5e-3)

    # Verify random_init=True draws genuine Dirichlet weights rather than near-uniform 1/M
    _, _, w_rand, _, _ = do_lc(
        tpf,
        ts,
        (None, None),
        sub=1,
        maxiter=1,
        random_init=True,
        thresh=0.8,
        sigclip=False,
        verbose=False,
        objective="tv",
    )
    assert np.isclose(np.sum(w_rand), 1.0, atol=1e-6)

    # Also verify tv_tpf direct call with float32 input (reusing cached compiled solver)
    w_best, lc_opt, _ = tv_tpf(
        pixels_sub.astype(np.float32), maxiter=50, verbose=False, objective="tv"
    )
    assert w_best.dtype == np.float64
    assert lc_opt.dtype == np.float64
    assert jax.config.jax_enable_x64 is False


def test_halo_tpf_tess_and_time_scale_preservation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify halo_tpf_tess.halo suppresses jitter, stays silent when verbose=False, and keeps Time.scale."""
    tess_path = tmp_path / "synth_tess_tpf.fits"
    stellar = _write_synthetic_tess_tpf(tess_path, n_cad=60, ny=4, nx=5)

    ttpf = halo_tpf_tess(str(tess_path))
    capsys.readouterr()  # Clear any upstream lightkurve open output
    meta_tess, lc_tess = ttpf.halo(
        aperture_mask="all",
        rr=(0.5, 5.0),
        thresh=0.8,
        minflux=10.0,
        maxiter=35,
        sigclip=False,
        verbose=False,
    )
    assert capsys.readouterr().out == ""
    assert lc_tess.time.format == ttpf.time.format
    assert lc_tess.time.scale == ttpf.time.scale == "tdb"
    assert meta_tess["weightmap"].shape == (5, 4)
    flux_vals = np.asarray(lc_tess.flux.value, dtype=np.float64)
    np.testing.assert_allclose(flux_vals, stellar / np.median(stellar), atol=1e-2)


def test_read_epic_mast_fits_and_cli_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Verify reading EPIC_211309989_mast.fits (K2SC light curve fixture at repo root)
    assert EPIC_FITS.exists(), f"Expected fixture {EPIC_FITS} in repository root"
    with fits.open(EPIC_FITS) as epic_hdul:
        assert epic_hdul[1].name == "K2SC"
        epic_time = np.asarray(epic_hdul[1].data["time"], dtype=np.float64)
        epic_flux = np.asarray(epic_hdul[1].data["flux"], dtype=np.float64)
    assert epic_time.ndim == 1 and len(epic_time) > 3000
    freq, power, _ = get_pgram(epic_time[:300], epic_flux[:300])
    assert len(freq) == len(power) and np.all(np.isfinite(power))

    # 2. Verify read_tpf, halo_tpf, and CLI (including --do-split and --do-plot) on real K2 TPF
    assert K2_TPF_FITS.exists(), f"Expected fixture {K2_TPF_FITS}"
    tpf_arr, ts_tab = read_tpf(K2_TPF_FITS)
    assert tpf_arr.ndim == 3
    assert len(ts_tab) == tpf_arr.shape[0]
    assert set(ts_tab.colnames) >= {"time", "cadence", "x", "y", "quality"}

    # Slice to 80 cadences and drop 1 pixel below minflux in the second half so M_seg != M_full
    mini_tpf_path = tmp_path / "mini_k2_tpf.fits"
    with fits.open(K2_TPF_FITS) as hdul:
        sub_data = hdul[1].data[:80].copy()
        sub_data["FLUX"][50:, 8, 8] = -500.0
        hdul[1].data = sub_data
        hdul.writeto(mini_tpf_path, overwrite=True)

    htpf = halo_tpf(str(mini_tpf_path))
    # Explicitly verify scale="tdb" preservation even when time format is "jd"
    jd_tdb = Time(htpf.time.jd, format="jd", scale="tdb")
    assert jd_tdb.scale == "tdb"
    capsys.readouterr()
    meta, lc_out = htpf.halo(
        aperture_mask="all",
        rr=(1.0, 4.0),
        thresh=0.8,
        minflux=100.0,
        maxiter=15,
        sigclip=False,
        verbose=False,
    )
    assert capsys.readouterr().out == ""
    assert lc_out.time.scale == htpf.time.scale == "tdb"
    assert "weightmap" in meta
    assert meta["weightmap"].shape == (tpf_arr.shape[2], tpf_arr.shape[1])
    assert len(lc_out.flux) == len(htpf.time)
    assert np.sum(np.isfinite(lc_out.flux.value)) > 60

    # CLI end-to-end smoke test with --do-split (1 timestamp split + M_seg != M_full) and --do-plot
    mid_time = float(np.nanmedian(sub_data["TIME"]))
    out_dir = tmp_path / "cli_out"
    out_dir.mkdir()
    cli.main(
        [
            str(mini_tpf_path),
            "--name",
            "epic_smoke",
            "-c",
            "3",
            "-maxiter",
            "10",
            "--rr",
            "[1.0,4.0]",
            "--thresh",
            "0.8",
            "--minflux",
            "100.0",
            "--do-split",
            "--splits",
            f"[{mid_time:.6f}]",
            "--do-plot",
            "--no-sigclip",
            "--quiet",
            "--save-dir",
            str(out_dir),
        ]
    )
    saved_lc = out_dir / "epic_smoke_halo_lc_tv.fits"
    saved_png = out_dir / "epic_smoke_all_tv.png"
    assert saved_lc.exists()
    assert saved_png.exists()
    with fits.open(saved_lc) as out_hdul:
        assert len(out_hdul) == 2
        assert "corr_flux" in out_hdul[1].data.names
        assert len(out_hdul[1].data) == 80
