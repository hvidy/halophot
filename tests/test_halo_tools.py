"""Regression and smoke tests for ``halophot.halo_tools``, ``halo_objectives``, and CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from halophot import cli, halo_objectives
from halophot.halo_tools import (
    DAY_TO_MUHZ,
    censor_tpf,
    diff_1,
    do_lc,
    get_pgram,
    halo_tpf,
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


def test_censor_tpf_do_quality_false_boolean_mask_regression() -> None:
    """Regression test: do_quality=False must use a bool mask, not repeat cadence index 1."""
    tpf, ts, _ = _make_synthetic_jitter_tpf(n_cad=80, ny=4, nx=4)
    ts["quality"][5] = 32  # Reaction wheel desaturation bit (masked in default quality bitmask)
    # Give every cadence a unique marker flux so accidental duplication of index 1 is obvious
    for k in range(80):
        tpf[k, 0, 0] = 500.0 + float(k)

    pixels_all, tsd_all, m_all, _, _ = censor_tpf(
        tpf, ts, thresh=0.8, do_quality=False, verbose=False
    )
    assert m_all.dtype == np.bool_
    assert len(tsd_all) == 80
    assert len(np.unique(tsd_all["time"])) == 80
    assert pixels_all.shape[1] == 80
    np.testing.assert_allclose(pixels_all[0, :], 500.0 + np.arange(80), rtol=1e-12)

    pixels_qual, tsd_qual, m_qual, _, _ = censor_tpf(
        tpf, ts, thresh=0.8, do_quality=True, verbose=False
    )
    assert m_qual.dtype == np.bool_
    assert len(tsd_qual) == 79
    assert 5 not in np.flatnonzero(m_qual)
    assert pixels_qual.shape[1] == 79


def test_legacy_removals_and_print_time(capsys: pytest.CaptureFixture[str]) -> None:
    for mod in ("autograd", "jaxopt", "statsmodels", "sklearn", "skimage"):
        assert mod not in sys.modules or mod == "sklearn"  # lightkurve may import sklearn lazily

    print_time(12.3)
    print_time(125.4)
    print_time(7325.6)
    captured = capsys.readouterr().out
    assert "2 m" in captured
    assert "2 h 2 m" in captured

    with pytest.raises(NotImplementedError, match="Phase 4"):
        remove_stars(np.ones((10, 4, 4)))


@pytest.mark.parametrize("obj_name", sorted(halo_objectives.mapping))
def test_all_objectives_supported_in_jax_func(obj_name: str) -> None:
    rng = np.random.default_rng(7)
    pvec = 100.0 + rng.normal(0.0, 1.0, size=(6, 32))
    w0 = np.ones(6, dtype=np.float64) / 6.0

    assert jax.config.jax_enable_x64 is False
    # Compile full XLA L-BFGS on representative 'tv' and 'owl' (1-arg vs 2-arg loss signatures)
    # and run remaining objectives under disable_jit to verify dispatch + float64 math without
    # redundant 6x L-BFGS XLA compilations.
    if obj_name in {"tv", "owl"}:
        res = jax_func(pvec, w0, obj_name, maxiter=10, lag=1)
    else:
        with jax.disable_jit():
            res = jax_func(pvec, w0, obj_name, maxiter=3, lag=1)
    assert jax.config.jax_enable_x64 is False

    params, sol = res
    assert params.shape == (6,)
    assert np.all(np.isfinite(np.asarray(params)))
    assert sol is not None

    with pytest.raises(ValueError, match="Unknown objective"):
        jax_func(pvec, w0, "nonexistent_objective", maxiter=5, lag=1)


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

    # Also verify tv_tpf direct call with float32 input (reusing cached compiled solver)
    w_best, lc_opt, _ = tv_tpf(
        pixels_sub.astype(np.float32), maxiter=50, verbose=False, objective="tv"
    )
    assert w_best.dtype == np.float64
    assert lc_opt.dtype == np.float64
    assert jax.config.jax_enable_x64 is False


def test_read_epic_mast_fits_and_cli_smoke(tmp_path: Path) -> None:
    # 1. Verify reading EPIC_211309989_mast.fits (K2SC light curve fixture at repo root)
    assert EPIC_FITS.exists(), f"Expected fixture {EPIC_FITS} in repository root"
    with fits.open(EPIC_FITS) as epic_hdul:
        assert epic_hdul[1].name == "K2SC"
        epic_time = np.asarray(epic_hdul[1].data["time"], dtype=np.float64)
        epic_flux = np.asarray(epic_hdul[1].data["flux"], dtype=np.float64)
    assert epic_time.ndim == 1 and len(epic_time) > 3000
    freq, power, _ = get_pgram(epic_time[:300], epic_flux[:300])
    assert len(freq) == len(power) and np.all(np.isfinite(power))

    # 2. Verify read_tpf, halo_tpf, and CLI on real K2 TPF fixture
    assert K2_TPF_FITS.exists(), f"Expected fixture {K2_TPF_FITS}"
    tpf_arr, ts_tab = read_tpf(K2_TPF_FITS)
    assert tpf_arr.ndim == 3
    assert len(ts_tab) == tpf_arr.shape[0]
    assert set(ts_tab.colnames) >= {"time", "cadence", "x", "y", "quality"}

    # Slice to 80 cadences for fast halo_tpf + CLI verification
    mini_tpf_path = tmp_path / "mini_k2_tpf.fits"
    with fits.open(K2_TPF_FITS) as hdul:
        hdul[1].data = hdul[1].data[:80]
        hdul.writeto(mini_tpf_path, overwrite=True)

    htpf = halo_tpf(str(mini_tpf_path))
    meta, lc_out = htpf.halo(
        aperture_mask="all",
        rr=(1.0, 4.0),
        thresh=0.8,
        minflux=100.0,
        maxiter=15,
        sigclip=False,
        verbose=False,
    )
    assert "weightmap" in meta
    assert meta["weightmap"].shape == (tpf_arr.shape[2], tpf_arr.shape[1])
    assert len(lc_out.flux) == len(htpf.time)
    assert np.sum(np.isfinite(lc_out.flux.value)) > 60

    # CLI end-to-end smoke test (reuses the compiled (M, N, maxiter=15) kernel above)
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
            "15",
            "--rr",
            "[1.0,4.0]",
            "--thresh",
            "0.8",
            "--minflux",
            "100.0",
            "--no-sigclip",
            "--quiet",
            "--save-dir",
            str(out_dir),
        ]
    )
    saved_lc = out_dir / "epic_smoke_halo_lc_tv.fits"
    assert saved_lc.exists()
    with fits.open(saved_lc) as out_hdul:
        assert len(out_hdul) == 2
        assert "corr_flux" in out_hdul[1].data.names
