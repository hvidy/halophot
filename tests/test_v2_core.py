"""Unit and numerical verification tests for the simplified ``halophot`` JAX core."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage

import halophot
from halophot.objectives import (
    OBJECTIVES,
    charbonnier_abs,
    compute_step_mask,
    normalized_flux,
    register_objective,
    resolve_objective,
    tv_exp,
    tv_o1,
    tv_o2,
)
from halophot.optimise import (
    OptimConfig,
    cadence_scaled_savgol_window,
    cadence_scaled_sigclip_mask,
    optimise_halo,
    orbit_block_jackknife,
)
from halophot.simplex import initial_logits, logits_to_weights, weights_to_logits

REPO_ROOT = Path(__file__).resolve().parents[1]
K2_TPF_FITS = REPO_ROOT / "data" / "ktwo205897543-c03_lpd-targ.fits.gz"


def test_direct_simplex_round_trip_gauge_and_finite_diff() -> None:
    """Verify symmetric softmax simplex round-trip, zero-sum gradient gauge, and FD Jacobian."""
    assert jax.config.jax_enable_x64 is False

    with jax.enable_x64(True):
        # M = 1 edge case
        w1 = logits_to_weights(jnp.array([2.5], dtype=jnp.float64))
        assert w1.shape == (1,) and float(w1[0]) == 1.0

        # Asymmetric non-unit inputs (away from uniform fixed point)
        rng = np.random.default_rng(2026)
        u_batch = jnp.asarray(rng.normal(0.4, 1.7, size=(8, 7)), dtype=jnp.float64)
        w_batch = logits_to_weights(u_batch)
        np.testing.assert_allclose(np.sum(np.asarray(w_batch), axis=-1), 1.0, atol=1e-14)
        assert np.all(np.asarray(w_batch) > 0.0)

        # Round-trip weights -> logits -> weights
        u_rt = weights_to_logits(w_batch)
        np.testing.assert_allclose(np.mean(np.asarray(u_rt), axis=-1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(logits_to_weights(u_rt)), np.asarray(w_batch), rtol=1e-13, atol=1e-13)

        # Translation invariance: shifting logits by arbitrary constant leaves weights unchanged
        np.testing.assert_allclose(
            np.asarray(logits_to_weights(u_batch + 17.3)),
            np.asarray(w_batch),
            rtol=1e-14,
            atol=1e-14,
        )

        # Gradient wrt logits sums identically to zero and matches central finite differences
        u_vec = u_batch[0]
        coeffs = jnp.linspace(0.7, 4.1, 7, dtype=jnp.float64)
        grad_ad = jax.grad(lambda x: jnp.dot(coeffs, logits_to_weights(x)))(u_vec)
        assert abs(float(jnp.sum(grad_ad))) < 1e-14

        h = 1e-6
        grad_fd = np.array(
            [
                float(
                    jnp.dot(coeffs, logits_to_weights(u_vec.at[k].add(h)))
                    - jnp.dot(coeffs, logits_to_weights(u_vec.at[k].add(-h)))
                )
                / (2.0 * h)
                for k in range(7)
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(np.asarray(grad_ad), grad_fd, rtol=1e-7, atol=1e-9)

    # Mixed-sign pixel fluxes: uniform when mean > 0, positive-pixel start when mean <= 0
    u_pos = initial_logits(np.array([100.0, 50.0, -5.0]))
    np.testing.assert_allclose(np.asarray(u_pos), np.zeros(3), atol=1e-15)
    u_neg = initial_logits(np.array([10.0, -50.0, -50.0]))
    w_neg = np.asarray(logits_to_weights(u_neg))
    assert w_neg[0] > 0.99 and float(np.dot(w_neg, [10.0, -50.0, -50.0])) > 0.0


def test_charbonnier_scale_invariance_and_downlink_gap_masking() -> None:
    """Verify cancellation-free Charbonnier across 0.1 ppm..10% and exact downlink gap invariance."""
    with jax.enable_x64(True):
        # Cancellation-free at |r| << eps
        r_tiny = jnp.array([0.0, 1e-13, -1e-13], dtype=jnp.float64)
        phi_tiny = charbonnier_abs(r_tiny, eps=1e-5)
        assert float(phi_tiny[0]) == 0.0
        np.testing.assert_allclose(np.asarray(phi_tiny[1:]), 0.5e-21, rtol=1e-10)
        g0 = jax.grad(lambda x: jnp.sum(charbonnier_abs(x, eps=1e-5)))(r_tiny)
        assert float(g0[0]) == 0.0

    # Dynamic eps_eff = epsilon * b_hat stays purely L1 across 6 orders of magnitude (0.1 ppm to 10%)
    rng = np.random.default_rng(101)
    n_cad = 250
    t = np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)
    for amp in (1e-7, 1e-5, 1e-1):
        p_quiet = 1000.0 * (1.0 + rng.normal(0.0, amp, size=n_cad))
        p_noisy = 1000.0 * (1.0 + rng.normal(0.0, 10.0 * amp, size=n_cad))
        fit = optimise_halo(np.vstack([p_quiet, p_noisy]), t, objective="tv", maxiter=200)
        assert fit.weights[0] > 0.95, f"Failed L1 scale invariance at amp={amp}: w[0]={fit.weights[0]}"
        assert jax.config.jax_enable_x64 is False

    # 2-orbit downlink gap: thermal step across gap must NEVER penalize quiet pixels
    dt = 2.0 / 1440.0
    n_half = 40
    t_orb = np.arange(n_half, dtype=np.float64) * dt
    t_gap_short = np.concatenate([t_orb, 1.5 + t_orb])
    t_gap_long = np.concatenate([t_orb, 45.0 + t_orb])

    step_gap = np.concatenate([np.full(n_half, 950.0), np.full(n_half, 1050.0)])
    sawtooth = 1000.0 + 60.0 * ((-1.0) ** np.arange(2 * n_half))
    pixels = np.vstack([step_gap, 1.1 * step_gap, sawtooth, 0.9 * sawtooth])

    sm_short = compute_step_mask(t_gap_short, lag=1)
    sm_long = compute_step_mask(t_gap_long, lag=1)
    assert bool(sm_short[n_half - 1]) is False
    assert bool(sm_short[n_half - 2]) is True
    assert bool(sm_short[n_half]) is True
    np.testing.assert_array_equal(sm_short, sm_long)

    with jax.enable_x64(True):
        w_unif = jnp.full(4, 0.25, dtype=jnp.float64)
        f_j = normalized_flux(w_unif, jnp.asarray(pixels, dtype=jnp.float64))
        for obj_name in ("tv", "tv_o2", "tv_exp", "l2v", "l3v", "owl"):
            _, fn = resolve_objective(obj_name)
            v_short = float(
                fn(
                    flux=f_j,
                    step_mask=jnp.asarray(sm_short),
                    weights=w_unif,
                    pixels=jnp.asarray(pixels),
                    times=jnp.asarray(t_gap_short),
                    dt_nom=jnp.asarray(dt),
                    lag=1,
                    eps=jnp.asarray(1e-6),
                    tau=3.0,
                    gap_tol=0.25,
                )
            )
            v_long = float(
                fn(
                    flux=f_j,
                    step_mask=jnp.asarray(sm_long),
                    weights=w_unif,
                    pixels=jnp.asarray(pixels),
                    times=jnp.asarray(t_gap_long),
                    dt_nom=jnp.asarray(dt),
                    lag=1,
                    eps=jnp.asarray(1e-6),
                    tau=3.0,
                    gap_tol=0.25,
                )
            )
            assert v_short == pytest.approx(v_long, rel=1e-13, abs=1e-13)

    fit_gap = optimise_halo(pixels, t_gap_short, objective="tv", maxiter=300)
    assert float(fit_gap.weights[0] + fit_gap.weights[1]) > 0.99


def test_custom_student_objectives_and_register_objective() -> None:
    """Verify (flux), (flux, step_mask), keyword-aware callables, and @register_objective."""
    rng = np.random.default_rng(77)
    n_cad = 120
    dt = 2.0 / 1440.0
    t = np.arange(n_cad, dtype=np.float64) * dt
    t[60:] += 1.0  # Insert a 1-day downlink gap at cadence 60

    # Pixel 0 has a gap jump but zero intra-orbit noise; pixels 1..3 have intra-orbit jitter
    p0 = np.concatenate([np.full(60, 1000.0), np.full(60, 1100.0)])
    p1 = 1000.0 + 40.0 * ((-1.0) ** np.arange(n_cad))
    p2 = 1000.0 + 25.0 * ((-1.0) ** np.arange(n_cad))
    p3 = 1000.0 + rng.normal(0.0, 20.0, size=n_cad)
    pixels = np.vstack([p0, p1, p2, p3])

    # 1. Minimal 1-positional-arg custom objective: def fn(flux)
    def simple_l2(flux):
        return jnp.mean(jnp.square(jnp.diff(flux)))

    fit1 = optimise_halo(pixels, t, objective=simple_l2, maxiter=200)
    assert fit1.loss_final < fit1.loss_init
    assert np.isclose(np.sum(fit1.weights), 1.0, atol=1e-12)

    # 2. Gap-safe 2-positional-arg custom objective: def fn(flux, step_mask)
    def gap_safe_tv(flux, step_mask):
        diffs = jnp.diff(flux)
        return jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-12), 0.0)) / jnp.sum(step_mask)

    fit2 = optimise_halo(pixels, t, objective=gap_safe_tv, maxiter=200)
    assert fit2.weights[0] > 0.99

    # 3. Keyword-aware custom objective using weights + pixels + times + dt_nom + lag
    def weight_reg_obj(flux, *, step_mask, weights, pixels, times, dt_nom, lag):
        assert pixels.shape[0] == weights.shape[0]
        assert times.shape[0] == flux.shape[0]
        diffs = flux[lag:] - flux[:-lag]
        tv = jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-12), 0.0)) / jnp.sum(step_mask)
        return tv + 0.05 * jnp.sum(jnp.square(weights)) + 0.0 * dt_nom

    fit3 = optimise_halo(pixels, t, objective=weight_reg_obj, maxiter=200)
    # Quadratic weight penalty keeps weights more spread than unregularized fit2
    assert fit3.weights[0] < fit2.weights[0]
    assert fit3.loss_final < fit3.loss_init

    # 4. @register_objective
    try:

        @register_objective("_unit_test_registered_tv")
        def _reg_tv(flux, step_mask):
            diffs = jnp.diff(flux)
            return jnp.sum(jnp.where(step_mask, jnp.sqrt(diffs**2 + 1e-12), 0.0)) / jnp.sum(step_mask)

        assert "_unit_test_registered_tv" in OBJECTIVES
        fit4 = optimise_halo(pixels, t, objective="_unit_test_registered_tv", maxiter=200)
        np.testing.assert_allclose(fit4.weights, fit2.weights, rtol=1e-5, atol=1e-5)
    finally:
        OBJECTIVES.pop("_unit_test_registered_tv", None)

    # 5. Unsupported required parameter raises informative TypeError before JIT
    def bad_obj(flux, *, unknown_required_param):
        return jnp.mean(flux) + unknown_required_param

    with pytest.raises(TypeError, match="unknown_required_param"):
        optimise_halo(pixels, t, objective=bad_obj, maxiter=10)


def test_real_k2_tpf_and_sirius_scale_lbfgs_convergence() -> None:
    """Regression test for B1: zero step-0 false convergence on K2 C03 and <100 steps on Sirius M=1808."""
    assert K2_TPF_FITS.exists()
    with fits.open(K2_TPF_FITS) as hdul:
        data = hdul[1].data
        flux_3d = np.asarray(data["FLUX"][:1000], dtype=np.float64)
        times = np.asarray(data["TIME"][:1000], dtype=np.float64)
        qual = np.asarray(data["QUALITY"][:1000], dtype=np.int32)

    good_cad = np.isfinite(times) & (qual == 0)
    flux_c = flux_3d[good_cad]
    t_c = times[good_cad]
    any_fin = np.any(np.isfinite(flux_c), axis=0)
    p99 = np.full(flux_c.shape[1:], np.nan)
    p99[any_fin] = np.nanpercentile(flux_c[:, any_fin], 99.0, axis=0)
    sat = ndimage.binary_dilation(
        np.isfinite(p99) & (p99 >= 1.66e5), structure=np.ones((3, 3), dtype=bool)
    )
    valid_pix = np.all(np.isfinite(flux_c), axis=0) & ~sat & (np.min(flux_c, axis=0) >= -100.0)
    pix_k2 = flux_c[:, valid_pix].T

    fit_k2 = optimise_halo(pix_k2, t_c, objective="tv", maxiter=500)
    sm = compute_step_mask(t_c)
    std_diff_ppm = float(np.std(np.diff(fit_k2.norm_flux)[sm]) * 1e6)
    assert fit_k2.n_steps > 10
    assert fit_k2.loss_final < 0.1 * fit_k2.loss_init
    assert std_diff_ppm < 600.0

    # Sirius-scale stamp (M = 1808, asymmetric wings + pointing jitter + Poisson noise):
    # TV drops scatter >100x with zero step-0 stall or one-hot collapse; smooth l2v converges < 100 steps.
    rng = np.random.default_rng(2026)
    m_sir, n_sir = 1808, 400
    t_sir = np.arange(n_sir, dtype=np.float64) * (2.0 / 1440.0)
    stellar = 1.0 + 0.002 * np.sin(2.0 * np.pi * t_sir / 0.15)
    dx = 0.03 * np.sin(2.0 * np.pi * t_sir / 0.05) + 0.015 * ((-1.0) ** np.arange(n_sir))
    x_pos = np.concatenate([np.full(m_sir * 3 // 4, 1.0), np.full(m_sir - m_sir * 3 // 4, -1.0)])
    flux_scale = np.geomspace(500.0, 5e4, m_sir)
    pix_sir = (
        flux_scale[:, None] * stellar[None, :] * (1.0 + x_pos[:, None] * dx[None, :])
        + rng.normal(0.0, np.sqrt(flux_scale / 120.0)[:, None], size=(m_sir, n_sir))
    )
    sm_sir = compute_step_mask(t_sir)
    fit_sir_tv = optimise_halo(pix_sir, t_sir, objective="tv", maxiter=100)
    ppm_sir_tv = float(np.std(np.diff(fit_sir_tv.norm_flux)[sm_sir]) * 1e6)
    assert fit_sir_tv.n_steps > 10
    assert fit_sir_tv.loss_final < 0.01 * fit_sir_tv.loss_init
    assert ppm_sir_tv < 150.0

    fit_sir_l2 = optimise_halo(pix_sir, t_sir, objective="l2v", maxiter=100)
    assert fit_sir_l2.converged is True
    assert 1 < fit_sir_l2.n_steps < 100
    assert fit_sir_l2.loss_final < 1e-3 * fit_sir_l2.loss_init


def test_sigclip_window_days_burst_and_algol_eclipse() -> None:
    """Regression test for B6: window_days honoured, 5-cadence 20s burst caught 5/5, Algol preserved."""
    assert cadence_scaled_savgol_window(2.0 / 1440.0, window_days=0.02) == 15

    rng = np.random.default_rng(314)
    dt_20s = 20.0 / 86400.0
    t_20s = np.arange(1000, dtype=np.float64) * dt_20s
    f_burst = 1.0 + 0.002 * np.sin(2.0 * np.pi * t_20s / 0.25) + rng.normal(0.0, 1e-4, size=1000)
    f_burst[500:505] += 50.0 * 1e-4  # 5-cadence (100 s) 50-sigma burst

    m_default = cadence_scaled_sigclip_mask(f_burst, t_20s, window_days=0.02)
    assert int(np.sum(~m_default[500:505])) == 5

    # window_days must actually change window half-width (not dead code!):
    # a 10-cadence burst fails at window_days=0.002 (hw=4) but is caught at window_days=0.02 (hw=15)
    f_wide = f_burst.copy()
    f_wide[500:510] += 50.0 * 1e-4
    m_narrow = cadence_scaled_sigclip_mask(f_wide, t_20s, window_days=0.002)
    m_wide = cadence_scaled_sigclip_mask(f_wide, t_20s, window_days=0.02)
    assert int(np.sum(~m_wide[500:510])) == 10
    assert int(np.sum(~m_narrow[500:510])) < 10

    # Deep 30% Algol primary eclipse + 4 spikes: spikes caught, eclipse trough preserved
    dt_2m = 120.0 / 86400.0
    t_2m = np.arange(1500, dtype=np.float64) * dt_2m
    f_algol = np.ones(1500, dtype=np.float64) + rng.normal(0.0, 2e-4, size=1500)
    phase = np.linspace(-1.0, 1.0, 180)
    f_algol[600:780] -= 0.30 * np.cos(0.5 * np.pi * phase) ** 2
    spikes = [100, 350, 950, 1200]
    f_algol[spikes] += 0.35
    m_algol = cadence_scaled_sigclip_mask(f_algol, t_2m, window_days=0.02)
    assert int(np.sum(~m_algol[spikes])) == 4
    assert int(np.sum(m_algol[650:730])) == 80  # Eclipse core 100% preserved


def test_all_builtin_objectives_and_orbit_block_jackknife() -> None:
    """Verify every built-in objective, OptimConfig, legacy_tv deprecation, and block jackknife."""
    rng = np.random.default_rng(99)
    n_cad = 160
    dt = 2.0 / 1440.0
    t = np.arange(n_cad, dtype=np.float64) * dt
    t[80:] += 1.5  # Orbit downlink gap
    stellar = 1.0 + 0.01 * np.sin(2.0 * np.pi * t / 0.12)
    jitter = 0.05 * ((-1.0) ** np.arange(n_cad))
    pixels = np.vstack(
        [
            1000.0 * stellar * (1.0 + jitter) + rng.normal(0.0, 0.5, size=n_cad),
            1000.0 * stellar * (1.0 - jitter) + rng.normal(0.0, 0.5, size=n_cad),
            1000.0 * stellar * (1.0 + 2.0 * jitter) + rng.normal(0.0, 5.0, size=n_cad),
        ]
    )

    for obj_name in ("tv", "tv_o2", "tv_exp", "l2v", "l3v", "owl"):
        fit = optimise_halo(pixels, t, objective=obj_name, maxiter=200)
        assert fit.weights.shape == (3,)
        assert np.isclose(np.sum(fit.weights), 1.0, atol=1e-12)
        assert fit.loss_final <= fit.loss_init + 1e-12
        assert np.all(np.isfinite(fit.norm_flux))

    # OptimConfig + orbit block jackknife + legacy_tv compatibility no-op
    cfg = OptimConfig(objective="tv", compute_uncertainty=True, jackknife_blocks=4, maxiter=200)
    with pytest.warns(DeprecationWarning, match="legacy_tv"):
        fit_jk = optimise_halo(pixels, t, config=cfg, legacy_tv=True)
    assert fit_jk.weight_std is not None and fit_jk.weight_std.shape == (3,)
    assert np.all(np.isfinite(fit_jk.weight_std)) and np.all(fit_jk.weight_std >= 0.0)
    assert fit_jk.jackknife_weights is not None and fit_jk.jackknife_weights.shape == (4, 3)

    w_std, w_folds = orbit_block_jackknife(pixels, t, fit_jk.weights, n_blocks=4, maxiter=200)
    np.testing.assert_allclose(w_std, fit_jk.weight_std, rtol=1e-6, atol=1e-8)
    assert w_folds.shape == (4, 3)

    # Verify direct tv_o1/tv_o2/tv_exp calls work standalone
    with jax.enable_x64(True):
        f_test = jnp.asarray(fit_jk.norm_flux, dtype=jnp.float64)
        assert float(tv_o1(f_test)) >= 0.0
        assert float(tv_o2(f_test)) >= 0.0
        assert float(tv_exp(f_test, times=jnp.asarray(t), dt_nom=dt, tau=2.0)) >= 0.0
    assert halophot.OBJECTIVES is OBJECTIVES
