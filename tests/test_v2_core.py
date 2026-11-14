"""Unit and numerical verification tests for the ``halophot`` v2.0 Zodiax / JAX core."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from halophot.halo_tools import tv_tpf
from halophot.models import HaloObservation, HaloWeights, bucket_pad_length
from halophot.objectives import (
    AVAILABLE_OBJECTIVES,
    charbonnier_penalty,
    halo_objective,
    photon_noise_penalty,
    tv_charbonnier_o2,
    tv_exp_scan,
)
from halophot.optimise import (
    _get_fisher_fn,
    build_value_and_grad,
    cadence_scaled_savgol_window,
    cadence_scaled_sigclip_mask,
    optimise_halo,
    orbit_block_jackknife,
    safe_eigen_projection,
)
from halophot.simplex import (
    flux_share_to_weights,
    ilr_to_simplex,
    simplex_to_ilr,
    uniform_weights_ilr,
    weights_to_flux_share,
    weights_to_ilr,
)


def test_ilr_round_trip_and_zero_nan_derivatives_at_origin() -> None:
    """Verify O(M) ILR round-trip, Charnes-Cooper bijection, and finite Jacobian/Hessian at z=0."""
    with jax.enable_x64(True):
        # M = 1 degenerate simplex edge case
        z_m1 = jnp.zeros(0, dtype=jnp.float64)
        t_m1 = ilr_to_simplex(z_m1)
        assert t_m1.shape == (1,) and float(t_m1[0]) == 1.0
        assert simplex_to_ilr(t_m1).shape == (0,)

        # Exact z = 0 evaluation, forward/reverse Jacobian, and Hessian for M = 8
        m_dim = 8
        z0 = jnp.zeros(m_dim - 1, dtype=jnp.float64)
        ilr_j = jax.jit(ilr_to_simplex)
        s2i_j = jax.jit(simplex_to_ilr)
        theta0 = ilr_j(z0)
        np.testing.assert_allclose(theta0, np.full(m_dim, 1.0 / m_dim), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(s2i_j(theta0), z0, atol=1e-14)

        jac_fwd = jax.jit(jax.jacfwd(ilr_to_simplex))(z0)
        jac_rev = jax.jit(jax.jacrev(ilr_to_simplex))(z0)
        assert jnp.all(jnp.isfinite(jac_fwd))
        assert jnp.all(jnp.isfinite(jac_rev))
        np.testing.assert_allclose(jac_fwd, jac_rev, rtol=1e-13, atol=1e-13)

        # Central finite-difference verification of Jacobian and Hessian at exact z = 0
        h = 1e-6
        jac_fd = np.zeros((m_dim, m_dim - 1), dtype=np.float64)
        for k in range(m_dim - 1):
            dz = jnp.zeros_like(z0).at[k].set(h)
            jac_fd[:, k] = np.asarray(ilr_j(z0 + dz) - ilr_j(z0 - dz)) / (2.0 * h)
        np.testing.assert_allclose(jac_rev, jac_fd, rtol=1e-8, atol=1e-10)

        lin_coeff = jnp.linspace(0.3, 2.7, m_dim, dtype=jnp.float64)

        def scalar_fn(z: jax.Array) -> jax.Array:
            return jnp.dot(lin_coeff, ilr_to_simplex(z))

        hess_ad = jax.jit(jax.hessian(scalar_fn))(z0)
        assert jnp.all(jnp.isfinite(hess_ad))

        grad_fn = jax.jit(jax.grad(scalar_fn))
        hess_fd = np.zeros((m_dim - 1, m_dim - 1), dtype=np.float64)
        for k in range(m_dim - 1):
            dz = jnp.zeros_like(z0).at[k].set(h)
            hess_fd[:, k] = np.asarray(grad_fn(z0 + dz) - grad_fn(z0 - dz)) / (2.0 * h)
        np.testing.assert_allclose(hess_ad, hess_fd, rtol=1e-7, atol=1e-9)

        # Batched random round-trip and Charnes-Cooper bijection across 5 orders of magnitude in P_bar
        rng = np.random.default_rng(2026)
        z_batch = jnp.asarray(rng.normal(0.0, 1.8, size=(12, m_dim - 1)), dtype=jnp.float64)
        theta_batch = ilr_to_simplex(z_batch)
        np.testing.assert_allclose(jnp.sum(theta_batch, axis=-1), 1.0, atol=1e-14)
        np.testing.assert_allclose(simplex_to_ilr(theta_batch), z_batch, rtol=1e-12, atol=1e-12)

        p_bar = jnp.geomspace(10.0, 5e5, m_dim, dtype=jnp.float64)
        w_rec = flux_share_to_weights(weights_to_flux_share(theta_batch, p_bar), p_bar)
        np.testing.assert_allclose(w_rec, theta_batch, rtol=1e-12, atol=1e-12)

        # Uniform physical weights w_i = 1/M initialization must be unbiased by P_bar
        z_unif = uniform_weights_ilr(p_bar, legacy_tv=False)
        w_unif = flux_share_to_weights(ilr_to_simplex(z_unif), p_bar)
        np.testing.assert_allclose(w_unif, np.full(m_dim, 1.0 / m_dim), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(weights_to_ilr(w_unif, p_bar), z_unif, rtol=1e-12, atol=1e-12)


def test_halo_weights_pytree_sole_differentiable_leaf_and_fd_grad() -> None:
    """Verify z_ilr is the sole array leaf on HaloWeights(zdx.Base) via eqx.partition."""
    rng = np.random.default_rng(77)
    m_pix, n_cad = 6, 48
    t = np.arange(n_cad, dtype=np.float64) * (2.0 / 1440.0)
    pixels = 500.0 + rng.normal(0.0, 5.0, size=(m_pix, n_cad))

    obs = HaloObservation.from_arrays(pixels, t, lag=1)
    weights = HaloWeights.from_uniform(obs.mean_pixel_flux, legacy_tv=False)
    assert jax.config.jax_enable_x64 is False

    diff_tree, static_tree = eqx.partition(weights, eqx.is_array)
    diff_leaves = jax.tree_util.tree_leaves(diff_tree)
    static_leaves = jax.tree_util.tree_leaves(static_tree)
    assert len(diff_leaves) == 1
    assert diff_leaves[0] is weights.z_ilr
    assert diff_leaves[0].shape == (m_pix - 1,)
    assert len(static_leaves) == 0

    # Verify build_value_and_grad differentiates strictly wrt z_ilr and matches finite differences
    with jax.enable_x64(True):
        vg_fn = build_value_and_grad("tv", epsilon=1e-5)
        loss_val, grad_tree = vg_fn(weights, obs)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad_tree, eqx.is_array))
        assert len(grad_leaves) == 1
        assert grad_tree.z_ilr.shape == (m_pix - 1,)
        assert np.isfinite(float(loss_val))
        assert np.all(np.isfinite(np.asarray(grad_tree.z_ilr)))

        h = 1e-6
        fd_grad = np.zeros(m_pix - 1, dtype=np.float64)
        for k in range(m_pix - 1):
            dz = jnp.zeros_like(weights.z_ilr).at[k].set(h)
            w_plus = HaloWeights(z_ilr=weights.z_ilr + dz, legacy_tv=False)
            w_minus = HaloWeights(z_ilr=weights.z_ilr - dz, legacy_tv=False)
            l_p = halo_objective(w_plus, obs, objective="tv", epsilon=1e-5)
            l_m = halo_objective(w_minus, obs, objective="tv", epsilon=1e-5)
            fd_grad[k] = float(l_p - l_m) / (2.0 * h)
        np.testing.assert_allclose(grad_tree.z_ilr, fd_grad, rtol=1e-6, atol=1e-8)


def test_charbonnier_no_cancellation_and_downlink_gap_invariance() -> None:
    """Verify cancellation-free Charbonnier at |r| << eps and exact mid-sector gap invariance."""
    with jax.enable_x64(True):
        # At r = 1e-13 << eps = 1e-5, naive sqrt(r^2 + eps^2) - eps suffers total float64
        # cancellation to 0.0, whereas r^2 / (sqrt(r^2 + eps^2) + eps) equals r^2 / (2 * eps) = 5e-22.
        r_tiny = jnp.array([0.0, 1e-13, -1e-13], dtype=jnp.float64)
        phi_tiny = charbonnier_penalty(r_tiny, epsilon=1e-5)
        assert float(phi_tiny[0]) == 0.0
        np.testing.assert_allclose(phi_tiny[1:], 0.5e-21, rtol=1e-10)
        grad_zero = jax.grad(lambda x: jnp.sum(charbonnier_penalty(x, epsilon=1e-5)))(r_tiny)
        assert float(grad_zero[0]) == 0.0

    # Construct a 2-orbit TESS-like observation separated by a mid-sector downlink gap
    dt = 2.0 / 1440.0
    n_half = 40
    t_orb1 = np.arange(n_half, dtype=np.float64) * dt
    t_gap_short = np.concatenate([t_orb1, 1.5 + t_orb1])
    t_gap_long = np.concatenate([t_orb1, 45.0 + t_orb1])

    # Pixels 0..1 have a large thermal step across the downlink gap (constant within each orbit),
    # while Pixels 2..3 have intra-orbit thruster sawtooth jitter on every cadence (M=4 shares XLA cache).
    step_across_gap = np.concatenate([np.full(n_half, 950.0), np.full(n_half, 1050.0)])
    sawtooth = 1000.0 + 60.0 * ((-1.0) ** np.arange(2 * n_half))
    pixels = np.vstack([step_across_gap, 1.1 * step_across_gap, sawtooth, 0.9 * sawtooth])

    obs_short = HaloObservation.from_arrays(pixels, t_gap_short, lag=1)
    obs_long = HaloObservation.from_arrays(pixels, t_gap_long, lag=1)

    # Index n_half - 1 (bridging orbit 1 -> orbit 2) MUST be False in valid_diff_mask
    assert bool(obs_short.valid_diff_mask[n_half - 1]) is False
    assert bool(obs_short.valid_diff_mask[n_half - 2]) is True
    assert bool(obs_short.valid_diff_mask[n_half]) is True

    w_probe = HaloWeights.from_uniform(obs_short.mean_pixel_flux)
    with jax.enable_x64(True):
        for obj_name in AVAILABLE_OBJECTIVES:
            val_short = float(halo_objective(w_probe, obs_short, objective=obj_name))
            val_long = float(halo_objective(w_probe, obs_long, objective=obj_name))
            assert val_short == pytest.approx(val_long, rel=1e-13, abs=1e-13)

    # Optimizer must select Pixels 0..1 (zero intra-orbit TV despite the downlink step!)
    fit = optimise_halo(pixels, t_gap_short, lambda_phot=0.0, lambda_z=1e-5, maxiter=151)
    assert float(fit.physical_weights[0] + fit.physical_weights[1]) > 0.995


def test_tv_exp_scan_matches_reference_and_scales_linearly() -> None:
    """Verify O(N) lax.scan tv_exp matches an explicit multi-lag loop and runs fast at N=16,384."""
    rng = np.random.default_rng(314)
    dt = 2.0 / 1440.0
    t = np.concatenate([np.arange(35) * dt, 2.0 + np.arange(35) * dt])
    pixels = 1000.0 + rng.normal(0.0, 8.0, size=(4, 70))
    tau = 3.5
    eps = 1e-5

    obs = HaloObservation.from_arrays(pixels, t, lag=1)
    w = HaloWeights.from_uniform(obs.mean_pixel_flux)

    with jax.enable_x64(True):
        f_tilde = w.evaluate_lightcurve(obs)
        scan_val, scan_count = tv_exp_scan(f_tilde, obs, tau=tau, epsilon=eps)

        # Explicit NumPy reference over lags ell = 1 .. ceil(4 * tau)
        f_np = np.asarray(f_tilde, dtype=np.float64)
        t_np = np.asarray(obs.times_padded, dtype=np.float64)
        vc_np = np.asarray(obs.valid_cadence_mask, dtype=bool)
        dt_nom_np = float(obs.dt_nom)
        l_max = min(int(math.ceil(4.0 * tau)), int(obs.n_valid) - 1)
        num_ref = 0.0
        den_ref = 0.0
        for ell in range(1, l_max + 1):
            dt_p = t_np[ell:] - t_np[:-ell]
            exp_dt = ell * dt_nom_np
            mask_ell = vc_np[:-ell] & vc_np[ell:] & (np.abs(dt_p - exp_dt) <= 0.25 * exp_dt)
            diffs = (f_np[ell:] - f_np[:-ell])[mask_ell]
            phi = (diffs * diffs) / (np.sqrt(diffs * diffs + eps * eps) + eps)
            w_ell = math.exp(-ell / tau)
            num_ref += w_ell * float(np.mean(phi))
            den_ref += w_ell
        ref_val = num_ref / den_ref

        assert float(scan_val) == pytest.approx(ref_val, rel=1e-12, abs=1e-14)
        assert float(scan_count) == float(jnp.sum(obs.valid_diff_mask))

        # Large-N linear-memory smoke test (N = 8,192 cadences: N x N float64 would be 537 MB!)
        n_large = 8192
        t_large = np.arange(n_large, dtype=np.float64) * (20.0 / 86400.0)
        p_large = np.vstack(
            [
                np.full(n_large, 1000.0),
                1000.0 + 30.0 * np.sin(0.2 * np.arange(n_large)),
            ]
        )
        obs_large = HaloObservation.from_arrays(p_large, t_large, lag=1)
        assert obs_large.N_pad == 8192
        vg_exp = build_value_and_grad("tv_exp", epsilon=1e-5, tau=3.0)
        w_large = HaloWeights.from_uniform(obs_large.mean_pixel_flux)
        val_l, grad_l = vg_exp(w_large, obs_large)
        assert np.isfinite(float(val_l))
        assert np.all(np.isfinite(np.asarray(grad_l.z_ilr)))


def test_solvers_safe_eigen_projection_jackknife_sigclip_and_legacy_tv() -> None:
    """Verify Tier-1 L-BFGS, Tier-2 zodiax solver, safe_eigen_projection, jackknife, and legacy_tv."""
    # 1. safe_eigen_projection on rank-deficient, zero-diagonal, and negative-eigenvalue matrices
    rank_def = np.array(
        [
            [4.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],  # Exact rank deficiency (lambda = 0)
            [0.0, 0.0, 0.0, 0.0],  # Exact zero diagonal
            [0.0, 0.0, 0.0, -1e-4],  # Negative numerical curvature
        ],
        dtype=np.float64,
    )
    proj_f = np.asarray(safe_eigen_projection(fmat=rank_def))
    proj_c = np.asarray(safe_eigen_projection(cov=rank_def))
    assert proj_f.shape == (4, 4) and np.all(np.isfinite(proj_f))
    assert proj_c.shape == (4, 4) and np.all(np.isfinite(proj_c))

    spd = np.array([[5.0, 1.2, -0.4], [1.2, 3.0, 0.5], [-0.4, 0.5, 2.0]], dtype=np.float64)
    p_spd = np.asarray(safe_eigen_projection(fmat=spd))
    np.testing.assert_allclose(p_spd.T @ spd @ p_spd, np.eye(3), atol=1e-10)

    # 2. Cadence-scaled Savitzky-Golay window sizing
    assert cadence_scaled_savgol_window(1.0 / 48.0, window_days=1.0) == 49
    assert cadence_scaled_savgol_window(2.0 / 1440.0, window_days=1.0) == 721
    assert cadence_scaled_savgol_window(20.0 / 86400.0, window_days=1.0) == 4321
    assert bucket_pad_length(75) == 128

    # 3. Synthetic complementary jitter + pulsation recovery across L-BFGS, Zodiax, sigclip, jackknife
    rng = np.random.default_rng(99)
    n_cad = 96
    dt = 1.0 / 48.0
    t = np.arange(n_cad, dtype=np.float64) * dt
    stellar = 1.0 + 0.02 * np.sin(2.0 * np.pi * t / 0.5)
    jitter = 0.15 * ((np.arange(n_cad) % 11) / 10.0 - 0.5)

    # 4 pixels with asymmetric complementary jitter sensitivities [-1.0, 0.4, -0.2, 1.2]
    # so uniform weights w = [0.25, 0.25, 0.25, 0.25] retain strong residual jitter
    means = np.array([2000.0, 2000.0, 800.0, 800.0])
    sens = np.array([-1.0, 0.4, -0.2, 1.2])
    pixels = means[:, None] * stellar[None, :] * (1.0 + sens[:, None] * jitter[None, :])
    pixels += rng.normal(0.0, 0.4, size=pixels.shape)

    # Inject a positive cosmic-ray outlier on cadence 33 to test cadence-scaled sigclip
    pixels_spiked = pixels.copy()
    pixels_spiked[:, 33] += 400.0

    fit_lbfgs = optimise_halo(
        pixels_spiked,
        t,
        solver="lbfgs",
        maxiter=151,
        sigclip=True,
        compute_uncertainty=True,
        jackknife_blocks=4,
    )
    assert jax.config.jax_enable_x64 is False
    assert bool(fit_lbfgs.inlier_mask[33]) is False
    assert int(np.sum(fit_lbfgs.inlier_mask)) >= n_cad - 2
    inliers = fit_lbfgs.inlier_mask
    np.testing.assert_allclose(
        fit_lbfgs.norm_flux[inliers],
        stellar[inliers] / np.mean(stellar[inliers]),
        atol=3e-3,
    )
    assert fit_lbfgs.weight_std is not None and fit_lbfgs.jackknife_weights is not None
    assert fit_lbfgs.weight_std.shape == (4,)
    assert np.all(np.isfinite(fit_lbfgs.weight_std)) and np.all(fit_lbfgs.weight_std >= 0.0)
    # Simplex sum invariance: sum_i w_i^{(-k)} == 1 across every jackknife fold
    np.testing.assert_allclose(np.sum(fit_lbfgs.jackknife_weights, axis=1), 1.0, atol=1e-12)

    # Tier-2 Zodiax preconditioned Adam solver must also suppress jitter and converge cleanly
    fit_zdx = optimise_halo(
        pixels,
        t,
        solver="zodiax",
        maxiter=151,
        learning_rate=0.2,
        precondition=True,
    )
    assert fit_zdx.loss_final < fit_zdx.loss_init
    np.testing.assert_allclose(fit_zdx.norm_flux, stellar / np.mean(stellar), atol=4e-3)

    # legacy_tv=True must match legacy unregularized w-simplex tv_tpf light curve (on M=4 with
    # 2-D jitter-cancelling nullspace) and exact unique weights [2/7, 5/7] on M=2!
    fit_legacy = optimise_halo(pixels, t, legacy_tv=True, epsilon=1e-5, maxiter=151)
    _, lc_legacy_ref, _ = tv_tpf(pixels, maxiter=100, verbose=False, objective="tv")
    np.testing.assert_allclose(
        fit_legacy.norm_flux,
        lc_legacy_ref / np.mean(lc_legacy_ref),
        atol=2e-3,
    )
    fit_legacy_2 = optimise_halo(pixels[:2], t, legacy_tv=True, epsilon=1e-5, maxiter=151)
    w_legacy_2, _, _ = tv_tpf(pixels[:2], maxiter=100, verbose=False, objective="tv")
    np.testing.assert_allclose(fit_legacy_2.physical_weights, w_legacy_2, atol=2e-3)

    # Verify orbit_block_jackknife reuses static N_pad across subsets
    obs_base = HaloObservation.from_arrays(pixels, t)
    sub_obs = obs_base.with_cadence_subset(np.arange(n_cad) >= 24)
    assert sub_obs.N_pad == obs_base.N_pad
    w_std, w_jk = orbit_block_jackknife(obs_base, fit_lbfgs.physical_weights, n_blocks=4)
    assert w_std.shape == (4,) and w_jk.shape == (4, 4)


def test_gap_invariance_when_lag_exceeds_segment_length() -> None:
    """Verify tv_exp_scan, tv_charbonnier_o2, and from_arrays never bridge cross-gap jumps."""
    n_half = 16
    dt = 2.0 / 1440.0
    t_seg1 = np.arange(n_half, dtype=np.float64) * dt
    t_seg2 = t_seg1 + 2.5  # 2.5-day TESS downlink gap
    t_gapped = np.concatenate([t_seg1, t_seg2])

    rng = np.random.default_rng(707)
    p_base = 1000.0 + rng.normal(0.0, 5.0, size=(3, 2 * n_half))
    p_shifted = p_base.copy()
    # Add a huge cross-gap step on segment 2 that leaves intra-segment differences unchanged
    p_shifted[:, n_half:] += 500.0

    # 1. lag = 18 > n_half = 16 must leave valid_diff_mask empty (never falling back across gap!)
    obs_lag18 = HaloObservation.from_arrays(p_base, t_gapped, lag=18)
    assert int(jnp.sum(obs_lag18.valid_diff_mask)) == 0

    # 2. tv_exp_scan with tau = 6.0 -> scan_len = ceil(4 * tau) = 24 > n_half = 16
    obs_a = HaloObservation.from_arrays(p_base, t_gapped, lag=1, legacy_tv=True)
    obs_b = HaloObservation.from_arrays(p_shifted, t_gapped, lag=1, legacy_tv=True)

    with jax.enable_x64(True):
        w_fixed = HaloWeights(z_ilr=jnp.array([0.2, -0.3], dtype=jnp.float64), legacy_tv=True)
        # Evaluate raw (unnormalized) linear combination w^T P(t) directly so intra-segment
        # differences match identically while the +500 cross-gap jump is present in f_b!
        w_phys = w_fixed.physical_weights(jnp.ones(3, dtype=jnp.float64))
        f_a = jnp.asarray(p_base.T, dtype=jnp.float64) @ w_phys
        f_b = jnp.asarray(p_shifted.T, dtype=jnp.float64) @ w_phys
        f_a_pad = jnp.pad(f_a, (0, obs_a.N_pad - f_a.shape[0]))
        f_b_pad = jnp.pad(f_b, (0, obs_b.N_pad - f_b.shape[0]))

        val_exp_a, _ = tv_exp_scan(f_a_pad, obs_a, tau=6.0, epsilon=1e-5)
        val_exp_b, _ = tv_exp_scan(f_b_pad, obs_b, tau=6.0, epsilon=1e-5)
        assert float(val_exp_a) == pytest.approx(float(val_exp_b), rel=1e-12, abs=1e-13)

        # 3. tv_charbonnier_o2 with obs.lag = 10 -> 2 * lag = 20 > n_half = 16 (zero valid triples)
        obs_o2 = HaloObservation.from_arrays(p_shifted, t_gapped, lag=10, legacy_tv=True)
        val_o2, count_o2 = tv_charbonnier_o2(f_b_pad, obs_o2, epsilon=1e-5)
        assert float(count_o2) == 0.0
        assert float(val_o2) == 0.0


def test_lambda_phot_subset_zero_mean_legacy_and_sigclip_jackknife_fixes() -> None:
    """Verify explicit lambda_phot persistence, zero-mean legacy_tv, NaN-safe sigclip, and JK guards."""
    # Choose n_cad = 72 (N_pad = 128) and M = 4 so solver calls reuse already-compiled XLA kernels!
    n_cad = 72
    dt = 2.0 / 1440.0
    t = np.arange(n_cad, dtype=np.float64) * dt
    p_pos = np.vstack(
        [
            1000.0 + 15.0 * np.sin(0.3 * np.arange(n_cad)),
            900.0 - 15.0 * np.sin(0.3 * np.arange(n_cad)),
            950.0 + 8.0 * np.cos(0.3 * np.arange(n_cad)),
            850.0 - 8.0 * np.cos(0.3 * np.arange(n_cad)),
        ]
    )
    pvar = np.full_like(p_pos, 25.0)

    # 1. Explicit lambda_phot > 0 must survive with_cadence_subset verbatim
    obs_explicit = HaloObservation.from_arrays(p_pos, t, pixel_var=pvar, lambda_phot=17.5)
    assert obs_explicit.lambda_phot == pytest.approx(17.5)
    assert obs_explicit.user_lambda_phot == pytest.approx(17.5)
    obs_sub = obs_explicit.with_cadence_subset(np.arange(n_cad) >= 16)
    assert obs_sub.lambda_phot == pytest.approx(17.5)
    assert obs_sub.user_lambda_phot == pytest.approx(17.5)

    # 2. Zero-mean pixel under legacy_tv=True must NOT be annihilated in evaluate_lightcurve,
    #    with_cadence_subset, or _get_fisher_fn
    zero_mean_wave = 25.0 * np.sin(2.0 * np.pi * np.arange(n_cad) / 18.0)
    assert abs(float(np.mean(zero_mean_wave))) < 1e-13
    p_zero = np.vstack([zero_mean_wave, 500.0 + 0.5 * zero_mean_wave])
    obs_zm = HaloObservation.from_arrays(p_zero, t, legacy_tv=True)
    with jax.enable_x64(True):
        w_zm = HaloWeights(z_ilr=jnp.zeros(1, dtype=jnp.float64), legacy_tv=True)
        lc_zm = np.asarray(w_zm.evaluate_lightcurve(obs_zm))[:n_cad]
        raw_zm = 0.5 * p_zero[0] + 0.5 * p_zero[1]
        expected_zm = raw_zm / np.mean(raw_zm)
        np.testing.assert_allclose(lc_zm, expected_zm, atol=1e-11)

        obs_zm_sub = obs_zm.with_cadence_subset(np.arange(n_cad) < 54)
        lc_zm_sub = np.asarray(w_zm.evaluate_lightcurve(obs_zm_sub))[:54]
        expected_sub = raw_zm[:54] / np.mean(raw_zm[:54])
        np.testing.assert_allclose(lc_zm_sub, expected_sub, atol=1e-11)

        fmat_zm = np.asarray(_get_fisher_fn("tv", 1)(w_zm, obs_zm))
        assert fmat_zm.shape == (1, 1) and float(fmat_zm[0, 0]) > 0.0

    # 3. cadence_scaled_sigclip_mask must NOT spread an isolated NaN across Savitzky-Golay windows
    #    and must clip cosmic-ray spikes via mad_std (including N = 5!)
    y_nan = 1.0 + 0.005 * np.sin(0.2 * np.arange(n_cad))
    y_nan[20] = np.nan
    y_nan[45] = 2.5  # Cosmic-ray spike
    inliers_nan = cadence_scaled_sigclip_mask(y_nan, t, window_days=1.0)
    assert not bool(inliers_nan[20])
    assert not bool(inliers_nan[45])
    assert int(np.sum(inliers_nan)) == n_cad - 2

    y_five = np.array([1.0, 1.001, 0.999, 1.0, 5.0], dtype=np.float64)
    t_five = np.arange(5, dtype=np.float64) * (1.0 / 48.0)
    inliers_five = cadence_scaled_sigclip_mask(y_five, t_five, window_days=0.1)
    assert list(inliers_five) == [True, True, True, True, False]

    # 4. sigclip=True objective consistency (loss_final <= loss_init on same active_obs)
    p_spike = p_pos.copy()
    p_spike[:, 30] += 800.0
    fit_sc = optimise_halo(
        p_spike, t, pixel_var=pvar, lambda_phot=17.5, maxiter=151, sigclip=True
    )
    assert not bool(fit_sc.inlier_mask[30])
    assert fit_sc.loss_final <= fit_sc.loss_init + 1e-12

    # 5. orbit_block_jackknife honours solver="zodiax" and returns NaN when < 2 folds succeed
    fit_zdx_jk = optimise_halo(
        p_pos,
        t,
        solver="zodiax",
        maxiter=151,
        learning_rate=0.2,
        precondition=True,
        compute_uncertainty=True,
        jackknife_blocks=3,
    )
    assert fit_zdx_jk.weight_std is not None
    assert np.all(np.isfinite(fit_zdx_jk.weight_std))

    obs_tiny = HaloObservation.from_arrays(p_pos[:, :3], t[:3])
    w_std_skip, w_jk_skip = orbit_block_jackknife(
        obs_tiny, np.full(4, 0.25), n_blocks=4
    )
    assert np.all(np.isnan(w_std_skip))
    assert np.all(np.isnan(w_jk_skip))

    nb_path = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "aldebaran.ipynb"
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert "extract_halo_lightcurve" in "".join(nb_data["cells"][0]["source"])


def test_lambda_phot_aperture_floor_invariance_and_m_over_n_guard() -> None:
    """Verify [HALO-03]: aperture-floor M-invariance, M >= 0.5*|V_ell| warning, and short-segment pull calibration."""
    with jax.enable_x64(True):
        # 1. Splitting constant total flux/variance across M=4 vs M=36 pixels keeps
        # photon_noise_penalty invariant up to dof_scale (no 1/M under-regularization).
        n_quiet = 200
        t_quiet = np.arange(n_quiet, dtype=np.float64) * 0.02
        penalties_over_dof: dict[int, float] = {}
        raw_penalties: dict[int, float] = {}
        for m_pix in (4, 36):
            p_m = np.full((m_pix, n_quiet), 3600.0 / m_pix, dtype=np.float64)
            pvar_m = np.full((m_pix, n_quiet), 36.0 / m_pix, dtype=np.float64)
            obs_m = HaloObservation.from_arrays(p_m, t_quiet, pixel_var=pvar_m)
            theta_init_m = jnp.full((m_pix,), 1.0 / m_pix, dtype=jnp.float64)
            pen_m = float(photon_noise_penalty(theta_init_m, obs_m))
            r_m = float(m_pix - 1) / float(n_quiet - 1)
            dof_scale_m = 1.0 + 16.0 * (r_m**2) + 160.0 * (r_m**4)
            raw_penalties[m_pix] = pen_m
            penalties_over_dof[m_pix] = pen_m / dof_scale_m

        assert raw_penalties[36] > raw_penalties[4]
        assert penalties_over_dof[4] == pytest.approx(0.025 * (n_quiet - 1), rel=1e-9)
        assert penalties_over_dof[36] == pytest.approx(0.025 * (n_quiet - 1), rel=1e-9)

        # 2. UserWarning emitted iff M >= 0.5 * |V_ell| (for M > 1).
        rng_warn = np.random.default_rng(20261114)
        with pytest.warns(UserWarning, match="elevated degrees of freedom"):
            optimise_halo(
                rng_warn.uniform(80.0, 120.0, size=(20, 36)),
                np.arange(36, dtype=np.float64) * 0.02,
                maxiter=25,
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimise_halo(
                rng_warn.uniform(80.0, 120.0, size=(8, 60)),
                np.arange(60, dtype=np.float64) * 0.02,
                maxiter=25,
            )
        assert not any("elevated degrees of freedom" in str(w.message) for w in caught)

        # 3. Short segment (N=36, M=45) with pulsation + transit + shot/sky noise:
        # auto-calibrated dof_scale prevents simplex white-noise interpolation.
        n_short, m_large = 36, 45
        t_short = np.arange(n_short, dtype=np.float64) * (30.0 / 1440.0)
        s_true = (
            1.0
            + 0.008 * np.sin(2.0 * np.pi * t_short / 0.25)
            - 0.012 * ((t_short > 0.30) & (t_short < 0.42)).astype(np.float64)
        )

        guarded_pulls: list[float] = []
        unguarded_pulls: list[float] = []
        for seed in range(8):
            rng = np.random.default_rng(9100 + seed)
            f0 = rng.uniform(80.0, 300.0, size=(m_large, 1))
            pvar = np.broadcast_to(f0 + 25.0, (m_large, n_short)).copy()
            p_mn = f0 * s_true[None, :] + rng.normal(0.0, np.sqrt(pvar))

            with pytest.warns(UserWarning, match="elevated degrees of freedom"):
                fit_g = optimise_halo(p_mn, t_short, pixel_var=pvar, maxiter=151)
                fit_u = optimise_halo(
                    p_mn,
                    t_short,
                    pixel_var=pvar,
                    lambda_phot=0.0,
                    lambda_z=1e-3,
                    maxiter=151,
                )

            for fit_res, bucket in ((fit_g, guarded_pulls), (fit_u, unguarded_pulls)):
                w_vec = fit_res.physical_weights
                raw_true = float(w_vec @ f0[:, 0]) * s_true
                raw_sig = np.sqrt((w_vec**2) @ pvar)
                pull = (fit_res.raw_flux - raw_true) / raw_sig
                bucket.append(float(np.std(pull, ddof=1)))

        assert float(np.mean(guarded_pulls)) >= 0.90
        assert float(np.mean(unguarded_pulls)) < 0.72
