"""Deterministic mean-field (N -> infinity) reduction of the block model.

The stochastic agent model of :mod:`credibility.model` admits an exact
large-population limit on balanced community blocks: within a community all
agents share the same expected value estimate, choices are replaced by the exact
DDM probability, decision times by the mean first-passage time, and rewards by
their Bernoulli means.  The state collapses to the community value table
``Q[c, k]`` (M communities x 2 arms).

This module provides the resulting deterministic map ``F: Q -> Q'`` (Eq. update
in the manuscript, in expectation), together with helpers for the bifurcation
diagram, the phase portrait / vector field, and a noise-free phase heatmap.
Everything here is analytic and cheap; it is the dynamical-systems companion to
the Monte-Carlo simulator and reproduces the same corrective/distortive regimes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import ddm


# --------------------------------------------------------------------------
def _self_consistent_state(Q, p, lam, B, iters=80, damp=0.5):
    """Solve the stationary augmented masses/confidence for a value table Q.

    Returns ``m`` (M,2) choice masses, ``C`` (M,) community confidence, and
    ``V`` (M,2) socially augmented values.  The anticipatory field couples the
    communities, so masses satisfy a small fixed-point solved by damped
    iteration.
    """
    M = Q.shape[0]
    a, s2, beta = p.a_thr, p.sigma ** 2, p.beta
    m = np.full((M, 2), 0.5)
    C = np.full(M, 0.5)
    for _ in range(iters):
        phi = m * C[:, None] if p.credibility_weight else m          # (M,2)
        S = B @ phi                                                  # (M,2)
        V = Q + lam * S
        Delta = V[:, 0] - V[:, 1]
        v = beta * Delta
        p0 = expit(2.0 * v * a / s2)                                 # P(arm 0)
        m_new = np.stack([p0, 1.0 - p0], axis=1)
        tau = ddm.mean_fpt(v, a, p.sigma)
        C_new = expit(p.kappa1 * np.abs(v) / a - p.kappa2 * np.log1p(tau / p.tau0))
        m = damp * m + (1 - damp) * m_new
        C = damp * C + (1 - damp) * C_new
    return m, C, V


def step(Q, p, lam=None):
    """One deterministic mean-field update ``F(Q)`` (in expectation)."""
    Q = np.asarray(Q, dtype=float)
    lam = p.lam if lam is None else lam
    B = p.coupling()
    mu = np.asarray(p.mu, dtype=float)
    m, C, _ = _self_consistent_state(Q, p, lam, B)
    span = p.alpha_max - p.alpha_min

    weight = B @ m                       # weight_c(k) = sum_d B_cd m_d(k)
    rmass = B @ (m * mu[None, :])        # rbar_d(k) = m_d(k) * mu_k
    cmass = B @ (m * C[:, None])         # confidence mass
    Qn = Q.copy()
    for k in (0, 1):
        a_pos = p.alpha_max - span * C   # good news (delta >= 0)
        a_neg = p.alpha_min + span * C   # bad news  (delta <  0)
        priv = m[:, k] * (mu[k] * a_pos * (1.0 - Q[:, k])
                          - (1.0 - mu[k]) * a_neg * Q[:, k])
        Cbar = cmass[:, k] / (weight[:, k] + p.eps_soc)
        soc_lr = p.eta * p.gamma * np.power(np.clip(Cbar, 0.0, 1.0), p.omega)
        soc = soc_lr * (rmass[:, k] - weight[:, k] * Q[:, k])
        Qn[:, k] = np.clip(Q[:, k] + priv + soc, 0.0, 1.0)
    return Qn


def run_to_fixed_point(Q0, p, lam=None, iters=1500, tol=1e-10):
    """Iterate the map to a fixed point; returns (Q*, converged)."""
    Q = np.asarray(Q0, dtype=float).copy()
    for _ in range(iters):
        Qn = step(Q, p, lam)
        if np.max(np.abs(Qn - Q)) < tol:
            return Qn, True
        Q = Qn
    return Q, False


def mass_on(Q, p, arm, lam=None):
    """Size-weighted population fraction choosing ``arm`` at value table Q."""
    B = p.coupling()
    m, _, _ = _self_consistent_state(Q, p, p.lam if lam is None else lam, B)
    sizes = np.asarray(p.sizes, float)
    return float((sizes * m[:, arm]).sum() / sizes.sum())


# --------------------------------------------------------------------------
def _ic(p, kind):
    """Two-community initial value tables for the canonical basins."""
    a = int(np.argmax(p.mu)); o = 1 - a
    hi, lo = 0.6, 0.4
    Q = np.full((p.M, 2), 0.5)
    if kind == "corrective":
        Q[:, a] = hi; Q[:, o] = lo
    elif kind == "distortive":
        Q[:, a] = lo; Q[:, o] = hi
    elif kind == "polarised":
        Q[0, a] = hi; Q[0, o] = lo
        Q[1, a] = lo; Q[1, o] = hi
    return Q


def lambda_star(p):
    """Symmetric-mode amplification threshold sigma^2 / (beta a C0 b_bar).

    The coupling B is row-stochastic, so for a *common* early lead the effective
    neighbour weight is the row sum b_bar = 1 (as in Proposition 3); permeability
    Gamma enters only the anti-symmetric/polarised modes.
    """
    tau0 = ddm.mean_fpt(np.array(0.0), p.a_thr, p.sigma)
    C0 = float(expit(p.kappa1 * 0.0 / p.a_thr - p.kappa2 * np.log1p(float(tau0) / p.tau0)))
    b_bar = 1.0
    return p.sigma ** 2 / (p.beta * p.a_thr * C0 * b_bar), C0


def infection_terminal(p, lams, z_seed=-0.05):
    """Asymmetric-seed terminal leads over lambda.

    Community 2 starts with a small *wrong* lead (z2<0), community 1 neutral.
    Returns an array (len(lams), 2) of terminal leads (z1*, z2*); both depend on
    the connectivity lambda and, through the off-diagonal coupling, on
    permeability Gamma.
    """
    B = p.coupling()
    out = np.empty((len(lams), 2))
    for i, lam in enumerate(lams):
        z = np.array([0.0, z_seed])
        for _ in range(800):
            zn = _amp_step(z, p, lam, B)
            if np.max(np.abs(zn - z)) < 1e-11:
                break
            z = zn
        out[i] = z
    return out


def infection(p, lams, z_seed=-0.05):
    """Community-1 terminal lead only (see :func:`infection_terminal`)."""
    return infection_terminal(p, lams, z_seed)[:, 0]


def infection_regime(p, lams, tol=0.05):
    """Classify the asymmetric-seed outcome into corrective / polarised /
    distortive over lambda.  Returns integer codes 0/1/2.
    """
    zt = infection_terminal(p, lams)
    z1, z2 = zt[:, 0], zt[:, 1]
    code = np.full(len(lams), 1, dtype=int)          # default polarised
    both_neutral = (np.abs(z1) < tol) & (np.abs(z2) < tol)
    both_wrong = (z1 < -tol) & (z2 < -tol)
    code[both_neutral] = 0                            # corrective (leads decay)
    code[both_wrong] = 2                              # distortive (both wrong)
    return code


# --------------------------------------------------------------------------
# Local amplification of an early lead (frozen-value analysis).
#
# The wrong-consensus regime is not a terminal attractor of the full map --
# truthful rewards eventually dominate -- but a *local amplification* of an
# early, confidence-weighted asymmetry, exactly the object of Proposition 3.
# Freeze values at the neutral (indifferent) point and let only the
# anticipatory-driven population lead z_c = m_c(0) - 1/2 evolve.  The
# self-consistent field a community feels from the leads {z_d} is
#   Delta_c = lambda * sum_d B_cd C_d (2 z_d),   v_c = beta Delta_c,
# with confidence C_d read from the same v_d.  z_c' = sigmoid(2 a v_c/sigma^2) - 1/2.
# Below lambda* only z=0 is stable; above, a symmetric pair +/- z* appears
# (pitchfork): +z* = corrective consensus, -z* = distortive consensus.
# --------------------------------------------------------------------------
def _amp_step(z, p, lam, B, citers=30):
    """One application of the frozen-value lead map z -> z'.

    Confidence is settled self-consistently at the *current* leads z, then a
    single population-lead update is applied.  A true one-step map, suitable for
    both iteration to fixed points and the vector field.
    """
    a, s2, beta = p.a_thr, p.sigma ** 2, p.beta
    z = np.asarray(z, float)
    C = np.full(z.shape, 0.5)
    for _ in range(citers):
        v = beta * lam * (B @ (C * 2.0 * z))
        C = expit(p.kappa1 * np.abs(v) / a - p.kappa2 * np.log1p(ddm.mean_fpt(v, a, p.sigma) / p.tau0))
    v = beta * lam * (B @ (C * 2.0 * z))
    return expit(2.0 * v * a / s2) - 0.5


def amplification_branch(p, lams, z0=0.02, iters=500):
    """Stable early-lead magnitude |z*| vs lambda (the pitchfork branch)."""
    B = p.coupling()
    zstar = np.empty(len(lams))
    for i, lam in enumerate(lams):
        z = np.full(p.M, z0)
        for _ in range(iters):
            zn = _amp_step(z, p, lam, B)
            if np.max(np.abs(zn - z)) < 1e-12:
                break
            z = zn
        zstar[i] = float(np.mean(np.abs(z)))
    return zstar


def _amp_step_batch(Z, p, lam, B, citers=25):
    """Vectorised frozen-value lead map over a batch Z of shape (n, M)."""
    a, s2, beta = p.a_thr, p.sigma ** 2, p.beta
    C = np.full(Z.shape, 0.5)
    for _ in range(citers):
        v = beta * lam * (2.0 * (Z * C) @ B.T)          # Delta_c per seed/community
        C = expit(p.kappa1 * np.abs(v) / a - p.kappa2 * np.log1p(ddm.mean_fpt(v, a, p.sigma) / p.tau0))
    v = beta * lam * (2.0 * (Z * C) @ B.T)
    return expit(2.0 * v * a / s2) - 0.5


def basin_probabilities(p, lams, n=240, bias=-0.06, noise=0.03, tol=0.05, seed=0):
    """Regime probabilities over a distribution of early leads.

    Finite populations seed a small, noisy early asymmetry.  For each lambda we
    draw ``n`` initial leads (community 2 biased toward the wrong arm, both
    communities perturbed by Gaussian noise), iterate the frozen-value map to its
    fixed point, and record the fraction ending corrective (no wrong lock-in),
    polarised (exactly one community wrong) or distortive (both wrong).  Returns
    an array (len(lams), 3) of probabilities [corrective, polarised, distortive].
    """
    rng = np.random.default_rng(seed)
    B = p.coupling()
    Z0 = np.empty((n, 2))
    Z0[:, 0] = noise * rng.standard_normal(n)
    Z0[:, 1] = bias + noise * rng.standard_normal(n)
    out = np.empty((len(lams), 3))
    for i, lam in enumerate(lams):
        Z = Z0.copy()
        for _ in range(400):
            Zn = _amp_step_batch(Z, p, lam, B)
            if np.max(np.abs(Zn - Z)) < 1e-9:
                break
            Z = Zn
        wrong = Z < -tol
        nwrong = wrong.sum(axis=1)                       # 0,1,2 communities wrong
        distort = np.mean(nwrong == 2)
        polar = np.mean(nwrong == 1)
        out[i] = [1.0 - distort - polar, polar, distort]
    return out


def basin_map(p, lam, gv, tol=0.05):
    """Basin of attraction over the early-lead plane at fixed lambda.

    Each point (z1, z2) on the grid ``gv x gv`` is iterated (frozen-value map) to
    its fixed point and labelled by the attractor reached: 0 neutral (both leads
    decay), 1 corrective (both lock onto the optimal arm), 2 distortive (both
    wrong), 3 polarised (opposite signs).  Returns an int array of shape
    (len(gv), len(gv)).
    """
    B = p.coupling()
    G1, G2 = np.meshgrid(gv, gv)
    Z = np.stack([G1.ravel(), G2.ravel()], axis=1)
    for _ in range(600):
        Zn = _amp_step_batch(Z, p, lam, B)
        if np.max(np.abs(Zn - Z)) < 1e-9:
            break
        Z = Zn
    z1 = Z[:, 0].reshape(G1.shape); z2 = Z[:, 1].reshape(G1.shape)
    code = np.zeros(z1.shape, dtype=int)
    code[(z1 > tol) & (z2 > tol)] = 1
    code[(z1 < -tol) & (z2 < -tol)] = 2
    code[((z1 > tol) & (z2 < -tol)) | ((z1 < -tol) & (z2 > tol))] = 3
    return code


def amplification_field(Z1, Z2, p, lam):
    """Increment (dz1, dz2) of the two-community lead map, for the vector field."""
    B = p.coupling()
    Z1, Z2 = np.broadcast_arrays(np.asarray(Z1, float), np.asarray(Z2, float))
    dZ1 = np.empty_like(Z1); dZ2 = np.empty_like(Z2)
    it = np.nditer(Z1, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        zn = _amp_step(np.array([Z1[idx], Z2[idx]]), p, lam, B)
        dZ1[idx] = zn[0] - Z1[idx]
        dZ2[idx] = zn[1] - Z2[idx]
    return dZ1, dZ2
