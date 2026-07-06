"""Core micro-dynamics and the quotient (meso) reduction.

Three engines, one model:

* :func:`simulate_blocks` -- the workhorse.  Under balanced block weights the
  social sums are *exactly* community aggregates, so the agent-level dynamics
  run in O(R*N) per trial instead of O(R*N^2).  This is an exact computation,
  not a mean-field approximation.

* :func:`simulate_dense` -- the same model on an arbitrary row-stochastic
  matrix ``W`` (O(R*N^2)).  Used for the stochastic-block robustness check and,
  in the tests, to confirm it reproduces :func:`simulate_blocks` on a balanced
  ``W``.

* :func:`simulate_meso` -- the deterministic community-level quotient model.
  Exact for the anticipatory social field; a moment-closure approximation for
  the stochastic value dynamics.  Stage 4 measures the micro--meso gap.

Timing within a trial follows the manuscript's simultaneous-update convention:
build the social signal from the previous period; form augmented values;
draw choice, decision time and confidence; draw rewards; compute *all*
prediction errors from the pre-update values; apply private and social updates
simultaneously.

Arms are indexed internally as 0 (the +a boundary, "arm 1") and 1 (the -a
boundary, "arm 2").
"""
from __future__ import annotations

import numpy as np

from . import ddm, confidence
from .config import Params
from .network import balanced_block_W, make_membership


# ---------------------------------------------------------------------------
# Shared helpers (also exercised directly by the test-suite)
# ---------------------------------------------------------------------------
def community_onehot(g, M):
    """(N, M) one-hot community-membership selector."""
    g = np.asarray(g)
    N = g.shape[0]
    C = np.zeros((N, M))
    C[np.arange(N), g] = 1.0
    return C


def community_means(x, Csel, sizes):
    """Per-community mean of an (R, N) array -> (R, M)."""
    return (x @ Csel) / sizes


def anticipatory_field_block(phi, B, g):
    """Exact agent social field from community masses.

    ``phi`` : (R, M, K) community (credibility-weighted) action masses.
    Returns ``S`` : (R, N, K) with ``S[:, i, k] = sum_d B[g_i, d] phi[:, d, k]``.
    """
    F = np.einsum("cd,rdk->rck", B, phi)        # (R, M, K)
    return F[:, g, :]                            # gather by membership -> (R, N, K)


def anticipatory_field_dense(cw, W):
    """Agent social field by explicit weighting: ``S[:, i, k]=sum_j W_ij cw[:,j,k]``."""
    # cw: (R, N, K) credibility-weighted action indicator; W: (N, N)
    return np.einsum("ij,rjk->rik", W, cw)


def _confidence_from_decision(p, params, v_abs, tau):
    if params.confidence_mode == "balance":
        return confidence.confidence_balance(p)
    return confidence.confidence_decision(v_abs, tau, params.a_thr,
                                          params.kappa1, params.kappa2, params.tau0)


def _private_lr(params, C, delta):
    if params.private_lr_mode == "constant":
        return np.full_like(C, params.alpha_const)
    span = params.alpha_max - params.alpha_min
    return np.where(delta < 0,
                    params.alpha_min + span * C,
                    params.alpha_max - span * C)


def _social_lr(params, Cbar):
    if params.social_lr_mode == "constant":
        return np.full_like(Cbar, params.gamma)
    return params.gamma * np.power(np.clip(Cbar, 0.0, 1.0), params.omega)


def value_update_block(Q, A, Rdraw, C, m_now, rbar_now, phi_now, B, g, bc_over_Nc, params):
    """Simultaneous private + retrospective-social value update (block form).

    All prediction errors are formed from the *pre-update* ``Q`` and applied
    together (the simultaneous-update convention).  ``m_now``/``rbar_now``/
    ``phi_now`` are current-trial community aggregates of the action mass,
    reward mass and confidence mass.
    """
    p = params
    R, N, _ = Q.shape
    arms = (0, 1)
    Q_chosen = np.take_along_axis(Q, A[:, :, None], axis=2)[:, :, 0]
    delta = Rdraw - Q_chosen
    alpha = _private_lr(p, C, delta)
    Sf = anticipatory_field_block(m_now, B, g)
    Rf = anticipatory_field_block(rbar_now, B, g)
    Cf = anticipatory_field_block(phi_now, B, g)
    inc = np.zeros((R, N, 2))
    for k in arms:
        self_ind = (A == k).astype(float)
        inc[:, :, k] += self_ind * alpha * delta                       # private
        weight_k = Sf[:, :, k] - bc_over_Nc * self_ind                 # exclude self
        rmass_k = Rf[:, :, k] - bc_over_Nc * self_ind * Rdraw
        cmass_k = Cf[:, :, k] - bc_over_Nc * self_ind * C
        delta_soc = rmass_k - weight_k * Q[:, :, k]
        Cbar_k = cmass_k / (weight_k + p.eps_soc)
        inc[:, :, k] += p.eta * _social_lr(p, Cbar_k) * delta_soc      # social
    return np.clip(Q + inc, 0.0, 1.0)


def value_update_dense(Q, A, Rdraw, C, W, params):
    """Simultaneous private + retrospective-social value update (dense form)."""
    p = params
    R, N, _ = Q.shape
    arms = (0, 1)
    Wns = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(Wns, 0.0)                                          # exclude self
    Q_chosen = np.take_along_axis(Q, A[:, :, None], axis=2)[:, :, 0]
    delta = Rdraw - Q_chosen
    alpha = _private_lr(p, C, delta)
    Ind = np.stack([(A == k).astype(float) for k in arms], axis=-1)
    inc = np.zeros((R, N, 2))
    for k in arms:
        inc[:, :, k] += Ind[:, :, k] * alpha * delta
        weight_k = np.einsum("ij,rj->ri", Wns, Ind[:, :, k])
        rmass_k = np.einsum("ij,rj->ri", Wns, Ind[:, :, k] * Rdraw)
        cmass_k = np.einsum("ij,rj->ri", Wns, Ind[:, :, k] * C)
        delta_soc = rmass_k - weight_k * Q[:, :, k]
        Cbar_k = cmass_k / (weight_k + p.eps_soc)
        inc[:, :, k] += p.eta * _social_lr(p, Cbar_k) * delta_soc
    return np.clip(Q + inc, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Workhorse: exact balanced-block agent simulator
# ---------------------------------------------------------------------------
def simulate_blocks(params: Params, n_reps: int, seed: int,
                    record_traj: bool = False, record_conf: bool = False):
    """Run ``n_reps`` independent replications of the agent model.

    Returns a dict with at least ``term_m`` (R, M, 2) terminal action
    frequencies and ``regret`` (R,).  Optional trajectory and confidence
    records are added when requested.
    """
    rng = np.random.default_rng(seed)
    p = params
    sizes = np.asarray(p.sizes, dtype=int)
    M, N = p.M, p.N
    g = make_membership(sizes)
    Csel = community_onehot(g, M)
    B = p.coupling()
    bc_over_Nc = (np.diag(B) / sizes)[g]          # per-agent self-weight W_ii  (N,)
    mu = np.asarray(p.mu, dtype=float)
    a_star = int(np.argmax(mu))
    mu_star = mu.max()
    lam = p.lam
    eta = p.eta

    R = n_reps
    # value estimates Q[r, i, arm]
    q0 = p.q_init_array()                          # (M, 2)
    Q = np.repeat(q0[g][None, :, :], R, axis=0).copy()   # (R, N, 2)

    A_prev = np.full((R, N), -1, dtype=int)        # no previous action at t=0
    C_prev = np.zeros((R, N))

    regret = np.zeros(R)
    if record_traj:
        traj_m = np.zeros((R, p.T + 1, M, 2))
        traj_phi = np.zeros((R, p.T + 1, M, 2))
        traj_Q = np.zeros((R, p.T + 1, M, 2))
    if record_conf:
        conf_C, conf_correct, conf_phase = [], [], []

    half_T = p.T // 2
    arms = (0, 1)

    for t in range(p.T):
        # 1. anticipatory social signal from previous period -----------------
        if t == 0:
            S = np.zeros((R, N, 2))
        else:
            cw_prev = np.stack(
                [(A_prev == k).astype(float) * (C_prev if p.credibility_weight else 1.0)
                 for k in arms], axis=-1)                       # (R, N, 2)
            phi_prev = np.stack([community_means(cw_prev[:, :, k], Csel, sizes)
                                 for k in arms], axis=-1)        # (R, M, 2)
            S = anticipatory_field_block(phi_prev, B, g)         # (R, N, 2)

        # 2-3. augmented values, contrast, drift -----------------------------
        V = Q + lam * S
        Delta = V[:, :, 0] - V[:, :, 1]
        v = p.beta * Delta

        # 4. choice, decision time, confidence -------------------------------
        A, p_chosen, p_up = ddm.simulate_choice(v, p.a_thr, p.sigma, rng)
        A = np.where(A == 1, 0, 1)                                # map {1,2}->{0,1}
        if p.rt_mode == "mean":
            tau = ddm.mean_fpt(v, p.a_thr, p.sigma)               # deterministic RT
        else:
            tau = ddm.sample_decision_time(v, p.a_thr, p.sigma, rng, p.rt_dispersion)
        C = _confidence_from_decision(p_chosen, p, np.abs(v), tau)

        # 5. rewards ---------------------------------------------------------
        mu_A = mu[A]
        Rdraw = (rng.random((R, N)) < mu_A).astype(float)

        # community aggregates of the CURRENT trial --------------------------
        Ind = np.stack([(A == k).astype(float) for k in arms], axis=-1)   # (R,N,2)
        m_now = np.stack([community_means(Ind[:, :, k], Csel, sizes) for k in arms], axis=-1)
        rbar_now = np.stack([community_means(Ind[:, :, k] * Rdraw, Csel, sizes) for k in arms], axis=-1)
        phi_now = np.stack([community_means(Ind[:, :, k] * C, Csel, sizes) for k in arms], axis=-1)

        # 6-8. simultaneous private + retrospective-social value update ------
        Q = value_update_block(Q, A, Rdraw, C, m_now, rbar_now, phi_now,
                               B, g, bc_over_Nc, p)

        # 9. bookkeeping ------------------------------------------------------
        regret += (mu_star - mu[A]).sum(axis=1)
        if record_traj:
            traj_m[:, t + 1] = m_now
            traj_phi[:, t + 1] = phi_now
            traj_Q[:, t + 1] = np.stack(
                [community_means(Q[:, :, k], Csel, sizes) for k in arms], axis=-1)
        if record_conf:
            correct = (A == a_star)
            phase = np.zeros((R, N), dtype=int) if t < half_T else np.ones((R, N), dtype=int)
            conf_C.append(C.ravel())
            conf_correct.append(correct.ravel())
            conf_phase.append(phase.ravel())

        A_prev, C_prev = A, C

    term_m = m_now                                  # (R, M, 2) terminal frequencies
    out = dict(term_m=term_m, regret=regret, a_star=a_star, mu_star=mu_star)
    if record_traj:
        traj_m[:, 0] = traj_m[:, 1]                  # pad t=0 for plotting
        traj_phi[:, 0] = 0.0
        traj_Q[:, 0] = np.repeat(q0[None], R, axis=0)
        out.update(traj_m=traj_m, traj_phi=traj_phi, traj_Q=traj_Q)
    if record_conf:
        out.update(conf_C=np.concatenate(conf_C),
                   conf_correct=np.concatenate(conf_correct),
                   conf_phase=np.concatenate(conf_phase))
    return out


# ---------------------------------------------------------------------------
# Dense engine on an arbitrary W (reference / robustness)
# ---------------------------------------------------------------------------
def simulate_dense(W, params: Params, n_reps: int, seed: int, record_traj: bool = False):
    """Same model on an explicit row-stochastic matrix ``W`` (O(R N^2))."""
    rng = np.random.default_rng(seed)
    p = params
    sizes = np.asarray(p.sizes, dtype=int)
    M, N = p.M, p.N
    g = make_membership(sizes)
    Csel = community_onehot(g, M)
    W = np.asarray(W, dtype=float)
    Wns = W.copy()
    np.fill_diagonal(Wns, 0.0)                       # retrospective excludes self
    mu = np.asarray(p.mu, dtype=float)
    a_star = int(np.argmax(mu)); mu_star = mu.max()
    R = n_reps
    q0 = p.q_init_array()
    Q = np.repeat(q0[g][None, :, :], R, axis=0).copy()
    A_prev = np.full((R, N), -1, dtype=int)
    C_prev = np.zeros((R, N))
    regret = np.zeros(R)
    if record_traj:
        traj_m = np.zeros((R, p.T + 1, M, 2))
    arms = (0, 1)
    for t in range(p.T):
        if t == 0:
            S = np.zeros((R, N, 2))
        else:
            cw_prev = np.stack(
                [(A_prev == k).astype(float) * (C_prev if p.credibility_weight else 1.0)
                 for k in arms], axis=-1)
            S = anticipatory_field_dense(cw_prev, W)
        V = Q + p.lam * S
        v = p.beta * (V[:, :, 0] - V[:, :, 1])
        A, p_chosen, _ = ddm.simulate_choice(v, p.a_thr, p.sigma, rng)
        A = np.where(A == 1, 0, 1)
        tau = ddm.sample_decision_time(v, p.a_thr, p.sigma, rng, p.rt_dispersion)
        C = _confidence_from_decision(p_chosen, p, np.abs(v), tau)
        Rdraw = (rng.random((R, N)) < mu[A]).astype(float)
        Ind = np.stack([(A == k).astype(float) for k in arms], axis=-1)
        Q = value_update_dense(Q, A, Rdraw, C, W, p)
        regret += (mu_star - mu[A]).sum(axis=1)
        if record_traj:
            traj_m[:, t + 1] = np.stack(
                [community_means(Ind[:, :, k], Csel, sizes) for k in arms], axis=-1)
        A_prev, C_prev = A, C
    m_now = np.stack([community_means(Ind[:, :, k], Csel, sizes) for k in arms], axis=-1)
    out = dict(term_m=m_now, regret=regret, a_star=a_star, mu_star=mu_star)
    if record_traj:
        out.update(traj_m=traj_m)
    return out


# ---------------------------------------------------------------------------
# Deterministic meso / quotient model
# ---------------------------------------------------------------------------
def simulate_meso(params: Params, record_traj: bool = True):
    """Deterministic community-level recursion (the quotient approximation)."""
    p = params
    M = p.M
    B = p.coupling()
    mu = np.asarray(p.mu, dtype=float)
    a_star = int(np.argmax(mu)); mu_star = mu.max()
    Qbar = p.q_init_array().copy()                  # (M, 2)
    phi_prev = np.zeros((M, 2))
    if record_traj:
        traj_m = np.zeros((p.T + 1, M, 2))
        traj_Q = np.zeros((p.T + 1, M, 2)); traj_Q[0] = Qbar
    arms = (0, 1)
    m_c = np.zeros((M, 2))
    for t in range(p.T):
        S = np.einsum("cd,dk->ck", B, phi_prev) if t > 0 else np.zeros((M, 2))
        V = Qbar + p.lam * S
        v = p.beta * (V[:, 0] - V[:, 1])
        m0 = ddm.choice_prob_up(v, p.a_thr, p.sigma)
        m_c = np.stack([m0, 1.0 - m0], axis=-1)                  # (M, 2)
        tau = ddm.mean_fpt(v, p.a_thr, p.sigma)
        p_chosen = np.maximum(m0, 1.0 - m0)
        Cc = _confidence_from_decision(p_chosen, p, np.abs(v), tau)   # (M,)
        phi_c = m_c * Cc[:, None]
        # private update (expected)
        inc = np.zeros((M, 2))
        for k in arms:
            d = mu[k] - Qbar[:, k]
            alpha = _private_lr(p, Cc, d)
            inc[:, k] += m_c[:, k] * alpha * d
        # retrospective social update (expected, exclude-self negligible at meso)
        weight = np.einsum("cd,dk->ck", B, m_c)
        cmass = np.einsum("cd,dk->ck", B, phi_c)
        for k in arms:
            delta_soc = weight[:, k] * (mu[k] - Qbar[:, k])
            Cbar = cmass[:, k] / (weight[:, k] + p.eps_soc)
            inc[:, k] += p.eta * _social_lr(p, Cbar) * delta_soc
        Qbar = np.clip(Qbar + inc, 0.0, 1.0)
        phi_prev = phi_c
        if record_traj:
            traj_m[t + 1] = m_c
            traj_Q[t + 1] = Qbar
    out = dict(term_m=m_c, a_star=a_star, mu_star=mu_star)
    if record_traj:
        traj_m[0] = traj_m[1]
        out.update(traj_m=traj_m, traj_Q=traj_Q)
    return out
