"""Verification suite for the Endogenous-Credibility model.

Covers the ten checks required by the implementation/validation contract plus
two engine-level validations:

  1.  row-stochasticity of W and B
  2.  bounded confidence
  3.  bounded values
  4.  simultaneous update logic
  5.  reproducibility under fixed seed
  6.  no-social baseline
  7.  isolated communities (B = I)
  8.  full mixing
  9.  quotient-field exactness under balanced block weights
  10. self-weighting convention (anticipatory includes self, retrospective excludes self)
  +   DDM analytic choice/RT vs Euler--Maruyama
  +   block vs dense engine equivalence on a balanced W
"""
import numpy as np
import pytest

from credibility import ddm, confidence, network, metrics
from credibility.config import Params
from credibility.model import (
    simulate_blocks, simulate_dense, simulate_meso,
    value_update_block, value_update_dense,
    anticipatory_field_block, anticipatory_field_dense,
    community_onehot, community_means,
)
from credibility.network import balanced_block_W, make_membership, two_community_B


# ---- 1. row-stochasticity ------------------------------------------------
def test_row_stochasticity_W_and_B():
    sizes = (5, 7, 3)
    B = np.array([[0.7, 0.2, 0.1], [0.15, 0.7, 0.15], [0.2, 0.2, 0.6]])
    assert network.is_row_stochastic(B)
    W, g = balanced_block_W(sizes, B)
    assert network.is_row_stochastic(W)
    rng = np.random.default_rng(0)
    Wsbm, _ = network.sample_weighted_sbm(sizes, B, rng)
    assert network.is_row_stochastic(Wsbm)


# ---- 2. bounded confidence ----------------------------------------------
def test_confidence_bounded():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 5, 10000)
    tau = np.abs(rng.normal(1, 1, 10000)) + 1e-3
    C = confidence.confidence_decision(np.abs(v), tau, a=1.0, kappa1=3, kappa2=1, tau0=0.5)
    assert np.all(C > 0) and np.all(C < 1)
    p = 0.5 + 0.5 * rng.random(10000)
    Cb = confidence.confidence_balance(p)
    assert np.all(Cb > 0) and np.all(Cb < 1)


# ---- 3. bounded values ---------------------------------------------------
def test_values_bounded():
    # Aggressive learning rates and strong social weights stress the clipping.
    p = Params(sizes=(40, 40), T=80, lam=1.5, eta=1.0,
               alpha_min=0.2, alpha_max=0.95, gamma=1.5)
    out = simulate_blocks(p, n_reps=8, seed=3, record_traj=True)
    Q = out["traj_Q"]
    assert np.isfinite(Q).all()
    assert Q.min() >= -1e-9 and Q.max() <= 1 + 1e-9
    # action frequencies are genuine probabilities
    assert out["term_m"].min() >= 0 and out["term_m"].max() <= 1
    assert np.allclose(out["term_m"].sum(axis=2), 1.0)


# ---- 4. simultaneous update ---------------------------------------------
def test_simultaneous_update_matches_reference():
    rng = np.random.default_rng(7)
    p = Params(sizes=(6,), T=1, lam=0.0, eta=0.8, gamma=0.7, omega=1.0,
               eps_soc=1e-3, private_lr_mode="confidence", social_lr_mode="confidence")
    N = 6
    W, g = balanced_block_W((N,), np.array([[1.0]]))
    Q = rng.random((1, N, 2))
    A = rng.integers(0, 2, (1, N))
    Rdraw = (rng.random((1, N)) < 0.5).astype(float)
    C = 0.2 + 0.6 * rng.random((1, N))
    Q_engine = value_update_dense(Q, A, Rdraw, C, W, p)

    # Independent reference: all PEs from the pre-update Q, applied together.
    Wns = W.copy(); np.fill_diagonal(Wns, 0.0)
    Q_ref = Q.copy()
    for i in range(N):
        for k in (0, 1):
            inc = 0.0
            if A[0, i] == k:                                   # private (chosen arm)
                d = Rdraw[0, i] - Q[0, i, k]
                a = (p.alpha_min + (p.alpha_max - p.alpha_min) * C[0, i]) if d < 0 \
                    else (p.alpha_max - (p.alpha_max - p.alpha_min) * C[0, i])
                inc += a * d
            wk = sum(Wns[i, j] for j in range(N) if A[0, j] == k)
            rk = sum(Wns[i, j] * Rdraw[0, j] for j in range(N) if A[0, j] == k)
            ck = sum(Wns[i, j] * C[0, j] for j in range(N) if A[0, j] == k)
            dsoc = rk - wk * Q[0, i, k]                        # pre-update Q used
            cbar = ck / (wk + p.eps_soc)
            inc += p.eta * p.gamma * (cbar ** p.omega) * dsoc
            Q_ref[0, i, k] = min(1.0, max(0.0, Q[0, i, k] + inc))
    assert np.allclose(Q_engine, Q_ref, atol=1e-12)

    # The test has teeth: a *sequential* private-then-social update differs.
    Q_seq = Q.copy()
    for i in range(N):                                         # private first
        k = A[0, i]
        d = Rdraw[0, i] - Q[0, i, k]
        a = (p.alpha_min + (p.alpha_max - p.alpha_min) * C[0, i]) if d < 0 \
            else (p.alpha_max - (p.alpha_max - p.alpha_min) * C[0, i])
        Q_seq[0, i, k] = np.clip(Q[0, i, k] + a * d, 0, 1)
    for i in range(N):                                         # social using updated Q
        for k in (0, 1):
            wk = sum(Wns[i, j] for j in range(N) if A[0, j] == k)
            rk = sum(Wns[i, j] * Rdraw[0, j] for j in range(N) if A[0, j] == k)
            ck = sum(Wns[i, j] * C[0, j] for j in range(N) if A[0, j] == k)
            cbar = ck / (wk + p.eps_soc)
            Q_seq[0, i, k] = np.clip(
                Q_seq[0, i, k] + p.eta * p.gamma * (cbar ** p.omega) * (rk - wk * Q_seq[0, i, k]), 0, 1)
    assert not np.allclose(Q_engine, Q_seq)


# ---- 5. seed reproducibility --------------------------------------------
def test_seed_reproducibility():
    p = Params(sizes=(50, 50), T=60)
    a = simulate_blocks(p, n_reps=10, seed=123)
    b = simulate_blocks(p, n_reps=10, seed=123)
    c = simulate_blocks(p, n_reps=10, seed=124)
    assert np.array_equal(a["term_m"], b["term_m"])
    assert np.allclose(a["regret"], b["regret"])
    assert not np.array_equal(a["term_m"], c["term_m"])


# ---- 6. no-social baseline ----------------------------------------------
def test_no_social_baseline_independent_of_B():
    # With lambda = eta = 0 the coupling B is irrelevant to the dynamics.
    p1 = Params(sizes=(60, 60), T=120, lam=0.0, eta=0.0, b_self=0.9, mu=(0.65, 0.5))
    p2 = p1.replace(b_self=0.5)
    a = simulate_blocks(p1, n_reps=12, seed=5)
    b = simulate_blocks(p2, n_reps=12, seed=5)
    assert np.allclose(a["term_m"], b["term_m"])
    # and the population mostly learns the better arm
    a_star = a["a_star"]
    assert a["term_m"][:, :, a_star].mean() > 0.6


# ---- 7. isolated communities (B = I) ------------------------------------
def test_isolated_communities_decouple():
    # B = I: community 1's outcome must not depend on community 2's init.
    base = Params(sizes=(40, 40), T=80, b_self=1.0, lam=0.7, eta=0.4,
                  q_init=((0.5, 0.5), (0.5, 0.5)))
    other = base.replace(q_init=((0.5, 0.5), (0.05, 0.95)))
    a = simulate_blocks(base, n_reps=10, seed=9)
    b = simulate_blocks(other, n_reps=10, seed=9)
    # community 0 (first block) identical; community 1 differs.
    assert np.allclose(a["term_m"][:, 0, :], b["term_m"][:, 0, :])
    assert not np.allclose(a["term_m"][:, 1, :], b["term_m"][:, 1, :])


# ---- 8. full mixing ------------------------------------------------------
def test_full_mixing_symmetry():
    # B with equal entries: communities are exchangeable -> equal mean mass.
    p = Params(sizes=(120, 120), T=150, b_self=0.5, lam=0.6, eta=0.3,
               mu=(0.6, 0.5), q_init=((0.5, 0.5), (0.5, 0.5)))
    out = simulate_blocks(p, n_reps=60, seed=11)
    m_star = out["term_m"][:, :, out["a_star"]].mean(axis=0)
    assert abs(m_star[0] - m_star[1]) < 0.05


# ---- 9. quotient-field exactness under balanced block weights -----------
def test_quotient_field_exact():
    sizes = (8, 6, 5)
    B = np.array([[0.6, 0.25, 0.15], [0.2, 0.7, 0.1], [0.3, 0.1, 0.6]])
    W, g = balanced_block_W(sizes, B)
    M = len(sizes)
    Csel = community_onehot(g, np.int64(M))
    sizes_arr = np.asarray(sizes)
    rng = np.random.default_rng(2)
    R, N = 4, sum(sizes)
    A_prev = rng.integers(0, 2, (R, N))
    C_prev = rng.random((R, N))
    # dense field including self (full W)
    cw = np.stack([(A_prev == k).astype(float) * C_prev for k in (0, 1)], axis=-1)
    S_dense = anticipatory_field_dense(cw, W)
    # block field via community masses
    phi = np.stack([community_means(cw[:, :, k], Csel, sizes_arr) for k in (0, 1)], axis=-1)
    S_block = anticipatory_field_block(phi, B, g)
    assert np.allclose(S_dense, S_block, atol=1e-12)


# ---- block vs dense engine equivalence (uses #9 for both channels) ------
def test_block_dense_value_update_equivalence():
    sizes = (7, 9)
    B = two_community_B(0.8)
    W, g = balanced_block_W(sizes, B)
    M = len(sizes); sizes_arr = np.asarray(sizes)
    Csel = community_onehot(g, M)
    p = Params(sizes=sizes, lam=0.5, eta=0.6, gamma=0.7)
    bc_over_Nc = (np.diag(B) / sizes_arr)[g]
    rng = np.random.default_rng(4)
    R, N = 5, sum(sizes)
    Q = rng.random((R, N, 2))
    A = rng.integers(0, 2, (R, N))
    Rdraw = (rng.random((R, N)) < 0.5).astype(float)
    C = rng.random((R, N))
    Ind = np.stack([(A == k).astype(float) for k in (0, 1)], axis=-1)
    m_now = np.stack([community_means(Ind[:, :, k], Csel, sizes_arr) for k in (0, 1)], axis=-1)
    rbar = np.stack([community_means(Ind[:, :, k] * Rdraw, Csel, sizes_arr) for k in (0, 1)], axis=-1)
    phi = np.stack([community_means(Ind[:, :, k] * C, Csel, sizes_arr) for k in (0, 1)], axis=-1)
    Qb = value_update_block(Q, A, Rdraw, C, m_now, rbar, phi, B, g, bc_over_Nc, p)
    Qd = value_update_dense(Q, A, Rdraw, C, W, p)
    assert np.allclose(Qb, Qd, atol=1e-12)


# ---- 10. self-weighting convention --------------------------------------
def test_self_weighting_convention():
    # Anticipatory channel INCLUDES self; retrospective channel EXCLUDES self.
    N = 4
    W, g = balanced_block_W((N,), np.array([[1.0]]))     # W_ii = 1/N > 0
    p = Params(sizes=(N,), eta=1.0, gamma=1.0, omega=1.0)
    # Agent 0 alone chooses arm 0; everyone else chooses arm 1.
    A = np.array([[0, 1, 1, 1]])
    C = np.array([[0.9, 0.5, 0.5, 0.5]])
    cw = np.stack([(A == k).astype(float) * C for k in (0, 1)], axis=-1)
    S = anticipatory_field_dense(cw, W)
    # Anticipatory: agent 0's signal on arm 0 must be > 0 (own action counts).
    assert S[0, 0, 0] > 0
    # Retrospective: agent 0 is the only arm-0 chooser, so excluding self gives
    # zero social weight on arm 0 -> no social update on arm 0.
    Q = np.full((1, N, 2), 0.5)
    Rdraw = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q_new = value_update_dense(Q, A, Rdraw, C, W, p)
    # Only the private update (chosen arm 0) moves agent 0's arm-0 value;
    # remove it and confirm no residual social contribution on arm 0.
    d = Rdraw[0, 0] - Q[0, 0, 0]
    a = p.alpha_max - (p.alpha_max - p.alpha_min) * C[0, 0]   # d >= 0 branch
    expected = np.clip(0.5 + a * d, 0, 1)
    assert np.isclose(Q_new[0, 0, 0], expected, atol=1e-12)


# ---- DDM: analytic choice probability vs Euler--Maruyama ----------------
@pytest.mark.parametrize("v", [-1.5, -0.5, 0.0, 0.8, 2.0])
def test_ddm_choice_prob_matches_euler(v):
    a, sigma = 1.0, 1.0
    rng = np.random.default_rng(int(100 + 10 * v))
    n = 6000
    action, _ = ddm.euler_maruyama_fpt(np.full(n, v), a, sigma, rng, dt=1e-3)
    emp = np.mean(action == 1)
    ana = ddm.choice_prob_up(v, a, sigma)
    assert abs(emp - ana) < 0.025


def test_ddm_mean_fpt_matches_euler():
    a, sigma = 1.0, 1.0
    rng = np.random.default_rng(77)
    for v in (0.0, 1.0, 2.0):
        _, tau = ddm.euler_maruyama_fpt(np.full(4000, v), a, sigma, rng, dt=5e-4)
        emp = tau.mean()
        ana = ddm.mean_fpt(v, a, sigma)
        assert abs(emp - ana) / ana < 0.08


# ---- meso engine sanity --------------------------------------------------
def test_meso_bounded_and_normalised():
    p = Params(sizes=(100, 100), T=120, lam=0.6, eta=0.3)
    out = simulate_meso(p)
    assert np.all(out["traj_Q"] >= -1e-9) and np.all(out["traj_Q"] <= 1 + 1e-9)
    assert np.allclose(out["term_m"].sum(axis=1), 1.0)


# ---- regime classification -----------------------------------------------
def test_regime_classification():
    # term_m shape (R, M, 2); a_star = 0.
    eff = np.array([[[0.95, 0.05], [0.92, 0.08]]])
    wrong = np.array([[[0.05, 0.95], [0.08, 0.92]]])
    pol = np.array([[[0.95, 0.05], [0.05, 0.95]]])
    unr = np.array([[[0.6, 0.4], [0.7, 0.3]]])
    assert metrics.classify_regime(eff, 0)[0] == metrics.EFFICIENT
    assert metrics.classify_regime(wrong, 0)[0] == metrics.WRONG
    assert metrics.classify_regime(pol, 0)[0] == metrics.POLARISED
    assert metrics.classify_regime(unr, 0)[0] == metrics.UNRESOLVED


# ---- mean-field reduction ------------------------------------------------
def test_meanfield_step_bounded_and_fixed_point():
    from credibility import meanfield as mf
    p = Params(sizes=(200, 200), mu=(0.6, 0.4), b_self=0.85)
    Q = np.full((2, 2), 0.5)
    Qn = mf.step(Q, p, lam=0.3)
    assert np.all(Qn >= -1e-12) and np.all(Qn <= 1 + 1e-12)
    # from a corrective seed the fixed point puts mass on the optimal arm.
    Qs, conv = mf.run_to_fixed_point(mf._ic(p, "corrective"), p, lam=0.3, iters=400)
    assert mf.mass_on(Qs, p, int(np.argmax(p.mu)), lam=0.3) > 0.6


def test_meanfield_amplification_threshold():
    """Early-lead pitchfork: |z*| is ~0 below lambda* and large above."""
    from credibility import meanfield as mf
    p = Params(sizes=(200, 200), mu=(0.53, 0.47), b_self=0.85, eta=0.12)
    lstar, C0 = mf.lambda_star(p)
    z_lo = mf.amplification_branch(p, [0.6 * lstar], z0=1e-3)[0]
    z_hi = mf.amplification_branch(p, [1.6 * lstar], z0=1e-3)[0]
    assert z_lo < 1e-2 < 0.1 < z_hi
    assert 0.0 < C0 < 1.0 and lstar > 0.0


def test_meanfield_basin_probabilities():
    """Below lambda* basins are corrective; above, distortive dominates; rows sum to 1."""
    from credibility import meanfield as mf
    p = Params(sizes=(200, 200), mu=(0.53, 0.47), b_self=0.85, eta=0.12)
    lstar, _ = mf.lambda_star(p)
    pr = mf.basin_probabilities(p, [0.5 * lstar, 1.7 * lstar], n=200)
    assert np.allclose(pr.sum(axis=1), 1.0)
    assert pr[0, 0] > 0.9            # corrective below threshold
    assert pr[1, 2] > 0.5            # distortive dominates above threshold
