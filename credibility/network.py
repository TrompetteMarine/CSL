"""Community networks and the quotient coupling matrix ``B``.

The baseline uses *balanced block weights* (Assumption 1 in the manuscript):
for ``i`` in community ``c`` and ``j`` in community ``d``,

    W_ij = B_cd / N_d .

Under this geometry the anticipatory social field is an exact community-level
object (Proposition 1/2), which is what makes the agent simulator both fast
and reducible to a quotient.  A weighted stochastic-block generator is also
provided for finite-sample / off-assumption robustness checks.
"""
from __future__ import annotations

import numpy as np


def make_membership(sizes):
    """Return the community id (0-indexed) of every agent given block sizes."""
    sizes = np.asarray(sizes, dtype=int)
    return np.repeat(np.arange(len(sizes)), sizes)


def balanced_block_W(sizes, B):
    """Dense N x N row-stochastic matrix with W_ij = B[c,d] / N_d.

    Self-weight ``W_ii = B_cc / N_c`` is included (the anticipatory channel
    treats own past confident action as inertia/memory; see the self-weighting
    convention).  Returns ``W`` and the membership array ``g``.
    """
    sizes = np.asarray(sizes, dtype=int)
    B = np.asarray(B, dtype=float)
    M = len(sizes)
    assert B.shape == (M, M), "B must be M x M"
    g = make_membership(sizes)
    N = sizes.sum()
    W = np.zeros((N, N))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    for c in range(M):
        rows = slice(starts[c], starts[c + 1])
        for d in range(M):
            cols = slice(starts[d], starts[d + 1])
            W[rows, cols] = B[c, d] / sizes[d]
    return W, g


def is_row_stochastic(M, tol=1e-10):
    """True iff every row of ``M`` is non-negative and sums to 1."""
    M = np.asarray(M, dtype=float)
    return bool(np.all(M >= -tol) and np.allclose(M.sum(axis=1), 1.0, atol=tol))


def spectral_gap(B):
    """Mesoscopic mixing kappa(B) = 1 - |lambda_2(B)| of the coupling matrix."""
    B = np.asarray(B, dtype=float)
    eig = np.sort(np.abs(np.linalg.eigvals(B)))[::-1]
    if len(eig) < 2:
        return 1.0
    return float(1.0 - eig[1])


def cohesion(B):
    """Diagonal of B: within-community share of social exposure."""
    return np.diag(np.asarray(B, dtype=float)).copy()


def permeability(B):
    """Gamma_c = 1 - B_cc: total cross-community permeability per community."""
    B = np.asarray(B, dtype=float)
    return 1.0 - np.diag(B)


def two_community_B(b_self, b_cross=None):
    """Symmetric 2x2 coupling matrix with diagonal ``b_self``.

    If ``b_cross`` is None it is set to ``1 - b_self`` so rows sum to 1.
    """
    if b_cross is None:
        b_cross = 1.0 - b_self
    return np.array([[b_self, b_cross], [b_cross, b_self]], dtype=float)


def sample_weighted_sbm(sizes, B, rng, concentration=8.0):
    """Random weighted, row-stochastic network with expected block exposures B.

    For each agent the within- and cross-community weights are drawn from
    Dirichlet distributions scaled by the target block masses ``B[c, d]`` and
    then row-normalised.  The expected block exposure equals ``B[c, d]`` while
    individual rows fluctuate -- used for the two-stage (graph x shocks)
    randomisation robustness check, *not* for the headline balanced-block runs.
    """
    sizes = np.asarray(sizes, dtype=int)
    B = np.asarray(B, dtype=float)
    M = len(sizes)
    g = make_membership(sizes)
    N = sizes.sum()
    starts = np.concatenate([[0], np.cumsum(sizes)])
    W = np.zeros((N, N))
    for i in range(N):
        c = g[i]
        for d in range(M):
            cols = slice(starts[d], starts[d + 1])
            nd = sizes[d]
            alpha = np.full(nd, concentration / nd)
            w = rng.dirichlet(alpha) * B[c, d]
            W[i, cols] = w
    W /= W.sum(axis=1, keepdims=True)
    return W, g
