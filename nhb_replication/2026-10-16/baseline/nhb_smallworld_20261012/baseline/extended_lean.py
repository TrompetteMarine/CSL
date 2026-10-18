"""Versioned reduced-form reinforcement learner; no DDM or time variable.

The primary score is expressed decisiveness max(p, 1-p), where p is the
logistic choice propensity. It is a convention, not probability of being right.
Arms are zero-indexed; all gains use pre-update values. Original modules are
left untouched for historical reproduction. See repair_v1/02 and 03.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class LeanParams:
    sizes: tuple[int, ...] = (80, 80)
    T: int = 260
    mu: tuple[float, float] = (.55, .45)
    B: tuple[tuple[float, ...], ...] = ((.85, .15), (.15, .85))
    q_init: tuple[tuple[float, float], ...] = ((.52, .50), (.42, .62))
    h: float = 12.
    lam: float = .6
    peer_gain: float = .15
    alpha_min: float = .05
    alpha_max: float = .4
    eps: float = 1e-3
    score: str = 'preference'
    valence: bool = True
    social_weighting: bool = True
    window: int = 100

    def replace(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)

    def validate(self):
        M = len(self.sizes)
        b = np.asarray(self.B, float)
        q = np.asarray(self.q_init, float)
        scalars = [self.h, self.lam, self.peer_gain, self.alpha_min,
                   self.alpha_max, self.eps, *self.mu]
        if (not np.isfinite(scalars).all() or b.shape != (M, M)
                or not np.isfinite(b).all() or np.any(b < 0)
                or not np.allclose(b.sum(1), 1, atol=1e-14, rtol=0)):
            raise ValueError('finite parameters and row-stochastic B required')
        if (q.shape != (M, 2) or not np.isfinite(q).all()
                or np.any((q < 0) | (q > 1))):
            raise ValueError('initial Q must lie in [0,1]')
        if (self.T < 1 or any(n < 1 or int(n) != n for n in self.sizes)
                or not 1 <= self.window <= self.T):
            raise ValueError('positive integral sizes/T and admissible window required')
        if not (self.h > 0 and self.lam >= 0 and self.peer_gain >= 0
                and 0 <= self.alpha_min <= self.alpha_max < 1
                and self.alpha_max + self.peer_gain <= 1 and self.eps > 0
                and all(0 < x < 1 for x in self.mu)):
            raise ValueError('nonnegative convex gains and interior reward means required')
        decisiveness(np.array([.5]), self.score)


def decisiveness(p, mode='preference'):
    """Predeclared score family, with no estimated mapping parameters."""
    p = np.asarray(p, float)
    u = np.abs(2*p - 1)
    if mode == 'preference':
        return .5 * (1 + u)
    if mode == 'quadratic':
        return .5 * (1 + u*u)
    if mode == 'zero_floor':
        return u
    if mode == 'constant':
        return np.full_like(p, .5)
    raise ValueError('unknown score map')


def signed_mass(delta, h=12., mode='preference'):
    p = expit(h*np.asarray(delta, float))
    return decisiveness(p, mode) * (2*p - 1)


def local_slope(h=12., mode='preference'):
    return float(decisiveness(np.array(.5), mode)) * h/2


def frozen_field(psi, B, lam, h=12., mode='preference', private_gap=None):
    psi = np.asarray(psi, float)
    gap = np.zeros_like(psi) if private_gap is None else np.asarray(private_gap)
    return signed_mass(gap + lam*(np.asarray(B) @ psi), h, mode)


def _means(x, slices):
    return np.stack([x[:, s].mean(axis=1) for s in slices], axis=1)


def dense_weights(sizes, B):
    g = np.repeat(np.arange(len(sizes)), sizes)
    b = np.asarray(B, float)
    return b[g[:, None], g[None, :]] / np.asarray(sizes)[g][None, :]


def private_expected_increment(q, mu, d, amin=.05, amax=.4):
    """Exact reward expectation given pre-reward Q, action and score."""
    pos = amax-(amax-amin)*d
    neg = amin+(amax-amin)*d
    return mu*pos*(1-np.asarray(q)) - (1-mu)*neg*np.asarray(q)


def simulate(p: LeanParams, reps: int, seed: int, engine='block',
             score_override=None, record=False, record_agents=False, choice_B=None):
    """Population replications with indexed independent choice/reward streams.

    `score_override` is solely for archived timing-law diagnostics: callable
    (delta, probability, action, time_rng) -> score. The selected paper model
    does not use it. Dense and block engines implement the same finite-N law.
    Terminal outputs use a full window; end_m is also saved for old comparability.
    """
    p.validate()
    if reps < 1 or engine not in ('block', 'dense'):
        raise ValueError('positive reps and supported engine required')
    sizes = np.asarray(p.sizes)
    N, M = int(sum(sizes)), len(sizes)
    g = np.repeat(np.arange(M), sizes)
    edges = np.r_[0, np.cumsum(sizes)]
    slices = [slice(edges[c], edges[c+1]) for c in range(M)]
    B = np.asarray(p.B, float)
    C = B if choice_B is None else np.asarray(choice_B, float)
    if (C.shape != B.shape or not np.isfinite(C).all() or np.any(C < 0)
            or not np.allclose(C.sum(1), 1., atol=1e-14, rtol=0)):
        raise ValueError('choice_B must be row stochastic with the same shape as B')
    Wchoice = None if engine == 'block' else dense_weights(sizes, C)
    selfw = (np.diag(B)/sizes)[g]
    W = dense_weights(sizes, B) if engine == 'dense' else None
    Wns = None if W is None else W-np.diag(np.diag(W))
    crng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    rrng = np.random.default_rng(np.random.SeedSequence([seed, 102]))
    trng = np.random.default_rng(np.random.SeedSequence([seed, 103]))
    Q = np.broadcast_to(np.asarray(p.q_init)[g], (reps, N, 2)).copy()
    y = np.zeros((reps, N))
    mu = np.asarray(p.mu)
    optimal = int(np.argmax(mu))
    regret = np.zeros(reps)
    masses = np.empty((p.T, reps, M))
    qmeans = np.empty((p.T, reps, M, 2)) if record else None
    first_correction = np.full(reps, p.T, dtype=int)
    consecutive = np.zeros(reps, dtype=int)
    gain_max = 0.
    clip_count = 0
    action_record = np.empty((p.T,reps,N),dtype=np.int8) if record_agents else None
    reward_record = np.empty((p.T,reps,N),dtype=np.int8) if record_agents else None
    for t in range(p.T):
        field = (_means(y, slices) @ C.T)[:, g] if W is None else y @ Wchoice.T
        delta = Q[:, :, 0]-Q[:, :, 1]+p.lam*field
        prob = expit(p.h*delta)
        A = (crng.random((reps, N)) >= prob).astype(np.int8)
        d = (decisiveness(prob, p.score) if score_override is None
             else np.asarray(score_override(delta, prob, A, trng)))
        if np.any((d < 0) | (d > 1)) or not np.isfinite(d).all():
            raise ValueError('score must be finite and bounded')
        rewards = (rrng.random((reps, N)) < mu[A]).astype(float)
        if record_agents:
            action_record[t] = A
            reward_record[t] = rewards
        Ind = np.stack([A == k for k in (0, 1)], axis=-1)
        chosen = np.take_along_axis(Q, A[:, :, None], axis=2)[:, :, 0]
        error = rewards-chosen
        if p.valence:
            span = p.alpha_max-p.alpha_min
            alpha = np.where(error < 0, p.alpha_min+span*d, p.alpha_max-span*d)
        else:
            alpha = np.full_like(d, .5*(p.alpha_min+p.alpha_max))
        ds = d if p.social_weighting else np.ones_like(d)
        inc = np.zeros_like(Q)
        for k in (0, 1):
            ik = Ind[:, :, k].astype(float)
            inc[:, :, k] += ik*alpha*error
            if W is None:
                w = (_means(ik, slices) @ B.T)[:, g]-selfw*ik
                r = (_means(ik*rewards, slices) @ B.T)[:, g]-selfw*ik*rewards
                cm = (_means(ik*ds, slices) @ B.T)[:, g]-selfw*ik*ds
            else:
                w = ik @ Wns.T
                r = (ik*rewards) @ Wns.T
                cm = (ik*ds) @ Wns.T
            # Only roundoff negatives are removed; substantive clipping is logged.
            w = np.maximum(w, 0)
            cm = np.maximum(cm, 0)
            gain = (p.peer_gain*cm/(w+p.eps) if p.social_weighting
                    else np.full_like(w, p.peer_gain))
            inc[:, :, k] += gain*(r-w*Q[:, :, k])
            gain_max = max(gain_max, float(np.max(ik*alpha+gain*w)))
        Qnext = Q+inc
        clip_count += int(np.count_nonzero((Qnext < -1e-12) | (Qnext > 1+1e-12)))
        Q = np.clip(Qnext, 0, 1)
        regret += (mu[optimal]-mu[A]).sum(axis=1)/N
        masses[t] = _means((A == optimal).astype(float), slices)
        corrected = np.all(masses[t] >= .5, axis=1)
        consecutive = np.where(corrected, consecutive+1, 0)
        first_correction = np.where((consecutive >= 5) & (first_correction == p.T),
                                    t-4, first_correction)
        y = (1-2*A)*ds
        if record:
            qmeans[t] = np.stack([_means(Q[:, :, k], slices) for k in (0, 1)], -1)
    out = dict(window_m=masses[-p.window:].mean(axis=0), end_m=masses[-1],
               regret=regret, first_correction=first_correction,
               censored=(first_correction == p.T), gain_max=gain_max,
               clip_count=clip_count, final_Q=Q,
               final_Q_variance=np.stack([Q[:, s].var(axis=1) for s in slices], 1))
    if record:
        out.update(mass_path=masses, Q_path=qmeans)
    if record_agents:
        out.update(actions=action_record,rewards=reward_record)
    return out


def classify(m, cutoff=.9):
    """0 efficient, 1 wrong, 2 polarised (all decisive), 3 unresolved."""
    m = np.asarray(m)
    eff = np.all(m >= cutoff, axis=1)
    wrong = np.all(m <= 1-cutoff, axis=1)
    decisive = np.all((m >= cutoff) | (m <= 1-cutoff), axis=1)
    return np.where(eff, 0, np.where(wrong, 1, np.where(decisive, 2, 3)))
