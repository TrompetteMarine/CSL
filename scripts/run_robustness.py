"""Robustness and sensitivity battery (Appendix D).

R1 terminal threshold (0.80/0.90/0.95)        R5 random-graph perturbations
R2 horizon (T = 200..1000)                    R6 phase-grid resolution
R3 initial-condition strength                 R7 DDM reaction-time approximation
R4 reward gap |mu1-mu2|

Each check is cached by its CSV output and may be run under a wall budget
(ECSL_BUDGET seconds); re-run until all CSVs exist. Writes paper/tables/*.csv
and a results/robustness_<scale>.json summary used by the figure and manuscript.
"""
import csv
import json
import os
import time
import numpy as np
import common
from credibility import scenarios, metrics, network, uncertainty as unc
from credibility.model import simulate_blocks, simulate_dense

BUDGET = float(os.environ.get("ECSL_BUDGET", "1e9"))
TABDIR = os.path.abspath(os.path.join(common.FIGDIR, "..", "tables"))
os.makedirs(TABDIR, exist_ok=True)
SUMMARY = f"{common.RESULTS}/robustness_{{}}.json"


def _probs(out, thr=0.9):
    a = out["a_star"]
    codes = metrics.classify_regime(out["term_m"], a, thr)
    pw = unc.proportion_summary(codes == metrics.WRONG)
    pe = unc.proportion_summary(codes == metrics.EFFICIENT)
    return dict(eff=float(np.mean(codes == metrics.EFFICIENT)),
                wrong=float(np.mean(codes == metrics.WRONG)),
                pol=float(np.mean(codes == metrics.POLARISED)),
                unr=float(np.mean(codes == metrics.UNRESOLVED)),
                wrong_lo=pw["lo"], wrong_hi=pw["hi"], eff_lo=pe["lo"], eff_hi=pe["hi"])


def _grid_terms(base, lams, gammas, reps, seed0, rt_mode="ig"):
    """Run the (lambda, Gamma) grid once; return [(term_m, a_star), ...]."""
    terms = []
    for gi, G in enumerate(gammas):
        for li, L in enumerate(lams):
            p = base.replace(lam=float(L), b_self=float(1 - G), rt_mode=rt_mode)
            out = simulate_blocks(p, reps, seed0 + gi * 100 + li)
            terms.append((out["term_m"], out["a_star"]))
    return terms


def _frac_from_terms(terms, thr=0.9):
    cnt = np.zeros(4)
    for term_m, a in terms:
        dom = np.bincount(metrics.classify_regime(term_m, a, thr), minlength=4).argmax()
        cnt[dom] += 1
    return cnt / cnt.sum()


def _area_fractions(base, lams, gammas, reps, seed0, thr=0.9, rt_mode="ig"):
    return _frac_from_terms(_grid_terms(base, lams, gammas, reps, seed0, rt_mode), thr)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _points(scale):
    b = scenarios.phase_base(scale)            # asymmetric: community 2 wrong-led
    return dict(
        near_zero=b.replace(lam=0.13, b_self=0.70),
        corrective=b.replace(lam=0.40, b_self=0.70),
        distortive=b.replace(lam=0.90, b_self=0.70),
        polar=b.replace(lam=0.90, b_self=0.98),
    )


# --- R1 threshold ---------------------------------------------------------
def r1(scale, summ):
    path = f"{TABDIR}/robustness_thresholds.csv"
    if "R1" in summ:
        return
    b = scenarios.phase_base(scale)
    lams = np.linspace(0.0, 1.6, 9); gammas = np.linspace(0.01, 0.5, 9)
    terms = _grid_terms(b, lams, gammas, 80, 9100)        # one grid, three thresholds
    rows = []; summ["R1"] = {}
    for thr in (0.80, 0.90, 0.95):
        fr = _frac_from_terms(terms, thr)
        rows.append([thr, *np.round(fr, 3)])
        summ["R1"][str(thr)] = dict(zip(metrics.REGIMES, fr.tolist()))
    _write_csv(path, ["threshold", *metrics.REGIMES], rows)
    print("R1 thresholds done")


# --- R2 horizon -----------------------------------------------------------
def r2(scale, summ):
    path = f"{TABDIR}/robustness_horizon.csv"
    if "R2" in summ:
        return
    pts = _points(scale); rows = []; summ["R2"] = {}
    for name in ("near_zero", "corrective", "distortive"):
        for T in (200, 300, 500, 1000):
            out = simulate_blocks(pts[name].replace(T=T), 200, 9200, record_traj=False)
            pr = _probs(out)
            rows.append([name, T, round(pr["eff"], 3), round(pr["wrong"], 3),
                         round(pr["unr"], 3)])
            summ["R2"][f"{name}_T{T}"] = pr
    _write_csv(path, ["point", "T", "P_eff", "P_wrong", "P_unresolved"], rows)
    print("R2 horizon done")


# --- R3 initial-condition strength ---------------------------------------
def r3(scale, summ):
    path = f"{TABDIR}/robustness_initial_conditions.csv"
    if "R3" in summ:
        return
    b = scenarios.phase_base(scale).replace(lam=0.90, b_self=0.70)
    rows = []; summ["R3"] = {}
    for delta in (0.0, 0.04, 0.08, 0.12, 0.16):
        q = ((0.5, 0.5), (0.5 - delta, 0.5 + delta))
        out = simulate_blocks(b.replace(q_init=q), 250, 9300)
        pr = _probs(out)
        rows.append([delta, round(pr["eff"], 3), round(pr["wrong"], 3), round(pr["pol"], 3)])
        summ["R3"][f"lead_{delta}"] = pr
    _write_csv(path, ["wrong_lead_delta", "P_eff", "P_wrong", "P_pol"], rows)
    print("R3 initial conditions done")


# --- R4 reward gap --------------------------------------------------------
def r4(scale, summ):
    path = f"{TABDIR}/robustness_reward_gap.csv"
    if "R4" in summ:
        return
    b = scenarios.phase_base(scale).replace(lam=0.90, b_self=0.70)
    rows = []; summ["R4"] = {}
    for g in (0.02, 0.05, 0.10, 0.20):
        out = simulate_blocks(b.replace(mu=(0.5 + g / 2, 0.5 - g / 2)), 250, 9400)
        pr = _probs(out)
        rows.append([g, round(pr["eff"], 3), round(pr["wrong"], 3), round(pr["wrong_lo"], 3),
                     round(pr["wrong_hi"], 3)])
        summ["R4"][f"gap_{g}"] = pr
    _write_csv(path, ["reward_gap", "P_eff", "P_wrong", "P_wrong_lo", "P_wrong_hi"], rows)
    print("R4 reward gap done")


# --- R5 random graphs -----------------------------------------------------
def _noisy_W(sizes, B, rng, cv=0.4):
    W, g = network.balanced_block_W(sizes, B)
    W = W * np.exp(rng.normal(0, cv, W.shape))          # multiplicative noise
    W /= W.sum(axis=1, keepdims=True)
    return W


def r5(scale, summ):
    path = f"{TABDIR}/robustness_random_graphs.csv"
    if "R5" in summ:
        return
    b = scenarios.phase_base(scale).replace(sizes=(70, 70))
    rng = np.random.default_rng(9500)
    rows = []; summ["R5"] = {}
    for name, lam, bs in (("corrective", 0.40, 0.70), ("distortive", 0.90, 0.70)):
        p = b.replace(lam=lam, b_self=bs)
        B = p.coupling(); sizes = p.sizes
        Wb, _ = network.balanced_block_W(sizes, B)
        Ws, _ = network.sample_weighted_sbm(sizes, B, rng)
        Wn = _noisy_W(sizes, B, rng)
        for gtype, W in (("balanced", Wb), ("weighted_sbm", Ws), ("noisy_W", Wn)):
            out = simulate_dense(W, p, 100, 9500)
            pr = _probs(out)
            rows.append([name, gtype, round(pr["eff"], 3), round(pr["wrong"], 3),
                         round(pr["pol"], 3)])
            summ["R5"][f"{name}_{gtype}"] = pr
    _write_csv(path, ["point", "graph", "P_eff", "P_wrong", "P_pol"], rows)
    print("R5 random graphs done")


# --- R6 grid resolution ---------------------------------------------------
def r6(scale, summ):
    path = f"{TABDIR}/robustness_grid_resolution.csv"
    if "R6" in summ:
        return
    b = scenarios.phase_base(scale)
    rows = []; summ["R6"] = {}
    # 7x7 fresh; 9x9 reused from R1 (thr 0.9); 13x13 from the production phase grid
    lams = np.linspace(0.0, 1.6, 7); gammas = np.linspace(0.01, 0.5, 7)
    fr7 = _area_fractions(b, lams, gammas, 80, 9600)
    rows.append(["7x7", *np.round(fr7, 3)])
    summ["R6"]["7x7"] = dict(zip(metrics.REGIMES, fr7.tolist()))
    if summ.get("R1", {}).get("0.9"):
        d = summ["R1"]["0.9"]; fr9 = np.array([d[r] for r in metrics.REGIMES])
        rows.append(["9x9", *np.round(fr9, 3)]); summ["R6"]["9x9"] = d
    pf = f"{common.RESULTS}/phase_{scale}.npz"
    if os.path.exists(pf):
        P = np.load(pf)["P"]; reg = P.argmax(2)
        fr = np.array([np.mean(reg == k) for k in range(4)])
        rows.append(["13x13", *np.round(fr, 3)])
        summ["R6"]["13x13"] = dict(zip(metrics.REGIMES, fr.tolist()))
    _write_csv(path, ["grid", *metrics.REGIMES], rows)
    print("R6 grid resolution done")


# --- R7 DDM reaction-time approximation -----------------------------------
def r7(scale, summ):
    path = f"{TABDIR}/robustness_ddm_rt.csv"
    if "R7" in summ:
        return
    b = scenarios.phase_base(scale)
    lams = np.linspace(0.0, 1.6, 7); gammas = np.linspace(0.01, 0.5, 7)
    rows = []; summ["R7"] = {}
    for rt in ("ig", "mean"):
        fr = _area_fractions(b, lams, gammas, 60, 9700, rt_mode=rt)
        rows.append([rt, *np.round(fr, 3)])
        summ["R7"][rt] = dict(zip(metrics.REGIMES, fr.tolist()))
    _write_csv(path, ["rt_mode", *metrics.REGIMES], rows)
    print("R7 DDM RT done")


CHECKS = [r1, r2, r3, r4, r5, r6, r7]


def main():
    scale = common.get_scale()
    spath = SUMMARY.format(scale)
    summ = json.load(open(spath)) if os.path.exists(spath) else {}
    t0 = time.time()
    for chk in CHECKS:
        chk(scale, summ)
        json.dump(summ, open(spath, "w"), indent=2, default=float)
        if time.time() - t0 > BUDGET:
            print(f"[budget] pausing after {chk.__name__}; re-run to continue.")
            return
    done = all(os.path.exists(f"{TABDIR}/{f}") for f in (
        "robustness_thresholds.csv", "robustness_horizon.csv",
        "robustness_initial_conditions.csv", "robustness_reward_gap.csv",
        "robustness_random_graphs.csv", "robustness_grid_resolution.csv",
        "robustness_ddm_rt.csv"))
    print("ALL ROBUSTNESS DONE" if done else "ROBUSTNESS INCOMPLETE")


if __name__ == "__main__":
    main()
