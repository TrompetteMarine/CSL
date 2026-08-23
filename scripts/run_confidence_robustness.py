"""Confidence-map calibration family and matched-marginal social placebo."""
import csv, sys
import itertools
import numpy as np
import common
from credibility import scenarios, metrics, uncertainty
from credibility.config import SCALE
from credibility.model import simulate_blocks


def summarise(label, out, ref, ref_code, seed):
    code = metrics.classify_regime(out["term_m"], out["a_star"])
    row = dict(variant=label, replications=len(code))
    for i, name in enumerate(metrics.REGIMES):
        s = uncertainty.proportion_summary(code == i)
        row.update({f"p_{name}": s["p"], f"{name}_lo": s["lo"], f"{name}_hi": s["hi"]})
    bm = uncertainty.bootstrap_mean(out["regret"], seed=seed)
    dw = uncertainty.paired_bootstrap_diff(code == metrics.WRONG, ref_code == metrics.WRONG, seed=seed + 1)
    dr = uncertainty.paired_bootstrap_diff(out["regret"], ref["regret"], seed=seed + 2)
    row.update(mean_regret=bm["mean"], regret_lo=bm["lo"], regret_hi=bm["hi"],
               delta_p_wrong=dw["diff"], delta_p_wrong_lo=dw["lo"], delta_p_wrong_hi=dw["hi"],
               delta_regret=dr["diff"], delta_regret_lo=dr["lo"], delta_regret_hi=dr["hi"])
    return row, code


def main():
    scale = common.get_scale(); smoke = "--smoke" in sys.argv
    reps = 1000 if scale == "publication" else SCALE[scale]["reps_traj"]
    base = scenarios.ablation_point(scale); seed = 2300
    ref = simulate_blocks(base, reps, seed); ref_code = metrics.classify_regime(ref["term_m"], ref["a_star"])
    grid = list(itertools.product((1.5, 2.25, 3.0, 4.5), (.5, .75, 1.0, 1.5), (.25, .5, .75, 1.0)))
    if smoke: grid = [(3.0, 1.0, .5)]
    rows = []
    for n, (k1, k2, tau0) in enumerate(grid):
        out = simulate_blocks(base.replace(kappa1=k1, kappa2=k2, tau0=tau0), reps, seed)
        row, _ = summarise(f"kappa1={k1};kappa2={k2};tau0={tau0}", out, ref, ref_code, 2400+n)
        row.update(family="parameter", kappa1=k1, kappa2=k2, tau0=tau0, scale=np.nan)
        rows.append(row)
    scales = (1.0,) if smoke else (.5, .75, 1.0, 1.5, 2.0)
    for n, s in enumerate(scales):
        out = simulate_blocks(base.replace(confidence_scale=s), reps, seed)
        row, _ = summarise(f"logit_scale={s}", out, ref, ref_code, 2600+n)
        row.update(family="rank_preserving", kappa1=np.nan, kappa2=np.nan, tau0=np.nan, scale=s)
        rows.append(row)
    placebo = simulate_blocks(base.replace(permute_social_confidence=True), reps, seed)
    prow, pcode = summarise("within-community social permutation", placebo, ref, ref_code, 2700)
    prow.update(family="placebo", kappa1=np.nan, kappa2=np.nan, tau0=np.nan, scale=np.nan)
    rows.append(prow)
    path = f"{common.RESULTS}/confidence_map_robustness.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    raw = []
    for r in range(reps):
        raw.append(dict(replication=r, baseline_regime=metrics.REGIMES[ref_code[r]],
                        permuted_regime=metrics.REGIMES[pcode[r]], baseline_regret=ref["regret"][r],
                        permuted_regret=placebo["regret"][r]))
    with open(f"{common.RESULTS}/confidence_permutation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw[0]); w.writeheader(); w.writerows(raw)
    with open(f"{common.FIGDIR}/../tables/confidence_map_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(f"confidence robustness {'validation-only' if smoke else 'complete'}: {len(rows)} variants")


if __name__ == "__main__":
    main()
