"""Matched-mass credibility-selectivity robustness with CRN comparisons."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common, plotstyle
from credibility import scenarios, metrics, uncertainty
from credibility.config import SCALE
from credibility.model import simulate_blocks

CHIS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def main():
    scale = common.get_scale()
    reps = 1000 if scale == "publication" else SCALE[scale]["reps_traj"]
    points = {"corrective": scenarios.efficient(scale),
              "contested": scenarios.ablation_point(scale),
              "distortive": scenarios.wrong(scale)}
    raw, summary = [], []
    for si, (name, base) in enumerate(points.items()):
        runs = {}
        for chi in CHIS:
            out = simulate_blocks(base.replace(selectivity_chi=chi), reps, 1800 + si)
            code = metrics.classify_regime(out["term_m"], out["a_star"])
            runs[chi] = (out, code)
        ref_out, ref_code = runs[1.0]
        for chi, (out, code) in runs.items():
            for r in range(reps):
                raw.append(dict(scenario=name, chi=chi, replication=r,
                                regime=metrics.REGIMES[code[r]], regret=out["regret"][r],
                                delta_wrong=float(code[r] == metrics.WRONG) - float(ref_code[r] == metrics.WRONG),
                                delta_regret=out["regret"][r] - ref_out["regret"][r]))
            row = dict(scenario=name, chi=chi, replications=reps)
            for i, regime in enumerate(metrics.REGIMES):
                s = uncertainty.proportion_summary(code == i)
                row.update({f"p_{regime}": s["p"], f"{regime}_lo": s["lo"], f"{regime}_hi": s["hi"]})
            reg = uncertainty.bootstrap_mean(out["regret"], seed=1900 + si)
            dw = uncertainty.paired_bootstrap_diff(code == metrics.WRONG, ref_code == metrics.WRONG,
                                                   seed=2000 + si)
            dr = uncertainty.paired_bootstrap_diff(out["regret"], ref_out["regret"], seed=2100 + si)
            row.update(mean_regret=reg["mean"], regret_lo=reg["lo"], regret_hi=reg["hi"],
                       delta_p_wrong=dw["diff"], delta_p_wrong_lo=dw["lo"], delta_p_wrong_hi=dw["hi"],
                       delta_regret=dr["diff"], delta_regret_lo=dr["lo"], delta_regret_hi=dr["hi"])
            summary.append(row)
    for filename, rows in [("credibility_selectivity.csv", raw),
                           ("credibility_selectivity_summary.csv", summary)]:
        with open(f"{common.RESULTS}/{filename}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with open(f"{common.FIGDIR}/../tables/credibility_selectivity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    plotstyle.use(); fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for name in points:
        rr = [r for r in summary if r["scenario"] == name]
        x = [r["chi"] for r in rr]
        axes[0].plot(x, [r["p_wrong"] for r in rr], "-o", ms=3, label=name)
        axes[1].plot(x, [r["mean_regret"] for r in rr], "-o", ms=3)
        axes[2].plot(x, [r["delta_p_wrong"] for r in rr], "-o", ms=3)
    axes[0].set_ylabel("P(wrong)"); axes[1].set_ylabel("mean regret")
    axes[2].set_ylabel(r"paired $\Delta$ P(wrong) vs $\chi=1$")
    for ax in axes: ax.set_xlabel(r"selectivity $\chi$"); ax.axvline(1, color=".5", ls=":")
    axes[0].legend(frameon=False, fontsize=7); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/credibility_selectivity.{ext}", dpi=300)
    print(f"credibility selectivity complete: {reps} replications/point")


if __name__ == "__main__":
    main()
