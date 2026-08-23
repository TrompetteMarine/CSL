"""Render Figures 2-6 from the saved result files (no simulation here).

Each figure is written as both PDF (for LaTeX) and PNG (for preview).
Figure 1 (the mechanism diagram) is a separate TikZ source, compiled by the
build script.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common
import plotstyle

plotstyle.use()
C1, C2 = plotstyle.PALETTE["blue"], plotstyle.PALETTE["orange"]   # community 1 / 2
REG_COLOR = plotstyle.REG_COLOR


def _load(name):
    return np.load(f"{common.RESULTS}/{name}_{SCALE}.npz", allow_pickle=True)


def _log10_odds(p):
    """Base-10 odds of choosing the better option; clip only for finite plotting."""
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    return np.log10(p / (1 - p))


# --------------------------------------------------------------------------
def fig2_baseline():
    d = _load("baseline")
    names = ["efficient", "wrong", "polarised"]
    titles = ["Efficient consensus", "Wrong consensus", "Polarisation"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, nm, ti in zip(axes, names, titles):
        mean = d[f"{nm}_mstar_mean"]; lo = d[f"{nm}_mstar_lo"]; hi = d[f"{nm}_mstar_hi"]
        t = np.arange(mean.shape[0])
        for c, col, lab in [(0, C1, "community 1"), (1, C2, "community 2")]:
            ax.plot(t, mean[:, c], color=col, lw=1.8, label=lab)
            ax.fill_between(t, lo[:, c], hi[:, c], color=col, alpha=0.18, lw=0)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_title(ti); ax.set_xlabel("trial $t$"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(r"mass on optimal arm $m_{c,t}(a^\star)$")
    axes[0].legend(loc="center right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_baseline")


def fig3_phase():
    d = _load("phase")
    P, lams, gammas = d["P"], d["lams"], d["gammas"]
    ext = [lams.min(), lams.max(), gammas.min(), gammas.max()]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35))
    panels = [("efficient", 0, "Greens"), ("wrong", 1, "Reds"), ("polarised", 2, "Purples")]
    for ax, (lab, k, cmap) in zip(axes[:3], panels):
        im = ax.imshow(P[:, :, k], origin="lower", extent=ext, aspect="auto",
                       cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"P({lab})"); ax.set_xlabel(r"social weight $\lambda$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(False)
    axes[0].set_ylabel(r"permeability $\Gamma=1-B_{cc}$")
    # dominant-regime categorical map
    reg = P.argmax(axis=2)
    names = ["efficient", "wrong", "polarised", "unresolved"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([REG_COLOR[n] for n in names])
    axd = axes[3]
    axd.imshow(reg, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    axd.set_title("dominant regime"); axd.set_xlabel(r"$\lambda$"); axd.grid(False)
    axd.legend(handles=[Patch(facecolor=REG_COLOR[n], label=n) for n in names],
               loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False,
               fontsize=6.5, handlelength=1.0)
    fig.tight_layout()
    _save(fig, "fig3_phase")


SHORT_ABL = {"full": "full", "no-anticipatory": r"$-$antic.",
             "no-retrospective": r"$-$retro.", "no-confidence-wt": r"$-$cred.wt",
             "constant-lr": "const.lr", "alt-confidence": "alt.conf."}


def fig4_ablations():
    import json
    d = _load("ablations")
    order = [str(x) for x in d["order"]]
    R = int(d["R"])
    readable = {"full": "Full model", "no-anticipatory": "No anticipation",
                "no-retrospective": "No retrospective", "no-confidence-wt": "No credibility",
                "constant-lr": "Constant learning", "alt-confidence": "Alt. confidence"}
    labels = [readable[o] for o in order]
    x = np.arange(len(order))
    eff, wro, pol = d["p_efficient"], d["p_wrong"], d["p_polarised"]
    unr = np.clip(1 - eff - wro - pol, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3))
    # (a) regime composition
    ax = axes[0]; bottom = np.zeros(len(order))
    for arr, lab in [(eff, "efficient"), (wro, "wrong"),
                     (pol, "polarised"), (unr, "unresolved")]:
        ax.bar(x, arr, bottom=bottom, color=REG_COLOR[lab], label=lab, width=0.7)
        bottom += arr
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("regime probability"); ax.set_ylim(0, 1)
    ax.set_title("(a) Which outcome occurs?")
    legend_names = ["efficient", "wrong",
                    "polarised (0 throughout)" if np.allclose(pol, 0) else "polarised",
                    "unresolved"]
    ax.legend(handles=[Patch(facecolor=REG_COLOR[n.split()[0]], label=n)
                       for n in legend_names],
              loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False, fontsize=6.2); ax.grid(axis="x")
    # (b) regret with bootstrap CI
    ax = axes[1]
    reg, lo, hi = d["regret"], d["regret_lo"], d["regret_hi"]
    ax.bar(x, reg, color=["#222" if o == "full" else "#8c8c8c" for o in order], width=0.7)
    ax.errorbar(x, reg, yerr=[reg - lo, hi - reg], fmt="none", ecolor="k", capsize=2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(r"mean group regret $\mathcal{R}_T$")
    ax.set_title("(b) How costly are the choices?")
    # (c) paired counterfactual effect on P(wrong)
    ax = axes[2]
    con = json.load(open(f"{common.RESULTS}/ablations_{SCALE}.json"))["contrasts_vs_full"]
    names = [o for o in order if o != "full"]
    diff = np.array([con[n]["d_p_wrong"]["diff"] for n in names])
    dlo = np.array([con[n]["d_p_wrong"]["lo"] for n in names])
    dhi = np.array([con[n]["d_p_wrong"]["hi"] for n in names])
    y = np.arange(len(names))
    cols = ["#2a7a36" if v > 0 else "#b2182b" for v in diff]
    for yi, value, lo_i, hi_i, col in zip(y, diff, dlo, dhi, cols):
        ax.errorbar(value, yi, xerr=[[value - lo_i], [hi_i - value]], fmt="o",
                    color=col, mec="white", mew=.7, ms=6, ecolor=col,
                    capsize=2.5, lw=1.3, zorder=3)
        if lo_i > 0 or hi_i < 0:
            ax.text(hi_i + .025 if value > 0 else lo_i - .025, yi, "*",
                    color=col, fontsize=9, va="center",
                    ha="left" if value > 0 else "right")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([readable[n] for n in names], fontsize=6.8)
    ax.set_xlabel("change in P(wrong) when mechanism is restored")
    ax.set_title("(c) Which mechanisms amplify or restrain error?")
    ax.annotate("", xy=(.12, .035), xytext=(.20, .035), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#b2182b", lw=1.4))
    ax.annotate("", xy=(.32, .035), xytext=(.24, .035), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#2a7a36", lw=1.4))
    ax.text(.105, .07, "restrains", transform=ax.transAxes, color="#b2182b",
            fontsize=5.8, ha="center")
    ax.text(.315, .07, "amplifies", transform=ax.transAxes, color="#2a7a36",
            fontsize=5.8, ha="center")
    fig.suptitle(f"What changes when each mechanism is removed? ($R={R}$)",
                 y=1.02, fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig4_ablations")


def fig5_confidence():
    d = _load("confidence")
    bins = np.linspace(0, 1, 31)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ce = d["correct_early"]; we = d["wrong_early"]
    cl = d["correct_late"]; wl = d["wrong_late"]
    a1.hist(ce, bins=bins, density=True, color=C1, alpha=0.6, label="correct choice")
    a1.hist(we, bins=bins, density=True, color="#b2182b", alpha=0.6, label="wrong choice")
    a1.axvline(ce.mean(), color=C1, lw=1.4, ls="--")
    a1.axvline(we.mean(), color="#b2182b", lw=1.4, ls="--")
    a1.set_title("Early phase ($t<T/2$)"); a1.set_xlabel("confidence $C$")
    a1.set_ylabel("density"); a1.legend(frameon=False)
    a2.hist(cl, bins=bins, density=True, color=C1, alpha=0.6, label="correct choice")
    a2.hist(wl, bins=bins, density=True, color="#b2182b", alpha=0.6, label="wrong choice")
    a2.axvline(cl.mean(), color=C1, lw=1.4, ls="--")
    a2.axvline(wl.mean(), color="#b2182b", lw=1.4, ls="--")
    a2.set_title("Late phase ($t\\geq T/2$)"); a2.set_xlabel("confidence $C$")
    a2.legend(frameon=False)
    fig.suptitle("Confidence attaches to wrong choices, especially early", y=1.02, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig5_confidence")


def fig6_quotient():
    from collections import deque
    dg = _load("quotient_grid")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    from matplotlib.colors import ListedColormap
    regime_names = ["efficient", "wrong", "polarised", "unresolved"]
    regime_cmap = ListedColormap([REG_COLOR[n] for n in regime_names])
    lams, gammas = dg["lams"], dg["gammas"]
    ext = [lams.min(), lams.max(), gammas.min(), gammas.max()]
    mismatch = dg["micro_dom"] != dg["meso_dom"]

    # (a,b) Direct visual comparison of benchmark and quotient regime maps.
    for ax, arr, title in [
        (axes[0, 0], dg["micro_dom"], "(a) Agent simulations define the benchmark"),
        (axes[0, 1], dg["meso_dom"],
         f"(b) The quotient differs in {mismatch.sum()} of {mismatch.size} cells"),
    ]:
        ax.imshow(arr, origin="lower", extent=ext, aspect="auto",
                  cmap=regime_cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_xlabel(r"social influence $\lambda$"); ax.grid(False)
        ax.set_title(title, fontsize=8.5)
    axes[0, 0].set_ylabel(r"permeability $\Gamma$")
    gi, li = np.where(mismatch)
    axes[0, 1].scatter(lams[li], gammas[gi], marker="x", s=18, color="black",
                       linewidth=.75, label="different regime", zorder=3)
    axes[0, 1].legend(loc="upper right", fontsize=5.8, frameon=True,
                      facecolor="white", edgecolor=".8")

    # (c) Quantify reliability by distance from a micro-model regime boundary.
    ax = axes[1, 0]
    regime = dg["micro_dom"]
    dist = np.full(regime.shape, -1, int); queue = deque()
    for i, j in np.ndindex(regime.shape):
        if any(0 <= i + di < regime.shape[0] and 0 <= j + dj < regime.shape[1]
               and regime[i + di, j + dj] != regime[i, j]
               for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            dist[i, j] = 0; queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < dist.shape[0] and 0 <= jj < dist.shape[1] and dist[ii, jj] < 0:
                dist[ii, jj] = dist[i, j] + 1; queue.append((ii, jj))
    masks = [dist <= 1, dist == 2, dist >= 3]
    labels = ["boundary or adjacent", "interior", "deep interior"]
    agreement = 100 * np.array([np.mean(dg["agree"][m]) for m in masks])
    counts = [int(m.sum()) for m in masks]
    colors = [plotstyle.PALETTE["orange"], REG_COLOR["efficient"], REG_COLOR["efficient"]]
    y = np.arange(3)
    for yi, value, col in zip(y, agreement, colors):
        ax.hlines(yi, 0, value, color=".72", lw=1.1, zorder=1)
        ax.plot(value, yi, "o", color=col, ms=7, mec="white", mew=.7, zorder=2)
    for yi, value, n in zip(y, agreement, counts):
        ax.text(min(value + 2, 102), yi, f"{value:.0f}%", color=".15",
                fontsize=7.2, weight="bold", ha="left", va="center")
        ax.text(2, yi + .17, f"N = {n} cells", fontsize=5.8, color=".4", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_ylim(2.4, -.4)
    ax.set_xlim(0, 105); ax.set_xlabel("same predicted regime (%)")
    ax.set_title("(c) Agreement rises from 71% to 100% inside regimes", fontsize=8.1)

    # (d) Locate the endpoint error and the agent-model regime transitions.
    axd = axes[1, 1]
    disc = dg["disc"]
    im = axd.imshow(100 * disc, origin="lower", extent=ext, aspect="auto", cmap="magma")
    axd.set_title("(d) Endpoint errors peak along the transition region", fontsize=8.3)
    axd.set_xlabel(r"social influence $\lambda$"); axd.set_ylabel(r"permeability $\Gamma$"); axd.grid(False)
    cb = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)
    cb.set_label("endpoint error (percentage points)", fontsize=7.2)
    fig.suptitle("The community model is reliable away from regime boundaries", fontsize=10.5)
    fig.legend(handles=[Patch(facecolor=REG_COLOR[n], label=n) for n in regime_names],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=4,
               frameon=False, fontsize=6.3)
    fig.tight_layout(rect=(0, .055, 1, .97))
    _save(fig, "fig6_quotient")


def fig6b_aggregation_dynamics():
    """Dynamic micro--aggregate fit, separated from Figure 6's regime validation."""
    dt = _load("quotient_traj")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharey=True)
    cols = [C1, C2, REG_COLOR["efficient"], plotstyle.PALETTE["purple"]]

    # (a,b) Two-community model-fit profiles.
    for ax, nm, title in [(axes[0], "efficient", "(a) Efficient consensus:\nquotient converges faster"),
                          (axes[1], "wrong", "(b) Wrong consensus:\nquotient lock-in is stronger")]:
        micro = _log10_odds(dt[f"{nm}_micro_mean"].mean(axis=1))
        meso = _log10_odds(dt[f"{nm}_meso"].mean(axis=1))
        t = np.arange(len(micro))
        ax.plot(t, micro, color=".15", lw=1.5, label="agent ensemble mean")
        ax.plot(t, meso, color=C1, lw=1.35, ls="--", label="community model")
        ax.axhline(0, color=".45", lw=.75, ls=":")
        ax.set_xlabel("trial $t$"); ax.set_ylim(-5.1, 4.1)
        ax.set_title(title, fontsize=8.3)

    # (c) Four-community modular stress test.
    ax = axes[2]
    t = np.arange(dt["many_micro_mean"].shape[0])
    for c, col in enumerate(cols):
        micro_odds = _log10_odds(dt["many_micro_mean"][:, c])
        meso_odds = _log10_odds(dt["many_meso"][:, c])
        ax.plot(t, micro_odds, color=col, lw=1.5,
                label=f"community {c+1}")
        ax.plot(t, meso_odds, color=col, lw=1.25, ls="--")
    ax.axhline(0, color=".45", lw=.75, ls=":")
    ax.set_xlabel("trial $t$")
    ax.set_title("(c) Four communities:\nsame direction, persistent tail gap", fontsize=8.3)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, .02), fontsize=5.5,
              frameon=False, ncol=2)
    axes[0].set_yticks([-4, -2, 0, 2, 4])
    axes[0].set_yticklabels(["0.01", "1", "50", "99", "99.99"])
    axes[0].set_ylabel("choosing the better option (%)\n(log-odds spacing)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015),
               ncol=2, frameon=False, fontsize=6.2)
    fig.suptitle("The quotient reproduces regime direction but not tail probabilities",
                 fontsize=10.3)
    fig.text(.995, .015, r"display floor: $p=10^{-5}$", ha="right", fontsize=5.4, color=".4")
    fig.tight_layout(rect=(0, .09, 1, .94))
    _save(fig, "fig6b_aggregation_dynamics")


def fig7_robustness():
    import json
    s = json.load(open(f"{common.RESULTS}/robustness_{SCALE}.json"))
    REGS = ["efficient", "wrong", "polarised", "unresolved"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    # (a) R1 threshold invariance
    ax = axes[0, 0]; thr = ["0.8", "0.9", "0.95"]; xs = np.arange(len(thr)); w = 0.2
    for i, r in enumerate(REGS):
        ax.bar(xs + (i - 1.5) * w, [s["R1"][t][r] for t in thr], w, color=REG_COLOR[r], label=r)
    ax.set_xticks(xs); ax.set_xticklabels(thr); ax.set_xlabel("terminal threshold")
    ax.set_ylabel("share of phase grid"); ax.set_ylim(0, 0.75)
    ax.set_title("(a) Conclusions do not depend on cutoff")
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    # (b) R2 horizon  (Wilson 95% CIs)
    ax = axes[0, 1]; Ts = [200, 300, 500, 1000]
    for pt, col, lab in [("near_zero", "#9e9e9e", "near-zero transmission"),
                         ("corrective", "#2a7a36", "corrective point"),
                         ("distortive", "#b2182b", "distortive point")]:
        d = [s["R2"][f"{pt}_T{T}"] for T in Ts]
        y = np.array([e["wrong"] for e in d])
        lo = np.array([e["wrong_lo"] for e in d]); hi = np.array([e["wrong_hi"] for e in d])
        ax.errorbar(Ts, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    fmt="-o", color=col, ms=3, capsize=2, elinewidth=0.9, label=lab)
    ax.set_xlabel("horizon $T$"); ax.set_ylabel("P(wrong)"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) Regimes persist as the horizon grows")
    ax.legend(fontsize=5.6, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.18), ncol=2)
    ax.text(225, .89, "wrong consensus", color="#b2182b", fontsize=6.2)
    ax.text(225, .07, "no wrong consensus", color="#2a7a36", fontsize=6.2)
    # (c) R3 initial-lead strength  (Wilson 95% CIs)
    ax = axes[0, 2]; deltas = [0.0, 0.04, 0.08, 0.12, 0.16]
    dd = [s["R3"][f"lead_{d}"] for d in deltas]
    for key, klo, khi, col, lab in [("wrong", "wrong_lo", "wrong_hi", "#b2182b", "P(wrong)"),
                                    ("eff", "eff_lo", "eff_hi", "#2a7a36", "P(eff)")]:
        y = np.array([e[key] for e in dd])
        lo = np.array([e[klo] for e in dd]); hi = np.array([e[khi] for e in dd])
        ax.errorbar(deltas, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    fmt="-o", color=col, ms=3, capsize=2, elinewidth=0.9, label=lab)
    ax.set_xlabel(r"early wrong-lead $\delta$"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(c) A small wrong lead triggers lock-in"); ax.legend(fontsize=6, frameon=False)
    # (d) R4 reward gap
    ax = axes[1, 0]; gaps = [0.02, 0.05, 0.1, 0.2]
    pw = np.array([s["R4"][f"gap_{g}"]["wrong"] for g in gaps])
    lo = np.array([s["R4"][f"gap_{g}"]["wrong_lo"] for g in gaps])
    hi = np.array([s["R4"][f"gap_{g}"]["wrong_hi"] for g in gaps])
    ax.errorbar(gaps, pw, yerr=[np.clip(pw - lo, 0, None), np.clip(hi - pw, 0, None)],
                fmt="-o", color="#b2182b", ms=3, capsize=2)
    ax.set_xlabel(r"reward gap $|\Delta\mu|$"); ax.set_ylabel("P(wrong)"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(d) Reward gaps:\nwrong lock-in persists", fontsize=8.5)
    ax.text(.025, .90, "P(wrong) remains 1", color="#b2182b", fontsize=6.5)
    # (e) R5 random graphs  (Wilson 95% CIs)
    ax = axes[1, 1]; gt = ["balanced", "weighted_sbm", "noisy_W"]; xs = np.arange(3); w = 0.35
    for sgn, pref, col, lab, marker in [(-1, "corrective", "#2a7a36", "corrective point", "o"),
                                        (+1, "distortive", "#b2182b", "distortive point", "D")]:
        e = [s["R5"][f"{pref}_{g}"] for g in gt]
        y = np.array([v["wrong"] for v in e])
        lo = np.array([v["wrong_lo"] for v in e]); hi = np.array([v["wrong_hi"] for v in e])
        ax.errorbar(xs + sgn * .08, y,
                    yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    fmt=marker, color=col, mfc=col, mec="white", mew=.7, ms=6,
                    capsize=3, elinewidth=1.2, label=lab, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(["balanced", "SBM", "noisy"], fontsize=7)
    ax.set_ylabel("P(wrong)"); ax.set_ylim(0, 1.05)
    ax.set_title("(e) Network perturbations:\noutcomes persist", fontsize=8.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.18), ncol=2)
    # (f) R6 grid resolution
    ax = axes[1, 2]; grids = ["7x7", "9x9", "13x13"]; xs = np.arange(3); w = 0.2
    for i, r in enumerate(REGS):
        ax.bar(xs + (i - 1.5) * w, [s["R6"][gr][r] for gr in grids], w, color=REG_COLOR[r])
    ax.set_xticks(xs); ax.set_xticklabels(grids, fontsize=7); ax.set_xlabel("phase grid")
    ax.set_ylabel("share of phase grid"); ax.set_ylim(0, 0.75)
    ax.set_title("(f) Conclusions survive finer grids")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("The qualitative conclusions survive alternative definitions and settings",
                 y=1.01, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig7_robustness")


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


if __name__ == "__main__":
    SCALE = common.get_scale()
    fig2_baseline()
    fig3_phase()
    fig4_ablations()
    fig5_confidence()
    fig6_quotient()
    fig6b_aggregation_dynamics()
    fig7_robustness()
    print("all figures done")
