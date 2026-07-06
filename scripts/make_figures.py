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
    labels = [SHORT_ABL.get(o, o) for o in order]
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
    ax.set_title("(a) Regime composition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              frameon=False, fontsize=6.5); ax.grid(axis="x")
    # (b) regret with bootstrap CI
    ax = axes[1]
    reg, lo, hi = d["regret"], d["regret_lo"], d["regret_hi"]
    ax.bar(x, reg, color=["#222" if o == "full" else "#8c8c8c" for o in order], width=0.7)
    ax.errorbar(x, reg, yerr=[reg - lo, hi - reg], fmt="none", ecolor="k", capsize=2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(r"mean group regret $\mathcal{R}_T$")
    ax.set_title("(b) Group regret (95% bootstrap CI)")
    # (c) paired counterfactual effect on P(wrong)
    ax = axes[2]
    con = json.load(open(f"{common.RESULTS}/ablations_{SCALE}.json"))["contrasts_vs_full"]
    names = [o for o in order if o != "full"]
    diff = np.array([con[n]["d_p_wrong"]["diff"] for n in names])
    dlo = np.array([con[n]["d_p_wrong"]["lo"] for n in names])
    dhi = np.array([con[n]["d_p_wrong"]["hi"] for n in names])
    y = np.arange(len(names))
    cols = ["#2a7a36" if v > 0 else "#b2182b" for v in diff]
    ax.barh(y, diff, color=cols, height=0.62)
    ax.errorbar(diff, y, xerr=[diff - dlo, dhi - diff], fmt="none", ecolor="k", capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([SHORT_ABL[n] for n in names], fontsize=7)
    ax.set_xlabel(r"$\Delta\,$P(wrong) $=$ full $-$ ablation")
    ax.set_title("(c) Counterfactual effect (paired)")
    fig.suptitle(f"Mechanism isolation under common random numbers ($R={R}$)",
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
    dt = _load("quotient_traj")
    dg = _load("quotient_grid")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    # (a) efficient, (b) wrong : 2-community micro vs meso
    for ax, nm, ti in [(axes[0, 0], "efficient", "Two communities: efficient"),
                       (axes[0, 1], "wrong", "Two communities: wrong consensus")]:
        micro = dt[f"{nm}_micro_mean"]; meso = dt[f"{nm}_meso"]
        t = np.arange(micro.shape[0])
        for c, col in [(0, C1), (1, C2)]:
            ax.plot(t, micro[:, c], color=col, lw=1.8,
                    label=f"micro c{c+1}" if c == 0 else None)
            ax.plot(t, meso[:, c], color=col, lw=1.3, ls="--")
        ax.set_title(ti); ax.set_xlabel("trial $t$")
        ax.set_ylabel(r"$m_{c,t}(a^\star)$"); ax.set_ylim(-0.02, 1.02)
    axes[0, 0].plot([], [], color="k", lw=1.8, label="micro (agent mean)")
    axes[0, 0].plot([], [], color="k", lw=1.3, ls="--", label="meso (quotient)")
    axes[0, 0].legend(loc="center right", frameon=False, fontsize=7)
    # (c) four-community modular
    axc = axes[1, 0]
    micro4 = dt["many_micro_mean"]; meso4 = dt["many_meso"]
    t = np.arange(micro4.shape[0])
    cols = ["#1f5fa8", "#d2691e", "#2a7a36", "#6a51a3"]
    for c in range(micro4.shape[1]):
        axc.plot(t, micro4[:, c], color=cols[c], lw=1.6)
        axc.plot(t, meso4[:, c], color=cols[c], lw=1.1, ls="--")
    axc.set_title("Four communities (modular ring)")
    axc.set_xlabel("trial $t$"); axc.set_ylabel(r"$m_{c,t}(a^\star)$"); axc.set_ylim(-0.02, 1.02)
    # (d) discrepancy heatmap
    axd = axes[1, 1]
    disc, lams, gammas = dg["disc"], dg["lams"], dg["gammas"]
    ext = [lams.min(), lams.max(), gammas.min(), gammas.max()]
    im = axd.imshow(disc, origin="lower", extent=ext, aspect="auto", cmap="magma")
    axd.set_title(r"micro--meso discrepancy $|\,\bar m^{\rm micro}-m^{\rm meso}|$")
    axd.set_xlabel(r"$\lambda$"); axd.set_ylabel(r"$\Gamma$"); axd.grid(False)
    fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "fig6_quotient")


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
    ax.set_ylabel("regime area fraction"); ax.set_ylim(0, 0.75); ax.set_title("(a) R1 threshold")
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    # (b) R2 horizon  (Wilson 95% CIs)
    ax = axes[0, 1]; Ts = [200, 300, 500, 1000]
    for pt, col in [("near_zero", "#9e9e9e"), ("corrective", "#2a7a36"), ("distortive", "#b2182b")]:
        d = [s["R2"][f"{pt}_T{T}"] for T in Ts]
        y = np.array([e["wrong"] for e in d])
        lo = np.array([e["wrong_lo"] for e in d]); hi = np.array([e["wrong_hi"] for e in d])
        ax.errorbar(Ts, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    fmt="-o", color=col, ms=3, capsize=2, elinewidth=0.9, label=pt)
    ax.set_xlabel("horizon $T$"); ax.set_ylabel("P(wrong)"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) R2 horizon"); ax.legend(fontsize=6, frameon=False)
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
    ax.set_title("(c) R3 initial conditions"); ax.legend(fontsize=6, frameon=False)
    # (d) R4 reward gap
    ax = axes[1, 0]; gaps = [0.02, 0.05, 0.1, 0.2]
    pw = np.array([s["R4"][f"gap_{g}"]["wrong"] for g in gaps])
    lo = np.array([s["R4"][f"gap_{g}"]["wrong_lo"] for g in gaps])
    hi = np.array([s["R4"][f"gap_{g}"]["wrong_hi"] for g in gaps])
    ax.errorbar(gaps, pw, yerr=[np.clip(pw - lo, 0, None), np.clip(hi - pw, 0, None)],
                fmt="-o", color="#b2182b", ms=3, capsize=2)
    ax.set_xlabel(r"reward gap $|\Delta\mu|$"); ax.set_ylabel("P(wrong)"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(d) R4 reward gap")
    # (e) R5 random graphs  (Wilson 95% CIs)
    ax = axes[1, 1]; gt = ["balanced", "weighted_sbm", "noisy_W"]; xs = np.arange(3); w = 0.35
    for sgn, pref, col, lab in [(-1, "corrective", "#2a7a36", "corrective"),
                                (+1, "distortive", "#b2182b", "distortive")]:
        e = [s["R5"][f"{pref}_{g}"] for g in gt]
        y = np.array([v["wrong"] for v in e])
        lo = np.array([v["wrong_lo"] for v in e]); hi = np.array([v["wrong_hi"] for v in e])
        ax.bar(xs + sgn * w / 2, y, w, color=col, label=lab,
               yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
               error_kw=dict(ecolor="k", elinewidth=0.9, capsize=2))
    ax.set_xticks(xs); ax.set_xticklabels(["balanced", "SBM", "noisy"], fontsize=7)
    ax.set_ylabel("P(wrong)"); ax.set_ylim(0, 1.05); ax.set_title("(e) R5 random graphs")
    ax.legend(fontsize=6, frameon=False)
    # (f) R6 grid resolution
    ax = axes[1, 2]; grids = ["7x7", "9x9", "13x13"]; xs = np.arange(3); w = 0.2
    for i, r in enumerate(REGS):
        ax.bar(xs + (i - 1.5) * w, [s["R6"][gr][r] for gr in grids], w, color=REG_COLOR[r])
    ax.set_xticks(xs); ax.set_xticklabels(grids, fontsize=7); ax.set_xlabel("phase grid")
    ax.set_ylabel("regime area fraction"); ax.set_ylim(0, 0.75); ax.set_title("(f) R6 grid resolution")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("Robustness of the phase structure", y=1.01, fontsize=10)
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
    fig7_robustness()
    print("all figures done")
