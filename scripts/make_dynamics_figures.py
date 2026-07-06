"""Dynamical-systems figures from the deterministic mean-field reduction:

  fig12_bifurcation.pdf   pitchfork in the early-lead order parameter vs lambda
  fig13_vectorfield.pdf   two-community lead phase portraits below/above lambda*
  fig14_phase_meanfield.pdf   21x21 (lambda, Gamma) amplification phase heatmap

All quantities are computed from credibility.meanfield (no Monte Carlo); they
are the analytic companion to the stochastic phase diagram.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import plotstyle
from credibility import meanfield as mf
from credibility.config import base_params

plotstyle.use()
P = PALETTE = plotstyle.PALETTE
GREEN, RED, BLUE, GREY, PURPLE = P["green"], P["red"], P["blue"], P["grey"], P["purple"]
PALETTE_INK = P["ink"]


def _tint(hexcol, f=0.55):
    """Lighten a colour toward white by fraction ``f`` (for pastel backgrounds)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexcol)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _base():
    # small reward gap: the corrective reward signal is weak, so anticipatory
    # amplification can decide the outcome (the regime of Proposition 3).
    return base_params("standard").replace(mu=(0.53, 0.47), b_self=0.85, eta=0.12)


# ==========================================================================
def fig_bifurcation():
    p = _base()
    lstar, C0 = mf.lambda_star(p)
    lams = np.linspace(0.0, 1.5, 76)
    zst = mf.amplification_branch(p, lams, z0=1e-3)      # stable |z*|
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    # shaded regime backdrop
    ax.axvspan(0, lstar, color=GREEN, alpha=0.05)
    ax.axvspan(lstar, 1.5, color=RED, alpha=0.05)
    # z=0 branch: stable below lstar (solid), unstable above (dashed)
    ax.plot(lams[lams < lstar], np.zeros((lams < lstar).sum()), color=PALETTE_INK, lw=1.8)
    ax.plot(lams[lams >= lstar], np.zeros((lams >= lstar).sum()), color=PALETTE_INK, lw=1.2, ls=(0, (4, 3)))
    # amplified pair +/- z*
    on = zst > 1e-4
    ax.plot(lams[on], zst[on], color=GREEN, lw=2.6, label=r"corrective consensus $+z^\star$", solid_capstyle="butt")
    ax.plot(lams[on], -zst[on], color=RED, lw=2.6, label=r"distortive consensus $-z^\star$", solid_capstyle="butt")
    ax.scatter([lams[on][0]], [zst[on][0]], s=14, color=GREEN, zorder=5)
    ax.scatter([lams[on][0]], [-zst[on][0]], s=14, color=RED, zorder=5)
    ax.axvline(lstar, color=BLUE, ls=":", lw=1.5, label=r"threshold $\lambda^\star=\sigma^2/(\beta a C_0\bar b)$")
    ax.axhline(0, color=PALETTE_INK, lw=0.5, alpha=0.3)
    ax.set_xlabel(r"anticipatory weight $\lambda$")
    ax.set_ylabel(r"early-lead order parameter $z^\star$")
    ax.set_xlim(0, 1.5); ax.set_ylim(-0.66, 0.72)
    ax.set_xticks([0.0, 0.5, lstar, 1.0, 1.5])
    ax.set_xticklabels(["0.0", "0.5", r"$\lambda^\star$", "1.0", "1.5"])
    ax.set_title(r"Amplification bifurcation of the early-lead order parameter")
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig12_bifurcation.{ext}")
    plt.close(fig)
    print("wrote fig12_bifurcation")


# ==========================================================================
def _fixed_points(p, lam):
    """Iterate the lead map from a spread of seeds; return unique fixed points."""
    B = p.coupling(); seeds = []
    for z1 in (-0.4, 0.0, 0.4):
        for z2 in (-0.4, 0.0, 0.4):
            z = np.array([z1, z2])
            for _ in range(600):
                zn = mf._amp_step(z, p, lam, B)
                if np.max(np.abs(zn - z)) < 1e-11:
                    break
                z = zn
            seeds.append(z)
    fps = []
    for z in seeds:
        if not any(np.allclose(z, f, atol=1e-3) for f in fps):
            fps.append(z)
    return fps


def fig_flowfield():
    """Two-panel flow portrait: streamlines coloured by speed + fixed points,
    below and above the amplification threshold."""
    from matplotlib.lines import Line2D
    p = _base()
    lstar, _ = mf.lambda_star(p)
    gv = np.linspace(-0.52, 0.52, 25)
    Z1, Z2 = np.meshgrid(gv, gv)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    for ax, lam, tag in [(axes[0], 0.20, r"$\lambda\ll\lambda^\star$: neutral state stable"),
                         (axes[1], 1.15, r"$\lambda>\lambda^\star$: any lead is amplified")]:
        dZ1, dZ2 = mf.amplification_field(Z1, Z2, p, lam)
        speed = np.hypot(dZ1, dZ2)
        strm = ax.streamplot(gv, gv, dZ1, dZ2, color=speed, cmap="viridis",
                             density=1.15, linewidth=0.9, arrowsize=0.9)
        for f in _fixed_points(p, lam):
            if np.max(np.abs(f)) < 0.05:
                col = PALETTE_INK
            elif f[0] > 0 and f[1] > 0:
                col = GREEN
            elif f[0] < 0 and f[1] < 0:
                col = RED
            else:
                col = PURPLE
            ax.plot(f[0], f[1], "o", ms=9, mfc=col, mec="white", mew=1.1, zorder=6)
        ax.axhline(0, color=PALETTE_INK, lw=0.5, alpha=0.28)
        ax.axvline(0, color=PALETTE_INK, lw=0.5, alpha=0.28)
        ax.set_xlabel(r"community 1 early lead $z_1$")
        ax.set_ylabel(r"community 2 early lead $z_2$")
        ax.set_title(tag, fontsize=9.5)
        ax.set_xlim(-0.58, 0.58); ax.set_ylim(-0.58, 0.58)
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.grid(False); ax.set_aspect("equal")
    handles = [Line2D([0], [0], marker="o", ls="", mfc=PALETTE_INK, mec="white", ms=8, label="neutral"),
               Line2D([0], [0], marker="o", ls="", mfc=GREEN, mec="white", ms=8, label="corrective"),
               Line2D([0], [0], marker="o", ls="", mfc=RED, mec="white", ms=8, label="distortive"),
               Line2D([0], [0], marker="o", ls="", mfc=PURPLE, mec="white", ms=8, label="polarised")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    cbar = fig.colorbar(strm.lines, ax=axes, fraction=0.038, pad=0.02)
    cbar.set_label("flow speed", fontsize=8.5)
    fig.suptitle(r"Early-lead flow field (fixed points) below and above $\lambda^\star$", y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig13_flowfield.{ext}")
    plt.close(fig)
    print("wrote fig13_flowfield")


def fig_vectorfield():
    """Phase-portrait sequence across lambda: flow + basins of attraction.

    The (z1, z2) early-lead plane is shaded by the attractor each point reaches
    (neutral / corrective / distortive / polarised) with the streamlines and
    fixed points overlaid, at three values of lambda spanning the amplification
    threshold.  This is the mean-field story as a portrait: below lambda* the
    whole plane flows to the neutral state; above it, the plane partitions into
    basins so the early lead decides the outcome.
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, BoundaryNorm
    p = _base()
    lstar, _ = mf.lambda_star(p)
    gb = np.linspace(-0.56, 0.56, 181)          # fine grid for basin shading
    gq = np.linspace(-0.5, 0.5, 21)             # coarse grid for streamlines
    Q1, Q2 = np.meshgrid(gq, gq)
    basin_cmap = ListedColormap(["#f2f2f2", _tint(GREEN), _tint(RED), _tint(PURPLE)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], basin_cmap.N)
    lams = [(0.20, r"$\lambda=0.2\ll\lambda^\star$"),
            (0.90, r"$\lambda=0.9>\lambda^\star$"),
            (1.35, r"$\lambda=1.35\gg\lambda^\star$")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6))
    for k, (ax, (lam, tag)) in enumerate(zip(axes, lams)):
        code = mf.basin_map(p, lam, gb)
        ax.pcolormesh(gb, gb, code, cmap=basin_cmap, norm=norm, shading="nearest",
                      rasterized=True, zorder=0)
        dZ1, dZ2 = mf.amplification_field(Q1, Q2, p, lam)
        ax.streamplot(gq, gq, dZ1, dZ2, color=PALETTE_INK, density=1.0,
                      linewidth=0.6, arrowsize=0.7, zorder=1)
        for f in _fixed_points(p, lam):
            if np.max(np.abs(f)) < 0.05:
                col = PALETTE_INK
            elif f[0] > 0 and f[1] > 0:
                col = GREEN
            elif f[0] < 0 and f[1] < 0:
                col = RED
            else:
                col = PURPLE
            ax.plot(f[0], f[1], "o", ms=8.5, mfc=col, mec="white", mew=1.1, zorder=6)
        ax.set_xlabel(r"community 1 early lead $z_1$")
        if k == 0:
            ax.set_ylabel(r"community 2 early lead $z_2$")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"({'abc'[k]})  {tag}", loc="left", fontsize=9.5)
        ax.set_xlim(-0.56, 0.56); ax.set_ylim(-0.56, 0.56)
        ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
        ax.grid(False); ax.set_aspect("equal")
    handles = [Line2D([0], [0], marker="s", ls="", mfc="#f2f2f2", mec="0.6", ms=9, label="neutral basin"),
               Line2D([0], [0], marker="s", ls="", mfc=_tint(GREEN), mec="0.6", ms=9, label="corrective basin"),
               Line2D([0], [0], marker="s", ls="", mfc=_tint(RED), mec="0.6", ms=9, label="distortive basin"),
               Line2D([0], [0], marker="s", ls="", mfc=_tint(PURPLE), mec="0.6", ms=9, label="polarised basin")]
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.20, wspace=0.10)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle(r"Early-lead phase portrait across the amplification threshold $\lambda^\star$", y=0.96)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig13_vectorfield.{ext}")
    plt.close(fig)
    print("wrote fig13_vectorfield")


# ==========================================================================
def fig_phase_meanfield(grid=21):
    """Mean-field phase diagram over (lambda, Gamma), gradient panels.

    Early leads are finite-population fluctuations: for each cell we sample a
    distribution of small early leads (community 2 biased wrong) and record the
    fraction of the mean-field basins ending corrective / polarised / distortive.
    Layout mirrors the Monte-Carlo diagram: three probability gradients plus the
    dominant regime.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    p0 = _base()
    lstar, _ = mf.lambda_star(p0)
    lams = np.linspace(0.0, 1.6, grid)
    gammas = np.linspace(0.01, 0.5, grid)
    Pc = np.empty((grid, grid)); Pp = np.empty((grid, grid)); Pd = np.empty((grid, grid))
    for gi, G in enumerate(gammas):
        pr = mf.basin_probabilities(p0.replace(b_self=float(1 - G)), lams, n=500)
        Pc[gi], Pp[gi], Pd[gi] = pr[:, 0], pr[:, 1], pr[:, 2]

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.35), constrained_layout=True)
    ink = PALETTE["ink"]
    panels = [(axes[0], Pc, "Greens", "a", "corrective"),
              (axes[1], Pp, "Purples", "b", "polarised"),
              (axes[2], Pd, "Reds", "c", "distortive")]
    for ax, Z, cmap, tag, name in panels:
        ax.grid(False)
        # flat cells at the true 21x21 resolution -- no spatial interpolation
        im = ax.pcolormesh(lams, gammas, Z, cmap=cmap, vmin=0, vmax=1,
                           shading="nearest", rasterized=True)
        # a single boundary contour only where a regime is the majority (P>0.5);
        # regimes that never reach 0.5 get no manufactured boundary
        if Z.max() > 0.5:
            ax.contour(lams, gammas, Z, levels=[0.5], colors=ink, linewidths=0.9)
        ax.axvline(lstar, color=ink, ls=(0, (4, 3)), lw=1.0, alpha=0.5)
        ax.set_title(rf"($\mathrm{{{tag}}}$)  $P(\mathrm{{{name}}})$", loc="left", fontsize=9.5)
        ax.set_xlabel(r"social weight $\lambda$")
        ax.set_xticks([0.0, lstar, 1.5]); ax.set_xticklabels(["0", r"$\lambda^\star$", "1.5"])
        ax.set_ylim(gammas.min(), gammas.max())
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=[0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel(r"permeability $\Gamma=1-B_{cc}$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # (d) majority regime: coloured only where one regime exceeds 0.5, else grey
    ax = axes[3]; ax.grid(False)
    stack = np.stack([Pc, Pp, Pd], axis=-1)
    dom = np.where(stack.max(axis=-1) >= 0.5, np.argmax(stack, axis=-1), 3)
    cmap = ListedColormap([PALETTE["green"], PALETTE["purple"], PALETTE["red"], "#e3e3e3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.pcolormesh(lams, gammas, dom, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    ax.axvline(lstar, color=ink, ls=(0, (4, 3)), lw=1.1, alpha=0.7)
    ax.set_title(r"($\mathrm{d}$)  majority regime ($P>0.5$)", loc="left", fontsize=9.5)
    ax.set_xlabel(r"social weight $\lambda$")
    ax.set_xticks([0.0, lstar, 1.5]); ax.set_xticklabels(["0", r"$\lambda^\star$", "1.5"])
    ax.set_yticklabels([]); ax.set_ylim(gammas.min(), gammas.max())
    lbl = {0: (PALETTE["green"], "corrective"), 1: (PALETTE["purple"], "polarised"),
           2: (PALETTE["red"], "distortive"), 3: ("#e3e3e3", "no clear majority")}
    present = sorted(set(int(v) for v in np.unique(dom)))
    handles = [Patch(fc=lbl[c][0], ec="0.6" if c == 3 else "none", label=lbl[c][1]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(present), fontsize=9,
               frameon=False, handlelength=1.2, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(r"Mean-field phase diagram: fraction of fluctuating early leads reaching each regime",
                 fontsize=10.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig14_phase_meanfield.{ext}")
    plt.close(fig)
    print(f"wrote fig14_phase_meanfield ({grid}x{grid})")


if __name__ == "__main__":
    import sys
    g = 21
    for a in sys.argv[1:]:
        if a.isdigit():
            g = int(a)
    fig_bifurcation()
    fig_flowfield()
    fig_vectorfield()
    # fig_phase_meanfield (the (lambda, Gamma) probability diagram) is retained
    # in the module but no longer featured; the phase portrait tells the story.
    if "--diagram" in sys.argv:
        fig_phase_meanfield(grid=g)
