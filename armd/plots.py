"""Figures. Every one is saved to the results directory and none of them pops a window."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, GREY = "#1f6fb2", "#c4761f", "#2e7d57", "#8a8a8a"


def _save(fig, results_dir: str, name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def loss_curves(history: dict, results_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(history["train"], label="train", color=BLUE)
    ax.plot(history["val"], label="validation", color=ORANGE)
    ax.set_xlabel("epoch")
    ax.set_ylabel("joint loss")
    ax.legend(frameon=False)
    ax.set_title("Training and validation loss")
    return _save(fig, results_dir, "loss_curves.png")


def auc_forest(metrics: pd.DataFrame, results_dir: str, name: str = "auc_forest.png") -> str:
    """Per-agent AUC with its interval, ordered by support.

    A bare histogram of point estimates hides that the extreme values all come from
    agents with a handful of test cultures; this does not.
    """
    ok = metrics[metrics["status"] == "ok"].sort_values("n_test")
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.24 * len(ok))))
    y = np.arange(len(ok))
    ax.hlines(y, ok["auc_lo"], ok["auc_hi"], color=GREY, lw=1.2)
    ax.plot(ok["auc"], y, "o", color=BLUE, ms=4)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  (n={n:,})" for a, n in zip(ok["antibiotic"], ok["n_test"])], fontsize=7)
    ax.set_xlabel("test AUC with 95% bootstrap interval")
    ax.set_title("Per-agent discrimination, ordered by test support")
    return _save(fig, results_dir, name)


def net_vs_baseline(compare: pd.DataFrame, results_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    sizes = 12 + 40 * (compare["n_test"] / compare["n_test"].max())
    ax.scatter(compare["xgb_auc"], compare["net_auc"], s=sizes, color=GREEN,
               alpha=0.8, edgecolor="white", linewidth=0.6)
    lo = float(min(compare[["xgb_auc", "net_auc"]].min())) - 0.02
    ax.plot([lo, 1.0], [lo, 1.0], "--", color=GREY, lw=1)
    ax.set_xlabel("per-agent gradient boosting AUC")
    ax.set_ylabel("shared network AUC")
    ax.set_title("Discrimination, network versus baseline\n(marker area is test support)", fontsize=10)
    return _save(fig, results_dir, "net_vs_xgb.png")


def propensity_overlap(pi: np.ndarray, a: np.ndarray, results_dir: str,
                       name: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 31)
    ax.hist(pi[a == 1], bins=bins, alpha=0.6, label="exposed", color=BLUE, density=True)
    ax.hist(pi[a == 0], bins=bins, alpha=0.6, label="unexposed", color=ORANGE, density=True)
    ax.set_xlabel("estimated P(exposed | X)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, results_dir, name)


def causal_forest(labels: Sequence[str], est: Sequence[float], lo: Sequence[float],
                  hi: Sequence[float], results_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 0.7 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    for yy, e, l, h in zip(y, est, lo, hi):
        ax.plot([l, h], [yy, yy], color=BLUE, lw=1.6)
        ax.plot(e, yy, "o", color="#16527f", ms=6)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("risk difference for collateral resistance")
    ax.set_title("Effect estimates with 95% intervals")
    return _save(fig, results_dir, "causal_estimates.png")


def calibration(tables: dict, results_dir: str) -> Optional[str]:
    if not tables:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect")
    for name, tab in tables.items():
        if len(tab):
            ax.plot(tab["predicted"], tab["observed"], "o-", ms=4, lw=1.2, label=name)
    ax.set_xlabel("predicted probability of resistance")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration on the test period")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, results_dir, "calibration.png")


def love_plot(before: np.ndarray, after: np.ndarray, results_dir: str) -> str:
    """Standardised differences before and after weighting, the conventional balance check."""
    ok = np.isfinite(before) & np.isfinite(after)
    b, a = np.abs(before[ok]), np.abs(after[ok])
    order = np.argsort(b)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y = np.arange(len(order))
    ax.plot(b[order], y, "o", ms=3, color=ORANGE, label="unadjusted")
    ax.plot(a[order], y, "o", ms=3, color=BLUE, label="weighted")
    ax.axvline(0.1, color=GREY, ls="--", lw=1)
    ax.set_yticks([])
    ax.set_xlabel("absolute standardised mean difference")
    ax.set_title("Covariate balance (dashed line at 0.10)")
    ax.legend(frameon=False)
    return _save(fig, results_dir, "balance.png")


def decision_frontier(curve: pd.DataFrame, results_dir: str,
                      default_lambda: Optional[float] = None) -> Optional[str]:
    """The trade-off the decision layer traces, drawn in the space it trades in.

    Plotted parametrically: immediate risk against downstream cost, with lambda as the
    label on each point rather than as an axis. Putting lambda on x and the two outcomes
    on twin y-axes would be the obvious alternative and would be wrong -- two scales on
    one frame invite a comparison between quantities that have no common unit, and the
    shape a reader needs here is the frontier itself, which only exists in this space.
    """
    if curve is None or not len(curve):
        return None
    x = curve["mean_collateral"].to_numpy(dtype=float)
    y = curve["mean_p_failure"].to_numpy(dtype=float)
    lam = curve["lambda"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, y, "-", color=BLUE, linewidth=2, zorder=2)
    ax.scatter(x, y, s=42, color=BLUE, zorder=3, edgecolor="white", linewidth=1.2)

    if default_lambda is not None:
        hit = np.isclose(lam, default_lambda)
        if hit.any():
            ax.scatter(x[hit], y[hit], s=120, color=ORANGE, zorder=4,
                       edgecolor="white", linewidth=1.5)

    wanted = {float(lam.min()), float(lam.max())}
    if default_lambda is not None:
        wanted.add(float(default_lambda))
    for xi, yi, li in zip(x, y, lam):
        if not any(np.isclose(li, w) for w in wanted):
            continue
        right_edge = np.isclose(li, lam.max())
        ax.annotate(
            f"$\\lambda$={li:g}", (xi, yi), textcoords="offset points",
            xytext=(-10, 10) if right_edge else (8, 6),
            ha="right" if right_edge else "left",
            fontsize=8.5, color="#333333",
        )

    ax.annotate("stewardship only", (x[0], y[0]), textcoords="offset points",
                xytext=(10, -4), fontsize=8, color=GREY, style="italic")
    ax.annotate("susceptibility only", (x[-1], y[-1]), textcoords="offset points",
                xytext=(-10, -14), ha="right", fontsize=8, color=GREY, style="italic")

    ax.set_xlabel("mean collateral cost of the recommended agent\n(risk difference)")
    ax.set_ylabel("mean predicted probability of failure")
    ax.set_title("What stewardship weight buys and costs")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _save(fig, results_dir, "decision_frontier.png")


def collateral_feasibility(probe: pd.DataFrame, results_dir: str,
                           name: str = "collateral_feasibility.png") -> str:
    """The screened space of exposure-to-outcome contrasts, and the network that survives.

    Two panels because they answer different questions. The matrix shows every candidate
    cell and *why* it fails, which is the more informative half: the pattern of failure is
    concentrated in overlap, not sample size. The graph shows only the survivors, which is
    what a reader wants to carry away.

    Nothing here encodes an effect. The screen deliberately stops at overlap, so the only
    thing a filled cell asserts is that the contrast could be estimated.
    """
    d = probe.copy()
    d["kind"] = np.where(
        d["status"] == "estimable", "estimable",
        np.where(d["status"] == "positivity", "positivity",
                 np.where(d["status"].astype(str).str.startswith("no outcome"),
                          "no outcome", "support")))

    expo = sorted(d["exposure"].unique())
    outs = sorted(d["outcome"].unique())
    code = {"estimable": 3, "positivity": 2, "support": 1, "no outcome": 0}
    M = np.zeros((len(expo), len(outs)))
    for r in d.itertuples():
        M[expo.index(r.exposure), outs.index(r.outcome)] = code[r.kind]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 6.0),
                                 gridspec_kw={"width_ratios": [1.28, 1]})

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f2f2f2", GREY, ORANGE, BLUE])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(outs)))
    ax.set_xticklabels(outs, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(expo)))
    ax.set_yticklabels(expo, fontsize=7.5)
    ax.set_xlabel("resistance outcome at the next culture", fontsize=9, labelpad=8)
    ax.set_ylabel("prior exposure", fontsize=9)
    ax.set_title("(a) every screened contrast", fontsize=10, loc="left")
    for i in range(len(expo)):
        if expo[i] in outs:
            ax.plot(outs.index(expo[i]), i, marker="o", ms=3.2, mfc="white",
                    mec="#333333", mew=0.8, zorder=3)
    ax.set_xticks(np.arange(-.5, len(outs), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(expo), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#cccccc", lw=0.5) for c in
               (BLUE, ORANGE, GREY, "#f2f2f2")]

    ok = d[d["status"] == "estimable"]
    nodes = sorted(set(ok["exposure"]) | set(ok["outcome"]))
    outdeg = {n: int((ok["exposure"] == n).sum()) for n in nodes}
    nodes.sort(key=lambda n: (-outdeg[n], n))
    ang = {n: 2 * np.pi * i / len(nodes) - np.pi / 2 for i, n in enumerate(nodes)}
    P = {n: (np.cos(a), np.sin(a)) for n, a in ang.items()}

    for r in ok.itertuples():
        x0, y0 = P[r.exposure]; x1, y1 = P[r.outcome]
        q = 1.0 - float(r.clipped_fraction)
        if r.exposure == r.outcome:
            a = ang[r.exposure]
            bx.plot([x0, x0 * 1.16], [y0, y0 * 1.16], color=BLUE, lw=0.8,
                    alpha=0.25 + 0.5 * q, solid_capstyle="round")
            continue
        cx, cy = (x0 + x1) * 0.5 * 0.42, (y0 + y1) * 0.5 * 0.42
        t = np.linspace(0, 1, 40)
        bez_x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        bez_y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        bx.plot(bez_x, bez_y, color=BLUE, lw=0.35 + 0.9 * q, alpha=0.16 + 0.42 * q,
                solid_capstyle="round", zorder=1)

    for n in nodes:
        x, y = P[n]
        is_expo = outdeg[n] > 0
        bx.scatter([x], [y], s=26 + 15 * outdeg[n], zorder=3,
                   facecolor=BLUE if is_expo else "white", edgecolor=BLUE, linewidth=1.1)
        a = ang[n]
        ha = "left" if np.cos(a) > -0.01 else "right"
        pad = 0.085 + 0.0045 * outdeg[n]
        bx.text(x + np.cos(a) * pad, y + np.sin(a) * pad, n, fontsize=7.2, ha=ha,
                va="center", color="#333333" if is_expo else GREY)

    bx.set_xlim(-1.72, 1.72); bx.set_ylim(-1.34, 1.30)
    bx.set_aspect("equal"); bx.axis("off")
    bx.set_title("(b) the contrasts that survive", fontsize=10, loc="left")
    bx.text(0, -1.26, "filled = estimable as an exposure   \u00b7   size = contrasts supported\n"
                      "line weight = overlap quality   \u00b7   no effect sizes are shown",
            ha="center", va="top", fontsize=7.6, color=GREY)

    fig.legend(handles, ["estimable", "fails positivity", "too little support", "no outcome"],
               loc="lower center", bbox_to_anchor=(0.34, 0.005), ncol=4, frameon=False,
               fontsize=8.5, handlelength=1.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
