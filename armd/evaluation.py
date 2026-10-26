"""Susceptibility metrics with uncertainty, and calibration.

Every metric carries a bootstrap interval, and the same support guards are applied to the
network and to the baseline, so the comparison is over the same set of agents.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def bootstrap_metric(y: np.ndarray, p: np.ndarray, metric, n_boot: int = 500, seed: int = 0):
    """Percentile interval for a metric, skipping resamples with a single class."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        draws.append(metric(yy, p[idx]))
    if not draws:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def per_agent_metrics(
    y: np.ndarray,
    mask: np.ndarray,
    proba: np.ndarray,
    agents: Sequence[str],
    min_events: int = 25,
    n_boot: int = 500,
    seed: int = 0,
    restrict_to: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Discrimination, average precision and calibration for every evaluable agent.

    ``restrict_to`` narrows the set of agents, which is how the network is held to the
    same support guards as the gradient boosting baseline.
    """
    keep = set(restrict_to) if restrict_to is not None else None
    rows = []
    for i, agent in enumerate(agents):
        if keep is not None and agent not in keep:
            continue
        m = mask[:, i] > 0
        yv = y[m, i].astype(int)
        if m.sum() < min_events or len(np.unique(yv)) < 2:
            rows.append({
                "antibiotic": agent, "n_test": int(m.sum()),
                "resistance_rate": float(yv.mean()) if len(yv) else np.nan,
                "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "auprc": np.nan, "brier": np.nan,
                "status": "insufficient_support" if m.sum() < min_events else "single_class",
            })
            continue
        p = proba[m, i]
        auc = roc_auc_score(yv, p)
        lo, hi = bootstrap_metric(yv, p, roc_auc_score, n_boot=n_boot, seed=seed + i)
        rows.append({
            "antibiotic": agent, "n_test": int(m.sum()), "resistance_rate": float(yv.mean()),
            "auc": float(auc), "auc_lo": lo, "auc_hi": hi,
            "auprc": float(average_precision_score(yv, p)),
            "brier": float(brier_score_loss(yv, p)), "status": "ok",
        })
    return pd.DataFrame(rows)


def support_weighted_mean(metrics: pd.DataFrame, column: str = "auc") -> float:
    ok = metrics[metrics["status"] == "ok"]
    if not len(ok):
        return float("nan")
    return float(np.average(ok[column], weights=ok["n_test"]))


def evaluable_agents(
    y: np.ndarray, mask: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray,
    agents: Sequence[str], min_train_events: int, min_train_positives: int, min_test_events: int,
) -> list[str]:
    """Agents that clear the support guards on both the training and the test period.

    Applying this once and using it for every model keeps the head-to-head comparison like
    for like.
    """
    out = []
    for i, agent in enumerate(agents):
        mtr = mask[train_rows, i] > 0
        mte = mask[test_rows, i] > 0
        ytr = y[train_rows, i][mtr]
        yte = y[test_rows, i][mte]
        if mtr.sum() < min_train_events or ytr.sum() < min_train_positives:
            continue
        if len(np.unique(ytr)) < 2 or mte.sum() < min_test_events or len(np.unique(yte)) < 2:
            continue
        out.append(agent)
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed versus predicted risk in equal-count bins."""
    if len(y) < n_bins * 2:
        return pd.DataFrame(columns=["bin", "n", "predicted", "observed"])
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return pd.DataFrame([
        {"bin": k, "n": len(b), "predicted": float(p[b].mean()), "observed": float(y[b].mean())}
        for k, b in enumerate(bins)
    ])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    tab = calibration_table(y, p, n_bins)
    if not len(tab):
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["observed"] - tab["predicted"]).abs()).sum())


def paired_bootstrap_delta(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Interval for the AUC difference between two models on the same units.

    The claim "model A beats model B on 7 of 34 agents" is a tally of point estimates
    and says nothing about whether any single difference is distinguishable from noise.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        draws.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    if not draws:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
        "lo": float(lo), "hi": float(hi),
    }
