"""Doubly robust estimation of the collateral-resistance effect.

Folds are grouped by patient (``StratifiedGroupKFold``), so no patient contributes to both
the nuisance fit and the held-out scoring. Standard errors are clustered on the patient,
because one patient contributes many cultures and their influence-function contributions
are correlated. The E-value for a confidence limit is 1 when that limit crosses the null.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass
class CausalEstimate:
    estimate: float
    se: float
    lo: float
    hi: float
    n: int
    n_clusters: int
    se_iid: float
    clipped_fraction: float
    baseline_risk: float
    label: str = ""

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in asdict(self).items()}


def aipw_influence(
    a: np.ndarray, y: np.ndarray, pi: np.ndarray, m0: np.ndarray, m1: np.ndarray
) -> np.ndarray:
    """Per-unit contribution to the augmented inverse probability weighted estimator."""
    return m1 - m0 + a * (y - m1) / pi - (1 - a) * (y - m0) / (1 - pi)


def clustered_se(psi: np.ndarray, groups: Optional[np.ndarray]) -> tuple[float, int]:
    """Cluster-robust standard error of the mean of ``psi``.

    With one observation per cluster this reduces exactly to the iid standard error.
    """
    n = len(psi)
    if groups is None:
        return float(psi.std(ddof=1) / np.sqrt(n)), n
    centred = psi - psi.mean()
    codes, uniques = _factorize(groups)
    g = len(uniques)
    sums = np.bincount(codes, weights=centred, minlength=g)
    if g < 2:
        return float("nan"), g
    se = float(np.sqrt(g / (g - 1.0) * (sums ** 2).sum()) / n)
    return se, g


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniques, codes = np.unique(values, return_inverse=True)
    return codes, uniques


def _default_learner(seed: int = 0):
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.06,
        l2_regularization=1.0, random_state=seed,
    )


def _fit_predict(X_tr, y_tr, X_te, learner_factory, seed):
    """Fit a probability model, degrading gracefully to the base rate when degenerate."""
    if len(y_tr) == 0:
        return np.zeros(len(X_te))
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), float(y_tr.mean()))
    clf = learner_factory(seed)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def crossfit_aipw(
    X: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_folds: int = 5,
    clip: float = 0.025,
    seed: int = 0,
    learner_factory: Callable = _default_learner,
    label: str = "",
    representer: Optional[Callable[[np.ndarray, np.ndarray], tuple]] = None,
) -> CausalEstimate:
    """Cross-fitted AIPW with patient-grouped folds and a cluster-robust interval.

    Every unit is scored by nuisance models that never saw it, nor any other culture
    from the same patient. The propensity is clipped into ``[clip, 1-clip]`` and the
    clipped fraction is reported, because clipping buys stability at the price of bias
    and a reader is entitled to know how much of it was used.

    ``representer`` optionally replaces the raw covariates with a *learned* adjustment
    set. It is called once per fold as ``representer(train_rows, test_rows)`` and must
    return ``(X_train, X_test)``, having fitted whatever it learns on the training rows
    alone. This is not a convenience: a representation fitted on all of the data and then
    handed to a cross-fitted estimator has already seen every held-out fold, which voids
    the guarantee cross-fitting exists to provide. Fitting it inside the fold keeps the
    held-out units genuinely held out.
    """
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("no units supplied to crossfit_aipw")
    if len(np.unique(a)) < 2:
        raise ValueError("treatment is constant; no contrast is identified")

    pi = np.zeros(n)
    m0 = np.zeros(n)
    m1 = np.zeros(n)

    if groups is not None:
        n_groups = len(np.unique(groups))
        folds = min(n_folds, n_groups)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, a.astype(int), groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, a.astype(int))

    for tr, te in split_iter:
        if representer is None:
            Xtr, Xte = X[tr], X[te]
        else:
            Xtr, Xte = representer(tr, te)
            Xtr = np.asarray(Xtr, dtype=float)
            Xte = np.asarray(Xte, dtype=float)
            if len(Xtr) != len(tr) or len(Xte) != len(te):
                raise ValueError(
                    f"representer returned {len(Xtr)}/{len(Xte)} rows for a fold of "
                    f"{len(tr)}/{len(te)}; it must preserve row order and count"
                )
        in1 = a[tr] == 1
        in0 = ~in1
        pi[te] = _fit_predict(Xtr, a[tr].astype(int), Xte, learner_factory, seed)
        m1[te] = _fit_predict(Xtr[in1], y[tr][in1].astype(int), Xte, learner_factory, seed)
        m0[te] = _fit_predict(Xtr[in0], y[tr][in0].astype(int), Xte, learner_factory, seed)

    clipped = float(np.mean((pi < clip) | (pi > 1 - clip)))
    pi = np.clip(pi, clip, 1 - clip)

    psi = aipw_influence(a, y, pi, m0, m1)
    est = float(psi.mean())
    se_iid = float(psi.std(ddof=1) / np.sqrt(n))
    se, n_clusters = clustered_se(psi, groups)

    untreated = y[a == 0]
    baseline = float(untreated.mean()) if len(untreated) else float("nan")

    return CausalEstimate(
        estimate=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
        n=n, n_clusters=n_clusters, se_iid=se_iid,
        clipped_fraction=clipped, baseline_risk=baseline, label=label,
    )


def crossfit_propensity(
    X: np.ndarray,
    a: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_folds: int = 5,
    seed: int = 0,
    learner_factory: Callable = _default_learner,
) -> np.ndarray:
    """Raw out-of-fold propensity scores on patient-grouped folds.

    Kept separate from the estimator so that overlap can be measured without computing an
    effect. The scores are returned unclipped: a clipped vector has no values outside the
    bounds, so an overlap diagnostic computed on it would always report perfect overlap.
    Callers clip after measuring.
    """
    a = np.asarray(a, dtype=float)
    pi = np.zeros(len(a))
    if groups is not None:
        folds = min(n_folds, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, a.astype(int), groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, a.astype(int))
    for tr, te in split_iter:
        pi[te] = _fit_predict(X[tr], a[tr].astype(int), X[te], learner_factory, seed)
    return pi


def bootstrap_ci(
    a: np.ndarray, y: np.ndarray, pi: np.ndarray, m0: np.ndarray, m1: np.ndarray,
    groups: Optional[np.ndarray] = None, n_boot: int = 500, seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval, resampling patients when grouped.

    Resampling patients rather than cultures keeps the interval valid under repeated
    measurements.
    """
    rng = np.random.default_rng(seed)
    point = float(aipw_influence(a, y, pi, m0, m1).mean())
    psi = aipw_influence(a, y, pi, m0, m1)
    if groups is None:
        n = len(psi)
        draws = [psi[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    else:
        codes, uniques = _factorize(groups)
        by_group = [np.where(codes == g)[0] for g in range(len(uniques))]
        ng = len(by_group)
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, ng, ng)
            idx = np.concatenate([by_group[g] for g in pick])
            draws.append(psi[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def e_value_rr(rr: float) -> float:
    """E-value for a risk ratio: the joint association an unmeasured confounder would
    need with both exposure and outcome to explain the estimate away."""
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def rr_from_rd(rd: float, baseline_risk: float) -> float:
    """Approximate risk ratio implied by a risk difference over an unexposed baseline."""
    p0 = float(np.clip(baseline_risk, 1e-6, 1 - 1e-6))
    p1 = float(np.clip(p0 + rd, 1e-6, 1 - 1e-6))
    return p1 / p0


def e_value_point(rd: float, baseline_risk: float) -> float:
    return e_value_rr(rr_from_rd(rd, baseline_risk))


def e_value_bound(estimate: float, lo: float, hi: float, baseline_risk: float) -> float:
    """E-value for the confidence limit nearer the null.

    Returns 1.0 when that limit crosses the null: an interval compatible with no effect needs
    no confounding to be explained away.
    """
    if not np.isfinite(estimate) or not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    bound = lo if estimate > 0 else hi
    if (estimate > 0 and bound <= 0) or (estimate < 0 and bound >= 0) or estimate == 0:
        return 1.0
    return e_value_rr(rr_from_rd(bound, baseline_risk))


def overlap_summary(pi: np.ndarray, a: np.ndarray) -> dict:
    """Numbers that make a positivity claim checkable rather than visual.

    The common-support share is the fraction of units whose propensity lies inside the
    range covered by both arms; a figure showing two overlapping histograms does not
    by itself establish this.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    p1, p0 = pi[a == 1], pi[a == 0]
    if not len(p1) or not len(p0):
        return {"common_support_fraction": float("nan")}
    lo = max(p1.min(), p0.min())
    hi = min(p1.max(), p0.max())
    inside = float(np.mean((pi >= lo) & (pi <= hi)))
    return {
        "exposed_n": int(len(p1)),
        "unexposed_n": int(len(p0)),
        "exposed_propensity_mean": float(p1.mean()),
        "unexposed_propensity_mean": float(p0.mean()),
        "common_support_low": float(lo),
        "common_support_high": float(hi),
        "common_support_fraction": inside,
    }


def standardised_differences(X: np.ndarray, a: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-covariate standardised mean difference between arms, optionally weighted.

    Reported before and after inverse probability weighting as the covariate balance check.
    """
    a = np.asarray(a, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    t, c = a == 1, a == 0
    if not t.any() or not c.any():
        return np.full(X.shape[1], np.nan)

    def _mean(mask):
        ww = w[mask]
        return (X[mask] * ww[:, None]).sum(axis=0) / max(ww.sum(), 1e-12)

    def _var(mask, mu):
        ww = w[mask]
        return ((X[mask] - mu) ** 2 * ww[:, None]).sum(axis=0) / max(ww.sum(), 1e-12)

    m1, m0 = _mean(t), _mean(c)
    v1, v0 = _var(t, m1), _var(c, m0)
    pooled = np.sqrt((v1 + v0) / 2.0)
    pooled = np.where(pooled < 1e-12, np.nan, pooled)
    return (m1 - m0) / pooled
