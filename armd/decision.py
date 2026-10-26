"""The decision layer: fusing immediate failure risk with downstream collateral cost.

This is the object the framework is actually for. The susceptibility head answers "will
this drug work now"; the causal heads answer "what does giving it cost later". Neither
is a recommendation on its own, and the fusion below is the smallest rule that turns the
pair into one.

The two terms are deliberately on the same scale. ``p_fail`` is a probability of
treatment failure; ``collateral`` is an absolute *increase* in the probability that the
patient's next infection resists the class, which is a risk difference and therefore also
a probability mass. A unit of one is comparable to a unit of the other, which is what
makes the weighted sum meaningful rather than a dimensionless score.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def fuse(p_fail: np.ndarray, collateral: np.ndarray, lam: float) -> np.ndarray:
    """L = lam * P(failure now) + (1 - lam) * P(collateral later).

    ``lam = 1`` recovers a pure susceptibility recommender, which is what current
    predictive decision support does. ``lam = 0`` recovers a pure stewardship rule that
    ignores whether the drug works. Neither endpoint is clinically sensible; the interest
    is in what happens between them.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], got {lam}")
    return lam * p_fail + (1.0 - lam) * collateral


def agent_collateral_vector(
    agents: Sequence[str],
    agent_class: Mapping[str, str],
    class_effects: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-agent collateral cost, and a mask of which agents carry a causal estimate.

    An agent inherits the estimate of its drug class: the causal question is asked at
    class grain because exposure is recorded at class grain. Agents whose class has no
    estimate are not eligible for recommendation -- scoring them as zero cost would make
    an unestimated class look maximally safe, which is precisely backwards.
    """
    cost = np.full(len(agents), np.nan, dtype=float)
    for i, ag in enumerate(agents):
        cls = agent_class.get(ag)
        if cls is not None and cls in class_effects:
            cost[i] = max(float(class_effects[cls]), 0.0)
    return cost, ~np.isnan(cost)


def _predictable(p_resistant: np.ndarray) -> np.ndarray:
    """Agents the susceptibility model scored for every unit.

    A per-agent baseline only fits agents that cleared the support guards, so its matrix
    carries NaN columns for the rest. Those columns must not reach the argmin: NaN comparisons
    are all false, so an unscored agent would be selected for every unit.
    """
    return np.isfinite(p_resistant).all(axis=0)


def recommend(
    p_resistant: np.ndarray,
    collateral: np.ndarray,
    eligible: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Index of the recommended agent for each unit under weight ``lam``.

    ``p_resistant`` is the probability that the organism resists the agent, which is the
    probability of treatment failure, so it enters the fusion directly.
    """
    p_fail = p_resistant[:, eligible]
    cost = np.broadcast_to(collateral[eligible], p_fail.shape)
    scores = fuse(p_fail, cost, lam)
    local = np.argmin(scores, axis=1)
    return np.flatnonzero(eligible)[local]


def frontier(
    p_resistant: np.ndarray,
    agents: Sequence[str],
    agent_class: Mapping[str, str],
    class_effects: Mapping[str, float],
    lambdas: Sequence[float] = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0),
) -> pd.DataFrame:
    """Sweep the trade-off weight and report what each setting would prescribe.

    The result is the frontier the decision layer traces: how much immediate efficacy a
    given appetite for stewardship costs, and how much collateral resistance it avoids.
    Everything here is computed from quantities available at decision time.
    """
    collateral, eligible = agent_collateral_vector(agents, agent_class, class_effects)
    eligible = eligible & _predictable(p_resistant)
    if not eligible.any():
        return pd.DataFrame(columns=["lambda", "mean_p_failure", "mean_collateral",
                                     "switched_from_pure_susceptibility", "top_agent",
                                     "n_distinct_agents"])
    baseline = recommend(p_resistant, collateral, eligible, lam=1.0)

    rows = []
    for lam in lambdas:
        choice = recommend(p_resistant, collateral, eligible, lam)
        p_fail = p_resistant[np.arange(len(choice)), choice]
        names, counts = np.unique(choice, return_counts=True)
        rows.append({
            "lambda": lam,
            "mean_p_failure": float(p_fail.mean()),
            "mean_collateral": float(np.nanmean(collateral[choice])),
            "switched_from_pure_susceptibility": float(np.mean(choice != baseline)),
            "top_agent": agents[int(names[np.argmax(counts)])],
            "n_distinct_agents": int(len(names)),
        })
    return pd.DataFrame(rows)


def observed_failure_check(
    p_resistant: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    agents: Sequence[str],
    agent_class: Mapping[str, str],
    class_effects: Mapping[str, float],
    lam: float,
) -> dict:
    """Observed resistance rate of the recommended agent, where it happened to be tested.

    This is a weak check and is reported as one. Susceptibility is observed only for
    agents a clinician chose to test, so the subset where the recommendation can be
    scored is not a random subset of units -- the same verification bias that affects the
    per-agent metrics affects this. It is included because a decision rule that never
    touches an observed outcome is unfalsifiable, not because it settles anything.
    """
    collateral, eligible = agent_collateral_vector(agents, agent_class, class_effects)
    eligible = eligible & _predictable(p_resistant)
    if not eligible.any():
        return {"n_scored": 0}
    choice = recommend(p_resistant, collateral, eligible, lam)
    rows = np.arange(len(choice))
    tested = mask[rows, choice] > 0
    if not tested.any():
        return {"n_scored": 0}
    return {
        "n_scored": int(tested.sum()),
        "fraction_of_units_scorable": float(tested.mean()),
        "observed_resistance_rate_of_recommendation": float(y_true[rows, choice][tested].mean()),
    }
