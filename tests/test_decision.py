"""The decision layer is the object the framework claims; it has to be tested like one."""

import numpy as np
import pandas as pd
import pytest

from armd import decision


AGENTS = ["Ciprofloxacin", "Levofloxacin", "Nitrofurantoin", "Meropenem"]
CLASSES = {"Ciprofloxacin": "Fluoroquinolone", "Levofloxacin": "Fluoroquinolone",
           "Nitrofurantoin": "Nitrofuran", "Meropenem": "Carbapenem"}


def test_fuse_recovers_each_pure_rule_at_the_endpoints():
    p_fail = np.array([0.2, 0.8])
    coll = np.array([0.9, 0.1])
    assert decision.fuse(p_fail, coll, 1.0) == pytest.approx(p_fail)
    assert decision.fuse(p_fail, coll, 0.0) == pytest.approx(coll)
    assert decision.fuse(p_fail, coll, 0.5) == pytest.approx((p_fail + coll) / 2)


def test_fuse_rejects_a_weight_outside_the_unit_interval():
    with pytest.raises(ValueError):
        decision.fuse(np.array([0.1]), np.array([0.1]), 1.5)


def test_a_class_without_an_estimate_is_ineligible_rather_than_free():
    """The failure mode worth guarding: an unestimated class scoring as zero collateral
    cost would be recommended *because* nothing is known about it."""
    cost, eligible = decision.agent_collateral_vector(
        AGENTS, CLASSES, {"Fluoroquinolone": 0.10, "Nitrofuran": 0.02}
    )
    assert eligible.tolist() == [True, True, True, False]
    assert np.isnan(cost[3])
    assert cost[0] == pytest.approx(0.10)


def test_a_protective_point_estimate_is_floored_at_zero():
    cost, _ = decision.agent_collateral_vector(
        ["Nitrofurantoin"], CLASSES, {"Nitrofuran": -0.04}
    )
    assert cost[0] == 0.0


def test_recommendation_shifts_away_from_the_costly_class_as_lambda_falls():
    p_res = np.array([[0.10, 0.90, 0.15, 0.90]])
    effects = {"Fluoroquinolone": 0.20, "Nitrofuran": 0.01, "Carbapenem": 0.02}
    cost, eligible = decision.agent_collateral_vector(AGENTS, CLASSES, effects)

    pure = decision.recommend(p_res, cost, eligible, lam=1.0)
    fused = decision.recommend(p_res, cost, eligible, lam=0.5)
    assert AGENTS[pure[0]] == "Ciprofloxacin"
    assert AGENTS[fused[0]] == "Nitrofurantoin"


def test_the_recommender_minimises_resistance_rather_than_susceptibility():
    """The recommender picks the agent least likely to fail.

    The head emits P(resistant). Treating it as P(susceptible) would flip the argmin and
    recommend the agent most likely to fail while still producing a smooth frontier.
    """
    p_res = np.array([[0.02, 0.97, 0.96, 0.95]])
    effects = {"Fluoroquinolone": 0.05, "Nitrofuran": 0.05, "Carbapenem": 0.05}
    cost, eligible = decision.agent_collateral_vector(AGENTS, CLASSES, effects)
    choice = decision.recommend(p_res, cost, eligible, lam=1.0)
    assert AGENTS[choice[0]] == "Ciprofloxacin", "recommender picked a likely-resistant agent"

    curve = decision.frontier(p_res, AGENTS, CLASSES, effects, lambdas=(1.0,))
    assert curve["mean_p_failure"].iloc[0] == pytest.approx(0.02)


def test_frontier_trades_immediate_efficacy_for_collateral_monotonically():
    rng = np.random.default_rng(0)
    p_res = rng.uniform(0.01, 0.7, size=(400, len(AGENTS)))
    effects = {"Fluoroquinolone": 0.18, "Nitrofuran": 0.01, "Carbapenem": 0.03}

    curve = decision.frontier(p_res, AGENTS, CLASSES, effects,
                              lambdas=(0.0, 0.25, 0.5, 0.75, 1.0))
    assert list(curve["lambda"]) == [0.0, 0.25, 0.5, 0.75, 1.0]

    assert curve["mean_p_failure"].is_monotonic_decreasing
    assert curve["mean_collateral"].is_monotonic_increasing
    assert curve.loc[curve["lambda"] == 1.0, "switched_from_pure_susceptibility"].iloc[0] == 0.0


def test_frontier_is_empty_when_no_class_has_an_estimate():
    p_res = np.zeros((5, len(AGENTS)))
    curve = decision.frontier(p_res, AGENTS, CLASSES, {})
    assert curve.empty


def test_observed_check_reports_how_little_of_it_is_scorable():
    """The check is weak by construction and must say so through n_scored."""
    p_res = np.array([[0.1, 0.9, 0.8, 0.9], [0.1, 0.9, 0.8, 0.9]])
    y = np.zeros((2, len(AGENTS)))
    mask = np.zeros((2, len(AGENTS)))
    mask[0, 0] = 1
    res = decision.observed_failure_check(
        p_res, y, mask, AGENTS, CLASSES, {"Fluoroquinolone": 0.05}, lam=1.0
    )
    assert res["n_scored"] == 1
    assert res["fraction_of_units_scorable"] == pytest.approx(0.5)


def test_an_agent_the_model_never_scored_is_not_recommended():
    """NaN columns must not win the argmin.

    A per-agent baseline leaves NaN columns for agents that did not clear the support guards,
    and NaN comparisons are all false. The eligibility mask must exclude any agent without a
    finite prediction for every unit.
    """
    p_res = np.array([[0.30, np.nan, 0.40, 0.50],
                      [0.35, np.nan, 0.45, 0.55]])
    effects = {"Fluoroquinolone": 0.05, "Nitrofuran": 0.02, "Carbapenem": 0.02}
    cost, eligible = decision.agent_collateral_vector(AGENTS, CLASSES, effects)

    curve = decision.frontier(p_res, AGENTS, CLASSES, effects, lambdas=(0.5, 1.0))
    assert not curve.empty
    assert curve["mean_p_failure"].notna().all(), "a NaN column reached the argmin"
    assert curve["top_agent"].iloc[-1] != "Levofloxacin"


def test_a_fully_unscored_panel_yields_an_empty_frontier():
    p_res = np.full((3, len(AGENTS)), np.nan)
    curve = decision.frontier(p_res, AGENTS, CLASSES, {"Fluoroquinolone": 0.05})
    assert curve.empty
