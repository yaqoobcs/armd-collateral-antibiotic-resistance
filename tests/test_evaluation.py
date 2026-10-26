import numpy as np
import pandas as pd
import pytest

from armd.evaluation import (
    calibration_table, evaluable_agents, expected_calibration_error,
    paired_bootstrap_delta, per_agent_metrics, support_weighted_mean,
)


def test_perfect_and_random_predictions():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.binomial(1, 0.3, (n, 2)).astype(float)
    mask = np.ones((n, 2))
    proba = np.zeros((n, 2))
    proba[:, 0] = y[:, 0] * 0.9 + 0.05
    proba[:, 1] = rng.uniform(size=n)
    m = per_agent_metrics(y, mask, proba, ["perfect", "noise"], n_boot=100)
    assert m.loc[m.antibiotic == "perfect", "auc"].iloc[0] == pytest.approx(1.0)
    assert 0.4 < m.loc[m.antibiotic == "noise", "auc"].iloc[0] < 0.6
    assert m.loc[m.antibiotic == "noise", "auc_lo"].iloc[0] < 0.5 < m.loc[m.antibiotic == "noise", "auc_hi"].iloc[0]


def test_small_support_is_flagged_not_scored():
    y = np.array([[1.0], [0.0], [1.0]])
    mask = np.ones((3, 1))
    proba = np.array([[0.9], [0.1], [0.8]])
    m = per_agent_metrics(y, mask, proba, ["rare"], min_events=25)
    assert m["status"].iloc[0] == "insufficient_support"
    assert np.isnan(m["auc"].iloc[0])


def test_untested_entries_are_excluded():
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    mask = np.array([[1.0], [1.0], [0.0], [0.0]])
    proba = np.array([[0.9], [0.1], [0.0], [1.0]])
    m = per_agent_metrics(y, mask, proba, ["a"], min_events=2, n_boot=20)
    assert m["n_test"].iloc[0] == 2
    assert m["auc"].iloc[0] == pytest.approx(1.0)


def test_support_weighted_mean_weights_by_n():
    m = pd.DataFrame({"antibiotic": ["a", "b"], "n_test": [1000, 10],
                      "auc": [0.9, 0.5], "status": ["ok", "ok"]})
    assert support_weighted_mean(m) == pytest.approx((1000 * 0.9 + 10 * 0.5) / 1010)


def test_support_weighted_mean_ignores_unscored_agents():
    m = pd.DataFrame({"antibiotic": ["a", "b"], "n_test": [1000, 10],
                      "auc": [0.9, np.nan], "status": ["ok", "insufficient_support"]})
    assert support_weighted_mean(m) == pytest.approx(0.9)


def test_evaluable_agents_applies_both_period_guards():
    rng = np.random.default_rng(0)
    n = 600
    agents = ["plenty", "no_train", "no_test", "all_negative"]
    mask = np.zeros((n, 4))
    y = np.zeros((n, 4))
    train = np.arange(0, 400)
    test = np.arange(400, 600)
    mask[:, 0] = 1
    y[:, 0] = rng.binomial(1, 0.4, n)
    mask[test, 1] = 1
    y[test, 1] = rng.binomial(1, 0.4, len(test))
    mask[train, 2] = 1
    y[train, 2] = rng.binomial(1, 0.4, len(train))
    mask[:, 3] = 1
    keep = evaluable_agents(y, mask, train, test, agents, 100, 10, 25)
    assert keep == ["plenty"]


def test_calibration_detects_a_miscalibrated_model():
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p_true).astype(float)
    good = expected_calibration_error(y, p_true)
    bad = expected_calibration_error(y, np.clip(p_true * 0.4, 0, 1))
    assert good < 0.03
    assert bad > good * 3
    assert len(calibration_table(y, p_true)) == 10


def test_paired_delta_interval_covers_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = rng.uniform(size=500)
    d = paired_bootstrap_delta(y, p, p, n_boot=200)
    assert d["delta"] == pytest.approx(0.0)
    assert d["lo"] <= 0.0 <= d["hi"]
