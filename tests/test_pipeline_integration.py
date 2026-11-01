"""End-to-end run on a synthetic release with a planted collateral effect.

Exercises everything downstream of the data load without the real release.
"""

import os

import numpy as np
import pytest

from armd.config import Config
from armd.synthetic import COLLATERAL_LOG_ODDS, generate


@pytest.fixture(scope="module")
def run_report(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    results_dir = tmp_path_factory.mktemp("results")
    generate(str(data_dir), n_patients=2000, seed=3)

    cfg = Config(
        sample_n_patients=None, temporal_cutoff_year=2019, max_epochs=15,
        n_bootstrap=60, top_k_organisms=10, min_train_events=40,
        min_train_positives=5, min_test_events=15, n_folds=4,
        results_dir=str(results_dir), device="cpu", seed=3,
    )
    from armd.pipeline import run

    return run(cfg, str(data_dir), verbose=False), cfg


def test_the_pipeline_completes(run_report):
    report, cfg = run_report
    assert report["cohort"]["culture_organism_units"] > 500
    assert report["agents_evaluable"] > 5
    assert report["splits"]["train"] > 0 and report["splits"]["test"] > 0


def test_gender_is_not_silently_zero(run_report):
    """Gender must survive the float encoding of the 0/1/Null column."""
    report, _ = run_report
    assert 0.3 < report["cohort"]["gender_positive_rate"] < 0.7


def test_susceptibility_beats_chance(run_report):
    """A wiring check, not a performance claim.

    The fixture trains for 15 epochs on 2000 synthetic patients, which under-trains the
    network on purpose to keep the suite fast; the threshold only has to be high enough
    that a mis-wired target matrix or a shuffled mask would fail it. The per-agent
    baseline picks the organism signal up quickly and is held to a stricter bar.
    """
    report, _ = run_report
    assert report["net_support_weighted_auc"] > 0.51
    assert report["baseline"]["xgb_support_weighted_auc"] > 0.55


def test_the_planted_collateral_effect_is_recovered(run_report):
    report, _ = run_report
    cf = report["causal"]["crossfit"]
    assert COLLATERAL_LOG_ODDS > 0
    assert cf["estimate"] > 0.05, cf
    assert cf["lo"] > 0.0, "the planted effect should be distinguishable from zero"


def test_adjustment_with_the_acute_block_moves_the_estimate_off_the_naive_contrast(run_report):
    """Confounding is planted through the patient frailty term, so an adjusted estimate that
    can see frailty must not reproduce the crude difference.

    This asserts on the arm that adjusts for the acute laboratory and vital block, because that
    is where the generator makes frailty visible: vitals carry ``2 * frailty`` against their
    own noise scale, while the pre-exposure block sees it only through an ADI score built as
    ``normal(50, 20) + 5 * frailty``.

    The bar is a margin, not just a sign, because cross-fitted AIPW is sensitive to the fold
    assignment on small samples.
    """
    report, _ = run_report
    naive = report["causal"]["crossfit_naive_contrast"]
    adjusted = report["causal"]["adjustment_with_acute_block"]["estimate"]
    assert naive - adjusted > 0.02, (naive, adjusted)


def test_primary_adjustment_moves_less_because_it_drops_the_acute_block(run_report):
    """The timing-safe adjustment set still moves the estimate, but by less.

    This is the cost of excluding covariates measured after the treatment window, and it
    is a cost worth seeing rather than hiding: the primary specification buys freedom from
    mediator and collider bias by giving up some measured confounding control. The
    synthetic generator does not simulate the timing pathology itself -- it draws severity
    independently of when exposure occurred -- so this test pins the size of the trade-off,
    not its direction on real data.
    """
    report, _ = run_report
    naive = report["causal"]["crossfit_naive_contrast"]
    primary = report["causal"]["crossfit"]["estimate"]
    acute = report["causal"]["adjustment_with_acute_block"]["estimate"]
    assert naive - primary > 0.0, (naive, primary)
    assert primary > acute, "dropping a measured confounder should adjust less, not more"


def test_negative_control_is_nearer_zero_than_the_main_effect(run_report):
    report, _ = run_report
    nc = report["causal"].get("negative_control")
    assert nc is not None
    assert abs(nc["estimate"]) < abs(report["causal"]["crossfit"]["estimate"])


def test_causal_sample_is_one_row_per_culture_order(run_report):
    report, _ = run_report
    c = report["causal"]
    assert c["units"] <= c["units_before_dedup"]
    assert c["patients"] <= c["units"]


def test_standard_error_is_clustered_on_patients(run_report):
    report, _ = run_report
    cf = report["causal"]["crossfit"]
    assert cf["n_clusters"] == report["causal"]["patients"]
    assert cf["n_clusters"] < cf["n"]


def test_diagnostics_are_reported(run_report):
    report, _ = run_report
    c = report["causal"]
    assert 0.0 <= c["overlap"]["common_support_fraction"] <= 1.0
    assert c["balance"]["max_abs_smd_weighted"] < c["balance"]["max_abs_smd_unadjusted"]
    assert np.isfinite(c["e_value_point"])
    assert "sensitivity_missing_as_negative" in c


def test_every_declared_artefact_is_written(run_report):
    report, cfg = run_report
    expected = [
        "loss_curves.png", "auc_forest.png", "net_vs_xgb.png", "crossfit_overlap.png",
        "causal_estimates.png", "balance.png", "calibration.png",
        "susceptibility_net_metrics.csv", "susceptibility_xgb_metrics.csv",
        "susceptibility_net_vs_xgb.csv", "predicted_antibiogram_example.csv",
        "run_report.json", "summary_report.md",
    ]
    present = set(os.listdir(cfg.results_dir))
    assert not (set(expected) - present), set(expected) - present


def test_comparison_is_over_the_same_agents(run_report):
    """The network and the baseline are compared over the same set of agents."""
    import pandas as pd

    report, cfg = run_report
    net = pd.read_csv(os.path.join(cfg.results_dir, "susceptibility_net_metrics.csv"))
    xgbm = pd.read_csv(os.path.join(cfg.results_dir, "susceptibility_xgb_metrics.csv"))
    assert set(net["antibiotic"]) == set(xgbm["antibiotic"])
