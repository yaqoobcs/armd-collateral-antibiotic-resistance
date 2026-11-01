"""The estimator has to recover an effect it was not told about."""

import numpy as np
import pytest

from armd.causal import (
    aipw_influence, bootstrap_ci, clustered_se, crossfit_aipw,
    e_value_bound, e_value_point, e_value_rr, overlap_summary,
    rr_from_rd, standardised_differences,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n=6000, d=5, delta=0.8, n_per_patient=1, seed=0, hidden_confounder=False,
             u_sd=0.7):
    """Confounded observational data with a known average treatment effect.

    A patient-level random effect drives both the treatment and the outcome, so
    repeated cultures from one patient are correlated. It is included in ``X`` unless
    ``hidden_confounder`` is set -- in which case the data violate conditional
    exchangeability on purpose and no adjustment can recover the truth.

    ``u_sd`` scales that patient effect. It matters for clustering: when the effect is
    measured and adjusted away, the influence function carries almost no within-patient
    correlation left to cluster on.
    """
    rng = np.random.default_rng(seed)
    n_patients = max(n // n_per_patient, 1)
    groups = np.repeat(np.arange(n_patients), n_per_patient)[:n]
    u = rng.normal(0, u_sd, n_patients)[groups]
    X = rng.normal(0, 1, size=(n, d))
    lin = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] + u
    pi = sigmoid(lin)
    a = rng.binomial(1, pi)
    base = -0.4 + 0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 3] + u
    p0, p1 = sigmoid(base), sigmoid(base + delta)
    y = np.where(a == 1, rng.binomial(1, p1), rng.binomial(1, p0))
    true_ate = float(np.mean(p1 - p0))
    if not hidden_confounder:
        X = np.column_stack([X, u])
    return X, a.astype(float), y.astype(float), groups, true_ate


def test_crossfit_recovers_a_known_effect():
    X, a, y, groups, true_ate = simulate(n=8000, seed=1)
    est = crossfit_aipw(X, a, y, groups=None, n_folds=5, seed=1)
    assert abs(est.estimate - true_ate) < 0.03, (est.estimate, true_ate)
    assert est.lo < true_ate < est.hi


def test_hidden_confounding_biases_the_estimate():
    """The estimator is not magic: with a confounder left out it is wrong, and that is
    exactly the situation the E-value is reported for."""
    X, a, y, groups, true_ate = simulate(n=8000, seed=1, hidden_confounder=True)
    est = crossfit_aipw(X, a, y, n_folds=5, seed=1)
    assert abs(est.estimate - true_ate) > 0.05


def test_unadjusted_contrast_is_biased_but_aipw_is_not():
    """If the adjustment were doing nothing, this test would fail."""
    X, a, y, groups, true_ate = simulate(n=8000, seed=2)
    naive = y[a == 1].mean() - y[a == 0].mean()
    est = crossfit_aipw(X, a, y, n_folds=5, seed=2)
    assert abs(naive - true_ate) > abs(est.estimate - true_ate)


def test_null_effect_interval_covers_zero():
    X, a, y, groups, true_ate = simulate(n=8000, delta=0.0, seed=3)
    est = crossfit_aipw(X, a, y, n_folds=5, seed=3)
    assert est.lo < 0.0 < est.hi


@pytest.mark.parametrize("fold_seed", [0, 1, 2, 3])
def test_clustering_widens_the_interval(fold_seed):
    """Repeated cultures per patient must not be counted as independent.

    The patient effect is left unmeasured and made large, which puts correlation into the
    influence function for clustering to find. The expected ratio is about 1.26, and it is
    asserted across four fold seeds.
    """
    X, a, y, groups, _ = simulate(
        n=6000, n_per_patient=12, seed=4, hidden_confounder=True, u_sd=1.5
    )
    est = crossfit_aipw(X, a, y, groups=groups, n_folds=5, seed=fold_seed)
    assert est.n_clusters == len(np.unique(groups))
    assert est.se > 1.15 * est.se_iid, (est.se, est.se_iid)
    assert (est.hi - est.lo) > 2 * 1.96 * est.se_iid


def test_clustering_is_a_no_op_when_the_patient_effect_is_adjusted_away():
    """The converse, and the reason the test above needs a hidden patient effect.

    With the patient effect measured and given to the nuisance models, what is left in the
    influence function is close to independent, so the clustered SE matches the iid one.
    """
    X, a, y, groups, _ = simulate(n=6000, n_per_patient=12, seed=4, u_sd=1.5)
    est = crossfit_aipw(X, a, y, groups=groups, n_folds=5, seed=0)
    assert est.se == pytest.approx(est.se_iid, rel=0.10)


@pytest.mark.parametrize(
    "patient_sd, m", [(0.5, 12), (1.0, 12), (2.0, 12), (1.0, 4)]
)
def test_clustered_se_matches_the_analytic_design_effect(patient_sd, m):
    """Check the SE machinery directly, with no estimator in the way.

    For an exchangeable within-cluster correlation the clustered SE should exceed the iid
    one by the design effect sqrt(1 + (m - 1) * icc). This test has no learner in it, so
    no scikit-learn release can move it.
    """
    k = 500
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(k), m)
    psi = rng.normal(0, patient_sd, k)[groups] + rng.normal(0, 1.0, k * m)

    se_grouped, n_clusters = clustered_se(psi, groups)
    se_iid = psi.std(ddof=1) / np.sqrt(len(psi))
    icc = patient_sd ** 2 / (patient_sd ** 2 + 1.0)
    expected = np.sqrt(1 + (m - 1) * icc)

    assert n_clusters == k
    assert se_grouped / se_iid == pytest.approx(expected, rel=0.08)


def test_grouped_folds_keep_patients_whole():
    from sklearn.model_selection import StratifiedGroupKFold
    X, a, y, groups, _ = simulate(n=3000, n_per_patient=5, seed=5)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in splitter.split(X, a.astype(int), groups=groups):
        assert not (set(groups[tr]) & set(groups[te]))


def test_clustered_se_reduces_to_iid_with_singleton_clusters():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=500)
    se_grouped, k = clustered_se(psi, np.arange(500))
    se_iid = psi.std(ddof=1) / np.sqrt(500)
    assert k == 500
    assert se_grouped == pytest.approx(se_iid, rel=1e-6)


def test_influence_function_equals_plugin_when_models_are_exact():
    n = 1000
    rng = np.random.default_rng(0)
    m0 = rng.uniform(0.1, 0.4, n)
    m1 = m0 + 0.1
    pi = np.full(n, 0.5)
    a = rng.binomial(1, pi).astype(float)
    y = np.where(a == 1, m1, m0)
    psi = aipw_influence(a, y, pi, m0, m1)
    assert psi.mean() == pytest.approx(0.1, abs=1e-12)


def test_constant_treatment_is_rejected():
    X, a, y, _, _ = simulate(n=500)
    with pytest.raises(ValueError):
        crossfit_aipw(X, np.ones_like(a), y)


def test_clipped_fraction_is_reported():
    X, a, y, groups, _ = simulate(n=3000, seed=6)
    est = crossfit_aipw(X, a, y, clip=0.30, seed=6)
    assert est.clipped_fraction > 0.0


def test_bootstrap_resamples_patients_not_rows():
    X, a, y, groups, _ = simulate(n=3000, n_per_patient=6, seed=7)
    rng = np.random.default_rng(0)
    pi = np.clip(rng.uniform(0.2, 0.8, len(y)), 0.05, 0.95)
    m0 = rng.uniform(0.1, 0.5, len(y))
    m1 = m0 + 0.05
    _, lo_g, hi_g = bootstrap_ci(a, y, pi, m0, m1, groups=groups, n_boot=300, seed=0)
    _, lo_i, hi_i = bootstrap_ci(a, y, pi, m0, m1, groups=None, n_boot=300, seed=0)
    assert (hi_g - lo_g) > (hi_i - lo_i)


def test_e_value_matches_published_values():
    assert e_value_rr(3.9) == pytest.approx(7.26, abs=0.01)
    assert e_value_rr(1.0) == pytest.approx(1.0)
    assert e_value_rr(0.5) == pytest.approx(e_value_rr(2.0))


def test_e_value_bound_is_one_when_the_interval_crosses_the_null():
    """An interval that crosses the null gives an E-value of exactly 1."""
    assert e_value_bound(estimate=0.05, lo=-0.01, hi=0.11, baseline_risk=0.2) == 1.0
    assert e_value_bound(estimate=-0.05, lo=-0.11, hi=0.01, baseline_risk=0.2) == 1.0


def test_e_value_bound_below_point_when_the_interval_excludes_the_null():
    ev_p = e_value_point(0.09, 0.20)
    ev_b = e_value_bound(0.09, 0.064, 0.116, 0.20)
    assert 1.0 < ev_b < ev_p


def test_published_e_values_reproduce():
    """Guards the E-values printed in the paper.

    The unexposed baseline risk recovered by inverting the published E-value is 0.310.
    """
    baseline = 0.310
    assert e_value_point(0.0899, baseline) == pytest.approx(1.90, abs=0.01)
    assert e_value_bound(0.0899, 0.064, 0.1159, baseline) == pytest.approx(1.71, abs=0.01)
    assert rr_from_rd(0.0899, baseline) == pytest.approx(1.29, abs=0.01)


def test_rr_from_rd():
    assert rr_from_rd(0.1, 0.2) == pytest.approx(1.5)
    assert rr_from_rd(0.0, 0.2) == pytest.approx(1.0)


def test_overlap_summary_flags_separation():
    rng = np.random.default_rng(0)
    a = np.r_[np.ones(200), np.zeros(200)]
    separated = np.r_[rng.uniform(0.80, 0.95, 200), rng.uniform(0.05, 0.20, 200)]
    nested = np.r_[np.linspace(0.05, 0.95, 200), np.linspace(0.05, 0.95, 200)]
    partial = np.r_[rng.uniform(0.10, 0.90, 200), rng.uniform(0.05, 0.95, 200)]
    assert overlap_summary(separated, a)["common_support_fraction"] == 0.0
    assert overlap_summary(nested, a)["common_support_fraction"] == 1.0
    assert 0.8 < overlap_summary(partial, a)["common_support_fraction"] < 1.0


def test_weighting_shrinks_standardised_differences():
    X, a, y, groups, _ = simulate(n=6000, seed=8)
    est = crossfit_aipw(X, a, y, seed=8)
    from armd.causal import crossfit_aipw as _cf
    pi = np.clip(1.0 / (1.0 + np.exp(-(0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2]))), 0.02, 0.98)
    w = np.where(a == 1, 1.0 / pi, 1.0 / (1.0 - pi))
    before = np.nanmax(np.abs(standardised_differences(X, a)))
    after = np.nanmax(np.abs(standardised_differences(X, a, weights=w)))
    assert after < before


def test_simultaneous_band_is_wider_than_the_pointwise_one():
    """A panel of estimates read as a ranking needs a multiplicity-aware band."""
    from armd.pipeline import simultaneous_interval

    est, se = 0.10, 0.02
    lo1, hi1, z1 = simultaneous_interval(est, se, n_comparisons=1)
    lo9, hi9, z9 = simultaneous_interval(est, se, n_comparisons=9)

    assert z1 == pytest.approx(1.96, abs=0.01), "a single comparison is the usual interval"
    assert z9 > z1
    assert (hi9 - lo9) > (hi1 - lo1)
    assert lo9 < lo1 and hi9 > hi1
    assert (lo9 + hi9) / 2 == pytest.approx(est)


def test_simultaneous_band_can_cover_zero_where_the_pointwise_one_does_not():
    """The point of the correction: a borderline class stops looking non-null."""
    from armd.pipeline import simultaneous_interval

    est, se = 0.055, 0.025
    lo1, _, _ = simultaneous_interval(est, se, n_comparisons=1)
    lo9, _, _ = simultaneous_interval(est, se, n_comparisons=9)
    assert lo1 > 0
    assert lo9 < 0


def test_crossfit_propensity_returns_unclipped_scores():
    """Clipping inside the propensity helper would void every overlap diagnostic.

    A clipped vector has nothing outside the clip bounds, so the overlap measure
    ``mean((pi < k) | (pi > 1 - k))`` would be identically zero. The helper returns raw scores
    and callers clip after measuring.
    """
    from armd.causal import crossfit_propensity

    X, a, y, groups, _ = simulate(n=3000, n_per_patient=3, seed=11, u_sd=4.0)
    pi = crossfit_propensity(X, a, groups=groups, n_folds=5, seed=0)

    assert pi.min() >= 0.0 and pi.max() <= 1.0
    outside = float(np.mean((pi < 0.025) | (pi > 0.975)))
    assert outside > 0.0, "a strongly separated design must put some units outside the bounds"

    clipped = np.clip(pi, 0.025, 0.975)
    assert float(np.mean((clipped < 0.025) | (clipped > 0.975))) == 0.0
