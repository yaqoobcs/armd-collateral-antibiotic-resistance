"""The outcome must be able to separate acquired resistance from species replacement."""
import numpy as np
import pandas as pd
import pytest

from armd.labels import collateral_outcome_map, order_organisms


def _frame(rows):
    return pd.DataFrame(rows, columns=["anon_id", "order_proc_id_coded",
                                       "order_time_jittered_utc", "organism",
                                       "antibiotic", "drug_class", "resistant"])


def test_species_replacement_counts_by_default_and_is_excluded_on_request():
    """The headline case: a susceptible E. coli followed by a resistant enterococcus.

    No organism acquired anything. The default outcome still scores it as an event,
    because the patient's next infection genuinely does resist the class. The
    same-organism restriction removes it, which is what isolates the acquired pathway.
    """
    t0 = pd.Timestamp("2020-01-01")
    df = _frame([
        ["p1", 1, t0, "Escherichia coli", "Ciprofloxacin", "Fluoroquinolone", 0],
        ["p1", 2, t0 + pd.Timedelta(days=30), "Enterococcus faecium",
         "Ciprofloxacin", "Fluoroquinolone", 1],
    ])
    default = collateral_outcome_map(df, "Fluoroquinolone", 365)
    same = collateral_outcome_map(df, "Fluoroquinolone", 365, require_shared_organism=True)

    assert default.loc[1] == 1, "species replacement should count in the default outcome"
    assert 1 not in same.index, "species replacement must be excluded by the restriction"


def test_same_organism_resistance_survives_the_restriction():
    t0 = pd.Timestamp("2020-01-01")
    df = _frame([
        ["p1", 1, t0, "Escherichia coli", "Ciprofloxacin", "Fluoroquinolone", 0],
        ["p1", 2, t0 + pd.Timedelta(days=30), "Escherichia coli",
         "Ciprofloxacin", "Fluoroquinolone", 1],
    ])
    same = collateral_outcome_map(df, "Fluoroquinolone", 365, require_shared_organism=True)
    assert same.loc[1] == 1


def test_a_partial_organism_overlap_is_kept():
    """Polymicrobial cultures share an organism if any organism is shared."""
    t0 = pd.Timestamp("2020-01-01")
    df = _frame([
        ["p1", 1, t0, "Escherichia coli", "Ciprofloxacin", "Fluoroquinolone", 0],
        ["p1", 1, t0, "Proteus mirabilis", "Ciprofloxacin", "Fluoroquinolone", 0],
        ["p1", 2, t0 + pd.Timedelta(days=10), "Escherichia coli",
         "Ciprofloxacin", "Fluoroquinolone", 1],
    ])
    same = collateral_outcome_map(df, "Fluoroquinolone", 365, require_shared_organism=True)
    assert same.loc[1] == 1


def test_order_organisms_collects_the_set_per_culture():
    t0 = pd.Timestamp("2020-01-01")
    df = _frame([
        ["p1", 1, t0, "Escherichia coli", "Ciprofloxacin", "Fluoroquinolone", 0],
        ["p1", 1, t0, "Proteus mirabilis", "Ciprofloxacin", "Fluoroquinolone", 1],
    ])
    orgs = order_organisms(df)
    assert orgs.loc[1] == frozenset({"Escherichia coli", "Proteus mirabilis"})


def test_the_restriction_never_adds_units():
    t0 = pd.Timestamp("2020-01-01")
    rng = np.random.default_rng(0)
    rows = []
    for p in range(40):
        for k in range(3):
            rows.append([f"p{p}", p * 10 + k, t0 + pd.Timedelta(days=30 * k),
                         rng.choice(["E. coli", "K. pneumoniae", "E. faecium"]),
                         "Ciprofloxacin", "Fluoroquinolone", int(rng.integers(0, 2))])
    df = _frame(rows)
    default = collateral_outcome_map(df, "Fluoroquinolone", 365)
    same = collateral_outcome_map(df, "Fluoroquinolone", 365, require_shared_organism=True)
    assert set(same.index) <= set(default.index)
