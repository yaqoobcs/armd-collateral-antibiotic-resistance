import numpy as np
import pandas as pd
import pytest

from armd.dataio import binary_from_numeric
from armd.drugclass import agent_to_class, audit_against_native, unmapped_agents
from armd.features import TabularEncoder, group_organisms, temporal_split
from armd.labels import (
    collapse_class_resistance, collateral_outcome_map, exposed_orders,
    exposure_indicators, next_culture_resistance,
)

T0 = pd.Timestamp("2020-01-01")


def days(n):
    return T0 + pd.Timedelta(days=n)


def frame(rows):
    return pd.DataFrame(rows, columns=["anon_id", "order_time_jittered_utc", "resistant"])


def test_next_culture_inside_horizon():
    f = frame([("p", days(0), 0), ("p", days(100), 1)])
    assert list(next_culture_resistance(f, 365)) == [1.0, pytest.approx(np.nan, nan_ok=True)]


def test_next_culture_beyond_horizon_is_missing():
    f = frame([("p", days(0), 0), ("p", days(400), 1)])
    assert np.isnan(next_culture_resistance(f, 365).iloc[0])


def test_horizon_boundary_is_inclusive():
    f = frame([("p", days(0), 0), ("p", days(365), 1)])
    assert next_culture_resistance(f, 365).iloc[0] == 1.0


def test_same_timestamp_cultures_are_not_each_others_successor():
    f = frame([("p", days(0), 1), ("p", days(0), 0), ("p", days(30), 1)])
    out = next_culture_resistance(f, 365)
    assert list(out[:2]) == [1.0, 1.0]


def test_patients_do_not_leak_into_each_other():
    f = frame([("a", days(0), 0), ("b", days(10), 1)])
    assert np.isnan(next_culture_resistance(f, 365)).all()


def test_missing_as_negative_codes_zero_instead_of_dropping():
    f = frame([("p", days(0), 0), ("p", days(400), 1)])
    out = next_culture_resistance(f, 365, missing_as_negative=True)
    assert list(out) == [0.0, 0.0]


def test_output_is_aligned_to_the_input_index_not_the_sort_order():
    f = frame([("p", days(100), 1), ("p", days(0), 0), ("p", days(200), 1)])
    f.index = [7, 3, 11]
    out = next_culture_resistance(f, 365)
    assert out.loc[3] == 1.0
    assert out.loc[7] == 1.0
    assert np.isnan(out.loc[11])


def test_empty_input():
    assert len(next_culture_resistance(frame([]), 365)) == 0


def test_class_resistance_is_the_maximum_over_agents():
    d = pd.DataFrame({
        "anon_id": ["p", "p"], "order_proc_id_coded": [1, 1],
        "order_time_jittered_utc": [days(0), days(0)],
        "drug_class": ["Fluoroquinolone", "Fluoroquinolone"], "resistant": [0, 1],
    })
    out = collapse_class_resistance(d, "Fluoroquinolone")
    assert len(out) == 1 and out["resistant"].iloc[0] == 1


def test_collateral_outcome_map_end_to_end():
    d = pd.DataFrame({
        "anon_id": ["p", "p"], "order_proc_id_coded": [1, 2],
        "order_time_jittered_utc": [days(0), days(60)],
        "drug_class": ["Fluoroquinolone"] * 2, "resistant": [0, 1],
    })
    m = collateral_outcome_map(d, "Fluoroquinolone", 365)
    assert m.to_dict() == {1: 1}


def test_exposure_window_is_a_backward_look():
    e = pd.DataFrame({
        "order_proc_id_coded": [1, 2, 3],
        "drug_class": ["Fluoroquinolone"] * 3,
        "time_to_culturetime": [10, 200, 90],
    })
    wide = exposure_indicators(e, 90)
    assert set(wide["order_proc_id_coded"]) == {1, 3}
    assert wide["prior_expo_Fluoroquinolone"].sum() == 2


def test_exposed_orders_selects_the_target_class_only():
    e = pd.DataFrame({"order_proc_id_coded": [1, 2],
                      "drug_class": ["Fluoroquinolone", "Cephalosporin"]})
    assert exposed_orders(e, "Fluoroquinolone") == {1}


@pytest.mark.parametrize("name,expected", [
    ("Ciprofloxacin", "Fluoroquinolone"),
    ("Meropenem/Vaborbactam", "Carbapenem"),
    ("Ceftazidime/Avibactam", "Cephalosporin"),
    ("Piperacillin/Tazobactam", "Penicillin"),
    ("Tigecycline", "Glycylcycline"),
    ("Minocycline", "Tetracycline"),
    ("Trimethoprim/Sulfamethoxazole", "Folate Pathway Inhibitor"),
    ("Chloramphenicol", "Phenicol"),
    ("Quinupristin/Dalfopristin", "Streptogramin"),
    ("Nitrofurantoin", "Nitrofuran"),
    (None, None),
    (3.5, None),
])
def test_agent_class_mapping(name, expected):
    assert agent_to_class(name) == expected


def test_every_agent_in_the_published_antibiogram_maps():
    published = [
        "Ceftaroline", "Daptomycin", "Amikacin", "Linezolid", "Ertapenem", "Meropenem",
        "Tigecycline", "Imipenem", "Metronidazole", "Nitrofurantoin",
        "Piperacillin/Tazobactam", "Ceftazidime/Avibactam", "Cefotetan", "Cefpodoxime",
        "Vancomycin", "Clarithromycin", "Cefotaxime", "Cefepime", "Aztreonam",
        "Fosfomycin", "Ceftolozane/Tazobactam", "Tobramycin",
        "Amoxicillin/Clavulanic Acid", "Gentamicin", "Ceftazidime", "Doripenem",
        "Meropenem/Vaborbactam", "Colistin", "Ticacarcillin/Clavulanic Acid",
        "Cefazolin", "Ceftriaxone", "Ciprofloxacin", "Levofloxacin", "Cefoxitin",
        "Chloramphenicol", "Penicillin", "Ticarcillin", "Cefuroxime",
        "Cephalexin/Cephalothin", "Streptomycin", "Moxifloxacin",
        "Trimethoprim/Sulfamethoxazole", "Doxycycline", "Oxacillin", "Ampicillin",
        "Tetracycline", "Cefiderocol", "Clindamycin", "Imipenem/Ebactam",
        "Erythromycin", "Minocycline", "Ampicillin/Sulbactam",
        "Quinupristin/Dalfopristin", "Piperacillin",
    ]
    assert len(published) == 54
    assert unmapped_agents(published) == []


def test_audit_reports_disagreement_with_the_native_taxonomy():
    e = pd.DataFrame({
        "medication_name": ["Ceftriaxone", "Ciprofloxacin"],
        "antibiotic_class": ["Beta Lactam", "Fluoroquinolone"],
    })
    audit = audit_against_native(e)
    row = audit[audit.medication_name == "Ceftriaxone"].iloc[0]
    assert row["antibiotic_class"] == "Beta Lactam"
    assert row["derived_class"] == "Cephalosporin"


def test_gender_survives_the_float_encoding():
    """demographics.gender is 0/1/Null, which pandas types float64."""
    assert list(binary_from_numeric(pd.Series([0.0, 1.0, np.nan]))) == [0.0, 1.0, 0.0]
    assert list(binary_from_numeric(pd.Series(["0", "1", "Null"]))) == [0.0, 1.0, 0.0]


def test_encoder_statistics_come_from_the_training_rows_only():
    df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1000.0]})
    enc = TabularEncoder(["x"], []).fit(df, np.array([0, 1, 2]))
    assert enc.means_[0] == pytest.approx(1.0)
    out = enc.transform(df)
    assert out[3, 0] > 100


def test_encoder_imputes_with_the_training_median():
    df = pd.DataFrame({"x": [2.0, 4.0, np.nan]})
    enc = TabularEncoder(["x"], []).fit(df, np.array([0, 1]))
    assert enc.medians_[0] == pytest.approx(3.0)
    assert enc.transform(df)[2, 0] == pytest.approx(0.0)


def test_unseen_categories_encode_as_all_zero():
    df = pd.DataFrame({"c": ["a", "b", "z"]})
    enc = TabularEncoder([], ["c"]).fit(df, np.array([0, 1]))
    out = enc.transform(df)
    assert out.shape[1] == 2
    assert out[2].sum() == 0.0


def test_missing_column_does_not_raise():
    enc = TabularEncoder(["absent"], ["also_absent"]).fit(pd.DataFrame({"x": [1.0]}), np.array([0]))
    assert enc.transform(pd.DataFrame({"x": [1.0]})).shape == (1, 1)


def test_top_organisms_are_chosen_on_the_training_rows():
    df = pd.DataFrame({"organism": ["A", "A", "A", "B", "B", "B", "B"]})
    grouped = group_organisms(df, np.array([0, 1, 2]), top_k=1)
    assert grouped.iloc[0] == "A"
    assert grouped.iloc[6] == "OTHER_ORGANISM"


def test_temporal_split_keeps_patients_out_of_both_train_and_validation():
    years = np.array([2015, 2016, 2017, 2018, 2022, 2023])
    patients = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
    rng = np.random.default_rng(0)
    tr, va, te = temporal_split(years, 2021, patients, 0.5, rng, group_by_patient=True)
    assert not set(patients[tr]) & set(patients[va])
    assert set(te.tolist()) == {4, 5}


def test_ungrouped_split_is_still_available_and_splits_by_row():
    years = np.array([2015] * 10)
    rng = np.random.default_rng(0)
    tr, va, te = temporal_split(years, 2021, None, 0.2, rng, group_by_patient=False)
    assert len(va) == 2 and len(tr) == 8 and len(te) == 0
