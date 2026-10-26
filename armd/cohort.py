"""Cohort assembly: from the raw tables to one analytic frame plus feature blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from armd import dataio
from armd.config import Config
from armd.drugclass import map_series, unmapped_agents
from armd.features import COUNT_COLS, LAB_STUBS, VITAL_STUBS, WARD_COLS
from armd.labels import collateral_outcome_map, exposure_indicators, exposed_orders

KEYS = ["anon_id", "order_proc_id_coded", "organism", "order_year"]


@dataclass
class Cohort:
    frame: pd.DataFrame
    definitive: pd.DataFrame
    exposure_window: pd.DataFrame
    agents: list
    expo_cols: list
    sev_cols: list
    sev_pre_cols: list
    sev_acute_cols: list
    base_numeric_cols: list
    notes: dict


def build(cfg: Config, data_dir: str, rng: np.random.Generator) -> Cohort:
    notes: dict = {}

    cohort = dataio.load_table(
        data_dir, "microbiology_cultures_cohort.csv", parse_dates=["order_time_jittered_utc"]
    )
    if cohort is None:
        raise FileNotFoundError("microbiology_cultures_cohort.csv is required")

    all_patients = cohort["anon_id"].dropna().unique()
    if cfg.sample_n_patients and cfg.sample_n_patients < len(all_patients):
        patients = set(rng.choice(all_patients, size=cfg.sample_n_patients, replace=False).tolist())
    else:
        patients = set(all_patients.tolist())
    notes["patients_sampled"] = len(patients)
    notes["patients_available"] = int(len(all_patients))

    cohort = cohort[cohort["anon_id"].isin(patients)].copy()
    cohort["susceptibility"] = cohort["susceptibility"].astype("string").str.strip()
    positive = cohort[
        (cohort["was_positive"] == 1) & cohort["organism"].notna() & cohort["antibiotic"].notna()
    ].copy()

    definitive = positive[positive["susceptibility"].isin(["Susceptible", "Resistant"])].copy()
    notes["results_total"] = int(len(positive))
    notes["results_definitive"] = int(len(definitive))
    notes["results_dropped_indeterminate"] = int(len(positive) - len(definitive))

    definitive["resistant"] = (definitive["susceptibility"] == "Resistant").astype(int)
    definitive["order_year"] = definitive["order_time_jittered_utc"].dt.year
    definitive["drug_class"] = map_series(definitive["antibiotic"]).values
    notes["agents_without_class"] = unmapped_agents(definitive["antibiotic"].unique())

    targets = (
        definitive.pivot_table(index=KEYS, columns="antibiotic", values="resistant", aggfunc="max")
        .reset_index()
    )
    agents = [c for c in targets.columns if c not in KEYS]

    ctx = definitive.drop_duplicates("order_proc_id_coded")[
        ["order_proc_id_coded", "culture_description", "ordering_mode"]
    ]
    demo = dataio.load_table(
        data_dir, "microbiology_cultures_demographics.csv",
        usecols=["order_proc_id_coded", "age", "gender"],
    )
    demo = demo.drop_duplicates("order_proc_id_coded") if demo is not None else None

    ward = dataio.load_table(
        data_dir, "microbiology_cultures_ward_info.csv",
        usecols=["order_proc_id_coded", *WARD_COLS],
    )
    ward = ward.drop_duplicates("order_proc_id_coded") if ward is not None else None

    nh = dataio.load_table(
        data_dir, "microbiology_cultures_nursing_home_visits.csv",
        usecols=["order_proc_id_coded", "nursing_home_visit_culture"],
    )
    if nh is not None and len(nh):
        nh = nh.drop_duplicates("order_proc_id_coded")[["order_proc_id_coded"]].assign(nursing_home=1)

    procedures = dataio.prior_count(
        data_dir, "microbiology_cultures_priorprocedures.csv",
        ["anon_id", "order_proc_id_coded", "procedure_time_to_culturetime"],
        "procedure_time_to_culturetime", "n_procedures", patients,
    )
    prior_org = dataio.prior_count(
        data_dir, "microbiology_culture_prior_infecting_organism.csv",
        ["anon_id", "order_proc_id_coded", "prior_infecting_organism_days_to_culutre"],
        "prior_infecting_organism_days_to_culutre", "n_prior_organisms", patients,
    )
    prior_res = dataio.prior_count(
        data_dir, "microbiology_cultures_microbial_resistance.csv",
        ["anon_id", "order_proc_id_coded", "resistant_time_to_culturetime"],
        "resistant_time_to_culturetime", "n_prior_resistance", patients,
    )

    labs = dataio.reduce_medians(data_dir, "microbiology_cultures_labs.csv", "median_", LAB_STUBS, patients)
    vitals = dataio.reduce_medians(data_dir, "microbiology_cultures_vitals.csv", "median_", VITAL_STUBS, patients)

    adi = dataio.load_table(
        data_dir, "microbiology_cultures_adi_scores.csv", usecols=["order_proc_id_coded", "adi_score"]
    )
    if adi is not None and len(adi):
        adi["adi_score"] = pd.to_numeric(adi["adi_score"], errors="coerce")
        adi = adi.groupby("order_proc_id_coded", as_index=False)["adi_score"].median()
    else:
        adi = pd.DataFrame({"order_proc_id_coded": pd.Series(dtype="int64"),
                            "adi_score": pd.Series(dtype="float64")})

    comorbid = dataio.chunked_prior_count(
        data_dir, "microbiology_cultures_comorbidity.csv", "anon_id", "order_proc_id_coded",
        "comorbidity_component_start_days_culture", "n_comorbidity", patients,
    )

    expo = dataio.load_table(
        data_dir, "microbiology_cultures_antibiotic_class_exposure.csv",
        usecols=["anon_id", "order_proc_id_coded", "medication_name", "antibiotic_class", "time_to_culturetime"],
    )
    if expo is not None:
        expo = dataio.restrict_to_patients(expo, patients)
        expo["time_to_culturetime"] = pd.to_numeric(expo["time_to_culturetime"], errors="coerce")
        expo["drug_class"] = map_series(expo["medication_name"]).values
        t = expo["time_to_culturetime"]
        expo_window = expo[(t >= 0) & (t <= cfg.exposure_window_days) & expo["drug_class"].notna()].copy()
    else:
        expo_window = pd.DataFrame(columns=["order_proc_id_coded", "drug_class"])
    expo_ind = exposure_indicators(expo_window, cfg.exposure_window_days)

    df = targets.copy()
    for table in (ctx, demo, ward, nh, procedures, prior_org, prior_res, expo_ind, labs, vitals, adi, comorbid):
        if table is None or not len(table.columns):
            continue
        df = df.merge(table, on="order_proc_id_coded", how="left")

    expo_cols = [c for c in df.columns if c.startswith("prior_expo_")]
    for c in list(COUNT_COLS) + expo_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in WARD_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0) if c in df.columns else 0.0
    if "n_comorbidity" not in df.columns:
        df["n_comorbidity"] = 0.0
    df["n_comorbidity"] = pd.to_numeric(df["n_comorbidity"], errors="coerce").fillna(0.0)
    if "adi_score" not in df.columns:
        df["adi_score"] = np.nan

    sev_acute_cols = [c for c in df.columns if c.startswith("median_")]
    sev_pre_cols = ["adi_score", "n_comorbidity"]
    sev_cols = sev_pre_cols + sev_acute_cols

    df["gender_bin"] = dataio.binary_from_numeric(df["gender"]) if "gender" in df.columns else 0.0
    notes["gender_positive_rate"] = float(np.mean(df["gender_bin"]))

    base_numeric = ["gender_bin", *WARD_COLS, *COUNT_COLS]

    target_orders = exposed_orders(expo_window, cfg.target_class)
    df["A"] = df["order_proc_id_coded"].isin(target_orders).astype(np.float32)

    y_target = collateral_outcome_map(
        definitive, cfg.target_class, cfg.followup_horizon_days, cfg.missing_as_negative
    )
    y_control = collateral_outcome_map(
        definitive, cfg.control_class, cfg.followup_horizon_days, cfg.missing_as_negative
    )
    df["Ycoll"] = df["order_proc_id_coded"].map(y_target)
    df["Yctrl"] = df["order_proc_id_coded"].map(y_control)
    df["causal_valid"] = df["Ycoll"].notna().astype(np.float32)

    notes["culture_organism_units"] = int(len(df))
    notes["culture_orders"] = int(df["order_proc_id_coded"].nunique())
    notes["patients_with_units"] = int(df["anon_id"].nunique())
    notes["causal_valid_units"] = int(df["causal_valid"].sum())
    notes["causal_valid_orders"] = int(df.loc[df.causal_valid == 1, "order_proc_id_coded"].nunique())
    notes["agents"] = len(agents)

    notes["sev_pre_cols"] = len(sev_pre_cols)
    notes["sev_acute_cols"] = len(sev_acute_cols)

    return Cohort(
        frame=df, definitive=definitive, exposure_window=expo_window, agents=agents,
        expo_cols=expo_cols, sev_cols=sev_cols, sev_pre_cols=sev_pre_cols,
        sev_acute_cols=sev_acute_cols, base_numeric_cols=base_numeric, notes=notes,
    )


def causal_sample(frame: pd.DataFrame, outcome_col: str = "Ycoll") -> pd.DataFrame:
    """One row per culture order among units with an observed outcome.

    The analytic row is culture x organism, so a culture that grew two organisms has two rows
    carrying the same treatment and outcome. Counting both would inflate n and shrink the
    standard error without adding an independent observation.
    """
    valid = frame[frame[outcome_col].notna()]
    return valid.drop_duplicates("order_proc_id_coded").copy()
