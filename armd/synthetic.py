"""Generate ARMD-shaped tables so the pipeline can be run and tested without the real data.

The synthetic release has the same column names, dtypes, null token and sign conventions as
the real one, including the misspelled ``prior_infecting_organism_days_to_culutre``, and a
planted collateral-resistance effect for a smoke run to recover.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

AGENTS = [
    "Ciprofloxacin", "Levofloxacin", "Moxifloxacin", "Ceftriaxone", "Cefazolin",
    "Cefepime", "Meropenem", "Ertapenem", "Ampicillin", "Piperacillin/Tazobactam",
    "Gentamicin", "Amikacin", "Tobramycin", "Vancomycin", "Linezolid",
    "Nitrofurantoin", "Trimethoprim/Sulfamethoxazole", "Tetracycline",
]
ORGANISMS = [
    "ESCHERICHIA COLI", "KLEBSIELLA PNEUMONIAE", "PSEUDOMONAS AERUGINOSA",
    "STAPHYLOCOCCUS AUREUS", "ENTEROCOCCUS FAECALIS", "PROTEUS MIRABILIS",
]
SPECIMENS = ["Urine", "Blood", "Respiratory", "Wound"]
AGE_BINS = ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
            "55-64 years", "65-74 years", "75-84 years", "85-89 years", "above 90"]
EXPOSURE_MEDS = [
    "Ciprofloxacin", "Levofloxacin", "Ceftriaxone", "Cefazolin", "Meropenem",
    "Vancomycin", "Nitrofurantoin", "Azithromycin", "Metronidazole", "Rifampin",
]
NATIVE_CLASS = {
    "Ciprofloxacin": "Fluoroquinolone", "Levofloxacin": "Fluoroquinolone",
    "Ceftriaxone": "Beta Lactam", "Cefazolin": "Beta Lactam", "Meropenem": "Beta Lactam",
    "Vancomycin": "Glycopeptide", "Nitrofurantoin": "Urinary Antiseptic",
    "Azithromycin": "Macrolide Lincosamide", "Metronidazole": "Nitroimidazole",
    "Rifampin": "Ansamycin",
}

COLLATERAL_LOG_ODDS = 1.6


def generate(out_dir: str, n_patients: int = 900, seed: int = 0) -> dict:
    """Write a small synthetic release and return the ground truth used to build it."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    patients = [f"JC{900000 + i}" for i in range(n_patients)]
    n_cultures = rng.integers(1, 6, n_patients)

    rows = []
    order_id = 500_000_000
    for pid, k in zip(patients, n_cultures):
        frailty = rng.normal()
        start = pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(rng.integers(0, 2500)))
        t = start
        for _ in range(int(k)):
            order_id += 1
            t = t + pd.Timedelta(days=int(rng.integers(30, 400)))
            rows.append({
                "anon_id": pid,
                "pat_enc_csn_id_coded": int(rng.integers(1e10, 9e10)),
                "order_proc_id_coded": order_id,
                "order_time_jittered_utc": t,
                "organism": rng.choice(ORGANISMS, p=[0.4, 0.2, 0.12, 0.14, 0.08, 0.06]),
                "culture_description": rng.choice(SPECIMENS, p=[0.5, 0.25, 0.15, 0.10]),
                "ordering_mode": rng.choice(["Inpatient", "Outpatient"], p=[0.55, 0.45]),
                "frailty": frailty,
            })
    cultures = pd.DataFrame(rows)
    n_cult = len(cultures)

    expo_rows = []
    fq_propensity = 1 / (1 + np.exp(-(-0.4 + 0.8 * cultures["frailty"].to_numpy())))
    fq_exposed = rng.binomial(1, fq_propensity).astype(bool)
    for i, row in enumerate(cultures.itertuples()):
        meds = []
        if fq_exposed[i]:
            meds.append(rng.choice(["Ciprofloxacin", "Levofloxacin"]))
        for med in rng.choice(EXPOSURE_MEDS, size=int(rng.integers(0, 3)), replace=False):
            meds.append(med)
        for med in meds:
            expo_rows.append({
                "anon_id": row.anon_id,
                "pat_enc_csn_id_coded": row.pat_enc_csn_id_coded,
                "order_proc_id_coded": row.order_proc_id_coded,
                "order_time_jittered_utc": row.order_time_jittered_utc,
                "medication_category": "SYN",
                "medication_name": med,
                "antibiotic_class": NATIVE_CLASS[med],
                "time_to_culturetime": int(rng.integers(1, 120)),
            })
    exposure = pd.DataFrame(expo_rows)

    org_effect = {o: rng.normal(0, 1.4) for o in ORGANISMS}
    agent_effect = {a: rng.normal(-1.0, 0.9) for a in AGENTS}
    fq_agents = {"Ciprofloxacin", "Levofloxacin", "Moxifloxacin"}

    prev_exposed = np.zeros(n_cult, dtype=bool)
    position = {}
    for idx, row in enumerate(cultures.itertuples()):
        prior = position.get(row.anon_id)
        if prior is not None:
            prev_exposed[idx] = fq_exposed[prior]
        position[row.anon_id] = idx

    result_rows = []
    for i, row in enumerate(cultures.itertuples()):
        tested = rng.choice(AGENTS, size=int(rng.integers(6, len(AGENTS) + 1)), replace=False)
        for agent in tested:
            lin = agent_effect[agent] + org_effect[row.organism] + 0.5 * row.frailty
            if agent in fq_agents and prev_exposed[i]:
                lin += COLLATERAL_LOG_ODDS
            p = 1 / (1 + np.exp(-lin))
            resistant = rng.binomial(1, p)
            label = "Resistant" if resistant else "Susceptible"
            if rng.random() < 0.03:
                label = "Intermediate"
            result_rows.append({
                "anon_id": row.anon_id,
                "pat_enc_csn_id_coded": row.pat_enc_csn_id_coded,
                "order_proc_id_coded": row.order_proc_id_coded,
                "order_time_jittered_utc": row.order_time_jittered_utc,
                "culture_description": row.culture_description,
                "ordering_mode": row.ordering_mode,
                "was_positive": 1,
                "organism": row.organism,
                "antibiotic": agent,
                "susceptibility": label,
            })
    cohort = pd.DataFrame(result_rows)

    def write(name, frame):
        frame.to_csv(os.path.join(out_dir, name), index=False)

    write("microbiology_cultures_cohort.csv", cohort)
    write("microbiology_cultures_antibiotic_class_exposure.csv", exposure)

    ids = cultures[["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered_utc"]]

    demo = ids[["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]].copy()
    demo["age"] = rng.choice(AGE_BINS, size=n_cult)
    gender = rng.integers(0, 2, n_cult).astype(object)
    gender[rng.random(n_cult) < 0.01] = "Null"
    demo["gender"] = gender
    write("microbiology_cultures_demographics.csv", demo)

    ward = ids.copy()
    setting = rng.choice(4, size=n_cult, p=[0.35, 0.40, 0.15, 0.10])
    for j, col in enumerate(["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]):
        ward[col] = (setting == j).astype(int)
    write("microbiology_cultures_ward_info.csv", ward)

    nh = ids[rng.random(n_cult) < 0.08].copy()
    nh["nursing_home_visit_culture"] = 1
    write("microbiology_cultures_nursing_home_visits.csv", nh)

    proc = ids.loc[ids.index.repeat(rng.integers(0, 4, n_cult))].copy()
    proc["procedure_description"] = rng.choice(
        ["urethral_catheter", "central_line", "mechanical_ventilation"], size=len(proc))
    proc["procedure_time_to_culturetime"] = rng.integers(1, 400, len(proc))
    write("microbiology_cultures_priorprocedures.csv", proc)

    po = ids.loc[ids.index.repeat(rng.integers(0, 3, n_cult))].copy()
    po["prior_organism"] = rng.choice(ORGANISMS, size=len(po))
    po["prior_infecting_organism_days_to_culutre"] = rng.integers(1, 900, len(po))
    write("microbiology_culture_prior_infecting_organism.csv", po)

    mr = ids.loc[ids.index.repeat(rng.integers(0, 3, n_cult))].copy()
    mr["organism"] = rng.choice(ORGANISMS, size=len(mr))
    mr["antibiotic"] = rng.choice(AGENTS, size=len(mr))
    mr["resistant_time_to_culturetime"] = rng.integers(1, 900, len(mr))
    write("microbiology_cultures_microbial_resistance.csv", mr)

    frail = cultures["frailty"].to_numpy()
    vitals = ids[["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]].copy()
    for stub, loc, scale in (("heartrate", 88, 16), ("resprate", 18, 4),
                             ("temp", 37.0, 0.7), ("sysbp", 122, 18)):
        vitals[f"median_{stub}"] = np.round(rng.normal(loc, scale, n_cult) + 2 * frail, 1)
    write("microbiology_cultures_vitals.csv", vitals)

    labs = ids[["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]].copy()
    for stub, loc, scale in (("wbc", 9.5, 4.0), ("cr", 1.1, 0.6), ("lactate", 1.6, 0.9),
                             ("plt", 240, 80), ("bun", 18, 9)):
        vals = np.round(rng.normal(loc, scale, n_cult) + 0.8 * frail, 2).astype(object)
        vals[rng.random(n_cult) < 0.15] = "Null"
        labs[f"median_{stub}"] = vals
    write("microbiology_cultures_labs.csv", labs)

    adi = ids[["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]].copy()
    adi["adi_score"] = np.clip(np.round(rng.normal(50, 20, n_cult) + 5 * frail), 1, 100)
    adi["adi_state_rank"] = rng.integers(1, 11, n_cult)
    write("microbiology_cultures_adi_scores.csv", adi)

    com = ids.loc[ids.index.repeat(np.clip((rng.poisson(3, n_cult) + (frail > 0) * 2), 0, None))].copy()
    com["comorbidity_component"] = rng.choice(["CHF", "DM", "CKD", "COPD", "CANCER"], size=len(com))
    com["comorbidity_component_start_days_culture"] = rng.integers(1, 2000, len(com))
    com["comorbidity_component_end_days_culture"] = "Null"
    write("microbiology_cultures_comorbidity.csv", com)

    return {
        "n_patients": n_patients,
        "n_cultures": n_cult,
        "n_results": len(cohort),
        "fluoroquinolone_exposure_rate": float(fq_exposed.mean()),
        "planted_collateral_log_odds": COLLATERAL_LOG_ODDS,
        "cultures_with_prior_exposure": int(prev_exposed.sum()),
    }
