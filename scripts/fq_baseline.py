"""Baseline characteristics, missingness and follow-up for the fluoroquinolone analysis.

STROBE items 14a-14c. Rebuilds the cohort exactly as ``armd.pipeline.run`` does (same
config, same seed, same random number order), asserts that the analysed orders match the
run report, and saves the out-of-fold propensity scores for the overlap figure.

Run from the repository root:  python -m scripts.fq_baseline ARMD_dataset results/full
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from armd import causal, dataio
from armd import cohort as cohort_mod
from armd.config import Config
from armd.features import TabularEncoder, CATEGORICAL_COLS, group_organisms, temporal_split
from armd.labels import collapse_class_resistance

data_dir, results_dir = sys.argv[1], sys.argv[2]
report = json.load(open(os.path.join(results_dir, "run_report.json")))
cfg_d = json.load(open(os.path.join(results_dir, "config.json")))
cfg = Config(**{k: v for k, v in cfg_d.items() if k in Config.__dataclass_fields__})

rng = np.random.default_rng(cfg.seed)
coh = cohort_mod.build(cfg, data_dir, rng)
df = coh.frame
train_idx, _, _ = temporal_split(df["order_year"].to_numpy(), cfg.temporal_cutoff_year,
                                 df["anon_id"].to_numpy(), cfg.val_fraction, rng,
                                 cfg.group_by_patient)
df["organism_g"] = group_organisms(df, train_idx, cfg.top_k_organisms)

base_encoder = TabularEncoder(coh.base_numeric_cols, list(CATEGORICAL_COLS)).fit(df, train_idx)
sev_pre_encoder = TabularEncoder(coh.sev_pre_cols, []).fit(df, train_idx)
Xcausal = np.concatenate([base_encoder.transform(df), sev_pre_encoder.transform(df)], axis=1)

sample = cohort_mod.causal_sample(df, "Ycoll")
rows = sample.index.to_numpy()
a = sample["A"].to_numpy(dtype=float)
groups = sample["anon_id"].to_numpy()
rc = report["causal"]
assert len(sample) == rc["units"] == 32724, len(sample)
assert sample["anon_id"].nunique() == rc["patients"]
assert int(a.sum()) == rc["class_effects"]["estimated"]["Fluoroquinolone"]["n_exposed"]

pi_raw = causal.crossfit_propensity(Xcausal[rows], a, groups, cfg.n_folds, cfg.seed)
pi = np.clip(pi_raw, cfg.propensity_clip, 1 - cfg.propensity_clip)
ov = causal.overlap_summary(pi, a)
np.savez_compressed(os.path.join(results_dir, "fq_propensity_oof.npz"), pi_raw=pi_raw, a=a)
w = np.where(a == 1, 1 / pi, 1 / (1 - pi))

s = sample
rowsdef: list[tuple[str, str, pd.Series]] = []


def add_cat(section, col, labels=None, top=None):
    v = s[col].astype("string").fillna("Not recorded")
    cats = v.value_counts().index.tolist()
    if top:
        keep = cats[:top]
        v = v.where(v.isin(keep), "Other")
        cats = keep + (["Other"] if (~s[col].astype("string").isin(keep)).any() else [])
    else:
        cats = sorted(cats, key=lambda c: (c == "Not recorded", c))
    for c in cats:
        rowsdef.append((section, (labels or {}).get(c, c), (v == c).astype(float), "bin"))


add_cat("Age, years", "age")
rowsdef.append(("Demographics", "Gender coded 1", s["gender_bin"].astype(float), "bin"))
for c, lab in [("hosp_ward_IP", "Inpatient"), ("hosp_ward_OP", "Outpatient"),
               ("hosp_ward_ER", "Emergency department"), ("hosp_ward_ICU", "Intensive care")]:
    rowsdef.append(("Setting of the index culture", lab, (s[c] > 0).astype(float), "bin"))
rowsdef.append(("Setting of the index culture", "Nursing home visit",
                (s["nursing_home"] > 0).astype(float), "bin"))
add_cat("Specimen", "culture_description", top=5)
add_cat("Ordering mode", "ordering_mode")
add_cat("Organism (analysed row)", "organism", top=6)
for c, lab in [("n_procedures", "Prior procedures, n"),
               ("n_prior_organisms", "Prior infecting organisms, n"),
               ("n_prior_resistance", "Prior resistant isolates, n"),
               ("n_comorbidity", "Comorbidity records, n"),
               ("adi_score", "Area Deprivation Index")]:
    rowsdef.append(("History and deprivation", lab, pd.to_numeric(s[c], errors="coerce"), "num"))


def smd(x, kind, weights):
    ok = x.notna().to_numpy()
    xv, av, wv = x.to_numpy(dtype=float)[ok], a[ok], weights[ok]
    m1 = np.average(xv[av == 1], weights=wv[av == 1])
    m0 = np.average(xv[av == 0], weights=wv[av == 0])
    if kind == "bin":
        v1, v0 = m1 * (1 - m1), m0 * (1 - m0)
    else:
        v1 = np.average((xv[av == 1] - m1) ** 2, weights=wv[av == 1])
        v0 = np.average((xv[av == 0] - m0) ** 2, weights=wv[av == 0])
    d = np.sqrt((v1 + v0) / 2)
    return float((m1 - m0) / d) if d > 0 else 0.0


out = []
for section, label, x, kind in rowsdef:
    rec = {"section": section, "label": label, "kind": kind}
    for g, mask in (("exposed", a == 1), ("unexposed", a == 0)):
        xv = x[mask]
        if kind == "bin":
            rec[f"{g}_n"] = int(xv.sum())
            rec[f"{g}_pct"] = float(100 * xv.mean())
        else:
            q = xv.dropna().quantile([0.25, 0.5, 0.75]).tolist()
            rec[f"{g}_q1"], rec[f"{g}_median"], rec[f"{g}_q3"] = q
            rec[f"{g}_mean"] = float(xv.mean())
    rec["smd_unweighted"] = smd(x, kind, np.ones_like(a))
    rec["smd_weighted"] = smd(x, kind, w)
    out.append(rec)
tab = pd.DataFrame(out)
tab.to_csv(os.path.join(results_dir, "fq_baseline_characteristics.csv"), index=False)

orders = set(s["order_proc_id_coded"])


def present(fname, cols):
    t = dataio.load_table(data_dir, fname, usecols=["order_proc_id_coded", *cols])
    t = t[t["order_proc_id_coded"].isin(orders)]
    if cols:
        t = t.dropna(subset=cols, how="all")
    return set(t["order_proc_id_coded"])


miss = []
n = len(s)


def record(var, missing_mask, handling):
    mm = np.asarray(missing_mask, dtype=bool)
    miss.append({"variable": var, "missing_n": int(mm.sum()), "missing_pct": 100 * mm.mean(),
                 "missing_exposed": int(mm[a == 1].sum()),
                 "missing_unexposed": int(mm[a == 0].sum()), "handling": handling})


record("Age band", s["age"].isna(), "missing category")
record("Gender", s["gender"].isna() if "gender" in s else np.zeros(n), "coded 0")
ward_orders = present("microbiology_cultures_ward_info.csv", [])
record("Care setting (ward record)", ~s["order_proc_id_coded"].isin(ward_orders), "all indicators 0")
record("Specimen type", s["culture_description"].isna(), "missing category")
record("Ordering mode", s["ordering_mode"].isna(), "missing category")
record("Area Deprivation Index", s["adi_score"].isna(), "training-period median")
acute = [c for c in s.columns if c.startswith("median_")]
record("Any acute laboratory or vital sign (sensitivity set only)",
       s[acute].isna().all(axis=1), "training-period median")
for c in acute:
    record(f"  {c.replace('median_', '')}", s[c].isna(), "training-period median")
pd.DataFrame(miss).to_csv(os.path.join(results_dir, "fq_missingness.csv"), index=False)

coll = collapse_class_resistance(coh.definitive, cfg.target_class)
coll = coll.sort_values(["anon_id", "order_time_jittered_utc"], kind="mergesort")
gap = pd.Series(np.nan, index=coll.index)
for _, grp in coll.groupby("anon_id", sort=False):
    t = grp["order_time_jittered_utc"].to_numpy()
    nxt = np.searchsorted(t, t, side="right")
    has = nxt < len(t)
    g = np.full(len(t), np.nan)
    g[has] = (t[nxt[has]] - t[has]) / np.timedelta64(1, "D")
    gap.loc[grp.index] = g
coll["gap_days"] = gap
gmap = coll.drop_duplicates("order_proc_id_coded").set_index("order_proc_id_coded")["gap_days"]
days = s["order_proc_id_coded"].map(gmap).to_numpy(dtype=float)
assert np.isfinite(days).all() and (days > 0).all() and (days <= cfg.followup_horizon_days).all()

fu = {}
for g, mask in (("all", np.ones(n, bool)), ("exposed", a == 1), ("unexposed", a == 0)):
    d = days[mask]
    fu[g] = {"n": int(mask.sum()), "median": float(np.median(d)),
             "q1": float(np.percentile(d, 25)), "q3": float(np.percentile(d, 75)),
             "within_30": float(np.mean(d <= 30)), "within_90": float(np.mean(d <= 90))}
summary = {"overlap_reproduced": ov,
           "clipped_fraction_reproduced": float(np.mean((pi_raw < cfg.propensity_clip)
                                                       | (pi_raw > 1 - cfg.propensity_clip))),
           "clipped_fraction_report": rc["crossfit"]["clipped_fraction"],
           "followup_days": fu,
           "n_orders": n, "n_exposed": int(a.sum())}
json.dump(summary, open(os.path.join(results_dir, "fq_baseline_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print(tab.to_string())
print(pd.DataFrame(miss).to_string())
