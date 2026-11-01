"""Two follow-up analyses on a finished run: an antibiogram baseline and a Reserve-free rule.

1. Antibiogram baseline. The hospital's own cumulative antibiogram, built from the training
   period: the resistance rate of each organism to each agent, with at least 30 isolates
   (CLSI M39), else the agent's overall training rate. It is scored on the same test rows
   and agents as the per-agent gradient boosting models.
2. Decision rule without WHO AWaRe Reserve agents (ceftazidime/avibactam and
   ceftolozane/tazobactam among the agents the rule can choose).

The cohort, split and features are rebuilt exactly as ``armd.pipeline.run`` does them (same
config, seed and random number order), the gradient boosting models are refitted with the
pipeline's own function, and the published metrics and frontier are reproduced before
anything new is reported. Class effects are read from the run, not re-estimated.

Run from the repository root:
    python -m scripts.antibiogram_reserve ARMD_dataset results/full
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from armd import cohort as cohort_mod, decision, evaluation
from armd.config import Config
from armd.drugclass import agent_to_class
from armd.features import CATEGORICAL_COLS, TabularEncoder, group_organisms, temporal_split
from armd.pipeline import _baseline

RESERVE = ("Ceftazidime/Avibactam", "Ceftolozane/Tazobactam")
MIN_ISOLATES = 30

data_dir, results_dir = sys.argv[1], sys.argv[2]
report = json.load(open(os.path.join(results_dir, "run_report.json")))
cfg_d = json.load(open(os.path.join(results_dir, "config.json")))
cfg = Config(**{k: v for k, v in cfg_d.items() if k in Config.__dataclass_fields__})

# ---- rebuild exactly as the pipeline does --------------------------------------------
rng = np.random.default_rng(cfg.seed)
coh = cohort_mod.build(cfg, data_dir, rng)
df, agents = coh.frame, coh.agents
train_idx, val_idx, test_idx = temporal_split(
    df["order_year"].to_numpy(), cfg.temporal_cutoff_year, df["anon_id"].to_numpy(),
    cfg.val_fraction, rng, cfg.group_by_patient)
assert {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)} == report["splits"]
df["organism_g"] = group_organisms(df, train_idx, cfg.top_k_organisms)
Xbase = TabularEncoder(coh.base_numeric_cols, list(CATEGORICAL_COLS)).fit(df, train_idx).transform(df)
Xexpo = df[coh.expo_cols].to_numpy(dtype=np.float32)
Xsucc = np.concatenate([Xbase, Xexpo], axis=1)

Yraw = df[agents].to_numpy(dtype=float)
M = (~np.isnan(Yraw)).astype(np.float32)
Y = np.nan_to_num(Yraw, nan=0.0).astype(np.float32)
keep = evaluation.evaluable_agents(Y, M, train_idx, test_idx, agents, cfg.min_train_events,
                                   cfg.min_train_positives, cfg.min_test_events)
assert len(keep) == report["agents_evaluable"], len(keep)

# ---- gradient boosting, refitted and checked against the run --------------------------
cache = os.path.join(results_dir, "xgb_test_predictions.npz")
if os.path.exists(cache):
    print("loading cached gradient boosting predictions")
    z = np.load(cache)
    xgb_proba, xgb_dense = z["proba"], z["dense"]
else:
    print("refitting the per-agent gradient boosting models")
    xgb_proba, _, xgb_dense = _baseline(Xsucc, Y, M, train_idx, test_idx, agents, keep, cfg, "cpu")
    np.savez_compressed(cache, proba=xgb_proba, dense=xgb_dense)
xgb_m = evaluation.per_agent_metrics(Y[test_idx], M[test_idx], xgb_proba, agents,
                                     cfg.min_test_events, cfg.n_bootstrap, cfg.seed,
                                     restrict_to=keep)
pub = pd.read_csv(os.path.join(results_dir, "susceptibility_xgb_metrics.csv"))
chk = xgb_m.merge(pub[["antibiotic", "auc"]], on="antibiotic", suffixes=("", "_pub"))
max_auc_diff = float((chk["auc"] - chk["auc_pub"]).abs().max())
print(f"  largest AUC difference from the published run: {max_auc_diff:.2e}")

# ---- antibiogram ---------------------------------------------------------------------
org = df["organism"].astype("string").to_numpy()
abg = np.full((len(test_idx), len(agents)), np.nan, dtype=np.float32)
abg_rows = []
for i, agent in enumerate(agents):
    if agent not in keep:
        continue
    mtr = M[train_idx, i] > 0
    tr = pd.DataFrame({"org": org[train_idx][mtr], "y": Y[train_idx, i][mtr]})
    overall = float(tr["y"].mean())
    g = tr.groupby("org")["y"].agg(["mean", "size"])
    rate = g.loc[g["size"] >= MIN_ISOLATES, "mean"].to_dict()
    abg[:, i] = [rate.get(o, overall) for o in org[test_idx]]
    mte = M[test_idx, i] > 0
    abg_rows.append({"antibiotic": agent, "organisms_with_own_rate": len(rate),
                     "test_results_with_own_rate": int(pd.Series(org[test_idx][mte]).isin(rate).sum()),
                     "test_results": int(mte.sum())})
abg_proba = np.where(M[test_idx] > 0, np.nan_to_num(abg), 0.0).astype(np.float32)
abg_m = evaluation.per_agent_metrics(Y[test_idx], M[test_idx], abg_proba, agents,
                                     cfg.min_test_events, cfg.n_bootstrap, cfg.seed,
                                     restrict_to=keep)

# Head-to-head on the agents the run compared, with a paired bootstrap per agent.
compared = pd.read_csv(os.path.join(results_dir, "susceptibility_net_vs_xgb.csv"))["antibiotic"]
rows = []
for agent in compared:
    i = agents.index(agent)
    m = M[test_idx][:, i] > 0
    y = Y[test_idx][m, i].astype(int)
    d = evaluation.paired_bootstrap_delta(y, xgb_dense[m, i], abg[m, i],
                                          n_boot=min(cfg.n_bootstrap, 400), seed=cfg.seed + i)
    rows.append({
        "antibiotic": agent, "n_test": int(m.sum()),
        "xgb_auc": float(xgb_m.set_index("antibiotic").loc[agent, "auc"]),
        "antibiogram_auc": float(abg_m.set_index("antibiotic").loc[agent, "auc"]),
        "delta_auc": d["delta"], "delta_lo": d["lo"], "delta_hi": d["hi"],
        "xgb_ece": evaluation.expected_calibration_error(y, xgb_dense[m, i]),
        "antibiogram_ece": evaluation.expected_calibration_error(y, abg[m, i]),
    })
h2h = pd.DataFrame(rows).merge(pd.DataFrame(abg_rows), on="antibiotic")
h2h.to_csv(os.path.join(results_dir, "susceptibility_xgb_vs_antibiogram.csv"), index=False)

# ---- decision rule, with and without Reserve agents ---------------------------------
ce = pd.read_csv(os.path.join(results_dir, "class_effects.csv"))
class_rd = {r.drug_class: float(r.risk_difference) for r in ce.itertuples() if r.positivity_ok}
agent_class = {a: agent_to_class(a) for a in agents}
cost, eligible = decision.agent_collateral_vector(agents, agent_class, class_rd)
predictable = decision._predictable(xgb_dense)

curve = decision.frontier(xgb_dense, agents, agent_class, class_rd)
pub_curve = pd.read_csv(os.path.join(results_dir, "decision_frontier.csv"))
max_curve_diff = float(np.abs(curve[["mean_p_failure", "mean_collateral"]].to_numpy()
                              - pub_curve[["mean_p_failure", "mean_collateral"]].to_numpy()).max())
print(f"  largest frontier difference from the published run: {max_curve_diff:.2e}")

no_res = xgb_dense.copy()
for a in RESERVE:
    no_res[:, agents.index(a)] = np.nan
curve_nr = decision.frontier(no_res, agents, agent_class, class_rd)
curve_nr.to_csv(os.path.join(results_dir, "decision_frontier_no_reserve.csv"), index=False)
chk_nr = decision.observed_failure_check(no_res, Y[test_idx], M[test_idx], agents, agent_class,
                                         class_rd, lam=cfg.lambda_decision)


def at_(c, lam):
    return c.loc[np.isclose(c["lambda"], lam)].iloc[0].to_dict()


# ---- decision rule restricted to the laboratory panel ---------------------------------
# A culture's candidates are the agents the laboratory reported for at least half of the
# training-period isolates of the same organism from the same specimen type (at least 30
# isolates; otherwise the specimen type's panel), less Reserve agents, with nitrofurantoin
# allowed for urine only.
PANEL_SHARE = 0.5
spec = df["culture_description"].astype("string").to_numpy()
tr_org, tr_spec, tr_M = org[train_idx], spec[train_idx], M[train_idx] > 0
panel_os, panel_s = {}, {}
for sp in np.unique(tr_spec):
    ms = tr_spec == sp
    panel_s[sp] = tr_M[ms].mean(axis=0) >= PANEL_SHARE
    for o in np.unique(tr_org[ms]):
        mo = ms & (tr_org == o)
        if mo.sum() >= MIN_ISOLATES:
            panel_os[(o, sp)] = tr_M[mo].mean(axis=0) >= PANEL_SHARE
te_org, te_spec = org[test_idx], spec[test_idx]
fallback = 0
cand = np.zeros((len(test_idx), len(agents)), dtype=bool)
for r, (o, sp) in enumerate(zip(te_org, te_spec)):
    if (o, sp) in panel_os:
        cand[r] = panel_os[(o, sp)]
    else:
        cand[r] = panel_s.get(sp, np.zeros(len(agents), dtype=bool))
        fallback += 1
nitro = agents.index("Nitrofurantoin")
urine_only = np.ones_like(cand)
urine_only[:, nitro] = te_spec == "URINE"
base_ok = (eligible & decision._predictable(no_res))[None, :]


def masked_frontier(p, allowed, lambdas=(0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)):
    """The frontier with a per-culture candidate set; cultures with no candidate are dropped."""
    keep_rows = allowed.any(axis=1)
    p, allowed = p[keep_rows], allowed[keep_rows]
    c = np.where(np.isnan(cost), 0.0, cost)

    def choose(lam):
        sc = np.where(allowed, decision.fuse(np.nan_to_num(p, nan=1.0), c[None, :], lam), np.inf)
        return np.argmin(sc, axis=1)

    ref = choose(1.0)
    out = []
    for lam in lambdas:
        ch = choose(lam)
        names, counts = np.unique(ch, return_counts=True)
        out.append({"lambda": lam,
                    "mean_p_failure": float(p[np.arange(len(ch)), ch].mean()),
                    "mean_collateral": float(c[ch].mean()),
                    "switched_from_pure_susceptibility": float(np.mean(ch != ref)),
                    "top_agent": agents[int(names[np.argmax(counts)])],
                    "n_distinct_agents": int(len(names))})
    return pd.DataFrame(out), keep_rows, choose


def observed(allowed_full, choose, keep_rows, lam):
    ch = choose(lam)
    rr = np.flatnonzero(keep_rows)
    tested = M[test_idx][rr, ch] > 0
    return {"n_scored": int(tested.sum()), "fraction_of_units_scorable": float(tested.mean()),
            "observed_resistance_rate_of_recommendation": float(Y[test_idx][rr, ch][tested].mean())}


variants = {"panel": cand & urine_only & base_ok, "urine_nitrofurantoin_only": urine_only & base_ok}
restricted = {}
for name, allowed in variants.items():
    cur, kr, ch = masked_frontier(no_res, allowed)
    cur.to_csv(os.path.join(results_dir, f"decision_frontier_{name}.csv"), index=False)
    sp_mix = pd.Series(te_spec[kr]).value_counts().to_dict()
    restricted[name] = {"cultures_with_candidates": int(kr.sum()), "cultures_total": int(len(kr)),
                        "specimens": sp_mix,
                        "mean_candidates": float(allowed[kr].sum(axis=1).mean()),
                        "lambda_1": at_(cur, 1.0), "lambda_0.5": at_(cur, 0.5),
                        "lambda_0.6": at_(cur, 0.6), "lambda_0.1": at_(cur, 0.1),
                        "observed_check_lambda_0.6": observed(allowed, ch, kr, cfg.lambda_decision)}
    print(f"--- {name}"); print(cur.to_string(index=False))
    by_spec = {}
    for lam in (1.0, cfg.lambda_decision):
        chosen = pd.Series([agents[i] for i in ch(lam)])
        for sp in np.unique(te_spec[kr]):
            top = chosen[te_spec[kr] == sp].value_counts(normalize=True).head(4).round(3)
            by_spec[f"{sp} lambda={lam}"] = top.to_dict()
            print(f"   {sp:12s} lambda={lam}: {top.to_dict()}")
    restricted[name]["top_agents_by_specimen"] = by_spec
restricted["panel"]["cultures_using_specimen_fallback"] = fallback


def at(c, lam):
    return c.loc[np.isclose(c["lambda"], lam)].iloc[0].to_dict()


ok = h2h
out = {
    "reproduction": {"max_auc_diff": max_auc_diff, "max_frontier_diff": max_curve_diff},
    "agents_mapped_to_usable_classes": int(eligible.sum()),
    "agents_scored_by_rule": int((eligible & predictable).sum()),
    "agents_scored_by_rule_without_reserve": int((eligible & decision._predictable(no_res)).sum()),
    "antibiogram": {
        "agents_compared": int(len(ok)),
        "xgb_mean_auc": float(ok["xgb_auc"].mean()),
        "antibiogram_mean_auc": float(ok["antibiogram_auc"].mean()),
        "xgb_better_significant": int((ok["delta_lo"] > 0).sum()),
        "antibiogram_better_significant": int((ok["delta_hi"] < 0).sum()),
        "xgb_median_ece": float(ok["xgb_ece"].median()),
        "antibiogram_median_ece": float(ok["antibiogram_ece"].median()),
        "share_of_test_results_with_own_rate":
            float(ok["test_results_with_own_rate"].sum() / ok["test_results"].sum()),
    },
    "no_reserve": {
        "reserve_agents_removed": list(RESERVE),
        "lambda_1": at(curve_nr, 1.0), "lambda_0.5": at(curve_nr, 0.5),
        "lambda_0.6": at(curve_nr, 0.6), "lambda_0.1": at(curve_nr, 0.1),
        "observed_check_lambda_0.6": chk_nr,
    },
    "restricted": restricted,
}
json.dump(out, open(os.path.join(results_dir, "antibiogram_reserve.json"), "w"), indent=2,
          default=str)
print(json.dumps(out, indent=2, default=str))
print(curve_nr.to_string(index=False))
