"""Collect the cohort-size sensitivity table (Table S5).

Every run uses the same code, seed and configuration; only ``sample_n_patients`` changes.
"""
import json, sys, os
import pandas as pd

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

RUNS = [("40,000", "results/n40000"),
        ("80,000", "results/n80000"),
        ("160,000", "results/n160000"),
        ("283,715 (all)", "results/full")]

rows = []
for label, d in RUNS:
    f = os.path.join(d, "run_report.json")
    if not os.path.exists(f):
        print(f"  [skip] {d} not present", file=sys.stderr); continue
    r = json.load(open(f))
    c, b = r["causal"], r["baseline"]
    cf, rp = c["crossfit"], c["representation"]
    ce = c["class_effects"]["estimated"]
    ok = {k: v for k, v in ce.items() if v["positivity_ok"]}
    excl = [k for k, v in ok.items()
            if v.get("lo_simultaneous", 0) > 0 or v.get("hi_simultaneous", 0) < 0]
    rows.append({
        "patients": label,
        "units": r["cohort"]["culture_organism_units"],
        "causal_orders": c["units"],
        "agents": r["agents_evaluable"],
        "net_auc": b["net_mean_auc"],
        "xgb_auc": b["xgb_mean_auc"],
        "auc_gap": b["xgb_mean_auc"] - b["net_mean_auc"],
        "rd": cf["estimate"],
        "ci_width": cf["hi"] - cf["lo"],
        "clip": cf["clipped_fraction"],
        "rep_rd": rp["estimate"],
        "rep_width_ratio": (rp["hi"] - rp["lo"]) / (cf["hi"] - cf["lo"]),
        "rep_clip_ratio": rp["clipped_fraction"] / cf["clipped_fraction"],
        "classes_estimated": len(ce),
        "classes_interpretable": len(ok),
        "classes_excluding_zero": len(excl),
    })

df = pd.DataFrame(rows)
df.to_csv("outputs/tables/sample_size_sensitivity.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
