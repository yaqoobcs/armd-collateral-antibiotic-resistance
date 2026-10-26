"""Draw the JAC-AMR figures at 300 dpi from the saved full-cohort results.

Nothing is re-estimated: the forest plot reads run_report.json, the frontier reads
decision_frontier.csv, and the overlap histogram reads the out-of-fold propensity scores
saved by scripts/fq_baseline.py. The cohort flow chart has its own generator,
scripts/cohort_flow_png.py.

Run from the repository root:  python -m scripts.jacamr_figures
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from armd import plots
import os

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

R = "results/full/"
OUT = "outputs/figures"
DPI = 300


def _save(fig, results_dir, name):
    path = f"{results_dir}/{name}"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


plots._save = _save

rep = json.load(open(R + "run_report.json"))
cfg = json.load(open(R + "config.json"))
c = rep["causal"]
cf, nc = c["crossfit"], c["negative_control"]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for yy, (lab, r) in enumerate([("Recent fluoroquinolone exposure", cf),
                               ("Negative control (nitrofuran)", nc)]):
    ax.plot([r["lo"], r["hi"]], [yy, yy], color=plots.BLUE, lw=2.2)
    ax.plot(r["estimate"], yy, "o", color="#16527f", ms=8)
ax.axvline(0, color=plots.GREY, ls="--", lw=1)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Recent fluoroquinolone exposure", "Negative control (nitrofuran)"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("risk difference for collateral resistance")
ax.set_title("Effect estimates with 95% intervals")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
_save(fig, OUT, "causal_estimates.png")

z = np.load(R + "fq_propensity_oof.npz")
clip = cfg["propensity_clip"]
pi, a = np.clip(z["pi_raw"], clip, 1 - clip), z["a"]
clipped = float(np.mean((z["pi_raw"] < clip) | (z["pi_raw"] > 1 - clip)))
assert np.isclose(clipped, cf["clipped_fraction"]), clipped
assert len(a) == cf["n"] and int(a.sum()) == c["overlap"]["exposed_n"]
plots.propensity_overlap(pi, a, OUT, "crossfit_overlap.png",
                         "Out-of-fold propensity, patient-grouped folds")

curve = pd.read_csv(R + "decision_frontier.csv")
plots.decision_frontier(curve, OUT, default_lambda=cfg["lambda_decision"])
print("written", OUT)
