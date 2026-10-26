# Class-level causal estimates of collateral antibiotic resistance

Analysis code for the article *Class-level causal estimates of collateral antibiotic
resistance from routine electronic health records* (Yaqoob T, Yaqoob M,
Ibrahim M).

The code uses the Antibiotic Resistance Microbiology Dataset (ARMD) to answer three questions:

1. **Collateral cost.** How much does recent exposure to an antibiotic class raise the
   probability that the patient's next positive culture is resistant to that class? The
   estimator is cross-fitted augmented inverse probability weighting (AIPW) with
   patient-grouped folds and patient-clustered confidence intervals.
2. **What the data can support.** Which classes pass an overlap check, defined as fewer than
   25% of orders needing their propensity score clipped. Classes that fail are reported but
   not interpreted.
3. **Decision rule.** For each agent the rule scores
   `λ × P(failure now) + (1 − λ) × collateral cost of the agent's class`. Failure
   probabilities come from per-agent gradient boosting models. Sweeping λ from 0 to 1
   traces the trade-off frontier.

The paper also reports a shared neural network as a comparison. It did not beat the per-agent
models, and using its learned representation as the causal adjustment set widened the
interval.

## Repository layout

| Path | Contents |
|---|---|
| `armd/` | The pipeline. Covers cohort construction, drug-class mapping, features, the shared network, training, causal estimation, the decision rule and plots. |
| `armd/synthetic.py` | Generates a synthetic ARMD-shaped release with a planted collateral effect, so the whole pipeline runs without the real data. |
| `tests/` | Unit and integration tests. None needs the real data. |
| `configs/full_run.json` | The exact configuration of the run reported in the paper (seed 20240611). |
| `scripts/` | Turn saved results into the paper's figures and supplementary tables (see below). |

## Installation

```bash
conda env create -f environment.yml
conda activate armd_env
pip install -e .
```

`environment.yml` requests the CUDA build of PyTorch. On a machine without an NVIDIA GPU,
remove the `pytorch-cuda` line. The paper's results were produced on the CPU.

The paper's run used Python 3.11, pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0, PyTorch 2.5.1
and XGBoost 3.2.0. On macOS, XGBoost must be imported before PyTorch, or the process crashes
because both ship an OpenMP runtime. `armd/__init__.py` handles this, so keep its import
order.

## Checking the installation without the data

```bash
pytest -q                                   # about 120 tests on synthetic data
python -m armd.run_analysis --smoke         # full pipeline on a generated release
```

The smoke run ends by printing the recovered risk difference for the planted effect. A value
near zero means something is broken.

## Data

ARMD is not redistributed here. Download it from Dryad under the terms set by its authors:

> Nateghi Haredasht F, Amrollahi F, Maddali MV, *et al.* Antibiotic Resistance Microbiology
> Dataset (ARMD). Dryad, 2025. https://doi.org/10.5061/dryad.jq2bvq8kp
>
> Described in: *Sci Data* 2025; **12**: 1299. https://doi.org/10.1038/s41597-025-05649-7

Put the CSV files in a directory named `ARMD_dataset/` at the repository root, or pass the
location with `--data-dir`. The pipeline stops with a message naming any required table that
is missing.

## Reproducing the paper

All commands run from the repository root.

**1. Main analysis** (all 283,715 patients):

```bash
python -m armd.run_analysis --config configs/full_run.json \
    --data-dir ARMD_dataset --results-dir results/full --device cpu
```

This writes `results/full/run_report.json`, which holds every number in the paper, together
with the per-class estimates, the decision frontier and diagnostic plots.

**2. Cohort-size sensitivity (Table S5).** Repeat step 1 with identical settings on random
subsets of patients:

```bash
for n in 40000 80000 160000; do
  python -m armd.run_analysis --config configs/full_run.json --data-dir ARMD_dataset \
      --results-dir results/n$n --sample-patients $n --device cpu
done
python -m scripts.sample_sensitivity        # -> outputs/tables/sample_size_sensitivity.csv
```

**3. Baseline characteristics, missing data and follow-up (Tables S6 and S7).** This step
rebuilds the fluoroquinolone cohort with the same seed. It stops with an error unless it
reproduces the 32,724 analysed orders and 4,646 exposed orders. It also saves the
out-of-fold propensity scores used for Figure 3.

```bash
python -m scripts.fq_baseline ARMD_dataset results/full
python -m scripts.fq_baseline_tex           # -> outputs/tables/supp_tab_{baseline,missing}.tex
```

**4. Figures.** Each figure is drawn from the saved results; nothing is re-estimated.

```bash
python -m scripts.cohort_flow_png           # Figure 1, cohort derivation
python -m scripts.jacamr_figures            # Figures 2-4: estimates, overlap, frontier
python -m scripts.cohort_flow_drawio        # editable draw.io version of Figure 1
python -m scripts.methodology_drawio        # analysis overview diagram (draw.io)
```

Figures are written to `outputs/figures/` at 300 dpi. To export the draw.io diagrams to
PNG, use draw.io:
`draw.io -x -f png -s 2.5 -o outputs/figures/methodology.png outputs/figures/methodology.drawio`.

## Notes on the data that affect the analysis

- **Timing.** Every `*_time_to_culturetime` field counts days **before** the culture order.
  Exposure is therefore receipt of a class in the 90 days before the index culture, not
  therapy started at the time of culture.
- **Sex coding.** Sex is coded 0, 1 or null, and the release does not say which value is
  which. It is read as a numeric field, so the null code cannot silently turn the column
  into strings.
- **Units of analysis.** A row is a culture paired with an organism it grew. For causal
  estimation, rows are reduced to one per culture order. Otherwise a culture that grew two
  organisms would count the same treatment and outcome twice.
- **Directory name.** The data directory may be named `ARMD_dataset`, `AMRD_dataset` or
  `data`; `armd/config.py` looks for all three.

## Citation

If you use this code, please cite the article (details to be added on publication) and the
ARMD dataset above.

## Licence

The code is released under the MIT Licence; see `LICENSE`. The licence covers the code only.
The ARMD data remain under the terms set by their authors on Dryad.
