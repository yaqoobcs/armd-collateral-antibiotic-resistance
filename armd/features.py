"""Feature blocks and a transformer whose statistics come from the training rows only.

Three blocks with different roles:

``base``      organism, specimen, care setting, demographics, prior-history counts.
              Shared: it feeds the encoder and therefore both heads.
``exposure``  prior antibiotic-class indicators. Susceptibility head only: the
              treatment is defined from these columns, so letting them reach the
              propensity model would make exposure almost perfectly predictable.
``severity``  laboratory and vital-sign summaries, comorbidity burden, deprivation.
              Causal heads only.

Everything that has to be estimated (medians, means, standard deviations, the category
vocabulary, the list of frequent organisms) is estimated on the training rows and then
applied to validation and test, so the test period cannot influence the feature space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

WARD_COLS = ("hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU")
COUNT_COLS = ("n_procedures", "n_prior_organisms", "n_prior_resistance", "nursing_home")
CATEGORICAL_COLS = ("organism_g", "culture_description", "ordering_mode", "age")

LAB_STUBS = ("wbc", "cr", "lactate", "plt", "bun")
VITAL_STUBS = ("heartrate", "resprate", "temp", "sysbp")


@dataclass
class TabularEncoder:
    """Median imputation, z-scoring and one-hot expansion, all fitted on train rows."""

    numeric_cols: Sequence[str]
    categorical_cols: Sequence[str]
    medians_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    stds_: Optional[np.ndarray] = None
    categories_: dict = field(default_factory=dict)
    feature_names_: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame, train_rows: np.ndarray) -> "TabularEncoder":
        num = self._numeric_matrix(df)
        train = num[train_rows]
        if train.size:
            all_nan = np.isnan(train).all(axis=0)
            medians = np.zeros(num.shape[1])
            if (~all_nan).any():
                medians[~all_nan] = np.nanmedian(train[:, ~all_nan], axis=0)
            self.medians_ = medians
        else:
            self.medians_ = np.zeros(num.shape[1])
        self.medians_ = np.nan_to_num(self.medians_, nan=0.0)
        filled = self._impute(num)
        self.means_ = filled[train_rows].mean(axis=0)
        self.stds_ = filled[train_rows].std(axis=0) + 1e-6

        self.categories_ = {}
        sub = df.iloc[train_rows]
        for col in self.categorical_cols:
            if col in df.columns:
                vals = sub[col].astype("string").fillna("__MISSING__")
                self.categories_[col] = sorted(vals.unique().tolist())

        self.feature_names_ = list(self.numeric_cols) + [
            f"{col}={lvl}" for col in self.categorical_cols
            for lvl in self.categories_.get(col, [])
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._impute(self._numeric_matrix(df))
        num = (num - self.means_) / self.stds_
        blocks = [num.astype(np.float32)]
        for col in self.categorical_cols:
            levels = self.categories_.get(col, [])
            if not levels:
                continue
            vals = (df[col].astype("string").fillna("__MISSING__")
                    if col in df.columns else pd.Series(["__MISSING__"] * len(df)))
            oh = np.zeros((len(df), len(levels)), dtype=np.float32)
            index = {lvl: i for i, lvl in enumerate(levels)}
            codes = vals.map(index).to_numpy()
            seen = ~pd.isna(codes)
            oh[np.where(seen)[0], codes[seen].astype(int)] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(df), 0), np.float32)

    def fit_transform(self, df: pd.DataFrame, train_rows: np.ndarray) -> np.ndarray:
        return self.fit(df, train_rows).transform(df)

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if not len(self.numeric_cols):
            return np.zeros((len(df), 0), dtype=np.float64)
        cols = []
        for c in self.numeric_cols:
            s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64))
        return np.column_stack(cols)

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        bad = np.isnan(out)
        if bad.any():
            out[bad] = np.take(self.medians_, np.where(bad)[1])
        return np.nan_to_num(out, nan=0.0)


def group_organisms(df: pd.DataFrame, train_rows: np.ndarray, top_k: int) -> pd.Series:
    """Collapse rare organisms, choosing the frequent ones from the training rows only."""
    top = df.iloc[train_rows]["organism"].value_counts().head(top_k).index
    return pd.Series(
        np.where(df["organism"].isin(top), df["organism"].astype("string"), "OTHER_ORGANISM"),
        index=df.index,
        dtype="string",
    )


def temporal_split(
    years: np.ndarray,
    cutoff_year: int,
    patients: Optional[np.ndarray],
    val_fraction: float,
    rng: np.random.Generator,
    group_by_patient: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train on earlier years, test on later ones; carve a validation set out of train.

    When ``group_by_patient`` the validation carve-out is by patient, so early stopping is not
    chosen on rows from a patient the model has already fitted.
    """
    is_test = years >= cutoff_year
    trainval = np.where(~is_test)[0]
    test = np.where(is_test)[0]

    if not len(trainval):
        return trainval, np.array([], dtype=int), test

    if group_by_patient and patients is not None:
        uniq = np.unique(patients[trainval])
        n_val = max(int(round(len(uniq) * val_fraction)), 1)
        val_patients = set(rng.permutation(uniq)[:n_val].tolist())
        in_val = np.array([p in val_patients for p in patients[trainval]])
        return trainval[~in_val], trainval[in_val], test

    perm = rng.permutation(trainval)
    n_val = int(len(trainval) * val_fraction)
    return perm[n_val:], perm[:n_val], test
