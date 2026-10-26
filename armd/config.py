"""Configuration and path resolution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

DATA_DIR_CANDIDATES = ("AMRD_dataset", "ARMD_dataset", "data")

REQUIRED_TABLES = ("microbiology_cultures_cohort.csv",)

OPTIONAL_TABLES = (
    "microbiology_cultures_demographics.csv",
    "microbiology_cultures_ward_info.csv",
    "microbiology_cultures_nursing_home_visits.csv",
    "microbiology_cultures_priorprocedures.csv",
    "microbiology_culture_prior_infecting_organism.csv",
    "microbiology_cultures_microbial_resistance.csv",
    "microbiology_cultures_antibiotic_class_exposure.csv",
    "microbiology_cultures_labs.csv",
    "microbiology_cultures_vitals.csv",
    "microbiology_cultures_adi_scores.csv",
    "microbiology_cultures_comorbidity.csv",
)


def resolve_data_dir(explicit: Optional[str] = None, root: str = ".") -> str:
    """Return the raw data directory, accepting the ARMD_dataset or AMRD_dataset spelling.

    Raises FileNotFoundError with an actionable message rather than letting a later ``None``
    table surface as an opaque AttributeError.
    """
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"data directory not found: {explicit}")
        return explicit
    for name in DATA_DIR_CANDIDATES:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "no data directory found; looked for "
        + ", ".join(DATA_DIR_CANDIDATES)
        + f" under {os.path.abspath(root)}"
    )


def missing_tables(data_dir: str, tables: Sequence[str]) -> list[str]:
    return [t for t in tables if not os.path.exists(os.path.join(data_dir, t))]


def check_required(data_dir: str) -> None:
    """Fail fast, with a message that names the files and where to get them."""
    missing = missing_tables(data_dir, REQUIRED_TABLES)
    if missing:
        raise FileNotFoundError(
            "required table(s) missing from "
            + os.path.abspath(data_dir)
            + ": "
            + ", ".join(missing)
            + ".\nThe cohort table is the analytic backbone; it carries culture, organism, "
            "antibiotic, susceptibility, was_positive, culture_description and ordering_mode. "
            "Download it from Dryad: https://doi.org/10.5061/dryad.jq2bvq8kp. "
            "To exercise the pipeline without the real data, run with --smoke."
        )


@dataclass
class Config:
    """Every knob the analysis exposes, in one place and serialisable to JSON."""

    data_dir: Optional[str] = None
    sample_n_patients: Optional[int] = 80_000
    top_k_organisms: int = 30

    target_class: str = "Fluoroquinolone"
    control_class: str = "Nitrofuran"
    exposure_window_days: int = 90
    followup_horizon_days: int = 365
    missing_as_negative: bool = False

    temporal_cutoff_year: int = 2021
    val_fraction: float = 0.15
    n_folds: int = 5
    group_by_patient: bool = True

    min_train_events: int = 100
    min_train_positives: int = 10
    min_test_events: int = 25
    n_bootstrap: int = 500

    batch_size: int = 4096
    encoder_dims: Sequence[int] = field(default_factory=lambda: [256, 128])
    rep_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 80
    patience: int = 8
    causal_loss_weight: float = 0.5

    propensity_clip: float = 0.025
    lambda_decision: float = 0.6

    seed: int = 20240611
    results_dir: str = "advanced_results"
    device: Optional[str] = None

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=list)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls(**json.load(fh))
