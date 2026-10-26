"""Command line entry point.

    python -m armd.run_analysis --smoke
    python -m armd.run_analysis --config configs/full_run.json --data-dir ARMD_dataset

``--smoke`` runs the whole pipeline on a generated synthetic release.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from armd.config import Config, OPTIONAL_TABLES, check_required, missing_tables, resolve_data_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ARMD joint susceptibility and collateral-resistance analysis")
    p.add_argument("--data-dir", default=None, help="raw CSV directory (AMRD_dataset or ARMD_dataset)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--target-class", default=None)
    p.add_argument("--control-class", default=None)
    p.add_argument("--sample-patients", type=int, default=None)
    p.add_argument("--cutoff-year", type=int, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--missing-as-negative", action="store_true",
                   help="code an absent next culture as no collateral-resistance event")
    p.add_argument("--config", default=None, help="JSON config file; command line flags win")
    p.add_argument("--smoke", action="store_true",
                   help="generate a small synthetic release and run on that")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.from_json(args.config) if args.config else Config()

    for flag, field in (
        ("results_dir", "results_dir"), ("target_class", "target_class"),
        ("control_class", "control_class"), ("seed", "seed"), ("device", "device"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(cfg, field, value)
    if args.sample_patients is not None:
        cfg.sample_n_patients = args.sample_patients if args.sample_patients > 0 else None
    if args.cutoff_year is not None:
        cfg.temporal_cutoff_year = args.cutoff_year
    if args.folds is not None:
        cfg.n_folds = args.folds
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.missing_as_negative:
        cfg.missing_as_negative = True

    tmp = None
    if args.smoke:
        from armd.synthetic import generate

        tmp = tempfile.mkdtemp(prefix="armd_smoke_")
        truth = generate(tmp, n_patients=900, seed=cfg.seed)
        data_dir = tmp
        cfg.sample_n_patients = None
        cfg.temporal_cutoff_year = 2019
        cfg.max_epochs = min(cfg.max_epochs, 25)
        cfg.n_bootstrap = 150
        cfg.top_k_organisms = 10
        cfg.min_train_events = 50
        cfg.min_test_events = 20
        cfg.results_dir = args.results_dir or "smoke_results"
        print("SMOKE RUN on synthetic data:", truth)
    else:
        try:
            data_dir = resolve_data_dir(args.data_dir)
            check_required(data_dir)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        absent = missing_tables(data_dir, OPTIONAL_TABLES)
        if absent:
            print("warning: optional tables absent, the corresponding features will be empty:")
            for name in absent:
                print("   -", name)

    cfg.data_dir = data_dir
    from armd.pipeline import run

    report = run(cfg, data_dir, verbose=not args.quiet)
    cfg.to_json(os.path.join(cfg.results_dir, "config.json"))
    print("\nresults written to", os.path.abspath(cfg.results_dir))
    print(open(os.path.join(cfg.results_dir, "summary_report.md")).read())

    if args.smoke:
        rd = report.get("causal", {}).get("crossfit", {}).get("estimate", float("nan"))
        print(f"smoke check: recovered risk difference {rd:+.4f} "
              f"(a planted positive effect; a value near zero means the pipeline is broken)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
