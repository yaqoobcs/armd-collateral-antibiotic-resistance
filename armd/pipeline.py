"""End-to-end analysis: cohort, model, baseline, causal estimation, exports."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from armd import causal, cohort as cohort_mod, decision, evaluation, plots, training
from armd.config import Config
from armd.drugclass import agent_to_class, audit_against_native
from armd.features import CATEGORICAL_COLS, TabularEncoder, group_organisms, temporal_split
from armd.model import SharedNet


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), device=device)


def run(cfg: Config, data_dir: str, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = training.pick_device(cfg.device)
    os.makedirs(cfg.results_dir, exist_ok=True)
    report: dict = {"config": {k: v for k, v in cfg.__dict__.items()}, "device": device}

    if verbose:
        print("[1/8] building the cohort")
    coh = cohort_mod.build(cfg, data_dir, rng)
    df, agents = coh.frame, coh.agents
    report["cohort"] = coh.notes
    if verbose:
        for k, v in coh.notes.items():
            print(f"   {k}: {v}")

    audit = audit_against_native(coh.exposure_window)
    if len(audit):
        audit.to_csv(os.path.join(cfg.results_dir, "drug_class_audit.csv"), index=False)
        report["drug_class_audit_rows"] = int(len(audit))

    if verbose:
        print("[2/8] splitting by period, grouping the validation carve-out by patient")
    years = df["order_year"].to_numpy()
    patients = df["anon_id"].to_numpy()
    train_idx, val_idx, test_idx = temporal_split(
        years, cfg.temporal_cutoff_year, patients, cfg.val_fraction, rng, cfg.group_by_patient
    )
    report["splits"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    if not len(train_idx) or not len(test_idx):
        raise ValueError(
            f"temporal cutoff {cfg.temporal_cutoff_year} leaves an empty partition "
            f"(train={len(train_idx)}, test={len(test_idx)}); adjust temporal_cutoff_year"
        )

    df["organism_g"] = group_organisms(df, train_idx, cfg.top_k_organisms)
    base_encoder = TabularEncoder(coh.base_numeric_cols, list(CATEGORICAL_COLS)).fit(df, train_idx)
    Xbase = base_encoder.transform(df)
    sev_encoder = TabularEncoder(coh.sev_cols, []).fit(df, train_idx)
    Xsev = sev_encoder.transform(df)
    sev_pre_encoder = TabularEncoder(coh.sev_pre_cols, []).fit(df, train_idx)
    Xsev_pre = sev_pre_encoder.transform(df)
    Xexpo = (df[coh.expo_cols].to_numpy(dtype=np.float32)
             if coh.expo_cols else np.zeros((len(df), 0), np.float32))
    Xsucc = np.concatenate([Xbase, Xexpo], axis=1)
    Xcausal = np.concatenate([Xbase, Xsev_pre], axis=1)
    Xcausal_acute = np.concatenate([Xbase, Xsev], axis=1)
    report["feature_dims"] = {"base": Xbase.shape[1], "exposure": Xexpo.shape[1],
                              "severity": Xsev.shape[1],
                              "severity_pre_exposure": Xsev_pre.shape[1],
                              "causal_primary": Xcausal.shape[1],
                              "causal_with_acute": Xcausal_acute.shape[1]}

    Ysucc = df[agents].to_numpy(dtype=float)
    Msucc = (~np.isnan(Ysucc)).astype(np.float32)
    Ysucc = np.nan_to_num(Ysucc, nan=0.0).astype(np.float32)
    A = df["A"].to_numpy(dtype=np.float32)
    Ycoll = np.nan_to_num(df["Ycoll"].to_numpy(dtype=float), nan=0.0).astype(np.float32)
    Mc = df["causal_valid"].to_numpy(dtype=np.float32)

    tensors = {
        "base": _to_tensor(Xbase, device), "expo": _to_tensor(Xexpo, device),
        "sev": _to_tensor(Xsev, device), "ys": _to_tensor(Ysucc, device),
        "ms": _to_tensor(Msucc, device), "a": _to_tensor(A, device),
        "yc": _to_tensor(Ycoll, device), "mc": _to_tensor(Mc, device),
    }

    if verbose:
        print(f"[3/8] training the shared network on {device}")
    model = SharedNet(Xbase.shape[1], Xexpo.shape[1], Xsev.shape[1], len(agents),
                      cfg.encoder_dims, cfg.rep_dim, cfg.dropout).to(device)
    trained = training.train(model, tensors, train_idx, val_idx, cfg, device, verbose)
    report["training"] = {"epochs": trained.epochs_run, "best_val_loss": trained.best_val,
                          "seconds": round(trained.seconds, 1)}
    plots.loss_curves(trained.history, cfg.results_dir)

    if verbose:
        print("[4/8] evaluating susceptibility")
    keep_agents = evaluation.evaluable_agents(
        Ysucc, Msucc, train_idx, test_idx, agents,
        cfg.min_train_events, cfg.min_train_positives, cfg.min_test_events,
    )
    report["agents_evaluable"] = len(keep_agents)

    preds = training.predict(model, tensors, test_idx, device)
    P_test = preds["susceptibility"]
    net_metrics = evaluation.per_agent_metrics(
        Ysucc[test_idx], Msucc[test_idx], P_test, agents,
        cfg.min_test_events, cfg.n_bootstrap, cfg.seed, restrict_to=keep_agents,
    )
    net_metrics.insert(1, "model", "shared_net")
    report["net_support_weighted_auc"] = evaluation.support_weighted_mean(net_metrics)
    report["net_median_auc"] = float(net_metrics.loc[net_metrics.status == "ok", "auc"].median())
    plots.auc_forest(net_metrics, cfg.results_dir)

    if verbose:
        print("[5/8] per-agent gradient boosting baseline")
    xgb_proba, xgb_rows, xgb_dense = _baseline(Xsucc, Ysucc, Msucc, train_idx, test_idx,
                                               agents, keep_agents, cfg, device)
    xgb_metrics = evaluation.per_agent_metrics(
        Ysucc[test_idx], Msucc[test_idx], xgb_proba, agents,
        cfg.min_test_events, cfg.n_bootstrap, cfg.seed, restrict_to=keep_agents,
    )
    xgb_metrics.insert(1, "model", "xgboost")

    compare = (
        net_metrics[net_metrics.status == "ok"][["antibiotic", "n_test", "resistance_rate", "auc", "auc_lo", "auc_hi"]]
        .rename(columns={"auc": "net_auc", "auc_lo": "net_lo", "auc_hi": "net_hi"})
        .merge(
            xgb_metrics[xgb_metrics.status == "ok"][["antibiotic", "auc"]].rename(columns={"auc": "xgb_auc"}),
            on="antibiotic", how="inner",
        )
    )
    deltas = []
    for _, row in compare.iterrows():
        i = agents.index(row["antibiotic"])
        m = Msucc[test_idx][:, i] > 0
        deltas.append(evaluation.paired_bootstrap_delta(
            Ysucc[test_idx][m, i].astype(int), P_test[m, i], xgb_proba[m, i],
            n_boot=min(cfg.n_bootstrap, 400), seed=cfg.seed + i,
        ))
    compare["delta_auc"] = [d["delta"] for d in deltas]
    compare["delta_lo"] = [d["lo"] for d in deltas]
    compare["delta_hi"] = [d["hi"] for d in deltas]
    compare["net_better"] = compare["delta_auc"] > 0
    compare["net_better_significant"] = compare["delta_lo"] > 0
    compare["xgb_better_significant"] = compare["delta_hi"] < 0

    report["baseline"] = {
        "agents_compared": int(len(compare)),
        "net_mean_auc": float(compare["net_auc"].mean()),
        "xgb_mean_auc": float(compare["xgb_auc"].mean()),
        "net_support_weighted_auc": float(np.average(compare["net_auc"], weights=compare["n_test"])),
        "xgb_support_weighted_auc": float(np.average(compare["xgb_auc"], weights=compare["n_test"])),
        "net_better_count": int(compare["net_better"].sum()),
        "net_better_significant": int(compare["net_better_significant"].sum()),
        "xgb_better_significant": int(compare["xgb_better_significant"].sum()),
    }
    if len(compare):
        plots.net_vs_baseline(compare, cfg.results_dir)

    cal_tables = {}
    for agent in compare.sort_values("n_test", ascending=False)["antibiotic"].head(3):
        i = agents.index(agent)
        m = Msucc[test_idx][:, i] > 0
        cal_tables[agent] = evaluation.calibration_table(Ysucc[test_idx][m, i], P_test[m, i])
        report.setdefault("calibration_ece", {})[agent] = evaluation.expected_calibration_error(
            Ysucc[test_idx][m, i], P_test[m, i]
        )
    for i, agent in enumerate(agents):
        if agent not in keep_agents:
            continue
        m = Msucc[test_idx][:, i] > 0
        if m.sum() < cfg.min_test_events or len(np.unique(Ysucc[test_idx][m, i])) < 2:
            continue
        report.setdefault("calibration_ece_all", {})[agent] = {
            "shared_net": evaluation.expected_calibration_error(
                Ysucc[test_idx][m, i], P_test[m, i]),
            "baseline": evaluation.expected_calibration_error(
                Ysucc[test_idx][m, i], xgb_dense[m, i]),
            "n_test": int(m.sum()),
        }
    plots.calibration(cal_tables, cfg.results_dir)

    if verbose:
        print("[6/8] causal estimation")
    ctx = {
        "Xcausal_acute": Xcausal_acute, "Xbase": Xbase, "Xsev_pre": Xsev_pre,
        "tensors": tensors, "train_idx": train_idx, "n_agents": len(agents),
        "coh": coh, "agents": agents,
        "causal_feature_names": list(base_encoder.feature_names_) + list(sev_pre_encoder.feature_names_)
        if hasattr(base_encoder, "feature_names_") else None,
    }
    report["causal"] = _causal(df, Xcausal, A, cfg, model, tensors, test_idx, device,
                               verbose, ctx)

    if verbose:
        print("[7/8] decision layer")
    report["decision"] = _decision(P_test, xgb_dense, Ysucc, Msucc, test_idx, agents,
                                   report["causal"], cfg, verbose)

    if verbose:
        print("[8/8] writing results")

    net_metrics.to_csv(os.path.join(cfg.results_dir, "susceptibility_net_metrics.csv"), index=False)
    xgb_metrics.to_csv(os.path.join(cfg.results_dir, "susceptibility_xgb_metrics.csv"), index=False)
    compare.to_csv(os.path.join(cfg.results_dir, "susceptibility_net_vs_xgb.csv"), index=False)

    dec = report.get("decision", {})
    if dec.get("available"):
        pd.DataFrame(dec["frontier"]).to_csv(
            os.path.join(cfg.results_dir, "decision_frontier.csv"), index=False)
        plots.decision_frontier(pd.DataFrame(dec["frontier"]), cfg.results_dir,
                                default_lambda=cfg.lambda_decision)
    ce = report.get("causal", {}).get("class_effects", {}).get("estimated", {})
    if ce:
        pd.DataFrame([
            {"drug_class": k, "risk_difference": v["estimate"], "lo": v["lo"], "hi": v["hi"],
             "lo_simultaneous": v.get("lo_simultaneous"), "hi_simultaneous": v.get("hi_simultaneous"),
             "n": v["n"], "patients": v["n_clusters"], "n_exposed": v.get("n_exposed"),
             "baseline_risk": v.get("baseline_risk"), "clipped_fraction": v.get("clipped_fraction"),
             "e_value_point": v.get("e_value_point"), "se": v["se"],
             "positivity_ok": v.get("positivity_ok")}
            for k, v in ce.items()
        ]).sort_values("risk_difference", ascending=False).to_csv(
            os.path.join(cfg.results_dir, "class_effects.csv"), index=False)

    antibiogram = predicted_antibiogram(model, tensors, int(test_idx[0]), agents, device)
    antibiogram.to_csv(os.path.join(cfg.results_dir, "predicted_antibiogram_example.csv"), index=False)

    with open(os.path.join(cfg.results_dir, "run_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    _write_summary(report, cfg)
    return report


def _baseline(Xsucc, Ysucc, Msucc, train_idx, test_idx, agents, keep_agents, cfg, device):
    """One gradient boosting classifier per agent, on the same features and split."""
    try:
        import xgboost as xgb
        use_xgb = True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        use_xgb = False
        print("  [warn] xgboost unavailable; falling back to HistGradientBoostingClassifier")

    proba = np.zeros((len(test_idx), len(agents)), dtype=np.float32)
    dense = np.full((len(test_idx), len(agents)), np.nan, dtype=np.float32)
    rows = []
    for i, agent in enumerate(agents):
        if agent not in keep_agents:
            continue
        mtr = Msucc[train_idx, i] > 0
        mte = Msucc[test_idx, i] > 0
        ytr = Ysucc[train_idx, i][mtr].astype(int)
        if use_xgb:
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.9,
                colsample_bytree=0.8, tree_method="hist",
                device="cuda" if device == "cuda" else "cpu",
                eval_metric="logloss", n_jobs=4, random_state=cfg.seed,
            )
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                                 learning_rate=0.06, random_state=cfg.seed)
        clf.fit(Xsucc[train_idx][mtr], ytr)
        dense[:, i] = clf.predict_proba(Xsucc[test_idx])[:, 1]
        proba[np.where(mte)[0], i] = dense[np.where(mte)[0], i]
        rows.append(agent)
    return proba, rows, dense


def _causal(df, Xcausal, A, cfg, model, tensors, test_idx, device, verbose, ctx) -> dict:
    """Cross-fitted AIPW on culture orders, plus the neural estimator and the checks."""
    out: dict = {"target_class": cfg.target_class, "control_class": cfg.control_class}

    sample = cohort_mod.causal_sample(df, "Ycoll")
    rows = sample.index.to_numpy()
    out["units_before_dedup"] = int((df["Ycoll"].notna()).sum())
    out["units"] = int(len(rows))
    out["patients"] = int(sample["anon_id"].nunique())

    Xc, a, y = Xcausal[rows], A[rows], sample["Ycoll"].to_numpy(dtype=float)
    groups = sample["anon_id"].to_numpy()
    out["exposure_rate"] = float(a.mean())
    out["outcome_rate"] = float(y.mean())

    if len(np.unique(a)) < 2 or len(np.unique(y)) < 2:
        out["error"] = "treatment or outcome is constant among valid units"
        return out

    cf = causal.crossfit_aipw(Xc, a, y, groups=groups, n_folds=cfg.n_folds,
                              clip=cfg.propensity_clip, seed=cfg.seed, label="crossfit_dml")
    out["crossfit"] = cf.as_dict()
    out["crossfit_naive_contrast"] = float(y[a == 1].mean() - y[a == 0].mean())

    acute = causal.crossfit_aipw(
        ctx["Xcausal_acute"][rows], a, y, groups=groups, n_folds=cfg.n_folds,
        clip=cfg.propensity_clip, seed=cfg.seed, label="with_acute_block",
    )
    out["adjustment_with_acute_block"] = acute.as_dict()

    if verbose:
        print("   fitting the fold-honest representation arm")
    representer = _representer_factory(rows, groups, df, ctx, cfg, device)
    rep_est = causal.crossfit_aipw(
        Xc, a, y, groups=groups, n_folds=cfg.n_folds, clip=cfg.propensity_clip,
        seed=cfg.seed, label="learned_representation", representer=representer,
    )
    out["representation"] = rep_est.as_dict()

    out["same_organism"] = _same_organism_estimate(df, Xcausal, A, cfg, ctx, verbose)

    out["recent_exposure_30d"] = _short_window_estimate(df, Xcausal, cfg, ctx, 30, verbose)

    out["class_effects"] = _class_effects(df, Xcausal, cfg, ctx, verbose)

    pi = _refit_propensity(Xc, a, groups, cfg)
    out["overlap"] = causal.overlap_summary(pi, a)
    w = np.where(a == 1, 1.0 / pi, 1.0 / (1.0 - pi))
    smd_before = causal.standardised_differences(Xc, a)
    smd_after = causal.standardised_differences(Xc, a, weights=w)
    names = ctx.get("causal_feature_names") or []
    if len(names) == Xc.shape[1]:
        worst = np.argsort(-np.abs(np.nan_to_num(smd_after)))[:5]
        out["worst_balanced_covariates"] = [
            {"covariate": names[j], "smd_weighted": float(smd_after[j]),
             "smd_unadjusted": float(smd_before[j])}
            for j in worst if np.isfinite(smd_after[j])
        ]
    out["balance"] = {
        "max_abs_smd_unadjusted": float(np.nanmax(np.abs(smd_before))),
        "max_abs_smd_weighted": float(np.nanmax(np.abs(smd_after))),
        "n_covariates_above_0.1_unadjusted": int(np.nansum(np.abs(smd_before) > 0.1)),
        "n_covariates_above_0.1_weighted": int(np.nansum(np.abs(smd_after) > 0.1)),
    }
    plots.love_plot(smd_before, smd_after, cfg.results_dir)
    plots.propensity_overlap(pi, a, cfg.results_dir, "crossfit_overlap.png",
                             "Out-of-fold propensity, patient-grouped folds")

    out["e_value_point"] = causal.e_value_point(cf.estimate, cf.baseline_risk)
    out["e_value_bound"] = causal.e_value_bound(cf.estimate, cf.lo, cf.hi, cf.baseline_risk)
    out["risk_ratio"] = causal.rr_from_rd(cf.estimate, cf.baseline_risk)

    neural = _neural_estimate(df, model, tensors, test_idx, A, cfg, device)
    out["neural_temporal"] = neural

    control_sample = cohort_mod.causal_sample(df, "Yctrl")
    if len(control_sample) > 50:
        rows_c = control_sample.index.to_numpy()
        yc = control_sample["Yctrl"].to_numpy(dtype=float)
        ac = A[rows_c]
        if len(np.unique(ac)) == 2 and len(np.unique(yc)) == 2:
            nc = causal.crossfit_aipw(
                Xcausal[rows_c], ac, yc, groups=control_sample["anon_id"].to_numpy(),
                n_folds=cfg.n_folds, clip=cfg.propensity_clip, seed=cfg.seed + 1,
                label="negative_control",
            )
            out["negative_control"] = nc.as_dict()

    out["sensitivity_missing_as_negative"] = _missing_as_negative(df, Xcausal, A, cfg)

    labels, est, lo, hi = [], [], [], []
    if "neural_temporal" in out and np.isfinite(out["neural_temporal"].get("estimate", np.nan)):
        labels.append("Neural temporal AIPW")
        est.append(out["neural_temporal"]["estimate"])
        lo.append(out["neural_temporal"]["lo"])
        hi.append(out["neural_temporal"]["hi"])
    labels.append("Cross-fitted DML")
    est.append(cf.estimate); lo.append(cf.lo); hi.append(cf.hi)
    if "negative_control" in out:
        labels.append(f"Negative control ({cfg.control_class})")
        est.append(out["negative_control"]["estimate"])
        lo.append(out["negative_control"]["lo"])
        hi.append(out["negative_control"]["hi"])
    plots.causal_forest(labels, est, lo, hi, cfg.results_dir)

    if verbose:
        print(f"   cross-fitted risk difference {cf.estimate:+.4f} "
              f"({cf.lo:+.4f} to {cf.hi:+.4f}), clustered on {cf.n_clusters:,} patients")
        print(f"   iid standard error would have been {cf.se_iid:.4f} versus {cf.se:.4f}")
    return out


def _decision(P_net, P_baseline, Ysucc, Msucc, test_idx, agents, causal_report, cfg,
              verbose) -> dict:
    """Evaluate the decision rule over the test period.

    Each test unit is scored for every eligible agent using the per-class collateral costs,
    and the trade-off weight is swept to trace the frontier: how much predicted immediate
    efficacy a given weight on stewardship gives up, and how much collateral resistance it
    avoids.
    """
    effects = (causal_report or {}).get("class_effects", {}).get("estimated", {})
    if not effects:
        return {"available": False, "reason": "no per-class effects were estimable"}

    usable = {cls: v for cls, v in effects.items() if v.get("positivity_ok", True)}
    dropped = sorted(set(effects) - set(usable))
    if not usable:
        return {"available": False, "reason": "no class passed the positivity guard"}
    class_rd = {cls: float(v["estimate"]) for cls, v in usable.items()}
    agent_class = {a: agent_to_class(a) for a in agents}
    cost, eligible = decision.agent_collateral_vector(agents, agent_class, class_rd)
    if not eligible.any():
        return {"available": False, "reason": "no evaluated agent maps to an estimated class"}

    primary = P_baseline if P_baseline is not None else P_net
    curve = decision.frontier(primary, agents, agent_class, class_rd)
    curve_net = decision.frontier(P_net, agents, agent_class, class_rd)
    check = decision.observed_failure_check(
        primary, Ysucc[test_idx], Msucc[test_idx], agents, agent_class, class_rd,
        lam=cfg.lambda_decision,
    )

    at_default = curve.loc[np.isclose(curve["lambda"], cfg.lambda_decision)]
    out = {
        "available": True,
        "lambda_default": cfg.lambda_decision,
        "probabilities_from": "per-agent baseline" if P_baseline is not None else "shared network",
        "eligible_agents": int(eligible.sum()),
        "classes_with_effects": sorted(class_rd),
        "classes_dropped_for_positivity": dropped,
        "frontier": curve.to_dict(orient="records"),
        "frontier_shared_net": curve_net.to_dict(orient="records"),
        "observed_check": check,
    }
    if len(at_default):
        out["at_default_lambda"] = at_default.iloc[0].to_dict()
    if verbose:
        pure = curve.loc[np.isclose(curve["lambda"], 1.0)]
        if len(pure) and len(at_default):
            p0, p1 = pure.iloc[0], at_default.iloc[0]
            print(f"   lambda=1.00 (susceptibility only): failure {p0['mean_p_failure']:.4f}, "
                  f"collateral {p0['mean_collateral']:.4f}")
            print(f"   lambda={cfg.lambda_decision:.2f} (fused):            "
                  f"failure {p1['mean_p_failure']:.4f}, collateral {p1['mean_collateral']:.4f}, "
                  f"switched {p1['switched_from_pure_susceptibility']:.1%}")
    return out


def _representer_factory(rows, groups, df, ctx, cfg, device):
    """Build a fold-honest representer for ``causal.crossfit_aipw``.

    For each fold the encoder is trained from scratch on every culture belonging to a
    patient who is *not* in the held-out fold, then used to embed both sides. Training on
    all such cultures rather than only the causal sample is deliberate: the susceptibility
    signal is most of what shapes the representation, and discarding it would test a
    different architecture from the one the paper describes.

    The cost of honesty here is five encoder fits instead of one. The encoder trains in
    seconds, so this is affordable; the alternative -- embedding once with a model that
    has seen every fold -- would make the cross-fitted interval meaningless.
    """
    anon = df["anon_id"].to_numpy()
    Xbase, Xsev_pre = ctx["Xbase"], ctx["Xsev_pre"]
    tensors, n_agents = ctx["tensors"], ctx["n_agents"]

    def representer(tr, te):
        held_out_patients = np.unique(groups[te])
        eligible = np.flatnonzero(~np.isin(anon, held_out_patients))
        rng = np.random.default_rng(cfg.seed + len(te))
        perm = rng.permutation(len(eligible))
        n_val = max(1, int(cfg.val_fraction * len(eligible)))
        val_rows, fit_rows = eligible[perm[:n_val]], eligible[perm[n_val:]]

        net = SharedNet(Xbase.shape[1], tensors["expo"].shape[1], tensors["sev"].shape[1],
                        n_agents, cfg.encoder_dims, cfg.rep_dim, cfg.dropout).to(device)
        training.train(net, tensors, fit_rows, val_rows, cfg, device, verbose=False)
        net.eval()
        with torch.no_grad():
            z = net.represent(tensors["base"]).detach().cpu().numpy()

        def block(idx):
            r = rows[idx]
            return np.concatenate([z[r], Xsev_pre[r]], axis=1)

        return block(tr), block(te)

    return representer


def _same_organism_estimate(df, Xcausal, A, cfg, ctx, verbose) -> dict:
    """Re-estimate restricting the outcome to next cultures sharing an organism."""
    from armd.labels import collateral_outcome_map

    coh = ctx["coh"]
    y_map = collateral_outcome_map(coh.definitive, cfg.target_class,
                                   cfg.followup_horizon_days, cfg.missing_as_negative,
                                   require_shared_organism=True)
    if not len(y_map):
        return {"available": False, "reason": "no same-organism successor pairs"}
    ycol = df["order_proc_id_coded"].map(y_map)
    valid = df[ycol.notna()].drop_duplicates("order_proc_id_coded")
    if len(valid) < MIN_CLASS_UNITS:
        return {"available": False, "reason": f"only {len(valid)} units"}
    rows = valid.index.to_numpy()
    a, y = A[rows], ycol.to_numpy(dtype=float)[rows]
    if len(np.unique(a)) < 2 or len(np.unique(y)) < 2:
        return {"available": False, "reason": "treatment or outcome constant"}
    est = causal.crossfit_aipw(
        Xcausal[rows], a, y, groups=valid["anon_id"].to_numpy(), n_folds=cfg.n_folds,
        clip=cfg.propensity_clip, seed=cfg.seed, label="same_organism",
    )
    out = est.as_dict()
    out["available"] = True
    out["retained_fraction_of_units"] = float(len(valid) / max(int(df["Ycoll"].notna().sum()), 1))
    if verbose:
        print(f"   same-organism outcome: {est.estimate:+.4f} "
              f"({est.lo:+.4f} to {est.hi:+.4f}) on {est.n:,.0f} orders")
    return out


def _short_window_estimate(df, Xcausal, cfg, ctx, window_days, verbose) -> dict:
    """Re-estimate with a tighter look-back, since selection pressure decays with time."""
    from armd.labels import exposed_orders

    coh = ctx["coh"]
    ew = coh.exposure_window
    if ew is None or not len(ew) or "time_to_culturetime" not in ew.columns:
        return {"available": False, "reason": "no exposure timing available"}
    t = pd.to_numeric(ew["time_to_culturetime"], errors="coerce")
    recent = ew[(t >= 0) & (t <= window_days)]
    a_all = df["order_proc_id_coded"].isin(
        exposed_orders(recent, cfg.target_class)).to_numpy(dtype=float)

    valid = df[df["Ycoll"].notna()].drop_duplicates("order_proc_id_coded")
    rows = valid.index.to_numpy()
    a, y = a_all[rows], valid["Ycoll"].to_numpy(dtype=float)
    if len(np.unique(a)) < 2 or int(a.sum()) < MIN_CLASS_EXPOSED:
        return {"available": False, "reason": f"only {int(a.sum())} exposed within {window_days}d"}
    est = causal.crossfit_aipw(
        Xcausal[rows], a, y, groups=valid["anon_id"].to_numpy(), n_folds=cfg.n_folds,
        clip=cfg.propensity_clip, seed=cfg.seed, label=f"exposure_{window_days}d",
    )
    out = est.as_dict()
    out["available"] = True
    out["window_days"] = window_days
    out["n_exposed"] = int(a.sum())
    if verbose:
        print(f"   {window_days}-day exposure window: {est.estimate:+.4f} "
              f"({est.lo:+.4f} to {est.hi:+.4f}), {int(a.sum()):,} exposed")
    return out


def simultaneous_interval(estimate: float, se: float, n_comparisons: int,
                          alpha: float = 0.05) -> tuple:
    """Bonferroni-adjusted interval, for reading a panel of estimates as a ranking.

    Nine class estimates presented together are nine opportunities to find a non-null, and
    pointwise intervals do not price that in. Bonferroni is conservative and crude, which
    is the right trade here: the panel is reported for interpretation rather than to
    support a specific rejection, and a reader comparing ranks deserves the pessimistic
    band rather than a sharper one resting on an assumed correlation structure the
    overlapping patient sets would violate anyway.
    """
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / (2.0 * max(n_comparisons, 1)))
    return float(estimate - z * se), float(estimate + z * se), float(z)


MIN_CLASS_UNITS = 400
MIN_CLASS_EXPOSED = 100

MAX_CLIPPED_FRACTION = 0.25


def _class_effects(df, Xcausal, cfg, ctx, verbose) -> dict:
    """Collateral-resistance effect for every drug class with enough support.

    The decision rule needs a cost for each candidate class, not only for the one class
    the paper analyses in depth. Classes that cannot support an estimate are reported as
    skipped, with the reason, rather than silently omitted -- an absent estimate is not a
    zero estimate, and the decision layer has to be able to tell the difference.
    """
    from armd.labels import collateral_outcome_map, exposed_orders

    coh = ctx["coh"]
    expo_window, definitive = coh.exposure_window, coh.definitive
    if expo_window is None or not len(expo_window):
        return {"estimated": {}, "skipped": {"all": "no exposure rows"}}

    candidates = sorted(set(expo_window["drug_class"].dropna().unique())
                        & set(definitive["drug_class"].dropna().unique()))
    estimated: dict = {}
    skipped: dict = {}

    for cls in candidates:
        y_map = collateral_outcome_map(definitive, cls, cfg.followup_horizon_days,
                                       cfg.missing_as_negative)
        if not len(y_map):
            skipped[cls] = "no next-culture outcome for this class"
            continue
        ycol = df["order_proc_id_coded"].map(y_map)
        valid = df[ycol.notna()].drop_duplicates("order_proc_id_coded")
        if len(valid) < MIN_CLASS_UNITS:
            skipped[cls] = f"only {len(valid)} units (floor {MIN_CLASS_UNITS})"
            continue
        rows_c = valid.index.to_numpy()
        a_c = df["order_proc_id_coded"].isin(exposed_orders(expo_window, cls)) \
                .to_numpy(dtype=float)[rows_c]
        y_c = ycol.to_numpy(dtype=float)[rows_c]
        n_exposed = int(a_c.sum())
        if n_exposed < MIN_CLASS_EXPOSED or n_exposed == len(a_c):
            skipped[cls] = f"only {n_exposed} exposed (floor {MIN_CLASS_EXPOSED})"
            continue
        if len(np.unique(y_c)) < 2:
            skipped[cls] = "outcome is constant"
            continue
        est = causal.crossfit_aipw(
            Xcausal[rows_c], a_c, y_c, groups=valid["anon_id"].to_numpy(),
            n_folds=cfg.n_folds, clip=cfg.propensity_clip, seed=cfg.seed, label=cls,
        )
        row = est.as_dict()
        row["n_exposed"] = n_exposed
        row["positivity_ok"] = bool(est.clipped_fraction <= MAX_CLIPPED_FRACTION)
        row["risk_ratio"] = (float(causal.rr_from_rd(est.estimate, est.baseline_risk))
                             if est.baseline_risk else float("nan"))
        if np.isfinite(row["risk_ratio"]) and row["risk_ratio"] > 0:
            row["e_value_point"] = float(causal.e_value_rr(row["risk_ratio"]))
        estimated[cls] = row
        if verbose:
            flag = "" if row["positivity_ok"] else "  [POSITIVITY]"
            print(f"   {cls:<34} rd {est.estimate:+.4f} ({est.lo:+.4f} to {est.hi:+.4f})"
                  f"  n={est.n:,.0f} exposed={n_exposed:,} base={est.baseline_risk:.3f}"
                  f" clip={est.clipped_fraction:.1%}{flag}")

    k = max(sum(1 for v in estimated.values() if v["positivity_ok"]), 1)
    for cls, row in estimated.items():
        lo, hi, z = simultaneous_interval(row["estimate"], row["se"], k)
        row["lo_simultaneous"], row["hi_simultaneous"] = lo, hi
        row["simultaneous_z"] = z
        row["n_comparisons"] = k

    return {"estimated": estimated, "skipped": skipped}


def _refit_propensity(Xc, a, groups, cfg) -> np.ndarray:
    """Out-of-fold propensities for the diagnostics, on the same folds as the estimator."""
    pi = causal.crossfit_propensity(Xc, a, groups, cfg.n_folds, cfg.seed)
    return np.clip(pi, cfg.propensity_clip, 1 - cfg.propensity_clip)


def _neural_estimate(df, model, tensors, test_idx, A, cfg, device) -> dict:
    """AIPW from the network's own heads on the test period.

    The nuisances were fitted on earlier years, so this is a sample split rather than
    cross-fitting. It is reported as a secondary estimate; the interval comes from a
    patient-level bootstrap and still does not account for nuisance estimation.
    """
    test_rows = df.index.to_numpy()[test_idx]
    sub = df.loc[test_rows]
    valid = sub["Ycoll"].notna()
    sub = sub[valid].drop_duplicates("order_proc_id_coded")
    if len(sub) < 50:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(sub))}

    positions = df.index.get_indexer(sub.index)
    preds = training.predict(model, tensors, positions, device)
    pi = np.clip(preds["propensity"], cfg.propensity_clip, 1 - cfg.propensity_clip)
    a = A[positions]
    y = sub["Ycoll"].to_numpy(dtype=float)
    if len(np.unique(a)) < 2:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(sub))}
    point, lo, hi = causal.bootstrap_ci(
        a, y, pi, preds["m0"], preds["m1"], groups=sub["anon_id"].to_numpy(),
        n_boot=cfg.n_bootstrap, seed=cfg.seed,
    )
    return {"estimate": point, "lo": lo, "hi": hi, "n": int(len(sub)),
            "patients": int(sub["anon_id"].nunique())}


def _missing_as_negative(df, Xcausal, A, cfg) -> dict:
    """Re-estimate coding an absent next culture as no collateral-resistance event.

    The primary analysis keeps only index cultures followed by another positive culture
    inside the horizon, which conditions on being cultured again. This variant bounds
    how much that choice matters instead of leaving it as a paragraph in the limitations.
    """
    from armd.labels import collateral_outcome_map

    sample = df.drop_duplicates("order_proc_id_coded").copy()
    rows = sample.index.to_numpy()
    y = sample["Ycoll"].fillna(0.0).to_numpy(dtype=float)
    a = A[rows]
    if len(np.unique(a)) < 2 or len(np.unique(y)) < 2:
        return {"estimate": float("nan")}
    est = causal.crossfit_aipw(
        Xcausal[rows], a, y, groups=sample["anon_id"].to_numpy(), n_folds=cfg.n_folds,
        clip=cfg.propensity_clip, seed=cfg.seed, label="missing_as_negative",
    )
    return est.as_dict()


def predicted_antibiogram(model, tensors, row: int, agents, device) -> pd.DataFrame:
    preds = training.predict(model, tensors, np.array([row]), device)
    p = preds["susceptibility"][0]
    out = pd.DataFrame({"antibiotic": agents, "predicted_susceptible": (1 - p).round(3)})
    out["drug_class"] = [agent_to_class(a) for a in agents]
    return out.sort_values("predicted_susceptible", ascending=False).reset_index(drop=True)


def _write_summary(report: dict, cfg: Config) -> None:
    c = report.get("causal", {})
    cf = c.get("crossfit", {})
    nc = c.get("negative_control", {})
    nt = c.get("neural_temporal", {})
    b = report.get("baseline", {})
    lines = [
        "# ARMD joint model — run summary", "",
        f"Device: {report.get('device')}   |   agents evaluable: {report.get('agents_evaluable')}",
        f"Culture–organism units: {report.get('cohort', {}).get('culture_organism_units'):,}"
        f"   |   culture orders: {report.get('cohort', {}).get('culture_orders'):,}"
        f"   |   patients: {report.get('cohort', {}).get('patients_with_units'):,}", "",
        "## Susceptibility", "",
        f"- Shared network, support-weighted test AUC: {report.get('net_support_weighted_auc'):.3f}",
        f"- Median per-agent test AUC: {report.get('net_median_auc'):.3f}",
        f"- Network vs baseline over {b.get('agents_compared')} agents: "
        f"{b.get('net_mean_auc', float('nan')):.3f} vs {b.get('xgb_mean_auc', float('nan')):.3f} mean AUC",
        f"- Network ahead on {b.get('net_better_count')} agents, "
        f"of which {b.get('net_better_significant')} with an interval excluding zero; "
        f"baseline significantly ahead on {b.get('xgb_better_significant')}", "",
        f"## Collateral resistance — {cfg.target_class}", "",
        f"- Units: {c.get('units'):,} culture orders from {c.get('patients'):,} patients "
        f"(before de-duplication: {c.get('units_before_dedup'):,})",
        f"- Unadjusted contrast: {c.get('crossfit_naive_contrast', float('nan')):+.4f}",
        f"- Cross-fitted AIPW: {cf.get('estimate', float('nan')):+.4f} "
        f"({cf.get('lo', float('nan')):+.4f} to {cf.get('hi', float('nan')):+.4f}), "
        f"clustered SE {cf.get('se', float('nan')):.4f} vs iid {cf.get('se_iid', float('nan')):.4f}",
        f"- Baseline risk among unexposed: {cf.get('baseline_risk', float('nan')):.3f}; "
        f"implied risk ratio {c.get('risk_ratio', float('nan')):.2f}",
        f"- E-value: {c.get('e_value_point', float('nan')):.2f} (point), "
        f"{c.get('e_value_bound', float('nan')):.2f} (confidence limit)",
        f"- Propensity clipped for {cf.get('clipped_fraction', float('nan')):.1%} of units; "
        f"common support covers {c.get('overlap', {}).get('common_support_fraction', float('nan')):.1%}",
    ]
    if nt:
        lines.append(f"- Neural temporal AIPW: {nt.get('estimate', float('nan')):+.4f} "
                     f"({nt.get('lo', float('nan')):+.4f} to {nt.get('hi', float('nan')):+.4f}) "
                     f"on {nt.get('n', 0):,} test-period orders")
    if nc:
        lines.append(f"- Negative control ({cfg.control_class}): {nc.get('estimate', float('nan')):+.4f} "
                     f"({nc.get('lo', float('nan')):+.4f} to {nc.get('hi', float('nan')):+.4f})")
    sens = c.get("sensitivity_missing_as_negative", {})
    if sens and np.isfinite(sens.get("estimate", np.nan)):
        lines.append(f"- Sensitivity, absent next culture coded as no event: "
                     f"{sens['estimate']:+.4f} ({sens['lo']:+.4f} to {sens['hi']:+.4f})")

    acute = c.get("adjustment_with_acute_block", {})
    repn = c.get("representation", {})
    if acute or repn:
        lines += ["", "## Adjustment-set specifications", "",
                  "The primary set omits the acute laboratory and vital block, which the "
                  "release measures in the 14 days before the index culture and therefore "
                  "after the exposure window."]
        if acute:
            lines.append(f"- Adding the acute block back: {acute.get('estimate', float('nan')):+.4f} "
                         f"({acute.get('lo', float('nan')):+.4f} to {acute.get('hi', float('nan')):+.4f})")
        if repn:
            lines.append(f"- Shared representation as the adjustment set (fold-honest): "
                         f"{repn.get('estimate', float('nan')):+.4f} "
                         f"({repn.get('lo', float('nan')):+.4f} to {repn.get('hi', float('nan')):+.4f}), "
                         f"clipped {repn.get('clipped_fraction', float('nan')):.1%}")

    so = c.get("same_organism", {})
    sw = c.get("recent_exposure_30d", {})
    if so.get("available") or sw.get("available"):
        lines += ["", "## Mechanism and exposure-recency sensitivities", ""]
        if so.get("available"):
            lines.append(f"- Same-organism outcome (acquired pathway only): "
                         f"{so['estimate']:+.4f} ({so['lo']:+.4f} to {so['hi']:+.4f}) on "
                         f"{so['n']:,.0f} orders, "
                         f"{so.get('retained_fraction_of_units', float('nan')):.1%} of the "
                         f"default sample retained")
        if sw.get("available"):
            lines.append(f"- Exposure within {sw['window_days']} days only: "
                         f"{sw['estimate']:+.4f} ({sw['lo']:+.4f} to {sw['hi']:+.4f}), "
                         f"{sw['n_exposed']:,} exposed")

    wb = c.get("worst_balanced_covariates", [])
    if wb:
        lines += ["", "Least-balanced covariates after weighting: " +
                  "; ".join(f"{x['covariate']} ({x['smd_weighted']:+.3f})" for x in wb)]

    ce = c.get("class_effects", {}).get("estimated", {})
    if ce:
        z = next(iter(ce.values())).get("simultaneous_z")
        k = next(iter(ce.values())).get("n_comparisons", len(ce))
        n_flagged = sum(1 for v in ce.values() if not v.get("positivity_ok", True))
        lines += ["", "## Collateral resistance by drug class", "",
                  f"{len(ce)} classes estimated, {n_flagged} of them flagged for a "
                  f"positivity violation and excluded from the decision layer. Pointwise "
                  f"95% intervals, and Bonferroni-simultaneous bands over the {k} "
                  f"interpretable classes (z={z:.2f}).", ""]
        for cls, v in sorted(ce.items(), key=lambda kv: -kv[1]["estimate"]):
            flag = "" if v.get("positivity_ok", True) else "  **positivity violated**"
            lines.append(f"- {cls}: {v['estimate']:+.4f} ({v['lo']:+.4f} to {v['hi']:+.4f}; "
                         f"simultaneous {v.get('lo_simultaneous', float('nan')):+.4f} to "
                         f"{v.get('hi_simultaneous', float('nan')):+.4f}), "
                         f"n={v['n']:,.0f} over {v['n_clusters']:,.0f} patients, "
                         f"{v.get('n_exposed', 0):,} exposed, base risk "
                         f"{v.get('baseline_risk', float('nan')):.3f}, "
                         f"clipped {v.get('clipped_fraction', float('nan')):.1%}{flag}")
        skipped = c.get("class_effects", {}).get("skipped", {})
        if skipped:
            lines.append(f"- Not estimable: " +
                         "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))

    d = report.get("decision", {})
    if d.get("available"):
        lines += ["", "## Decision layer", "",
                  f"Eligible agents: {d.get('eligible_agents')} across "
                  f"{len(d.get('classes_with_effects', []))} classes with an estimate. "
                  f"Probabilities from the {d.get('probabilities_from', 'n/a')}.", "",
                  "| lambda | mean P(failure) | mean collateral | switched vs pure |",
                  "|---:|---:|---:|---:|"]
        for row in d.get("frontier", []):
            lines.append(f"| {row['lambda']:.2f} | {row['mean_p_failure']:.4f} | "
                         f"{row['mean_collateral']:.4f} | "
                         f"{row['switched_from_pure_susceptibility']:.1%} |")
        chk = d.get("observed_check", {})
        if chk.get("n_scored"):
            lines.append("")
            lines.append(f"Observed resistance rate of the recommendation at "
                         f"lambda={d.get('lambda_default')}: "
                         f"{chk['observed_resistance_rate_of_recommendation']:.3f} "
                         f"over {chk['n_scored']:,} scorable units "
                         f"({chk['fraction_of_units_scorable']:.1%} of the test period). "
                         f"Verification bias applies: only agents a clinician chose to test "
                         f"can be scored.")

    with open(os.path.join(cfg.results_dir, "summary_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
