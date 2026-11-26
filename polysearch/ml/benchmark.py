"""Nested group CV on the checksum-pinned provisional Monash MMA table."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .estimators import fit_estimator, predict, predict_gp, group_weights
from ..data.monash_source import DOI, digest

FEATURES = ["temperature_c", "ct_cm", "residence_min"]
TARGET = "conversion_fraction"
MODELS = ("ridge", "catboost", "gp", "kinetic")
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "catboost": [{"depth": d, "iterations": n, "learning_rate": 0.05,
                  "l2_leaf_reg": 5} for d in (3, 5) for n in (200, 500)],
    "gp": [{}],
    "kinetic": [{"regularization": r} for r in (0.0, 0.001, 0.01, 0.1)],
}
SENSITIVITY = ("MMA-13", "MMA-16")


def load_table(table, quality):
    report = json.loads(Path(quality).read_text(encoding="utf-8"))
    if report.get("doi") != DOI or report.get("monomer") != "MMA":
        raise ValueError("Unexpected source DOI or monomer")
    if digest(table) != report["table_sha256"]:
        raise ValueError("Prepared table checksum mismatch")
    with Path(table).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != report["counts"]["accepted"] or not rows:
        raise ValueError("Row count does not match quality report")
    X = np.asarray([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.asarray([float(r[TARGET]) for r in rows])
    groups = np.asarray([r["split_group"] for r in rows])
    experiments = np.asarray([r["experiment_id"] for r in rows])
    folds = np.asarray([int(r["fold"]) for r in rows])
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite input")
    if np.any((y < 0) | (y > 1)) or np.any(X[:, 2] <= 0) or np.any(X[:, 1] < 0):
        raise ValueError("Invalid measurements")
    if any(r["monomer"] != "MMA" or not r["experiment_id"] or not r["split_group"] for r in rows):
        raise ValueError("Unexpected row metadata")
    for labels in (groups, experiments):
        for label in np.unique(labels):
            if len(np.unique(folds[labels == label])) != 1:
                raise ValueError("Group leakage between folds")
    for label in np.unique(experiments):
        if len(np.unique(groups[experiments == label])) != 1:
            raise ValueError("Experiment split across condition groups")
    conditions = defaultdict(set)
    for x, group in zip(X, groups):
        conditions[tuple(x[:2])].add(group)
    if any(len(g) != 1 for g in conditions.values()):
        raise ValueError("Identical conditions assigned to different groups")
    if set(folds) != set(range(report["folds"])) or len(np.unique(groups)) != report["condition_groups"]:
        raise ValueError("Unexpected fold or group count")
    return rows, X, y, groups, experiments, folds, report


def macro_mae(y, pred, groups):
    return float(np.mean([np.mean(np.abs(y[groups == g] - pred[groups == g]))
                          for g in np.unique(groups)]))


def metrics(y, pred, groups):
    err = y - pred
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 and np.ptp(y) > 0 else None,
            "group_macro_mae": macro_mae(y, pred, groups),
            "group_macro_rmse": float(np.sqrt(np.mean([
                np.mean(err[groups == g] ** 2) for g in np.unique(groups)])))}


def tune(name, X, y, groups, seed):
    grid = GRIDS[name]
    if len(grid) == 1:
        return grid[0], []
    cv = GroupKFold(n_splits=3)
    results = []
    for params in grid:
        pred = np.full(len(y), np.nan)
        for tr, va in cv.split(X, y, groups):
            assert not set(groups[tr]) & set(groups[va])
            model = fit_estimator(name, params, X[tr], y[tr], groups[tr], seed)
            pred[va] = predict(model, X[va])
        results.append({"params": params, "inner_group_macro_mae": macro_mae(y, pred, groups)})
    return min(results, key=lambda r: r["inner_group_macro_mae"])["params"], results


def conformal_group_quantile(scores, groups, coverage):
    """Order statistic of one maximum standardized error per independent group."""
    maxima = np.sort([np.max(scores[groups == g]) for g in np.unique(groups)])
    rank = math.ceil((len(maxima) + 1) * coverage - 1e-12)
    if rank > len(maxima):
        return None
    return float(maxima[rank - 1])


def interval_metrics(y, lower, upper, groups):
    hit = (y >= lower) & (y <= upper)
    return {"row_coverage": float(hit.mean()),
            "group_macro_row_coverage": float(np.mean([hit[groups == g].mean() for g in np.unique(groups)])),
            "whole_group_coverage": float(np.mean([hit[groups == g].all() for g in np.unique(groups)])),
            "mean_width": float(np.mean(upper - lower)),
            "groups": int(len(np.unique(groups)))}


def limits(mean, std, q):
    return np.clip(mean - q * std, 0, 1), np.clip(mean + q * std, 0, 1)


def run_cv(X, y, groups, experiments, folds, seed, intervals=True):
    predictions = {name: np.full(len(y), np.nan) for name in MODELS}
    gp_lo, gp_hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    conf_lo, conf_hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_reports = []
    all_warnings = []
    for fold in sorted(set(folds)):
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(groups[tr]) & set(groups[te])
        fold_seed = seed + int(fold)
        fold_report = {"fold": int(fold), "train_groups": sorted(set(groups[tr])),
                       "test_groups": sorted(set(groups[te])),
                       "test_experiments": sorted(set(experiments[te])), "models": {}}
        for name in MODELS:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                params, inner = tune(name, X[tr], y[tr], groups[tr], fold_seed)
                model = fit_estimator(name, params, X[tr], y[tr], groups[tr], fold_seed)
                pred = predict(model, X[te])
                predictions[name][te] = pred
                if name == "gp" and intervals:
                    mu, sd = predict_gp(model, X[te])
                    gp_lo[te], gp_hi[te] = limits(mu, sd, norm.ppf(0.95))
            all_warnings.extend({"fold": int(fold), "model": name, "message": str(w.message)}
                                for w in caught)
            fold_report["models"][name] = {"metrics": metrics(y[te], pred, groups[te]),
                                            "params": params, "inner_search": inner,
                                            "seconds": round(time.perf_counter() - start, 3)}
        if intervals:
            # Random split of the OUTER training groups, with four calibration groups.
            # With n=4, 80% is the largest finite rank; 90% cannot be estimated.
            unique = np.unique(groups[tr])
            shuffled = np.random.default_rng(fold_seed).permutation(unique)
            cal_groups = shuffled[:4]
            ca = tr[np.isin(groups[tr], cal_groups)]
            fit = tr[~np.isin(groups[tr], cal_groups)]
            if len(np.unique(groups[fit])) < 3:
                raise ValueError("Insufficient fit groups for group conformal")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gp = fit_estimator("gp", {}, X[fit], y[fit], groups[fit], fold_seed)
                cm, cs = predict_gp(gp, X[ca])
                q = conformal_group_quantile(np.abs(y[ca] - cm) / cs, groups[ca], 0.8)
                if q is None:
                    raise ValueError("Insufficient calibration groups")
                mu, sd = predict_gp(gp, X[te])
                conf_lo[te], conf_hi[te] = limits(mu, sd, q)
            all_warnings.extend({"fold": int(fold), "model": "gp_group_conformal",
                                 "message": str(w.message)} for w in caught)
            fold_report["calibration"] = {"fit_groups": sorted(set(groups[fit])),
                                          "calibration_groups": sorted(set(groups[ca])),
                                          "q80": q, "q90": None,
                                          "nominal_gp90": interval_metrics(y[te], gp_lo[te], gp_hi[te], groups[te]),
                                          "group_conformal80": interval_metrics(y[te], conf_lo[te], conf_hi[te], groups[te])}
        fold_reports.append(fold_report)
    if any(not np.all(np.isfinite(p)) for p in predictions.values()):
        raise ValueError("Incomplete OOF predictions")
    report = {"rows": len(y), "condition_groups": len(np.unique(groups)),
              "experiments": len(np.unique(experiments)),
              "models": {name: metrics(y, p, groups) for name, p in predictions.items()},
              "fold_reports": fold_reports,
              "warnings": all_warnings}
    if intervals:
        report["intervals"] = {
            "gp_nominal_90": interval_metrics(y, gp_lo, gp_hi, groups),
            "gp_group_split_conformal_80": interval_metrics(y, conf_lo, conf_hi, groups),
            "group_conformal_90": None,
            "note": "80% group split conformal uses a separate GP with 4 calibration groups per fold. "
                    "Finite-sample guarantee assumes exchangeable independent groups; "
                    "these small fixed experimental conditions are not established exchangeable. "
                    "Observed coverage is empirical; no adaptive-candidate coverage claim."}
    return report, predictions, (gp_lo, gp_hi, conf_lo, conf_hi)


def write_predictions(path, rows, predictions, intervals):
    names = [*MODELS, "gp90_lower", "gp90_upper", "conformal80_lower", "conformal80_upper"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0].keys(), *names])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({**row, **{k: float(v[i]) for k, v in predictions.items()},
                             **{k: float(v[i]) for k, v in zip(names[4:], intervals)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/processed/monash-v1/mma_conversion.csv"))
    parser.add_argument("--quality", type=Path, default=Path("data/processed/monash-v1/quality.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/monash-ml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, X, y, groups, experiments, folds, quality = load_table(args.table, args.quality)
    with threadpool_limits(limits=1):
        main_report, predictions, intervals = run_cv(X, y, groups, experiments, folds, args.seed)
        keep = ~np.isin(experiments, SENSITIVITY)
        sensitivity, _, _ = run_cv(X[keep], y[keep], groups[keep], experiments[keep],
                                  folds[keep], args.seed, intervals=False)
    report = {"doi": DOI, "table_sha256": digest(args.table),
              "quality_status": quality["quality_status"], "features": FEATURES, "target": TARGET,
              "seed": args.seed, "selection_metric": "group_macro_mae",
              "protocol": "fixed five outer condition-group folds; 3 group inner folds for hyperparameters",
              "environment": {"python": platform.python_version(),
                              **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "catboost", "threadpoolctl")}},
              "primary": main_report,
              "sensitivity": {"excluded_experiments": list(SENSITIVITY), **sensitivity},
              "limitation": "Provisional MMA measurements, correlated within reactions. Not an independent final test, "
                            "not a synthesis-count benchmark, not evidence of transfer to industrial chemistry."}
    report["lowest_oof_macro_mae"] = min(MODELS, key=lambda n: main_report["models"][n]["group_macro_mae"])
    write_predictions(args.out / "oof_predictions.csv", rows, predictions, intervals)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # Compact machine-readable line for CI verification and review.
    print("ML_SUMMARY " + json.dumps({"rows": main_report["rows"], "condition_groups": main_report["condition_groups"],
        "models": main_report["models"], "intervals": main_report["intervals"],
        "sensitivity": {"rows": sensitivity["rows"], "condition_groups": sensitivity["condition_groups"],
                        "models": sensitivity["models"]},
        "lowest_oof_macro_mae": report["lowest_oof_macro_mae"], "table_sha256": report["table_sha256"],
        "environment": report["environment"], "warnings": len(main_report["warnings"])}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
