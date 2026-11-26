"""Archive replay with one measured MMA reaction per query.

The evaluator owns the endpoint labels. Policies see only pool conditions and
outcomes revealed after their selections. This is a retrospective finite pool,
not an experiment or a prospective proof of savings.
"""
import argparse
import csv
import json
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .policies import POLICIES, choose, SIGNAL_SD, NOISE_SD, LENGTH_SCALE, RIDGE_ALPHA
from ..data.monash_source import DOI, digest
from ..ml.benchmark import load_table

ENDPOINT_MIN = 6.0
WINDOW_MIN = 0.25
TARGET = 0.15
NEAR_OPTIMUM_MARGIN = 0.01
INITIAL = 3
BUDGET = 10
SEEDS = tuple(range(64))
FLAGGED = ("MMA-13", "MMA-16")


def endpoint_pool(rows, endpoint=ENDPOINT_MIN, window=WINDOW_MIN, exclude=()):
    """Median of actual in-window readings, not an interpolation or surrogate."""
    if not np.isfinite(endpoint) or endpoint <= 0 or not np.isfinite(window) or window < 0:
        raise ValueError("Invalid endpoint")
    by_reaction = defaultdict(list)
    for row in rows:
        by_reaction[row["experiment_id"]].append(row)
    items, omitted = [], []
    groups = set()
    for reaction, values in sorted(by_reaction.items()):
        if reaction in exclude:
            omitted.append({"experiment_id": reaction, "reason": "sensitivity_exclusion"})
            continue
        conditions = {(r["split_group"], float(r["temperature_c"]), float(r["ct_cm"]))
                      for r in values}
        if len(conditions) != 1:
            raise ValueError("Inconsistent reaction conditions")
        group, temp, ratio = next(iter(conditions))
        if group in groups:
            raise ValueError("Multiple reactions for one condition: define replicate budget first")
        groups.add(group)
        selected = [r for r in values if abs(float(r["residence_min"]) - endpoint) <= window + 1e-10]
        if not selected:
            omitted.append({"experiment_id": reaction, "reason": "no_measured_endpoint_window"})
            continue
        outcomes = np.array([float(r["conversion_fraction"]) for r in selected])
        if not np.isfinite(outcomes).all() or np.any((outcomes < 0) | (outcomes > 1)):
            raise ValueError("Invalid endpoint conversion")
        items.append({"experiment_id": reaction, "split_group": group,
                      "temperature_c": temp, "ct_cm": ratio,
                      "conversion_fraction": float(np.median(outcomes)),
                      "endpoint_measurements": len(selected),
                      "residence_min_values": [float(r["residence_min"]) for r in selected],
                      "source_rows": [{"file": r["source_file"], "row": int(r["source_row"]),
                                       "sha256": r["source_sha256"]} for r in selected]})
    return items, omitted


class Archive:
    def __init__(self, labels):
        self._labels = np.array(labels, dtype=float, copy=True)
        if self._labels.ndim != 1 or not len(self._labels) or not np.isfinite(self._labels).all():
            raise ValueError("Invalid archive labels")
        self._observed = {}

    def reveal(self, index):
        index = int(index)
        if index < 0 or index >= len(self._labels) or index in self._observed:
            raise ValueError("Invalid or repeated archive query")
        self._observed[index] = float(self._labels[index])
        return self._observed[index]

    def history(self):
        return list(self._observed.keys()), list(self._observed.values())


def replay_once(X, y, initial, policy, seed, budget, chooser=choose):
    X = np.array(X, dtype=float, copy=True)
    X.setflags(write=False)
    archive = Archive(y)
    if not len(initial) <= budget <= len(X):
        raise ValueError("Budget must include initial design and fit pool")
    rng = np.random.default_rng(seed)
    for index in initial:
        archive.reveal(index)
    while len(archive.history()[0]) < budget:
        indices, values = archive.history()
        index = chooser(policy, X, tuple(indices), tuple(values), rng)
        archive.reveal(index)
    indices, values = archive.history()
    return indices, values


def first_hit(values, threshold):
    hits = np.flatnonzero(np.maximum.accumulate(values) >= threshold - 1e-12)
    return int(hits[0] + 1) if len(hits) else None


def summarize(values, oracle, target=TARGET):
    best = np.maximum.accumulate(np.asarray(values, dtype=float), axis=1)
    regret = np.maximum(oracle - best, 0)
    budget = best.shape[1]
    target_hits = [first_hit(v, target) for v in values]
    near_hits = [first_hit(v, oracle - NEAR_OPTIMUM_MARGIN) for v in values]
    def hits_summary(hits):
        success = [h for h in hits if h is not None]
        return {"success_fraction": len(success)/len(hits),
                "median_hit_cost_successes_only": float(np.median(success)) if success else None,
                "mean_capped_cost_budget_plus_one_for_misses": float(np.mean(
                    [budget+1 if h is None else h for h in hits])),
                "misses": len(hits)-len(success)}
    curve = []
    for b in range(budget):
        curve.append({"budget": b+1, "mean_best": float(best[:, b].mean()),
                      "mean_regret": float(regret[:, b].mean()),
                      "median_best": float(np.median(best[:, b])),
                      "best_q25": float(np.quantile(best[:, b], 0.25)),
                      "best_q75": float(np.quantile(best[:, b], 0.75)),
                      "target_hit_fraction": float(np.mean(best[:, b] >= target-1e-12)),
                      "near_optimum_hit_fraction": float(np.mean(best[:, b] >= oracle-NEAR_OPTIMUM_MARGIN-1e-12))})
    return {"curve": curve, "target": hits_summary(target_hits), "near_optimum": hits_summary(near_hits)}


def benchmark(items, seeds=SEEDS, n_initial=INITIAL, budget=BUDGET, target=TARGET):
    if not n_initial < budget <= len(items) or not len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Insufficient endpoint candidates or invalid replay configuration")
    X = np.array([[r["temperature_c"], r["ct_cm"]] for r in items])
    y = np.array([r["conversion_fraction"] for r in items])
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("Duplicate conditions in finite pool")
    oracle = float(y.max())  # Evaluation only, never passed to choose().
    histories = {p: [] for p in POLICIES}
    traces = []
    for seed in seeds:
        initial = np.random.default_rng(seed).choice(len(X), n_initial, replace=False).tolist()
        for p in POLICIES:
            indices, values = replay_once(X, y, initial, p, seed+100_000, budget)
            histories[p].append(values)
            best = np.maximum.accumulate(values)
            for step, (index, value, current) in enumerate(zip(indices, values, best), start=1):
                traces.append({"seed": seed, "policy": p, "budget": step,
                               "phase": "initial" if step <= n_initial else "sequential",
                               "experiment_id": items[index]["experiment_id"],
                               "observed_conversion": value, "best_conversion": float(current),
                               "simple_regret": max(0.0, oracle-float(current))})
    result = {p: summarize(histories[p], oracle, target) for p in POLICIES}
    random_best = np.maximum.accumulate(histories["random"], axis=1)
    for p in POLICIES:
        delta = np.maximum.accumulate(histories[p], axis=1) - random_best
        result[p]["paired_vs_random"] = [
            {"budget": i+1, "mean_best_difference": float(delta[:, i].mean()),
             "win_fraction": float(np.mean(delta[:, i] > 1e-12)),
             "tie_fraction": float(np.mean(np.abs(delta[:, i]) <= 1e-12)),
             "loss_fraction": float(np.mean(delta[:, i] < -1e-12))}
            for i in range(budget)]
    return {"pool_size": len(items), "oracle_best_observed": oracle,
            "target_attainable": oracle >= target,
            "oracle_experiments": [r["experiment_id"] for r in items
                                   if abs(r["conversion_fraction"] - oracle) <= 1e-12],
            "target_conversion": target, "near_optimum_margin": NEAR_OPTIMUM_MARGIN,
            "initial": n_initial, "budget": budget, "seeds": list(seeds),
            "policies": result}, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/processed/monash-v1/mma_conversion.csv"))
    parser.add_argument("--quality", type=Path, default=Path("data/processed/monash-v1/quality.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/monash-bo"))
    args = parser.parse_args()
    rows, *_, quality = load_table(args.table, args.quality)
    args.out.mkdir(parents=True, exist_ok=True)
    results, all_traces = {}, []
    with threadpool_limits(limits=1):
        for name, exclude in (("primary", ()), ("sensitivity", FLAGGED)):
            items, omitted = endpoint_pool(rows, exclude=exclude)
            result, traces = benchmark(items)
            results[name] = {**result, "omitted": omitted, "endpoint_pool": items}
            all_traces.extend({"scenario": name, **r} for r in traces)
    report = {"doi": DOI, "table_sha256": digest(args.table),
              "quality_status": quality["quality_status"],
              "endpoint_minutes": ENDPOINT_MIN, "window_minutes": WINDOW_MIN,
              "aggregation": "median of actual in-window readings within one reaction",
              "policy_priors": {"signal_sd": SIGNAL_SD, "noise_sd": NOISE_SD,
                                "length_scale_normalized": LENGTH_SCALE, "ridge_alpha": RIDGE_ALPHA},
              "environment": {"python": platform.python_version(),
                              **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}},
              "scenarios": results,
              "limitations": "Retrospective exploration on one provisional fixed archive already used for ML CV. "
                "Equal query cost is an assumption; repeated seeds are not independent chemical replications. "
                "Target 15% is an illustrative benchmark, not an engineering specification. "
                "The observed oracle is not the global chemical optimum. GP-EI uses an observed incumbent "
                "and fixed prior scales, not noisy EI; residual GP ignores trend uncertainty. "
                "This does not establish physical experiment savings or safe operating conditions."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out / "traces.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_traces[0]))
        writer.writeheader()
        writer.writerows(all_traces)
    compact = {"table_sha256": report["table_sha256"], "quality_status": report["quality_status"],
               "endpoint_minutes": ENDPOINT_MIN, "window_minutes": WINDOW_MIN,
               "scenarios": {name: {k: v for k, v in r.items() if k not in {"endpoint_pool", "policies"}}
                             | {"policies": {p: {"curve": [m for m in s["curve"] if m["budget"] in {3,5,8,10}],
                                                 "target": s["target"], "near_optimum": s["near_optimum"],
                                                 "paired_vs_random": [m for m in s["paired_vs_random"] if m["budget"] in {5,8,10}]}
                                             for p, s in r["policies"].items()}}
                             for name, r in results.items()},
               "environment": report["environment"]}
    print("BO_SUMMARY " + json.dumps(compact, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
