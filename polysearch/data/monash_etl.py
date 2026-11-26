"""Create an auditable MMA conversion table from reviewed Monash v1 mappings.

Raw CSV formats and units must be explicitly established from source records.
Rows from one reaction (and replicates at identical conditions) share one fold.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .monash_source import DOI, digest

FIELDS = ["experiment_id", "split_group", "fold", "monomer", "temperature_c",
          "ct_cm", "residence_min", "conversion_fraction", "source_file",
          "source_row", "source_sha256", "timesweep"]


def finite(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite measurement")
    return result


def source_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root / "extracted") or not path.is_file():
        raise ValueError(f"Not an extracted release file: {name}")
    return path


def canonical_condition(monomer, temperature, ratio):
    # Identical chemistry settings stay in one fold, even across replicate runs.
    key = json.dumps([monomer, temperature + 0.0, ratio + 0.0], separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def validate_mapping(mapping, inventory, root):
    if mapping.get("doi") != DOI or inventory.get("doi") != DOI:
        raise ValueError("Mapping and inventory must refer to pinned Monash v1")
    if mapping.get("monomer") != "MMA" or not mapping.get("review_note"):
        raise ValueError("Expected reviewed MMA mapping with provenance note")
    sources = mapping.get("sources", [])
    if not sources:
        raise ValueError("No reviewed source mappings supplied")
    known = {entry["path"]: entry["sha256"] for entry in inventory["files"]}
    if mapping.get("summary_path"):
        name = mapping["summary_path"]
        sha = mapping.get("summary_sha256")
        if not sha or known.get(name) != sha or digest(source_path(root, name)) != sha:
            raise ValueError("Condition workbook checksum mismatch")
    conditions = {}
    for source in sources:
        name = source["path"]
        path = source_path(root, name)
        if known.get(name) != source.get("sha256") or digest(path) != source["sha256"]:
            raise ValueError(f"Source hash does not match inventory and mapping: {name}")
        experiment = source["experiment_id"]
        if not isinstance(experiment, str) or not experiment.strip():
            raise ValueError("Missing independent experiment ID")
        temp = finite(source["temperature_c"])
        ratio = finite(source["ct_cm"])
        if temp <= -273.15 or ratio < 0 or not source.get("condition_evidence"):
            raise ValueError("Invalid or undocumented experimental conditions")
        if experiment in conditions and conditions[experiment] != (temp, ratio):
            raise ValueError("One reaction ID has inconsistent conditions")
        conditions[experiment] = (temp, ratio)
        if source["residence_unit"] not in {"min", "s"} or source["conversion_unit"] not in {"fraction", "percent"}:
            raise ValueError("Units must be explicit: min/s and fraction/percent")
        if source["delimiter"] not in {",", ";", "\t"}:
            raise ValueError("Unsupported delimiter")
        if source.get("header_row", 1) < 1:
            raise ValueError("header_row must be >= 1")
        if source.get("measurement_status") not in {"reviewed", "provisional"} or not source.get("selection_evidence"):
            raise ValueError("Review valid timesweeps before preparing measurements")
        if source.get("measurement_status") == "provisional" and (not source.get("status_column") or not source.get("accepted_status") or not source.get("timesweep_column")):
            raise ValueError("Provisional sources require explicit platform status and timesweep fields")
        exclusions = source.get("exclude_rows", [])
        for item in exclusions:
            if not (1 <= int(item["start"]) <= int(item["end"])) or not item.get("reason"):
                raise ValueError("Each excluded row range needs a documented reason")
    return sources


def prepare(root, mapping, folds=5, seed=42):
    root = Path(root)
    inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    sources = validate_mapping(mapping, inventory, root)
    rows, rejected = [], []
    seen_rows = set()
    counts = Counter()
    for source in sources:
        path = source_path(root, source["path"])
        temp = finite(source["temperature_c"])
        ratio = finite(source["ct_cm"])
        experiment = source["experiment_id"]
        split_group = canonical_condition("MMA", temp, ratio)
        header_row = int(source.get("header_row", 1))
        exclusions = source.get("exclude_rows", [])
        with path.open(encoding=source.get("encoding", "utf-8-sig"), newline="") as stream:
            for _ in range(header_row - 1):
                next(stream, None)
            reader = csv.DictReader(stream, delimiter=source["delimiter"], strict=True)
            names = reader.fieldnames or []
            mapped = {source["residence_column"], source["conversion_column"]}
            mapped.update(source[k] for k in ("status_column", "timesweep_column") if source.get(k))
            if len(names) != len(set(names)) or not mapped <= set(names):
                raise ValueError(f"Mapped columns missing or ambiguous: {source['path']}: {names}")
            if source["residence_column"] == source["conversion_column"]:
                raise ValueError("Residence and conversion columns cannot coincide")
            for raw in reader:
                # Physical end line number of this CSV record, even for quoted multiline values.
                line = header_row - 1 + reader.line_num
                key = (source["path"], line)
                if key in seen_rows:
                    raise ValueError(f"Source row mapped more than once: {key}")
                seen_rows.add(key)
                counts["read"] += 1
                reason = next((e["reason"] for e in exclusions if int(e["start"]) <= line <= int(e["end"])), None)
                if reason:
                    reason = "reviewed_exclusion: " + str(reason)
                elif source.get("status_column") and str(raw.get(source["status_column"], "")).strip() not in source["accepted_status"]:
                    reason = "outside_timesweep: " + str(raw.get(source["status_column"], ""))
                else:
                    try:
                        if None in raw or any(v is None for v in raw.values()):
                            raise ValueError("malformed row")
                        residence = finite(raw[source["residence_column"]])
                        conversion = finite(raw[source["conversion_column"]])
                        if source["residence_unit"] == "s":
                            residence /= 60
                        if source["conversion_unit"] == "percent":
                            conversion /= 100
                        if residence <= 0:
                            raise ValueError("non-positive residence time")
                        if not 0 <= conversion <= 1:
                            raise ValueError("conversion outside [0,1]")
                    except (TypeError, ValueError) as error:
                        reason = "invalid_measurement: " + str(error)
                if reason:
                    rejected.append({"source_file": source["path"], "source_row": line,
                                     "experiment_id": experiment, "reason": reason})
                    counts["rejected"] += 1
                    continue
                rows.append({"experiment_id": experiment, "split_group": split_group,
                             "monomer": "MMA", "temperature_c": temp, "ct_cm": ratio,
                             "residence_min": residence, "conversion_fraction": conversion,
                             "source_file": source["path"], "source_row": line,
                             "source_sha256": source["sha256"],
                             "timesweep": raw.get(source.get("timesweep_column", ""), "")})
                counts["accepted"] += 1
    if not rows:
        raise ValueError("No accepted measurements")
    groups = sorted({r["split_group"] for r in rows})
    if not 2 <= folds <= len(groups):
        raise ValueError(f"Need at least {folds} distinct condition groups; found {len(groups)}")
    # Equal group count per fold; the number of spectral rows is not the unit of independence.
    random.Random(seed).shuffle(groups)
    assigned = {group: i % folds for i, group in enumerate(groups)}
    for row in rows:
        row["fold"] = assigned[row["split_group"]]
    rows.sort(key=lambda r: (r["experiment_id"], r["residence_min"], r["source_file"], r["source_row"]))
    by_experiment = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment_id"]].append(row)
    sweep_diagnostics = {}
    for experiment, values in sorted(by_experiment.items()):
        sweeps = defaultdict(list)
        for row in values:
            if row["timesweep"]:
                sweeps[row["timesweep"]].append(row)
        summaries = []
        for label, sweep in sweeps.items():
            ordered = sorted(sweep, key=lambda r: r["residence_min"])
            first, last = ordered[0], ordered[-1]
            summaries.append({"timesweep": label, "n": len(sweep), "t_min": first["residence_min"],
                              "t_max": last["residence_min"], "conversion_min": min(r["conversion_fraction"] for r in sweep),
                              "conversion_max": max(r["conversion_fraction"] for r in sweep)})
        flags = []
        labels = list(sweeps)
        for a, b in zip(labels, labels[1:]):
            # Compare nearest available pair only if their residence times are close.
            pairs = [(abs(x["residence_min"] - y["residence_min"]), x, y) for x in sweeps[a] for y in sweeps[b]]
            gap, left, right = min(pairs, key=lambda p: p[0])
            jump = abs(left["conversion_fraction"] - right["conversion_fraction"])
            if gap > 0.25:
                flags.append({"between": [a,b], "flag": "no_comparable_boundary", "time_gap_min": gap})
            elif jump > 0.05:
                flags.append({"between": [a,b], "flag": "possible_jump", "time_gap_min": gap, "conversion_gap": jump})
        sweep_diagnostics[experiment] = {"sweeps": summaries, "flags": flags}
    report = {
        "quality_status": "provisional" if any(s["measurement_status"] == "provisional" for s in sources) else "reviewed",
        "missing_sources": mapping.get("missing_sources", []),
        "timesweep_diagnostics": sweep_diagnostics,
        "exclusion_reasons": dict(Counter(r["reason"].split(":")[0] for r in rejected)),
        "doi": DOI, "monomer": "MMA", "task": "predict conversion fraction from residence time and reaction conditions",
        "counts": dict(counts), "independent_experiments": len(by_experiment),
        "condition_groups": len(groups), "folds": folds, "seed": seed,
        "fold_group_counts": dict(sorted(Counter(assigned.values()).items())),
        "fold_row_counts": dict(sorted(Counter(r["fold"] for r in rows).items())),
        "experiments": {key: {"rows": len(values), "fold": values[0]["fold"],
                               "temperature_c": values[0]["temperature_c"],
                               "ct_cm": values[0]["ct_cm"],
                               "residence_min_range": [min(v["residence_min"] for v in values),
                                                       max(v["residence_min"] for v in values)]}
                        for key, values in sorted(by_experiment.items())},
        "ranges": {col: {"min": min(r[col] for r in rows), "max": max(r[col] for r in rows),
                         "median": statistics.median(r[col] for r in rows)}
                   for col in ("temperature_c", "ct_cm", "residence_min", "conversion_fraction")},
        "duplicate_feature_rows": len(rows) - len({(r["experiment_id"], r["temperature_c"],
                                                    r["ct_cm"], r["residence_min"]) for r in rows}),
        "warning": "Rows are correlated within reaction curves. No evidence of independent synthesis savings."
    }
    return rows, rejected, report


def write_outputs(out, rows, rejected, report, mapping_path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = out / "mma_conversion.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "rejected.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_file", "source_row", "experiment_id", "reason"])
        writer.writeheader()
        writer.writerows(rejected)
    report["mapping_sha256"] = digest(mapping_path)
    report["table_sha256"] = digest(table)
    (out / "quality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/monash-v1"))
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/monash-v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows, rejected, report = prepare(args.root, mapping, args.folds, args.seed)
    write_outputs(args.out, rows, rejected, report, args.mapping)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
