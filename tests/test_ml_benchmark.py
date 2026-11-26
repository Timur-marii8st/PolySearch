import json
import csv

import numpy as np
import pytest

from polysearch.data.monash_source import DOI, digest
from polysearch.ml.estimators import KineticRegressor, group_weights
from polysearch.ml.benchmark import conformal_group_quantile, interval_metrics, macro_mae, load_table, run_cv


def test_group_weights():
    g = np.array(["a", "a", "a", "b"])
    w = group_weights(g)
    assert w[g == "a"].sum() == pytest.approx(w[g == "b"].sum())


def test_kinetics_monotone_and_zero():
    t = np.linspace(0, 10, 20)
    X = np.column_stack([np.full(20, 90), np.full(20, 0.05), t])
    y = -np.expm1(-0.02 * t)
    model = KineticRegressor().fit(X, y)
    p = model.predict(X)
    assert p[0] == 0
    assert np.all(np.diff(p) >= 0)
    assert np.max(np.abs(p-y)) < 0.001


def test_group_conformal_finite_rank():
    g = np.array(["a", "a", "b", "c", "d"])
    scores = np.array([0.1, 9, 2, 3, 4])
    assert conformal_group_quantile(scores, g, 0.8) == 9
    assert conformal_group_quantile(scores, g, 0.9) is None


def test_group_metrics_not_row_weighted():
    y = np.zeros(4)
    p = np.array([0, 0, 0, 1])
    g = np.array(["a", "a", "a", "b"])
    assert macro_mae(y, p, g) == pytest.approx(0.5)
    m = interval_metrics(y, p, p, g)
    assert m["row_coverage"] == 0.75
    assert m["whole_group_coverage"] == 0.5


def test_load_table_rejects_group_leakage(tmp_path):
    table = tmp_path / "data.csv"
    quality = tmp_path / "quality.json"
    rows = [dict(temperature_c=90, ct_cm=0.05, residence_min=i+1, conversion_fraction=0.1,
                 monomer="MMA", experiment_id="e1", split_group="g1", fold=i)
            for i in range(2)]
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    quality.write_text(json.dumps(dict(doi=DOI, monomer="MMA", table_sha256=digest(table),
                                       counts={"accepted": 2}, folds=2, condition_groups=1)))
    with pytest.raises(ValueError, match="leakage"):
        load_table(table, quality)


def test_load_table_rejects_changed_bytes(tmp_path):
    table, quality = tmp_path / "data.csv", tmp_path / "quality.json"
    table.write_text("changed")
    quality.write_text(json.dumps(dict(doi=DOI, monomer="MMA", table_sha256="wrong")))
    with pytest.raises(ValueError, match="checksum"):
        load_table(table, quality)


def test_group_cv_no_overlap(monkeypatch):
    # Small deterministic smoke fixture; no scientific claim.
    import polysearch.ml.benchmark as b
    monkeypatch.setattr(b, "GRIDS", {"ridge": [{"alpha": 0.1}], "catboost": [{"depth": 2, "iterations": 5}],
                                      "gp": [{}], "kinetic": [{"regularization": 0.01}]})
    g = np.repeat([f"g{i}" for i in range(10)], 3)
    folds = np.repeat(np.arange(10) % 5, 3)
    X = np.column_stack([np.repeat(np.linspace(80,100,10),3),
                         np.repeat(np.linspace(0.01,0.1,10),3), np.tile([1,3,5],10)])
    y = -np.expm1(-0.02 * X[:,2])
    report, preds, _ = run_cv(X,y,g,g,folds,42,intervals=True)
    assert all(np.all(np.isfinite(p)) for p in preds.values())
    for f in report["fold_reports"]:
        assert not set(f["train_groups"]) & set(f["test_groups"])
        cal = f["calibration"]
        assert not set(cal["fit_groups"]) & set(cal["calibration_groups"])
        assert not set(cal["calibration_groups"]) & set(f["test_groups"])
        assert cal["q90"] is None
