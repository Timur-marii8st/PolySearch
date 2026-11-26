import numpy as np
import pytest
from polysearch.models import Factor, Target, Constraint, ProjectConfig, Trial, ResultInput, validate_result
from polysearch.engine import Space, recommend, loss, calibrate


def config(**kwargs):
    return ProjectConfig(factors=[Factor(name="x", lower=0, upper=1)],
                         targets=[Target(name="y", lower=0.8, upper=1.0, scale=0.2)], **kwargs)


def trial(x, y, role="train"):
    return Trial(id=str(x), parameters={"x": x}, outcomes={"y": y},
                 role=role, source="manual", created_at="2026-01-01T00:00:00Z")


def test_loss():
    np.testing.assert_allclose(loss(config(), np.array([[0.6], [0.9], [1.2]])), [1, 0, 1])


def test_invalid_constraints():
    with pytest.raises(ValueError):
        Space(config(constraints=[Constraint(coefficients={"x": 1}, upper=-1)]))


def test_cold_start():
    c = config(constraints=[Constraint(coefficients={"x": 1}, upper=0.25)])
    r = recommend(c, [], batch_size=3, pool_size=64)
    assert len(r["recommendations"]) == 3
    assert all(v["prediction"] is None and v["parameters"]["x"] <= 0.25
               for v in r["recommendations"])


def test_calibration_rank():
    assert calibrate(np.zeros(3), np.ones(3), np.array([1, 2, 3]), 0.9) is None
    assert calibrate(np.zeros(3), np.ones(3), np.array([1, 2, 3]), 0.75) == 3


def test_gp_and_calibration_isolation():
    c = config()
    train = [trial(0.1, 0.1), trial(0.3, 0.3), trial(0.5, 0.5)]
    a = recommend(c, train + [trial(0.9, 0.9, "calibration")], pool_size=64)
    b = recommend(c, train + [trial(0.9, 9000, "calibration")], pool_size=64)
    assert a["recommendations"][0]["parameters"] == b["recommendations"][0]["parameters"]
    assert a["recommendations"][0]["score"] == b["recommendations"][0]["score"]
    assert a["report"]["training_trial_ids"] == ["0.1", "0.3", "0.5"]
    assert a["recommendations"][0]["prediction"] is not None
    assert a["report"]["probability_calibrated"] is False


def test_failed_not_training_and_no_duplicates():
    c = config()
    failed = Trial(id="f", parameters={"x": 0.4}, status="failed", source="manual", created_at="")
    r = recommend(c, [failed], pool_size=64)
    assert r["report"]["training_trial_ids"] == []
    assert abs(r["recommendations"][0]["parameters"]["x"] - 0.4) > 1e-7


def test_missing_measurement():
    with pytest.raises(ValueError):
        validate_result(config(), ResultInput(parameters={"x": 0.2}))
