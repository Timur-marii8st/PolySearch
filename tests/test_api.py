from fastapi.testclient import TestClient
from polysearch.api import create_app

CONFIG = {"factors": [{"name": "x", "lower": 0, "upper": 1}],
          "targets": [{"name": "y", "lower": 0.8, "scale": 0.2}]}


def test_closed_loop(tmp_path):
    db = tmp_path / "state.db"
    with TestClient(create_app(db)) as c:
        assert c.post("/api/demo", json={"n_initial": 6}).status_code == 200
        assert c.post("/api/demo", json={}).status_code == 409
        response = c.post("/api/recommend", json={})
        assert response.status_code == 200, response.text
        rec = response.json()["recommendations"][0]
        payload = {"recommendation_id": rec["id"]}
        assert c.post("/api/simulate", json=payload).status_code == 200
        assert c.post("/api/simulate", json=payload).status_code == 409
    with TestClient(create_app(db)) as c:
        state = c.get("/api/state").json()
        assert len(state["trials"]) == 7
        assert state["recommendations"] == []
        assert c.post("/api/recommend", json={}).status_code == 200


def test_csv_atomic_and_real_mode(tmp_path):
    with TestClient(create_app(tmp_path / "state.db")) as c:
        assert c.post("/api/config", json={"config": CONFIG}).status_code == 200
        assert c.post("/api/import.csv", json={"csv": "x,y\n0.1,0.2\n2,0.8"}).status_code == 422
        assert c.get("/api/state").json()["trials"] == []
        assert c.post("/api/import.csv", json={"csv": "x,y\n0.1,0.2\n0.5,0.7"}).status_code == 200
        assert len(c.get("/api/state").json()["trials"]) == 2
        assert c.post("/api/simulate", json={"recommendation_id": "fake"}).status_code == 409


def test_stale_result(tmp_path):
    with TestClient(create_app(tmp_path / "state.db")) as c:
        assert c.post("/api/config", json={"config": CONFIG}).status_code == 200
        r = c.post("/api/recommend", json={}).json()["recommendations"][0]
        result = {"parameters": r["parameters"], "outcomes": {"y": 0.9},
                  "recommendation_id": r["id"]}
        assert c.post("/api/results", json=result).status_code == 200
        assert c.post("/api/results", json=result).status_code == 409


def test_bad_config(tmp_path):
    with TestClient(create_app(tmp_path / "state.db")) as c:
        cfg = dict(CONFIG, constraints=[{"coefficients": {"x": 1}, "upper": -1}])
        assert c.post("/api/config", json={"config": cfg}).status_code == 422
        assert c.get("/api/state").json()["config"] is None
