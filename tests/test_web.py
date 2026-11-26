"""HTTP integration checks for the packaged local interface."""
from fastapi.testclient import TestClient
from polysearch.api import create_app


def test_web_assets(tmp_path):
    with TestClient(create_app(tmp_path / "web.db")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "PolySearch" in page.text
        assert '/static/style.css' in page.text
        assert '/static/app.js' in page.text
        for asset in ("/static/style.css", "/static/app.js"):
            response = client.get(asset)
            assert response.status_code == 200
            assert response.text.strip()
        assert client.get("/static/does-not-exist.js").status_code == 404


def test_web_and_api_coexist(tmp_path):
    with TestClient(create_app(tmp_path / "web.db")) as client:
        assert client.get("/").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/api/state").json()["config"] is None
        assert client.post("/api/demo", json={"n_initial": 3}).status_code == 200
        assert client.get("/api/state").json()["config"]["mode"] == "demo"
        assert len(client.get("/api/export.json").json()["trials"]) == 3
