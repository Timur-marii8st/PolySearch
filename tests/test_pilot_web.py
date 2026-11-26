from fastapi.testclient import TestClient
from polysearch.api import create_app


def test_pilot_page_and_assets(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as client:
        response = client.get("/pilot")
        assert response.status_code == 200
        assert "Экспериментальные кампании" in response.text
        assert client.get("/static/pilot.js").status_code == 200
        assert 'href="/pilot"' in client.get("/").text
        assert client.get("/api/pilot/campaigns").json() == []
