import pytest


@pytest.fixture(autouse=True)
def test_host_allowlist(monkeypatch):
    # Starlette TestClient uses the synthetic host testserver.
    monkeypatch.setenv("POLYSEARCH_TRUSTED_HOSTS", "testserver,localhost,127.0.0.1,[::1]")
