import os
from fastapi.testclient import TestClient
import pytest

from polysearch.api import create_app
from polysearch.security import Accounts, COOKIE, authorized
from polysearch.storage import Store

CFG={"factors":[{"name":"x","lower":0,"upper":1}],"targets":[{"name":"y","lower":0.8,"scale":0.2}]}
PROTOCOL={"name":"test","actor":"spoofed","objective":"fixture","baseline_method":"fixture",
          "measurement_protocol":"fixture","safety_review":"No physical experiment","cost_policy":"fixture",
          "budget_per_arm":2,"confirmations_required":1}
PASSWORD="test-only-strong-password"
HEADERS={"X-PolySearch-Request":"1","Origin":"http://localhost:8000"}


@pytest.fixture
def secured(tmp_path,monkeypatch):
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    path=tmp_path/"secure.sqlite3"
    accounts=Accounts(Store(path).db)
    for role in ("admin","researcher","reviewer","viewer"):
        accounts.provision(role,PASSWORD,role)
    return create_app(path),accounts


def login(c,role):
    r=c.post("/api/auth/login",json={"username":role,"password":PASSWORD},headers=HEADERS)
    assert r.status_code==200,r.text
    assert "httponly" in r.headers["set-cookie"].lower()
    assert "samesite=strict" in r.headers["set-cookie"].lower()


def test_auth_and_csrf(secured):
    app,_=secured
    with TestClient(app) as c:
        assert c.get("/api/state").status_code==401
        assert c.get("/",follow_redirects=False).status_code==303
        assert c.get("/health").status_code==200
        assert c.get("/ready").status_code==200
        assert c.get("/static/login.js").status_code==200
        assert c.post("/api/auth/login",json={"username":"admin","password":PASSWORD}).status_code==403
        login(c,"admin")
        assert c.get("/api/state").status_code==200
        assert c.get("/api/auth/me").json()["user"]["role"]=="admin"
        assert c.post("/api/config",json={"config":CFG}).status_code==403
        assert c.post("/api/config",json={"config":CFG},headers={**HEADERS,"Origin":"https://other.example"}).status_code==403
        assert c.post("/api/config",json={"config":CFG},headers=HEADERS).status_code==200
        assert c.get("/metrics").status_code==200
        assert c.post("/api/auth/logout",headers=HEADERS).status_code==200
        assert c.get("/api/state").status_code==401


def test_rbac_and_actor_binding(secured):
    app,_=secured
    with TestClient(app) as admin, TestClient(app) as researcher, TestClient(app) as reviewer, TestClient(app) as viewer:
        for c,role in [(admin,"admin"),(researcher,"researcher"),(reviewer,"reviewer"),(viewer,"viewer")]:
            login(c,role)
        assert admin.post("/api/config",json={"config":CFG},headers=HEADERS).status_code==200
        assert researcher.post("/api/config",json={"config":CFG},headers=HEADERS).status_code==403
        assert viewer.post("/api/pilot/campaigns",json=PROTOCOL,headers=HEADERS).status_code==403
        assert viewer.get("/api/export.json").status_code==200
        assert viewer.get("/metrics").status_code==403
        r=researcher.post("/api/pilot/campaigns",json=PROTOCOL,headers=HEADERS)
        assert r.status_code==201,r.text
        campaign=r.json()
        assert campaign["protocol"]["actor"]=="researcher"
        assert campaign["events"][0]["actor"]=="researcher"
        path="/api/pilot/campaigns/"+campaign["id"]
        d={"expected_revision":campaign["revision"],"actor":"fake","note":"fixture"}
        assert researcher.post(path+"/activate",json=d,headers=HEADERS).status_code==403
        r=reviewer.post(path+"/activate",json=d,headers=HEADERS)
        assert r.status_code==200,r.text
        assert r.json()["events"][-1]["actor"]=="reviewer"
        assert viewer.get(path+"/report").status_code==200
        assert not authorized("admin","POST","/api/unknown")


def test_session_revocation_and_no_plaintext(secured):
    app,accounts=secured
    with TestClient(app) as c:
        login(c,"researcher")
        token=c.cookies[COOKIE]
        with accounts.db.transaction() as s:
            assert token not in str(s.execute("SELECT token_hash FROM sessions").fetchall())
            assert PASSWORD not in str(s.execute("SELECT password_hash FROM accounts").fetchall())
        accounts.disable("researcher")
        assert c.get("/api/state").status_code==401


def test_login_throttle(secured):
    _,accounts=secured
    for _ in range(12):
        assert accounts.login("missing","incorrect","fixture-peer")[1]==401
    assert accounts.login("admin",PASSWORD,"fixture-peer")[1]==429


def test_insecure_nonloopback_refused(tmp_path,monkeypatch):
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://example.org")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    with pytest.raises(ValueError,match="loopback"):
        create_app(tmp_path/"refused.sqlite3")
