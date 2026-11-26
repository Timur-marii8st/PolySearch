"""Integration tests enabled only by an explicit disposable PostgreSQL DSN."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from polysearch.storage import Store
from polysearch.security import Accounts
from polysearch.admin import migrate
from polysearch.api import create_app

DSN=os.environ.get("POLYSEARCH_TEST_POSTGRES_DSN")
pytestmark=pytest.mark.skipif(not DSN,reason="Requires disposable PostgreSQL test database")


def test_transaction_and_migration(tmp_path,monkeypatch):
    pg=Store(DSN)
    with pg.db.transaction() as db:
        db.execute("DELETE FROM integration_inbox")
        db.execute("DELETE FROM pilot_campaigns")
        db.execute("DELETE FROM state")
    sqlite=Store(tmp_path/"source.db")
    data={"config":None,"trials":[],"recommendations":[],"report":None,"revision":42}
    sqlite.save(data)
    with sqlite.db.transaction() as db:
        db.execute("INSERT INTO pilot_campaigns VALUES (?,?)",("fixture",json.dumps({"id":"fixture","created_at":"2026"})))
    assert migrate(str(tmp_path/"source.db"),pg)=={"workspace_rows":1,"campaigns":1,"inbox_items":0}
    assert pg.load()==data
    with pytest.raises(ValueError,match="overwrite"):
        migrate(str(tmp_path/"source.db"),pg)
    with pytest.raises(RuntimeError):
        with pg.lock:
            pg.save({**data,"revision":99})
            raise RuntimeError("rollback")
    assert pg.load()["revision"]==42
    other=Store(DSN)
    def increment(store):
        with store.lock:
            snapshot=store.load()
            snapshot["revision"]+=1
            store.save(snapshot)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(increment,[pg,other]*12))
    assert pg.load()["revision"]==66


def test_postgres_api(tmp_path,monkeypatch):
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    store=Store(DSN)
    accounts=Accounts(store.db)
    accounts.provision("fixture-admin","test-only-long-password","admin",replace=True)
    with store.db.transaction() as db:
        db.execute("DELETE FROM state")
        db.execute("DELETE FROM integration_inbox")
        db.execute("DELETE FROM pilot_campaigns")
        db.execute("DELETE FROM login_attempts")
    headers={"X-PolySearch-Request":"1"}
    with TestClient(create_app(DSN)) as c:
        assert c.post("/api/auth/login",json={"username":"fixture-admin","password":"test-only-long-password"},headers=headers).status_code==200
        assert c.post("/api/config",json={"config":{"factors":[{"name":"x","lower":0,"upper":1}],
                      "targets":[{"name":"y","lower":0.8,"scale":0.2}]}},headers=headers).status_code==200
        p={"name":"restore-fixture","actor":"fake","objective":"fixture","baseline_method":"fixture",
           "measurement_protocol":"fixture","safety_review":"No physical work","cost_policy":"fixture",
           "budget_per_arm":2,"confirmations_required":1}
        result=c.post("/api/pilot/campaigns",json=p,headers=headers)
        assert result.status_code==201,result.text
        cid=result.json()["id"]
        assert c.get("/api/pilot/campaigns/"+cid).json()["events"][0]["actor"]=="fixture-admin"
    assert store.load()["config"]["mode"]=="real"
