import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from polysearch.api import create_app
from polysearch.engine import recommend, MAX_TRAIN_TRIALS
from polysearch.hardening import DEFAULT_BODY_LIMIT
from polysearch.models import ProjectConfig, Trial

CFG = {"factors": [{"name":"x","lower":0,"upper":1}],
       "targets": [{"name":"y","lower":.8,"scale":.2}]}


def test_rebinding_host_and_local_csrf(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYSEARCH_TRUSTED_HOSTS", "testserver,localhost,127.0.0.1")
    with TestClient(create_app(tmp_path/"p.db")) as c:
        assert c.get("/health",headers={"Host":"external.invalid"}).status_code==400
        assert c.post("/api/config",json={"config":CFG},headers={"Origin":"https://external.invalid"}).status_code==403
        assert c.post("/api/config",json={"config":CFG},headers={"Sec-Fetch-Site":"cross-site"}).status_code==403
        assert c.post("/api/config",json={"config":CFG},headers={"Origin":"http://testserver"}).status_code==200


def test_body_budget_and_media_type(tmp_path):
    with TestClient(create_app(tmp_path/"p.db")) as c:
        r=c.post("/api/config",content=b"x"*(DEFAULT_BODY_LIMIT+1),headers={"Content-Type":"application/json"})
        assert r.status_code==413
        def chunks():
            for _ in range(5):
                yield b"x"*65536
        r=c.post("/api/config",content=chunks(),headers={"Content-Type":"application/json"})
        assert r.status_code==413
        assert c.post("/api/config",content=json.dumps({"config":CFG}),headers={"Content-Type":"text/plain"}).status_code==415
        assert c.get("/api/state").json()["config"] is None
        assert c.post("/api/config",json={"config":CFG}).status_code==200


def test_errors_do_not_echo_input(tmp_path):
    secret="FIXTURE-SENSITIVE-TEXT"
    with TestClient(create_app(tmp_path/"p.db")) as c:
        r=c.post("/api/auth/login",json={"username":"user","password":secret*20})
        assert r.status_code==422
        assert secret not in r.text
        assert all(set(item)=={"loc","msg","type"} for item in r.json()["detail"])


def test_gp_budget_is_explicit():
    cfg=ProjectConfig.model_validate(CFG)
    t=Trial(id="fixture",parameters={"x":.5},outcomes={"y":.5},source="manual",
            created_at=datetime.now(timezone.utc).isoformat())
    with pytest.raises(ValueError,match="training limit"):
        recommend(cfg,[t]*(MAX_TRAIN_TRIALS+1))
