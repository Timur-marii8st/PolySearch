"""Integration checks for file exchange, never actual laboratory data."""
import copy
from datetime import datetime, timezone
import os

from fastapi.testclient import TestClient
import pytest

from polysearch.api import create_app
from polysearch.security import Accounts
from polysearch.storage import Store
from polysearch.exchange import Inbox

CFG={"factors":[{"name":"x","lower":0,"upper":1,"unit":"a.u."}],
     "targets":[{"name":"y","lower":.8,"scale":.2,"unit":"a.u."}]}
PROTOCOL={"name":"software fixture","actor":"fixture","objective":"fixture only",
          "baseline_method":"fixture","measurement_protocol":"Synthetic fixture, no physical run",
          "safety_review":"No physical work","cost_policy":"Fixture costs only",
          "budget_per_arm":2,"confirmations_required":1}


def call(client, path, body=None, code=200, headers=None):
    response=client.get(path,headers=headers) if body is None else client.post(path,json=body,headers=headers)
    assert response.status_code==code,response.text
    return response.json()


def setup(client, headers=None):
    call(client,"/api/config",{"config":CFG},headers=headers)
    c=call(client,"/api/pilot/campaigns",PROTOCOL,201,headers)
    root="/api/pilot/campaigns/"+c["id"]
    def change(c,tail,**data):
        return call(client,root+tail,{"expected_revision":c["revision"],"actor":"fixture",**data},headers=headers)
    c=change(c,"/activate",note="fixture")
    c=change(c,"/plans",arm="baseline",parameters={"x":.5},rationale="fixture")
    pid=c["plans"][-1]["id"]
    return c,root,pid,change


def envelope(c,pid,order,external_id="fixture-result"):
    return {"schema_version":"polysearch.lab.v1","source_system":"fixture-lims",
            "external_id":external_id,"campaign_id":c["id"],"plan_id":pid,
            "work_order_sha256":order["work_order_sha256"],"run_id":"fixture-run",
            "observed_at":datetime.now(timezone.utc).isoformat(),
            "actual_parameters":order["parameters"],"outcomes":{"y":{"value":.9,"unit":"a.u."}},
            "status":"completed","evidence":"fixture only","currency":"RUB","cost_minor":100,
            "elapsed_minutes":1}


def approved(client,headers=None):
    c,root,pid,change=setup(client,headers)
    order_path="/api/exchange/campaigns/"+c["id"]+"/orders/"+pid
    call(client,order_path,code=409,headers=headers)
    c=change(c,"/plans/"+pid+"/review",decision="approve",note="fixture")
    order=call(client,order_path,headers=headers)
    return c,root,pid,order


def stage(client,result,code=201,headers=None):
    return call(client,"/api/exchange/inbox",{"actor":"fixture","result":result},code,headers)


def review(client,item,c,decision="accept",code=200,headers=None):
    return call(client,"/api/exchange/inbox/"+item["id"]+"/review",
                {"actor":"fixture","expected_revision":c["revision"],"decision":decision,"note":"checked fixture"},code,headers)


def full_flow(cl,headers=None):
    c,root,pid,order=approved(cl,headers)
    result=envelope(c,pid,order)
    item=stage(cl,result,headers=headers)
    assert item["state"]=="pending"
    assert stage(cl,result,headers=headers)["id"]==item["id"]
    assert call(cl,root+"/report",headers=headers)["arms"]["baseline"]["attempts"]==0
    done=review(cl,item,c,headers=headers)
    assert done["state"]=="accepted"
    assert review(cl,item,c,headers=headers)==done
    assert stage(cl,result,headers=headers)==done
    saved=call(cl,root,headers=headers)["plans"][-1]["measurement"]
    assert saved["integration"]["inbox_id"]==item["id"]
    assert saved["integration"]["payload_sha256"]==item["payload_sha256"]
    assert call(cl,root+"/report",headers=headers)["arms"]["baseline"]["attempts"]==1
    return c,item


def test_exchange_end_to_end(tmp_path):
    db=tmp_path/"fixture.db"
    with TestClient(create_app(db)) as cl:
        assert cl.get("/exchange").status_code==200
        assert cl.get("/static/exchange.js").status_code==200
        c,item=full_flow(cl)
    with TestClient(create_app(db)) as cl:
        assert call(cl,"/api/exchange/inbox/"+item["id"])["state"]=="accepted"
        assert call(cl,"/api/exchange/inbox?campaign_id="+c["id"])["total"]==1


@pytest.mark.parametrize("invalid,code",[
    ("unit",422),("hash",409),("currency",422),("missing",422),("timestamp",422),
    ("numeric",422),("deviation",422),("failed",422)])
def test_reject_invalid_result(tmp_path,invalid,code):
    with TestClient(create_app(tmp_path/"p.db")) as cl:
        c,_,pid,order=approved(cl)
        r=envelope(c,pid,order)
        if invalid=="unit": r["outcomes"]["y"]["unit"]="%"
        elif invalid=="hash": r["work_order_sha256"]="0"*64
        elif invalid=="currency": r["currency"]="USD"
        elif invalid=="missing": r["outcomes"]={}
        elif invalid=="timestamp": r["observed_at"]="2000-01-01T00:00:00Z"
        elif invalid=="numeric": r["outcomes"]["y"]["value"]="0.9"
        elif invalid=="deviation": r["actual_parameters"]["x"]["value"]=2
        else: r["status"]="failed"
        stage(cl,r,code)
        assert call(cl,"/api/exchange/inbox")["total"]==0


def test_conflict_stale_and_rejection(tmp_path):
    with TestClient(create_app(tmp_path/"p.db")) as cl:
        c,root,pid,order=approved(cl)
        r=envelope(c,pid,order)
        item=stage(cl,r)
        other=copy.deepcopy(r)
        other["cost_minor"]=101
        stage(cl,other,409)
        review(cl,item,{**c,"revision":c["revision"]-1},code=409)
        result=review(cl,item,c,"reject")
        assert result["state"]=="rejected"
        assert review(cl,item,c,"reject")==result
        review(cl,item,c,"accept",409)
        assert call(cl,root+"/report")["arms"]["baseline"]["attempts"]==0


def test_already_recorded_plan_cannot_be_accepted(tmp_path):
    with TestClient(create_app(tmp_path/"p.db")) as cl:
        c,root,pid,order=approved(cl)
        item=stage(cl,envelope(c,pid,order))
        updated=call(cl,root+"/plans/"+pid+"/measurement",{
            "expected_revision":c["revision"],"actor":"fixture","run_id":"manual-run",
            "actual_parameters":{"x":.5},"outcomes":{"y":.9},"status":"completed",
            "evidence":"fixture","cost_minor":1,"elapsed_minutes":1})
        review(cl,item,updated,code=409)
        assert call(cl,"/api/exchange/inbox/"+item["id"])["state"]=="pending"
        assert call(cl,root+"/report")["arms"]["baseline"]["attempts"]==1


def test_atomic_accept_rollback(tmp_path,monkeypatch):
    with TestClient(create_app(tmp_path/"p.db"),raise_server_exceptions=False) as cl:
        c,root,pid,order=approved(cl)
        item=stage(cl,envelope(c,pid,order))
        original=Inbox.save
        def fail(self,value):
            if value["state"]=="accepted":
                raise RuntimeError("fixture rollback")
            return original(self,value)
        monkeypatch.setattr(Inbox,"save",fail)
        response=cl.post("/api/exchange/inbox/"+item["id"]+"/review",json={
            "expected_revision":c["revision"],"actor":"fixture","decision":"accept","note":"fixture"})
        assert response.status_code==500
        assert call(cl,root+"/report")["arms"]["baseline"]["attempts"]==0
        assert call(cl,"/api/exchange/inbox/"+item["id"])["state"]=="pending"
        monkeypatch.setattr(Inbox,"save",original)
        assert review(cl,item,c)["state"]=="accepted"


def test_exchange_roles(tmp_path,monkeypatch):
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    db=tmp_path/"secure.db"
    accounts=Accounts(Store(db).db)
    for name in ("admin","researcher","reviewer","viewer"):
        accounts.provision(name,"fixture-only-password",name)
    app=create_app(db)
    headers={"X-PolySearch-Request":"1"}
    with TestClient(app) as admin,TestClient(app) as researcher,TestClient(app) as reviewer,TestClient(app) as viewer:
        for client,name in ((admin,"admin"),(researcher,"researcher"),(reviewer,"reviewer"),(viewer,"viewer")):
            call(client,"/api/auth/login",{"username":name,"password":"fixture-only-password"},headers=headers)
        c,_,pid,order=approved(admin,headers)
        result=envelope(c,pid,order)
        call(viewer,"/api/exchange/inbox",code=403)
        stage(viewer,result,403,headers)
        stage(reviewer,result,403,headers)
        item=stage(researcher,result,headers=headers)
        assert item["received_by"]=="researcher"
        review(researcher,item,c,code=403,headers=headers)
        assert review(reviewer,item,c,headers=headers)["decision"]["actor"]=="reviewer"


@pytest.mark.skipif(not os.environ.get("POLYSEARCH_TEST_POSTGRES_DSN"),reason="Disposable PostgreSQL only")
def test_exchange_postgres(monkeypatch):
    dsn=os.environ["POLYSEARCH_TEST_POSTGRES_DSN"]
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    store=Store(dsn)
    with store.db.transaction() as tx:
        for name in ("integration_inbox","pilot_campaigns","state","login_attempts"):
            tx.execute("DELETE FROM "+name)
    Accounts(store.db).provision("exchange-admin","fixture-only-password","admin",replace=True)
    headers={"X-PolySearch-Request":"1"}
    with TestClient(create_app(dsn)) as cl:
        call(cl,"/api/auth/login",{"username":"exchange-admin","password":"fixture-only-password"},headers=headers)
        full_flow(cl,headers)
