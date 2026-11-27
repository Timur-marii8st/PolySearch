"""Software fixtures only; no observed chemical process data."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from polysearch.api import create_app
from polysearch.ml.workbench import StudyRequest, analyze, parse
from polysearch.security import Accounts
from polysearch.storage import Store

CFG={"factors":[{"name":"x","lower":0,"upper":1,"unit":"a.u."}],
     "targets":[{"name":"property","lower":.6,"scale":1,"unit":"a.u."}]}
GROUPS={"train":[.05,.15,.25,.35,.65,.75,.85,.95],
        "calibration":[.1,.2,.3,.4,.5,.6,.7,.8,.9],
        "validation":[.12,.42,.82]}


def fixture(**overrides):
    rows=["group_id,role,x,property"]
    for role,values in GROUPS.items():
        for i,x in enumerate(values):
            rows.append(f"{role}-{i},{role},{x},{.2+.6*x:.8f}")
    result={"name":"software fixture","dataset_kind":"synthetic_fixture","provenance":"No physical experiment",
            "config":CFG,"csv":"\n".join(rows)+"\n","max_interval_width":{"property":.2},
            "coverage":.9,"budget_minor":1000,"candidate_pool_reviewed":True,
            "candidates":[{"id":"A","parameters":{"x":.45},"cost_minor":500},
                          {"id":"B","parameters":{"x":.99},"cost_minor":500},
                          {"id":"C","parameters":{"x":.55},"cost_minor":700}]}
    return {**result,**overrides}


def post(client,body,headers=None,expected=201):
    r=client.post("/api/research/runs",json=body,headers=headers)
    assert r.status_code==expected,r.text
    return r.json()


def test_run_persistence_and_workspace_isolation(tmp_path):
    path=tmp_path/"r.db"
    with TestClient(create_app(path)) as client:
        assert client.get("/research").status_code==200
        assert client.get("/static/research.js").status_code==200
        before=client.get("/api/state").json()
        record=post(client,fixture())
        output=record["result"]
        assert record["status"]=="research_only"
        assert len(record["record_sha256"])==64
        assert set(v["model"] for v in output["selection"])=={"mean","ridge","gp"}
        assert output["calibration"][0]["groups"]==9
        assert output["counts"]["train"]["groups"]==8
        assert output["counts"]["validation"]["groups"]==3
        assert output["validation"]["property"]["groups"]==3
        assert output["candidate_predictions"][1]["properties"]["property"]["status"]=="abstain"
        assert output["candidate_predictions"][1]["properties"]["property"]["prediction"] is None
        assert output["collection_plan"]["spent_minor"]<=1000
        assert len({x["id"] for x in output["collection_plan"]["suggestions"]})==len(output["collection_plan"]["suggestions"])
        assert client.get("/api/state").json()==before
        assert client.get("/api/research/runs").json()["total"]==1
    with TestClient(create_app(path)) as client:
        assert client.get("/api/research/runs/"+record["id"]).json()==record
        assert client.get("/api/research/runs/missing").status_code==404


def test_validation_labels_not_used_for_selection_or_predictions(tmp_path):
    with TestClient(create_app(tmp_path/"r.db")) as client:
        req=fixture()
        a=post(client,req)
        req["csv"]="\n".join(line.rsplit(",",1)[0]+",0.01" if ",validation," in line else line
                             for line in req["csv"].splitlines())+"\n"
        b=post(client,req)
        assert a["input_sha256"]!=b["input_sha256"]
        assert a["model_version"]==b["model_version"]
        for k in ("selected_model","selection","calibration","candidate_predictions","collection_plan","training_sha256"):
            assert a["result"][k]==b["result"][k]
        assert a["result"]["validation"]!=b["result"]["validation"]


def test_no_calibration_abstains_and_budget_is_label_free():
    req=fixture(model="ridge")
    req["csv"]="\n".join(l for l in req["csv"].splitlines() if ",calibration," not in l)+"\n"
    r=analyze(StudyRequest.model_validate(req))
    assert r["calibration"][0]["quantile"] is None
    assert r["candidate_predictions"][0]["properties"]["property"]["reason"]=="interval_unavailable"
    assert all(q["properties"]["property"]["prediction"] is None for q in r["candidate_predictions"])
    req["csv"]="\n".join(line.rsplit(",",1)[0]+",0.01" if ",train," in line else line
                             for line in req["csv"].splitlines())+"\n"
    other=analyze(StudyRequest.model_validate(req))
    assert r["collection_plan"]==other["collection_plan"]


@pytest.mark.parametrize("fault",["group_overlap","condition_overlap","candidate_repeat","pool_not_reviewed","bad_number","all_missing","bad_header"])
def test_invalid_csv_and_candidates_are_not_persisted(tmp_path,fault):
    req=fixture()
    if fault=="group_overlap":
        req["csv"]+="train-0,validation,0.11,0.3\n"
    elif fault=="condition_overlap":
        req["csv"]+="extra,validation,0.05,0.3\n"
    elif fault=="candidate_repeat":
        req["candidates"][0]["parameters"]["x"]=.05
    elif fault=="pool_not_reviewed":
        req["candidate_pool_reviewed"]=False
    elif fault=="bad_number":
        req["csv"]=req["csv"].replace(",0.05,",",SENSITIVE_VALUE,")
    elif fault=="all_missing":
        req["csv"]+="extra,validation,0.11,\n"
    else:
        req["csv"]=req["csv"].replace("group_id,role","group_id,role,role",1)
    with TestClient(create_app(tmp_path/"r.db")) as client:
        r=client.post("/api/research/runs",json=req)
        assert r.status_code==422,r.text
        assert "SENSITIVE_VALUE" not in r.text
        assert client.get("/api/research/runs").json()["total"]==0


def test_coupled_masked_labels_and_identity():
    cfg=copy.deepcopy(CFG)
    cfg["targets"]=[{"name":n,"unit":u,"lower":0,"scale":s} for n,u,s in
                    (("conversion_fraction","fraction",1),("mn_g_mol","g/mol",1000),
                     ("mw_g_mol","g/mol",1000),("dispersity","1",1))]
    rows=["group_id,role,x,conversion_fraction,mn_g_mol,mw_g_mol,dispersity"]
    for role,values in GROUPS.items():
        for i,x in enumerate(values):
            mn=1000+100*x;d=1.2+.1*x;mw=mn*d
            rows.append(f"{role}-{i},{role},{x},{.2+.6*x},{mn if i%3!=0 else ''},{mw},{d}")
    req=StudyRequest.model_validate(fixture(config=cfg,csv="\n".join(rows)+"\n",coupled_mass=True,
        max_interval_width={"conversion_fraction":1,"mn_g_mol":10000,"mw_g_mol":10000,"dispersity":10}))
    r=analyze(req)
    assert r["selected_model"]=="coupled"
    p=r["candidate_predictions"][0]["properties"]
    assert all(v["status"]=="research_supported" for v in p.values())
    assert p["mw_g_mol"]["prediction"]>=p["mn_g_mol"]["prediction"]
    assert p["mw_g_mol"]["prediction"]/p["mn_g_mol"]["prediction"]==pytest.approx(p["dispersity"]["prediction"])


def test_rdkit_adapter_refuses_unseen_molecule():
    pytest.importorskip("rdkit")
    req=fixture(molecular=True,model="rdkit_ridge")
    req["csv"]="\n".join(l+(",smiles" if i==0 else ",C=C") for i,l in enumerate(req["csv"].splitlines()))+"\n"
    for c in req["candidates"]:
        c["smiles"]="C=C"
    req["candidates"][0]["smiles"]="C=CC"
    result=analyze(StudyRequest.model_validate(req))
    assert result["candidate_predictions"][0]["properties"]["property"]["reason"]=="unseen_molecule"


def test_role_enforcement_and_actor(tmp_path,monkeypatch):
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    path=tmp_path/"r.db"
    accounts=Accounts(Store(path).db)
    for role in ("researcher","reviewer","viewer"):
        accounts.provision(role,"fixture-only-password",role)
    app=create_app(path);headers={"X-PolySearch-Request":"1"}
    with TestClient(app) as cl:
        assert cl.get("/api/research/runs").status_code==401
        for role in ("researcher","reviewer","viewer"):
            assert cl.post("/api/auth/login",json={"username":role,"password":"fixture-only-password"},headers=headers).status_code==200
            assert cl.get("/research").status_code==200
            if role=="researcher":
                record=post(cl,fixture(),headers)
                assert record["actor"]=="researcher"
            else:
                post(cl,fixture(),headers,403)
                assert cl.get("/api/research/runs/"+record["id"]).status_code==200
            assert cl.post("/api/auth/logout",headers=headers).status_code==200


def test_integrity_detection(tmp_path):
    path=tmp_path/"r.db"
    with TestClient(create_app(path)) as cl:
        r=post(cl,fixture())
        r["input"]["name"]="changed"
        with Store(path).db.transaction() as tx:
            tx.execute("UPDATE research_runs SET payload=? WHERE id=?",(json.dumps(r),r["id"]))
        assert cl.get("/api/research/runs/"+r["id"]).status_code==409


@pytest.mark.skipif(not os.environ.get("POLYSEARCH_TEST_POSTGRES_DSN"),reason="Disposable PostgreSQL only")
def test_research_postgres_migration(tmp_path,monkeypatch):
    from polysearch.admin import migrate
    dsn=os.environ["POLYSEARCH_TEST_POSTGRES_DSN"]
    source=tmp_path/"r.db"
    with TestClient(create_app(source)) as c:
        record=post(c,fixture())
    target=Store(dsn)
    with target.db.transaction() as tx:
        for name in ("research_runs","integration_inbox","pilot_campaigns","state","login_attempts"):
            tx.execute("DELETE FROM "+name)
    assert migrate(str(source),target)["research_runs"]==1
    monkeypatch.setenv("POLYSEARCH_AUTH","required")
    monkeypatch.setenv("POLYSEARCH_ORIGIN","http://localhost:8000")
    monkeypatch.setenv("POLYSEARCH_COOKIE_SECURE","0")
    Accounts(target.db).provision("research-fixture-admin","fixture-only-password","admin",replace=True)
    headers={"X-PolySearch-Request":"1"}
    with TestClient(create_app(dsn)) as cl:
        assert cl.post("/api/auth/login",json={"username":"research-fixture-admin","password":"fixture-only-password"},headers=headers).status_code==200
        assert cl.get("/api/research/runs/"+record["id"]).json()==record
        another=post(cl,fixture(),headers)
        assert cl.get("/api/research/runs").json()["total"]==2
        assert cl.get("/api/research/runs/"+another["id"]).json()==another
