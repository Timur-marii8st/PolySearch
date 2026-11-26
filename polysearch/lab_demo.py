"""Offline disposable software demo; never connects to a laboratory or user database."""
import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from .api import create_app
from .security import auth_enabled


def run(output):
    if auth_enabled():
        raise RuntimeError("Run only in local fixture mode (POLYSEARCH_AUTH=local)")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="polysearch-fixture-") as temp:
        with TestClient(create_app(Path(temp)/"demo.sqlite3"), base_url="http://localhost") as client:
            def post(path,body,expected=200):
                response=client.post(path,json=body)
                assert response.status_code==expected,response.text
                return response.json()
            def get(path):
                response=client.get(path)
                response.raise_for_status()
                return response.json()
            post("/api/config",{"config":{"factors":[{"name":"x","lower":0,"upper":1,"unit":"a.u."}],
                "targets":[{"name":"y","lower":0.8,"scale":0.2,"unit":"a.u."}]}})
            c=post("/api/pilot/campaigns",{"name":"SOFTWARE FIXTURE — not chemical data","actor":"fixture",
                "objective":"Illustrate a workflow, not chemical performance","baseline_method":"Fixture only",
                "measurement_protocol":"Generated numbers, no physical runs","safety_review":"No physical work",
                "cost_policy":"Invented fixture accounting only","budget_per_arm":3,"confirmations_required":1},201)
            root="/api/pilot/campaigns/"+c["id"]
            def change(c,suffix,**body):
                return post(root+suffix,{"expected_revision":c["revision"],"actor":"fixture",**body})
            c=change(c,"/activate",note="Software fixture only")
            for arm in ("baseline","polysearch"):
                payload={"arm":arm,"rationale":"Software fixture only"}
                if arm=="baseline":payload["parameters"]={"x":.5}
                c=change(c,"/plans",**payload)
                pid=c["plans"][-1]["id"]
                c=change(c,"/plans/"+pid+"/review",decision="approve",note="Fixture, not a lab approval")
                order=get("/api/exchange/campaigns/"+c["id"]+"/orders/"+pid)
                result={"schema_version":"polysearch.lab.v1","source_system":"mock-lab-FIXTURE",
                    "external_id":"fixture-"+arm,"campaign_id":c["id"],"plan_id":pid,
                    "work_order_sha256":order["work_order_sha256"],"run_id":"fixture-"+arm,
                    "observed_at":datetime.now(timezone.utc).isoformat(),"actual_parameters":order["parameters"],
                    "outcomes":{"y":{"value":.9,"unit":"a.u."}},"status":"completed",
                    "evidence":"GENERATED SOFTWARE FIXTURE, NO CHEMICAL EXPERIMENT","currency":"RUB",
                    "cost_minor":10000,"elapsed_minutes":10}
                item=post("/api/exchange/inbox",{"actor":"fixture","result":result},201)
                assert post("/api/exchange/inbox",{"actor":"fixture","result":result},201)["id"]==item["id"]
                accepted=post("/api/exchange/inbox/"+item["id"]+"/review",{"actor":"fixture",
                    "expected_revision":c["revision"],"decision":"accept","note":"Fixture only"})
                assert accepted["state"]=="accepted"
                c=get(root)
                for name,doc in ((arm+"-order.json",order),(arm+"-result.json",result)):
                    (output/name).write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            c=change(c,"/close",note="End of software fixture")
            report=get(root+"/report")
            assert report["arms"]["baseline"]["attempts"]==1
            assert report["arms"]["polysearch"]["attempts"]==1
            (output/"report.json").write_text(json.dumps({"warning":"SYNTHETIC SOFTWARE FIXTURE, NOT EVIDENCE OF SAVINGS","report":report},indent=2)+"\n")
            (output/"campaign.json").write_text(json.dumps(get(root),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            (output/"README.txt").write_text("All numbers are generated fixtures; no laboratory connection, physical run or evidence of savings.\n")
            print("LAB_DEMO_OK: two fixture arms, work-order export, idempotent import, review, report, disposable DB.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.out)


if __name__=="__main__":
    main()
