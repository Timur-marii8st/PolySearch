from fastapi.testclient import TestClient

from polysearch.api import create_app
from polysearch.pilot import verify_events

CFG = {"factors": [{"name": "x", "lower": 0, "upper": 1}],
       "targets": [{"name": "y", "lower": .8, "scale": .2}]}
PROTOCOL = dict(name="fixture", actor="researcher", objective="Test target", baseline_method="Manual sequential",
                measurement_protocol="Independent synthetic test fixture only", safety_review="No physical work",
                cost_policy="Count all attempts, including failures", budget_per_arm=3, confirmations_required=2)


def post(client, path, payload, status=200):
    response = client.post(path, json=payload)
    assert response.status_code == status, response.text
    return response.json()


def create(client, **overrides):
    post(client, "/api/config", {"config": CFG})
    c = post(client, "/api/pilot/campaigns", {**PROTOCOL, **overrides}, 201)
    return c, "/api/pilot/campaigns/" + c["id"]


def change(client, path, c, suffix, **payload):
    return post(client, path + suffix, {"expected_revision": c["revision"], "actor": "reviewer", **payload})


def activate(client, path, c):
    return change(client, path, c, "/activate", note="Fixture approval")


def plan(client, path, c, arm, **kwargs):
    return change(client, path, c, "/plans", arm=arm, rationale="Predefined test plan", **kwargs)


def approve(client, path, c):
    return change(client, path, c, "/plans/" + c["plans"][-1]["id"] + "/review",
                  decision="approve", note="Fixture review")


def measure(client, path, c, y=.9, **overrides):
    p = c["plans"][-1]
    payload = dict(actual_parameters=p["parameters"], outcomes={"y": y}, status="completed",
                   evidence="Test fixture", cost_minor=10000, elapsed_minutes=30)
    payload.update(overrides)
    return change(client, path, c, "/plans/" + p["id"] + "/measurement", **payload)


def test_campaign_isolated_and_persistent(tmp_path):
    db = tmp_path / "pilot.db"
    with TestClient(create_app(db)) as cl:
        c, path = create(cl)
        c = activate(cl, path, c)
        c = plan(cl, path, c, "baseline", parameters={"x": .5})
        c = approve(cl, path, c)
        c = measure(cl, path, c)
        post(cl, "/api/demo", {"reset": True, "n_initial": 3})
        assert cl.get(path).json() == c
    with TestClient(create_app(db)) as cl:
        assert cl.get(path).json() == c
        assert verify_events(c["events"])
        assert cl.get(path + "/report").json()["arms"]["baseline"]["attempts"] == 1


def test_protocol_gate_revision_and_approval(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        c, path = create(cl)
        post(cl, path + "/plans", {"expected_revision": c["revision"], "actor":"x", "arm":"baseline",
                                  "parameters":{"x": .5}, "rationale":"test"}, 409)
        c = activate(cl, path, c)
        c = plan(cl, path, c, "baseline", parameters={"x": .5})
        p = c["plans"][-1]
        post(cl, path + "/plans/" + p["id"] + "/measurement", {
            "expected_revision":c["revision"], "actor":"x","actual_parameters":{"x":.5},
            "outcomes":{"y":.9},"status":"completed","evidence":"x","cost_minor":1,"elapsed_minutes":1},409)
        post(cl, path + "/plans/" + p["id"] + "/review", {"expected_revision":0,"actor":"x",
                                                         "decision":"approve","note":"x"},409)
        c = approve(cl, path, c)
        c = measure(cl, path, c)
        assert c["plans"][-1]["state"] == "completed"


def test_repeat_confirmation_and_costs(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        c, path = create(cl)
        c = activate(cl, path, c)
        c = measure(cl, path, approve(cl, path, plan(cl, path, c, "baseline", parameters={"x":.1})),
                    y=.2, status="failed", failure_reason="Fixture failure", cost_minor=5000)
        c = measure(cl, path, approve(cl, path, plan(cl, path, c, "baseline", parameters={"x":.8})))
        assert not cl.get(path+"/report").json()["arms"]["baseline"]["goal_confirmed"]
        original = c["plans"][-1]["id"]
        c = measure(cl, path, approve(cl, path, plan(cl, path, c, "baseline", repeat_of=original)))
        r = cl.get(path+"/report").json()
        assert r["arms"]["baseline"]["first_confirmed_goal"]["attempts"] == 3
        assert r["arms"]["baseline"]["first_confirmed_goal"]["cost_minor"] == 25000
        assert r["comparison_to_confirmed_goal"] is None
        post(cl,path+"/plans",{"expected_revision":c["revision"],"actor":"x","arm":"baseline",
                              "parameters":{"x":.3},"rationale":"extra"},409)


def test_snapshot_replay_and_arm_isolation(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        c, path = create(cl)
        c = activate(cl, path, c)
        c = measure(cl,path,approve(cl,path,plan(cl,path,c,"baseline",parameters={"x":.5})))
        c = plan(cl,path,c,"polysearch")
        p = c["plans"][-1]
        assert p["snapshot"]["history"] == []
        result = cl.get(path+"/plans/"+p["id"]+"/replay").json()
        assert result["snapshot_integrity"] and result["reproduced"]
        assert cl.get(path+"/export.json").json()["campaign"]["plans"][-1]["model_version"]


def test_deviations_count_but_do_not_train(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        c,path = create(cl)
        c=activate(cl,path,c)
        c=approve(cl,path,plan(cl,path,c,"polysearch"))
        c=measure(cl,path,c,actual_parameters={"x":2},deviation_note="Fixture deviation",cost_minor=321)
        r=cl.get(path+"/report").json()["arms"]["polysearch"]
        assert r["attempts"]==1 and r["deviations"]==1 and r["total_cost_minor"]==321
        c=plan(cl,path,c,"polysearch")
        assert c["plans"][-1]["snapshot"]["history"]==[]


def test_reject_and_close(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        c,path=create(cl)
        c=activate(cl,path,c)
        c=plan(cl,path,c,"baseline",parameters={"x":.5})
        post(cl,path+"/close",{"expected_revision":c["revision"],"actor":"x","note":"close"},409)
        c=change(cl,path,c,"/plans/"+c["plans"][-1]["id"]+"/review",decision="reject",note="not approved")
        assert cl.get(path+"/report").json()["arms"]["baseline"]["attempts"]==0
        c=change(cl,path,c,"/close",note="end fixture")
        assert c["state"]=="closed"
        assert cl.get(path+"/attempts.csv").status_code==200


def test_demo_cannot_create_pilot(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        post(cl,"/api/demo",{"n_initial":3})
        post(cl,"/api/pilot/campaigns",PROTOCOL,409)


def test_protocol_confirmation_budget(tmp_path):
    with TestClient(create_app(tmp_path / "p.db")) as cl:
        post(cl,"/api/config",{"config":CFG})
        post(cl,"/api/pilot/campaigns",{**PROTOCOL,"budget_per_arm":1},422)
