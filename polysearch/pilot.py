"""Local, isolated pilot campaigns. Human review records are not authenticated signatures."""
import csv
import hashlib
import io
import json
import math
import platform
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import Field, FiniteFloat, StringConstraints
from threadpoolctl import threadpool_limits

from . import __version__
from .engine import Space, recommend
from .models import Name, ProjectConfig, ResultInput, StrictModel, Trial, validate_result

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Actor = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
Arm = Literal["baseline", "polysearch"]
OPEN = {"proposed", "approved"}
DONE = {"completed", "failed"}
SEARCH = {"batch_size": 1, "pool_size": 1536, "mc_samples": 128}


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid4())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def runtime_manifest():
    root = Path(__file__).parent
    code = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in ("engine.py", "models.py", "pilot.py")}
    return {"polysearch": __version__, "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "pydantic", "threadpoolctl")},
            "source_sha256": code, "blas_threads": 1}


class CreateCampaign(StrictModel):
    name: Actor
    actor: Actor
    objective: Text
    baseline_method: Text
    measurement_protocol: Text
    safety_review: Text
    cost_policy: Text
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "RUB"
    budget_per_arm: int = Field(ge=1, le=100)
    confirmations_required: int = Field(default=2, ge=1, le=5)
    include_workspace_history: bool = False
    history_provenance: Text | None = None


class Mutation(StrictModel):
    expected_revision: int = Field(ge=0)
    actor: Actor


class Decision(Mutation):
    note: Text


class Propose(Mutation):
    arm: Arm
    parameters: dict[Name, FiniteFloat] | None = None
    repeat_of: str | None = None
    rationale: Text


class Review(Decision):
    decision: Literal["approve", "reject"]


class Measurement(Mutation):
    run_id: Actor
    actual_parameters: dict[Name, FiniteFloat]
    outcomes: dict[Name, FiniteFloat] = Field(default_factory=dict)
    status: Literal["completed", "failed"]
    evidence: Text
    cost_minor: int = Field(ge=0, le=10**12)
    elapsed_minutes: Annotated[FiniteFloat, Field(ge=0, le=10**7)]
    deviation_note: Text | None = None
    failure_reason: Text | None = None


def conflict(message):
    raise HTTPException(409, message)


def checked(config, parameters, outcomes=None, status="failed"):
    try:
        r = ResultInput(parameters=parameters, outcomes=outcomes or {}, status=status)
        validate_result(config, r)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def append_event(campaign, kind, actor, **data):
    events = campaign["events"]
    event = {"seq": len(events) + 1, "at": now(), "kind": kind, "actor": actor,
             "data": data, "previous_sha256": events[-1]["sha256"] if events else None}
    event["sha256"] = fingerprint(event)
    events.append(event)
    campaign["revision"] += 1


def verify_events(events):
    previous = None
    for i, event in enumerate(events, 1):
        unsigned = {k: v for k, v in event.items() if k != "sha256"}
        if event["seq"] != i or event["previous_sha256"] != previous or fingerprint(unsigned) != event["sha256"]:
            return False
        previous = event["sha256"]
    return True


class CampaignStore:
    """Separate table; workspace reset never erases a pilot campaign."""
    def __init__(self, path, lock):
        self.path, self.lock = path, lock
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS pilot_campaigns (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def get(self, campaign_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload FROM pilot_campaigns WHERE id=?", (campaign_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Campaign not found")
        return json.loads(row[0])

    def save(self, c):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO pilot_campaigns VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (c["id"], json.dumps(c, ensure_ascii=False, allow_nan=False)))

    def list(self):
        with closing(sqlite3.connect(self.path)) as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM pilot_campaigns ORDER BY rowid DESC")]


def completed_goal(config, plan):
    m = plan.get("measurement")
    return bool(m and m["status"] == "completed" and not m["protocol_deviation"] and
                all((t.lower is None or m["outcomes"][t.name] >= t.lower)
                    and (t.upper is None or m["outcomes"][t.name] <= t.upper)
                    for t in config.targets))


def arm_attempts(c, arm):
    return [p for p in c["plans"] if p["arm"] == arm and p["state"] in DONE]


def arm_history(c, arm):
    history = [Trial.model_validate(t) for t in c["initial_history"]]
    for p in arm_attempts(c, arm):
        m = p["measurement"]
        if m["protocol_deviation"]:
            continue
        history.append(Trial(id=p["id"], parameters=m["actual_parameters"], outcomes=m["outcomes"],
                             status=m["status"], source="manual", created_at=m["at"],
                             note="Pilot evidence: " + m["evidence"][:1000]))
    return history


def arm_summary(c, arm):
    config = ProjectConfig.model_validate(c["config"])
    attempts = arm_attempts(c, arm)
    successes = {}
    first = None
    cost, minutes = 0, 0.0
    for i, p in enumerate(attempts, 1):
        m = p["measurement"]
        cost += m["cost_minor"]
        minutes += m["elapsed_minutes"]
        if completed_goal(config, p):
            key = tuple(sorted(m["actual_parameters"].items()))
            successes[key] = successes.get(key, 0) + 1
            if first is None and successes[key] >= c["protocol"]["confirmations_required"]:
                first = {"attempts": i, "cost_minor": cost, "elapsed_minutes": minutes,
                         "parameters": m["actual_parameters"]}
    return {"attempts": len(attempts), "remaining": c["protocol"]["budget_per_arm"] - len(attempts),
            "failed": sum(p["state"] == "failed" for p in attempts),
            "deviations": sum(p["measurement"]["protocol_deviation"] for p in attempts),
            "rejected_plans": sum(p["arm"] == arm and p["state"] == "rejected" for p in c["plans"]),
            "cancelled_plans": sum(p["arm"] == arm and p["state"] == "cancelled" for p in c["plans"]),
            "total_cost_minor": cost, "total_elapsed_minutes": minutes,
            "goal_confirmed": first is not None, "first_confirmed_goal": first}


def report(c):
    arms = {a: arm_summary(c, a) for a in ("baseline", "polysearch")}
    a, b = arms["baseline"]["first_confirmed_goal"], arms["polysearch"]["first_confirmed_goal"]
    comparison = None
    if a is not None and b is not None:
        comparison = {k + "_difference_baseline_minus_polysearch": a[k] - b[k]
                      for k in ("attempts", "cost_minor", "elapsed_minutes")}
        comparison["cost_reduction_fraction"] = (a["cost_minor"] - b["cost_minor"]) / a["cost_minor"] if a["cost_minor"] else None
    return {"campaign_id": c["id"], "state": c["state"], "currency": c["protocol"]["currency"],
            "arms": arms, "comparison_to_confirmed_goal": comparison,
            "initial_history_count": len(c["initial_history"]),
            "event_chain_valid": verify_events(c["events"]),
            "interpretation": "Descriptive comparison of recorded attempts only; common historical data are not counted "
                              "in prospective arm costs. No causal or industrial efficiency claim. "
                              "A missing confirmation is not assigned an invented cost or saving."}


def pilot_router(workspace):
    router = APIRouter(prefix="/api/pilot", tags=["pilot"])
    store = CampaignStore(workspace.path, workspace.lock)

    def mutable(cid, request, states=("active",)):
        c = store.get(cid)
        if c["revision"] != request.expected_revision:
            conflict("Stale campaign revision; refresh before changing it")
        if c["state"] not in states:
            conflict("Campaign is not in the required state")
        return c

    def plan_by_id(c, pid):
        for p in c["plans"]:
            if p["id"] == pid:
                return p
        raise HTTPException(404, "Plan not found")

    @router.get("/campaigns")
    def list_campaigns():
        with store.lock:
            return [{"id": c["id"], "name": c["protocol"]["name"], "state": c["state"],
                     "revision": c["revision"], "created_at": c["created_at"]} for c in store.list()]

    @router.post("/campaigns", status_code=201)
    def create(request: CreateCampaign):
        with store.lock:
            source = workspace.load()
            if not source["config"] or source["config"]["mode"] != "real":
                conflict("Configure a real workspace before creating a pilot")
            config = ProjectConfig.model_validate(source["config"])
            if request.confirmations_required > request.budget_per_arm:
                raise HTTPException(422, "Confirmation count exceeds the budget")
            history = []
            if request.include_workspace_history:
                if not request.history_provenance:
                    raise HTTPException(422, "Document common historical data provenance")
                history = source["trials"]
                for row in history:
                    t = Trial.model_validate(row)
                    if t.source == "synthetic":
                        raise HTTPException(422, "Synthetic trials cannot be pilot history")
                    checked(config, t.parameters, t.outcomes, t.status)
            protocol = request.model_dump(mode="json")
            c = {"id": uid(), "created_at": now(), "state": "draft", "revision": 0,
                 "protocol": protocol, "config": config.model_dump(mode="json"), "initial_history": history,
                 "workspace_revision": source["revision"], "plans": [], "events": []}
            c["protocol_sha256"] = fingerprint({"protocol": protocol, "config": c["config"], "initial_history": history})
            append_event(c, "created", request.actor, protocol_sha256=c["protocol_sha256"])
            store.save(c)
            return c

    @router.get("/campaigns/{cid}")
    def get_campaign(cid: str):
        with store.lock:
            return store.get(cid)

    @router.post("/campaigns/{cid}/activate")
    def activate(cid: str, request: Decision):
        with store.lock:
            c = mutable(cid, request, ("draft",))
            c["state"] = "active"
            append_event(c, "protocol_approved", request.actor, note=request.note,
                         protocol_sha256=c["protocol_sha256"])
            store.save(c)
            return c

    @router.post("/campaigns/{cid}/plans")
    def propose(cid: str, request: Propose):
        with store.lock:
            c = mutable(cid, request)
            arm = request.arm
            if any(p["arm"] == arm and p["state"] in OPEN for p in c["plans"]):
                conflict("Resolve the open plan in this arm before proposing another")
            if len(arm_attempts(c, arm)) >= c["protocol"]["budget_per_arm"]:
                conflict("Experiment budget exhausted")
            cfg = ProjectConfig.model_validate(c["config"])
            snapshot, candidate = None, None
            if request.repeat_of:
                original = plan_by_id(c, request.repeat_of)
                if original["arm"] != arm or original["state"] != "completed" or original["measurement"]["protocol_deviation"]:
                    raise HTTPException(422, "Repeat requires a completed, compliant trial in this arm")
                if request.parameters is not None:
                    raise HTTPException(422, "Repeat uses the original exact parameters")
                parameters = original["measurement"]["actual_parameters"]
                method = "confirmation_repeat"
            elif arm == "baseline":
                if request.parameters is None:
                    raise HTTPException(422, "Baseline requires explicit parameters")
                parameters = request.parameters
                method = "manual_baseline"
            else:
                if request.parameters is not None:
                    raise HTTPException(422, "PolySearch parameters must come from the model")
                history = arm_history(c, arm)
                snapshot = {"runtime": runtime_manifest(), "config": c["config"],
                            "history": [t.model_dump(mode="json") for t in history], "search": SEARCH}
                try:
                    with threadpool_limits(limits=1):
                        result = recommend(cfg, history, **SEARCH)
                except ValueError as e:
                    raise HTTPException(422, str(e)) from e
                candidate = result["recommendations"][0]
                parameters = candidate["parameters"]
                snapshot["model_report"] = result["report"]
                method = "independent_gp" if candidate["prediction"] else "initial_design"
            checked(cfg, parameters)
            if not request.repeat_of and any(p["measurement"]["actual_parameters"] == parameters
                                             for p in arm_attempts(c, arm)):
                raise HTTPException(422, "Use explicit repeat_of for repeated conditions")
            plan = {"id": uid(), "arm": arm, "state": "proposed", "created_at": now(),
                    "parameters": parameters, "method": method, "repeat_of": request.repeat_of,
                    "rationale": request.rationale, "candidate": candidate, "snapshot": snapshot,
                    "model_version": fingerprint(snapshot) if snapshot else None, "review": None, "measurement": None}
            c["plans"].append(plan)
            append_event(c, "plan_proposed", request.actor, plan_id=plan["id"], arm=arm,
                         plan_sha256=fingerprint(plan), model_version=plan["model_version"])
            store.save(c)
            return c

    @router.post("/campaigns/{cid}/plans/{pid}/review")
    def review(cid: str, pid: str, request: Review):
        with store.lock:
            c = mutable(cid, request)
            p = plan_by_id(c, pid)
            if p["state"] != "proposed":
                conflict("Only a proposed plan can be reviewed")
            p["state"] = "approved" if request.decision == "approve" else "rejected"
            p["review"] = {"actor": request.actor, "at": now(), "decision": request.decision, "note": request.note}
            append_event(c, "plan_reviewed", request.actor, plan_id=pid, decision=request.decision, note=request.note)
            store.save(c)
            return c

    @router.post("/campaigns/{cid}/plans/{pid}/measurement")
    def measure(cid: str, pid: str, request: Measurement):
        with store.lock:
            c = mutable(cid, request)
            p = plan_by_id(c, pid)
            if p["state"] != "approved":
                conflict("Only an approved plan can receive a measurement")
            if any(q.get("measurement") and q["measurement"]["run_id"] == request.run_id for q in c["plans"]):
                conflict("Run ID already recorded in this campaign")
            if request.status == "failed" and not request.failure_reason:
                raise HTTPException(422, "Document the failure reason")
            cfg = ProjectConfig.model_validate(c["config"])
            # Validate outcomes against the planned configuration. Preserve actual
            # out-of-range parameters as evidence of deviations rather than hiding them.
            checked(cfg, p["parameters"], request.outcomes, request.status)
            if set(request.actual_parameters) != {f.name for f in cfg.factors}:
                raise HTTPException(422, "Actual parameter names must match the configuration")
            deviation = request.actual_parameters != p["parameters"] or bool(request.deviation_note)
            if deviation and not request.deviation_note:
                raise HTTPException(422, "Document the deviation from the approved parameters")
            m = request.model_dump(mode="json", exclude={"expected_revision"})
            m.update(at=now(), protocol_deviation=deviation)
            p["measurement"] = m
            p["state"] = request.status
            append_event(c, "measurement_recorded", request.actor, plan_id=pid, measurement_sha256=fingerprint(m))
            store.save(c)
            return c

    @router.post("/campaigns/{cid}/plans/{pid}/cancel")
    def cancel(cid: str, pid: str, request: Decision):
        with store.lock:
            c = mutable(cid, request)
            p = plan_by_id(c, pid)
            if p["state"] != "approved":
                conflict("Only an approved, unexecuted plan can be cancelled")
            p["state"] = "cancelled"
            p["cancellation"] = {"actor": request.actor, "at": now(), "note": request.note}
            append_event(c, "plan_cancelled", request.actor, plan_id=pid, note=request.note)
            store.save(c)
            return c

    @router.post("/campaigns/{cid}/close")
    def close(cid: str, request: Decision):
        with store.lock:
            c = mutable(cid, request, ("draft", "active"))
            if any(p["state"] in OPEN for p in c["plans"]):
                conflict("Resolve pending plans before closing")
            c["state"] = "closed"
            append_event(c, "campaign_closed", request.actor, note=request.note)
            store.save(c)
            return c

    @router.get("/campaigns/{cid}/report")
    def get_report(cid: str):
        with store.lock:
            return report(store.get(cid))

    @router.get("/campaigns/{cid}/export.json")
    def export(cid: str):
        with store.lock:
            c = store.get(cid)
            return {"campaign": c, "report": report(c)}

    @router.get("/campaigns/{cid}/attempts.csv")
    def attempts_csv(cid: str):
        with store.lock:
            c = store.get(cid)
            out = io.StringIO()
            fields = ["id", "run_id", "arm", "status", "approved_parameters", "actual_parameters", "outcomes",
                      "cost_minor", "currency", "elapsed_minutes", "protocol_deviation", "evidence", "model_version"]
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            # Escape spreadsheet formula prefixes in arbitrary text.
            def safe(value):
                return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else value
            for p in c["plans"]:
                if p["state"] not in DONE:
                    continue
                m = p["measurement"]
                writer.writerow({"id": p["id"], "run_id": safe(m["run_id"]), "arm": p["arm"], "status": p["state"],
                                 "approved_parameters": json.dumps(p["parameters"]),
                                 "actual_parameters": json.dumps(m["actual_parameters"]),
                                 "outcomes": json.dumps(m["outcomes"]),
                                 "cost_minor": m["cost_minor"], "currency": c["protocol"]["currency"],
                                 "elapsed_minutes": m["elapsed_minutes"], "protocol_deviation": m["protocol_deviation"],
                                 "evidence": safe(m["evidence"]), "model_version": p["model_version"] or ""})
            return Response(out.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="pilot-attempts.csv"'})

    @router.get("/campaigns/{cid}/plans/{pid}/replay")
    def replay(cid: str, pid: str):
        with store.lock:
            c = store.get(cid)
            p = plan_by_id(c, pid)
            snapshot = p["snapshot"]
            if not snapshot:
                raise HTTPException(422, "This plan has no model snapshot")
            integrity = fingerprint(snapshot) == p["model_version"]
            same_runtime = snapshot["runtime"] == runtime_manifest()
            if not integrity or not same_runtime:
                return {"snapshot_integrity": integrity, "same_runtime": same_runtime, "reproduced": None}
            with threadpool_limits(limits=1):
                result = recommend(ProjectConfig.model_validate(snapshot["config"]),
                                   [Trial.model_validate(t) for t in snapshot["history"]], **snapshot["search"])
            candidate = result["recommendations"][0]
            expected = p["candidate"]
            # Exact content hash can differ from floating-point variation; report both.
            exact = fingerprint(candidate) == fingerprint(expected)
            keys = list(expected["parameters"])
            reproduced = (candidate["acquisition"] == expected["acquisition"] and
                          candidate["training_trial_ids"] == expected["training_trial_ids"] and
                          bool(np.allclose([candidate["parameters"][k] for k in keys],
                                           [expected["parameters"][k] for k in keys], rtol=1e-9, atol=1e-10)) and
                          math.isclose(candidate["score"], expected["score"], rel_tol=1e-8, abs_tol=1e-10))
            return {"snapshot_integrity": integrity, "same_runtime": same_runtime,
                    "reproduced": reproduced, "exact_candidate_hash_match": exact,
                    "recalculated_candidate": candidate}

    return router
