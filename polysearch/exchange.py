"""Versioned, human-gated LIMS/ELN file exchange. No equipment control or outbound calls."""
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import AwareDatetime, Field, FiniteFloat, StringConstraints

from .models import Name, ProjectConfig, StrictModel
from .pilot import (Actor, Text, CampaignStore, Measurement, Mutation, fingerprint,
                    now, uid, conflict, validate_measurement, record_measurement)
from .security import actor_name

VERSION = "polysearch.lab.v1"
ExternalID = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
SHA = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Quantity(StrictModel):
    value: Annotated[FiniteFloat, Field(strict=True)]
    unit: str = Field(max_length=160)


class LabResult(StrictModel):
    schema_version: Literal["polysearch.lab.v1"]
    source_system: ExternalID
    external_id: ExternalID
    campaign_id: ExternalID
    plan_id: ExternalID
    work_order_sha256: SHA
    run_id: Actor
    observed_at: AwareDatetime
    actual_parameters: dict[Name, Quantity] = Field(max_length=12)
    outcomes: dict[Name, Quantity] = Field(default_factory=dict, max_length=8)
    status: Literal["completed", "failed"]
    evidence: Text
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    cost_minor: int = Field(strict=True, ge=0, le=10**12)
    elapsed_minutes: Annotated[FiniteFloat, Field(strict=True, ge=0, le=10**7)]
    deviation_note: Text | None = None
    failure_reason: Text | None = None


class Submit(StrictModel):
    actor: Actor
    result: LabResult


class Review(Mutation):
    decision: Literal["accept", "reject"]
    note: Text


class Inbox:
    def __init__(self, db):
        self.db = db

    def get(self, iid):
        with self.db.transaction() as tx:
            row = tx.execute("SELECT payload FROM integration_inbox WHERE id=?", (iid,)).fetchone()
        if row is None:
            raise HTTPException(404, "Inbox item not found")
        return json.loads(row[0])

    def lookup(self, source, external_id):
        with self.db.transaction() as tx:
            row = tx.execute("SELECT payload FROM integration_inbox WHERE source_system=? AND external_id=?",
                             (source, external_id)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, item):
        with self.db.transaction() as tx:
            tx.execute("INSERT INTO integration_inbox (id,source_system,external_id,payload) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (item["id"], item["result"]["source_system"], item["result"]["external_id"],
                        json.dumps(item, ensure_ascii=False, allow_nan=False)))

    def list(self, campaign_id, limit, offset):
        with self.db.transaction() as tx:
            # JSON text is portable across legacy SQLite and PostgreSQL. Filter
            # in application code for pilot-sized datasets.
            items = [json.loads(r[0]) for r in tx.execute("SELECT payload FROM integration_inbox").fetchall()]
        if campaign_id:
            items = [r for r in items if r["result"]["campaign_id"] == campaign_id]
        items.sort(key=lambda x: (x["received_at"], x["id"]), reverse=True)
        return {"total": len(items), "items": items[offset:offset + limit]}


def approved_plan(campaign, plan_id):
    if campaign["state"] != "active":
        conflict("Campaign is not active")
    for p in campaign["plans"]:
        if p["id"] == plan_id:
            if p["state"] != "approved" or not p["review"] or p["review"]["decision"] != "approve":
                conflict("Only an approved, unexecuted plan can be exchanged")
            return p
    raise HTTPException(404, "Plan not found")


def work_order(c, p):
    config = ProjectConfig.model_validate(c["config"])
    document = {
        "schema_version": VERSION, "kind": "approved_work_order",
        "campaign_id": c["id"], "plan_id": p["id"], "arm": p["arm"],
        "protocol_sha256": c["protocol_sha256"],
        "approved_at": p["review"]["at"], "approved_by": p["review"]["actor"],
        "approval_note": p["review"]["note"],
        "parameters": {f.name: {"value": p["parameters"][f.name], "unit": f.unit} for f in config.factors},
        "targets": [t.model_dump(mode="json") for t in config.targets],
        "constraints": [x.model_dump(mode="json") for x in config.constraints],
        "currency": c["protocol"]["currency"],
        "measurement_protocol": c["protocol"]["measurement_protocol"],
        "safety_review": c["protocol"]["safety_review"],
        "cost_policy": c["protocol"]["cost_policy"],
        "model_version": p["model_version"],
    }
    return {**document, "work_order_sha256": fingerprint(document)}


def as_measurement(c, p, result, actor):
    order = work_order(c, p)
    if result.work_order_sha256 != order["work_order_sha256"]:
        conflict("Work-order fingerprint mismatch")
    if result.currency != order["currency"]:
        raise HTTPException(422, "Currency must match the approved protocol")
    config = ProjectConfig.model_validate(c["config"])
    factors = {f.name: f.unit for f in config.factors}
    targets = {t.name: t.unit for t in config.targets}
    if set(result.actual_parameters) != set(factors):
        raise HTTPException(422, "Actual factor names do not match the work order")
    if not set(result.outcomes) <= set(targets):
        raise HTTPException(422, "Unknown target")
    for quantities, units in ((result.actual_parameters, factors), (result.outcomes, targets)):
        for key, q in quantities.items():
            if q.unit != units[key]:
                raise HTTPException(422, "Unit mismatch: " + key)
    approved_at = datetime.fromisoformat(order["approved_at"])
    if result.observed_at < approved_at or result.observed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise HTTPException(422, "Measurement timestamp must follow approval and not be in the future")
    request = Measurement(
        expected_revision=c["revision"], actor=actor, run_id=result.run_id,
        actual_parameters={k: q.value for k, q in result.actual_parameters.items()},
        outcomes={k: q.value for k, q in result.outcomes.items()}, status=result.status,
        evidence=result.evidence, cost_minor=result.cost_minor, elapsed_minutes=result.elapsed_minutes,
        deviation_note=result.deviation_note, failure_reason=result.failure_reason)
    validate_measurement(c, p, request)
    return request


def exchange_router(workspace):
    router = APIRouter(prefix="/api/exchange", tags=["laboratory exchange"])
    store = CampaignStore(workspace.db)
    inbox = Inbox(workspace.db)

    @router.get("/campaigns/{cid}/orders/{pid}")
    def order(cid: str, pid: str):
        with store.lock:
            c = store.get(cid)
            return work_order(c, approved_plan(c, pid))

    @router.get("/inbox")
    def items(campaign_id: str | None = None, limit: int = Query(50, ge=1, le=100),
              offset: int = Query(0, ge=0)):
        with store.lock:
            return inbox.list(campaign_id, limit, offset)

    @router.get("/inbox/{iid}")
    def item(iid: str):
        with store.lock:
            return inbox.get(iid)

    @router.post("/inbox", status_code=201)
    def stage(request: Submit):
        result = request.result
        payload = result.model_dump(mode="json")
        sha = fingerprint(payload)
        actor = actor_name(request.actor)
        with store.lock:
            existing = inbox.lookup(result.source_system, result.external_id)
            if existing:
                if existing["payload_sha256"] != sha:
                    conflict("External ID already used for a different payload")
                return existing
            c = store.get(result.campaign_id)
            p = approved_plan(c, result.plan_id)
            as_measurement(c, p, result, actor)
            item = {"id": uid(), "state": "pending", "received_at": now(),
                    "received_by": actor, "result": payload, "payload_sha256": sha,
                    "decision": None}
            inbox.save(item)
            return item

    @router.post("/inbox/{iid}/review")
    def review(iid: str, request: Review):
        actor = actor_name(request.actor)
        with store.lock:
            item = inbox.get(iid)
            if item["state"] != "pending":
                desired = "accepted" if request.decision == "accept" else "rejected"
                if item["state"] == desired:
                    return item
                conflict("Inbox item already has a final decision")
            if fingerprint(item["result"]) != item["payload_sha256"]:
                conflict("Inbox payload integrity mismatch")
            result = LabResult.model_validate(item["result"])
            c = store.get(result.campaign_id)
            if c["revision"] != request.expected_revision:
                conflict("Stale campaign revision; refresh before reviewing")
            decision = {"actor": actor, "at": now(), "note": request.note,
                        "decision": request.decision}
            if request.decision == "accept":
                p = approved_plan(c, result.plan_id)
                measurement = as_measurement(c, p, result, actor)
                provenance = {"schema_version": VERSION, "inbox_id": iid,
                              "source_system": result.source_system, "external_id": result.external_id,
                              "payload_sha256": item["payload_sha256"],
                              "work_order_sha256": result.work_order_sha256,
                              "observed_at": result.observed_at.isoformat(),
                              "received_by": item["received_by"], "review": decision}
                record_measurement(c, p, measurement, provenance=provenance)
                store.save(c)
                item["state"] = "accepted"
            else:
                # Rejecting a staged result does not invent an experiment or modify
                # an already closed campaign. The decision stays in the inbox.
                item["state"] = "rejected"
            item["decision"] = decision
            inbox.save(item)
            return item

    return router
