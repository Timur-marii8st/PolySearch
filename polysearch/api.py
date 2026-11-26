"""Local single-project FastAPI service with SQLite persistence."""
import csv
import io
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from scipy.stats import qmc

from .demo import configuration, evaluate
from .engine import Space, recommend
from .models import ProjectConfig, ResultInput, Trial, StrictModel, validate_result


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")

    def load(self):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"config": None, "trials": [], "recommendations": [],
                                              "report": None, "revision": 0}

    def save(self, state):
        payload = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO state VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (payload,))


class ConfigRequest(StrictModel):
    config: ProjectConfig
    reset: bool = False


class DemoRequest(StrictModel):
    seed: int = Field(default=42, ge=0, le=2**32-1)
    n_initial: int = Field(default=12, ge=0, le=500)
    reset: bool = False


class RecommendRequest(StrictModel):
    batch_size: int = Field(default=1, ge=1, le=3)


class SimulateRequest(StrictModel):
    recommendation_id: str


class CSVRequest(StrictModel):
    csv: str = Field(max_length=2_000_000)


def create_app(db_path=None):
    app = FastAPI(title="PolySearch", version="0.1.0")
    store = Store(db_path or os.environ.get("POLYSEARCH_DB", "polysearch.sqlite3"))
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def workspace():
        return FileResponse(static / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    def current():
        state = store.load()
        if not state["config"]:
            raise HTTPException(409, "Initialize a project first")
        return state, ProjectConfig.model_validate(state["config"])

    def invalidate(state):
        state["revision"] += 1
        state["recommendations"] = []
        state["report"] = None

    def make_trial(config, result, source):
        validate_result(config, result)
        return Trial(**result.model_dump(), id=str(uuid4()), source=source,
                     created_at=datetime.now(timezone.utc).isoformat()).model_dump(mode="json")

    def checked_trial(state, config, result, source):
        if result.recommendation_id:
            candidate = next((r for r in state["recommendations"]
                              if r["id"] == result.recommendation_id), None)
            if candidate is None:
                raise HTTPException(409, "Stale or unknown recommendation")
            if result.parameters != candidate["parameters"]:
                raise HTTPException(422, "Parameters do not match recommendation")
        try:
            return make_trial(config, result, source)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    def initialize(config, reset, results):
        previous = store.load()
        if previous["config"] and not reset:
            raise HTTPException(409, "Project exists; explicit reset required")
        try:
            Space(config)
            trials = [make_trial(config, r, "synthetic") for r in results]
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        state = {"config": config.model_dump(mode="json"), "trials": trials,
                 "recommendations": [], "report": None, "revision": previous["revision"] + 1}
        store.save(state)
        return state

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/state")
    @app.get("/api/export.json")
    def state():
        with store.lock:
            return store.load()

    @app.post("/api/config")
    def config(payload: ConfigRequest):
        if payload.config.mode != "real":
            raise HTTPException(422, "Use /api/demo to initialize demo mode")
        with store.lock:
            return initialize(payload.config, payload.reset, [])

    @app.post("/api/demo")
    def demo(payload: DemoRequest):
        cfg = configuration(payload.seed)
        samples = qmc.LatinHypercube(3, seed=payload.seed).random(payload.n_initial)
        results = [evaluate(dict(zip([f.name for f in cfg.factors], x.tolist()))) for x in samples]
        with store.lock:
            return initialize(cfg, payload.reset, results)

    @app.post("/api/recommend")
    def next_experiments(payload: RecommendRequest):
        with store.lock:
            state, cfg = current()
            try:
                result = recommend(cfg, [Trial.model_validate(t) for t in state["trials"]],
                                   batch_size=payload.batch_size)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            for candidate in result["recommendations"]:
                candidate.update(id=str(uuid4()), revision=state["revision"])
            state.update(result)
            store.save(state)
            return result

    @app.post("/api/results")
    def result(payload: ResultInput):
        with store.lock:
            state, cfg = current()
            trial = checked_trial(state, cfg, payload, "manual")
            state["trials"].append(trial)
            invalidate(state)
            store.save(state)
            return trial

    @app.post("/api/simulate")
    def simulate(payload: SimulateRequest):
        with store.lock:
            state, cfg = current()
            if cfg.mode != "demo":
                raise HTTPException(409, "Simulator is disabled in real mode")
            candidate = next((r for r in state["recommendations"]
                              if r["id"] == payload.recommendation_id), None)
            if candidate is None:
                raise HTTPException(409, "Stale or unknown recommendation")
            result = evaluate(candidate["parameters"])
            result.recommendation_id = payload.recommendation_id
            trial = checked_trial(state, cfg, result, "synthetic")
            state["trials"].append(trial)
            invalidate(state)
            store.save(state)
            return trial

    @app.post("/api/import.csv")
    def import_csv(payload: CSVRequest):
        with store.lock:
            state, cfg = current()
            factors = [f.name for f in cfg.factors]
            targets = [t.name for t in cfg.targets]
            reader = csv.DictReader(io.StringIO(payload.csv.lstrip("\ufeff")), strict=True)
            try:
                header = reader.fieldnames or []
            except csv.Error as error:
                raise HTTPException(422, f"Invalid CSV header: {error}") from error
            required = set(factors + targets)
            if (len(header) != len(set(header)) or not required <= set(header)
                    or set(header) - required - {"status", "role", "note"}):
                raise HTTPException(422, "Invalid CSV header")
            pending = []
            try:
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError("Incorrect number of CSV fields")
                    result = ResultInput(parameters={k: float(row[k]) for k in factors},
                                         outcomes={k: float(row[k]) for k in targets if row[k].strip()},
                                         status=row.get("status") or "completed",
                                         role=row.get("role") or "train", note=row.get("note") or "")
                    pending.append(make_trial(cfg, result, "csv"))
                if not pending:
                    raise ValueError("CSV contains no records")
            except (ValueError, csv.Error) as error:
                raise HTTPException(422, f"CSV line {reader.line_num}: {error}") from error
            state["trials"].extend(pending)
            invalidate(state)
            store.save(state)
            return {"imported": len(pending), "revision": state["revision"]}

    return app
