"""Immutable research runs, separate from the operational workspace and pilot."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from . import __version__
from .ml.workbench import CATALOG, StudyRequest, analyze, rdkit_available
from .security import actor_name

MAX_RUNS=100
# Bound expensive training per process. Do not hold the DB lock during fitting.
SLOT=BoundedSemaphore(1)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def manifest(molecular):
    root=Path(__file__).parent
    sources=["research.py","models.py","ml/workbench.py","ml/molecular.py","ml/applicability.py","ml/data_planning.py"]
    libs=["numpy","scipy","scikit-learn","threadpoolctl","pydantic"]+(["rdkit"] if molecular else [])
    return {"polysearch":__version__,"python":platform.python_version(),"platform":platform.platform(),
            "dependencies":{p:version(p) for p in libs},
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            "model_catalog":CATALOG}


def research_router(store):
    router=APIRouter(prefix="/api/research",tags=["research"])

    @router.get("/catalog")
    def catalog():
        return {"models":CATALOG,"rdkit_available":rdkit_available(),"max_runs":MAX_RUNS,
                "status":"research_only","production_optimizer":"unchanged"}

    @router.get("/runs")
    def runs(limit:int=Query(20,ge=1,le=100),offset:int=Query(0,ge=0)):
        with store.db.transaction() as tx:
            rows=tx.execute("SELECT id,created_at,summary FROM research_runs ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",(limit,offset)).fetchall()
            total=tx.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]
        return {"items":[{"id":r[0],"created_at":r[1],**json.loads(r[2])} for r in rows],"total":total}

    @router.get("/runs/{rid}")
    def get_run(rid:str):
        with store.db.transaction() as tx:
            row=tx.execute("SELECT payload FROM research_runs WHERE id=?",(rid,)).fetchone()
        if row is None:
            raise HTTPException(404,"Research run not found")
        record=json.loads(row[0])
        recorded=record.pop("record_sha256")
        if digest(record)!=recorded:
            raise HTTPException(409,"Research record integrity mismatch")
        return {**record,"record_sha256":recorded}

    @router.post("/runs",status_code=201)
    def create(request:StudyRequest):
        if not SLOT.acquire(blocking=False):
            raise HTTPException(503,"Another research calculation is running")
        try:
            with store.db.transaction() as tx:
                if tx.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]>=MAX_RUNS:
                    raise HTTPException(409,"Research run limit reached; arrange offline retention with administrator")
            try:
                output=analyze(request)
                snapshot=request.model_dump(mode="json")
                runtime=manifest(request.molecular)
            except (ValueError,OverflowError,ArithmeticError,ValidationError) as e:
                # Avoid echoing raw data, including malformed numeric strings.
                if isinstance(e,ValueError) and not isinstance(e,ValidationError) and str(e).startswith(("could not convert","invalid literal")):
                    raise HTTPException(422,"Invalid numeric CSV field") from e
                raise HTTPException(422,"Research input or numerical calculation invalid: "+str(e)[:300]) from e
            record={"id":str(uuid4()),"created_at":datetime.now(timezone.utc).isoformat(),
                    "actor":actor_name("local-operator"),"status":"research_only","input":snapshot,
                    "input_sha256":digest(snapshot),"runtime":runtime,"runtime_sha256":digest(runtime),
                    "model_version":digest({"runtime":runtime,"input":snapshot,"selected_model":output["selected_model"]}),
                    "result":output}
            record["record_sha256"]=digest(record)
            summary={"name":request.name,"dataset_kind":request.dataset_kind,"actor":record["actor"],
                     "model":output["selected_model"],"model_version":record["model_version"]}
            with store.db.transaction() as tx:
                if tx.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]>=MAX_RUNS:
                    raise HTTPException(409,"Research run limit reached")
                tx.execute("INSERT INTO research_runs (id,created_at,summary,payload) VALUES (?,?,?,?)",
                           (record["id"],record["created_at"],json.dumps(summary,ensure_ascii=False),json.dumps(record,ensure_ascii=False,allow_nan=False)))
            return record
        finally:
            SLOT.release()

    return router
