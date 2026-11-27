"""Unified, bounded research model interface for user-supplied grouped measurements.

No code execution, artifact deserialization, equipment control, or automatic
promotion to the pilot optimizer. Selection uses proper training groups only.
"""
import csv
import io
import math
from collections import Counter
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, FiniteFloat, StringConstraints, model_validator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from ..models import ProjectConfig, ResultInput, StrictModel, validate_result
from .molecular import TARGETS, PropertyModel, molecular_features, conformal_quantile
from .applicability import BoundedMolecularModel
from .data_planning import minimum_calibration_groups

MAX_ROWS=600
MAX_GP_ROWS=180
MAX_CANDIDATES=100
ID=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=120)]
MAGNITUDE=1e12
CATALOG={
    "mean": {"name":"Weighted group mean","version":"1","status":"baseline","features":"numeric"},
    "ridge": {"name":"Group-weighted Ridge","version":"1","status":"research","features":"numeric"},
    "gp": {"name":"Fixed-prior Matérn GP","version":"1","status":"research","features":"numeric"},
    "rdkit_ridge": {"name":"Bounded RDKit Ridge","version":"1","status":"optional_research","features":"molecular"},
    "coupled": {"name":"Coupled polymer mass heads","version":"1","status":"research","features":"numeric"},
    "rdkit_coupled": {"name":"Bounded RDKit coupled heads","version":"1","status":"optional_research","features":"molecular"},
}


class Candidate(StrictModel):
    id: ID
    parameters: dict[str,FiniteFloat]
    smiles: str | None = Field(default=None,max_length=512)
    cost_minor: int = Field(strict=True,ge=1,le=10**12)


class StudyRequest(StrictModel):
    name: ID
    dataset_kind: Literal["real","synthetic_fixture"]
    provenance: Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)]
    config: ProjectConfig
    csv: str = Field(min_length=1,max_length=200_000)
    model: Literal["auto","mean","ridge","gp","rdkit_ridge","coupled","rdkit_coupled"]="auto"
    molecular: bool=False
    coupled_mass: bool=False
    coverage: Literal[.8,.9,.95]=.9
    max_interval_width: dict[str,FiniteFloat]
    currency: Annotated[str,StringConstraints(pattern=r"^[A-Z]{3}$")]="RUB"
    budget_minor: int=Field(default=0,strict=True,ge=0,le=10**12)
    max_suggestions: int=Field(default=3,strict=True,ge=0,le=20)
    candidate_pool_reviewed: bool=False
    candidates: list[Candidate]=Field(default_factory=list,max_length=MAX_CANDIDATES)

    @model_validator(mode="after")
    def contract(self):
        factors=[f.name for f in self.config.factors]
        targets=[t.name for t in self.config.targets]
        if set(factors+targets)&{"group_id","role","smiles"}:
            raise ValueError("Reserved research CSV column")
        if self.config.mode!="real":
            raise ValueError("Use research dataset_kind to label fixture data; config.mode must be real")
        if set(self.max_interval_width)!=set(targets) or any(not 0<v<=MAGNITUDE for v in self.max_interval_width.values()):
            raise ValueError("Explicit positive interval width for every target required")
        if any(max(abs(f.lower),abs(f.upper))>MAGNITUDE or f.upper-f.lower<1e-12 for f in self.config.factors):
            raise ValueError("Factor range exceeds research numerical limits")
        if any(not 1e-12<=t.scale<=MAGNITUDE for t in self.config.targets):
            raise ValueError("Target scale exceeds research numerical limits")
        if self.coupled_mass:
            units={t.name:t.unit for t in self.config.targets}
            if targets!=list(TARGETS) or units!={"conversion_fraction":"fraction","mn_g_mol":"g/mol","mw_g_mol":"g/mol","dispersity":"1"}:
                raise ValueError("Coupled mode requires canonical targets in documented order and units")
        if self.model.startswith("rdkit_") and not self.molecular:
            raise ValueError("RDKit model requires molecular mode")
        if self.model in ("coupled","rdkit_coupled") and not self.coupled_mass:
            raise ValueError("Coupled model requires coupled_mass mode")
        if self.candidates and not self.candidate_pool_reviewed:
            raise ValueError("Explicit operator-reviewed candidate pool required")
        if len({c.id for c in self.candidates})!=len(self.candidates):
            raise ValueError("Duplicate candidate ID")
        return self


def rdkit_available():
    try:
        import rdkit
        return True
    except ImportError:
        return False


def canonical(s,enabled):
    if not enabled:
        if s:
            raise ValueError("SMILES supplied without molecular mode")
        return ""
    if not s or len(s)>512 or not rdkit_available():
        raise ValueError("Molecular mode needs RDKit and one valid SMILES per row")
    return molecular_features(s)[0]


def checked(cfg,parameters,outcomes):
    # Generic workspace validator permits partial outcomes on failed status.
    # Here partial observed labels are *not* failed trials: status is only an
    # internal adapter to reuse factor/constraint and target-name checks.
    if any(abs(v)>MAGNITUDE for v in (*parameters.values(),*outcomes.values())):
        raise ValueError("Numerical value exceeds research limit")
    validate_result(cfg,ResultInput(parameters=parameters,outcomes=outcomes,status="failed"))


def parse(request):
    cfg=request.config
    factors=[f.name for f in cfg.factors];targets=[t.name for t in cfg.targets]
    header=["group_id","role",*factors,*targets]+(["smiles"] if request.molecular else [])
    reader=csv.DictReader(io.StringIO(request.csv.lstrip("\ufeff")),strict=True)
    if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or set(reader.fieldnames)!=set(header):
        raise ValueError("CSV headers must exactly match group_id, role, factors, targets and optional smiles")
    rows=[]; group_role={}; group_molecule={}; point_group={}
    for row in reader:
        if len(rows)>=MAX_ROWS:
            raise ValueError("Research CSV exceeds 600 observations")
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Invalid CSV row width")
        group=row["group_id"].strip();role=row["role"].strip()
        if not group or len(group)>120 or role not in ("train","calibration","validation"):
            raise ValueError("Invalid group or role")
        if group in group_role and group_role[group]!=role:
            raise ValueError("A group cannot cross train/calibration/validation")
        parameters={k:float(row[k]) for k in factors}
        outcomes={k:float(row[k]) for k in targets if row[k].strip()}
        if not outcomes:
            raise ValueError("Every row needs at least one observed target")
        checked(cfg,parameters,outcomes)
        s=canonical(row.get("smiles"),request.molecular)
        if group in group_molecule and group_molecule[group]!=s:
            raise ValueError("One research group cannot contain multiple molecules")
        if request.coupled_mass:
            for t,v in outcomes.items():
                if t=="conversion_fraction" and not 0<=v<=1 or t in ("mn_g_mol","mw_g_mol") and v<=0 or t=="dispersity" and v<1:
                    raise ValueError("Nonphysical polymer label")
            if {"mn_g_mol","mw_g_mol"}<=outcomes.keys() and outcomes["mw_g_mol"]<outcomes["mn_g_mol"]:
                raise ValueError("Observed Mw below Mn")
            if set(TARGETS[1:])<=outcomes.keys():
                ratio=outcomes["mw_g_mol"]/outcomes["mn_g_mol"]
                if abs(outcomes["dispersity"]-ratio)>max(.05,.05*ratio):
                    raise ValueError("Observed dispersity inconsistent with Mw/Mn")
        key=(s,*[parameters[k] for k in factors])
        if key in point_group and point_group[key]!=group:
            raise ValueError("Identical condition occurs in multiple groups; group replicates together")
        point_group[key]=group
        group_role[group]=role;group_molecule[group]=s
        rows.append({"group_id":group,"role":role,"parameters":parameters,"outcomes":outcomes,"smiles":s})
    if not rows:
        raise ValueError("Empty dataset")
    for t in targets:
        n=len({r["group_id"] for r in rows if r["role"]=="train" and t in r["outcomes"]})
        v=len({r["group_id"] for r in rows if r["role"]=="validation" and t in r["outcomes"]})
        if n<6 or v<2:
            raise ValueError("Each target needs at least six train and two validation groups: "+t)
    candidate_rows=[];candidate_keys=set()
    for c in request.candidates:
        checked(cfg,c.parameters,{})
        s=canonical(c.smiles,request.molecular)
        key=(s,*[c.parameters[k] for k in factors])
        if key in point_group or key in candidate_keys or c.id in group_role:
            raise ValueError("Candidate repeats a measured condition or group/candidate ID")
        candidate_keys.add(key)
        candidate_rows.append({**c.model_dump(mode="json"),"smiles":s})
    return rows,candidate_rows


def arrays(cfg,rows):
    factors=[f.name for f in cfg.factors];targets=[t.name for t in cfg.targets]
    raw=np.array([[r["parameters"][f] for f in factors] for r in rows],float).reshape(len(rows),len(factors))
    low=np.array([f.lower for f in cfg.factors]);span=np.array([f.upper-f.lower for f in cfg.factors])
    return ((raw-low)/span,np.array([r["smiles"] for r in rows],str),
            np.array([[r.get("outcomes",{}).get(t,np.nan) for t in targets] for r in rows],float).reshape(len(rows),len(targets)),
            np.array([r.get("group_id",r.get("id")) for r in rows],str))


def weights(g):
    _,idx,count=np.unique(g,return_inverse=True,return_counts=True)
    return 1./count[idx]


class Independent:
    def __init__(self,key,scales):
        self.key,self.scales=key,np.asarray(scales)
        self.molecular=key=="rdkit_ridge"

    def fit(self,X,S,y,g):
        self.heads=[]
        for j in range(y.shape[1]):
            k=np.isfinite(y[:,j]);w=weights(g[k]);z=y[k,j]/self.scales[j]
            if not len(z):
                raise ValueError("No observed training labels")
            self.heads.append(None)
            # Descriptor clipping must be target-specific; each target may
            # have distinct observed molecules. Store independent bounds.
            if self.molecular:
                d=np.asarray([molecular_features(s)[1] for s in S[k]],float)
                low,high=d.min(axis=0),d.max(axis=0)
                span=high-low
                design=np.column_stack([X[k],(d-low)/np.where(span>1e-12,span,1.)])
            else:
                low=high=None;design=X[k]
            if self.key=="mean":
                model=float(np.average(z,weights=w))
            elif self.key in ("ridge","rdkit_ridge"):
                model=Ridge(alpha=.1).fit(design,z,sample_weight=w)
            else:
                if len(z)>MAX_GP_ROWS:
                    raise ValueError("GP research row cap exceeded")
                # Fixed prior/noise in declared target-scale units. No tuning
                # on calibration or validation. Group-balanced observation noise.
                kernel=ConstantKernel(1.,constant_value_bounds="fixed")*Matern(
                    length_scale=np.full(design.shape[1],.35),length_scale_bounds="fixed",nu=2.5)
                model=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True,alpha=.02/w).fit(design,z)
            self.heads[-1]=(model,low,high)
        return self

    def predict(self,X,S):
        p=[]
        for j,(model,low,high) in enumerate(self.heads):
            design=X
            if self.molecular:
                d=np.asarray([molecular_features(s)[1] for s in S],float)
                design=np.column_stack([X,(np.clip(d,low,high)-low)/np.where(high-low>1e-12,high-low,1.)])
            v=np.full(len(X),model) if self.key=="mean" else model.predict(design)
            p.append(v*self.scales[j])
        return np.column_stack(p)


class Coupled:
    def __init__(self,key):
        self.model=BoundedMolecularModel(alpha=10.,molecular=key=="rdkit_coupled") if key=="rdkit_coupled" else PropertyModel("numeric",True,10.)

    def fit(self,X,S,y,g):
        self.model.fit(X,S,np.empty((len(X),0),str),y,g)
        return self

    def predict(self,X,S):
        return self.model.predict(X,S,np.empty((len(X),0),str))


def adapter(key,scales):
    return Coupled(key) if key in ("coupled","rdkit_coupled") else Independent(key,scales)


def macro(y,p,g):
    return float(np.mean([np.mean(np.abs(y[g==k]-p[g==k])) for k in np.unique(g)]))


def eligible_models(request,y):
    if request.coupled_mass:
        keys=["coupled"] + (["rdkit_coupled"] if request.molecular else [])
    else:
        keys=["mean","ridge"]
        if max(np.isfinite(y).sum(axis=0))<=MAX_GP_ROWS:
            keys.append("gp")
        if request.molecular:
            keys.append("rdkit_ridge")
    if request.model!="auto":
        if request.model not in keys:
            raise ValueError("Requested model not eligible for dataset/features/row limits")
        return [request.model]
    return keys


def choose(request,X,S,y,g):
    scales=np.asarray([t.scale for t in request.config.targets])
    cv=list(GroupKFold(n_splits=3).split(X,groups=g))
    results=[]
    for key in eligible_models(request,y):
        p=np.full_like(y,np.nan)
        for tr,te in cv:
            if set(g[tr])&set(g[te]):
                raise ValueError("Group CV leakage")
            # Six groups overall does not ensure sufficient labels per fold.
            if any(len(set(g[tr][np.isfinite(y[tr,j])]))<2 for j in range(y.shape[1])):
                raise ValueError("Insufficient observed training groups in a CV fold; revise group split")
            p[te]=adapter(key,scales).fit(X[tr],S[tr],y[tr],g[tr]).predict(X[te],S[te])
        if not np.isfinite(p).all():
            raise ValueError("Non-finite CV predictions")
        errs=[macro(y[k,j],p[k,j],g[k]) for j in range(y.shape[1]) if (k:=np.isfinite(y[:,j])).any()]
        results.append({"model":key,"normalized_group_mae":float(np.mean(np.array(errs)/scales)),
                        "target_mae":dict(zip([t.name for t in request.config.targets],errs))})
    winner=min(results,key=lambda r:(r["normalized_group_mae"],eligible_models(request,y).index(r["model"])))["model"]
    return winner,results


def predict_bounds(request,X,S,y,g,Xc,Sc,yc,gc,Xq,Sq,p,pcal):
    """Per-target group conformal and conservative exact-molecule support."""
    result=[];scores=[]
    for j,t in enumerate(request.config.targets):
        k=np.isfinite(yc[:,j])
        res=np.abs(yc[k,j]-pcal[k,j])
        gs=gc[k]
        a=[float(res[gs==key].max()) for key in np.unique(gs)]
        q=conformal_quantile(a,request.coverage)
        scores.append({"target":t.name,"groups":len(a),"quantile":q,
                       "minimum_finite_groups":minimum_calibration_groups(request.coverage)})
    for i,(x,s) in enumerate(zip(Xq,Sq)):
        row={}
        for j,t in enumerate(request.config.targets):
            mask=np.isfinite(y[:,j]) & (S==s)
            n=len(set(g[mask]))
            q=scores[j]["quantile"]
            reason="supported"
            if n<3:
                reason="unseen_molecule" if request.molecular and not np.any(S==s) else "insufficient_target_groups"
            else:
                z=X[mask]
                if np.any(x<z.min(axis=0)-1e-10) or np.any(x>z.max(axis=0)+1e-10):
                    reason="outside_observed_range"
                elif np.linalg.norm(z-x,axis=1).min()/np.sqrt(len(x))>.35:
                    reason="sparse_process_support"
            if reason=="supported":
                if q is None:
                    reason="interval_unavailable"
                elif 2*q>request.max_interval_width[t.name]:
                    reason="interval_too_wide"
            ok=reason=="supported"
            row[t.name]={"status":"research_supported" if ok else "abstain","reason":reason,
                        "prediction":float(p[i,j]) if ok else None,
                        "interval":[float(p[i,j]-q),float(p[i,j]+q)] if ok else None}
            if ok and request.coupled_mass:
                # Intersect marginal bounds with known algebraic/physical
                # support. This does not create a joint coverage guarantee.
                lo,hi=row[t.name]["interval"]
                if t.name=="conversion_fraction":
                    lo,hi=max(0.,lo),min(1.,hi)
                elif t.name=="dispersity":
                    lo=max(1.,lo)
                else:
                    lo=max(0.,lo)
                row[t.name]["interval"]=[lo,hi]
        result.append(row)
    return result,scores


def collection_plan(request,training,candidates):
    """Greedy diversity per known group cost; ignores every measured outcome."""
    if not candidates or not request.max_suggestions:
        return {"suggestions":[],"spent_minor":0,"remaining_minor":request.budget_minor}
    X,S,_,_=arrays(request.config,training)
    Q,T,_,_=arrays(request.config,candidates)
    if request.molecular:
        cats=sorted(set(S)|set(T))
        X=np.column_stack([X,np.column_stack([S==m for m in cats]).astype(float)])
        Q=np.column_stack([Q,np.column_stack([T==m for m in cats]).astype(float)])
    chosen=[];spent=0;seen=X.copy()
    while len(chosen)<request.max_suggestions:
        choices=[i for i,c in enumerate(candidates) if i not in chosen and c["cost_minor"]<=request.budget_minor-spent]
        if not choices:
            break
        dist=np.linalg.norm(Q[choices,None,:]-seen[None,:,:],axis=2).min(axis=1)/np.sqrt(Q.shape[1])
        score=dist/np.asarray([candidates[i]["cost_minor"] for i in choices])
        ix=choices[int(np.argmax(score))]
        chosen.append(ix);spent+=candidates[ix]["cost_minor"];seen=np.vstack([seen,Q[ix]])
    return {"suggestions":[{"id":candidates[i]["id"],"cost_minor":candidates[i]["cost_minor"],
                            "status":"operator_review_required","method":"maximin_per_cost"} for i in chosen],
            "spent_minor":spent,"remaining_minor":request.budget_minor-spent}


def analyze(request):
    rows,candidates=parse(request)
    roles={k:[r for r in rows if r["role"]==k] for k in ("train","calibration","validation")}
    X,S,y,g=arrays(request.config,roles["train"])
    Xc,Sc,yc,gc=arrays(request.config,roles["calibration"])
    Xv,Sv,yv,gv=arrays(request.config,roles["validation"])
    Xq,Sq,_,_=arrays(request.config,candidates)
    scales=[t.scale for t in request.config.targets]
    with threadpool_limits(limits=1):
        key,cv=choose(request,X,S,y,g)
        model=adapter(key,scales).fit(X,S,y,g)
        pv=model.predict(Xv,Sv)
        pc=model.predict(Xc,Sc) if len(Xc) else np.empty_like(yc)
        pq=model.predict(Xq,Sq) if len(Xq) else np.empty((0,len(scales)))
        if not all(np.isfinite(p).all() for p in (pv,pc,pq)):
            raise ValueError("Non-finite model prediction")
        valid,cal=predict_bounds(request,X,S,y,g,Xc,Sc,yc,gc,Xv,Sv,pv,pc)
        result,_=predict_bounds(request,X,S,y,g,Xc,Sc,yc,gc,Xq,Sq,pq,pc)
    validation={}
    for j,t in enumerate(request.config.targets):
        k=np.isfinite(yv[:,j])
        q=cal[j]["quantile"]
        hit=np.abs(yv[k,j]-pv[k,j])<=q if q is not None else None
        gs=gv[k]
        accepted=np.array([r[t.name]["status"]=="research_supported" for r in valid])&k
        validation[t.name]={"observed_rows":int(k.sum()),"groups":len(set(gs)),
            "group_macro_mae":macro(yv[k,j],pv[k,j],gs),
            "whole_group_coverage":float(np.mean([hit[gs==group].all() for group in np.unique(gs)])) if hit is not None else None,
            "accepted_rows":int(accepted.sum()),
            "accepted_group_macro_mae":macro(yv[accepted,j],pv[accepted,j],gv[accepted]) if accepted.any() else None,
            # Diagnostic only: never label these retrospective numbers as a recommendation.
            "diagnostic_scatter":[{"observed":float(yv[i,j]),"predicted":float(pv[i,j]),"supported":bool(accepted[i])}
                                  for i in np.flatnonzero(k)[:200]]}
    return {"selected_model":key,"selection":cv,"calibration":cal,"validation":validation,
            "counts":{k:{"rows":len(v),"groups":len({r["group_id"] for r in v})} for k,v in roles.items()},
            "candidate_predictions":[{**c,"properties":p} for c,p in zip(candidates,result)],
            "collection_plan":collection_plan(request,roles["train"],candidates),
            "limitations":"Research only. Model choice on train group CV; independent declared calibration and validation. "
             "Group identity and physical independence must be verified by the operator. Marginal intervals not simultaneous "
             "and not guaranteed after selection or on new chemistry. Collection plan uses geometry/cost only; "
             "no chemical safety approval, no automatic pilot promotion, no proven physical savings."}
