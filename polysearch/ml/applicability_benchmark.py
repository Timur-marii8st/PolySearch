"""Retrospective applicability, interval, and molecular adaptation study."""
import argparse
import json
from pathlib import Path
from collections import Counter
from importlib.metadata import version

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import molecular_benchmark as base
from .molecular import TARGETS, PropertyModel, JointGroupCalibration, consistency
from .applicability import (BoundedMolecularModel, Domain, MarginalGroupCalibration,
                            selective_predictions, similarity)
from .benchmark import macro_mae
from ..data.monash_source import digest

# Predeclared engineering tolerances, illustrative only. Never tuned on test labels.
WIDTHS={"conversion_width":.20,"mass_ratio":3.,"dispersity_width":.30}
ALPHAS=(1.,10.,100.)
ADAPT_SEEDS=tuple(range(8))


def fit(data,idx,alpha=10.,molecular=True):
    X,S,C,y,g,*_=data
    return BoundedMolecularModel(alpha,molecular).fit(X[idx],S[idx],C[idx],y[idx],g[idx])


def predict(model,data,idx):
    X,S,C,*_=data
    p=model.predict(X[idx],S[idx],C[idx])
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction")
    return p


def tune(data,idx,molecular=True):
    X,S,C,y,g,*_=data
    folds=list(GroupKFold(n_splits=3).split(X[idx],groups=g[idx]))
    scores=[]
    for a in ALPHAS:
        p=np.full((len(idx),4),np.nan)
        for tr,va in folds:
            train,test=idx[tr],idx[va]
            if set(g[train])&set(g[test]):
                raise ValueError("Inner group leakage")
            p[va]=predict(fit(data,train,a,molecular),data,test)
        scores.append((base.losses(y[idx],p,g[idx]),a))
    return min(scores)[1]


def select_report(y,p,g,accept,reasons):
    out={}
    for j,target in enumerate(TARGETS):
        obs=np.isfinite(y[:,j])
        k=obs&accept[:,j]
        groups=np.unique(g[obs])
        out[target]={"observed_rows":int(obs.sum()),"accepted_rows":int(k.sum()),
            "accepted_fraction":float(k.sum()/obs.sum()) if obs.any() else None,
            "observed_groups":len(groups),"groups_with_any_accepted":len(set(g[k])),
            "fully_accepted_groups":sum(bool(accept[g==a,j][obs[g==a]].all()) for a in groups),
            "all_macro_mae":macro_mae(y[obs,j],p[obs,j],g[obs]) if obs.any() else None,
            "accepted_macro_mae":macro_mae(y[k,j],p[k,j],g[k]) if k.any() else None,
            "reasons":dict(Counter(str(a) for a in reasons[obs,j]))}
    return out


def interval_report(y,lo,hi,g,accepted=None):
    out={}
    for j,t in enumerate(TARGETS):
        obs=np.isfinite(y[:,j])
        valid=obs&np.isfinite(lo[:,j])&np.isfinite(hi[:,j])
        if accepted is not None:
            valid&=accepted[:,j]
        hit=(y[:,j]>=lo[:,j])&(y[:,j]<=hi[:,j])
        gs=np.unique(g[valid])
        width=hi[valid,j]-lo[valid,j]
        out[t]={"observed_rows":int(obs.sum()),"evaluated_rows":int(valid.sum()),
                "evaluated_groups":len(gs),
                "row_coverage":float(hit[valid].mean()) if valid.any() else None,
                "whole_group_coverage_on_evaluated_labels":float(np.mean([hit[valid&(g==a)].all() for a in gs])) if len(gs) else None,
                "median_width":float(np.median(width)) if valid.any() else None,
                "median_multiplicative_width":float(np.median(hi[valid,j]/lo[valid,j])) if j in (1,2) and valid.any() and np.all(lo[valid,j]>0) else None}
    return out


def group_cv(data):
    X,S,C,y,g,f,M=data
    p=np.full_like(y,np.nan)
    supported=np.zeros(y.shape,bool)
    accepted=np.zeros(y.shape,bool)
    reasons=np.empty(y.shape,object)
    final_reasons=np.empty(y.shape,object)
    bounds={kind:(np.full_like(y,np.nan),np.full_like(y,np.nan)) for kind in ("joint90","marginal80","marginal90")}
    fold_details=[]
    for fold in sorted(set(f)):
        tr,te=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
        keys=np.random.default_rng(42+int(fold)).permutation(np.unique(g[tr]))
        ncal=max(9,int(np.ceil(.25*len(keys))))
        ca=tr[np.isin(g[tr],keys[:ncal])]
        fi=tr[~np.isin(g[tr],keys[:ncal])]
        if set(g[ca])&set(g[fi]) or set(g[ca])&set(g[te]) or set(g[fi])&set(g[te]):
            raise ValueError("Split leakage")
        alpha=tune(data,fi)
        model=fit(data,fi,alpha)
        p[te]=predict(model,data,te)
        proper_pred=predict(model,data,fi)
        cal_pred=predict(model,data,ca)
        jc=JointGroupCalibration().fit(y[fi],proper_pred,y[ca],cal_pred,g[ca])
        mc=MarginalGroupCalibration().fit(y[ca],cal_pred,g[ca])
        for kind,b in (("joint90",jc.interval(p[te],.9)),("marginal80",mc.interval(p[te],.8)),
                       ("marginal90",mc.interval(p[te],.9))):
            if b is not None:
                bounds[kind][0][te],bounds[kind][1][te]=b
        domain=Domain().fit(X[fi],S[fi],C[fi],y[fi],g[fi])
        supported[te],reasons[te],_=domain.assess(X[te],S[te],C[te])
        sel,accepted[te]=selective_predictions(p[te],bounds["marginal90"][0][te],
                  bounds["marginal90"][1][te],supported[te],reasons[te],**WIDTHS)
        final_reasons[te]=np.array([[v[t]["reason"] for t in TARGETS] for v in sel])
        fold_details.append({"fold":int(fold),"alpha":alpha,"proper_train_groups":sorted(set(g[fi])),
            "calibration_groups":sorted(set(g[ca])),"test_groups":sorted(set(g[te])),
            "calibration_target_groups":{t:len(mc.scores[j]) for j,t in enumerate(TARGETS)}})
    if not np.isfinite(p).all():
        raise ValueError("Incomplete held-out predictions")
    return {"rows":len(y),"groups":len(set(g)),"metrics":base.target_metrics(y,p,g),
            "consistency":consistency(p),"domain_only":select_report(y,p,g,supported,reasons),
            "width_and_domain":select_report(y,p,g,accepted,final_reasons),
            "intervals":{k:interval_report(y,*b,g) for k,b in bounds.items()},
            "accepted_marginal90_intervals":interval_report(y,*bounds["marginal90"],g,accepted),
            "fold_details":fold_details}


def lomo(data):
    X,S,C,y,g,f,M=data
    models=("raw_v3","bounded","process_only")
    pred={k:np.full_like(y,np.nan) for k in models}
    accept=np.zeros(y.shape,bool)
    detail=[]
    adaptation_records=[]
    for mol in sorted(set(S)):
        tr,te=np.flatnonzero(S!=mol),np.flatnonzero(S==mol)
        if set(g[tr])&set(g[te]):
            raise ValueError("LOMO leakage")
        a=base.tune(data,tr,"molecular_coupled")
        pred["raw_v3"][te]=base.predict(base.model_fit(data,tr,"molecular_coupled",a),data,te)
        for name,mode in (("bounded",True),("process_only",False)):
            alpha=tune(data,tr,mode)
            pred[name][te]=predict(fit(data,tr,alpha,mode),data,te)
        domain=Domain().fit(X[tr],S[tr],C[tr],y[tr],g[tr])
        accept[te],_,near=domain.assess(X[te],S[te],C[te])
        detail.append({"codes":sorted(set(M[te])),"molecule":mol,"groups":len(set(g[te])),
                       "similarity":float(near[0]),"accepted_predictions":int(accept[te].sum()),
                       "models":{k:base.target_metrics(y[te],p[te],g[te]) for k,p in pred.items()}})
        keys=np.unique(g[te])
        if len(keys)<5:
            continue
        for seed in ADAPT_SEEDS:
            chosen=np.random.default_rng(seed).choice(keys,3,replace=False)
            k=te[np.isin(g[te],chosen)]
            test=te[~np.isin(g[te],chosen)]
            train=np.r_[tr,k]
            if set(g[train])&set(g[test]):
                raise ValueError("Adaptation leakage")
            pooled=fit(data,train,10.)
            p=predict(pooled,data,test)
            adap_domain=Domain().fit(X[train],S[train],C[train],y[train],g[train])
            ok,why,_=adap_domain.assess(X[test],S[test],C[test])
            # Local fit may not have enough observed groups of every target.
            # Don't pick the adaptation groups using future target availability.
            local=None
            if all(len(set(g[k][np.isfinite(y[k,j])]))>=2 for j in range(4)):
                local=predict(fit(data,k,10.,False),data,test)
            adaptation_records.append({"codes":sorted(set(M[te])),"molecule":mol,"seed":seed,
                "adaptation_groups":sorted(chosen),"test_groups":sorted(set(g[test])),
                "zero_bounded":base.target_metrics(y[test],pred["bounded"][test],g[test]),
                "adapted":base.target_metrics(y[test],p,g[test]),
                "local_3":base.target_metrics(y[test],local,g[test]) if local is not None else None,
                "domain_only":select_report(y[test],p,g[test],ok,why)})
    if any(not np.isfinite(p).all() for p in pred.values()) or accept.any():
        raise ValueError("Incomplete or unexpected zero-shot domain acceptance")
    return {"molecules":len(detail),"groups":len(set(g)),
            "models":{k:base.target_metrics(y,v,g) for k,v in pred.items()},
            "zero_shot_domain_accepted":int(accept.sum()),"holdouts":detail,
            "adaptation_records":adaptation_records,
            "adaptation_summary":summarize_adapt(adaptation_records)}


def summarize_adapt(records):
    result={}
    for j,t in enumerate(TARGETS):
        paired=[r for r in records if r["adapted"][t].get("group_macro_mae") is not None]
        local=[r for r in paired if r["local_3"] and r["local_3"][t].get("group_macro_mae") is not None]
        # Average over paired molecule/seed runs, not independent replicates.
        def avg(rs,key):
            return float(np.mean([r[key][t]["group_macro_mae"] for r in rs])) if rs else None
        obs=sum(r["domain_only"][t]["observed_rows"] for r in paired)
        acc=sum(r["domain_only"][t]["accepted_rows"] for r in paired)
        result[t]={"paired_runs":len(paired),"zero_bounded_macro_mae":avg(paired,"zero_bounded"),
            "adapted_macro_mae":avg(paired,"adapted"),"local_comparable_runs":len(local),
            "local_3_macro_mae":avg(local,"local_3"),"adapted_on_local_comparable":avg(local,"adapted"),
            "domain_accepted_fraction":acc/obs if obs else None}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monash",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    parser.add_argument("--raft",type=Path,default=Path("data/processed/raft-v1"))
    parser.add_argument("--raft-raw",type=Path,default=Path("data/raw/raft-v1"))
    parser.add_argument("--registry",type=Path,default=Path("data/molecular/monomers-v1.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/applicability"))
    args=parser.parse_args()
    registry=base.load_registry(args.registry,args.raft_raw)
    monash,mq=base.read_dataset(args.monash)
    raft,rq=base.read_raft(args.raft)
    report={"registry_sha256":digest(args.registry),"monash_hashes":mq["hashes"],"raft_sha256":rq["table_sha256"],
            "versions":{k:version(k) for k in ("rdkit","numpy","scipy","scikit-learn")},
            "width_tolerances":WIDTHS,"domain":{"min_groups":3,"range_pad":.1,"max_distance":.5},
            "results":{},"limitations":"Retrospective provisional data. Guard is an engineering abstention heuristic, "
            "not a verified safety boundary. Marginal coverage is not simultaneous or selective coverage guarantee. "
            "Known molecule alone does not establish process equivalence. Adaptation results are exploratory; "
            "only domain support, not calibrated intervals, evaluated after three-group adaptation. "
            "No new experiments; server API unchanged."}
    with threadpool_limits(limits=1):
        for name,rows in (("monash",monash),("raft",raft)):
            data=base.make_data(rows,registry[name],name)
            report["results"][name]={"group_cv":group_cv(data),"lomo":lomo(data)}
            print("DOMAIN_DATASET_DONE "+name,flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    compact={}
    for name,r in report["results"].items():
        cv,lo=r["group_cv"],r["lomo"]
        compact[name]={"cv":{k:v for k,v in cv.items() if k!="fold_details"},
                       "lomo":{k:v for k,v in lo.items() if k not in ("holdouts","adaptation_records")}}
    print("DOMAIN_SUMMARY "+json.dumps(compact,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
