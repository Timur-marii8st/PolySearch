"""Group learning curves, label-free active design and calibration-size study.

Monash and RAFT are separate. Target eligibility is a retrospective archive
restriction, not known availability in a prospective lab. Candidate design
uses process condition, molecule and recipe only, not observed time grids.
"""
import argparse
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from . import molecular_benchmark as base
from .molecular import TARGETS, molecular_features
from .data_planning import POLICIES, ordered_groups, minimum_calibration_groups
from .molecular import conformal_quantile
from ..data.monash_source import digest

SEEDS=tuple(range(5))
CAL_SEEDS=(10,11,12)
ALPHA=10.
BUDGETS={"monash":(3,5,8,12,20,30),"raft":(3,5,10,20,40,80,120,200)}
CAL_SIZES=(4,9,14,19,29,39)
COVERAGES=(.8,.9)
WIDTHS=(.20,3.,3.,.30)


def mask_target(data,j):
    X,S,C,y,g,f,M=data
    keep=np.isfinite(y[:,j])
    return X[keep],S[keep],C[keep],y[keep,j],g[keep],f[keep]


def group_geometry(X,S,C,g):
    """X last column is transformed measurement time; omit it entirely."""
    ids=np.unique(g)
    rows=[]; categories=[]
    for group in ids:
        k=g==group
        process=np.unique(X[k,:-1],axis=0)
        mol=np.unique(S[k])
        context=np.unique(C[k],axis=0)
        if len(process)!=1 or len(mol)!=1 or len(context)!=1:
            raise ValueError("Inconsistent condition metadata inside group")
        rows.append([*process[0],*molecular_features(mol[0])[1]])
        categories.append(context[0])
    Z=np.asarray(rows,float)
    cat=np.asarray(categories,str).reshape(len(ids),-1)
    if cat.shape[1]:
        Z=np.column_stack([Z,OneHotEncoder(sparse_output=False).fit_transform(cat)])
    return ids,Z


def labels(y,j):
    y=np.asarray(y,float)
    if not np.isfinite(y).all() or (j==0 and np.any((y<0)|(y>1))) or (j>0 and np.any(y<=0)):
        raise ValueError("Invalid target observations")
    return y if j==0 else np.log(y)


def decode(z,j):
    if j==0:
        return np.clip(z,0,1)
    p=np.exp(np.clip(z,-20,20))
    return np.maximum(p,1.) if j==3 else p


class Head:
    """Fixed-alpha target-specific Ridge baseline, not the production model."""
    def __init__(self,j):
        self.j=j

    @staticmethod
    def desc(S):
        return np.asarray([molecular_features(s)[1] for s in S],float)

    def design(self,X,S,C):
        z=self.scaler.transform(np.column_stack([X,np.clip(self.desc(S),self.low,self.high)]))
        if self.encoder is not None:
            z=np.column_stack([z,self.encoder.transform(C)])
        return z

    def fit(self,X,S,C,y,g):
        d=self.desc(S)
        self.low,self.high=d.min(axis=0),d.max(axis=0)
        self.scaler=StandardScaler().fit(np.column_stack([X,d]))
        self.encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(C) if C.shape[1] else None
        _,idx,n=np.unique(g,return_inverse=True,return_counts=True)
        self.model=Ridge(alpha=ALPHA).fit(self.design(X,S,C),labels(y,self.j),sample_weight=1./n[idx])
        return self

    def predict(self,X,S,C):
        return decode(self.model.predict(self.design(X,S,C)),self.j)


def fit_predict(d,j,train,test):
    X,S,C,y,g,f=d
    if set(g[train])&set(g[test]):
        raise ValueError("Train/test group leakage")
    return Head(j).fit(X[train],S[train],C[train],y[train],g[train]).predict(X[test],S[test],C[test])


def group_errors(y,p,g):
    return [float(np.mean(np.abs(y[g==key]-p[g==key]))) for key in np.unique(g)]


def learning_curves(d,j,dataset):
    X,S,C,y,g,f=d
    folds=sorted(set(f))
    minimum=min(len(set(g[f!=fold])) for fold in folds)
    budgets=[b for b in BUDGETS[dataset] if b<=minimum]
    if not budgets:
        raise ValueError("Insufficient independent groups")
    curves={p:{b:[] for b in budgets} for p in POLICIES}
    records=[]
    for fold in folds:
        tr,te=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
        ids,geometry=group_geometry(X[tr],S[tr],C[tr],g[tr])
        if set(ids)&set(g[te]):
            raise ValueError("Candidate/test leakage")
        for seed in SEEDS:
            initial=np.random.default_rng(seed).choice(len(ids),budgets[0],replace=False).tolist()
            for policy in POLICIES:
                order=ordered_groups(geometry,initial,policy,seed+1000,budgets[-1])
                rec={"fold":int(fold),"seed":seed,"policy":policy,"initial_groups":ids[initial].tolist(),
                     "order":ids[order].tolist(),"test_groups":sorted(set(g[te])),"points":[]}
                for b in budgets:
                    selected=ids[order[:b]]
                    fit=tr[np.isin(g[tr],selected)]
                    p=fit_predict(d,j,fit,te)
                    e=group_errors(y[te],p,g[te])
                    curves[policy][b].extend(e)
                    rec["points"].append({"budget":b,"test_macro_mae":float(np.mean(e)),
                                          "observed_train_rows":len(fit)})
                records.append(rec)
    summary={}
    for policy,series in curves.items():
        summary[policy]=[{"groups":b,"mean_group_mae":float(np.mean(errors)),
                          "median_group_mae":float(np.median(errors)),
                          "p90_group_absolute_error":float(np.quantile(errors,.9)),
                          "group_evaluations":len(errors)} for b,errors in series.items()]
    return {"budgets":budgets,"seeds":list(SEEDS),"policies":summary,"trajectories":records}


def group_scores(y,p,g,j):
    a=np.abs(labels(y,j)-labels(p,j))
    return np.array([a[g==key].max() for key in np.unique(g)])


def bounds(p,q,j):
    center=labels(p,j)
    if j==0:
        return np.clip(center-q,0,1),np.clip(center+q,0,1)
    if np.any(center+q>700):
        return None
    lo=np.exp(np.maximum(center-q,-700))
    hi=np.exp(center+q)
    if j==3:
        lo=np.maximum(lo,1.)
    return lo,hi


def width_ok(lo,hi,j):
    if j in (1,2):
        return np.log(hi)-np.log(lo)<=np.log(WIDTHS[j])
    return hi-lo<=WIDTHS[j]


def calibration_curve(d,j,dataset):
    X,S,C,y,g,f=d
    folds=sorted(set(f))
    min_groups=min(len(set(g[f!=fold])) for fold in folds)
    min_train=8 if dataset=="monash" else 20
    counts=[n for n in CAL_SIZES if n<=min_groups-min_train]
    if not counts:
        return {"status":"insufficient_groups","counts":[]}
    max_cal=max(counts)
    accum={(n,c):{"hits":[],"width":[],"passes":0,"rows":0,"fold_runs":0}
           for n in counts for c in COVERAGES}
    records=[]
    for fold in folds:
        tr,te=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
        for seed in CAL_SEEDS:
            keys=np.random.default_rng(seed).permutation(np.unique(g[tr]))
            pool=keys[:max_cal]
            fi=tr[~np.isin(g[tr],pool)]
            model=Head(j).fit(X[fi],S[fi],C[fi],y[fi],g[fi])
            pred_test=model.predict(X[te],S[te],C[te])
            rec={"fold":int(fold),"seed":seed,"fit_groups":sorted(set(g[fi])),
                 "calibration_order":pool.tolist(),"test_groups":sorted(set(g[te])),"points":[]}
            if set(g[fi])&set(pool) or set(g[te])&set(pool) or set(g[te])&set(g[fi]):
                raise ValueError("Calibration leakage")
            for n in counts:
                ca=tr[np.isin(g[tr],pool[:n])]
                pca=model.predict(X[ca],S[ca],C[ca])
                scores=group_scores(y[ca],pca,g[ca],j)
                for c in COVERAGES:
                    q=conformal_quantile(scores,c)
                    point={"calibration_groups":n,"coverage":c,"quantile":q}
                    if q is not None:
                        b=bounds(pred_test,q,j)
                        if b is not None:
                            lo,hi=b
                            hit=(y[te]>=lo)&(y[te]<=hi)
                            gh=[bool(hit[g[te]==key].all()) for key in np.unique(g[te])]
                            widths=hi-lo
                            a=accum[n,c]
                            a["hits"].extend(gh);a["width"].extend(widths.tolist())
                            a["passes"]+=int(width_ok(lo,hi,j).sum());a["rows"]+=len(lo);a["fold_runs"]+=1
                            point.update(whole_group_coverage=float(np.mean(gh)),median_width=float(np.median(widths)))
                    rec["points"].append(point)
            records.append(rec)
    summary=[]
    for (n,c),a in accum.items():
        summary.append({"calibration_groups":n,"nominal_coverage":c,"finite_rank_possible":n>=minimum_calibration_groups(c),
            "evaluated_fold_runs":a["fold_runs"],"total_fold_runs":len(folds)*len(CAL_SEEDS),
            "whole_group_coverage":float(np.mean(a["hits"])) if a["hits"] else None,
            "median_width":float(np.median(a["width"])) if a["width"] else None,
            "fraction_within_illustrative_width":a["passes"]/a["rows"] if a["rows"] else None})
    return {"status":"completed","fixed_reserved_calibration_groups":max_cal,
            "seeds":list(CAL_SEEDS),"summary":summary,"fold_details":records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monash",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    parser.add_argument("--raft",type=Path,default=Path("data/processed/raft-v1"))
    parser.add_argument("--raft-raw",type=Path,default=Path("data/raw/raft-v1"))
    parser.add_argument("--registry",type=Path,default=Path("data/molecular/monomers-v1.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/data-planning"))
    args=parser.parse_args()
    registry=base.load_registry(args.registry,args.raft_raw)
    monash,mq=base.read_dataset(args.monash);raft,rq=base.read_raft(args.raft)
    report={"registry_sha256":digest(args.registry),"monash_hashes":mq["hashes"],"raft_sha256":rq["table_sha256"],
            "versions":{k:version(k) for k in ("rdkit","numpy","scipy","scikit-learn")},
            "alpha":ALPHA,"minimum_calibration_groups":{str(c):minimum_calibration_groups(c) for c in COVERAGES},
            "results":{},"limitations":"Retrospective provisional archives, target-eligible groups known only in archive. "
            "Policies are label-free design heuristics (not adaptive GP or production recommendations). "
            "Each target uses its own group pool; do not sum budgets across targets or equate groups to physical runs. "
            "Repeated seeds/folds are descriptive, not independent replications. No prospective sample-size or coverage guarantee."}
    with threadpool_limits(limits=1):
        for dataset,rows in (("monash",monash),("raft",raft)):
            data=base.make_data(rows,registry[dataset],dataset)
            result={}
            for j,t in enumerate(TARGETS):
                d=mask_target(data,j)
                result[t]={"observed_rows":len(d[3]),"groups":len(set(d[4])),
                           "learning":learning_curves(d,j,dataset),"calibration":calibration_curve(d,j,dataset)}
                print("DATA_PLANNING_DONE "+dataset+" "+t,flush=True)
            report["results"][dataset]=result
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    compact={name:{t:{"rows":v["observed_rows"],"groups":v["groups"],
                      "learning":v["learning"]["policies"],
                      "calibration":v["calibration"].get("summary",[])} for t,v in values.items()}
             for name,values in report["results"].items()}
    print("PLANNING_SUMMARY "+json.dumps(compact,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
