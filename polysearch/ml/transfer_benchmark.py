"""Grouped multimonomer property prediction and zero/few-shot transfer.

No chemical identity is learned from held-out labels. NMR and SEC observations
are not merged; a model is fitted separately to the observed mask of each target.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .benchmark import metrics, macro_mae
from ..data.monash_source import DOI, digest

MONOMERS=("BA","MMA","VA")
TARGETS=("conversion_fraction","mn_g_mol","mw_g_mol","dispersity")
ALPHAS=(.01,.1,1.,10.,100.)


def read_dataset(root):
    root=Path(root)
    report=json.loads((root/"quality.json").read_text())
    if report["doi"]!=DOI:
        raise ValueError("Unexpected dataset DOI")
    observations=[]
    for name in ("conversion.csv","sec.csv"):
        path=root/name
        if digest(path)!=report["hashes"][name]:
            raise ValueError("Processed table checksum mismatch")
        with path.open(encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream):
                observations.append({**row,"channel":"nmr" if name=="conversion.csv" else "sec"})
    if len(observations)!=report["conversion_rows"]+report["sec_rows"]:
        raise ValueError("Unexpected table row count")
    return observations,report


def target_table(rows,target):
    selected=[r for r in rows if r.get(target) not in (None,"")]
    X=np.array([[float(r["temperature_c"]),float(r["ct_cm"]),float(r["residence_min"])]
                for r in selected])
    M=np.array([r["monomer"] for r in selected])
    y=np.array([float(r[target]) for r in selected])
    groups=np.array([r["split_group"] for r in selected])
    folds=np.array([int(r["fold"]) for r in selected])
    if not len(selected) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("No valid observed targets: "+target)
    if np.any(X[:,0]<=-273.15) or np.any(X[:,1]<0) or np.any(X[:,2]<=0):
        raise ValueError("Invalid features")
    for group in np.unique(groups):
        keep=groups==group
        if len(set(folds[keep]))!=1 or len(set(M[keep]))!=1 or len(np.unique(X[keep,:2],axis=0))!=1:
            raise ValueError("Group leakage or inconsistent condition")
    if target=="conversion_fraction" and np.any((y<0)|(y>1)):
        raise ValueError("Invalid conversion")
    if target!="conversion_fraction" and (np.any(y<=0) or target=="dispersity" and np.any(y<1)):
        raise ValueError("Invalid positive property")
    return X,M,y,groups,folds


def design(X):
    return np.column_stack([1000*(1/363.15-1/(X[:,0]+273.15)),
                            np.log1p(X[:,1]/.05),np.log1p(X[:,2])])


def group_weights(groups):
    _,idx,n=np.unique(groups,return_inverse=True,return_counts=True)
    return len(groups)/(len(n)*n[idx])


class PropertyHead:
    def __init__(self,target,alpha,identity):
        self.target,self.alpha,self.identity=target,alpha,identity

    def _features(self,X,M):
        z=self.scaler.transform(design(X))
        if not self.identity:
            return z
        if any(m not in self.seen for m in M):
            raise ValueError("Monomer identity not observed in training; use explicit zero-shot numeric model")
        one=np.column_stack([M==m for m in self.seen]).astype(float)
        return np.column_stack([z,one,(z[:,:,None]*one[:,None,:]).reshape(len(z),-1)])

    def fit(self,X,M,y,g):
        self.scaler=StandardScaler().fit(design(X))
        self.seen=tuple(sorted(set(M)))
        z=self._features(X,M)
        labels=y if self.target=="conversion_fraction" else np.log(y)
        self.model=Ridge(alpha=self.alpha).fit(z,labels,sample_weight=group_weights(g))
        return self

    def predict(self,X,M):
        raw=self.model.predict(self._features(X,M))
        if self.target=="conversion_fraction":
            return np.clip(raw,0,1)
        p=np.exp(np.clip(raw,-20,20))
        return np.maximum(p,1.) if self.target=="dispersity" else p


def choose_alpha(target,X,M,y,g,identity):
    """Nested validation; never use outer labels for selecting a head."""
    n=min(3,len(np.unique(g)))
    if n<2:
        return 1.,[]
    search=[]
    for a in ALPHAS:
        losses=[]
        for tr,va in GroupKFold(n_splits=n).split(X,y,g):
            if identity and not set(M[va])<=set(M[tr]):
                # No arbitrary category extrapolation for identity-aware models.
                continue
            model=PropertyHead(target,a,identity).fit(X[tr],M[tr],y[tr],g[tr])
            losses.append(macro_mae(y[va],model.predict(X[va],M[va]),g[va]))
        if losses:
            search.append({"alpha":a,"mean_group_mae":float(np.mean(losses))})
    return (min(search,key=lambda r:r["mean_group_mae"])["alpha"] if search else 1.),search


class MaskedPropertyRegressor:
    """Independent observed-label heads, with no fabricated joint measurements."""
    def fit(self,rows,identity=True):
        self.heads={}
        self.fit_groups={}
        for target in TARGETS:
            X,M,y,g,_=target_table(rows,target)
            if len(np.unique(g))<2:
                raise ValueError("Insufficient independent groups for "+target)
            alpha,_=choose_alpha(target,X,M,y,g,identity)
            self.heads[target]=PropertyHead(target,alpha,identity).fit(X,M,y,g)
            self.fit_groups[target]=len(np.unique(g))
        return self

    def predict(self,X,monomer):
        X=np.asarray(X,float)
        M=np.array([monomer]*len(X))
        return {target:head.predict(X,M) for target,head in self.heads.items()}


def in_domain(target,X,M,y,g,f):
    names=("local_ridge","pooled_identity_ridge","pooled_numeric_ridge")
    pred={k:np.full(len(y),np.nan) for k in names}
    detail=[]
    for fold in sorted(set(f)):
        tr,te=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
        if set(g[tr])&set(g[te]):
            raise ValueError("Outer group overlap")
        for monomer in sorted(set(M[te])):
            local_tr=tr[M[tr]==monomer]
            local_te=te[M[te]==monomer]
            if len(np.unique(g[local_tr]))<2:
                raise ValueError("Insufficient monomer groups")
            for name in names:
                idx=local_tr if name=="local_ridge" else tr
                identity=name=="pooled_identity_ridge"
                a,_=choose_alpha(target,X[idx],M[idx],y[idx],g[idx],identity)
                model=PropertyHead(target,a,identity).fit(X[idx],M[idx],y[idx],g[idx])
                pred[name][local_te]=model.predict(X[local_te],M[local_te])
                detail.append({"fold":int(fold),"monomer":monomer,"model":name,"alpha":a,
                               "fit_groups":sorted(set(g[idx])),"test_groups":sorted(set(g[local_te]))})
    if any(not np.isfinite(v).all() for v in pred.values()):
        raise ValueError("Missing out-of-fold predictions")
    return {"rows":len(y),"condition_groups":len(np.unique(g)),
            "models":{k:{"all":metrics(y,p,g),
                         "monomers":{m:metrics(y[M==m],p[M==m],g[M==m]) for m in sorted(set(M))}}
                      for k,p in pred.items()},"fold_details":detail},pred


def transfer(target,X,M,y,g):
    """Leave-one-monomer-out plus paired adaptation using exactly three groups.

    The zero-shot model has only numeric reaction features. Few-shot subsets
    are drawn from a held-out monomer with fixed seeds; all remaining groups
    are evaluation-only. This is exploratory retrospective transfer.
    """
    out={}
    for monomer in sorted(set(M)):
        src=np.flatnonzero(M!=monomer)
        held=np.flatnonzero(M==monomer)
        a,_=choose_alpha(target,X[src],M[src],y[src],g[src],False)
        model=PropertyHead(target,a,False).fit(X[src],M[src],y[src],g[src])
        zero=metrics(y[held],model.predict(X[held],M[held]),g[held])
        groups=np.unique(g[held])
        if len(groups)<5:
            out[monomer]={"zero_shot":zero,"few_shot":None,"groups":len(groups)}
            continue
        few=[]
        for seed in range(8):
            adaptation=np.random.default_rng(seed).choice(groups,3,replace=False)
            k=held[np.isin(g[held],adaptation)]
            te=held[~np.isin(g[held],adaptation)]
            if set(g[k])&set(g[te]) or set(g[src])&set(g[te]):
                raise ValueError("Transfer group overlap")
            result={"seed":seed,"adaptation_groups":sorted(adaptation),"test_groups":sorted(set(g[te]))}
            result["zero_on_same_test"]=metrics(y[te],model.predict(X[te],M[te]),g[te])
            for name,idx,identity in (("local_3",k,False),("pooled_adapt_3",np.r_[src,k],True)):
                # Fixed alpha: no hyperparameter search on the small adaptation sample.
                head=PropertyHead(target,1.,identity).fit(X[idx],M[idx],y[idx],g[idx])
                result[name]=metrics(y[te],head.predict(X[te],M[te]),g[te])
            few.append(result)
        out[monomer]={"groups":len(groups),"zero_shot":zero,"few_shot":{
            "adaptation_groups":3,"seeds":8,
            "mean_macro_mae":{name:float(np.mean([r[name]["group_macro_mae"] for r in few]))
                              for name in ("zero_on_same_test","local_3","pooled_adapt_3")},
            "runs":few}}
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/monash-transfer"))
    args=parser.parse_args()
    rows,quality=read_dataset(args.root)
    full={}
    with threadpool_limits(limits=1):
        for target in TARGETS:
            X,M,y,g,f=target_table(rows,target)
            report,_=in_domain(target,X,M,y,g,f)
            full[target]={"in_domain":report,"transfer":transfer(target,X,M,y,g)}
        # Demonstrate the multi-property interface only on known monomers.
        # This fit uses all data and is NOT used to calculate held-out metrics.
        model=MaskedPropertyRegressor().fit(rows)
    report={"doi":DOI,"quality_status":quality["quality_status"],"hashes":quality["hashes"],
            "targets":full,"multihead_fit_groups":model.fit_groups,
            "limitations":"Provisional dataset; group CV and cross-monomer transfer within one lab. "
              "No new external validation. NMR/SEC timestamps separate, independently fitted target heads; "
              "Mw>=Mn and D=Mw/Mn are not enforced jointly. No cost or physical optimization claim."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    compact={k:{"rows":v["in_domain"]["rows"],"groups":v["in_domain"]["condition_groups"],
                "in_domain":{m:p["all"] for m,p in v["in_domain"]["models"].items()},
                "transfer":{m:{"zero":r["zero_shot"]["group_macro_mae"],
                               "few_shot":r["few_shot"]["mean_macro_mae"] if r["few_shot"] else None}
                            for m,r in v["transfer"].items()}}
             for k,v in full.items()}
    print("TRANSFER_SUMMARY "+json.dumps(compact,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
