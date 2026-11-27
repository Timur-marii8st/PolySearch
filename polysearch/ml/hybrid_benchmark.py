"""Exploratory nested grouped validation of physics-inspired surrogates."""
import argparse
import csv
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import benchmark as base
from .estimators import fit_estimator
from .hybrid import RateSurrogate, KineticResidualGP
from ..data.monash_source import DOI, digest

NEW = ("rate_ridge", "rate_gp", "kinetic_residual_gp")
MODELS = (*base.MODELS, *NEW)
GRIDS = {
    "rate_ridge": [{"alpha": a, "use_gp": False} for a in (.01,.1,1.,10.)],
    "rate_gp": [{"alpha": a, "length": l, "signal": s, "noise": .25}
                for a in (.1,1.) for l in (.7,1.5) for s in (.25,.7)],
    "kinetic_residual_gp": [{"regularization": r, "length": l, "signal": .03, "noise": .015, "shrink": sh}
                            for r in (.001,.01) for l in (.7,1.5) for sh in (.5,1.)],
}


def fit(name, params, X, y, groups, seed):
    if name.startswith("rate_"):
        return RateSurrogate(**params).fit(X,y,groups)
    if name == "kinetic_residual_gp":
        return KineticResidualGP(**params).fit(X,y,groups)
    return fit_estimator(name,params,X,y,groups,seed)


def tune(name, X, y, groups, seed):
    if name not in GRIDS:
        return base.tune(name,X,y,groups,seed)
    cv = GroupKFold(n_splits=3)
    results = []
    for params in GRIDS[name]:
        pred = np.full(len(y),np.nan)
        for tr, va in cv.split(X,y,groups):
            if set(groups[tr]) & set(groups[va]):
                raise ValueError("Inner group leakage")
            model = fit(name,params,X[tr],y[tr],groups[tr],seed)
            pred[va] = model.predict(X[va])
        results.append({"params":params,"inner_group_macro_mae":base.macro_mae(y,pred,groups)})
    return min(results,key=lambda x:x["inner_group_macro_mae"])["params"],results


def shape_diagnostics(model, X, groups):
    violations, count, zero = 0,0,0
    for g in np.unique(groups):
        cond=X[groups==g][0,:2]
        horizon=float(X[groups==g,2].max())
        grid=np.column_stack([np.tile(cond,(65,1)),np.linspace(0,horizon,65)])
        p=np.clip(model.predict(grid),0,1)
        violations+=int(np.sum(np.diff(p)<-1e-7))
        count+=len(p)-1
        zero+=int(abs(p[0])>1e-7)
    return {"decreasing_steps":violations,"grid_steps":count,"nonzero_at_t0_groups":zero}


def evaluate(X,y,groups,experiments,folds,seed=42,intervals=True,models=MODELS):
    pred={name:np.full(len(y),np.nan) for name in models}
    prob_models=[n for n in ("rate_gp","kinetic_residual_gp") if n in models]
    bounds={n:{kind:(np.full(len(y),np.nan),np.full(len(y),np.nan))
               for kind in ("nominal90","group_conformal80")} for n in prob_models} if intervals else {}
    detail=[]
    shape={n:{"decreasing_steps":0,"grid_steps":0,"nonzero_at_t0_groups":0} for n in models}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        for fold in sorted(set(folds)):
            tr,te=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            if set(groups[tr]) & set(groups[te]):
                raise ValueError("Outer group leakage")
            fs=seed+int(fold)
            fr={"fold":int(fold),"train_groups":sorted(set(groups[tr])),
                "test_groups":sorted(set(groups[te])),"models":{}}
            for name in models:
                params, search=tune(name,X[tr],y[tr],groups[tr],fs)
                model=fit(name,params,X[tr],y[tr],groups[tr],fs)
                p=np.clip(model.predict(X[te]),0,1)
                pred[name][te]=p
                diag=shape_diagnostics(model,X[te],groups[te])
                for k,v in diag.items():
                    shape[name][k]+=v
                fr["models"][name]={"params":params,"inner_search":search,
                    "metrics":base.metrics(y[te],p,groups[te]),"shape":diag}
                if name in bounds:
                    _,_,lo,hi=model.distribution(X[te])
                    bounds[name]["nominal90"][0][te]=lo
                    bounds[name]["nominal90"][1][te]=hi
                    shuffled=np.random.default_rng(fs).permutation(np.unique(groups[tr]))
                    cal_groups=shuffled[:4]
                    ca=tr[np.isin(groups[tr],cal_groups)]
                    fi=tr[~np.isin(groups[tr],cal_groups)]
                    # Tune AGAIN using only proper training groups. Never use
                    # calibration labels in selection or fitting.
                    cparams,_=tune(name,X[fi],y[fi],groups[fi],fs)
                    calibrated=fit(name,cparams,X[fi],y[fi],groups[fi],fs)
                    mu,sd,_,_=calibrated.distribution(X[ca])
                    q=base.conformal_group_quantile(np.abs(y[ca]-mu)/sd,groups[ca],.8)
                    if q is None:
                        raise ValueError("Insufficient calibration groups")
                    mu,sd,_,_=calibrated.distribution(X[te])
                    lo,hi=base.limits(mu,sd,q)
                    bounds[name]["group_conformal80"][0][te]=lo
                    bounds[name]["group_conformal80"][1][te]=hi
                    fr["models"][name]["calibration"]={"fit_groups":sorted(set(groups[fi])),
                        "calibration_groups":sorted(set(groups[ca])),"params":cparams,"q80":q,"q90":None}
            detail.append(fr)
        warning_count=len(caught)
    if any(not np.isfinite(p).all() for p in pred.values()):
        raise ValueError("Incomplete OOF prediction")
    if any(not np.isfinite(a).all() for kinds in bounds.values() for pair in kinds.values() for a in pair):
        raise ValueError("Incomplete OOF intervals")
    report={"rows":len(y),"condition_groups":len(np.unique(groups)),
            "models":{n:base.metrics(y,p,groups) for n,p in pred.items()},
            "shape":shape,"folds":detail,"warning_count":warning_count,
            "intervals":{n:{k:base.interval_metrics(y,*pair,groups) for k,pair in kinds.items()}
                         for n,kinds in bounds.items()}}
    return report,pred,bounds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table",type=Path,default=Path("data/processed/monash-v1/mma_conversion.csv"))
    parser.add_argument("--quality",type=Path,default=Path("data/processed/monash-v1/quality.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/monash-hybrid"))
    args=parser.parse_args()
    rows,X,y,groups,experiments,folds,quality=base.load_table(args.table,args.quality)
    full,pred,bounds=evaluate(X,y,groups,experiments,folds)
    keep=~np.isin(experiments,base.SENSITIVITY)
    sensitivity,_,_=evaluate(X[keep],y[keep],groups[keep],experiments[keep],folds[keep],intervals=False)
    report={"doi":DOI,"table_sha256":digest(args.table),"quality_status":quality["quality_status"],
            "primary":full,"sensitivity":{"excluded":list(base.SENSITIVITY),**sensitivity},
            "environment":{"python":platform.python_version(),**{k:version(k) for k in ("numpy","scipy","scikit-learn","catboost")}},
            "limitations":"Repeated exploratory evaluation of the same provisional archive. "
             "Rate model is an apparent first-order approximation, not a mechanistic simulator. "
             "Model intervals ignore trend estimation uncertainty; rate latent intervals omit measurement noise. "
             "80% group conformal uses four groups, exchangeability not established. No independent prospective validation."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    fields=[*rows[0],*pred,*[f"{n}_{k}_{side}" for n,ks in bounds.items() for k in ks for side in ("lower","upper")]]
    with (args.out/"oof.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for i,row in enumerate(rows):
            writer.writerow({**row,**{n:float(p[i]) for n,p in pred.items()},
                **{f"{n}_{k}_{side}":float(a[i]) for n,ks in bounds.items() for k,pair in ks.items()
                   for side,a in zip(("lower","upper"),pair)}})
    compact={k:v for k,v in report.items() if k not in ("primary","sensitivity")}
    compact.update(primary={k:v for k,v in full.items() if k!="folds"},
                   sensitivity={k:v for k,v in report["sensitivity"].items() if k!="folds"})
    print("HYBRID_SUMMARY "+json.dumps(compact,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
