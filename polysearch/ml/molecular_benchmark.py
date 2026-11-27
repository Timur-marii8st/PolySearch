"""V3 molecular and coupled-mass benchmarks; independent datasets, shared group splits."""
import argparse
import csv
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .molecular import (TARGETS, DESCRIPTORS, PropertyModel, JointGroupCalibration,
                        log_labels, molecular_features, consistency)
from .benchmark import metrics, macro_mae
from .transfer_benchmark import read_dataset, design
from ..data.monash_source import digest
from ..data.raft_etl import WORKBOOK_SHA
from ..data.raft_source import DOI as RAFT_DOI

MODES={"process_context":("numeric",False),"identity":("identity",False),
       "molecular_independent":("molecular",False),"molecular_coupled":("molecular",True)}
ALPHAS=(.1,1.,10.,100.)
SEED=42


def load_registry(path,raft_root):
    from openpyxl import load_workbook
    registry=json.loads(Path(path).read_text(encoding="utf-8"))
    info=json.loads((Path(raft_root)/"inventory.json").read_text())
    matches=[b for b in info["workbooks"] if b["sha256"]==WORKBOOK_SHA]
    if len(matches)!=1 or registry["raft"]["workbook_sha256"]!=WORKBOOK_SHA:
        raise ValueError("Pinned RAFT legend not found")
    bookpath=Path(raft_root)/matches[0]["path"]
    if digest(bookpath)!=WORKBOOK_SHA:
        raise ValueError("RAFT workbook hash mismatch")
    book=load_workbook(bookpath,read_only=True,data_only=True)
    try:
        actual={str(int(r[2])):str(r[0]).strip() for r in book["Legend for Abbreviations"].iter_rows(values_only=True)
                if isinstance(r[2],(int,float)) and not isinstance(r[2],bool) and r[0]}
    finally:
        book.close()
    if set(actual)!=set(registry["raft"]["entries"]) or any(actual[k]!=v[0].strip() for k,v in registry["raft"]["entries"].items()):
        raise ValueError("Molecular identities differ from pinned author legend")
    result={}
    for dataset in ("monash","raft"):
        entries=registry[dataset]["entries"]
        canonical={k:{"name":v[0],"smiles":molecular_features(v[1])[0],
                      "descriptors":dict(zip(DESCRIPTORS,molecular_features(v[1])[1]))} for k,v in entries.items()}
        if len(set(v["smiles"] for v in canonical.values()))!=len(canonical):
            raise ValueError("Duplicate molecular identities")
        result[dataset]=canonical
    return result


def make_data(rows,registry,dataset):
    if dataset=="monash":
        code="monomer"; group="split_group"
        X=design(np.array([[float(r["temperature_c"]),float(r["ct_cm"]),float(r["residence_min"])]
                           for r in rows]))
        context=np.empty((len(rows),0),str)
    else:
        code="monomer_code";group="condition_group"
        X=np.array([[np.log1p(float(r["nominal_time_h"]))] for r in rows])
        context=np.array([[r["raft_agent_code"],r["solvent_code"]] for r in rows])
    M=np.array([r[code] for r in rows])
    smiles=np.array([registry[m]["smiles"] for m in M])
    y=np.array([[float(r[t]) if r.get(t) not in (None,"") else np.nan for t in TARGETS] for r in rows])
    g=np.array([r[group] for r in rows])
    log_labels(y)
    if not np.isfinite(X).all() or np.any(~np.isfinite(y).any(axis=1)):
        raise ValueError("Invalid feature or empty observed-label row")
    for key in np.unique(g):
        if len(np.unique(smiles[g==key]))!=1:
            raise ValueError("Mixed molecules in one reaction group")
    if dataset=="monash":
        folds=np.array([int(r["fold"]) for r in rows])
    else:
        folds=np.full(len(y),-1)
        for k,(_,te) in enumerate(GroupKFold(n_splits=5).split(X,groups=g)):
            folds[te]=k
    if np.any(folds<0) or any(len(np.unique(folds[g==key]))!=1 for key in np.unique(g)):
        raise ValueError("Group leakage across outer folds")
    return X,smiles,context,y,g,folds,M


def model_fit(data,idx,name,alpha):
    X,S,C,y,g,_,_=data
    mode,coupled=MODES[name]
    return PropertyModel(mode,coupled,alpha).fit(X[idx],S[idx],C[idx],y[idx],g[idx])


def predict(model,data,idx):
    X,S,C,*_=data
    p=model.predict(X[idx],S[idx],C[idx])
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prediction")
    return p


def losses(y,p,g):
    yy,pp=log_labels(y),log_labels(p)
    out=[]
    for j in range(len(TARGETS)):
        keep=np.isfinite(yy[:,j])
        if keep.any():
            out.append(macro_mae(yy[keep,j],pp[keep,j],g[keep]))
    return float(np.mean(out))


def tune(data,idx,name):
    X,S,C,y,g,_,_=data
    if len(set(g[idx]))<3:
        raise ValueError("Insufficient proper training groups")
    inner=list(GroupKFold(n_splits=3).split(X[idx],groups=g[idx]))
    result=[]
    for a in ALPHAS:
        p=np.full((len(idx),len(TARGETS)),np.nan)
        for tr,va in inner:
            fit,valid=idx[tr],idx[va]
            if set(g[fit])&set(g[valid]):
                raise ValueError("Inner group leakage")
            model=model_fit(data,fit,name,a)
            p[va]=predict(model,data,valid)
        if not np.isfinite(p).all():
            raise ValueError("Missing validation predictions")
        result.append((losses(y[idx],p,g[idx]),a))
    return min(result)[1]


def target_metrics(y,p,g):
    report={}
    for j,target in enumerate(TARGETS):
        mask=np.isfinite(y[:,j])
        if mask.any():
            report[target]={"rows":int(mask.sum()),"groups":len(set(g[mask])),
                            **metrics(y[mask,j],p[mask,j],g[mask])}
        else:
            report[target]={"rows":0,"groups":0}
    return report


def coverage(y,lo,hi,g):
    present=np.isfinite(y)
    covered=(y>=lo)&(y<=hi)
    group_hit=[bool(covered[g==key][present[g==key]].all()) for key in np.unique(g)]
    per={}
    for j,t in enumerate(TARGETS):
        mask=present[:,j]
        if mask.any():
            width=hi[mask,j]-lo[mask,j]
            # Relative width for positive properties is often more informative.
            relative=width/np.maximum(np.abs(y[mask,j]),1e-12)
            per[t]={"labels":int(mask.sum()),"coverage":float(covered[mask,j].mean()),
                    "median_width":float(np.median(width)),
                    "median_relative_width":float(np.median(relative)) if j else None}
    return {"whole_group_coverage":float(np.mean(group_hit)),"groups":len(group_hit),"targets":per}


def group_cv(data,calibrate=True):
    X,S,C,y,g,f,M=data
    pred={k:np.full_like(y,np.nan) for k in MODES}
    intervals={c:(np.full_like(y,np.nan),np.full_like(y,np.nan)) for c in (.8,.9)}
    fold_reports=[]
    for fold in sorted(set(f)):
        tr,te=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
        if set(g[tr])&set(g[te]):
            raise ValueError("Outer group overlap")
        rec={"fold":int(fold),"train_groups":sorted(set(g[tr])),"test_groups":sorted(set(g[te])),"alphas":{}}
        for name in MODES:
            alpha=tune(data,tr,name)
            pred[name][te]=predict(model_fit(data,tr,name,alpha),data,te)
            rec["alphas"][name]=alpha
        if calibrate:
            # 9 groups is the minimum for a finite 90% split-conformal quantile.
            keys=np.random.default_rng(SEED+int(fold)).permutation(np.unique(g[tr]))
            ncal=max(9,int(np.ceil(.2*len(keys))))
            if len(keys)-ncal<6:
                raise ValueError("Insufficient groups for proper train and calibration")
            ca=tr[np.isin(g[tr],keys[:ncal])]
            fi=tr[~np.isin(g[tr],keys[:ncal])]
            if set(g[fi])&set(g[ca]) or set(g[ca])&set(g[te]):
                raise ValueError("Calibration leakage")
            a=tune(data,fi,"molecular_coupled")
            model=model_fit(data,fi,"molecular_coupled",a)
            cal=JointGroupCalibration().fit(y[fi],predict(model,data,fi),y[ca],predict(model,data,ca),g[ca])
            p=predict(model,data,te)
            rec["calibration"]={"fit_groups":sorted(set(g[fi])),"calibration_groups":sorted(set(g[ca])),
                                "alpha":a,"scale":cal.scale.tolist(),"n":cal.groups}
            for c in intervals:
                bounds=cal.interval(p,c)
                if bounds is not None:
                    intervals[c][0][te],intervals[c][1][te]=bounds
        fold_reports.append(rec)
    if any(not np.isfinite(p).all() for p in pred.values()):
        raise ValueError("Incomplete OOF prediction")
    result={"rows":len(y),"groups":len(set(g)),"models":{k:{"targets":target_metrics(y,p,g),
                     "consistency":consistency(p)} for k,p in pred.items()},"fold_details":fold_reports}
    if calibrate:
        result["joint_intervals"]={}
        for c,(lo,hi) in intervals.items():
            ok=np.isfinite(lo).all(axis=1)&np.isfinite(hi).all(axis=1)
            result["joint_intervals"][str(c)]={"evaluation_rows":int(ok.sum()),"unavailable_rows":int((~ok).sum()),
                **(coverage(y[ok],lo[ok],hi[ok],g[ok]) if ok.any() else {})}
    return result


def nearest_similarity(held,train):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    q=generator.GetFingerprint(Chem.MolFromSmiles(held))
    fps=[generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in sorted(set(train))]
    return float(max(DataStructs.BulkTanimotoSimilarity(q,fps)))


def leave_one_molecule_out(data):
    X,S,C,y,g,f,M=data
    result={}
    predictions={name:np.full_like(y,np.nan) for name in MODES}
    for mol in sorted(set(S)):
        tr,te=np.flatnonzero(S!=mol),np.flatnonzero(S==mol)
        if set(g[tr])&set(g[te]):
            raise ValueError("Molecular holdout leakage")
        record={"codes":sorted(set(M[te])),"canonical_smiles":mol,
                "nearest_training_tanimoto":nearest_similarity(mol,S[tr]),"groups":len(set(g[te])),"models":{}}
        for name in MODES:
            a=tune(data,tr,name)
            p=predict(model_fit(data,tr,name,a),data,te)
            predictions[name][te]=p
            record["models"][name]={"alpha":a,"targets":target_metrics(y[te],p,g[te])}
        result[mol]=record
    # Holdout folds differ by whole monomer, with group-macro metrics over all groups.
    aggregate={name:{"targets":target_metrics(y,p,g),"consistency":consistency(p)}
               for name,p in predictions.items()}
    return {"molecules":len(result),"aggregate":aggregate,"holdouts":result}


def read_raft(root):
    root=Path(root)
    q=json.loads((root/"quality.json").read_text())
    path=root/"observations.csv"
    if q["doi"]!=RAFT_DOI or q["table_sha256"]!=digest(path):
        raise ValueError("Unexpected RAFT processed source")
    with path.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=q["timepoint_rows"]:
        raise ValueError("RAFT row count mismatch")
    return rows,q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monash",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    parser.add_argument("--raft",type=Path,default=Path("data/processed/raft-v1"))
    parser.add_argument("--raft-raw",type=Path,default=Path("data/raw/raft-v1"))
    parser.add_argument("--registry",type=Path,default=Path("data/molecular/monomers-v1.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/molecular-v3"))
    args=parser.parse_args()
    registry=load_registry(args.registry,args.raft_raw)
    monash,mq=read_dataset(args.monash)
    raft,rq=read_raft(args.raft)
    flagged={e for m in mq["monomers"].values() for e in m["flagged_experiments"]}
    result={}
    with threadpool_limits(limits=1):
        for name,rows in (("monash",monash),("raft",raft)):
            data=make_data(rows,registry[name],name)
            result[name]={"group_cv":group_cv(data),"leave_one_molecule_out":leave_one_molecule_out(data)}
            if name=="monash":
                clean=[r for r in rows if r["experiment_id"] not in flagged]
                result[name]["sensitivity"]={"excluded_experiments":sorted(flagged),
                    "group_cv":group_cv(make_data(clean,registry[name],name),calibrate=False)}
            print("MOLECULAR_DATASET_DONE "+name,flush=True)
    report={"registry_sha256":digest(args.registry),"registry":registry,
            "versions":{k:version(k) for k in ("rdkit","numpy","scipy","scikit-learn")},
            "monash_hashes":mq["hashes"],"raft_sha256":rq["table_sha256"],
            "quality":{"monash":mq["quality_status"],"raft":rq["status"]},
            "results":result,
            "limitations":"Research only. Name-derived SMILES, no supplier CAS verification. "
            "RAFT agents and solvents categorical; RAFT times nominal. Both sources provisional, not pooled. "
            "Monomer descriptors are not proof of transfer. Coupled masses enforce algebraic identities, "
            "not chemistry or process safety. Joint group conformal on observed labels; exchangeability "
            "not established, no coverage guarantee for new molecules or adaptive experimental conditions."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    compact={}
    for name,v in result.items():
        cv=v["group_cv"]
        lo=v["leave_one_molecule_out"]
        compact[name]={"rows":cv["rows"],"groups":cv["groups"],"group_cv":cv["models"],
                       "joint_intervals":cv["joint_intervals"],
                       "lomo":lo["aggregate"],"molecules":lo["molecules"],
                       "holdouts":[{"codes":r["codes"],"similarity":r["nearest_training_tanimoto"],
                           "conversion_macro_mae":{m:p["targets"]["conversion_fraction"].get("group_macro_mae")
                                                    for m,p in r["models"].items()}} for r in lo["holdouts"].values()]}
        if "sensitivity" in v:
            compact[name]["sensitivity"]={"excluded":v["sensitivity"]["excluded_experiments"],
                        "rows":v["sensitivity"]["group_cv"]["rows"],
                        "groups":v["sensitivity"]["group_cv"]["groups"],
                        "models":v["sensitivity"]["group_cv"]["models"]}
    print("MOLECULAR_SUMMARY "+json.dumps(compact,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
