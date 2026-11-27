"""Independent provisional RAFT prediction benchmark using nominal hours.

Never combines RAFT with Monash. Hold all repeated recipe codes together.
Categories are nominal strings, not numerical chemistry descriptors.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .benchmark import metrics, macro_mae
from ..data.monash_source import digest

TARGETS=("conversion_fraction","mn_g_mol","mw_g_mol","dispersity")


def weights(groups):
    _,idx,counts=np.unique(groups,return_inverse=True,return_counts=True)
    return len(groups)/(len(counts)*counts[idx])


def fit(X,y,g,target,alpha,identity):
    parts=[("time",StandardScaler(),[0])]
    if identity:
        parts.append(("recipe",OneHotEncoder(handle_unknown="ignore"),[1,2,3]))
    model=make_pipeline(ColumnTransformer(parts,sparse_threshold=0),Ridge(alpha=alpha))
    label=y if target=="conversion_fraction" else np.log(y)
    model.fit(X,label,ridge__sample_weight=weights(g))
    return model


def predict(model,X,target):
    y=model.predict(X)
    if target=="conversion_fraction":
        return np.clip(y,0,1)
    y=np.exp(np.clip(y,-20,20))
    return np.maximum(y,1) if target=="dispersity" else y


def tune(X,y,g,target,identity):
    scores=[]
    for a in (.1,1.,10.):
        p=np.full(len(y),np.nan)
        for tr,va in GroupKFold(n_splits=3).split(X,y,g):
            if set(g[tr])&set(g[va]):
                raise ValueError("Inner group leakage")
            p[va]=predict(fit(X[tr],y[tr],g[tr],target,a,identity),X[va],target)
        scores.append((macro_mae(y,p,g),a))
    return min(scores)[1]


def benchmark(rows,target):
    rows=[r for r in rows if r.get(target) not in (None,"")]
    g=np.array([r["condition_group"] for r in rows])
    if len(set(g))<8:
        return {"status":"insufficient_groups","rows":len(rows),"groups":len(set(g))}
    X=np.array([[float(np.log1p(float(r["nominal_time_h"]))),r["monomer_code"],
                 r["raft_agent_code"],r["solvent_code"]] for r in rows],dtype=object)
    y=np.array([float(r[target]) for r in rows])
    predictions={"time_only":np.full(len(y),np.nan),"time_recipe":np.full(len(y),np.nan)}
    folds=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,g):
        if set(g[tr])&set(g[te]):
            raise ValueError("Outer group overlap")
        rec={"train_groups":sorted(set(g[tr])),"test_groups":sorted(set(g[te]))}
        for name in predictions:
            identity=name=="time_recipe"
            a=tune(X[tr],y[tr],g[tr],target,identity)
            predictions[name][te]=predict(fit(X[tr],y[tr],g[tr],target,a,identity),X[te],target)
            rec[name]={"alpha":a}
        folds.append(rec)
    return {"status":"completed","rows":len(y),"groups":len(set(g)),
            "models":{k:metrics(y,p,g) for k,p in predictions.items()},"fold_details":folds}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/processed/raft-v1"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/raft-benchmark"))
    args=parser.parse_args()
    q=json.loads((args.root/"quality.json").read_text())
    table=args.root/"observations.csv"
    if digest(table)!=q["table_sha256"]:
        raise ValueError("Table checksum mismatch")
    with table.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=q["timepoint_rows"]:
        raise ValueError("Row count mismatch")
    with threadpool_limits(limits=1):
        results={t:benchmark(rows,t) for t in TARGETS}
    report={"doi":q["doi"],"table_sha256":q["table_sha256"],"status":q["status"],
            "targets":results,"limitations":"Provisional candidates, nominal times and recipe codes. "
              "Group CV within one published lab archive; not curated author reproduction or external transfer. "
              "No predictions on a new experimental system and no claim of economic savings."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("RAFT_BENCHMARK "+json.dumps({t:{k:v for k,v in r.items() if k!="fold_details"} for t,r in results.items()},allow_nan=False),flush=True)


if __name__=="__main__":
    main()
