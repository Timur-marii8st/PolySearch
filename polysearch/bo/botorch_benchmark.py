"""Small exploratory BoTorch archive and synthetic cost/multiobjective smoke."""
import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import policies as base
from .replay import benchmark, endpoint_pool, FLAGGED
from .botorch_backend import choose as bo_choose, candidate_set, hypervolume2
from ..ml.benchmark import load_table
from ..data.monash_source import digest

REAL_POLICIES=("random","ridge_greedy","gp_ei","botorch_qlognei")
REAL_SEEDS=tuple(range(8))
TOY_SEEDS=tuple(range(4))
TOY_BUDGET=16


def real_choose(policy,X,observed_indices,observed_y,rng):
    if policy!="botorch_qlognei":
        return base.choose(policy,X,observed_indices,observed_y,rng)
    return bo_choose(X,observed_indices,observed_y,np.ones(len(X)),1.,
                     seed=int(rng.integers(2**30)),noise_sd=.02)["index"]


def toy_problem():
    # Dimensionless, intentionally conflicting objectives; no chemical claims.
    X=np.array([(a,b) for a in np.linspace(0,1,5) for b in np.linspace(0,1,5)])
    y1=np.exp(-3*((X[:,0]-.85)**2+(X[:,1]-.75)**2))
    y2=np.exp(-3*((X[:,0]-.15)**2+(X[:,1]-.25)**2))
    costs=1+np.floor(3*X[:,0]+X[:,1]).astype(int)
    return X,np.column_stack([y1,y2]),costs


def toy_replay(X,y,costs,initial,policy,seed,budget=TOY_BUDGET):
    # Oracle labels remain local to the evaluator; chooser receives only history.
    ids=list(initial)
    vals=[y[i].copy() for i in ids]
    spent=int(costs[ids].sum())
    if spent>budget:
        raise ValueError("Initial design exceeds budget")
    trace=[{"index":i,"cost":int(costs[i]),"phase":"initial"} for i in ids]
    rng=np.random.default_rng(seed)
    while True:
        remaining=np.setdiff1d(np.flatnonzero(costs<=budget-spent),ids)
        if not len(remaining):
            break
        if policy=="random":
            i=int(rng.choice(remaining))
        else:
            i=bo_choose(X,ids,vals,costs,budget-spent,reference=(0.,0.),noise_sd=.02,
                        cost_aware=policy=="qlognehvi_per_cost",seed=int(rng.integers(2**30)))["index"]
        if i not in remaining:
            raise ValueError("Illegal choice")
        ids.append(i)
        vals.append(y[i].copy())
        spent+=int(costs[i])
        trace.append({"index":i,"cost":int(costs[i]),"phase":"sequential"})
    return {"spent":spent,"queries":len(ids),"hypervolume":hypervolume2(vals),"trace":trace}


def main():
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table",type=Path,default=Path("data/processed/monash-v1/mma_conversion.csv"))
    parser.add_argument("--quality",type=Path,default=Path("data/processed/monash-v1/quality.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/botorch-research"))
    args=parser.parse_args()
    rows,*_,quality=load_table(args.table,args.quality)
    real={}
    with threadpool_limits(limits=1):
        # Paired reduced seed count, because each step fits a new BoTorch model.
        for scenario,exclude in (("primary",()),("sensitivity",FLAGGED)):
            items,omitted=endpoint_pool(rows,exclude=exclude)
            r,traces=benchmark(items,seeds=REAL_SEEDS,budget=8,policies=REAL_POLICIES,chooser=real_choose)
            real[scenario]={"report":r,"traces":traces,"omitted":omitted}
            print("BOTORCH_ARCHIVE_DONE "+scenario,flush=True)
        X,y,costs=toy_problem()
        toy={p:[] for p in ("random","qlognehvi","qlognehvi_per_cost")}
        for seed in TOY_SEEDS:
            initial=np.random.default_rng(seed).choice(len(X),3,replace=False).tolist()
            for policy in toy:
                toy[policy].append({"seed":seed,**toy_replay(X,y,costs,initial,policy,seed+100)})
            print("BOTORCH_TOY_DONE "+str(seed),flush=True)
    metrics={p:{"mean_hypervolume":float(np.mean([r["hypervolume"] for r in runs])),
                "mean_queries":float(np.mean([r["queries"] for r in runs])),
                "mean_spent":float(np.mean([r["spent"] for r in runs]))} for p,runs in toy.items()}
    summary={"table_sha256":digest(args.table),"quality_status":quality["quality_status"],
             "real_seeds":list(REAL_SEEDS),"real_budget":8,
             "real":{n:{"pool_size":v["report"]["pool_size"],"oracle":v["report"]["oracle_best_observed"],
                         "policies":{p:{"best_at5":s["curve"][4]["mean_best"],"best_at8":s["curve"][7]["mean_best"],
                                        "near_at8":s["curve"][7]["near_optimum_hit_fraction"]}
                                     for p,s in v["report"]["policies"].items()}} for n,v in real.items()},
             "toy_seeds":list(TOY_SEEDS),"toy_cost_budget":TOY_BUDGET,"toy_oracle_hypervolume":hypervolume2(y),
             "toy_metrics":metrics,"environment":{"python":platform.python_version(),**{k:version(k) for k in ("torch","botorch","gpytorch","numpy","scipy")}},
             "limitations":"Real archive has only conversion and assumed equal cost; eight seeds are exploratory. "
             "Toy objectives and costs are invented dimensionless functions, not chemical data or monetary savings. "
             "q=1, fixed known noise assumption, no production deployment or independent validation."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps({**summary,"real_detail":real,"toy_traces":toy},indent=2,allow_nan=False),encoding="utf-8")
    print("BOTORCH_SUMMARY "+json.dumps(summary,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
