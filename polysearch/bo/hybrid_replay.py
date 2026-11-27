"""Compare structured rate policies with the historical finite-pool baselines."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import policies as base
from .replay import benchmark, endpoint_pool, FLAGGED, ENDPOINT_MIN
from ..ml.benchmark import load_table
from ..ml.hybrid import RateSurrogate, conversion
from ..data.monash_source import digest

POLICIES = (*base.POLICIES, "rate_ridge_greedy", "rate_gp_ei")


def choose(policy, X, observed_indices, observed_y, rng):
    if policy in base.POLICIES:
        return base.choose(policy,X,observed_indices,observed_y,rng)
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    observed=np.asarray(observed_indices,dtype=int)
    y=np.asarray(observed_y,dtype=float)
    remaining=np.setdiff1d(np.arange(len(X)),observed)
    if not len(remaining):
        raise ValueError("Pool exhausted")
    # Endpoint approximation: all actual window medians are assigned 6 min,
    # consistent with the earlier archive replay. No unobserved curve is used.
    design=np.column_stack([X,np.full(len(X),ENDPOINT_MIN)])
    model=RateSurrogate(alpha=1.,use_gp=policy=="rate_gp_ei",length=1.,signal=.5,noise=.25)
    model.fit(design[observed],y,np.arange(len(y)))
    if policy=="rate_ridge_greedy":
        score=model.predict(design[remaining])
    else:
        mean,sd=model.log_rate(design[remaining])
        samples=conversion(mean[:,None]+sd[:,None]*rng.normal(size=(1,128)),ENDPOINT_MIN)
        score=np.maximum(samples-y.max(),0).mean(axis=1)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite acquisition")
    tied=np.flatnonzero(np.isclose(score,score.max(),atol=1e-12,rtol=1e-10))
    return int(remaining[int(rng.choice(tied))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table",type=Path,default=Path("data/processed/monash-v1/mma_conversion.csv"))
    parser.add_argument("--quality",type=Path,default=Path("data/processed/monash-v1/quality.json"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/monash-hybrid-bo"))
    args=parser.parse_args()
    rows,*_,quality=load_table(args.table,args.quality)
    results,all_traces={},[]
    with threadpool_limits(limits=1):
        for name,exclude in (("primary",()),("sensitivity",FLAGGED)):
            items,omitted=endpoint_pool(rows,exclude=exclude)
            report,traces=benchmark(items,policies=POLICIES,chooser=choose)
            results[name]={**report,"omitted":omitted}
            all_traces.extend({"scenario":name,**r} for r in traces)
    full={"table_sha256":digest(args.table),"quality_status":quality["quality_status"],
          "scenarios":results,
          "limitations":"Same previously inspected provisional archive; equal query costs assumed. "
          "Fixed engineering priors, no hidden-label fitting. Rate model uses endpoint only, not full curves. "
          "No independent scientific or physical savings claim."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(full,indent=2,allow_nan=False),encoding="utf-8")
    with (args.out/"traces.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_traces[0]))
        writer.writeheader()
        writer.writerows(all_traces)
    compact={n:{"pool_size":r["pool_size"],"oracle":r["oracle_best_observed"],
                "policies":{p:{"curve":[v for v in m["curve"] if v["budget"] in (3,5,8,10)],
                               "target":m["target"],"near_optimum":m["near_optimum"]}
                            for p,m in r["policies"].items()}}
             for n,r in results.items()}
    print("HYBRID_BO_SUMMARY "+json.dumps({"table_sha256":full["table_sha256"],"scenarios":compact},allow_nan=False),flush=True)


if __name__=="__main__":
    main()
