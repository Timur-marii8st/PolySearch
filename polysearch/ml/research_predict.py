"""Read-only, selective molecular research CLI. Not an equipment or production API."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import molecular_benchmark as base
from .applicability import Domain, MarginalGroupCalibration, selective_predictions
from .applicability_benchmark import fit, predict, tune
from .molecular import molecular_features, TARGETS
from .transfer_benchmark import design


def process_query(dataset, query):
    """Accept only explicit raw units and one known data-source protocol."""
    if not isinstance(query,dict) or not isinstance(query.get("smiles"),str) or not query["smiles"] or len(query["smiles"])>512:
        raise ValueError("Provide one nonempty SMILES (maximum 512 characters)")
    canonical,_=molecular_features(query["smiles"])
    if dataset=="monash":
        fields={"smiles","temperature_c","ct_cm","residence_min"}
        if set(query)!=fields:
            raise ValueError("Monash query requires only smiles, temperature_c, ct_cm, residence_min")
        raw=[query[k] for k in ("temperature_c","ct_cm","residence_min")]
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) for v in raw):
            raise ValueError("Process values must be numeric")
        v=np.asarray(raw,float)
        if not np.isfinite(v).all() or v[0]<=-273.15 or v[1]<0 or v[2]<=0:
            raise ValueError("Invalid process values")
        return design(v[None,:])[0],canonical,[]
    if dataset=="raft":
        if set(query)!={"smiles","nominal_time_h","raft_agent_code","solvent_code"}:
            raise ValueError("RAFT query requires only smiles, nominal_time_h, raft_agent_code, solvent_code")
        t=query["nominal_time_h"]
        if isinstance(t,bool) or not isinstance(t,(float,int)) or not np.isfinite(t) or t<=0:
            raise ValueError("Positive finite nominal hours required")
        if any(not isinstance(query[k],str) or not query[k] or len(query[k])>32 for k in ("raft_agent_code","solvent_code")):
            raise ValueError("Explicit reagent and solvent codes required")
        return np.array([np.log1p(t)]),canonical,[query["raft_agent_code"],query["solvent_code"]]
    raise ValueError("Unknown source protocol")


def evaluate_queries(data,dataset,queries,widths,coverage=.9):
    if not isinstance(queries,list) or not 1<=len(queries)<=100:
        raise ValueError("Provide 1 to 100 queries")
    if set(widths)!={"conversion_width","mass_ratio","dispersity_width"} or coverage not in (.8,.9):
        raise ValueError("Explicit width tolerances and marginal coverage 0.8 or 0.9 required")
    X,S,C,y,g,*_=data
    keys=np.random.default_rng(42).permutation(np.unique(g))
    ncal=max(9,int(np.ceil(.25*len(keys))))
    ca=np.flatnonzero(np.isin(g,keys[:ncal]))
    fi=np.flatnonzero(~np.isin(g,keys[:ncal]))
    if set(g[fi])&set(g[ca]) or len(set(g[fi]))<6:
        raise ValueError("Insufficient independent proper/calibration groups")
    alpha=tune(data,fi)
    model=fit(data,fi,alpha)
    cal=MarginalGroupCalibration().fit(y[ca],predict(model,data,ca),g[ca])
    domain=Domain().fit(X[fi],S[fi],C[fi],y[fi],g[fi])
    prepared=[process_query(dataset,q) for q in queries]
    x=np.array([p[0] for p in prepared]);smiles=np.array([p[1] for p in prepared])
    context=np.array([p[2] for p in prepared],dtype=str).reshape(len(queries),-1)
    ok,why,near=domain.assess(x,smiles,context)
    p=model.predict(x,smiles,context)
    lo,hi=cal.interval(p,coverage)
    result,_=selective_predictions(p,lo,hi,ok,why,**widths)
    return {"dataset":dataset,"status":"research_only","alpha":alpha,
            "proper_train_groups":len(set(g[fi])),"calibration_groups":len(set(g[ca])),
            "target_calibration_groups":{t:len(cal.scores[j]) for j,t in enumerate(TARGETS)},
            "nominal_marginal_group_coverage":coverage,"width_tolerances":widths,
            "results":[{"canonical_smiles":str(smiles[i]),"nearest_training_tanimoto":float(near[i]),
                        "properties":r} for i,r in enumerate(result)],
            "warning":"Provisional research estimates, not safety approval or operating recommendations. "
            "Marginal intervals do not establish simultaneous or selective coverage. "
            "A refused property is null, not zero; obtain independent measurements."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["monash","raft"],required=True)
    parser.add_argument("--query",type=Path,required=True,help="JSON file with queries and width_tolerances")
    parser.add_argument("--monash",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    parser.add_argument("--raft",type=Path,default=Path("data/processed/raft-v1"))
    parser.add_argument("--raft-raw",type=Path,default=Path("data/raw/raft-v1"))
    parser.add_argument("--registry",type=Path,default=Path("data/molecular/monomers-v1.json"))
    args=parser.parse_args()
    if args.query.stat().st_size>100_000:
        raise ValueError("Query exceeds size limit")
    q=json.loads(args.query.read_text(encoding="utf-8"))
    if not isinstance(q,dict) or set(q)!={"queries","width_tolerances","coverage"}:
        raise ValueError("Expected queries, width_tolerances and coverage")
    registry=base.load_registry(args.registry,args.raft_raw)
    rows,_=base.read_dataset(args.monash) if args.dataset=="monash" else base.read_raft(args.raft)
    data=base.make_data(rows,registry[args.dataset],args.dataset)
    with threadpool_limits(limits=1):
        output=evaluate_queries(data,args.dataset,q["queries"],q["width_tolerances"],q["coverage"])
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
