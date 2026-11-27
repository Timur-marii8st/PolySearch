"""Group-based research design and budget accounting; no experiment execution."""
from decimal import Decimal, ROUND_CEILING
import json
import argparse
from pathlib import Path

import numpy as np


POLICIES=("random","maximin","ridge_variance")


def minimum_calibration_groups(coverage):
    """Minimum n for a finite split-conformal order statistic.

    This is only a mathematical rank condition, not sufficient data for
    useful precision, exchangeability or a verified coverage claim.
    """
    if isinstance(coverage,bool) or not isinstance(coverage,(int,float)) or not 0<coverage<1:
        raise ValueError("Coverage must be a number in (0,1)")
    c=Decimal(str(coverage))
    return int((c/(1-c)).to_integral_value(rounding=ROUND_CEILING))


def ordered_groups(geometry,initial,policy,seed,budget,regularization=1.):
    """Select unique group indices using only declared pre-measurement features.

    The entire candidate geometry may be known, but never labels, measured
    sampling times, or outcomes of held-out test groups. Ridge variance is
    a fixed-prior D-optimal design heuristic, not learned chemical uncertainty.
    """
    X=np.asarray(geometry,float)
    if X.ndim!=2 or len(X)<2 or not np.isfinite(X).all() or X.shape[1]<1:
        raise ValueError("Invalid group geometry")
    if policy not in POLICIES or not np.isfinite(regularization) or regularization<=0:
        raise ValueError("Invalid policy or prior")
    ids=list(initial)
    if (any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in ids)
            or not len(ids)<=budget<=len(X) or not ids or len(set(ids))!=len(ids)
            or any(i<0 or i>=len(X) for i in ids)):
        raise ValueError("Invalid initial groups or budget")
    rng=np.random.default_rng(seed)
    span=np.ptp(X,axis=0)
    Z=(X-X.min(axis=0))/np.where(span>1e-12,span,1.)
    # Scale the geometry dimension so distance does not inflate with descriptor count.
    Z=Z/np.sqrt(Z.shape[1])
    A=np.column_stack([np.ones(len(Z)),Z])
    while len(ids)<budget:
        remaining=np.setdiff1d(np.arange(len(X)),ids)
        if policy=="random":
            choice=int(rng.choice(remaining))
        else:
            if policy=="maximin":
                score=np.linalg.norm(Z[remaining,None,:]-Z[np.asarray(ids)[None,:],:],axis=2).min(axis=1)
            else:
                precision=regularization*np.eye(A.shape[1])+A[ids].T@A[ids]
                solved=np.linalg.solve(precision,A[remaining].T).T
                score=np.log1p(np.maximum(0.,np.sum(A[remaining]*solved,axis=1)))
            tied=np.flatnonzero(np.isclose(score,score.max(),rtol=1e-10,atol=1e-12))
            choice=int(remaining[int(rng.choice(tied))])
        ids.append(choice)
    return ids


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum or value>10**12:
        raise ValueError(f"{name} must be an integer between {minimum} and 10^12")
    return value


def plan_budget(request):
    """Budget three mutually exclusive group roles. No inferred costs or savings."""
    if not isinstance(request,dict) or set(request)!={"currency","budget_minor","coverage","phases"}:
        raise ValueError("Expected currency, budget_minor, coverage, phases")
    currency=request["currency"]
    if not isinstance(currency,str) or len(currency)!=3 or not currency.isascii() or not currency.isalpha() or currency!=currency.upper():
        raise ValueError("Use a three-letter currency code")
    budget=_integer(request["budget_minor"],"budget_minor")
    coverage=request["coverage"]
    minimum=minimum_calibration_groups(coverage)
    phases=request["phases"]
    if not isinstance(phases,dict) or set(phases)!={"train","calibration","validation"}:
        raise ValueError("Exactly train, calibration and validation phases required")
    result={}
    for role in ("train","calibration","validation"):
        q=phases[role]
        if not isinstance(q,dict) or set(q)!={"existing_groups","target_groups","cost_per_new_group_minor"}:
            raise ValueError("Explicit group counts and costs required for "+role)
        existing=_integer(q["existing_groups"],role+" existing")
        target=_integer(q["target_groups"],role+" target",1)
        cost=_integer(q["cost_per_new_group_minor"],role+" cost",1)
        if role=="calibration" and target<minimum:
            raise ValueError(f"Calibration target must be at least {minimum} groups for finite rank")
        additional=max(target-existing,0)
        result[role]={"existing_groups":existing,"target_groups":target,"additional_groups":additional,
                      "cost_per_new_group_minor":cost,"additional_cost_minor":additional*cost}
    required=sum(v["additional_cost_minor"] for v in result.values())
    return {"currency":currency,"budget_minor":budget,"coverage":coverage,
            "minimum_calibration_groups_finite_rank":minimum,"phases":result,
            "additional_groups":sum(v["additional_groups"] for v in result.values()),
            "required_budget_minor":required,"shortfall_minor":max(0,required-budget),
            "unallocated_budget_minor":max(0,budget-required),"within_budget":required<=budget,
            "status":"budget_feasible_not_scientifically_validated" if required<=budget else "revise_budget_or_scope",
            "instructions":["Confirm costs and independent group definition with the laboratory.",
                "Lock disjoint proper-training, calibration and validation assignments before collecting labels.",
                "Select training groups only from technologist-approved candidates.",
                "Keep calibration and validation outcomes inaccessible during model selection.",
                "Assess observed error, interval width and coverage on independent validation before any deployment."],
            "limitations":"Costs are user assumptions, in minor units; counts must be disjoint, target-specific "
            "and reviewed. This is arithmetic planning, not a power calculation, validated sample size, "
            "chemical operating plan or evidence of savings."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.request.stat().st_size>100_000:
        raise ValueError("Request too large")
    result=plan_budget(json.loads(args.request.read_text(encoding="utf-8")))
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:
            f.write(text)
    else:
        print(text,end="")


if __name__=="__main__":
    main()
