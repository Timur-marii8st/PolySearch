"""Optional BoTorch research acquisition on a finite feasible candidate pool.

All objectives are supplied in maximization orientation. Costs are known
before selection, positive, and in a common unit. This is q=1 only.
"""
import numpy as np


def candidate_set(X, observed_indices, observed_y, costs, remaining_budget):
    X=np.asarray(X,dtype=float)
    obs=np.asarray(observed_indices)
    y=np.asarray(observed_y,dtype=float)
    costs=np.asarray(costs,dtype=float)
    if X.ndim!=2 or len(X)<2 or X.shape[1]<1 or not np.isfinite(X).all():
        raise ValueError("Invalid pool geometry")
    if len(np.unique(X,axis=0))!=len(X):
        raise ValueError("Duplicate candidate geometry")
    if obs.ndim!=1 or obs.dtype.kind not in "iu" or len(obs)<2 or len(set(obs))!=len(obs) or np.any(obs<0) or np.any(obs>=len(X)):
        raise ValueError("Need distinct observed indices")
    if y.ndim==1:
        y=y[:,None]
    if y.ndim!=2 or y.shape[0]!=len(obs) or y.shape[1] not in (1,2) or not np.isfinite(y).all():
        raise ValueError("Invalid observed objective matrix")
    if costs.shape!=(len(X),) or np.any(costs<=0) or not np.isfinite(costs).all():
        raise ValueError("Costs must be known, finite and strictly positive")
    if not np.isfinite(remaining_budget) or remaining_budget<0:
        raise ValueError("Invalid remaining budget")
    remaining=np.setdiff1d(np.flatnonzero(costs<=remaining_budget+1e-10),obs)
    if not len(remaining):
        raise ValueError("No unmeasured affordable candidate")
    return X,obs,y,costs,remaining


def choose(X, observed_indices, observed_y, costs, remaining_budget, *,
           seed=42, noise_sd=.02, cost_aware=False, reference=None, mc_samples=64):
    """Select by qLogNEI or qLogNEHVI, optionally divide expected gain by cost.

    Only previously observed labels are passed; never fit on archive outcomes
    outside observed_indices. Noisy acquisition uses assumed known Gaussian
    observation variance, not an experimentally estimated noise model.
    """
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X,obs,y,costs,remaining=candidate_set(X,observed_indices,observed_y,costs,remaining_budget)
    if not np.isfinite(noise_sd) or noise_sd<=0 or mc_samples<16:
        raise ValueError("Positive assumed noise and at least 16 MC samples required")
    if y.shape[1]==2:
        reference=np.asarray(reference,dtype=float)
        if reference.shape!=(2,) or not np.isfinite(reference).all():
            raise ValueError("A fixed two-objective maximization reference is required")
    elif reference is not None:
        raise ValueError("Reference point applies only to multiobjective optimization")
    # Pool geometry is available before experiments; no hidden labels used.
    span=np.ptp(X,axis=0)
    Z=torch.as_tensor((X-X.min(axis=0))/np.where(span>0,span,1.),dtype=torch.double)
    Y=torch.as_tensor(y.copy(),dtype=torch.double)
    tx=Z[obs]
    # Standardize is SingleTaskGP's default outcome transform; fixed variance
    # supplied in original objective units and transformed alongside Y.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model=SingleTaskGP(tx,Y,train_Yvar=torch.full_like(Y,noise_sd**2))
        mll=ExactMarginalLogLikelihood(model.likelihood,model)
        fit_gpytorch_mll(mll,optimizer_kwargs={"options":{"maxiter":40}})
        model.eval()
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]),seed=int(seed))
        if Y.shape[-1]==1:
            acq=qLogNoisyExpectedImprovement(model=model,X_baseline=tx,sampler=sampler,
                                             prune_baseline=False,cache_root=False)
            kind="qLogNEI"
        else:
            acq=qLogNoisyExpectedHypervolumeImprovement(model=model,ref_point=reference.tolist(),
                X_baseline=tx,sampler=sampler,prune_baseline=False,cache_root=False)
            kind="qLogNEHVI"
        with torch.no_grad():
            values=acq(Z[remaining].unsqueeze(-2)).reshape(-1).cpu().numpy()
    if cost_aware:
        values=values-np.log(costs[remaining])
    if not np.isfinite(values).all() or len(values)!=len(remaining):
        raise ValueError("Non-finite or malformed acquisition values")
    i=int(np.argmax(values))
    return {"index":int(remaining[i]),"acquisition":kind+("/cost" if cost_aware else ""),
            "log_score":float(values[i]),"estimated_cost":float(costs[remaining[i]])}


def hypervolume2(Y, reference=(0.,0.)):
    """Exact dominated area for two maximized objectives; evaluator only."""
    Y=np.asarray(Y,dtype=float)
    r=np.asarray(reference,dtype=float)
    if Y.ndim!=2 or Y.shape[1]!=2 or r.shape!=(2,) or not np.isfinite(Y).all() or not np.isfinite(r).all():
        raise ValueError("Invalid hypervolume geometry")
    A=Y-r
    A=A[(A>0).all(axis=1)]
    if not len(A):
        return 0.
    A=A[np.argsort(A[:,0])]
    height=np.maximum.accumulate(A[::-1,1])[::-1]
    return float(np.sum(np.diff(np.r_[0.,A[:,0]])*height))
