"""Research-only applicability checks and bounded molecular extrapolation.

No guard here establishes process safety or calibrated coverage on new chemistry.
All support statistics, labels, and residuals used by a fitted object must be
from the declared proper-training/calibration groups, never the outer test.
"""
from functools import lru_cache

import numpy as np
from .molecular import PropertyModel, TARGETS, molecular_features, log_labels, conformal_quantile


@lru_cache(maxsize=256)
def fingerprint(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    canonical,_=molecular_features(smiles)
    return rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024).GetFingerprint(Chem.MolFromSmiles(canonical))


def similarity(query, training):
    from rdkit import DataStructs
    if not len(training):
        raise ValueError("No reference molecules")
    return float(max(DataStructs.BulkTanimotoSimilarity(fingerprint(query),
                       [fingerprint(s) for s in sorted(set(training))])))


class BoundedMolecularModel:
    """Clip molecular descriptors to training ranges, not measured properties.

    Constant descriptor dimensions are mapped to the observed constant. The
    coupled linear model is still experimental; clipping is not proof of OOD
    accuracy. Process coordinates remain unmodified; the domain guard decides
    whether a prediction may be shown.
    """
    def __init__(self, alpha=1., molecular=True):
        self.alpha,self.molecular=alpha,molecular

    @staticmethod
    def _desc(smiles):
        return np.asarray([molecular_features(s)[1] for s in smiles],float)

    def fit(self,X,S,C,y,g):
        X=np.asarray(X,float)
        if self.molecular:
            d=self._desc(S)
            self.low,self.high=d.min(axis=0),d.max(axis=0)
        self.model=PropertyModel("numeric",True,self.alpha).fit(self.features(X,S),S,C,y,g)
        return self

    def features(self,X,S):
        X=np.asarray(X,float)
        if not self.molecular:
            return X
        return np.column_stack([X,np.clip(self._desc(S),self.low,self.high)])

    def predict(self,X,S,C):
        return self.model.predict(self.features(X,S),S,C)


class Domain:
    """Target-specific support for exactly known molecules and context levels.

    Distances use each target/molecule's observed process ranges. Conditions
    outside those ranges (plus 10% engineering tolerance) are refused. Constant
    dimensions must match. A known molecule requires at least three groups with
    observed labels of the particular target. This is a conservative heuristic.
    """
    def __init__(self,min_groups=3,range_pad=.1,max_distance=.5):
        if min_groups<2 or range_pad<0 or max_distance<=0:
            raise ValueError("Invalid domain thresholds")
        self.min_groups,self.range_pad,self.max_distance=min_groups,range_pad,max_distance

    def fit(self,X,S,C,y,g):
        X=np.asarray(X,float); S=np.asarray([molecular_features(s)[0] for s in S])
        y=log_labels(y)
        C=np.asarray(C,str).reshape(len(X),-1)
        g=np.asarray(g)
        if len(X)!=len(y) or len(X)!=len(S) or len(X)!=len(g) or not np.isfinite(X).all():
            raise ValueError("Invalid support data")
        self.molecules=sorted(set(S))
        self.support={}
        for mol in self.molecules:
            for j in range(len(TARGETS)):
                mask=(S==mol)&np.isfinite(y[:,j])
                if not mask.any():
                    continue
                z=X[mask];low,high=z.min(axis=0),z.max(axis=0)
                self.support[mol,j]={"groups":len(set(g[mask])),"low":low,"high":high,
                    "points":np.unique(z,axis=0),
                    "levels":[set(C[mask,k]) for k in range(C.shape[1])]}
        self.n_features=X.shape[1];self.n_context=C.shape[1]
        return self

    def assess(self,X,S,C):
        X=np.asarray(X,float); C=np.asarray(C,str).reshape(len(X),-1)
        if X.ndim!=2 or X.shape[1]!=self.n_features or len(S)!=len(X) or C.shape[1]!=self.n_context or not np.isfinite(X).all():
            raise ValueError("Invalid query features")
        allowed=np.zeros((len(X),len(TARGETS)),bool)
        reasons=np.empty(allowed.shape,object)
        nearest=[]
        for i,(x,s,c) in enumerate(zip(X,S,C)):
            mol=molecular_features(s)[0]
            nearest.append(similarity(mol,self.molecules))
            for j in range(len(TARGETS)):
                p=self.support.get((mol,j))
                if mol not in self.molecules:
                    reason="unseen_molecule"
                elif p is None or p["groups"]<self.min_groups:
                    reason="insufficient_target_groups"
                elif any(value not in levels for value,levels in zip(c,p["levels"])):
                    reason="unseen_context_level"
                else:
                    span=p["high"]-p["low"]
                    constant=span<=1e-12
                    tolerance=self.range_pad*span+1e-10
                    if np.any(x<p["low"]-tolerance) or np.any(x>p["high"]+tolerance):
                        reason="outside_process_range"
                    else:
                        scale=np.where(constant,1.,span)
                        distance=np.linalg.norm((p["points"]-x)/scale,axis=1).min()/np.sqrt(len(x))
                        reason="sparse_process_support" if distance>self.max_distance else "supported"
                reasons[i,j]=reason
                allowed[i,j]=reason=="supported"
        return allowed,reasons,np.asarray(nearest)


class MarginalGroupCalibration:
    """One max absolute transformed residual per observed group and target.

    Finite quantiles are computed independently for each target. Unlike the
    v3 simultaneous joint interval, these are marginal per-target intervals;
    they do not have a simultaneous four-target guarantee.
    """
    def fit(self,y,p,g):
        yy,pp=log_labels(y),log_labels(p)
        g=np.asarray(g)
        self.scores=[]
        for j in range(len(TARGETS)):
            ok=np.isfinite(yy[:,j])
            self.scores.append(np.asarray([np.max(np.abs(yy[ok&(g==key),j]-pp[ok&(g==key),j]))
                                            for key in np.unique(g[ok])]))
        return self

    def interval(self,p,coverage=.9):
        z=log_labels(p)
        lo=np.full_like(z,np.nan);hi=np.full_like(z,np.nan)
        for j,score in enumerate(self.scores):
            q=conformal_quantile(score,coverage)
            if q is None:
                continue
            a,b=z[:,j]-q,z[:,j]+q
            if j==0:
                lo[:,j],hi[:,j]=np.clip(a,0,1),np.clip(b,0,1)
            else:
                if np.any(b>700):
                    continue
                lo[:,j],hi[:,j]=np.exp(np.maximum(a,-700)),np.exp(b)
                if j==3:
                    lo[:,j]=np.maximum(lo[:,j],1)
        return lo,hi


def width_gate(lo,hi,conversion_width,mass_ratio,dispersity_width):
    """User-supplied tolerance on interval width, not physical acceptance limits."""
    if not 0<conversion_width<=1 or mass_ratio<=1 or dispersity_width<=0:
        raise ValueError("Positive explicit width tolerances required")
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    if lo.shape!=hi.shape or lo.ndim!=2 or lo.shape[1]!=4:
        raise ValueError("Expected interval matrices")
    ok=np.isfinite(lo)&np.isfinite(hi)&(lo<=hi)
    ok[:,0]&=(hi[:,0]-lo[:,0])<=conversion_width
    for j in (1,2):
        ok[:,j]&=(lo[:,j]>0)&(np.log(np.maximum(hi[:,j],1e-300))-np.log(np.maximum(lo[:,j],1e-300))<=np.log(mass_ratio))
    ok[:,3]&=(hi[:,3]-lo[:,3])<=dispersity_width
    return ok


def selective_predictions(p,lo,hi,domain_ok,reasons,*,conversion_width,mass_ratio,dispersity_width):
    """Fail closed: unavailable targets are null, never silently replaced by zero."""
    p=np.asarray(p,float)
    domain_ok=np.asarray(domain_ok,bool)
    reasons=np.asarray(reasons,object).copy()
    good=width_gate(lo,hi,conversion_width,mass_ratio,dispersity_width)
    if p.shape!=good.shape or domain_ok.shape!=good.shape or reasons.shape!=good.shape:
        raise ValueError("Shape mismatch")
    finite=np.isfinite(lo)&np.isfinite(hi)
    good&=np.isfinite(p)
    reasons[domain_ok&~finite]="interval_unavailable"
    reasons[domain_ok&finite&~good]="interval_too_wide"
    eligible=domain_ok&good
    result=[]
    for i in range(len(p)):
        row={}
        for j,t in enumerate(TARGETS):
            row[t]={"status":"research_supported" if eligible[i,j] else "abstain",
                    "reason":str(reasons[i,j]),
                    "prediction":float(p[i,j]) if eligible[i,j] else None,
                    "interval":[float(lo[i,j]),float(hi[i,j])] if eligible[i,j] else None}
        result.append(row)
    return result,eligible
