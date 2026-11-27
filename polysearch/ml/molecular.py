"""Molecular descriptors and physically coupled mass heads (research only)."""
from functools import lru_cache
import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS=("conversion_fraction","mn_g_mol","mw_g_mol","dispersity")
DESCRIPTORS=("MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds",
             "RingCount","NumAromaticRings","FractionCSP3","HeavyAtomCount","NumHeteroatoms","MolMR")


@lru_cache(maxsize=256)
def molecular_features(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()==0 or len(Chem.GetMolFrags(mol))!=1:
        raise ValueError("Expected one valid monomer SMILES")
    values=tuple(float(getattr(Descriptors,k)(mol)) for k in DESCRIPTORS)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite molecular descriptor")
    return Chem.MolToSmiles(mol,isomericSmiles=True),values


def group_weights(groups):
    _,idx,n=np.unique(groups,return_inverse=True,return_counts=True)
    return 1.0/n[idx]


class Features:
    def __init__(self,mode):
        if mode not in ("numeric","identity","molecular"):
            raise ValueError("Unknown feature mode")
        self.mode=mode

    def _numbers(self,X,smiles):
        X=np.asarray(X,float)
        if X.ndim!=2 or not np.isfinite(X).all():
            raise ValueError("Invalid process features")
        if self.mode=="molecular":
            return np.column_stack([X,np.asarray([molecular_features(s)[1] for s in smiles])])
        return X

    def _categories(self,smiles,context):
        context=np.asarray(context,str).reshape(len(smiles),-1)
        if self.mode=="identity":
            return np.column_stack([smiles,context])
        return context

    def fit(self,X,smiles,context):
        self.scaler=StandardScaler().fit(self._numbers(X,smiles))
        cat=self._categories(smiles,context)
        self.encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(cat) if cat.shape[1] else None
        return self

    def transform(self,X,smiles,context):
        z=self.scaler.transform(self._numbers(X,smiles))
        if self.encoder is not None:
            z=np.column_stack([z,self.encoder.transform(self._categories(smiles,context))])
        return z


def log_labels(y):
    y=np.asarray(y,float)
    if y.ndim!=2 or y.shape[1]!=4 or np.isinf(y).any():
        raise ValueError("Expected four targets with NaN for missing observations")
    if np.any((y[:,0]<0)|(y[:,0]>1)) or np.any(y[:,1:3]<=0) or np.any(y[:,3]<1):
        raise ValueError("Nonphysical label")
    out=y.copy()
    out[:,1:]=np.log(out[:,1:])
    return out


class PropertyModel:
    """Independent Ridge or jointly fitted latent log Mn and log dispersity.

    Joint residuals: log Mn=a, log Mw=a+b, log D=b. Each observed
    target contributes its own group-weighted loss. At inference project b>=0.
    Conversion has a separate head; no imputed labels or shared-time merge.
    """
    def __init__(self,mode="molecular",coupled=True,alpha=1.):
        if alpha<=0:
            raise ValueError("Positive regularization required")
        self.mode,self.coupled,self.alpha=mode,coupled,alpha

    def fit(self,X,smiles,context,y,groups):
        y=log_labels(y)
        groups=np.asarray(groups)
        self.features=Features(self.mode).fit(X,smiles,context)
        Z=self.features.transform(X,smiles,context)
        if any(len(np.unique(groups[np.isfinite(y[:,j])]))<2 for j in range(4)):
            raise ValueError("Need at least two observed groups per target")
        self.heads={}
        for j in range(4 if not self.coupled else 1):
            keep=np.isfinite(y[:,j])
            self.heads[j]=Ridge(alpha=self.alpha).fit(Z[keep],y[keep,j],sample_weight=group_weights(groups[keep]))
        if self.coupled:
            A=np.column_stack([np.ones(len(Z)),Z])
            width=A.shape[1]
            blocks,labels=[],[]
            for j in (1,2,3):
                keep=np.isfinite(y[:,j])
                a=A[keep]
                zero=np.zeros_like(a)
                block=np.column_stack([a,zero] if j==1 else [a,a] if j==2 else [zero,a])
                w=np.sqrt(group_weights(groups[keep]))
                blocks.append(block*w[:,None])
                labels.append(y[keep,j]*w)
            penalty=np.sqrt(self.alpha)*np.eye(width*2)
            penalty[0,0]=penalty[width,width]=0.
            coeff=np.linalg.lstsq(np.vstack([*blocks,penalty]),np.concatenate([*labels,np.zeros(width*2)]),rcond=None)[0]
            self.mass_coef=coeff.reshape(2,width)
        return self

    def predict(self,X,smiles,context):
        z=self.features.transform(X,smiles,context)
        conv=np.clip(self.heads[0].predict(z),0.,1.)
        if self.coupled:
            a,b=(np.column_stack([np.ones(len(z)),z])@self.mass_coef.T).T
            a=np.clip(a,-20,20)
            b=np.clip(b,0,20)
            mn=np.exp(a)
            d=np.exp(b)
            return np.column_stack([conv,mn,mn*d,d])
        logs=np.column_stack([self.heads[j].predict(z) for j in (1,2,3)])
        masses=np.exp(np.clip(logs,-20,20))
        masses[:,2]=np.maximum(masses[:,2],1.)
        return np.column_stack([conv,masses])


def consistency(pred):
    p=np.asarray(pred,float)
    ratio=p[:,2]/p[:,1]
    return {"rows":len(p),"mw_below_mn":int(np.sum(p[:,2]<p[:,1])),
            "ratio_error_gt_5pct":int(np.sum(np.abs(p[:,3]-ratio)>0.05*ratio)),
            "max_ratio_error":float(np.max(np.abs(p[:,3]-ratio)))}


def conformal_quantile(scores,coverage):
    a=np.sort(np.asarray(scores,float))
    k=math.ceil((len(a)+1)*coverage-1e-12)
    return None if k>len(a) or not len(a) else float(a[k-1])


class JointGroupCalibration:
    """One max normalized residual per calibration group across observed targets.

    Log-space errors for positive properties. Scales from proper train only.
    Marginal bounds are not joint feasible draws; group exchangeability assumed.
    """
    def fit(self,train_y,train_pred,cal_y,cal_pred,groups):
        ty,cy=log_labels(train_y),log_labels(cal_y)
        tp,cp=log_labels(train_pred),log_labels(cal_pred)
        self.scale=np.maximum(np.nanmedian(np.abs(ty-tp),axis=0),1e-6)
        if not np.isfinite(self.scale).all():
            raise ValueError("Insufficient proper-train labels")
        residual=np.abs(cy-cp)/self.scale
        groups=np.asarray(groups)
        self.scores=np.array([np.nanmax(residual[groups==g]) for g in np.unique(groups)])
        if not np.isfinite(self.scores).all():
            raise ValueError("Invalid calibration group")
        self.groups=len(self.scores)
        return self

    def interval(self,pred,coverage):
        q=conformal_quantile(self.scores,coverage)
        if q is None:
            return None
        center=log_labels(pred)
        lo,hi=center-q*self.scale,center+q*self.scale
        lo[:,0]=np.clip(lo[:,0],0,1);hi[:,0]=np.clip(hi[:,0],0,1)
        if np.any(hi[:,1:]>700):
            return None
        lo[:,1:]=np.exp(np.maximum(lo[:,1:],-745))
        hi[:,1:]=np.exp(hi[:,1:])
        lo[:,3]=np.maximum(lo[:,3],1.)
        return lo,hi
