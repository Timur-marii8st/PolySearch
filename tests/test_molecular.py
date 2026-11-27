import numpy as np
import pytest

from polysearch.ml.molecular import PropertyModel, consistency, conformal_quantile, JointGroupCalibration, molecular_features


def fixture():
    x=np.arange(12,dtype=float)[:,None]/10
    s=np.array(["C=C","C=CC"]*6)
    c=np.empty((12,0),str)
    g=np.array([f"g{i//2}" for i in range(12)])
    mn=1000+100*x[:,0]
    d=1.2+.05*x[:,0]
    y=np.column_stack([.1+.2*x[:,0],mn,mn*d,d])
    y[0,1]=np.nan
    y[1,2]=np.nan
    y[2,3]=np.nan
    y[3,0]=np.nan
    return x,s,c,y,g


def test_coupled_masked_mass():
    x,s,c,y,g=fixture()
    model=PropertyModel("numeric",True,.1).fit(x,s,c,y,g)
    pred=model.predict(x,s,c)
    assert np.isfinite(pred).all()
    assert np.all((0<=pred[:,0])&(pred[:,0]<=1))
    assert np.all(pred[:,2]>=pred[:,1])
    np.testing.assert_allclose(pred[:,2]/pred[:,1],pred[:,3])
    assert consistency(pred)["ratio_error_gt_5pct"]==0


def test_group_calibration():
    x,s,c,y,g=fixture()
    pred=PropertyModel("numeric",True,.1).fit(x,s,c,y,g).predict(x,s,c)
    cal=JointGroupCalibration().fit(y[:4],pred[:4],y[4:],pred[4:],g[4:])
    assert cal.groups==4
    assert cal.interval(pred,.9) is None
    lo,hi=cal.interval(pred,.8)
    assert np.all(lo<=pred) and np.all(pred<=hi)
    assert conformal_quantile([1,2,3,4],.8)==4


def test_rdkit_optional():
    pytest.importorskip("rdkit")
    a,v=molecular_features("C=C(C)C(=O)OC")
    b,w=molecular_features("COC(=O)C(=C)C")
    assert a==b
    np.testing.assert_allclose(v,w)
    with pytest.raises(ValueError):
        molecular_features("not-smiles")
    x,s,c,y,g=fixture()
    m=PropertyModel("molecular",True,.1).fit(x,s,c,y,g)
    p=m.predict(x[:1],np.array(["C=COC(=O)C"]),c[:1])
    assert np.isfinite(p).all()
