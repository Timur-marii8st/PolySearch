import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from polysearch.ml.hybrid import RateSurrogate, KineticResidualGP, condition_rates
from polysearch.ml.hybrid_benchmark import evaluate


def fixture():
    groups=np.repeat([f"g{i}" for i in range(10)],5)
    X=np.column_stack([np.repeat(np.linspace(80,100,10),5),
                       np.repeat(np.linspace(.01,.1,10),5),np.tile([1,3,5,7,9],10)])
    rates=.01+.0003*(X[:,0]-80)+.1*X[:,1]
    y=-np.expm1(-rates*X[:,2])
    return X,y,groups,np.repeat(np.arange(10)%5,5)


def test_one_rate_per_condition_and_monotonicity():
    X,y,g,_=fixture()
    cond, rates=condition_rates(X,y,g)
    assert cond.shape==(10,3) and np.isfinite(rates).all()
    # Duplicating the entire curve does not change its fitted apparent rate.
    _,dup=condition_rates(np.tile(X,(2,1)),np.tile(y,2),np.tile(g,2))
    np.testing.assert_allclose(rates,dup,atol=1e-7)
    model=RateSurrogate().fit(X,y,g)
    assert model.group_count_==10
    q=np.column_stack([np.full(30,90),np.full(30,.05),np.linspace(0,12,30)])
    p,sd,lo,hi=model.distribution(q)
    assert p[0]==0 and lo[0]==0 and hi[0]==0
    assert np.all(np.diff(p)>=0) and np.all((0<=p)&(p<=1))
    assert np.all(lo<=p) and np.all(p<=hi) and np.all(sd>0)


def test_bad_condition_rejected():
    X,y,g,_=fixture()
    X[1,0]+=1
    with pytest.raises(ValueError,match="one condition"):
        condition_rates(X,y,g)


def test_residual_model_bounds():
    X,y,g,_=fixture()
    model=KineticResidualGP().fit(X,y,g)
    q=np.array([[90,.04,0],[90,.04,5]])
    p,sd,lo,hi=model.distribution(q)
    assert p[0]==0
    assert np.all((0<=lo)&(hi<=1)&(lo<=p)&(p<=hi))


def test_nested_calibration_disjoint(monkeypatch):
    import polysearch.ml.hybrid_benchmark as b
    monkeypatch.setitem(b.GRIDS,"rate_gp",[{"alpha":1.,"length":1.}])
    X,y,g,f=fixture()
    with threadpool_limits(limits=1):
        report,pred,bounds=evaluate(X,y,g,g,f,models=("rate_gp",))
    assert np.isfinite(pred["rate_gp"]).all()
    for fold in report["folds"]:
        cal=fold["models"]["rate_gp"]["calibration"]
        assert not set(fold["test_groups"]) & set(cal["fit_groups"])
        assert not set(fold["test_groups"]) & set(cal["calibration_groups"])
        assert not set(cal["fit_groups"]) & set(cal["calibration_groups"])
        assert cal["q90"] is None
