import numpy as np
import pytest

from polysearch.ml.applicability import (BoundedMolecularModel, Domain, MarginalGroupCalibration,
    selective_predictions)
from polysearch.ml.molecular import TARGETS


def fixture():
    x=np.arange(12,dtype=float)[:,None]/10
    s=np.array(["C=C"]*12,dtype="U64")
    c=np.array([["A"]]*12)
    g=np.array([f"g{i//2}" for i in range(12)])
    mn=1000+100*x[:,0]
    d=1.2+.05*x[:,0]
    y=np.column_stack([.1+.2*x[:,0],mn,mn*d,d])
    return x,s,c,y,g


def test_marginal_group_quantiles_missing_target():
    x,s,c,y,g=fixture()
    p=y.copy()
    p[:,0]+=.05
    y[:8,1]=np.nan
    cal=MarginalGroupCalibration().fit(y,p,g)
    lo,hi=cal.interval(p,.8)
    assert np.isfinite(lo[:,0]).all()
    assert np.isnan(lo[:,1]).all()  # only two observed calibration groups
    assert np.all(lo[:,0]<=y[:,0]+1e-12) and np.all(hi[:,0]>=y[:,0])


def test_selective_prediction_fail_closed():
    p=np.array([[.4,1000,1200,1.2]])
    lo=np.array([[.1,900,np.nan,1.1]])
    hi=np.array([[.7,1100,np.nan,1.3]])
    ok=np.array([[True,False,True,True]])
    reasons=np.array([["supported","unseen_molecule","supported","supported"]],object)
    rows,accepted=selective_predictions(p,lo,hi,ok,reasons,conversion_width=.2,mass_ratio=3,dispersity_width=.3)
    assert accepted.tolist()==[[False,False,False,True]]
    assert rows[0][TARGETS[0]]["reason"]=="interval_too_wide"
    assert rows[0][TARGETS[1]]["prediction"] is None
    assert rows[0][TARGETS[2]]["reason"]=="interval_unavailable"
    assert rows[0][TARGETS[3]]["prediction"]==1.2


def test_domain_unseen_molecule_context_and_range():
    pytest.importorskip("rdkit")
    x,s,c,y,g=fixture()
    d=Domain().fit(x,s,c,y,g)
    ok,why,sim=d.assess(np.array([[.5],[.5],[.5],[9.]]),
                         np.array(["C=C","C=CC","C=C","C=C"]),
                         np.array([["A"],["A"],["B"],["A"]]))
    assert ok[0].all() and not ok[1:].any()
    assert why[1,0]=="unseen_molecule" and why[2,0]=="unseen_context_level"
    assert why[3,0]=="outside_process_range"
    assert sim[0]==1.


def test_target_specific_support_and_descriptor_clip():
    pytest.importorskip("rdkit")
    x,s,c,y,g=fixture()
    y[:10,1]=np.nan
    d=Domain().fit(x,s,c,y,g)
    ok,why,_=d.assess(x[:1],s[:1],c[:1])
    assert not ok[0,1] and why[0,1]=="insufficient_target_groups"
    x,s,c,y,g=fixture()
    s[6:]="C=CC"
    model=BoundedMolecularModel().fit(x,s,c,y,g)
    query=np.array(["C=CCCCCCCCCC"])
    z=model.features(x[:1],query)[:,1:]
    assert (z>=model.low).all() and (z<=model.high).all()
    p=model.predict(x[:1],query,c[:1])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p[:,2]/p[:,1],p[:,3])


def test_research_query_validation():
    pytest.importorskip("rdkit")
    from polysearch.ml.research_predict import process_query
    q={"smiles":"C=C","temperature_c":90,"ct_cm":.1,"residence_min":5}
    x,s,c=process_query("monash",q)
    assert x.shape==(3,) and s=="C=C" and c==[]
    with pytest.raises(ValueError):
        process_query("monash",{**q,"residence_min":True})
    with pytest.raises(ValueError):
        process_query("monash",{**q,"extra":1})
    with pytest.raises(ValueError):
        process_query("raft",q)
