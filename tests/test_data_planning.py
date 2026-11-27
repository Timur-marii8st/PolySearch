import numpy as np
import pytest

from polysearch.ml.data_planning import minimum_calibration_groups, ordered_groups, plan_budget, POLICIES


@pytest.mark.parametrize("coverage,n",[(.8,4),(.9,9),(.95,19),(.99,99)])
def test_finite_rank(coverage,n):
    assert minimum_calibration_groups(coverage)==n


@pytest.mark.parametrize("policy",POLICIES)
def test_unique_nested_order(policy):
    X=np.array([[0,0],[1,0],[0,1],[1,1],[.5,.5],[.1,.4]])
    a=ordered_groups(X,[0,2],policy,42,5)
    b=ordered_groups(X,[0,2],policy,42,4)
    assert a[:2]==[0,2] and len(set(a))==5 and a[:4]==b


def test_budget_does_not_invent_cost_or_count():
    request={"currency":"RUB","budget_minor":10000,"coverage":.9,"phases":{
        "train":{"existing_groups":3,"target_groups":5,"cost_per_new_group_minor":1000},
        "calibration":{"existing_groups":0,"target_groups":9,"cost_per_new_group_minor":1000},
        "validation":{"existing_groups":0,"target_groups":2,"cost_per_new_group_minor":2000}}}
    r=plan_budget(request)
    assert r["additional_groups"]==13 and r["required_budget_minor"]==15000
    assert r["shortfall_minor"]==5000 and not r["within_budget"]
    request["phases"]["calibration"]["target_groups"]=8
    with pytest.raises(ValueError,match="at least 9"):
        plan_budget(request)


def test_geometry_no_observed_sampling_time():
    pytest.importorskip("rdkit")
    from polysearch.ml.learning_benchmark import group_geometry
    X=np.array([[.1,.2,1.],[.1,.2,3.],[.5,.2,1.],[.5,.2,7.]])
    S=np.array(["C=C","C=C","C=CC","C=CC"])
    C=np.empty((4,0),str);g=np.array(["a","a","b","b"])
    a,z=group_geometry(X,S,C,g)
    X[:,2]=[100,200,300,400]
    b,q=group_geometry(X,S,C,g)
    assert a.tolist()==b.tolist()
    np.testing.assert_array_equal(z,q)


def test_learning_and_calibration_disjoint(monkeypatch):
    pytest.importorskip("rdkit")
    import polysearch.ml.learning_benchmark as b
    monkeypatch.setitem(b.BUDGETS,"monash",(3,5))
    monkeypatch.setattr(b,"SEEDS",(0,))
    monkeypatch.setattr(b,"CAL_SEEDS",(0,))
    monkeypatch.setattr(b,"CAL_SIZES",(4,9))
    n=25
    g=np.repeat([f"g{i}" for i in range(n)],2)
    X=np.column_stack([np.repeat(np.linspace(0,1,n),2),np.tile([1.,2.],n)])
    S=np.repeat(["C=C","C=CC","C=CCC","C=CCCC","C=CCCCC"]*5,2)
    C=np.empty((2*n,0),str)
    y=.1+.1*X[:,0]+.01*X[:,1]
    d=(X,S,C,y,g,np.repeat(np.arange(n)%5,2))
    r=b.learning_curves(d,0,"monash")
    for item in r["trajectories"]:
        assert not set(item["order"])&set(item["test_groups"])
        assert item["order"][:3]==item["initial_groups"]
    c=b.calibration_curve(d,0,"monash")
    for item in c["fold_details"]:
        assert not set(item["fit_groups"])&set(item["calibration_order"])
        assert not set(item["test_groups"])&set(item["calibration_order"])
        assert not set(item["test_groups"])&set(item["fit_groups"])
    assert next(x for x in c["summary"] if x["calibration_groups"]==4 and x["nominal_coverage"]==.9)["whole_group_coverage"] is None
