import numpy as np
import pytest

from polysearch.bo.botorch_backend import candidate_set, hypervolume2, choose
from polysearch.bo.hybrid_replay import choose as hybrid_choose
from polysearch.bo.replay import replay_once


def test_cost_and_geometry_validation():
    X=np.array([[0,0],[0,1],[1,0],[1,1]],float)
    _,_,_,_,remaining=candidate_set(X,[0,1],[.2,.3],[1,1,4,2],2)
    assert remaining.tolist()==[3]
    with pytest.raises(ValueError,match="affordable"):
        candidate_set(X,[0,1],[.2,.3],[1,1,4,2],1)
    with pytest.raises(ValueError,match="strictly positive"):
        candidate_set(X,[0,1],[.2,.3],[1,1,0,2],2)


def test_hypervolume():
    assert hypervolume2([[1,2],[2,1],[.5,.5]])==pytest.approx(3.)
    assert hypervolume2([[-1,1],[1,-1]])==0.


@pytest.mark.parametrize("policy",["rate_gp_ei","rate_ridge_greedy"])
def test_hybrid_archive_labels_hidden(policy):
    X=np.array([[80,.05],[80,.1],[90,.05],[90,.1],[100,.05],[100,.1]])
    y=np.array([.05,.08,.1,.12,.15,.18])
    other=np.array([.05,.08,.1,.8,.9,.7])
    a=replay_once(X,y,[0,1,2],policy,42,4,chooser=hybrid_choose)[0]
    b=replay_once(X,other,[0,1,2],policy,42,4,chooser=hybrid_choose)[0]
    assert a==b and len(set(a))==4


def test_botorch_acquisitions_optional():
    pytest.importorskip("botorch")
    import torch
    torch.set_num_threads(1)
    X=np.array([[0,0],[0,1],[1,0],[1,1],[.5,.5]],float)
    costs=[1,1,1,3,2]
    single=choose(X,[0,1,2],[.2,.4,.5],costs,2,seed=12,mc_samples=32)
    assert single["index"]==4 and single["acquisition"]=="qLogNEI"
    multi=choose(X,[0,1,2],[[.2,.8],[.4,.5],[.5,.4]],costs,2,
                 seed=12,reference=(0.,0.),cost_aware=True,mc_samples=32)
    assert multi["index"]==4 and multi["acquisition"]=="qLogNEHVI/cost"
