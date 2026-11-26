import numpy as np
import pytest

from polysearch.bo.policies import POLICIES, choose, expected_improvement
from polysearch.bo.replay import Archive, endpoint_pool, replay_once, benchmark, summarize


def row(exp, group, temp, ratio, t, conv, line):
    return {"experiment_id": exp, "split_group": group, "temperature_c": temp,
            "ct_cm": ratio, "residence_min": t, "conversion_fraction": conv,
            "source_file": exp + ".csv", "source_row": line, "source_sha256": "fixture"}


def test_endpoint_uses_measured_window_only():
    rows = [row("A", "g1", 80, .05, 5.75, .10, 2), row("A", "g1", 80, .05, 6.25, .14, 3),
            row("A", "g1", 80, .05, 7, .99, 4), row("B", "g2", 90, .05, 7, .5, 2)]
    pool, omitted = endpoint_pool(rows)
    assert len(pool) == 1 and pool[0]["conversion_fraction"] == pytest.approx(.12)
    assert pool[0]["endpoint_measurements"] == 2
    assert omitted == [{"experiment_id": "B", "reason": "no_measured_endpoint_window"}]


def test_archive_forbids_repeat():
    archive = Archive([.1, .2])
    assert archive.reveal(0) == .1
    with pytest.raises(ValueError, match="repeated"):
        archive.reveal(0)
    assert archive.history() == ([0], [.1])


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_receives_no_unseen_labels(policy):
    X = np.array([[0,0], [0,1], [1,0], [1,1], [.4,.6], [.6,.4]])
    y1 = np.array([.1,.2,.3,.4,.5,.6])
    y2 = np.array([.1,.2,.3,.9,.8,.7])
    a = replay_once(X,y1,[0,1,2],policy,42,4)[0]
    b = replay_once(X,y2,[0,1,2],policy,42,4)[0]
    assert a == b and len(set(a)) == 4


def test_budget_counts_initial_and_misses():
    r = summarize([[.02,.03,.05],[.15,.16,.17]], .20)
    assert r["target"]["success_fraction"] == .5
    assert r["target"]["mean_capped_cost_budget_plus_one_for_misses"] == 2.5
    assert r["curve"][0]["mean_best"] == pytest.approx(.085)


def test_benchmark_shared_initial_and_no_repeats():
    items = [{"experiment_id": str(i), "temperature_c": 80+i, "ct_cm": .01*i,
              "conversion_fraction": .02*i} for i in range(7)]
    report, traces = benchmark(items, seeds=(1,2), n_initial=3, budget=5)
    for seed in (1,2):
        orders = [[r["experiment_id"] for r in traces if r["seed"] == seed and r["policy"] == p]
                  for p in POLICIES]
        assert all(order[:3] == orders[0][:3] and len(set(order)) == 5 for order in orders)
    assert all(s["curve"][-1]["budget"] == 5 for s in report["policies"].values())


def test_ei_zero_variance():
    assert expected_improvement([.1,.4],[0,0],.2).tolist() == pytest.approx([0,.2])
