import csv
import json

import numpy as np
import pytest

from polysearch.data.monash_source import digest
from polysearch.data.monash_multimonomer import sec_rows
from polysearch.data.monash_etl import canonical_condition
from polysearch.ml.transfer_benchmark import PropertyHead, MaskedPropertyRegressor, target_table, transfer


def test_sec_uses_own_time_and_deduplicates(tmp_path):
    p=tmp_path/"extracted"/"test.csv"
    p.parent.mkdir()
    p.write_text("tres,conversion,GPC_number,Mn,Mw,D,tres_GPC\n"
                 "4,0.2,1,1000,1200,1.2,6\n"
                 "5,0.3,1,1000,1200,1.2,6\n"
                 "7,0.4,2,900,800,0.9,8\n"
                 "8,0.4,3,1200,,,9\n")
    sha=digest(p)
    src={"path":"extracted/test.csv","sha256":sha,"experiment_id":"BA-1",
         "temperature_c":80,"ct_cm":.1,"encoding":"utf-8","delimiter":","}
    verified={src["path"]:{"sha256":sha,"columns":{"id":"GPC_number","mn":"Mn","mw":"Mw",
                "dispersity":"D","residence":"tres_GPC","mass_unit":"g/mol","time_unit":"min"}}}
    group=canonical_condition("BA",80,.1)
    rows,rejected,missing=sec_rows(tmp_path,[src],verified,{group:0},"BA")
    assert len(rows)==2 and len(rejected)==2 and not missing
    assert rows[0]["residence_min"]==6 and rows[1]["mw_g_mol"] is None
    assert rows[0]["source_row"]==2
    assert rows[0]["fold"]==0


def test_property_heads_masks_and_unseen_monomer():
    rows=[]
    for i,m in enumerate(("BA","MMA","VA")):
        for j in range(5):
            for t in (2.,4.):
                base={"temperature_c":80+j,"ct_cm":.05+j*.01,"residence_min":t,
                      "monomer":m,"split_group":f"{m}-{j}","fold":j}
                rows.append({**base,"conversion_fraction":.1+.01*j})
                rows.append({**base,"mn_g_mol":1000+100*j,"mw_g_mol":1200+120*j,"dispersity":1.2})
    model=MaskedPropertyRegressor().fit(rows)
    pred=model.predict(np.array([[80,.05,2]]),"BA")
    assert set(pred)=={"conversion_fraction","mn_g_mol","mw_g_mol","dispersity"}
    assert all(v.shape==(1,) and np.isfinite(v).all() for v in pred.values())
    assert pred["dispersity"][0]>=1
    with pytest.raises(ValueError,match="not observed"):
        model.predict(np.array([[80,.05,2]]),"UNKNOWN")
    X,M,y,g,f=target_table(rows,"conversion_fraction")
    assert len(y)==30
    r=transfer("conversion_fraction",X,M,y,g)
    for m in r:
        for run in r[m]["few_shot"]["runs"]:
            assert not set(run["adaptation_groups"])&set(run["test_groups"])
