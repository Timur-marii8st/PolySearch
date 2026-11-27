from polysearch.data.raft_etl import observations, TIME_BLOCKS, selected, decode
import pytest


def test_explicit_selection_negative_precedence():
    assert selected("take for AI",None)
    assert selected("",1)
    assert not selected("do not take for AI",1)
    assert not selected("take for AI",0)
    assert not selected("",None)
    assert decode("MRG-046-G-A-1-A-DMF")==("1","A","DMF")


def test_raft_no_imputation_and_flagged_sec():
    a,b,c,d=[[None]*91 for _ in range(4)]
    b[1]="sample determiner"
    b[90]="use data for AI"
    for t,(start,conv) in TIME_BLOCKS.items():
        i=start-1
        a[i]=f"t={t}h"
        b[i:i+3]=["Mn","Mw","Ð"]
        d[i:i+2]=["[g/mol]","[g/mol]"]
        if conv:
            b[conv-1]="conversion"
    row=[None]*91
    row[0]="take for AI";row[1]="MRG-046-G-A-1-A-DMF";row[2]=1
    row[14]=0.
    row[15:18]=[1000,1200,1.2]
    row[23:25]=[2.5,.4]
    row[25:28]=[2000,2400,1.2]
    row[29]="True"
    row[33:35]=[2.0,.6]
    obs,counts=observations([a,b,c,d,row])
    assert len(obs)==2 and counts["selected_rows"]==1
    assert obs[0]["nominal_time_h"]==1 and obs[0]["mn_g_mol"]==1000
    assert obs[1]["nominal_time_h"]==2 and obs[1]["conversion_fraction"]==.6
    assert obs[1]["mn_g_mol"] is None


def test_unmarked_only_in_provisional_mode():
    a,b,c,d=[[None]*91 for _ in range(4)]
    b[1]="sample determiner";b[90]="use data for AI"
    for t,(start,conv) in TIME_BLOCKS.items():
        i=start-1;a[i]=f"t={t}h";b[i:i+3]=["Mn","Mw","Ð"];d[i:i+2]=["[g/mol]","[g/mol]"]
        if conv:b[conv-1]="conversion"
    row=[None]*91;row[1]="MRG-046-G-A-1-A-DMF";row[2]=1
    row[15:18]=[1000,1200,1.2];row[23:25]=[2.5,.4]
    assert observations([a,b,c,d,row])[0]==[]
    obs,_=observations([a,b,c,d,row],include_unmarked=True)
    assert len(obs)==1 and obs[0]["selection_basis"]=="unmarked_provisional"
    assert observations([a,b,c,d,row,row],include_unmarked=True)[0]==[]
