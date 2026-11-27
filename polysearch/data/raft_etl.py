"""Conservative extraction of observed RAFT workbook labels, not author-curated replication."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .raft_source import DOI
from .monash_source import digest

WORKBOOK_SHA = "9c40f72053e66a7a6c4bdd5d14a92e2af1eb9a89470dc1045824c6f6f8741060"
# Explicit positions (1-indexed), verified against the pinned workbook header.
TIME_BLOCKS = {0:(4,15),1:(16,25),2:(26,35),4:(36,45),
               6:(46,None),8:(54,63),10:(64,None),15:(72,81)}
FIELDS = ["source_doi","experiment_id","condition_group","monomer_code","raft_agent_code","solvent_code",
          "reactor","nominal_time_h","conversion_fraction","mn_g_mol","mw_g_mol","dispersity",
          "author_note","author_use_ai","selection_basis","source_sheet","source_row","source_columns","workbook_sha256"]


def number(value):
    # Reject strings (including NA/ooc/#VALUE!), booleans and nonfinite numbers.
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return None
    return float(value)


def norm(value):
    return " ".join(str(value or "").strip().lower().split())


def selection(note,flag):
    n=norm(note)
    explicit_yes=n=="take for ai" or flag==1 or norm(flag) in {"yes","true"}
    explicit_no="do not take" in n or "discard" in n or flag==0 or norm(flag) in {"no","false"}
    if explicit_no:
        return "explicit_rejection"
    if explicit_yes:
        return "explicit_positive"
    if not n and flag is None:
        return "unmarked_provisional"
    return "ambiguous_flag"


def selected(note,flag):
    return selection(note,flag)=="explicit_positive"


def decode(sample):
    parts=str(sample).strip().split("-")
    if len(parts)!=7 or parts[0]!="MRG" or not parts[1].isdigit() or not parts[4].isdigit() or not parts[5].isalpha() or not parts[6].isalpha():
        raise ValueError("Unrecognized sample determiner")
    return parts[4],parts[5],parts[6]


def header_check(rows):
    a,b,_,units=rows[:4]
    if norm(b[1])!="sample determiner" or norm(b[90])!="use data for ai":
        raise ValueError("RAFT workbook header changed")
    for t,(start,conv) in TIME_BLOCKS.items():
        i=start-1
        if norm(a[i])!=f"t={t}h" or [norm(x) for x in b[i:i+3]]!=["mn","mw","ð"] or [norm(x) for x in units[i:i+2]]!=["[g/mol]","[g/mol]"]:
            raise ValueError("RAFT time block header changed")
        if conv is not None and norm(b[conv-1])!="conversion":
            raise ValueError("RAFT conversion column changed")


def sec_flagged(values):
    # The first column is a tailing/fronting ratio or False; the remaining
    # flags are categorical indicators. Treat unknown nonempty text conservatively.
    first,*other=values
    if isinstance(first,(int,float)) and not isinstance(first,bool):
        bad=not .7<float(first)<1.79
    else:
        bad=norm(first) not in {"","false","no","0"}
    return bad or any(norm(v) not in {"","false","no","0"} for v in other)


def observations(rows,include_unmarked=False):
    header_check(rows)
    counts=Counter()
    result=[]
    id_counts=Counter(str(row[1]).strip() for row in rows[4:] if row[1] and str(row[1]).startswith("MRG-"))
    for line,row in enumerate(rows[4:],5):
        sample=row[1]
        if not sample or not str(sample).startswith("MRG-"):
            continue
        counts["sample_rows"]+=1
        try:
            mon,agent,solvent=decode(sample)
        except ValueError:
            counts["unrecognized_id"]+=1
            continue
        basis=selection(row[0],row[90])
        counts["flag_"+basis]+=1
        if basis!="explicit_positive" and not (include_unmarked and basis=="unmarked_provisional"):
            counts["not_selected"]+=1
            continue
        if id_counts[str(sample).strip()]!=1:
            counts["duplicate_id_rows_quarantined"]+=1
            continue
        counts["selected_rows"]+=1
        group=hashlib.sha256(json.dumps([mon,agent,solvent],separators=(",",":")).encode()).hexdigest()[:16]
        for t,(start,conv_col) in TIME_BLOCKS.items():
            i=start-1
            if t==0:
                # The workbook uses zero conversion as a reference convention;
                # do not count reference/initial mass as independent target observations.
                continue
            c=number(row[conv_col-1]) if conv_col else None
            if c is not None and (not 0<=c<=1 or number(row[conv_col-2]) is None):
                counts["conversion_invalid_or_no_integral"]+=1
                c=None
            mn,mw,d=map(number,row[i:i+3])
            if sec_flagged(row[i+3:i+8]):
                if any(v is not None for v in (mn,mw,d)):
                    counts["sec_flagged_timepoints"]+=1
                mn=mw=d=None
            if mn is not None and not 0<mn<=100000:
                mn=None
            if mw is not None and not 0<mw<=100000:
                mw=None
            if d is not None and d<1:
                d=None
            if mn is not None and mw is not None and (mw<mn or d is not None and abs(d-mw/mn)>max(.05,.05*mw/mn)):
                counts["sec_inconsistent"]+=1
                mn=mw=d=None
            if all(v is None for v in (c,mn,mw,d)):
                continue
            result.append({"source_doi":DOI,"experiment_id":sample,"condition_group":group,
                "monomer_code":mon,"raft_agent_code":agent,"solvent_code":solvent,"reactor":row[2],
                "nominal_time_h":t,"conversion_fraction":c,"mn_g_mol":mn,"mw_g_mol":mw,"dispersity":d,
                "author_note":row[0] or "","author_use_ai":row[90] if row[90] is not None else "",
                "selection_basis":basis,
                "source_sheet":"Data","source_row":line,"source_columns":f"{start}:{start+7};conversion={conv_col}",
                "workbook_sha256":WORKBOOK_SHA})
    return result,counts


def prepare(root,out):
    from openpyxl import load_workbook
    root,out=Path(root),Path(out)
    info=json.loads((root/"inventory.json").read_text())
    if info["doi"]!=DOI:
        raise ValueError("Unexpected RAFT source DOI")
    bookmeta=[b for b in info["workbooks"] if b["sha256"]==WORKBOOK_SHA and b["archive_member"].endswith("evaluation table (NMR and SEC).xlsx")]
    if len(bookmeta)!=1:
        raise ValueError("Pinned evaluation workbook missing")
    path=root/bookmeta[0]["path"]
    if digest(path)!=WORKBOOK_SHA:
        raise ValueError("Workbook checksum mismatch")
    book=load_workbook(path,read_only=True,data_only=True)
    try:
        rows=list(book["Data"].iter_rows(max_col=91,values_only=True))
    finally:
        book.close()
    explicit,strict_counts=observations(rows)
    obs,counts=observations(rows,include_unmarked=True)
    if not obs:
        raise ValueError("No provisional observed labels")
    out.mkdir(parents=True,exist_ok=True)
    table=out/"observations.csv"
    for file,entries in ((table,obs),(out/"explicit_observations.csv",explicit)):
        with file.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(entries)
    targets=("conversion_fraction","mn_g_mol","mw_g_mol","dispersity")
    report={"doi":DOI,"workbook_sha256":WORKBOOK_SHA,"table_sha256":digest(table),
        "status":"provisional_numeric_qc_nominal_time",
        "strict_explicit_subset":{"counts":dict(strict_counts),"timepoint_rows":len(explicit),
          "experiments":len({r["experiment_id"] for r in explicit}),"sha256":digest(out/"explicit_observations.csv")},
        "selection_basis_rows":dict(Counter(r["selection_basis"] for r in obs)),
        "counts":dict(counts),"timepoint_rows":len(obs),
        "experiments":len({r["experiment_id"] for r in obs}),
        "condition_groups":len({r["condition_group"] for r in obs}),
        "monomer_codes":sorted({r["monomer_code"] for r in obs}),
        "raft_agent_codes":sorted({r["raft_agent_code"] for r in obs}),
        "solvent_codes":sorted({r["solvent_code"] for r in obs}),
        "targets":{t:{"observed_rows":sum(r[t] is not None for r in obs),
                       "groups":len({r["condition_group"] for r in obs if r[t] is not None})} for t in targets},
        "limitations":"Explicit-positive and unmarked rows form provisional numeric-QC candidates; explicit rejection and ambiguous flags excluded. "
        "Duplicate IDs quarantined. Strict explicit subset preserved separately. "
        "Not a reproduction of the published 234 curated curves. Nominal sampling times, not corrected exact times. "
        "Source codes retained; no imputed labels, no pooling with Monash, no cross-source accuracy claim."}
    (out/"quality.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("RAFT_QC "+json.dumps(report,allow_nan=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/raw/raft-v1"))
    parser.add_argument("--out",type=Path,default=Path("data/processed/raft-v1"))
    args=parser.parse_args()
    prepare(args.root,args.out)


if __name__=="__main__":
    main()
