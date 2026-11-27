"""Produce separately timed NMR and SEC observations from the pinned release."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .monash_etl import prepare, FIELDS, finite, canonical_condition, source_path, write_outputs
from .monash_source import DOI, digest

MONOMERS=("BA","MMA","VA")
SEC_FIELDS=["experiment_id","split_group","fold","monomer","design_phase","temperature_c",
            "ct_cm","residence_min","gpc_id","mn_g_mol","mw_g_mol","dispersity",
            "source_file","source_row","source_sha256"]
CONV_FIELDS=[*FIELDS,"design_phase"]


def write_csv(path,fields,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as out:
        w=csv.DictWriter(out,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sec_rows(root, sources, verified, fold_by_group, monomer):
    rows, rejected=[],[]
    missing=[]
    for src in sources:
        rel=src["path"]
        if rel not in verified or verified[rel]["sha256"]!=src["sha256"]:
            raise ValueError("Unverified SEC columns or source hash")
        cols=verified[rel]["columns"]
        if cols is None:
            missing.append(src["experiment_id"])
            continue
        if cols["mass_unit"]!="g/mol" or cols["time_unit"]!="min":
            raise ValueError("Unverified SEC units")
        path=source_path(root,rel)
        if digest(path)!=src["sha256"]:
            raise ValueError("SEC source checksum mismatch")
        group=canonical_condition(monomer,float(src["temperature_c"]),float(src["ct_cm"]))
        if group not in fold_by_group:
            raise ValueError("SEC condition missing from conversion grouping; review explicitly")
        by_id=defaultdict(list)
        with path.open(encoding=src["encoding"],newline="") as f:
            reader=csv.DictReader(f,delimiter=src["delimiter"],strict=True)
            needed={cols[k] for k in ("id","mn","mw","dispersity","residence")}
            if not needed<=set(reader.fieldnames or []):
                raise ValueError("Mapped SEC header absent")
            for raw in reader:
                line=reader.line_num
                names=[cols[k] for k in ("id","mn","mw","dispersity","residence")]
                vals=[raw.get(k) for k in names]
                if not any(v is not None and str(v).strip() for v in vals):
                    continue
                try:
                    if None in raw or any(v is None for v in vals):
                        raise ValueError("malformed SEC record")
                    # An ID and independent time must be known, even if a target is missing.
                    id_num=finite(vals[0])
                    if not id_num.is_integer() or id_num<0:
                        raise ValueError("invalid injection ID")
                    gid=str(int(id_num))
                    t=finite(vals[4])
                    if t<=0:
                        raise ValueError("nonpositive SEC residence time")
                    # Do not silently fill missing labels; retain each valid observed target.
                    mn,mw,d=[finite(v) if str(v).strip() else None for v in vals[1:4]]
                    if any(v is not None and v<=0 for v in (mn,mw)) or (d is not None and d<1):
                        raise ValueError("nonphysical mass or dispersity")
                    if mn is not None and mw is not None and mw<mn:
                        raise ValueError("Mw below Mn")
                    if d is not None and mn is not None and mw is not None and abs(d-mw/mn)>max(.05,.05*mw/mn):
                        raise ValueError("dispersity inconsistent with Mw/Mn")
                    if mn is None and mw is None and d is None:
                        raise ValueError("no observed SEC target")
                    by_id[gid].append({"experiment_id":src["experiment_id"],"split_group":group,
                        "fold":fold_by_group[group],"monomer":monomer,
                        "design_phase":src.get("design_phase","original"),
                        "temperature_c":float(src["temperature_c"]),"ct_cm":float(src["ct_cm"]),
                        "residence_min":t,"gpc_id":gid,"mn_g_mol":mn,"mw_g_mol":mw,
                        "dispersity":d,"source_file":rel,"source_row":line,"source_sha256":src["sha256"]})
                except (TypeError,ValueError) as e:
                    rejected.append({"experiment_id":src["experiment_id"],"source_file":rel,
                                     "source_row":line,"reason":str(e)})
        for gid,items in by_id.items():
            first=items[0]
            key=lambda r:tuple(r[k] for k in ("residence_min","mn_g_mol","mw_g_mol","dispersity"))
            if any(key(x)!=key(first) for x in items[1:]):
                rejected.extend({"experiment_id":x["experiment_id"],"source_file":rel,
                    "source_row":x["source_row"],"reason":"conflicting repeated injection ID"} for x in items)
            else:
                rows.append(first)
                rejected.extend({"experiment_id":x["experiment_id"],"source_file":rel,
                    "source_row":x["source_row"],"reason":"duplicate injection ID"} for x in items[1:])
    return rows,rejected,missing


def prepare_all(root,mapping_dir,out):
    root, mapping_dir, out=Path(root),Path(mapping_dir),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    release=json.loads((mapping_dir/"release-kinetics-v1.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["doi"]!=DOI or release["doi"]!=DOI:
        raise ValueError("Unexpected release DOI")
    actual={(x["id"],x["name"],x["size"],x["md5"]) for x in manifest["files"]}
    expected={(x["id"],x["name"],x["size"],x["md5"]) for x in release["files"]}
    if actual!=expected:
        raise ValueError("Kinetic release manifest differs from pinned selection")
    sec_mapping=json.loads((mapping_dir/"sec-columns-v1.json").read_text())
    if sec_mapping["doi"]!=DOI:
        raise ValueError("Unexpected SEC mapping DOI")
    verified={s["path"]:s for s in sec_mapping["sources"]}
    all_conv,all_sec,all_rejected=[],[],[]
    summary={"doi":DOI,"quality_status":"provisional","monomers":{},
             "note":"NMR and SEC are separate observations with independent residence times. "
             "No imputation or nearest-time merge. SEC physical checks are not expert chromatogram QC."}
    for monomer in MONOMERS:
        mapping_path=mapping_dir/f"mapping.{monomer.lower()}-v1.json"
        mapping=json.loads(mapping_path.read_text())
        conv,rejected,report=prepare(root,mapping,folds=5,seed=42)
        write_outputs(out/monomer.lower(),conv,rejected,report,mapping_path)
        phases={s["experiment_id"]:s.get("design_phase","original") for s in mapping["sources"]}
        conv=[{**r,"design_phase":phases[r["experiment_id"]]} for r in conv]
        groups={r["split_group"]:r["fold"] for r in conv}
        sec,sec_rej,missing=sec_rows(root,mapping["sources"],verified,groups,monomer)
        all_conv.extend(conv)
        all_sec.extend(sec)
        all_rejected.extend(sec_rej)
        summary["monomers"][monomer]={
            "conversion_rows":len(conv),"experiments":report["independent_experiments"],
            "condition_groups":report["condition_groups"],"conversion_rejected":report["counts"].get("rejected",0),
            "conversion_ranges":report["ranges"],
            "missing_sources":report["missing_sources"],
            "flagged_experiments":[e for e,v in report["timesweep_diagnostics"].items() if any(f["flag"]=="possible_jump" for f in v["flags"])],
            "sec_rows":len(sec),"sec_rejected":len(sec_rej),
            "sec_groups":len({r["split_group"] for r in sec}),
            "sec_no_columns":missing,
            "sec_targets":{t:sum(r[t] is not None for r in sec) for t in ("mn_g_mol","mw_g_mol","dispersity")},
            "phases":dict(Counter(phases.values()))}
    if not all_sec:
        raise ValueError("No valid SEC data; inspect source before proceeding")
    write_csv(out/"conversion.csv",CONV_FIELDS,all_conv)
    write_csv(out/"sec.csv",SEC_FIELDS,all_sec)
    write_csv(out/"sec_rejected.csv",["experiment_id","source_file","source_row","reason"],all_rejected)
    summary["conversion_rows"]=len(all_conv)
    summary["sec_rows"]=len(all_sec)
    summary["condition_groups"]=len({r["split_group"] for r in all_conv})
    summary["hashes"]={name:digest(out/name) for name in ("conversion.csv","sec.csv","sec_rejected.csv")}
    (out/"quality.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("MULTIMONOMER_QC "+json.dumps(summary,ensure_ascii=True,allow_nan=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/raw/monash-kinetics-v1"))
    parser.add_argument("--mapping-dir",type=Path,default=Path("data/monash"))
    parser.add_argument("--out",type=Path,default=Path("data/processed/monash-kinetics-v1"))
    args=parser.parse_args()
    prepare_all(args.root,args.mapping_dir,args.out)


if __name__=="__main__":
    main()
