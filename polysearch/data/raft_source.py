"""Inspect the pinned published RAFT workbook, not digitized plots or fitted curves."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import zipfile

from .monash_source import digest

DOI = "10.5281/zenodo.15230215"
URL = "https://zenodo.org/records/15230215/files/Bizbalt/RAFT-knowledgebase-v1.0.0.zip?download=1"
MD5 = "e7dc1c91cee856b8cc286fae7cc78d5f"
MAX_ARCHIVE = 16 * 1024 * 1024
MAX_WORKBOOK = 12 * 1024 * 1024


def checked_url(url):
    p = urlsplit(url)
    if p.scheme != "https" or p.hostname != "zenodo.org" or p.port not in (443,None) or p.username or p.password:
        raise ValueError("Unexpected Zenodo HTTPS download endpoint")
    return url


class Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, checked_url(newurl))


def acquire(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    archive=root/"raft-v1.zip"
    if not archive.exists() or digest(archive,"md5")!=MD5:
        temp=archive.with_suffix(".part")
        try:
            with build_opener(Redirect()).open(Request(checked_url(URL)),timeout=90) as response,temp.open("wb") as out:
                size=0
                for chunk in iter(lambda:response.read(65536),b""):
                    size+=len(chunk)
                    if size>MAX_ARCHIVE:
                        raise ValueError("Archive exceeds size limit")
                    out.write(chunk)
            if digest(temp,"md5")!=MD5:
                raise ValueError("Pinned RAFT archive checksum mismatch")
            temp.replace(archive)
        finally:
            temp.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as z:
        books=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".xlsx")]
        # Keep all small workbook candidates; use fixed output filenames only.
        if not books or len(books)>10 or sum(i.file_size for i in books)>MAX_WORKBOOK:
            raise ValueError("Unexpected workbook inventory or size")
        result=[]
        for n,i in enumerate(books):
            path=root/f"workbook-{n}.xlsx"
            with z.open(i) as stream:
                raw=stream.read(MAX_WORKBOOK+1)
            if len(raw)!=i.file_size:
                raise ValueError("Workbook size mismatch")
            path.write_bytes(raw)
            result.append({"archive_member":i.filename,"path":path.name,"sha256":digest(path)})
    return {"doi":DOI,"license":"CC BY 4.0","archive_md5":MD5,"archive_sha256":digest(archive),"workbooks":result}


def inventory(root, manifest):
    from openpyxl import load_workbook
    result=[]
    for item in manifest["workbooks"]:
        book=load_workbook(Path(root)/item["path"],read_only=True,data_only=True)
        try:
            result.append({**item,"sheets":[{"name":s.title,"rows":s.max_row,"columns":s.max_column,
                "preview":[list(r) for r in s.iter_rows(min_row=1,max_row=min(s.max_row or 1,8),
                    max_col=min(s.max_column or 1,120),values_only=True)]} for s in book.worksheets]})
        finally:
            book.close()
    return {**manifest,"workbooks":result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/raw/raft-v1"))
    args=parser.parse_args()
    report=inventory(args.root,acquire(args.root))
    (args.root/"inventory.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    for book in report["workbooks"]:
        print("RAFT_WORKBOOK "+json.dumps(book,ensure_ascii=True,default=str),flush=True)


if __name__=="__main__":
    main()
