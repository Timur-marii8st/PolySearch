"""Download the pinned Monash Figshare release and inspect raw file schemas.

No chemistry interpretation is performed during discovery.
"""
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

ARTICLE_ID = 29208134
VERSION = 1
DOI = "10.26180/29208134.v1"
METADATA_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{VERSION}"
MAX_BYTES = 512 * 1024 * 1024
CHUNK = 1024 * 1024


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked_url(url):
    """Permit HTTPS Figshare endpoints and observed public object-storage buckets."""
    p = urlsplit(url)
    host = p.hostname
    direct = host in {"api.figshare.com", "figshare.com", "ndownloader.figshare.com",
                      "monash.figshare.com"}
    # AWS exposes the same Figshare bucket via global and regional S3 hosts.
    s3 = r"s3(?:(?:[.-]dualstack)?[.-][a-z0-9-]+)?\.amazonaws\.com"
    bucket_path = bool(re.fullmatch(s3, host or "") and p.path.startswith("/pfigshare-u-files/"))
    bucket_host = bool(re.fullmatch(r"pfigshare-u-files\." + s3, host or ""))
    # Monash institutional repository redirects to its own object storage.
    monash_bucket = host == "au-east.erc.monash.edu.au" and p.path.startswith("/fpfiles/")
    if (p.scheme != "https" or p.username is not None or p.password is not None
            or p.port not in (None, 443) or not (direct or bucket_path or bucket_host or monash_bucket)):
        raise ValueError(f"Download URL is outside the approved Figshare HTTPS endpoints (host={host!r}, scheme={p.scheme!r}, bucket={p.path.split('/')[1:2]!r})")
    return url


class _FigshareRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _checked_url(newurl))


def _https_open(url, timeout, headers=None):
    request = Request(_checked_url(url), headers=headers or {})
    # The custom handler checks every redirect before following it.
    return build_opener(_FigshareRedirect()).open(request, timeout=timeout)


def read_metadata():
    with _https_open(METADATA_URL, timeout=60, headers={"Accept": "application/json"}) as r:
        payload = json.loads(r.read(8 * CHUNK))
    if int(payload["id"]) != ARTICLE_ID or int(payload["version"]) != VERSION:
        raise ValueError("Unexpected Figshare release")
    doi = str(payload.get("doi", "")).lower().removeprefix("https://doi.org/")
    if doi != DOI:
        raise ValueError(f"Unexpected DOI: {doi}")
    if not payload.get("files"):
        raise ValueError("Release has no downloadable files")
    return payload


def _safe_name(name):
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe file name: {name!r}")
    return name


def _safe_member(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts or ":" in p.parts[0]:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return Path(*p.parts)


def _download(file, destination):
    expected = int(file["size"])
    md5 = file.get("computed_md5") or file.get("supplied_md5")
    if not 0 < expected <= MAX_BYTES:
        raise ValueError("File exceeds download budget")
    if destination.exists() and destination.stat().st_size == expected and md5 and digest(destination, "md5") == md5.lower():
        return
    url = str(file.get("download_url") or file.get("downloadUrl")
              or f"https://api.figshare.com/v2/file/download/{int(file['id'])}")
    _checked_url(url)
    temp = destination.with_suffix(destination.suffix + ".part")
    try:
        size = 0
        with _https_open(url, timeout=90) as source, temp.open("wb") as out:
            for chunk in iter(lambda: source.read(CHUNK), b""):
                size += len(chunk)
                if size > expected:
                    raise ValueError("Download exceeded declared size")
                out.write(chunk)
        if size != expected or (md5 and digest(temp, "md5") != md5.lower()):
            raise ValueError("Release size or checksum mismatch")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def _extract(path, target, remaining=MAX_BYTES):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > 20000 or sum(i.file_size for i in infos) > remaining:
            raise ValueError("Archive exceeds extraction budget")
        used = 0
        seen = set()
        for info in infos:
            relative = _safe_member(info.filename)
            if relative in seen:
                raise ValueError("Duplicate ZIP member")
            seen.add(relative)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, output.open("wb") as out:
                shutil.copyfileobj(source, out, CHUNK)
            used += info.file_size
        return used


def acquire(root, subset='mma'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    release = read_metadata()
    files = release["files"]
    if subset == "mma":
        files = [f for f in files if re.fullmatch(r"\d+mma\d.*\.csv", f["name"], re.I)
                 or f["name"].lower() == "summary_for_folders.xlsx"]
        if len(files) != 16 or len([f for f in files if f["name"].lower().endswith(".csv")]) != 15:
            raise ValueError("Unexpected pinned MMA selection")
    elif subset != "all":
        raise ValueError("Unknown release subset")
    if sum(int(f["size"]) for f in files) > MAX_BYTES:
        raise ValueError("Release exceeds configured total download budget")
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    extracted = root / "extracted"
    # Never clear arbitrary user paths; use a fresh inventory directory.
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir()
    manifest = {"doi": DOI, "article_id": ARTICLE_ID, "version": VERSION,
                "title": release.get("title"), "license": release.get("license"), "subset": subset,
                "authors": release.get("authors"), "files": []}
    def fetch_one(f):
        name = _safe_name(f["name"])
        file_id = int(f["id"])
        folder = raw / str(file_id)
        folder.mkdir(exist_ok=True)
        path = folder / name
        print("MONASH_DOWNLOAD " + json.dumps({"id": file_id, "name": name, "size": f["size"]}), flush=True)
        _download(f, path)
        return path

    with ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = list(executor.map(fetch_one, files))
    remaining = MAX_BYTES
    for f, path in zip(files, downloaded):
        name = _safe_name(f["name"])
        file_id = int(f["id"])
        manifest["files"].append({"id": file_id, "name": name, "size": path.stat().st_size,
                                  "sha256": digest(path), "md5": digest(path, "md5"),
                                  "path": path.relative_to(root).as_posix()})
        target = extracted / str(file_id)
        if zipfile.is_zipfile(path) and path.suffix.lower() not in {".xlsx", ".xlsm"}:
            remaining -= _extract(path, target, remaining)
            for nested in sorted(target.rglob("*.zip")):
                remaining -= _extract(nested, nested.parent / (nested.stem + "_unpacked"), remaining)
        else:
            target.mkdir(exist_ok=True)
            shutil.copy2(path, target / name)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _csv_preview(path):
    with path.open("rb") as stream:
        sample = stream.read(65536)
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return {"error": "Unsupported encoding"}
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))[:5]
    return {"encoding": encoding, "delimiter": delimiter,
            "preview": [[str(v)[:200] for v in row[:30]] for row in rows]}


def inventory(root):
    root = Path(root)
    results = []
    for path in sorted((root / "extracted").rglob("*")):
        if not path.is_file():
            continue
        item = {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                "sha256": digest(path)}
        if path.suffix.lower() == ".csv":
            item.update(_csv_preview(path))
        elif path.suffix.lower() in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook
            book = load_workbook(path, read_only=True, data_only=True)
            try:
                item["sheets"] = [
                    {"name": sheet.title, "rows": sheet.max_row, "columns": sheet.max_column,
                     "preview": [[str(v)[:200] if v is not None else None for v in row]
                                 for row in sheet.iter_rows(min_row=1, max_row=min(sheet.max_row or 1, 100),
                                                            max_col=min(sheet.max_column or 1, 20), values_only=True)]}
                    for sheet in book.worksheets]
            finally:
                book.close()
        elif path.suffix.lower() == ".zip":
            item["nested_zip"] = True
        results.append(item)
    report = {"doi": DOI, "files": results}
    (root / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/monash-v1"))
    parser.add_argument("--subset", choices=["mma", "all"], default="mma")
    args = parser.parse_args()
    manifest = acquire(args.root, args.subset)
    report = inventory(args.root)
    print("MONASH_MANIFEST " + json.dumps(manifest, ensure_ascii=True), flush=True)
    print("MONASH_FILE_COUNT " + str(len(report["files"])))
    csvs = [f for f in report["files"] if f["path"].lower().endswith(".csv")]
    print("MONASH_CSV_COUNT " + str(len(csvs)))
    for item in report["files"]:
        if "sheets" in item:
            print("MONASH_WORKBOOK " + json.dumps(item, ensure_ascii=True))
    for item in csvs:
        if "mma" in Path(item["path"]).name.lower():
            print("MONASH_MMA_CSV " + json.dumps(item, ensure_ascii=True))
    if csvs:
        print("MONASH_CSV_SAMPLE " + json.dumps(csvs[0], ensure_ascii=True))
    for item in report["files"]:
        if item.get("nested_zip"):
            print("MONASH_NESTED_ZIP " + json.dumps(item, ensure_ascii=True))


if __name__ == "__main__":
    main()
