"""Offline synthetic fixtures exercise ETL, not chemical performance."""
import json
import zipfile

import pytest

from polysearch.data.monash_source import DOI, digest, _extract, _safe_name
from polysearch.data.monash_etl import prepare


def fixture(tmp_path):
    root = tmp_path / "release"
    path = root / "extracted"
    path.mkdir(parents=True)
    sources = []
    inventory = []
    for i, (temp, ratio) in enumerate([(70, 0.05), (80, 0.05), (90, 0.15), (90, 0.15)]):
        f = path / f"run{i}.csv"
        f.write_text("time_s,conv_pct\n60,15\n120,30\n180,120\n240,40\n", encoding="utf-8")
        relative = f.relative_to(root).as_posix()
        sha = digest(f)
        inventory.append({"path": relative, "sha256": sha})
        sources.append(dict(path=relative, sha256=sha, experiment_id=f"reaction{i}",
                            temperature_c=temp, ct_cm=ratio, condition_evidence="fixture only",
                            delimiter=",", header_row=1, residence_column="time_s",
                            residence_unit="s", conversion_column="conv_pct",
                            conversion_unit="percent", measurement_status="reviewed",
                            selection_evidence="fixture only", exclude_rows=[
                                {"start": 5, "end": 5, "reason": "synthetic excluded timesweep"}]))
    (root / "inventory.json").write_text(json.dumps({"doi": DOI, "files": inventory}))
    return root, {"doi": DOI, "monomer": "MMA", "review_note": "synthetic test only", "sources": sources}


def test_units_quality_and_condition_groups(tmp_path):
    root, mapping = fixture(tmp_path)
    rows, rejected, report = prepare(root, mapping, folds=3)
    assert report["counts"] == {"read": 16, "accepted": 8, "rejected": 8}
    assert report["independent_experiments"] == 4
    assert report["condition_groups"] == 3
    assert len(rejected) == 8
    assert rows[0]["residence_min"] == 1
    assert rows[0]["conversion_fraction"] == 0.15
    assert rows[0]["source_row"] == 2
    folds = {r["experiment_id"]: r["fold"] for r in rows}
    assert folds["reaction2"] == folds["reaction3"]
    assert rows == prepare(root, mapping, folds=3)[0]


def test_hash_mismatch(tmp_path):
    root, mapping = fixture(tmp_path)
    (root / mapping["sources"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        prepare(root, mapping, folds=3)


def test_unreviewed_source_rejected(tmp_path):
    root, mapping = fixture(tmp_path)
    mapping["sources"][0]["measurement_status"] = "unreviewed"
    with pytest.raises(ValueError, match="Review"):
        prepare(root, mapping, folds=3)


def test_duplicate_source_rows_rejected(tmp_path):
    root, mapping = fixture(tmp_path)
    mapping["sources"].append(mapping["sources"][0])
    with pytest.raises(ValueError, match="mapped more than once"):
        prepare(root, mapping, folds=3)


def test_group_count_not_spectrum_count(tmp_path):
    root, mapping = fixture(tmp_path)
    with pytest.raises(ValueError, match="condition groups"):
        prepare(root, mapping, folds=5)


def test_wrong_doi(tmp_path):
    root, mapping = fixture(tmp_path)
    mapping["doi"] = "wrong"
    with pytest.raises(ValueError, match="pinned"):
        prepare(root, mapping, folds=3)


def test_zip_path_traversal(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.csv", "test")
    with pytest.raises(ValueError, match="Unsafe"):
        _extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape.csv").exists()


def test_filename_traversal():
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_name("../outside.csv")


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://api.figshare.com/v2/file/download/1",
    "https://api.figshare.com.evil.example/1", "https://user@api.figshare.com/1",
    "https://api.figshare.com:444/1", "https://127.0.0.1/1",
    "https://s3-eu-west-1.amazonaws.com/unrelated-bucket/file",
])
def test_external_url_rejected(url):
    from polysearch.data.monash_source import _checked_url
    with pytest.raises(ValueError):
        _checked_url(url)


def test_figshare_https_urls_allowed():
    from polysearch.data.monash_source import _checked_url, _FigshareRedirect
    from urllib.request import Request
    initial = "https://ndownloader.figshare.com/files/55492559"
    target = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/55492559/file.csv?X-Amz-Expires=10"
    assert _checked_url(initial) == initial
    assert _checked_url(target) == target
    regional = "https://s3.ap-southeast-2.amazonaws.com/pfigshare-u-files/55492559/file.csv"
    virtual = "https://pfigshare-u-files.s3.dualstack.ap-southeast-2.amazonaws.com/55492559/file.csv"
    assert _checked_url(regional) == regional
    assert _checked_url(virtual) == virtual
    handler = _FigshareRedirect()
    redirected = handler.redirect_request(Request(initial), None, 302, "Found", {}, target)
    assert redirected.full_url == target
    with pytest.raises(ValueError):
        handler.redirect_request(Request(initial), None, 302, "Found", {}, "http://127.0.0.1/secret")
