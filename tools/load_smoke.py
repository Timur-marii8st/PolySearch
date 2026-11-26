"""Bounded HTTP load check; ONLY use the fresh disposable CI Compose instance.

No external target option: loopback, dedicated fixture credentials, explicit opt-in,
and empty campaign/workspace checks before writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen
from http.cookies import SimpleCookie

BASE = "http://127.0.0.1:8000"
HEADER = {"Content-Type": "application/json", "X-PolySearch-Request": "1"}


def request(path, data=None, cookie=None):
    headers = dict(HEADER)
    if cookie:
        headers["Cookie"] = cookie
    body = json.dumps(data).encode() if data is not None else None
    start = time.perf_counter()
    with urlopen(Request(BASE + path, data=body, headers=headers), timeout=30) as r:
        response = json.load(r)
        return response, (time.perf_counter() - start) * 1000, r.headers


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered)-1, __import__("math").ceil(len(ordered)*p)-1))]


def stats(values):
    return {"requests": len(values), "p50_ms": round(statistics.median(values), 2),
            "p95_ms": round(percentile(values, .95), 2), "max_ms": round(max(values), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--disposable-fixture", action="store_true", required=True)
    args = p.parse_args()
    password = os.environ["POLYSEARCH_LOAD_FIXTURE_PASSWORD"]
    _, _, headers = request("/api/auth/login", {"username":"load-fixture-admin","password":password})
    jar = SimpleCookie()
    jar.load(headers["Set-Cookie"])
    cookie = "polysearch_session=" + jar["polysearch_session"].value
    def req(path, data=None):
        return request(path, data, cookie)
    state, _, _ = req("/api/state")
    campaigns, _, _ = req("/api/pilot/campaigns")
    if state["config"] is not None or state["trials"] or campaigns:
        raise RuntimeError("Refusing to load-test nonempty workspace")
    cfg = {"factors":[{"name":"x","lower":0,"upper":1}],
           "targets":[{"name":"y","lower":.8,"scale":.2}]}
    req("/api/config", {"config":cfg})
    start = time.perf_counter()
    def read(i):
        return req("/api/state")[1]
    with ThreadPoolExecutor(max_workers=8) as executor:
        reads = list(executor.map(read, range(80)))
    read_seconds = time.perf_counter()-start
    start = time.perf_counter()
    def write(i):
        # Distinct synthetic test rows; never physical measurements.
        result, elapsed, _ = req("/api/results", {"parameters":{"x":(i+1)/81},
                     "outcomes":{"y":(i+1)/81},"note":f"LOAD_FIXTURE_{i}"})
        return result["id"], elapsed
    with ThreadPoolExecutor(max_workers=8) as executor:
        writes = list(executor.map(write, range(40)))
    write_seconds = time.perf_counter()-start
    final, _, _ = req("/api/state")
    assert len(final["trials"]) == 40
    assert len({x["id"] for x in final["trials"]}) == 40
    assert {x["id"] for x in final["trials"]} == {i for i,_ in writes}
    _, model_ms, _ = req("/api/recommend", {"batch_size":1})
    result = {"fixture": True, "workers":8, "read":{**stats(reads),"requests_per_sec":round(80/read_seconds,2)},
              "write":{**stats([v for _,v in writes]),"requests_per_sec":round(40/write_seconds,2)},
              "gp_recommend_ms":round(model_ms,2), "persisted_rows":40, "lost_writes":0,
              "note":"One disposable CI Compose instance, 40 synthetic rows, no production SLA or capacity claim."}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("LOAD_SUMMARY "+json.dumps(result),flush=True)


if __name__=="__main__":
    main()
