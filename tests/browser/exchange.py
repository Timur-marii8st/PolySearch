"""Authenticated Chromium acceptance scenario against disposable fixture-only server."""
import json
from datetime import datetime, timezone
from pathlib import Path
import os

from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("POLYSEARCH_EXCHANGE_URL", "http://127.0.0.1:8001")
OUT = Path(os.environ.get("POLYSEARCH_BROWSER_OUT", "artifacts/browser"))
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"X-PolySearch-Request": "1"}
PASSWORD = "browser-fixture-only-password"


def login(browser, role):
    context = browser.new_context(viewport={"width":1440,"height":1000}, accept_downloads=True)
    page = context.new_page()
    page.goto(BASE + "/exchange" if role != "viewer" else BASE)
    expect(page).to_have_url(BASE+"/login")
    page.get_by_label("Имя пользователя").fill(role)
    page.get_by_label("Пароль").fill(PASSWORD)
    page.get_by_role("button",name="Войти",exact=True).click()
    expect(page.locator(".session-panel")).to_contain_text(role)
    return context,page


def post(page,path,data,status=200):
    response=page.request.post(BASE+path,data=data,headers=HEADERS)
    assert response.status==status,response.text()
    return response.json()


with sync_playwright() as p:
    browser=p.chromium.launch()
    admin,ap=login(browser,"admin")
    post(ap,"/api/config",{"config":{"factors":[{"name":"x","lower":0,"upper":1,"unit":"a.u."}],
         "targets":[{"name":"y","lower":.8,"scale":.2,"unit":"a.u."}]}})
    c=post(ap,"/api/pilot/campaigns",{"name":"LAB EXCHANGE FIXTURE","actor":"fixture","objective":"No physical run",
        "baseline_method":"Fixture only","measurement_protocol":"Software fixture","safety_review":"No physical work",
        "cost_policy":"Fixture cost","budget_per_arm":2,"confirmations_required":1},201)
    root="/api/pilot/campaigns/"+c["id"]
    def change(c,tail,**data):
        return post(ap,root+tail,{"expected_revision":c["revision"],"actor":"fixture",**data})
    c=change(c,"/activate",note="Fixture")
    c=change(c,"/plans",arm="baseline",parameters={"x":.5},rationale="Fixture")
    pid=c["plans"][-1]["id"]
    c=change(c,"/plans/"+pid+"/review",decision="approve",note="Fixture")
    researcher,rp=login(browser,"researcher")
    errors=[]
    rp.on("pageerror",lambda e:errors.append(str(e)))
    rp.goto(BASE+"/exchange")
    expect(rp.locator("#campaign-info")).to_contain_text("ревизия")
    with rp.expect_download() as downloaded:
        rp.get_by_role("button",name="Скачать задание JSON").click()
    order=json.loads(downloaded.value.path().read_text())
    assert order["plan_id"]==pid
    result={"schema_version":"polysearch.lab.v1","source_system":"mock-lab","external_id":"fixture-result",
            "campaign_id":c["id"],"plan_id":pid,"work_order_sha256":order["work_order_sha256"],
            "run_id":"fixture-run","observed_at":datetime.now(timezone.utc).isoformat(),
            "actual_parameters":order["parameters"],"outcomes":{"y":{"value":.9,"unit":"a.u."}},
            "status":"completed","evidence":"Software fixture only","currency":"RUB","cost_minor":15000,"elapsed_minutes":30}
    rp.locator("#result-file").set_input_files({"name":"fixture.json","mimeType":"application/json","buffer":json.dumps(result).encode()})
    expect(rp.locator("#preview")).to_contain_text("fixture-result")
    expect(rp.locator("#stage-form input[name=actor]")).to_have_value("researcher")
    rp.get_by_role("button",name="Отправить на проверку").click()
    expect(rp.locator("#message")).to_contain_text("На проверке")
    expect(rp.locator("#inbox tbody tr")).to_have_count(1)
    assert rp.request.get(BASE+root+"/report").json()["arms"]["baseline"]["attempts"]==0
    reviewer,qp=login(browser,"reviewer")
    qp.on("pageerror",lambda e:errors.append(str(e)))
    qp.on("dialog",lambda d:d.accept())
    qp.goto(BASE+"/exchange")
    qp.get_by_role("button",name="fixture-result",exact=True).click()
    expect(qp.locator("#detail input[name=actor]")).to_have_value("reviewer")
    qp.locator("#detail textarea[name=note]").fill("Verified software fixture")
    qp.get_by_role("button",name="Принять измерение").click()
    expect(qp.locator("#inbox tbody tr")).to_contain_text("Принят")
    saved=qp.request.get(BASE+root).json()["plans"][-1]["measurement"]
    assert saved["integration"]["review"]["actor"]=="reviewer"
    assert saved["integration"]["received_by"]=="researcher"
    assert qp.request.get(BASE+root+"/report").json()["arms"]["baseline"]["attempts"]==1
    qp.reload()
    expect(qp.locator("#inbox tbody tr")).to_contain_text("Принят")
    qp.screenshot(path=str(OUT/"exchange.png"),full_page=True)
    viewer,vp=login(browser,"viewer")
    assert vp.request.get(BASE+"/api/exchange/inbox").status==403
    vp.get_by_role("button",name="Выйти").click()
    expect(vp).to_have_url(BASE+"/login")
    assert vp.request.get(BASE+"/api/state").status==401
    assert not errors,errors
    print("Authenticated exchange browser smoke passed: login, export, stage, reviewer acceptance, persistence, viewer denial, logout.")
    browser.close()
