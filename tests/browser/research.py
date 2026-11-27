"""Authenticated browser fixture; no physical chemistry or customer data."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

BASE=os.environ.get("POLYSEARCH_EXCHANGE_URL","http://127.0.0.1:8001")
OUT=Path(os.environ.get("POLYSEARCH_BROWSER_OUT","artifacts/browser"))
OUT.mkdir(parents=True,exist_ok=True)
PASSWORD="browser-fixture-only-password"

with sync_playwright() as p:
    browser=p.chromium.launch()
    ctx=browser.new_context(viewport={"width":1440,"height":1000},accept_downloads=True)
    page=ctx.new_page()
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto(BASE+"/research")
    expect(page).to_have_url(BASE+"/login")
    page.get_by_label("Имя пользователя").fill("researcher")
    page.get_by_label("Пароль").fill(PASSWORD)
    page.get_by_role("button",name="Войти",exact=True).click()
    expect(page.locator(".session-panel")).to_contain_text("researcher")
    page.goto(BASE+"/research")
    expect(page.get_by_role("button",name="Загрузить тестовый пример")).to_be_enabled()
    page.get_by_role("button",name="Загрузить тестовый пример").click()
    csv=page.locator("textarea[name=csv]").input_value()
    page.locator("#research-file").set_input_files({"name":"fixture.csv","mimeType":"text/csv","buffer":csv.encode()})
    expect(page.locator("#research-message")).to_contain_text("CSV загружен")
    page.get_by_role("button",name="Обучить и проверить модели").click()
    expect(page.locator("#research-message")).to_contain_text("Исследовательский расчёт сохранён",timeout=60000)
    expect(page.locator("#research-kind")).to_contain_text("ПРОГРАММНАЯ ФИКСТУРА")
    expect(page.locator("#selection-table tbody tr")).to_have_count(3)
    expect(page.locator("#research-scatter svg")).to_be_visible()
    expect(page.locator("#candidate-table")).to_contain_text("Вне диапазона")
    with page.expect_download() as download:
        page.get_by_role("button",name="Экспорт расчёта JSON").click()
    data=json.loads(download.value.path().read_text())
    assert data["actor"]=="researcher" and data["input"]["dataset_kind"]=="synthetic_fixture"
    assert data["result"]["counts"]["train"]["groups"]==8
    page.reload()
    expect(page.locator("#research-title")).to_contain_text("Тестовый пример")
    page.screenshot(path=str(OUT/"research.png"),full_page=True)
    viewer=browser.new_context()
    vp=viewer.new_page()
    vp.goto(BASE+"/research")
    vp.get_by_label("Имя пользователя").fill("viewer")
    vp.get_by_label("Пароль").fill(PASSWORD)
    vp.get_by_role("button",name="Войти",exact=True).click()
    expect(vp.locator(".session-panel")).to_contain_text("viewer")
    vp.goto(BASE+"/research")
    expect(vp.locator("#research-title")).to_contain_text("Тестовый пример")
    expect(vp.locator("#research-create")).to_be_hidden()
    assert vp.request.post(BASE+"/api/research/runs",data=data["input"],headers={"X-PolySearch-Request":"1"}).status==403
    assert not errors,errors
    print("RESEARCH_BROWSER_OK")
    browser.close()
