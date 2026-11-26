"""Run against an isolated local server with Playwright (not part of pytest discovery)."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("POLYSEARCH_BASE_URL", "http://127.0.0.1:8000")
OUT = Path(os.environ.get("POLYSEARCH_BROWSER_OUT", "artifacts/browser"))
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(BASE)
    expect(page.get_by_text("Создайте первый проект")).to_be_visible()
    page.get_by_role("button", name="Запустить демо").click()
    expect(page.locator("#mode-label")).to_contain_text("Синтетическое демо")
    expect(page.locator("#stats .stat").first).to_contain_text("12")
    expect(page.locator("#chart svg")).to_be_visible()
    page.screenshot(path=str(OUT / "overview.png"), full_page=True)

    page.get_by_role("link", name="Следующий опыт").click()
    page.get_by_role("button", name="Рассчитать рекомендации").click()
    expect(page.locator(".rec-card").first).to_be_visible(timeout=60000)
    expect(page.locator(".rec-card").first).to_contain_text("Некалиброванный GP-интервал")
    page.screenshot(path=str(OUT / "recommend.png"), full_page=True)
    page.get_by_role("button", name="Симулировать опыт").first.click()
    expect(page.locator("#message")).to_contain_text("Синтетическое измерение добавлено")
    assert len(page.request.get(BASE + "/api/state").json()["trials"]) == 13
    expect(page.locator(".rec-card")).to_have_count(0)

    page.get_by_role("link", name="Настройки проекта").click()
    # Rebuild into a one-factor real project using the current form.
    for _ in range(2):
        page.locator("#factor-rows .editor-row").last.get_by_role("button", name="Удалить фактор").click()
        page.locator("#target-rows .editor-row").last.get_by_role("button", name="Удалить свойство").click()
    factor = page.locator("#factor-rows .editor-row").first
    factor.get_by_label("Имя").fill("x")
    factor.get_by_label("Нижняя").fill("0")
    factor.get_by_label("Верхняя").fill("1")
    target = page.locator("#target-rows .editor-row").first
    target.get_by_label("Имя").fill("y")
    target.get_by_label("Нижняя").fill("0.8")
    target.get_by_label("Верхняя").fill("")
    target.get_by_label("Масштаб").fill("0.2")
    page.get_by_role("button", name="Создать реальный проект").click()
    expect(page.locator("#mode-label")).to_contain_text("Реальный проект")
    expect(page.locator("#stats .stat").first).to_contain_text("0")

    page.get_by_role("link", name="Журнал опытов").click()
    page.locator("#csv-file").set_input_files({"name":"test.csv","mimeType":"text/csv",
        "buffer":b"x,y\n0.1,0.2\n0.5,0.6\n0.8,0.7\n"})
    page.get_by_role("button", name="Импортировать CSV").click()
    expect(page.locator("#message")).to_contain_text("Импортировано записей: 3")
    expect(page.locator("#journal-table tbody tr")).to_have_count(3)
    page.get_by_role("link", name="Следующий опыт").click()
    page.get_by_role("button", name="Рассчитать рекомендации").click()
    expect(page.locator(".rec-card").first).to_be_visible(timeout=60000)
    expect(page.get_by_role("button", name="Симулировать опыт")).to_have_count(0)
    page.get_by_role("button", name="Внести результат").first.click()
    expect(page.locator("#trial-binding")).to_contain_text("Параметры зафиксированы")
    expect(page.locator('#trial-fields [data-kind="factor"]')).to_have_attribute("readonly", "")
    page.get_by_label("Измерение: y", exact=True).fill("0.85")
    page.get_by_role("button", name="Сохранить измерение").click()
    expect(page.locator("#message")).to_contain_text("Измерение сохранено")
    expect(page.locator("#journal-table tbody tr")).to_have_count(4)
    page.reload()
    expect(page.locator("#journal-table tbody tr")).to_have_count(4)
    page.screenshot(path=str(OUT / "journal.png"), full_page=True)

    page.get_by_role("link", name="Проверка качества").click()
    expect(page.locator("#research-table tbody tr")).to_have_count(4)
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("link", name="Обзор", exact=True).click()
    expect(page.locator("#chart svg")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=str(OUT / "mobile.png"), full_page=True)
    assert not errors, json.dumps(errors, ensure_ascii=False)
    print("Browser smoke passed: demo, recommendation, simulation, real config, CSV, bound result, persistence, quality and mobile.")
    browser.close()
