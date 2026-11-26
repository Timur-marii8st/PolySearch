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

    # Pilot interface: isolated synthetic test fixture (never a real chemical experiment).
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.get_by_role("link", name="Пилотные кампании").click()
    expect(page.locator("#create-button")).to_be_enabled()
    form = page.locator("#create-form")
    for name, value in {
        "name":"Browser fixture", "actor":"Researcher", "objective":"Test target",
        "baseline_method":"Manual baseline fixture", "measurement_protocol":"Synthetic software fixture only",
        "safety_review":"No physical work", "cost_policy":"All attempts",
        "budget_per_arm":"3", "confirmations_required":"1"
    }.items():
        form.locator("[name='"+name+"']").fill(value)
    form.get_by_role("button", name="Создать черновик").click()
    expect(page.locator("#campaign-status")).to_have_text("Черновик")
    action = page.locator("#campaign-action")
    action.locator("[name=actor]").fill("Reviewer")
    action.locator("[name=note]").fill("Fixture protocol approved")
    action.get_by_role("button", name="Согласовать протокол").click()
    expect(page.locator("#campaign-status")).to_have_text("Активна")

    proposal = page.locator("#propose-form")
    proposal.locator("[name=actor]").fill("Researcher")
    proposal.locator("[name=rationale]").fill("Fixture baseline")
    proposal.locator("[name=arm]").select_option("baseline")
    proposal.locator("[name='factor:x']").fill("0.5")
    proposal.get_by_role("button", name="Сформировать план").click()
    plan = page.locator("#plan-detail")
    expect(plan.get_by_role("button", name="Согласовать", exact=True)).to_be_visible()
    plan.locator("form [name=actor]").fill("Reviewer")
    plan.locator("form [name=note]").fill("Fixture reviewed")
    plan.get_by_role("button", name="Согласовать", exact=True).click()
    expect(plan.get_by_role("button", name="Записать измерение")).to_be_visible()
    measurement = plan.locator("form").first
    for name, value in {"actor":"Researcher","run_id":"fixture-1","evidence":"synthetic fixture",
                        "outcome:y":"0.9","cost_minor":"15000","elapsed_minutes":"30"}.items():
        measurement.locator("[name='"+name+"']").fill(value)
    measurement.get_by_role("button", name="Записать измерение").click()
    expect(page.locator("#plans tbody tr")).to_have_count(1)
    expect(page.locator("#plans tbody tr")).to_contain_text("Завершён")
    expect(page.locator("#arm-stats")).to_contain_text("150 RUB")

    proposal.locator("[name=arm]").select_option("polysearch")
    proposal.locator("[name=rationale]").fill("Fixture next suggestion")
    proposal.get_by_role("button", name="Сформировать план").click()
    expect(plan.get_by_role("button", name="Проверить воспроизводимость")).to_be_visible()
    plan.get_by_role("button", name="Проверить воспроизводимость").click()
    expect(page.locator("#pilot-message")).to_contain_text("повтор: совпал")
    plan.locator("form [name=actor]").fill("Reviewer")
    plan.locator("form [name=note]").fill("Fixture rejection")
    plan.get_by_role("button", name="Отклонить").click()
    expect(page.locator("#plans tbody tr").first).to_contain_text("Отклонён")
    action.locator("[name=actor]").fill("Reviewer")
    action.locator("[name=note]").fill("Fixture closed")
    action.get_by_role("button", name="Закрыть кампанию").click()
    expect(page.locator("#campaign-status")).to_have_text("Закрыта")
    page.reload()
    expect(page.locator("#campaign-status")).to_have_text("Закрыта")
    expect(page.locator("#plans tbody tr")).to_have_count(2)
    page.screenshot(path=str(OUT / "pilot.png"), full_page=True)
    print("Pilot browser smoke passed: create, activate, baseline, approval, measurement, cost, ML proposal, replay, reject, close, persistence.")
    assert not errors, json.dumps(errors, ensure_ascii=False)
    print("Browser smoke passed: demo, recommendation, simulation, real config, CSV, bound result, persistence, quality and mobile.")
    browser.close()
