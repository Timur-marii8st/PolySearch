# API

Контракты новой реализации v0.1.0; подробности запуска — [IMPLEMENTATION](IMPLEMENTATION.md).

| Метод | Маршрут | Назначение |
|---|---|---|
| GET | /health | Проверка доступности сервиса |
| POST | /api/config | Инициализация реального проекта |
| POST | /api/import.csv | Атомарный импорт CSV (JSON с полем csv) |
| GET | /api/state | Состояние проекта |
| POST | /api/recommend | Рекомендации на текущей истории |
| POST | /api/demo | Создание синтетического проекта |
| POST | /api/results | Запись результата |
| POST | /api/simulate | Синтетический результат, только demo |
| GET | /api/export.json | Экспорт снимка |

```json
{"batch_size": 1}
```
Пример тела POST /api/recommend. Результат включает ID и параметры кандидата, прогнозы и ссылки на train-историю.

```json
{"seed": 42, "n_initial": 12}
```
POST /api/demo создаёт синтетическую историю. Если проект уже существует, требуется явное поле reset: true; иначе 409.

```json
{
  "parameters": {"factor_a": 0.5, "factor_b": 0.5, "factor_c": 0.5},
  "outcomes": {"mfr": 15.0, "modulus": 1268.8, "impact": 10.7056349186},
  "status": "completed",
  "role": "train"
}
```
Иллюстративная структура POST /api/results, не реальный режим синтеза. При передаче recommendation_id параметры должны соответствовать актуальной рекомендации. Ошибки валидации — 422; устаревший ID или симуляция в real — 409.

Локальный запуск: Python 3.11/3.12, Uvicorn с фабрикой polysearch.api:create_app, 127.0.0.1:8000, Swagger /docs. POLYSEARCH_DB задаёт путь к SQLite. Один процесс, локальный доступ. Новая запись делает прежние рекомендации недействительными.
