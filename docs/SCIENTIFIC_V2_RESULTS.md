# Научное ядро v2 — фактические результаты

Коммит [4681702](https://github.com/Timur-marii8st/PolySearch/commit/4681702a56bac1c6625361f254e3363bd3d29fcb).
[Гибридный ML и архивный BO](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250676) · [BoTorch](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250708).

Источник Monash MMA v1 (CC BY 4.0, DOI 10.26180/29208134.v1, атрибуция в data/monash/README.md). SHA256 таблицы: `f422958f34ae742ece88017c16bf3ddb9e95e23c150195bddeaa50bceb8e8809`. 785 строк, 15 групп, качество provisional. Это повторное исследование известного архива, а не новый независимый тест. Научные кандидаты не заменяют generic GP в рабочем API и пилоте.

## Сравнение семи суррогатов

Пять внешних групповых фолдов и внутренние три фолда; для конформной калибровки подбор повторяется без calibration. Ошибки в процентных пунктах конверсии. Sensitivity исключает MMA-13 и MMA-16 (698 строк, 13 групп).

| Модель | Group macro MAE, п.п. | RMSE, п.п. | R² | Sensitivity macro MAE, п.п. | Убывающие шаги сетки | Ненулевой t=0 |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 3.2137 | 3.712 | 0.5608 | 2.9909 | 0/960 | 14/15 |
| Kinetic Residual GP | 3.2139 | 3.904 | 0.5144 | 3.2382 | 18/960 | 0/15 |
| Rate Ridge | 3.2463 | 3.844 | 0.5290 | 3.0254 | 0/960 | 0/15 |
| Rate GP | 3.2507 | 4.043 | 0.4790 | 3.0603 | 0/960 | 0/15 |
| Kinetic | 3.2563 | 3.905 | 0.5139 | 3.0278 | 0/960 | 0/15 |
| CatBoost | 3.7884 | 4.506 | 0.3529 | 3.1486 | 91/960 | 15/15 |
| GP | 4.8642 | 5.974 | -0.1375 | 4.8439 | 284/960 | 15/15 |

Rate GP и Rate Ridge соблюдают выбранную форму кривой; обычный GP имеет 284 убывающих шага из 960. Это математические свойства принятого приближения, не доказательство корректной кинетики. Ни один новый кандидат не превзошёл Ridge по обеим постановкам. Разница Ridge и Kinetic Residual GP на полном наборе менее 0,001 п.п.; после исключения сомнительных реакций Ridge лучше.

### Интервалы

| Модель | Интервал | Покрытие строк | Покрытие групп | Средняя ширина, п.п. |
|---|---|---:|---:|---:|
| Rate GP | nominal90 | 45.61% | 6.67% | 7.75 |
| Rate GP | group_conformal80 | 99.87% | 93.33% | 77.14 |
| Kinetic Residual GP | nominal90 | 26.62% | 0.00% | 2.96 |
| Kinetic Residual GP | group_conformal80 | 99.87% | 93.33% | 28.91 |

Номинальные интервалы сильно недопокрывают; group conformal покрывает 14/15 групп, но ценой большой ширины (особенно Rate GP). Калибровка только на четырёх группах: конечный 90%-ный квантиль недоступен. Интервалы пока не годятся как подтверждённая уверенность в достижении цели. Обменность групп не установлена.

## Архивный BO: семь стратегий, 64 paired seed

Одинаковые 3 стартовые реакции, бюджет 10, конверсия в окне 6 ± 0,25 мин. В rate-модели медиана приближённо относится к 6 мин. Результаты используют только раскрытые endpoints.

| Стратегия | Полный пул: best @5 | Достижение 15% @5 | Sensitivity: best @10 | В пределах 1 п.п. от максимума @10 |
|---|---:|---:|---:|---:|
| Random | 18.17% | 64.06% | 13.05% | 78.13% |
| Maximin | 19.97% | 78.13% | 12.64% | 46.88% |
| Ridge Greedy | 21.61% | 100.00% | 13.22% | 92.19% |
| GP-EI | 20.80% | 92.19% | 13.07% | 79.69% |
| Ridge + GP-EI | 21.22% | 95.31% | 13.15% | 85.94% |
| Rate Ridge Greedy | 21.17% | 95.31% | 13.24% | 93.75% |
| Rate GP-EI | 21.56% | 98.44% | 13.22% | 92.19% |

Rate GP-EI практически сравнялся с Ridge Greedy на полном пуле; на очищенном варианте Rate Ridge достигает близкой к максимуму конверсии в 60/64 траекториях против 59/64 у Ridge. Разница мала. Порог 15% недостижим после исключения двух реакций (oracle 13,32%). Условная равная стоимость запроса не является фактической стоимостью лаборатории.

## BoTorch и GPyTorch

Отдельное окружение CPU torch 2.10.0+cpu, BoTorch 0.18.1, GPyTorch 1.15.2. Реальная часть — восемь paired seed, бюджет 8 реакций. Не сравнивать напрямую с прогоном 64 seed.

| Стратегия | Полный пул: best @5 | Полный пул: best @8 | Sensitivity: best @8 | Sensitivity: near-optimum @8 |
|---|---:|---:|---:|---:|
| Random | 19.23% | 19.23% | 13.00% | 75.00% |
| Ridge Greedy | 21.64% | 22.34% | 13.16% | 87.50% |
| GP-EI | 19.89% | 22.34% | 12.68% | 50.00% |
| BoTorch qLogNEI | 21.40% | 22.34% | 13.01% | 75.00% |

qLogNEI достиг наблюдаемого максимума во всех 8 траекториях полного архива, но не превосходит Ridge в основном или sensitivity сценариях по средней лучшей конверсии при фиксированном бюджете.

### Синтетическая многокритериальная задача

25 вариантов, две конфликтующие безразмерные цели, известные условные стоимости, бюджет 16 условных единиц; четыре paired seed. Oracle hypervolume всего пула 0.5233.

| Стратегия | Mean hypervolume | Mean queries | Mean spent |
|---|---:|---:|---:|
| Random | 0.3701 | 6.50 | 16.00 |
| qLogNEHVI | 0.3887 | 6.50 | 16.00 |
| qLogNEHVI/cost | 0.3821 | 6.75 | 16.00 |

В этом небольшом тесте qLogNEHVI лучше случайного выбора по средней hypervolume; cost-aware вариант не улучшил qLogNEHVI. Деление приобретения на стоимость — эвристика, не глобальная оптимизация бюджета. Эти результаты не химические измерения, не рубли и не экономия на производстве.

## Решение
- Оставить Ridge базовым точечным кандидатом; Rate Ridge — структурно ограниченный кандидат, Rate GP и Kinetic Residual GP — экспериментальные.
- Не включать BoTorch в сервер автоматически. Optional research dependencies не охвачены базовым серверным аудитом безопасности. Требуются отдельная проверка зависимостей, новые независимые группы и сравнительная валидация.
- Следующая научная задача: отдельное расширение датасетов BA/VA и нового источника, сопоставление условий/единиц, QC и проверка переносимости; собрать измерения дополнительных свойств и фактическую стоимость опытов у заказчика. Новые датасеты в этом этапе не интегрированы.
- Не обещать эффективность до проспективного пилота.

## Воспроизводимость

[Протокол](SCIENTIFIC_V2.md). Сводки [ML](../reports/monash-hybrid-v2.json), [BO](../reports/monash-hybrid-bo-v2.json), [BoTorch](../reports/botorch-research-v1.json). [Полные ML/BO-артефакты](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250676/artifacts/15201860627) · [BoTorch-артефакт](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250708/artifacts/15201121479) (срок хранения GitHub Actions).

Python CI: 90 passed, 4 skipped (3 PostgreSQL-only, 1 BoTorch-only) на Python 3.11 и 3.12. Отдельный research workflow: 5 passed, включая qLogNEI/qLogNEHVI. Состояние CI коммита 4681702:
- [Monash archive BO](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250700): success
- [Python tests](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250678): success
- [Server integration](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250895): success
- [Beta release checks](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250742): success
- [Browser smoke](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250876): success
- [Monash hybrid scientific benchmark](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250676): success
- [Monash ML benchmark](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250693): success
- [BoTorch research validation](https://github.com/Timur-marii8st/PolySearch/actions/runs/46005250708): success
