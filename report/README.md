# Отчёт по геометрии E5

Итоговый документ: `e5_geometry_report.docx`, подготовленный для Google Docs.
Целевой Google-документ: `1DHd-aVHgI9b8x4faeHXhaylTkHiL8s0YBDEUD54xUbw`.
Создание локального файла само по себе не означает, что Google-документ обновлён.

## Источники

Основной прогон: `../guardrails_geometry_results/run_20261018T134157_442709Z`.
Архивные диагностики: `../guarails_results/analysis`.
История обучения: `../guardrails_training_history` и `../rq3_training`.

## Сборка

`analyze_report.py` использует сохранённые эмбеддинги, проверяет совпадение строк
и меток, воспроизводит 39 исходных значений и добавляет сопоставимые метрики.
Обучение и извлечение новых эмбеддингов не выполняются.

```sh
.venv/bin/python report/analyze_report.py
```

Зависимости анализа: NumPy, pandas, SciPy, scikit-learn.

`build_report.py` строит DOCX из агрегированных CSV и 16 уникальных исходных PNG.
Зависимости сборки: python-docx 1.2.0, Pillow 13.0.0; сборка выполнена во временном
окружении `/tmp/e5-report-runtime`.

```sh
/tmp/e5-report-runtime/bin/python report/build_report.py
```

После сборки применяется `google_docs_title_sanitize.py` из навыка documents.
Для проверки вёрстки документ экспортируется в PDF и постраничные PNG; эти
промежуточные файлы лежат в игнорируемой папке `qa/` и не являются отдельными
пользовательскими результатами.

`data/analysis_manifest.json` содержит SHA-256 исходных массивов и максимальное
расхождение при воспроизведении. `figure_manifest.json` сопоставляет номера
рисунков с оригинальными файлами; `design_tokens.json` фиксирует оформление.
