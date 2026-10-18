# Геометрия embedding guardrails

Сравнение `intfloat/e5-small-v2`, CE и CE+SupCon на одних и тех же 1 699 запросах WildGuardTest. Основной воспроизводимый анализ — [`guardrails_geometry_colab.ipynb`](guardrails_geometry_colab.ipynb). Модели берутся из Hugging Face по зафиксированным коммитам, веса моделей в Git не хранятся.

## Что здесь лежит

- [`report/e5_geometry_report.docx`](report/e5_geometry_report.docx) — отчёт: сводка геометрии трёх моделей, исходные графики, выводы и ограничения; дополнительные расчёты сохранены в [`report/data`](report/data).
- [`guardrails_geometry_colab.ipynb`](guardrails_geometry_colab.ipynb) — CE/CE+SupCon, PCA, спектр, косинусы, kNN, TwoNN и история обучения.
- [`guarails_results/analysis`](guarails_results/analysis) — архив агрегированных результатов прежнего разбора CE+SupCon (голова, слои, t-SNE).
- [`guardrails_geometry_results/run_20261018T134157_442709Z`](guardrails_geometry_results/run_20261018T134157_442709Z) — последний полный прогон: агрегированные CSV, PNG, интерактивный HTML, JSON-манифест и локальные рабочие файлы.
- [`rq4_followup/GEOMETRY_METRICS.md`](rq4_followup/GEOMETRY_METRICS.md) — выбранные метрики и точные места их расчёта.
- [`rq4_followup/prepare.py`](rq4_followup/prepare.py), [`rq4_followup/evaluate.py`](rq4_followup/evaluate.py) — анализ PC1 и подготовка/оценка  разметки смысловой близости.
- [`rq3_training`](rq3_training) и [`guardrails_training_history`](guardrails_training_history) — конфигурации, логи и агрегированная история пяти обучающих прогонов.

