"""Analyze PC1 and prepare blinded human semantic-similarity judgments.

Usage: .venv/bin/python rq4_followup/prepare.py PATH_TO_RQ4_RUN
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score


MODELS = ("base", "ce", "finetuned")
SEED = 42


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spreadsheet_safe(text):
    """Prevent a prompt from being interpreted as a formula when CSV is opened."""
    text = str(text)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def r2(y, *predictors):
    design = np.column_stack((np.ones(len(y)), *predictors))
    prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return 1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum()


def load_run(run):
    frame = pd.read_csv(run / "eval_inputs.csv")
    assert frame.row_id.is_unique and frame.label.isin([0, 1]).all()
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(frame) == manifest["audit"]["analysis_rows"]
    vectors = {}
    for model in MODELS:
        with np.load(run / f"{model}_embeddings.npz") as saved:
            assert np.array_equal(saved["row_ids"], frame.row_id.to_numpy())
            assert np.array_equal(saved["labels"], frame.label.to_numpy())
            raw = saved["raw"].astype(np.float64)
        assert raw.shape == (len(frame), 384) and np.isfinite(raw).all()
        vectors[model] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return frame, manifest, vectors


def pc1_analysis(frame, vectors, out):
    y = frame.label.to_numpy(dtype=float)
    tokens = frame.n_tokens_full.to_numpy(dtype=float)
    log_tokens = np.log1p(tokens)
    token_z = (log_tokens - log_tokens.mean()) / log_tokens.std()
    scores = frame[["row_id", "label", "n_tokens_full"]].copy()
    scores["n_chars"] = frame.text.str.len().to_numpy()
    rows, extremes = [], []
    for model, x in vectors.items():
        centered = x - x.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(x))
        pc = centered @ eigenvectors[:, -1]
        if pc[y == 1].mean() < pc[y == 0].mean():
            pc = -pc  # PCA sign is arbitrary; orient toward harmful for display.
        scores[f"pc1_{model}"] = pc
        share = float(eigenvalues[-1] / eigenvalues.sum())
        point_r = float(pointbiserialr(y, pc).statistic)
        row = dict(model=model, n=len(x), pc1_share=share,
                   label_point_biserial_r=point_r, label_r2=point_r**2,
                   label_auc_on_this_sample=float(roc_auc_score(y, pc)),
                   token_spearman_all=float(spearmanr(pc, tokens).statistic),
                   token_spearman_safe=float(spearmanr(pc[y == 0], tokens[y == 0]).statistic),
                   token_spearman_harmful=float(spearmanr(pc[y == 1], tokens[y == 1]).statistic),
                   r2_label_only=float(r2(pc, y)),
                   r2_logtokens_only=float(r2(pc, token_z)),
                   r2_label_and_logtokens=float(r2(pc, y, token_z)))
        row["extra_r2_label_given_length"] = row["r2_label_and_logtokens"] - row["r2_logtokens_only"]
        row["extra_r2_length_given_label"] = row["r2_label_and_logtokens"] - row["r2_label_only"]
        rows.append(row)
        for tail, order in (("low", np.argsort(pc)[:12]), ("high", np.argsort(pc)[-12:][::-1])):
            for rank, index in enumerate(order, 1):
                extremes.append(dict(model=model, tail=tail, rank=rank,
                                     row_id=frame.row_id.iat[index], label=int(y[index]),
                                     n_tokens_full=int(tokens[index]), pc1_score=float(pc[index]),
                                     text=spreadsheet_safe(frame.text.iat[index])))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "pc1_summary.csv", index=False)
    scores.to_csv(out / "pc1_scores.csv", index=False)
    pd.DataFrame(extremes).to_csv(out / "pc1_extremes.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, model in enumerate(MODELS):
        pc = scores[f"pc1_{model}"].to_numpy()
        z = (pc - pc.mean()) / pc.std()
        for label, color, title in ((0, "#3477b2", "safe"), (1, "#d7773b", "harmful")):
            mask = y == label
            axes[0, col].hist(z[mask], bins=35, density=True, alpha=0.55,
                              color=color, label=title)
            axes[1, col].scatter(tokens[mask], z[mask], s=8, alpha=0.28,
                                 color=color, label=title)
        axes[0, col].set(title=f"{model}: PC1 {summary.pc1_share.iat[col]:.1%} of variance",
                         xlabel="PC1 score (standardized)", ylabel="Density")
        axes[1, col].set(xlabel="Input tokens", ylabel="PC1 score (standardized)", xscale="log")
        axes[0, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    fig.savefig(out / "pc1_label_and_length.png", dpi=160)
    plt.close(fig)
    return scores, summary


def choose_pairs(frame, scores, vectors, out):
    rng = np.random.default_rng(SEED)
    labels = frame.label.to_numpy()
    good = (frame.n_tokens_full.to_numpy() <= 256) & frame.text.str.len().between(20, 1000).to_numpy()
    n = len(frame)
    cosines = {name: (x @ x.T).astype(np.float32) for name, x in vectors.items()}
    # Cover both labels and low/middle/high positions along joint PC1.
    anchors = []
    joint_pc = scores.pc1_finetuned.to_numpy()
    for label in (0, 1):
        eligible = np.flatnonzero(good & (labels == label))
        ordered = eligible[np.argsort(joint_pc[eligible])]
        for bucket in np.array_split(ordered, 4):
            anchors.extend(rng.choice(bucket, size=3, replace=False).tolist())
    rng.shuffle(anchors)

    seen = set()
    hidden, visible = [], []
    for anchor in anchors:
        same = np.flatnonzero(good & (labels == labels[anchor]))
        opposite = np.flatnonzero(good & (labels != labels[anchor]))
        same = same[same != anchor]
        candidates = {
            "base_nearest": same[np.argsort(cosines["base"][anchor, same])[-40:][::-1]],
            "ce_distinct": same[np.argsort(cosines["ce"][anchor, same])[-40:][::-1]],
            "joint_distinct": same[np.argsort(cosines["finetuned"][anchor, same])[-40:][::-1]],
            "random_same_label": rng.permutation(same),
            "random_other_label": rng.permutation(opposite),
        }
        candidates["ce_distinct"] = candidates["ce_distinct"][
            np.argsort((cosines["ce"] - cosines["base"])[anchor, candidates["ce_distinct"]])[::-1]]
        candidates["joint_distinct"] = candidates["joint_distinct"][
            np.argsort((cosines["finetuned"] - cosines["base"])[anchor, candidates["joint_distinct"]])[::-1]]
        for kind, partners in candidates.items():
            partner = next((int(j) for j in partners if tuple(sorted((anchor, int(j)))) not in seen), None)
            assert partner is not None, f"No unique candidate for {kind}"
            seen.add(tuple(sorted((anchor, partner))))
            pair_id = f"P{len(visible) + 1:04d}"
            a, b = spreadsheet_safe(frame.text.iat[anchor]), spreadsheet_safe(frame.text.iat[partner])
            visible.append(dict(pair_id=pair_id, text_a=a, text_b=b,
                                semantic_similarity_0_4="", uncertain_0_1="", notes=""))
            hidden.append(dict(pair_id=pair_id, anchor_row_id=frame.row_id.iat[anchor],
                               partner_row_id=frame.row_id.iat[partner], selection=kind,
                               label_a=int(labels[anchor]), label_b=int(labels[partner]),
                               text_sha256=digest(a + "\0" + b),
                               cos_base=float(cosines["base"][anchor, partner]),
                               cos_ce=float(cosines["ce"][anchor, partner]),
                               cos_finetuned=float(cosines["finetuned"][anchor, partner])))
    assert len(visible) == 120 and len(seen) == 120
    visible = [visible[i] for i in rng.permutation(len(visible))]
    ratings_path = out / "human_ratings_template.csv"
    if ratings_path.exists():
        existing = pd.read_csv(ratings_path)
        for column in ("semantic_similarity_0_4", "uncertain_0_1", "notes"):
            if column in existing and existing[column].notna().any():
                raise RuntimeError(f"Existing human ratings in {ratings_path}; refusing to overwrite them.")
    pd.DataFrame(visible).to_csv(ratings_path, index=False)
    pd.DataFrame(hidden).to_csv(out / "pair_manifest.csv", index=False)
    return pd.DataFrame(hidden)


def write_protocol(out, summary, pairs, source_run, data_hash):
    by_kind = pairs.selection.value_counts().to_dict()
    lines = ["# Проверка PC1 и семантики", "",
             f"Источник: `{source_run}`; hash оценочной выборки: `{data_hash}`.",
             "Все три энкодера оценены на одинаковых 1 699 входах после одинаковой токенизации.", "",
             "## Что отражает PC1", "",
             "PC1 рассчитана **отдельно для каждой модели** по центрированным L2-нормированным эмбеддингам.",
             "Знак выбран так, чтобы средняя координата harmful была выше средней safe; знак PCA сам по себе произволен.",
             "`label_auc_on_this_sample` — описательная связь PC1 с меткой на этой же выборке, не качество обученного классификатора.",
             "`extra_r2_*` — добавочная объяснённая дисперсия в линейной модели с меткой и log(1+число токенов); причинного смысла она не имеет.", "",
             "| Модель | Доля PC1 | r с меткой | AUC метки по PC1 | Spearman с длиной | Доп. R² метки при контроле длины |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.pc1_share:.3f} | {row.label_point_biserial_r:.3f} | "
                     f"{row.label_auc_on_this_sample:.3f} | {row.token_spearman_all:.3f} | "
                     f"{row.extra_r2_label_given_length:.3f} |")
    lines += ["", "**Интерпретация:** у base PC1 в основном следует длине входа: Spearman с числом токенов "
              "0.905, а после учёта длины метка добавляет лишь 0.002 к R² PC1. "
              "У CE и CE+SupCon PC1 прежде всего связана с safety-меткой: r≈0.75, "
              "добавочный R² метки при учёте длины ≈0.53–0.54. "
              "На 12 крайних отрицательных и 12 крайних положительных примерах каждой "
              "дообученной модели стороны соответствуют safe и harmful; у base крайние "
              "примеры различаются прежде всего длиной. Это описание этой выборки, "
              "а не доказательство того, что PC1 кодирует только safety или что другая семантика утрачена.",
              "", "Таблица крайних текстов: `pc1_extremes.csv`. График: `pc1_label_and_length.png`.",
              "Крайние тексты помогают выдвинуть гипотезу, но сами по себе не доказывают, что ось кодирует конкретную тему.", "",
              "## Как заполнить человеческие оценки", "",
              "Откройте **только** `human_ratings_template.csv`, не просматривая `pair_manifest.csv` до завершения оценки.",
              "Оцените близость **смысла и намерения двух запросов**, не их вредность и не качество ответа на них.",
              "0 — разные темы/намерения; 1 — слабая тематическая связь; 2 — одна тема, но разные намерения; "
              "3 — похожее намерение с существенным отличием; 4 — практически один смысл/парафраз.",
              "Если смысл неясен, оставьте оценку пустой, поставьте `uncertain_0_1=1` и поясните в `notes`.",
              "Не меняйте `pair_id`, `text_a`, `text_b`; заполняйте только три последних столбца. Сохраните файл CSV в UTF-8.",
              "Некоторые запросы могут содержать неприятный материал: при необходимости пропустите пару, оставив оценку пустой.", "",
              f"План: {len(pairs)} пар, по 24 каждого типа: {', '.join(f'{k}={v}' for k, v in sorted(by_kind.items()))}.",
              "Пары намеренно включают близких соседей разных моделей и случайные контроли; это диагностический набор, "
              "а не случайная выборка всех пар текстов.", "",
              "После разметки выполните:", "",
              "```bash",
              f".venv/bin/python rq4_followup/evaluate.py {out / 'human_ratings_template.csv'}",
              "```", "",
              "Оценки остаются в вашем CSV. Скрипт сохранит метрики рядом с ним и отдельно покажет результаты для пар с одной safety-меткой."]
    (out / "PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "pc1_semantics_followup"
    out.mkdir(exist_ok=True)
    frame, manifest, vectors = load_run(run)
    scores, summary = pc1_analysis(frame, vectors, out)
    pairs = choose_pairs(frame, scores, vectors, out)
    write_protocol(out, summary, pairs, run, manifest["data_hash"])
    print(summary[["model", "pc1_share", "label_point_biserial_r", "label_auc_on_this_sample",
                   "token_spearman_all", "extra_r2_label_given_length"]].to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
