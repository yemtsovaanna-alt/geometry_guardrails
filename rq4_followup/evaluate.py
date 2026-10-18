"""Score human semantic judgments against the three frozen encoders.

Usage: .venv/bin/python rq4_followup/evaluate.py PATH_TO_FILLED_CSV
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODELS = ("base", "ce", "finetuned")
SEED = 42


def digest(a, b):
    return hashlib.sha256((a + "\0" + b).encode("utf-8")).hexdigest()


def correlation(frame, model):
    if len(frame) < 10 or frame.semantic_similarity_0_4.nunique() < 2:
        return np.nan
    return float(spearmanr(frame.semantic_similarity_0_4, frame[f"cos_{model}"]).statistic)


def bootstrap(frame, repeats=1000):
    anchors = frame.anchor_row_id.unique()
    rng = np.random.default_rng(SEED)
    by_anchor = {anchor: frame.loc[frame.anchor_row_id == anchor] for anchor in anchors}
    samples = []
    for _ in range(repeats):
        selected = rng.choice(anchors, size=len(anchors), replace=True)
        sampled = pd.concat([by_anchor[a] for a in selected], ignore_index=True)
        samples.append([correlation(sampled, model) for model in MODELS])
    return np.asarray(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv", type=Path)
    args = parser.parse_args()
    ratings_path = args.ratings_csv.resolve()
    ratings = pd.read_csv(ratings_path, dtype={"pair_id": str, "text_a": str, "text_b": str})
    manifest = pd.read_csv(ratings_path.parent / "pair_manifest.csv", dtype={"pair_id": str})
    required = {"pair_id", "text_a", "text_b", "semantic_similarity_0_4", "uncertain_0_1"}
    assert required.issubset(ratings.columns), f"Missing columns: {required - set(ratings.columns)}"
    assert ratings.pair_id.is_unique and manifest.pair_id.is_unique
    assert set(ratings.pair_id) == set(manifest.pair_id), "Pair IDs differ from manifest."
    frame = ratings.merge(manifest, on="pair_id", validate="one_to_one")
    assert all(digest(a, b) == h for a, b, h in
               zip(frame.text_a, frame.text_b, frame.text_sha256)), "Pair text was changed."
    score = pd.to_numeric(frame.semantic_similarity_0_4, errors="coerce")
    invalid = frame.semantic_similarity_0_4.notna() & ~score.isin([0, 1, 2, 3, 4])
    assert not invalid.any(), f"Invalid scores at: {frame.loc[invalid, 'pair_id'].tolist()}"
    uncertain = pd.to_numeric(frame.uncertain_0_1, errors="coerce").fillna(0)
    assert uncertain.isin([0, 1]).all(), "uncertain_0_1 must be blank, 0 or 1."
    frame["semantic_similarity_0_4"] = score
    frame["uncertain_0_1"] = uncertain.astype(int)
    rated = frame.loc[score.notna() & (uncertain == 0)].copy()
    coverage = dict(total_pairs=len(frame), rated_certain=len(rated),
                    uncertain=int((uncertain == 1).sum()), missing=int(score.isna().sum()))
    lines = ["# Человеческие оценки смыслового сходства", "",
             f"Размечено с уверенной оценкой: {coverage['rated_certain']} из {coverage['total_pairs']}; "
             f"пропущено: {coverage['missing']}; отмечено как неуверенные: {coverage['uncertain']}.", "",
             "Spearman ρ измеряет согласие косинуса модели с оценкой человека. "
             "Интервалы ниже — кластерный bootstrap по 24 опорным текстам; они не учитывают "
             "вариацию обучения или межэкспертные расхождения.",
             "Этот набор пар выбран для диагностики, поэтому метрики нельзя выдавать за "
             "качество на случайной выборке всех пар или на независимом стандартном бенчмарке.", ""]
    rows = []
    if len(rated) >= 20:
        for group, subset in (("all", rated),
                              ("same_safety_label", rated.loc[rated.label_a == rated.label_b]),
                              ("different_safety_label", rated.loc[rated.label_a != rated.label_b])):
            if len(subset) < 10:
                continue
            boot = bootstrap(subset)
            for col, model in enumerate(MODELS):
                valid = boot[:, col][np.isfinite(boot[:, col])]
                low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
                rows.append(dict(group=group, model=model, n_pairs=len(subset),
                                 spearman_rho=correlation(subset, model),
                                 bootstrap_low=low, bootstrap_high=high))
            if group == "all" or group == "same_safety_label":
                lines.append(f"## {group}: {len(subset)} пар")
                lines.append("")
                lines.append("| Модель | Spearman ρ | 95% bootstrap интервал |")
                lines.append("|---|---:|---:|")
                for row in rows[-3:]:
                    lines.append(f"| {row['model']} | {row['spearman_rho']:.3f} | "
                                 f"[{row['bootstrap_low']:.3f}, {row['bootstrap_high']:.3f}] |")
                lines.append("")
                for left, right in (("ce", "base"), ("finetuned", "base"), ("finetuned", "ce")):
                    a, b = MODELS.index(left), MODELS.index(right)
                    differences = boot[:, a] - boot[:, b]
                    differences = differences[np.isfinite(differences)]
                    if len(differences):
                        low, high = np.quantile(differences, [0.025, 0.975])
                        lines.append(f"Δρ {left} − {right}: "
                                     f"{correlation(subset, left) - correlation(subset, right):+.3f} "
                                     f"(95% bootstrap [{low:+.3f}, {high:+.3f}]).")
                lines.append("")
        pd.DataFrame(rows).to_csv(ratings_path.parent / "semantic_metrics.csv", index=False)
    else:
        lines.append("Для расчёта нужно минимум 20 уверенно оценённых пар.")
    report = ratings_path.parent / "semantic_evaluation.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Coverage:", coverage)
    print("Report:", report)


if __name__ == "__main__":
    main()
