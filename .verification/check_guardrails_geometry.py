"""Verify delivered notebook math and optionally run all cells on real test inputs.

.venv/bin/python .verification/check_guardrails_geometry.py [--smoke]
Smoke mode requires downloaded base assets in .verification/rq4_base and the
existing fine-tuned model / WildGuardTest caches. It does not edit notebook output.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".verification/matplotlib"),
                  HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", USE_TF="0",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TQDM_DISABLE="1")

import nbformat
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors


def helpers(notebook):
    namespace = dict(np=np, pd=pd, NearestNeighbors=NearestNeighbors, logsumexp=logsumexp,
                     SEED=42, N_PAIRS=10000, TWONN_DISCARD=0.1, KS=(5, 10, 20),
                     TEXT_COLUMN="prompt", LABEL_COLUMN="prompt_harm_label",
                     LABEL_MAP={"safe": 0, "harmful": 1})
    wanted = {"normalize", "sample_pairs", "pair_cosines", "exact_mean_cosine", "spectrum",
              "neighbors", "twonn", "evaluate", "clean_data", "remove_duplicates", "summary",
              "trajectory_row", "class_cosines"}
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.id == "install":
            continue
        tree = ast.parse(cell.source)
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        exec(compile(ast.Module(body=selected, type_ignores=[]), cell.id, "exec"), namespace)
    return namespace


def check_math(notebook):
    ns = helpers(notebook)
    normalize = ns["normalize"]
    rng = np.random.default_rng(31)
    U = normalize(rng.normal(size=(80, 8)))
    y = np.array([0] * 50 + [1] * 30)
    groups = {"all": np.arange(80), "safe": np.flatnonzero(y == 0), "harmful": np.flatnonzero(y == 1)}
    # All ordered off-diagonal pairs: compare against an independent dense reference.
    pairs = {g: np.where(~np.eye(len(ids), dtype=bool)) for g, ids in groups.items()}
    ns.update(y=y, GROUPS=groups, PAIRS=pairs)
    result = ns["evaluate"](U)
    geometry = result[0].set_index("group")
    for group, ids in groups.items():
        V = U[ids]
        gram = V @ V.T
        offdiag = ~np.eye(len(V), dtype=bool)
        cosines = gram[offdiag]
        np.testing.assert_allclose(geometry.loc[group, "mean_cosine_exact"], cosines.mean(), atol=1e-14)
        np.testing.assert_allclose(geometry.loc[group, "mean_cosine_mc"], cosines.mean(), atol=1e-14)
        expected = np.log(np.exp(-2 * (2 - 2 * cosines)).mean())
        np.testing.assert_allclose(geometry.loc[group, "uniformity_t2"], expected, atol=1e-14)
    expected_alignment = np.mean([np.square(U[a][:, None] - U[a][None, :]).sum(2)[~np.eye(len(a), dtype=bool)].mean()
                                  for a in [groups["safe"], groups["harmful"]]])
    np.testing.assert_allclose(result[2], expected_alignment, atol=1e-14)
    # Independent pair-matrix check of the newly reported class separation.
    cos = ns["class_cosines"](U, y)
    cross = (U[y == 0] @ U[y == 1].T).mean()
    np.testing.assert_allclose(cos["cos_safe_harm"], cross, atol=1e-14)
    np.testing.assert_allclose(cos["cosine_gap_macro"],
                               1 - expected_alignment / 2 - cross, atol=1e-14)
    np.testing.assert_allclose(ns["summary"](result)["cos_within_safe"],
                               geometry.loc["safe", "mean_cosine_exact"], atol=1e-14)
    ids, distances = ns["neighbors"](U, 20)
    dense = np.linalg.norm(U[:, None] - U[None, :], axis=2)
    np.fill_diagonal(dense, np.inf)
    np.testing.assert_array_equal(ids, np.argsort(dense, axis=1)[:, :20])
    np.testing.assert_allclose(distances, np.take_along_axis(dense, ids, axis=1), atol=1e-12)
    local = (y[ids[:, :10]] == y[:, None]).mean(1)
    np.testing.assert_allclose(result[1].set_index("k").loc[10, "macro"], (local[:50].mean() + local[50:].mean()) / 2)
    np.testing.assert_allclose(result[1]["null_macro"], (len(y) - 2) / (2 * (len(y) - 1)))

    # Centred isotropic axes have PR=d; antipodal collapse has zero cosine but PR=1.
    axes = np.concatenate([np.eye(8), -np.eye(8)])
    stats, fractions = ns["spectrum"](axes)
    np.testing.assert_allclose(stats["participation_ratio"], 8)
    np.testing.assert_allclose(stats["pc1_share"], 1 / 8)
    collapsed = np.tile([1., 0., 0.], (80, 1))
    tied_ids, tied_distances = ns["neighbors"](collapsed, 20)
    assert not np.any(tied_ids == np.arange(80)[:, None])
    assert all(len(set(row)) == 20 for row in tied_ids)
    assert ns["spectrum"](collapsed)[0]["participation_ratio"] == 0
    dim, invalid = ns["twonn"](tied_distances)
    assert np.isnan(dim) and invalid == 1
    # Non-axis duplicates also have exactly zero distance (avoid cancellation).
    generic_duplicates = np.repeat(U[:40], 2, axis=0)
    _, duplicate_distances = ns["neighbors"](generic_duplicates, 2)
    np.testing.assert_array_equal(duplicate_distances[:, 0], 0)
    assert ns["twonn"](duplicate_distances)[1] == 1
    antipodal = np.vstack([collapsed[:40], -collapsed[:40]])
    np.testing.assert_allclose(ns["spectrum"](antipodal)[0]["participation_ratio"], 1)
    assert abs(ns["exact_mean_cosine"](antipodal)) < 0.02

    # TwoNN regression against the analytic Pareto CDF, then a sampled 3D cloud.
    n, dimension = 10000, 7
    mu = (1 - (np.arange(n) + 0.5) / n) ** (-1 / dimension)
    estimate, _ = ns["twonn"](np.column_stack([np.ones(n), mu]))
    np.testing.assert_allclose(estimate, dimension, rtol=0.002)
    _, cloud_d = ns["neighbors"](rng.random((2500, 3)), 2)
    cloud_estimate, _ = ns["twonn"](cloud_d)
    assert 2.5 < cloud_estimate < 3.6, cloud_estimate
    for seed in [0, 42]:
        left, right = ns["sample_pairs"](80, 1000, seed)
        assert (left != right).all()
        np.testing.assert_array_equal((left, right), ns["sample_pairs"](80, 1000, seed))

    # Text preparation counts conflicting/prefixed duplicates instead of leaking them.
    raw = pd.DataFrame({"prompt": ["a", "query: a", "b", "b", "", None, "c", "d"],
                        "prompt_harm_label": ["safe", "safe", "safe", "harmful", "safe", "safe", None, "harmful"]})
    cleaned, audit = ns["clean_data"](raw)
    assert cleaned["input_text"].tolist() == ["query: a", "query: d"]
    assert audit["missing_removed"] == 3
    assert audit["text_duplicates"] == {"conflicting_rows": 2, "duplicate_rows": 1}
    print("PASS: notebook schema/syntax; cosine, spectrum, alignment/uniformity, kNN, TwoNN, duplicates.", flush=True)


def smoke(notebook):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    base = ROOT / ".verification/rq4_base"
    final = ROOT / ".cache/guardrails/hub/models--pineee--e5-small-v2-hivetrace-finetune-CE-SupCon/snapshots/d7905a98d3000ce2690c09171ba02ab3e9315add"
    arrow = ROOT / ".cache/guardrails/datasets/allenai___wildguardmix/wildguardtest/0.0.0/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/wildguardmix-test.arrow"
    frame = Dataset.from_file(str(arrow)).to_pandas()
    small = pd.concat([frame.loc[frame.prompt_harm_label == c].drop_duplicates("prompt").sample(70, random_state=7)
                       for c in ["harmful", "unharmful"]], ignore_index=True)
    inputs = ROOT / ".verification/rq4_smoke_inputs.csv"
    small.to_csv(inputs, index=False)
    t1, t2 = AutoTokenizer.from_pretrained(base), AutoTokenizer.from_pretrained(final)
    text = ["query: " + s for s in small.prompt]
    assert t1(text, truncation=True, max_length=512)["input_ids"] == t2(text, truncation=True, max_length=512)["input_ids"]
    output_root = ROOT / ".verification/rq4_smoke"
    namespace = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.id == "install":
            continue
        source = cell.source
        if cell.id == "config":
            source = source.replace("REQUIRE_GPU = True", "REQUIRE_GPU = False")
            source += f'\nDATA_SOURCE = "local"\nLOCAL_DATA_PATH = {str(inputs)!r}\nUSE_DRIVE = False\n'
            source += f'BASE.update(source={str(base)!r}, revision=None)\nFINAL.update(source={str(final)!r}, revision=None)\n'
        if cell.id == "storage":
            source = source.replace('Path("guardrails_geometry_results")', f'Path({str(output_root)!r})')
        print("RUN", cell.id, flush=True)
        exec(compile(source, cell.id, "exec"), namespace)
    assert len(namespace["df"]) >= 100
    comparison = namespace["comparison"]
    assert np.isfinite(comparison.to_numpy()).all()
    assert comparison.loc["knn_macro@10", "finetuned"] != comparison.loc["knn_macro@10", "base"]
    run = namespace["RUN"]
    for filename in ["before_after.csv", "geometry_by_class.csv", "knn_purity.csv", "manifest.json",
                     "eval_inputs.csv", "pairs.npz", "base_embeddings.npz", "finetuned_embeddings.npz",
                     "cosines_and_spectra.png", "summary.txt"]:
        assert (run / filename).exists(), filename
    # A second encode must reuse the on-disk cache, preserving exact arrays.
    cached, _ = namespace["encode"](namespace["BASE"])
    with np.load(run / "base_embeddings.npz") as saved:
        np.testing.assert_array_equal(cached, saved["raw"])
    # Exercise optional trajectory with fixture weights/steps, never claim real training steps.
    namespace["CHECKPOINTS"] = [dict(name="fixture_step_100", source=str(final), revision=None, step=100)]
    namespace["FINAL"]["step"] = 200
    logpath = run / "fixture_loss.csv"
    pd.DataFrame({"step": [99, 100, 201], "loss_total": [7., 0.8, 9.], "loss_ce": [3., 0.2, 4.]}).to_csv(logpath, index=False)
    namespace["LOSS_CSV"] = logpath
    checkpoint_code = next(c.source for c in notebook.cells if c.id == "checkpoints")
    exec(compile(checkpoint_code, "checkpoints_fixture", "exec"), namespace)
    trajectory = namespace["trajectory"].set_index("step")
    assert trajectory.loc[100, "loss_total"] == 0.8
    assert pd.isna(trajectory.loc[200, "loss_total"]) and pd.isna(trajectory.loc[0, "loss_total"])
    print("PASS: all cells, real base/fine-tuned weights, real 140-row WildGuardTest subset, cache and checkpoint/log branch.")
    print("SMOKE OUTPUT:", run)


def check_checkpoint_join(notebook):
    """Exercise exact-step joins with archive column names and an earlier best model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = helpers(notebook)
    rng = np.random.default_rng(6)
    U = ns["normalize"](rng.normal(size=(80, 8)))
    labels = np.repeat([0, 1], 40)
    groups = {"all": np.arange(80), "safe": np.arange(40), "harmful": np.arange(40, 80)}
    ns.update(y=labels, GROUPS=groups,
              PAIRS={g: ns["sample_pairs"](len(ids), 500) for g, ids in groups.items()})
    result = ns["evaluate"](U)
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)
        log = run / "archive_columns.csv"
        pd.DataFrame({"step": [99, 100, 201], "loss": [7., 0.8, 9.],
                      "train_loss_ce": [3., 0.2, 4.], "train_loss_supcon": [4., 0.6, 5.]}).to_csv(log, index=False)
        ns.update(pd=pd, plt=plt, json=json, RUN=run, LOSS_CSV=log, LOSS_FAMILY="fixture",
                  REVIEW_RATIO=0.5, display=lambda *args: None,
                  BASE=dict(name="base", step=0), FINAL=dict(name="finetuned", step=100),
                  CHECKPOINTS=[dict(name="fixture_later", step=200)],
                  results={"base": result, "finetuned": result},
                  model_identities={"base": {}, "finetuned": {}}, encode=lambda spec: (U, {}))
        source = next(c.source for c in notebook.cells if c.id == "checkpoints")
        exec(compile(source, "checkpoint_join_fixture", "exec"), ns)
        trajectory = ns["trajectory"].set_index("step")
        assert trajectory.loc[100, "loss_total"] == 0.8
        assert trajectory.loc[100, "loss_ce"] == 0.2
        assert pd.isna(trajectory.loc[200, "loss_total"]) and pd.isna(trajectory.loc[0, "loss_total"])
    print("PASS: archive loss aliases, exact-step joins, best checkpoint earlier than later weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "guardrails_geometry_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    check_math(notebook)
    check_checkpoint_join(notebook)
    if args.smoke:
        smoke(notebook)
