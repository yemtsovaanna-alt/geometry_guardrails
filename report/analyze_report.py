"""Reproduce report metrics from saved embeddings; never re-encode or train."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/e5-report-matplotlib')
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import skew
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'guardrails_geometry_results/run_20261018T134157_442709Z'
OUT = ROOT / 'report/data'
OUT.mkdir(parents=True, exist_ok=True)
reference = pd.read_csv(RUN / 'three_way_comparison.csv', index_col='metric')
frame = pd.read_csv(RUN / 'eval_inputs.csv')
y = frame.label.to_numpy()
n = len(y)
pairs = np.load(RUN / 'pairs.npz')['all']
checks, metrics, by_class, neighborhoods = [], {}, [], {}

def spectral(x):
    z = x - x.mean(0)
    ev = np.linalg.eigvalsh(z.T @ z / len(x))[::-1].clip(0)
    p = ev / ev.sum()
    positive = p[p > 0]
    return dict(participation_ratio=float(1 / (p @ p)),
                pc1_share=float(p[0]), variance_trace=float(ev.sum()),
                effective_rank=float(np.exp(-(positive * np.log(positive)).sum())),
                dims_90pct=int(np.searchsorted(p.cumsum(), .9) + 1))

for model in ['base', 'ce', 'finetuned']:
    saved = np.load(RUN / f'{model}_embeddings.npz')
    assert np.array_equal(saved['row_ids'], frame.row_id.to_numpy())
    assert np.array_equal(saved['labels'], y)
    x = saved['raw'].astype(np.float64)
    assert x.shape == (n, 384) and np.isfinite(x).all()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cosine = (x @ x.T).clip(-1, 1)
    distance = (1 - cosine).clip(0)
    np.fill_diagonal(distance, 0)
    row = spectral(x)
    row['mean_cosine_exact'] = float((cosine.sum() - np.trace(cosine)) / (n*(n-1)))
    sampled = cosine[pairs[0], pairs[1]]
    row['mean_cosine_mc'] = float(sampled.mean())
    row['uniformity_t2'] = float(logsumexp(-4 * (1 - sampled)) - np.log(len(sampled)))
    means, traces, within = {}, {}, {}
    for c, group in [(0, 'safe'), (1, 'harmful')]:
        mask = y == c
        means[c] = x[mask].mean(0)
        stats = spectral(x[mask])
        traces[c] = stats['variance_trace']
        block = cosine[np.ix_(mask, mask)]
        nc = int(mask.sum())
        within[c] = float((block.sum() - np.trace(block)) / (nc*(nc-1)))
        by_class.append(dict(model=model, group=group, n=nc, **stats))
    row['cos_within_safe'], row['cos_within_harm'] = within[0], within[1]
    row['cos_safe_harm'] = float(means[0] @ means[1])
    row['cosine_gap_macro'] = (within[0]+within[1])/2-row['cos_safe_harm']
    row['label_alignment_macro'] = 2-within[0]-within[1]
    row['within_scatter'] = sum(float((y==c).mean())*traces[c] for c in [0,1])
    between = sum(float((y==c).mean())*float(np.square(means[c]-x.mean(0)).sum()) for c in [0,1])
    row['between_scatter'] = between
    row['between_over_within'] = between/row['within_scatter']
    row['centroid_cosine'] = float(means[0]@means[1]/np.linalg.norm(means[0])/np.linalg.norm(means[1]))
    sil = silhouette_samples(distance, y, metric='precomputed')
    row['silhouette_macro'] = float(np.mean([sil[y==c].mean() for c in [0,1]]))
    row['silhouette_micro'] = float(sil.mean())
    np.fill_diagonal(distance, np.inf)
    ids = np.argsort(distance, axis=1, kind='stable')[:, :100]
    neighborhoods[model] = ids[:, :10]
    for k in [1, 5, 10, 20, 50, 100]:
        local = (y[ids[:, :k]] == y[:, None]).mean(1)
        row[f'knn_macro@{k}'] = float(np.mean([local[y==c].mean() for c in [0,1]]))
    local = (y[ids[:, :10]] == y[:, None]).mean(1)
    row['knn_safe@10'] = float(local[y==0].mean())
    row['knn_harmful@10'] = float(local[y==1].mean())
    row['knn_micro@10'] = float(local.mean())
    first_d = np.linalg.norm(x - x[ids[:, 0]], axis=1)
    second_d = np.linalg.norm(x - x[ids[:, 1]], axis=1)
    a = np.sort(np.log(second_d/first_d))[:int(n*.9)]
    b = -np.log1p(-np.arange(len(a))/n)
    row['twonn_id'] = float(a@b/(a@a))
    row['hubness_skew@10'] = float(skew(np.bincount(ids[:, :10].ravel(), minlength=n)))
    own = np.where(y[:,None]==y[None,:], distance, np.inf).min(1)
    opposite = np.where(y[:,None]!=y[None,:], distance, np.inf).min(1)
    row['other_class_closer_fraction'] = float((opposite < own).mean())
    for metric in reference.index:
        delta = row[metric]-float(reference.loc[metric, model])
        checks.append(dict(model=model, metric=metric, difference=delta))
        assert abs(delta) < 1e-8, (model, metric, delta)
    metrics[model] = row
    print(model, {k: round(row[k], 5) for k in ['effective_rank','dims_90pct','silhouette_macro','other_class_closer_fraction','between_over_within']})

df = pd.DataFrame(metrics)
df['joint_minus_base'] = df.finetuned-df.base
df['joint_minus_ce'] = df.finetuned-df.ce
df.to_csv(OUT / 'all_geometry_metrics.csv', index_label='metric')
pd.DataFrame(by_class).to_csv(OUT / 'spectral_by_class.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'reproduction_checks.csv', index=False)
overlap = {}
for a,b in [('base','ce'),('base','finetuned'),('ce','finetuned')]:
    score = np.array([len(set(i)&set(j))/10 for i,j in zip(neighborhoods[a],neighborhoods[b])])
    overlap[a+'__'+b] = dict(micro=float(score.mean()),macro=float(np.mean([score[y==c].mean() for c in [0,1]])))
(OUT / 'neighbor_overlap.json').write_text(json.dumps(overlap, indent=2))
sources = [RUN / f'{m}_embeddings.npz' for m in metrics] + [RUN/'eval_inputs.csv',RUN/'pairs.npz']
(OUT / 'analysis_manifest.json').write_text(json.dumps(dict(
    source_run=str(RUN.relative_to(ROOT)), n=n, dimension=384,
    geometry='L2-normalized, float64; covariance divided by n',
    all_39_original_metrics_reproduced=True,
    max_absolute_difference=max(abs(c['difference']) for c in checks),
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2))
print('All 39 original values reproduced. Neighborhood overlap:', overlap)
