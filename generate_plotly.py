"""Generate missing Plotly JSON files for blog posts."""
import numpy as np
import pandas as pd
import scipy.stats
import hashlib
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path

repo = Path(__file__).parent
data_dir = repo / "data" / "plotly"
data_dir.mkdir(parents=True, exist_ok=True)


def generate_samples(n_users, n_samples, seed=0):
    np.random.seed(seed)
    def encoder(x):
        uid = hashlib.md5(str(x).encode()).hexdigest()
        test_flg = hash(str(x).encode()) % 2
        return (uid, 'test' if test_flg else 'control')
    df = pd.DataFrame(
        list(map(encoder, np.random.randint(0, n_users, 2 * n_samples))),
        columns=['user_id', 'group'],
    )
    return df.assign(metric=scipy.stats.lognorm.rvs(3, loc=100, size=2 * n_samples))


df = generate_samples(1000, 10000)

# ── Bootstrap classical ──────────────────────────────────────────────────────
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    return data[indices]

np.random.seed(42)
a_scores = list(map(np.mean, get_bootstrap_samples(df[df.group == 'test'].metric.values, 1000)))
b_scores = list(map(np.mean, get_bootstrap_samples(df[df.group == 'control'].metric.values, 1000)))

fig = make_subplots(rows=2, cols=2,
    subplot_titles=('Raw Metric - Test', 'Raw Metric - Control',
                    'Bootstrap Means - Test', 'Bootstrap Means - Control'))
colors = {'test': '#636efa', 'control': '#EF553B'}
for col, (grp, label) in enumerate([('test', a_scores), ('control', b_scores)], start=1):
    raw = df[df.group == grp].metric.values.tolist()
    fig.add_trace(go.Histogram(x=raw, nbinsx=100, name=f'Raw {grp}',
                               marker_color=colors[grp], showlegend=False), row=1, col=col)
    fig.add_trace(go.Histogram(x=label, name=f'Bootstrap {grp}',
                               marker_color=colors[grp], showlegend=False), row=2, col=col)
fig.update_layout(title={'text': 'Classical Bootstrap: Raw vs Bootstrap Mean Distribution', 'x': 0.5},
                  template='plotly_dark', height=600)
fig.write_json(data_dir / 'bootstrap-classical.json')
print('bootstrap-classical.json OK')

# ── Bootstrap Poisson ────────────────────────────────────────────────────────
def get_poisson_samples(df, metric='metric', n=100):
    m = len(df)
    weights = scipy.stats.poisson.rvs(1, size=(n, m)).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ df[metric].values

np.random.seed(42)
a_pois = get_poisson_samples(df[df.group == 'test'], 'metric', 1000)
b_pois = get_poisson_samples(df[df.group == 'control'], 'metric', 1000)

fig2 = make_subplots(rows=2, cols=2,
    subplot_titles=('Raw Metric - Test', 'Raw Metric - Control',
                    'Poisson Bootstrap Means - Test', 'Poisson Bootstrap Means - Control'))
for col, (grp, label) in enumerate([('test', a_pois), ('control', b_pois)], start=1):
    raw = df[df.group == grp].metric.values.tolist()
    fig2.add_trace(go.Histogram(x=raw, nbinsx=100, name=f'Raw {grp}',
                                marker_color=colors[grp], showlegend=False), row=1, col=col)
    fig2.add_trace(go.Histogram(x=label.tolist(), name=f'Poisson bootstrap {grp}',
                                marker_color=colors[grp], showlegend=False), row=2, col=col)
fig2.update_layout(title={'text': 'Poisson Bootstrap: Raw vs Bootstrap Mean Distribution', 'x': 0.5},
                   template='plotly_dark', height=600)
fig2.write_json(data_dir / 'bootstrap-poisson.json')
print('bootstrap-poisson.json OK')

# ── Bucketing distribution ───────────────────────────────────────────────────
def make_bucket(df, ids='user_id', groups='group', num_buck=100):
    return (
        df.assign(bucket=df.apply(lambda x: 1 + hash(x[ids]) % num_buck, axis=1))
        .groupby([groups, 'bucket'])
        .agg({'metric': 'sum'})
        .reset_index()
    )

bucketed = make_bucket(df)
fig3 = make_subplots(rows=2, cols=2,
    subplot_titles=('Raw Metric - Test', 'Raw Metric - Control',
                    'Bucket Sums - Test', 'Bucket Sums - Control'))
for col, grp in enumerate(['test', 'control'], start=1):
    fig3.add_trace(go.Histogram(x=df[df.group == grp].metric.values.tolist(), nbinsx=100,
                                name=f'Raw {grp}', marker_color=colors[grp], showlegend=False), row=1, col=col)
    fig3.add_trace(go.Histogram(x=bucketed[bucketed.group == grp].metric.values.tolist(),
                                name=f'Buckets {grp}', marker_color=colors[grp], showlegend=False), row=2, col=col)
fig3.update_layout(title={'text': 'Bucketing: Raw vs Bucket-Sum Distribution', 'x': 0.5},
                   template='plotly_dark', height=600)
fig3.write_json(data_dir / 'bucketing-distribution.json')
print('bucketing-distribution.json OK')
