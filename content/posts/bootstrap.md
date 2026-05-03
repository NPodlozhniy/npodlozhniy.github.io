---
title: Bootstrap Methods for AB Testing
author: Nikita Podlozhniy
summary: >-
  Classical and Poisson bootstrap as non-parametric tools for normalizing skewed
  metrics and enabling valid T-tests on small samples
date: '2022-08-16'
format:
  hugo-md:
    output-file: bootstrap.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - bootstrap
  - variance-reduction
  - non-parametric
categories:
  - posts
---


<!--
<script src="https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js" integrity="sha384-c9c+LnTbwQ3aujuU7ULEPVvgLs+Fn6fJUvIGTsuu1ZcCf11fiEubah0ttpca4ntM sha384-6V1/AdqZRWk1KAlWbKBlGhN7VG4iE/yAZcO6NZPMF8od0vukrvr0tg4qY6NSrItx" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js" integrity="sha384-ZvpUoO/+PpLXR1lu4jmpXWu80pZlYUAfxl5NsBMWOEPSjUn/6Z/hRTt8+pR6L4N2" crossorigin="anonymous" data-relocate-top="true"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
-->


## Background

We at HomeBuddy frequently encounter metrics with heavy-tailed, non-normal distributions --- revenue per session, time on site, number of clicks.
Standard T-tests assume the sample mean is approximately normally distributed, which holds asymptotically by the Central Limit Theorem, but can require very large samples when the underlying distribution is highly skewed.

**Bootstrap** is a non-parametric resampling technique that estimates the empirical sampling distribution of any statistic without any distributional assumptions.
By bootstrapping the mean, we can apply a T-test even on small samples whose raw distribution would otherwise invalidate the normality assumption.

Two implementations are widely used: **Classical bootstrap** (sampling with replacement) and **Poisson bootstrap** (random weights from $\mathrm{Poisson}(1)$).

### Prerequisites

Python 3.11.4 with standard scientific stack: NumPy, SciPy, pandas, Plotly.

## Synthetic Data

We simulate a dataset where each row is a single measurement from a user assigned to test or control.
The metric follows a lognormal distribution --- a common approximation for revenue or engagement signals.

<details class="code-fold">
<summary>Data generation</summary>

``` python
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
df.sample(3)
```

</details>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

|       | user_id                          | group   | metric     |
|-------|----------------------------------|---------|------------|
| 2512  | f2fc990265c712c49d51a18a32b39f0c | test    | 100.555829 |
| 19645 | 98d6f58ab0dafbb86b083a001561bb34 | control | 100.458050 |
| 8912  | 07a96b1f61097ccb54be14d6a47439b0 | control | 100.369466 |

</div>

## Classical Bootstrap

The best estimate of the population distribution is the empirical distribution of the observed sample.
Classical bootstrap draws $n$ samples **with replacement** from the observed data, repeating this process $B$ times to build the empirical sampling distribution of the target statistic.

The weight of each observation in a single bootstrap draw follows a multinomial distribution:

$$ \forall\, \text{resample}:\; (w_1, \ldots, w_n) \sim \mathrm{Multinomial}_n\!\left(\tfrac{1}{n}, \ldots, \tfrac{1}{n}\right) $$

A confidence interval is then obtained as empirical quantiles of the bootstrapped statistic distribution.

<details class="code-fold">
<summary>Classical bootstrap implementation</summary>

``` python
def get_bootstrap_samples(data, n_samples):
    """Draw n_samples resamples (with replacement) equal in size to data."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    return data[indices]


def get_stat_intervals(stat, alpha):
    """Two-sided confidence interval via empirical quantiles."""
    return np.percentile(stat, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])


np.random.seed(42)
a_scores = list(map(np.mean, get_bootstrap_samples(df[df.group == 'test'].metric.values, 1000)))
b_scores = list(map(np.mean, get_bootstrap_samples(df[df.group == 'control'].metric.values, 1000)))

print('Classical bootstrap — 95% CI')
print('Test:   ', np.round(get_stat_intervals(a_scores, 0.05), 2))
print('Control:', np.round(get_stat_intervals(b_scores, 0.05), 2))
```

</details>

    Classical bootstrap — 95% CI
    Test:    [164.83 232.18]
    Control: [152.18 196.49]

The raw metric distribution is heavily right-skewed (lognormal). After bootstrapping the mean, the resulting distribution of bootstrap means is approximately normal --- validating the use of a T-test.

<details class="code-fold">
<summary>Classical bootstrap: raw vs bootstrapped distribution</summary>

``` python
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path


def find_git_repo(path=None):
    source = Path(path) if path else Path.cwd()
    for p in source.parents:
        if (p / '.git').is_dir():
            return p


fig = make_subplots(rows=2, cols=2,
    subplot_titles=('Raw Metric — Test', 'Raw Metric — Control',
                    'Bootstrap Means — Test', 'Bootstrap Means — Control'))

for col, (grp, label) in enumerate([('test', a_scores), ('control', b_scores)], start=1):
    raw = df[df.group == grp].metric.values
    fig.add_trace(go.Histogram(x=raw, nbinsx=100, name=f'Raw {grp}',
                               marker_color='#636efa' if grp == 'test' else '#EF553B',
                               showlegend=False), row=1, col=col)
    fig.add_trace(go.Histogram(x=label, name=f'Bootstrap {grp}',
                               marker_color='#636efa' if grp == 'test' else '#EF553B',
                               showlegend=False), row=2, col=col)

fig.update_layout(title={'text': 'Classical Bootstrap: Raw vs Bootstrap Mean Distribution',
                         'x': 0.5},
                  template='plotly_dark', height=600)
fig.show()
```

</details>
{{< plotly obj=bootstrap-classical >}}

## Poisson Bootstrap

An alternative to drawing resamples is to assign a random weight to each observation.
If weights are drawn independently from $\mathrm{Poisson}(1)$, the resulting weighted mean approximates the classical bootstrap mean while allowing **distributed computation** --- each worker can independently draw its own set of weights and compute a local weighted sum, with no need to share data.

$$ \forall\, i:\; w_i \sim \mathrm{Poisson}(1) $$

The weighted mean per bootstrap iteration is then:

$$ \bar{x}^{(b)} = \frac{\sum_i w_i^{(b)} x_i}{\sum_i w_i^{(b)}} $$

<details class="code-fold">
<summary>Poisson bootstrap implementation</summary>

``` python
def get_poisson_samples(df, metric='metric', n=100):
    m = len(df)
    weights = scipy.stats.poisson.rvs(1, size=(n, m)).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ df[metric].values


np.random.seed(42)
a_pois = get_poisson_samples(df[df.group == 'test'], 'metric', 1000)
b_pois = get_poisson_samples(df[df.group == 'control'], 'metric', 1000)

print('Poisson bootstrap — 95% CI')
print('Test:   ', np.round(get_stat_intervals(a_pois, 0.05), 2))
print('Control:', np.round(get_stat_intervals(b_pois, 0.05), 2))
```

</details>

    Poisson bootstrap — 95% CI
    Test:    [164.24 234.02]
    Control: [153.27 194.47]

<details class="code-fold">
<summary>Poisson bootstrap: raw vs bootstrapped distribution</summary>

``` python
fig2 = make_subplots(rows=2, cols=2,
    subplot_titles=('Raw Metric — Test', 'Raw Metric — Control',
                    'Poisson Bootstrap Means — Test', 'Poisson Bootstrap Means — Control'))

for col, (grp, label) in enumerate([('test', a_pois), ('control', b_pois)], start=1):
    raw = df[df.group == grp].metric.values
    fig2.add_trace(go.Histogram(x=raw, nbinsx=100, name=f'Raw {grp}',
                                marker_color='#636efa' if grp == 'test' else '#EF553B',
                                showlegend=False), row=1, col=col)
    fig2.add_trace(go.Histogram(x=label, name=f'Poisson bootstrap {grp}',
                                marker_color='#636efa' if grp == 'test' else '#EF553B',
                                showlegend=False), row=2, col=col)

fig2.update_layout(title={'text': 'Poisson Bootstrap: Raw vs Bootstrap Mean Distribution',
                          'x': 0.5},
                   template='plotly_dark', height=600)
fig2.show()
```

</details>
{{< plotly obj=bootstrap-poisson >}}

## Conclusion

Both bootstrap variants transform a heavily skewed metric distribution into an approximately normal distribution of bootstrap means, enabling a valid T-test regardless of the original distribution shape.

|  | Classical | Poisson |
|------------------------|------------------------|------------------------|
| **Works on small samples** | Yes | Yes |
| **Distributable** | No | Yes |
| **Computation** | Faster (simple resampling) | Slower (matrix multiply) |
| **Implementation** | `np.random.randint` | `scipy.stats.poisson` |

For large-scale distributed data pipelines (Spark, Flink), Poisson bootstrap is the practical choice since each partition can independently compute its weighted sum.
For single-machine analysis, classical bootstrap is simpler and faster.
