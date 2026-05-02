---
title: 'Bucketing: Variance Reduction and Faster AB Tests'
author: Nikita Podlozhniy
summary: >-
  How aggregating observations into buckets normalises skewed metrics, reduces
  variance, and cuts computation — with a probability proof for choosing the
  right bucket count
date: '2026-02-01'
format:
  hugo-md:
    output-file: bucketing.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - bucketing
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

We at HomeBuddy deal with metrics that are measured at the **event level** but compared at the **user level**: each user may have dozens of sessions, purchases, or page views.
Working with millions of raw rows is computationally expensive, and the raw metric distribution is typically heavy-tailed, making a T-test unreliable without a large enough sample.

**Bucketing** solves both problems at once: users are randomly partitioned into $b$ buckets, and the metric is aggregated within each bucket.
The resulting $b$ bucket-level values are nearly normally distributed by the Central Limit Theorem, enabling a T-test with far fewer data points --- and dramatically reduced storage and computation.

### Prerequisites

Python 3.11.4 with NumPy, SciPy, pandas, and Plotly.

## Synthetic Data

We reuse the same lognormal simulation from the [Delta Method](../delta-method) post: 1 000 unique users, 10 000 rows per group, metric sampled from $\mathrm{LogNormal}(3, 100)$.

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
| 2512  | f2fc990265c712c49d51a18a32b39f0c | control | 100.555829 |
| 19645 | 98d6f58ab0dafbb86b083a001561bb34 | test    | 100.458050 |
| 8912  | 07a96b1f61097ccb54be14d6a47439b0 | test    | 100.369466 |

</div>

## Bucketing

Each user is deterministically assigned to one of $b$ buckets using the hash of their ID modulo $b$.
The metric is then **summed** within each bucket, yielding $b$ approximately normally distributed values per group.

<details class="code-fold">
<summary>Bucketing function</summary>

``` python
def make_bucket(df, ids='user_id', groups='group', num_buck=100):
    return (
        df.assign(bucket=df.apply(lambda x: 1 + hash(x[ids]) % num_buck, axis=1))
        .groupby([groups, 'bucket'])
        .agg({'metric': 'sum'})
        .reset_index()
    )


bucketed = make_bucket(df)
bucketed.sample(3)
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

|     | group   | bucket | metric       |
|-----|---------|--------|--------------|
| 180 | test    | 83     | 18836.189558 |
| 2   | control | 3      | 35560.334840 |
| 169 | test    | 72     | 19833.689535 |

</div>

## Distribution Before and After Bucketing

The raw metric is lognormally distributed --- highly right-skewed.
After bucketing into 100 groups, the bucket sums are approximately normal, justifying the T-test.

<details class="code-fold">
<summary>Distribution comparison chart</summary>

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
                    'Bucket Sums — Test', 'Bucket Sums — Control'))

colors = {'test': '#636efa', 'control': '#EF553B'}
for col, grp in enumerate(['test', 'control'], start=1):
    fig.add_trace(go.Histogram(
        x=df[df.group == grp].metric.values, nbinsx=100,
        name=f'Raw {grp}', marker_color=colors[grp], showlegend=False), row=1, col=col)
    fig.add_trace(go.Histogram(
        x=bucketed[bucketed.group == grp].metric.values,
        name=f'Buckets {grp}', marker_color=colors[grp], showlegend=False), row=2, col=col)

fig.update_layout(
    title={'text': 'Bucketing: Raw vs Bucket-Sum Distribution', 'x': 0.5},
    template='plotly_dark', height=600)
fig.show()
```

</details>
{{< plotly obj=bucketing-distribution >}}

## Choosing the Number of Buckets

A critical constraint: **the number of users must substantially exceed the number of buckets**.
If $n \approx b$, many buckets will be empty, making the statistic meaningless.

Let $P(n, b)$ be the probability that all $b$ buckets contain at least one of $n$ users.
Counting **successful arrangements** as inserting $b-1$ dividers in $n-1$ positions yields $\binom{n-1}{b-1}$.
Counting **all arrangements** (dividers can repeat) gives $(n+1)^{b-1}$.
Therefore:

$$ P(n, b) = \frac{\binom{n-1}{b-1}}{(n + 1)^{b - 1}} $$

Applying Stirling's approximation $n! \approx \sqrt{2\pi n}\,(n/e)^n$:

$$ P(n, b) \approx \frac{n!}{(n-b)!\cdot b! \cdot n^b} \xrightarrow{n \to \infty} 0 $$

Two particularly informative regimes:

- $b \sim n$: $\;P \sim 1/n^n \to 0$ (vanishingly small probability)
- $b \sim n/2$: $\;P \sim \sqrt{2/(\pi n)}\,(4/n)^{n/2} \to 0$ (also zero)

**Practical rule of thumb:** keep $b \ll n$, typically $b \leq n/10$.
With 1 000 users, use at most 100 buckets; with 10 000 users, 200--500 buckets is reasonable.

<details class="code-fold">
<summary>Empty bucket rate vs bucket count</summary>

``` python
n_users = df.user_id.nunique() // 2  # per group
print(f'Users per group: {n_users}')

for b in [10, 50, 100, 200, 500]:
    grp = df[df.group == 'test']
    bucketed_check = make_bucket(grp, num_buck=b)
    filled = bucketed_check.bucket.nunique()
    print(f'  {b:4d} buckets → {filled:4d} non-empty  ({100*(b-filled)/b:.1f}% empty)')
```

</details>

    Users per group: 500
        10 buckets →   10 non-empty  (0.0% empty)
        50 buckets →   50 non-empty  (0.0% empty)
       100 buckets →   99 non-empty  (1.0% empty)
       200 buckets →  185 non-empty  (7.5% empty)
       500 buckets →  320 non-empty  (36.0% empty)

## Conclusion

Bucketing is one of the simplest variance-reduction techniques available in AB testing:

- **Normalises** the metric distribution, making T-tests valid on far fewer data points.
- **Reduces computation**: instead of testing millions of rows, you compare $b$ bucket aggregates.
- **Distributable**: bucket assignment is a hash modulo --- trivially parallelisable in any SQL or Spark query.

The main constraint is bucket count: choose $b$ significantly smaller than the number of users per group to avoid empty buckets that would bias the test statistic.
For our typical experiment sizes at HomeBuddy, 100--200 buckets gives a good trade-off between normalisation effectiveness and statistical efficiency.
