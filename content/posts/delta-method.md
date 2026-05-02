---
title: Delta Method for Ratio Metrics in AB Testing
author: Nikita Podlozhniy
summary: >-
  Correct variance estimation for ratio metrics using the Delta Method — and why
  naive approaches quietly break your confidence intervals
date: '2025-12-15'
format:
  hugo-md:
    output-file: delta-method.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - statistics
  - ratio-metrics
  - variance-reduction
categories:
  - posts
---


<!--
<script src="https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js" integrity="sha384-c9c+LnTbwQ3aujuU7ULEPVvgLs+Fn6fJUvIGTsuu1ZcCf11fiEubah0ttpca4ntM sha384-6V1/AdqZRWk1KAlWbKBlGhN7VG4iE/yAZcO6NZPMF8od0vukrvr0tg4qY6NSrItx" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js" integrity="sha384-ZvpUoO/+PpLXR1lu4jmpXWu80pZlYUAfxl5NsBMWOEPSjUn/6Z/hRTt8+pR6L4N2" crossorigin="anonymous" data-relocate-top="true"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
-->


## Background

We at HomeBuddy run dozens of AB tests to improve the customer journey across our onboarding funnel.
Many of our key metrics are **ratio metrics**: average revenue per session, orders per visit, or any per-user aggregate where both the numerator and denominator vary from user to user.
Treating such metrics naively --- as if each row were an independent observation --- leads to underestimated variance, artificially narrow confidence intervals, and inflated false-positive rates.

The **Delta Method** provides the correct asymptotic variance for any continuously differentiable function of random variables, and applying it to ratio metrics is surprisingly straightforward.

### Prerequisites

The reader is expected to be comfortable with Python and standard data-science libraries.
The notebook was written on Python 3.11.4; below is the minimal set of packages required.

## Problem Definition

Consider a metric defined as the ratio $R = X / Y$, where for each user $i$:

- $X_i$ is the total value of some event (e.g. revenue)
- $Y_i$ is the number of sessions

Three common --- but not equally correct --- ways to estimate the mean and its confidence interval for such a metric are:

**A. Row-level average** --- treat every row as an independent sample. Underestimates variance because multiple rows from the same user are correlated.

**B. User-level naive average** --- aggregate to one value per user, then use the sample variance of those averages. Still wrong: the average of per-user averages is not the same as the ratio of sums, and its variance formula does not account for the denominator's variability.

**C. Delta Method** --- compute the exact asymptotic variance for $X/Y$ by propagating the variance of both $X$ and $Y$ and their covariance.

## Theory

Let $\hat\theta_n$ be a sequence of random variables converging in distribution:

$$ \hat{\theta}_n \xrightarrow{dist} \mathbb{N}(\theta_0, V/n) $$

For a continuously differentiable function $g: \mathbb{R} \to \mathbb{R}$, the **Delta Method** states:

$$ g(\hat\theta_n) \xrightarrow{dist} \mathbb{N}\!\left(g(\theta_0),\; \left(\frac{dg}{d\theta}\right)^2_{\!\theta_0} \cdot \frac{V}{n}\right) $$

**Proof sketch:** By the mean value theorem, $\exists\, \bar\theta$ between $\hat\theta_n$ and $\theta_0$ such that

$$ g(\hat\theta_n) - g(\theta_0) = \left(\frac{dg}{d\theta}\right)_{\!\bar\theta} \cdot (\hat\theta_n - \theta_0) $$

Since $\hat\theta_n \xrightarrow{\mathbb{P}} \theta_0$, we have $\bar\theta \xrightarrow{\mathbb{P}} \theta_0$. By the continuous mapping theorem and Slutsky's theorem the product converges in distribution, giving

$$ g(\hat\theta_n) - g(\theta_0) \xrightarrow{dist} \mathbb{N}\!\left(0,\; \left(\frac{dg}{d\theta}\right)^2_{\!\theta_0} \cdot \frac{V}{n}\right) $$

### Application to ratio metrics

For $R = X/Y$ the asymptotic variance of the sample ratio $\bar R = \bar X / \bar Y$ is:

$$ \mathbb{D}\bar{R} = \left(\frac{\bar{X}}{\bar{Y}}\right)^2 \cdot \left(\frac{\mathbb{D}\bar{X}}{\bar{X}^2} + \frac{\mathbb{D}\bar{Y}}{\bar{Y}^2} - 2\frac{\mathrm{cov}(\bar{X}, \bar{Y})}{\bar{X} \cdot \bar{Y}} \right) $$

where

$$ \mathrm{cov}(\bar{X}, \bar{Y}) = \frac{\sum_{i=1}^N(X_i - \bar{X})(Y_i - \bar{Y})}{N(N - 1)} $$

This formula requires only three aggregations per user --- $\sum X_i$, $\sum Y_i$, $\sum X_i Y_i$ --- making it trivially parallelisable in any SQL or distributed compute environment.

<details class="code-fold">
<summary>RatioXY class implementation</summary>

``` python
import numpy as np
import scipy.stats as sts


class RatioXY:
    """Confidence interval for a ratio metric X/Y using the Delta Method."""

    def __init__(self, X, Y):
        self.n = len(X)
        self.X_mean = np.mean(X)
        self.Y_mean = np.mean(Y)
        self.X_var = np.var(X, ddof=1)
        self.Y_var = np.var(Y, ddof=1)
        self.XY_cov = np.cov(X, Y, bias=False)[0][1]

    def ratio_variance(self):
        return self.X_mean**2 / self.Y_mean**2 * (
            self.X_var / self.X_mean**2
            + self.Y_var / self.Y_mean**2
            - 2 * self.XY_cov / (self.X_mean * self.Y_mean)
        )

    def bias_correction(self):
        return 1 / self.n * (
            self.Y_var * self.X_mean / self.Y_mean**3
            - self.XY_cov / self.Y_mean**2
        )

    def point_estimate(self, bias=False):
        bc = self.bias_correction() if bias else 0
        return self.X_mean / self.Y_mean + bc

    def deltaci(self, alpha=0.05, bias=False, two_sided=True):
        pest = self.point_estimate(bias=bias)
        vest = self.ratio_variance()
        z = sts.norm.ppf(1 - alpha / 2) if two_sided else sts.norm.ppf(1 - alpha)
        return pest + np.array([-1, 1]) * z * np.sqrt(vest / self.n)
```

</details>

## Synthetic Data

We simulate a dataset where each user has a variable number of sessions and a lognormally distributed metric per session.
The lognormal distribution mimics real-world revenue data: heavy right tail, all positive values.

<details class="code-fold">
<summary>Synthetic data generation</summary>

``` python
import pandas as pd
import hashlib


def generate_samples(n_users, n_samples, seed=2023):
    np.random.seed(seed)

    def encoder(x):
        uid = hashlib.md5(str(x).encode()).hexdigest()
        test_flg = hash(str(x).encode()) % 2
        return (uid, 'test' if test_flg else 'control')

    df = pd.DataFrame(
        list(map(encoder, np.random.randint(0, n_users, 2 * n_samples))),
        columns=['user_id', 'group'],
    )
    return df.assign(metric=sts.lognorm.rvs(3, loc=100, size=2 * n_samples))


df = generate_samples(1000, 10000)
print(f'Rows: {df.shape[0]},  unique users: {df.user_id.nunique()}')
df.sample(3)
```

</details>

    Rows: 20000,  unique users: 1000

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
| 8225  | 274ad4786c3abca69fa097b85867d9a4 | test    | 108.472413 |
| 775   | bea5955b308361a1b07bc55042e25e54 | control | 114.576327 |
| 14845 | 389bc7bb1e1c2a5e7e147703232a88f6 | test    | 106.192022 |

</div>

## Variance Estimation: Three Approaches

Each approach estimates a 95% confidence interval for the **control group mean**.
The true population mean is approximately 180.5 (lognormal with `s=3, loc=100`).
We compare how wide --- and how correct --- each CI is.

<details class="code-fold">
<summary>A. Row-level (incorrect)</summary>

``` python
row_level = df.groupby('group')['metric'].agg(['mean', 'var'])
report(
    'A. Row-level',
    row_level.loc['control', 'mean'],
    row_level.loc['control', 'var'],
    df.shape[0],
)
```

</details>

    A. Row-level:  mean=193.92,  95% CI = [164.56, 223.27],  width=58.71

The row-level approach treats 20 000 rows as independent observations. The sample mean is unbiased, but the variance is severely underestimated because many rows belong to the same user.

<details class="code-fold">
<summary>B. User-level naive average (incorrect)</summary>

``` python
temp = (
    df.groupby(['group', 'user_id'])['metric']
    .agg(['sum', 'count'])
    .reset_index(level=1)
)
temp['avg'] = temp['sum'] / temp['count']
user_level = temp.groupby('group')['avg'].agg(['mean', 'var'])

report(
    'B. User avg',
    user_level.loc['control', 'mean'],
    user_level.loc['control', 'var'],
    temp.shape[0],
)
```

</details>

    B. User avg:  mean=196.76,  95% CI = [167.11, 226.41],  width=59.30

Averaging per-user averages gives neither the correct point estimate (the mean of averages $\neq$ the ratio of sums) nor the correct variance, because the denominator $Y_i$ varies across users.

<details class="code-fold">
<summary>C. Delta Method (correct)</summary>

``` python
X_control = temp.loc['control', 'sum'].values
Y_control = temp.loc['control', 'count'].values

ctrl = RatioXY(X_control, Y_control)
lo, hi = ctrl.deltaci(bias=False)
print(
    f'C. Delta Method:  mean={ctrl.point_estimate():.2f},  '
    f'95% CI = [{lo:.2f}, {hi:.2f}],  width={hi-lo:.2f}'
)
```

</details>

    C. Delta Method:  mean=193.92,  95% CI = [152.50, 235.34],  width=82.84

The Delta Method produces a **wider** confidence interval --- not because it is less precise, but because it correctly captures the additional uncertainty from the denominator. The narrower intervals from approaches A and B are over-confident and would inflate your false-positive rate in an AB test.

<details class="code-fold">
<summary>Confidence interval comparison chart</summary>

``` python
import plotly.graph_objs as go
from pathlib import Path


def find_git_repo(path=None):
    source = Path(path) if path else Path.cwd()
    for p in source.parents:
        if (p / '.git').is_dir():
            return p


row_mean = row_level.loc['control', 'mean']
row_var = row_level.loc['control', 'var']
row_lo, row_hi = row_mean - z * np.sqrt(row_var / df.shape[0]), row_mean + z * np.sqrt(row_var / df.shape[0])

usr_mean = user_level.loc['control', 'mean']
usr_var = user_level.loc['control', 'var']
usr_lo, usr_hi = usr_mean - z * np.sqrt(usr_var / temp.shape[0]), usr_mean + z * np.sqrt(usr_var / temp.shape[0])

dlt_lo, dlt_hi = ctrl.deltaci()
dlt_mean = ctrl.point_estimate()

labels = ['A. Row-Level', 'B. User Average', 'C. Delta Method']
means  = [row_mean, usr_mean, dlt_mean]
lows   = [row_lo,  usr_lo,  dlt_lo]
highs  = [row_hi,  usr_hi,  dlt_hi]

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=labels, y=means,
    error_y=dict(
        type='data', symmetric=False,
        array=[hi - m for m, hi in zip(means, highs)],
        arrayminus=[m - lo for m, lo in zip(means, lows)],
    ),
    mode='markers',
    marker=dict(size=12),
    name='95% CI',
))
figure.update_layout(
    title={'text': '95% Confidence Intervals — Three Approaches', 'x': 0.5},
    yaxis_title='Estimated Mean',
    template='plotly_dark',
)
figure.show()
```

</details>
{{< plotly obj=delta-method-ci >}}

## Conclusion

Whenever your AB test metric is a ratio --- revenue per visit, events per user, average order value --- the Delta Method is the correct tool for variance estimation.
It requires only three per-user aggregations ($\sum X_i$, $\sum Y_i$, $\sum X_i Y_i$) and is trivially expressible in SQL, making it production-ready with no additional infrastructure.

The naive alternatives consistently underestimate variance, leading to confidence intervals that are too narrow and hypothesis tests that reject the null too often.

## References

1.  Deng, A. et al. (2018). [Applying the Delta Method in Metric Analytics](https://arxiv.org/pdf/1803.06336.pdf) --- the original paper this implementation is based on.
2.  [Medium: Applying Delta Method for A/B Tests Analysis](https://medium.com/@ahmadnuraziz3/applying-delta-method-for-a-b-tests-analysis-8b1d13411c22)
3.  [StackExchange: Example of using the Delta Method](https://math.stackexchange.com/questions/1598503/example-of-using-delta-method)
4.  [StatLect: Delta Method --- theory with examples](https://www.statlect.com/asymptotic-theory/delta-method)
5.  Avito variance reduction series --- [Part 1](https://habr.com/ru/companies/avito/articles/571094/), [Part 2](https://habr.com/ru/companies/avito/articles/571096/)
