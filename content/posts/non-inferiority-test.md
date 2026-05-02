---
title: Superiority vs Non-Inferiority Trials in AB Testing
author: Nikita Podlozhniy
summary: >-
  Not every experiment needs to show improvement — sometimes proving you haven't
  made things worse is enough. A practical guide to non-inferiority trial
  design.
date: '2026-03-01'
format:
  hugo-md:
    output-file: non-inferiority-test.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - hypothesis-testing
  - sample-size
categories:
  - posts
---


<!--
<script src="https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js" integrity="sha384-c9c+LnTbwQ3aujuU7ULEPVvgLs+Fn6fJUvIGTsuu1ZcCf11fiEubah0ttpca4ntM sha384-6V1/AdqZRWk1KAlWbKBlGhN7VG4iE/yAZcO6NZPMF8od0vukrvr0tg4qY6NSrItx" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js" integrity="sha384-ZvpUoO/+PpLXR1lu4jmpXWu80pZlYUAfxl5NsBMWOEPSjUn/6Z/hRTt8+pR6L4N2" crossorigin="anonymous" data-relocate-top="true"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
-->


## Background

Most AB tests at HomeBuddy ask: *"is the new variant better?"* --- a **superiority trial**.
But sometimes the question is different: we want to ship a change for cost, performance, or architectural reasons, and we need to verify that it does **not harm** the key metric by more than a tolerable margin.
This is a **non-inferiority trial**.

The distinction is not cosmetic --- it changes the null hypothesis, the test statistic, and critically, the **sample size formula**.
Using a superiority design when you actually want non-inferiority leads to either an underpowered test (too few users to detect the margin) or a miscalibrated one (wrong null hypothesis).

## Problem Definition

Let $\mu_E$ and $\mu_C$ be the true means of the experimental and control groups respectively.
Both trial types test a claim about $\mu_E - \mu_C$, but they differ in what they try to prove.

**Superiority trial**

$$H_0: \mu_E - \mu_C \le 0 \quad \text{vs} \quad H_1: \mu_E - \mu_C > 0$$

Rejecting $H_0$ means the treatment is strictly better than control.

**Non-inferiority trial**

$$H_0: \mu_E - \mu_C \le -\Delta \quad \text{vs} \quad H_1: \mu_E - \mu_C > -\Delta$$

where $\Delta > 0$ is the **non-inferiority margin** --- the largest degradation you are willing to tolerate.
Rejecting $H_0$ means the treatment is not worse than control by more than $\Delta$.

## Side-by-Side Comparison

| Feature | Superiority Trial | Non-Inferiority Trial |
|:-----------------------|:-----------------------|:-----------------------|
| **Goal** | Prove $\mu_E > \mu_C$ | Prove $\mu_E \ge \mu_C - \Delta$ |
| **Null hypothesis** | $\mu_E - \mu_C \le 0$ | $\mu_E - \mu_C \le -\Delta$ |
| **Alternative** | $\mu_E - \mu_C > 0$ | $\mu_E - \mu_C > -\Delta$ |
| **Test type** | Two-sided (typically) | One-sided (always) |
| **Null center** | $0$ | $-\Delta$ |
| **Test statistic** | $Z = \dfrac{(\bar x_E - \bar x_C) - 0}{SE}$ | $Z = \dfrac{(\bar x_E - \bar x_C) - (-\Delta)}{SE}$ |
| **Key parameter** | $\delta$ (MDE) | $\Delta$ (margin) |
| **Sample size** | See below | See below |

## Sample Size Formulas

Let $\sigma^2$ be the common variance, $\alpha$ the significance level, and $1-\beta$ the desired power.

**Superiority** (two-sided, detects effect $\delta$):

$$ n \approx \frac{\left(Z_{1-\alpha/2} + Z_{1-\beta}\right)^2 \cdot 2\sigma^2}{\delta^2} $$

**Non-inferiority** (one-sided, with anticipated true effect $\delta_{\text{ant}}$):

$$ n \approx \frac{\left(Z_{1-\alpha} + Z_{1-\beta}\right)^2 \cdot 2\sigma^2}{(\Delta - \delta_{\text{ant}})^2} $$

Typically $\delta_{\text{ant}} = 0$ is assumed (no true difference), which simplifies the denominator to $\Delta^2$.

**Key insight:** the non-inferiority formula uses the **one-sided** critical value $Z_{1-\alpha}$ (not $Z_{1-\alpha/2}$), and the effect in the denominator is the margin $\Delta$, not the MDE $\delta$.
For a tight margin (small $\Delta$), non-inferiority tests require **more** users than an equivalent superiority test.

<details class="code-fold">
<summary>Sample size calculator</summary>

``` python
from scipy.stats import norm
import numpy as np


def n_superiority(delta, sigma2, alpha=0.05, power=0.8):
    """Per-group sample size for a two-sided superiority test."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(z**2 * 2 * sigma2 / delta**2))


def n_non_inferiority(margin, sigma2, alpha=0.05, power=0.8, anticipated_delta=0.0):
    """Per-group sample size for a one-sided non-inferiority test."""
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return int(np.ceil(z**2 * 2 * sigma2 / (margin - anticipated_delta)**2))


# Example: conversion rate ~ 20%, margin / MDE = 1 pp
p = 0.20
sigma2 = p * (1 - p)
effect = 0.01  # 1 percentage point

n_sup  = n_superiority(effect, sigma2)
n_ni   = n_non_inferiority(effect, sigma2)

print(f'Conversion rate: {p:.0%},  effect / margin: {effect:.1%}')
print(f'Superiority   n per group: {n_sup:>8,}')
print(f'Non-inferiority n per group: {n_ni:>8,}')
```

</details>

    Conversion rate: 20%,  effect / margin: 1.0%
    Superiority   n per group:   25,117
    Non-inferiority n per group:   19,785

<details class="code-fold">
<summary>Comparison across margins and conversion rates</summary>

``` python
import pandas as pd

rows = []
for p in [0.05, 0.10, 0.20]:
    for margin_pct in [0.01, 0.02, 0.05]:
        s2 = p * (1 - p)
        rows.append({
            'Baseline p': f'{p:.0%}',
            'Margin': f'{margin_pct:.0%}',
            'Superiority n': n_superiority(margin_pct, s2),
            'Non-inferiority n': n_non_inferiority(margin_pct, s2),
        })

pd.DataFrame(rows).set_index(['Baseline p', 'Margin'])
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

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Superiority n</th>
<th data-quarto-table-cell-role="th">Non-inferiority n</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Baseline p</th>
<th data-quarto-table-cell-role="th">Margin</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" data-quarto-table-cell-role="th" data-valign="top">5%</td>
<td data-quarto-table-cell-role="th">1%</td>
<td>7457</td>
<td>5874</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2%</td>
<td>1865</td>
<td>1469</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5%</td>
<td>299</td>
<td>235</td>
</tr>
<tr>
<td rowspan="3" data-quarto-table-cell-role="th" data-valign="top">10%</td>
<td data-quarto-table-cell-role="th">1%</td>
<td>14128</td>
<td>11129</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2%</td>
<td>3532</td>
<td>2783</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5%</td>
<td>566</td>
<td>446</td>
</tr>
<tr>
<td rowspan="3" data-quarto-table-cell-role="th" data-valign="top">20%</td>
<td data-quarto-table-cell-role="th">1%</td>
<td>25117</td>
<td>19785</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2%</td>
<td>6280</td>
<td>4947</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5%</td>
<td>1005</td>
<td>792</td>
</tr>
</tbody>
</table>

</div>

## When to Use Each

**Use a superiority trial when:**
- You expect the new variant to improve a metric and want to prove it.
- The cost of a false positive (shipping a harmful change) is high.

**Use a non-inferiority trial when:**
- You are shipping a change for non-metric reasons (performance, cost, tech debt) and need to confirm no regression.
- You are replacing an old component and the new one is expected to be equivalent, not better.
- You want to retire a feature and need to prove its removal doesn't degrade key metrics.

A common mistake is to run a superiority test and interpret a non-significant result as proof of non-inferiority --- it is not.
"We couldn't detect a difference" is not the same as "we proved there is no meaningful difference."
Only a properly designed non-inferiority test with a pre-specified margin gives you that guarantee.

## Conclusion

Superiority and non-inferiority trials answer fundamentally different questions and require different statistical designs.
Choosing the right one before you run the experiment --- not after --- is essential for valid inference.

For practical implementation: non-inferiority tests at HomeBuddy use one-sided Z-tests with the margin as the null center.
The sample size is calculated against the margin $\Delta$, not the MDE, and the significance level uses the one-sided critical value $Z_{1-\alpha}$ instead of $Z_{1-\alpha/2}$.

## References

1.  [Non-inferiority trials --- FDA guidance](https://www.fda.gov/media/78504/download)
2.  [Superiority, equivalence, and non-inferiority trials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3418891/)
3.  [Sample size for non-inferiority studies](https://www.statmethods.net/stats/power.html)
