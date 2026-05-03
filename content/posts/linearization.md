---
title: 'Linearization: Turning Ratio Metrics into Per-User Metrics'
author: Nikita Podlozhniy
summary: >-
  A simple algebraic transformation that converts ratio metrics into independent
  per-user values, making T-tests and sensitivity techniques directly applicable
date: '2026-04-01'
format:
  hugo-md:
    output-file: linearization.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - ratio-metrics
  - variance-reduction
  - statistics
categories:
  - posts
---


## Background

We at HomeBuddy track many **ratio metrics**: average session duration, pages per visit, revenue per order.
These metrics share a common structure — a sum of events in the numerator divided by a count of sessions (or users) in the denominator, where both vary per user.

$$\mathcal{R} = \frac{\sum_{u \in A} X(u)}{\sum_{u \in A} Y(u)}$$

The problem: $\mathcal{R}$ is **not** a simple average of independent observations.
Each user $u$ contributes $Y(u)$ observations to the denominator, creating dependence between rows.
Naively applying a T-test to the raw rows yields invalid p-values.
Collapsing to a per-user mean loses the weighting — a user with 100 sessions gets the same weight as one with 1 session, which distorts the direction of the test.

**Linearization** resolves this with an elegant algebraic trick: it produces a **per-user scalar** that is both unbiased for $\mathcal{R}$ and fully compatible with standard T-tests and sensitivity methods.

## Four Approaches to Ratio Metrics

| Approach | Description | Problem |
|:-----------------------|:-----------------------|:-----------------------|
| **Naive per-user average** | Mean of per-user means | Distorts weighting; direction of test may flip |
| **Bootstrap** | Resample the ratio statistic | Computationally expensive at scale |
| **Delta Method** | Asymptotic variance for X/Y | Requires 3 aggregations per user; see [dedicated post](../delta-method) |
| **Linearization** | Per-user scalar via Taylor expansion | Simple; enables all per-user techniques |

## Theory

### Taylor Expansion

Expand $\mathcal{R}(X, Y)$ around the control group means $(\mu_X, \mu_Y)$ to first order:

$$\mathcal{R}(X, Y) \approx \frac{\mu_X}{\mu_Y} + \frac{1}{\mu_Y}(X - \mu_X) - \frac{\mu_X}{\mu_Y^2}(Y - \mu_Y)$$

$$= \frac{1}{\mu_Y}\left(X - \frac{\mu_X}{\mu_Y}\cdot Y\right) + const$$

Denoting $\alpha = \mu_X / \mu_Y = \mathcal{R}_{control}$ (the control group ratio), the **linearized metric** per user is:

$$L(u) = X(u) - \alpha \cdot Y(u)$$

The per-user values $\{L(u)\}$ are independent and identically distributed — T-test ready.

### Why It Works

The group-level mean of $L$ satisfies:

$$\overline{L_A} = \overline{X_A} - \alpha \cdot \overline{Y_A} = \overline{Y_A}\left(\mathcal{R}_A - \alpha\right)$$

Since $\overline{Y_A} > 0$, the sign of $\overline{L_A}$ equals the sign of $(\mathcal{R}_A - \alpha)$.
This means the T-test on $L$ **rejects whenever $\mathcal{R}_{test} \neq \mathcal{R}_{control}$** — exactly what we want.

The parameter $\alpha$ is estimated from the control group:

$$\hat\alpha = \frac{\sum_{u \in control} X(u)}{\sum_{u \in control} Y(u)}$$

## Implementation

<details class="code-fold">
<summary>Linearization function</summary>

``` python
def linearization(control: list[list], test: list[list]) -> tuple[list, list]:
    """
    Convert ratio-metric observations into linearized per-user scalars.

    Parameters
    ----------
    control, test : list of lists, one inner list per user with all their observations

    Returns
    -------
    (linearized_control, linearized_test) ” one scalar per user, T-test compatible
    """
    total_x = sum(sum(row) for row in control)
    total_y = sum(len(row) for row in control)
    alpha = total_x / total_y  # control ratio estimate

    linearized_control = [sum(row) - len(row) * alpha for row in control]
    linearized_test    = [sum(row) - len(row) * alpha for row in test]
    return linearized_control, linearized_test
```

</details>

## Synthetic Data

We simulate an AB test where each user has a persistent quality effect that correlates all their sessions — the key property that makes row-level tests invalid for ratio metrics.
The test group gets a 5% uplift on the ratio metric for the single-run p-value comparison.

<details class="code-fold">
<summary>Data generation</summary>

``` python
def generate_samples(n_users, n_samples, seed=0, effect=0.0):
    np.random.seed(seed)

    # User-level quality — each user has a fixed base that correlates all their sessions
    user_quality = np.random.lognormal(0, 1, n_users)

    def encoder(x):
        uid = hashlib.md5(str(x).encode()).hexdigest()
        # deterministic group split via MD5 (avoids Python hash randomisation)
        test_flg = int(uid, 16) % 2
        return (uid, 'test' if test_flg else 'control', float(user_quality[x]))

    rows = list(map(encoder, np.random.randint(0, n_users, 2 * n_samples)))
    df = pd.DataFrame(rows, columns=['user_id', 'group', 'user_quality'])

    # Row metric = user quality * session noise (creates within-user correlation)
    row_noise = sts.lognorm.rvs(0.5, size=2 * n_samples)
    metric = df['user_quality'].values * row_noise * 100

    is_test = (df['group'] == 'test').values
    metric[is_test] *= (1 + effect)
    return df[['user_id', 'group']].assign(metric=metric)


df = generate_samples(100, 10000, effect=0.05)

# Build per-user observation lists
grouped = df.groupby(['group', 'user_id'])['metric'].apply(list)
control_obs = [grouped['control'][uid] for uid in grouped['control'].index]
test_obs    = [grouped['test'][uid]    for uid in grouped['test'].index]

print(f'Control users: {len(control_obs)}, test users: {len(test_obs)}')
print(f'Avg obs per control user: {np.mean([len(r) for r in control_obs]):.1f}')
```

</details>

    Control users: 48, test users: 52
    Avg obs per control user: 200.2

<details class="code-fold">
<summary>Apply linearization and compare approaches</summary>

``` python
lin_control, lin_test = linearization(control_obs, test_obs)

# Naive: row-level T-test (incorrect)
ctrl_rows = df[df.group == 'control'].metric.values
test_rows = df[df.group == 'test'].metric.values
_, p_naive = sts.ttest_ind(test_rows, ctrl_rows)

# Per-user average T-test (biased weighting)
ctrl_avg = [np.mean(r) for r in control_obs]
test_avg = [np.mean(r) for r in test_obs]
_, p_avg = sts.ttest_ind(test_avg, ctrl_avg)

# Linearization T-test (correct)
_, p_lin = sts.ttest_ind(lin_test, lin_control)

print(f'Row-level T-test p-value:      {p_naive:.4f}')
print(f'Per-user average p-value:      {p_avg:.4f}')
print(f'Linearization p-value:         {p_lin:.4f}')
```

</details>

    Row-level T-test p-value:      0.0000
    Per-user average p-value:      0.0715
    Linearization p-value:         0.0635

## Correctness and Power

A simulation over 500 AA (no-effect) trials verifies that the linearized T-test holds the FPR at $\alpha = 5\%$, while the row-level approach inflates it to ~88% due to within-user correlation.
A subsequent AB test with a 50% effect checks true power once the FPR is controlled.

<details class="code-fold">
<summary>AA / AB simulation</summary>

``` python
def simulate(effect=0.0, n_tests=500, alpha=0.05):
    row_fp, avg_fp, lin_fp = 0, 0, 0
    for i in range(n_tests):
        d = generate_samples(100, 10000, seed=i, effect=effect)
        g = d.groupby(['group', 'user_id'])['metric'].apply(list)
        ctrl = [g['control'][u] for u in g['control'].index]
        tst  = [g['test'][u]    for u in g['test'].index]

        ctrl_rows = d[d.group == 'control'].metric.values
        test_rows = d[d.group == 'test'].metric.values

        _, p1 = sts.ttest_ind(test_rows, ctrl_rows)
        _, p2 = sts.ttest_ind([np.mean(r) for r in tst], [np.mean(r) for r in ctrl])
        lc, lt = linearization(ctrl, tst)
        _, p3 = sts.ttest_ind(lt, lc)

        if p1 < alpha: row_fp += 1
        if p2 < alpha: avg_fp += 1
        if p3 < alpha: lin_fp += 1

    def ci(n): 
        lo, hi = proportion_confint(n, n_tests, alpha=0.05, method='wilson')
        return f'{n/n_tests:.3f} [{lo:.3f},{hi:.3f}]'

    label = 'AA FPR' if effect == 0 else f'AB power ({effect:.0%} effect)'
    print(f'{label}')
    print(f'  Row-level:      {ci(row_fp)}')
    print(f'  Per-user avg:   {ci(avg_fp)}')
    print(f'  Linearization:  {ci(lin_fp)}')
    print()


simulate(effect=0.0)
simulate(effect=0.50)
```

</details>

    AA FPR
      Row-level:      0.884 [0.853,0.909]
      Per-user avg:   0.036 [0.023,0.056]
      Linearization:  0.038 [0.024,0.059]

    AB power (50% effect)
      Row-level:      0.972 [0.954,0.983]
      Per-user avg:   0.370 [0.329,0.413]
      Linearization:  0.378 [0.337,0.421]

## Linearization vs Delta Method

Both linearization and the [Delta Method](../delta-method) address ratio metrics, but they differ in approach and output:

|  | Linearization | Delta Method |
|:-----------------------|:-----------------------|:-----------------------|
| **Output** | Per-user scalar (T-test ready) | Asymptotic confidence interval |
| **Variance reduction** | Yes — enables CUPED on the linearized metric | Not directly |
| **Bias** | Slight bias: α estimated from control only | Asymptotically unbiased |
| **Enables bucketing/CUPED** | Yes | No |
| **SQL friendliness** | Very high | Very high |

The key advantage of linearization is that once you have per-user scalars, you can apply **any** per-user technique: CUPED, bucketing, stratification.
The Delta Method gives you a confidence interval directly but stops there.

## Conclusion

Linearization is a lightweight transformation that unlocks the full toolkit of per-user AB methods for ratio metrics.
The formula is a single line: `L(u) = X(u) - alpha * Y(u)` where `alpha` is the control group ratio.

Practical notes:
- Estimate $\alpha$ from the **control group only** — using test data would introduce a circular dependency.
- The linearized metric has mean $\approx 0$ in the control group by construction — this is expected and correct.
- For further variance reduction, apply [CUPED](../cuped) or [bucketing](../bucketing) to the linearized values.

## References

1.  Deng, A. et al. (2018). [Applying the Delta Method in Metric Analytics](https://arxiv.org/pdf/1803.06336.pdf)
2.  [Video: Sensitivity improvement for ratio metrics](https://www.youtube.com/watch?v=z8CqaOQgYcI&t=1204s) — includes the square-root reweighting trick
3.  [Video: Advantages of linearization vs reweighting](https://www.youtube.com/watch?v=HpinAY5QfCo)
4.  Avito variance reduction series — [Part 1](https://habr.com/ru/companies/avito/articles/571094/), [Part 2](https://habr.com/ru/companies/avito/articles/571096/)
