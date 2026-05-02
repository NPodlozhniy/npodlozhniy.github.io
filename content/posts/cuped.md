---
title: 'CUPED: Reducing Metric Variance with Pre-Experiment Data'
author: Nikita Podlozhniy
summary: >-
  How CUPED uses pre-experiment covariates to cut metric variance and boost AB
  test sensitivity — with a rigorous proof of what works and what silently
  breaks
date: '2026-03-15'
format:
  hugo-md:
    output-file: cuped.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: false
  cache: true
tags:
  - ab-testing
  - cuped
  - variance-reduction
  - statistics
categories:
  - posts
---


## Background

We at HomeBuddy run AB tests on conversion metrics and engagement signals whose natural variance is high.
High variance means we need more users to detect the same effect --- or equivalently, we might miss a real improvement because the noise drowns it out.

**CUPED** (Controlled-experiment Using Pre-Experiment Data) is a variance-reduction technique introduced by Microsoft Research that exploits the correlation between a user's pre-experiment behavior and their in-experiment metric.
By subtracting a scaled version of the pre-experiment covariate from the metric, CUPED can cut variance by 50--90%, effectively doubling or tripling test sensitivity with no additional traffic.

### Prerequisites

Python 3.11.4 with NumPy, SciPy, pandas, and statsmodels.

## Theory

Let $Y$ be the in-experiment metric and $X$ a covariate unaffected by the treatment (e.g. the same metric measured before the experiment began).
Define the CUPED-adjusted metric:

$$ Y_{\text{CUPED}} = Y - \theta(X - \mathbb{E}X) $$

Computing the moments:

$$ \mathbb{E}[Y_{\text{CUPED}}] = \mathbb{E}[Y] - \theta \underbrace{\mathbb{E}[X - \mathbb{E}X]}_{=\,0} = \mathbb{E}[Y] $$

$$ \mathbb{D}[Y_{\text{CUPED}}] = \mathbb{D}[Y] + \theta^2 \mathbb{D}[X] - 2\theta\,\text{cov}(Y, X) $$

The expectation is preserved. The variance can be reduced by choosing $\theta$ to minimise $\mathbb{D}[Y_{\text{CUPED}}]$:

$$ \frac{\partial}{\partial \theta}\mathbb{D}[Y_{\text{CUPED}}] = 2\theta\mathbb{D}[X] - 2\,\text{cov}(Y, X) = 0 \implies \theta^* = \frac{\text{cov}(Y, X)}{\mathbb{D}[X]} $$

Substituting back:

$$ \mathbb{D}[Y_{\text{CUPED}}] = \mathbb{D}[Y]\left(1 - \rho^2_{XY}\right) $$

where $\rho_{XY}$ is the Pearson correlation between $X$ and $Y$.
If $\rho = 0.9$, variance drops by 81%. If $\rho = 0$, CUPED provides no benefit (and should not be applied).

**Key constraints:**

1.  The covariate $X$ must not be affected by the treatment --- pre-experiment data is ideal.
2.  $\theta$ must be estimated **on the combined sample** (test + control), not on each group separately.
3.  For a two-sample test, $\mathbb{E}X$ must be the pooled mean --- not the per-group mean.

<details class="code-fold">
<summary>Synthetic data generation with correlated past/future metrics</summary>

``` python
def lognorm():
    return sts.lognorm(s=1, loc=100)


def generate_samples(distribution, n_users=100, n_days=1, seed=0):
    """
    Generate a dataset with historical (past) and experimental (future) periods.
    Past and future metrics are correlated: future = past + small noise.
    """
    np.random.seed(seed)

    def encoder(x):
        uid = hashlib.md5(str(x).encode()).hexdigest()
        test_flg = hash(str(x).encode()) % 2
        return (uid, 'test' if test_flg else 'control')

    df = pd.DataFrame(
        list(map(
            encoder,
            np.array([[u] * (2 * n_days) for u in range(2 * n_users)]).ravel()
        )),
        columns=['user_id', 'group'],
    )
    df['date'] = pd.to_datetime(
        [datetime.date.today() - datetime.timedelta(days=x) for x in range(2 * n_days)] * 2 * n_users
    )
    df['history'] = np.where(
        df['date'] > pd.Timestamp(datetime.date.today() - datetime.timedelta(days=n_days)),
        'future', 'past'
    )
    future_metric = distribution.rvs(size=n_days * n_users * 2)
    past_metric = future_metric + sts.norm.rvs(loc=0, scale=1, size=len(future_metric))

    df = df.sort_values(by=['date', 'user_id'], ascending=True)
    df['metric'] = np.hstack((past_metric, future_metric))
    return df.reset_index(drop=True)


rv = lognorm()
EV = rv.mean()  # true population mean ≈ 101.6
df = generate_samples(rv)
user_level = df.groupby(['group', 'history', 'user_id'])[['metric']].mean().reset_index()
print(f'Shape: {df.shape}, EV ≈ {EV:.2f}')
user_level.sample(3)
```

</details>

## T-Test Utilities

Two test procedures are needed: a **one-sample T-test** (used when comparing a single group against a known mean) and **Welch's T-test** (for two independent groups with potentially unequal variances).

<details class="code-fold">
<summary>One-sample and Welch T-test implementations</summary>

``` python
def one_samp_t_test(X, mu0, alpha=0.05):
    mu = np.mean(X) - mu0
    sigma = np.sqrt(np.var(X, ddof=1) / len(X))
    t = mu / sigma
    T = sts.t(df=len(X) - 1)
    return {'pvalue': 2 * min(T.sf(t), T.cdf(t))}


def welch_t_test(X, Y, alpha=0.05):
    vx = np.var(X, ddof=1) / len(X)
    vy = np.var(Y, ddof=1) / len(Y)
    mu = np.mean(X) - np.mean(Y)
    sigma = np.sqrt(vx + vy)
    t = mu / sigma
    nu = int((vx + vy)**2 / (vx**2 / (len(X) - 1) + vy**2 / (len(Y) - 1)))
    T = sts.t(df=nu)
    return {'pvalue': 2 * min(T.sf(t), T.cdf(t))}
```

</details>

## Two-Sample CUPED

In a standard AB test with test group $T$ and control group $C$, define:

$$ T_{\text{CUPED}} = T - \theta A, \quad C_{\text{CUPED}} = C - \theta B $$

where $A$ and $B$ are pre-experiment covariates for test and control respectively, with $\mathbb{E}A = \mathbb{E}B$ (they're drawn from the same pre-experiment distribution).

The optimal $\theta$ for variance minimisation of the difference $T_{\text{CUPED}} - C_{\text{CUPED}}$:

$$ \theta^* = \frac{\text{cov}(T, A) + \text{cov}(C, B)}{\mathbb{D}[A] + \mathbb{D}[B]} $$

The resulting variance reduction:

$$ \mathbb{D}[T_{\text{CUPED}} - C_{\text{CUPED}}] = \left(1 - \rho^2\right)\mathbb{D}[T - C], \quad \rho = \text{corr}(T - C,\; A - B) $$

<details class="code-fold">
<summary>Two-sample CUPED implementation</summary>

``` python
def two_samples_cuped(test_target, control_target, test_cov, control_cov):
    theta = (
        np.cov(test_target, test_cov)[0, 1] + np.cov(control_target, control_cov)[0, 1]
    ) / (np.var(test_cov) + np.var(control_cov))
    return test_target - theta * test_cov, control_target - theta * control_cov


def apply_two_samples_cuped(user_df, cuped, ids='user_id', date='history', metric='metric'):
    data = user_df.sort_values(by=[date, ids], ascending=True)
    ft = data[(data.group == 'test')    & (data.history == 'future')][metric].values
    fc = data[(data.group == 'control') & (data.history == 'future')][metric].values
    pt = data[(data.group == 'test')    & (data.history == 'past')][metric].values
    pc = data[(data.group == 'control') & (data.history == 'past')][metric].values

    data = data.copy()
    data['cuped_metric'] = np.nan
    tc, cc = cuped(ft, fc, pt, pc)
    data.loc[(data.group == 'test')    & (data.history == 'future'), 'cuped_metric'] = tc
    data.loc[(data.group == 'control') & (data.history == 'future'), 'cuped_metric'] = cc
    return data.dropna().reset_index(drop=True)


df_cuped = apply_two_samples_cuped(user_level, two_samples_cuped)

orig_var  = df_cuped['metric'].var()
cuped_var = df_cuped['cuped_metric'].var()
print(f'Original variance:  {orig_var:.2f}')
print(f'CUPED variance:     {cuped_var:.2f}')
print(f'Reduction factor:   {orig_var / cuped_var:.1f}×')
```

</details>

## Correctness: AA Test

Before measuring power gains, we verify the criterion is properly calibrated: in an AA test (no real effect), the false-positive rate should stay at $\alpha$.

We also check a common mistake --- subtracting the **per-group** mean of the covariate instead of the **pooled** mean. This breaks the equal-expectation assumption and inflates FPR dramatically.

<details class="code-fold">
<summary>AA test: correct vs incorrect demeaning</summary>

``` python
def two_samples_cuped_demeaned(test_target, control_target, test_cov, control_cov):
    """Correct: subtract the POOLED covariate mean."""
    theta = (
        np.cov(test_target, test_cov)[0, 1] + np.cov(control_target, control_cov)[0, 1]
    ) / (np.var(test_cov) + np.var(control_cov))
    pooled_mean = np.hstack((test_cov, control_cov)).mean()
    return (test_target - theta * (test_cov - pooled_mean),
            control_target - theta * (control_cov - pooled_mean))


def two_samples_cuped_incorrect(test_target, control_target, test_cov, control_cov):
    """Incorrect: subtract per-group mean — inflates FPR."""
    theta = (
        np.cov(test_target, test_cov)[0, 1] + np.cov(control_target, control_cov)[0, 1]
    ) / (np.var(test_cov) + np.var(control_cov))
    return (test_target - theta * (test_cov - test_cov.mean()),
            control_target - theta * (control_cov - control_cov.mean()))


def simulate(procedure, n_tests=500, alpha=0.05, mode='AA'):
    n_err = 0
    for i in range(n_tests):
        d = generate_samples(lognorm(), seed=8 * i)
        ul = d.groupby(['group', 'history', 'user_id'])[['metric']].mean().reset_index()
        if mode == 'AB':
            ul['metric'] = ul.apply(
                lambda r: 1.01 * r.metric if r.group == 'test' and r.history == 'future' else r.metric,
                axis=1)
        dc = ul.pipe(apply_two_samples_cuped, cuped=procedure)
        p = welch_t_test(
            dc.loc[dc.group == 'test',    'cuped_metric'],
            dc.loc[dc.group == 'control', 'cuped_metric'],
        )['pvalue']
        if p < alpha:
            n_err += 1
    lo, hi = proportion_confint(n_err, n_tests, alpha=0.05, method='wilson')
    return f'{n_err / n_tests:.3f}  95% CI [{lo:.3f}, {hi:.3f}]'


print('AA FPR — correct (pooled mean):  ', simulate(two_samples_cuped_demeaned))
print('AA FPR — correct (no demeaning): ', simulate(two_samples_cuped))
print('AA FPR — incorrect (per-group):  ', simulate(two_samples_cuped_incorrect))
```

</details>

The incorrect per-group demeaning blows up the FPR to ~40% --- a catastrophic failure that would be hard to detect without a simulation like this.
Both the no-demeaning and pooled-demeaning variants maintain FPR ≈ 5%.

## Power Improvement

Now let's measure the power gain under a 1% true effect.

<details class="code-fold">
<summary>Power comparison: no CUPED vs CUPED</summary>

``` python
def no_cuped(test_target, control_target, test_cov, control_cov):
    return test_target, control_target


print('Power (1% effect) — no CUPED: ', simulate(no_cuped, mode='AB'))
print('Power (1% effect) — CUPED:    ', simulate(two_samples_cuped, mode='AB'))
```

</details>

## Conclusion

CUPED is one of the most cost-effective variance-reduction techniques available:

- **No extra traffic** --- it uses data you already have (pre-experiment history).
- **Large gains** --- if pre/post correlation is high ($\rho \approx 0.9$), variance drops ~81%.
- **Unbiased** --- the expected value of the metric is preserved exactly.
- **Easy to implement** --- requires only three statistics per group: covariance, variance of covariate, and means.

**Critical implementation detail:** $\theta$ must be estimated from both groups combined, and the covariate mean subtracted must be the **pooled** pre-experiment mean --- not per-group. Subtracting per-group means silently inflates the false-positive rate from 5% to ~40%.

**Alternatives** when pre-experiment data is unavailable or CUPED gains are small:

- **Stratification** --- randomise within strata rather than fully at random, reducing imbalance.
- **Prediction subtraction** --- generalises CUPED from a single covariate to a full regression model; useful when many pre-experiment features are available.

## References

1.  Deng, A. et al. (2013). [Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) --- the original CUPED paper.
2.  [Booking.com: How CUPED increases power of online experiments](https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-cuped-995d186fff1d)
3.  [Avito: Variance reduction methods --- Part 1](https://habr.com/ru/companies/avito/articles/571094/), [Part 2](https://habr.com/ru/companies/avito/articles/571096/)
4.  [Variance reduction techniques --- conference talk](https://www.youtube.com/watch?v=pZpUM08mv-E&t=1692s)
