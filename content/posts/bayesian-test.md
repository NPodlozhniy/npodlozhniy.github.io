---
title: "Bayesian A/B Test is NOT immune to peeking"
summary: "Practical guide: PyMC examples, hierarchical models, loss functions, HDI, and safe stopping rules"
author: "Nikita Podlozhniy"
date: "2025-11-30"
format:
    hugo-md:
        output-file: "bayesian-test.md"
        html-math-method: katex
        code-fold: true
jupyter: python3
execute:
    enabled: false
    cache: true
tags:
    - ab-testing
    - bayesian-inference
    - monte-carlo
    - pymc
    - sequential-testing
categories:
    - posts
---

<!--
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
-->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>


# Bayesian A/B Testing - Practice

This interactive notebook demonstrates a concise, pragmatic approach to Bayesian A/B testing using PyMC and analytic Beta-Binomial formulas.

What you'll find here:

-   A minimal, runnable PyMC example to obtain posterior samples.
-   A hierarchical model example for multiple related tests (shrinkage).
-   Analytic Beta-Binomial updates and closed-form PoS / Expected Loss expressions.
-   A Monte Carlo harness comparing frequentist sequential z-tests and several Bayesian stopping rules.

TL;DR: Posterior summaries like the Probability of Superiority (PoS) are fantastic for interpretation, but if you stare at them until they cross a threshold (peeking), you will break your error guarantees. 🛑 If you care about long-run false positives, use decision-theoretic rules (Expected Loss) or precision-aware metrics (HDI).

## 🥧 Part 1: A Simple Slice of PyMC

Let's kick things off with the basics: two variants, A and B, and a binary outcome (conversion: yes/no). Our goal? Use Markov Chains to sample the success probabilities and figure out if A is actually beating B.

### Step 1: Set up imports

Grab PyMC along with must have DS and BI libraries.

<details>
<summary>Code</summary>

``` python
import pymc as pm
from pymc import Uniform, Bernoulli

from matplotlib import pyplot as plt

import seaborn as sns

import pandas as pd
import numpy as np

from scipy import stats as sts
```

</details>

### Step 2: Simulate observed data

Generate sample data: two variants with known conversion rates. In practice, these would be your real observed counts from the experiment.

``` python
a_default, b_default = 0.06, 0.04
a_count, b_count = 200, 150

rng = np.random.default_rng(seed=42)

a_bernoulli_samples = rng.binomial(n=1, p=a_default, size=a_count)
b_bernoulli_samples = rng.binomial(n=1, p=b_default, size=b_count)

print(
    "Point Estimate"
        f"\n- A: {a_bernoulli_samples.sum() / a_count :.3f}"
        f"\n- B: {b_bernoulli_samples.sum() / b_count :.3f}"
)
```

    Point Estimate
    - A: 0.040
    - B: 0.020

Quick sanity check: plug-in estimates (observed proportions). These will be compared with posterior estimates below.

### Step 3: Define the Bayesian model

We treat success probabilities as independent random variables. Since we don't know much yet, we use a uniform prior (weakly informative - go to for proportions). The observed data (Bernoulli trials) are the likelihood. We also track *deterministic* difference \$= A - B \$ , - because that's what we actually care about!

``` python
with pm.Model() as my_model:

    A_prior = Uniform('A_prior', lower=0, upper=1)
    B_prior = Uniform('B_prior', lower=0, upper=1)

    A_observed = Bernoulli('A_observed', p=A_prior, observed=a_bernoulli_samples)
    B_observed = Bernoulli('B_observed', p=B_prior, observed=b_bernoulli_samples)

    delta = pm.Deterministic("delta", A_prior - B_prior)
```

### Step 4: Sample from the posterior

PyMC unleashes the "No-U-Turn Sampler" by default to draw samples from the joint posterior. The `tune` parameter controls burn-in iterations (discarded); `draws` are the kept samples used for inference.

``` python
with my_model:
    idata = pm.sample(draws=5000, tune=1000, cores=-1)
```

### Step 5: Visualize the posteriors

Plot those posterior distributions of each variant and their difference. The vertical black line shows the true (generating) difference; the red line marks zero (no difference). If the posterior difference doesn't touch zero, you're onto something.

<details>
<summary>Code</summary>

``` python
# --- Setup Dark Mode ---
plt.style.use('dark_background')

# Ensure texts are white (sometimes needed depending on Jupyter setup)
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white"
})

fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

# Neon colors for pop against dark background
colors = ['#00FFFF', '#FF00FF', '#32CD32'] # Cyan, Magenta, Lime

# --- Plot 1: Posterior P(A) ---
ax = axes[0]
data_a = idata.posterior['A_prior'].values.ravel()
ax.hist(data_a, bins=50, density=True, color=colors[0], alpha=0.7, 
        edgecolor='black', linewidth=1.2, label="Posterior $P(A)$")
ax.set_title("Posterior Probability of A", fontsize=14, loc='left', fontweight='bold')
ax.legend(loc='upper right', frameon=False)
ax.set_ylabel("Density")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

# --- Plot 2: Posterior P(B) ---
ax = axes[1]
data_b = idata.posterior['B_prior'].values.ravel()
ax.hist(data_b, bins=50, density=True, color=colors[1], alpha=0.7, 
        edgecolor='black', linewidth=1.2, label="Posterior $P(B)$")
ax.set_title("Posterior Probability of B", fontsize=14, loc='left', fontweight='bold')
ax.legend(loc='upper right', frameon=False)
ax.set_ylabel("Density")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

# --- Plot 3: Difference ---
ax = axes[2]
data_delta = idata.posterior['delta'].values.ravel()
ax.hist(data_delta, bins=50, density=True, color=colors[2], alpha=0.7, 
        edgecolor='black', linewidth=1.2, label="Difference ($A - B$)")
ax.set_title("Difference in Probabilities", fontsize=14, loc='left', fontweight='bold')
ax.set_ylabel("Density")
ax.set_xlabel("Delta Value")

# Vertical Lines
ax.axvline(a_default - b_default, color='white', linestyle=':', linewidth=2, label="Expected Diff")
ax.axvline(0, color='#FF4444', linestyle='-', linewidth=2, label="Zero Difference")

ax.legend(loc='upper right', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

# --- Saving with Transparent Background ---

# The magic argument is transparent=True + bbox_inches='tight' cuts off extra whitespace around the labels
# plt.savefig('dark_bayes_plot_transparent.png', dpi=300, bbox_inches='tight', transparent=True)
# to evaluate probability (idata.posterior['delta'].values > 0).mean()

plt.show()
```

</details>

<img src="BayesianAB_files/figure-markdown_strict/fig-bayesian-plot-1-output-1.png" id="fig-bayesian-plot-1" alt="Figure 1: Posterior distributions for the conversion rates of variants A and B" />

<p style="text-align: center;">
Figure 1. Probability of superiority: 83.2%
</p>

## 🏗️ Part 2: Hierarchical Models (The "Robin Hood" Approach)

When you run many related experiments or compare several variants in the same domain, hierarchical (multilevel) models are a practical way to borrow strength across groups. They reduce variance for small groups (shrinkage) and improve estimation stability. Below is a compact PyMC implementation that models group probabilities as draws from a shared Beta(a, b) prior.

This pattern is useful for dashboarding many A/B results together or for pooling information when sample sizes vary across tests.

### What is hierarchical models

In a hierarchical (or multilevel) model, you assume that the parameters for each group are related and drawn from a common, overarching distribution. This shared distribution is governed by hyper-parameters.

All group-level parameters are drawn from a single, shared distribution defined by hyper-parameters a and b. For example each test's probability is drawn from a shared Beta(a,b), where a and b are themselves parameters to be estimated.

The estimate for any single group is influenced both by its own data and the data from all other groups (via the shared a and b). This is called partial pooling, Say a and b are estimated to best fit all test data, and then every test's probability is pulled slightly toward the overall average defined by a and b.

Estimates for small groups are pulled toward the average (a process called shrinkage), leading to more stable, less extreme estimates

### Which distribution may be used?

Non-Informative Prior for Beta Hyperparameters:
$$p(a,b) \propto (a+b)^{-5/2}$$

Simplification or approximation of the Jeffreys Prior for the hyper-parameters of the Beta distribution, which is used as a conjugate prior for binomial or Bernoulli likelihoods (common in multiple testing models, e.g., estimating the probability of a true null hypothesis)

When $a$ and $b$ are the shape parameters of Beta distribution, the actual Jeffreys Prior is defined by the Fisher Information matrix:

$$p(a,b) \propto \sqrt{\det(\mathbf{I}(a,b))}$$

The determinant of the Fisher Information matrix for the Beta distribution's parameters is a complex function involving the trigamma function ($\psi'$). Specifically, a known simplification used in some computational Bayesian contexts is related to the mean and total effective count of the Beta distribution.

The term $\tau = a+b$ is often interpreted as the total effective sample size (or precision) of the Beta distribution. The exponent $-5/2$ is a specific value that results from one of the approximations designed to make the prior less influential on the posterior, often for the standard deviation or variance of the underlying distribution.

### Example

``` python
# --- 1. Data Definition ---
trials = np.array([842, 854, 862, 821, 839])
successes = np.array([27, 47, 69, 52, 35])

N_GROUPS = len(trials)
# Constraint for Beta parameters (a and b must be > 0)
ALPHA_MIN = 0.01

with pm.Model() as hierarchical_model:

    # --- 2. Hyper-parameter Priors (a and b) ---

    # The Beta shape parameters a and b must be positive.
    # We define them with a minimally informative uniform prior.
    a = pm.Uniform("a", lower=ALPHA_MIN, upper=100)
    b = pm.Uniform("b", lower=ALPHA_MIN, upper=100)

    # --- 3. Custom Precision Prior (pm.Potential) ---
    # The original prior was: log p(a, b) = log((a+b)^-2.5) = -2.5 * log(a + b)
    PRIOR_EXPONENT = -2.5

    # Use pm.Potential to add the custom log-prior term to the model's log(P)
    log_precision_prior = PRIOR_EXPONENT * np.log(a + b)
    pm.Potential("beta_precision_potential", log_precision_prior)

    # --- 4. Group-level Prior (occurrences) ---
    # 'occurrences' is the probability for each group
    # drawn from a Beta distribution defined by the hyper-parameters a and b.
    occurrences = pm.Beta("occurrences", alpha=a, beta=b, shape=N_GROUPS)

    # --- 5. Likelihood (l_obs) ---
    # The observed successes follow a Binomial distribution.
    likelihood = pm.Binomial("likelihood", n=trials, p=occurrences, observed=successes)

    # --- 6. Sampling ---
    # Sampling is now done with pm.sample()
    # 5000 draws, 1000 tune (burn-in)
    idata = pm.sample(draws=1000, tune=1000, cores=-1, chains=2, random_seed=13)

# To view the results:
print(pm.summary(idata))
```

<details>
<summary>Code</summary>

``` python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Activate Dark Mode
plt.style.use('dark_background')

# 2. Use the figure-level function sns.displot
g = sns.displot(
    idata.posterior.occurrences[0, :, :].values, 
    kind='kde', # Use KDE for a smoother distribution line
    color='#00FFFF', # Neon Cyan for high contrast
    fill=True, 
    alpha=0.6,
    height=5, 
    aspect=1.5 # Set figure size/ratio
)

# 3. Apply final aesthetic touches to the single axis
ax = g.ax 
ax.set_title("Distribution of Occurrences", fontsize=16, loc='left', fontweight='bold', pad=20)
ax.set_xlabel("Occurrences Count", fontsize=12)
ax.set_ylabel("Density", fontsize=12)

# Ensure grid lines are subtle
ax.grid(True, alpha=0.3, linestyle='--')

# 4. Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
```

</details>

<img src="BayesianAB_files/figure-markdown_strict/fig-bayesian-plot-2-output-1.png" id="fig-bayesian-plot-2" alt="Figure 2: Distribution of occurrences for multiple hypotheses" />

<p style="text-align: center;">
Figure 2. Posterior probabilities in Hierarchical Model
</p>
<details>
<summary>Code</summary>

``` python
plt.style.use('dark_background')

diff_1_vs_4 = (idata.posterior.occurrences[:, :, 1] - idata.posterior.occurrences[:, :, 4]).values.ravel()
prob_v1_gt_v4 = (diff_1_vs_4 > 0).mean()

# Setup Figure and Axis
fig, ax = plt.subplots(figsize=(10, 5))

# Define color (Neon Magenta)
neon_color = '#FF00FF'

# 1. Create the KDE plot
sns.kdeplot(
    diff_1_vs_4, 
    ax=ax,
    fill=True, 
    color=neon_color, 
    alpha=0.6,
    linewidth=2,
    # The label includes the P(V1 > V4) calculation
    label=f"P(V1 > V4)"
)

# 2. Add Zero Line (Crucial for interpretation)
# This white dashed line marks the threshold for the probability calculation
ax.axvline(0, color='white', linestyle='--', linewidth=1.5, label="V1 = V4 (Zero Difference)")

# 3. Apply final aesthetic touches
ax.set_title(
    f"Posterior Distribution of Difference: V1 vs. V4", 
    fontsize=16, 
    loc='left', 
    fontweight='bold', 
    pad=20
)
ax.set_xlabel("V1 - V4 (Difference)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)

# Clean up
ax.legend(loc='upper left', frameon=False, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

plt.show()
```

</details>

<img src="BayesianAB_files/figure-markdown_strict/fig-bayesian-plot-3-output-1.png" id="fig-bayesian-plot-3" alt="Figure 3: Posterior distribution of difference" />

<p style="text-align: center;">
Figure 3. Probability of superiority: 88.7%
</p>

## ⚡ Part 3: The Need for Speed (Analytic Bayesian Solutions)

MCMC is great, but sometimes you need speed. For the math nerds among us, the Beta-Binomial conjugacy is pure magic. ✨

If you have a Beta prior and Binomial data, the posterior is ... drumroll ... just another Beta distribution! No complex sampling required - just simple arithmetic. Say there is $Beta(\alpha, \beta)$ prior and $k$ successes in $n$ trials, the posterior is $Beta(\alpha + k, \beta + n - k)$. This gives us closed-form solutions for the Probability of Superiority (PoS) instantly.

Say we have 10 heads from ten coin flips, what is the probability to get get a head in the next flip?

Using Bayesian approach, where

\$ \$ - hypothesis, $\mathcal{D}$ - data

$P(\mathcal{H}) - prior$

\$P( \| ) - likelihood \$

\$P( \| ) - posterior \$

$$ P(\mathcal{H} | \mathcal{D}) \propto P(\mathcal{D} | \mathcal{H}) P(\mathcal{H}) $$

It can be shown that if

$$ P(\mathcal{H}) = {Beta}(p; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} p^{\alpha-1}(1-p)^{\beta-1} $$

$$ P(\mathcal{D} | \mathcal{H}) = {Binom}(p; k, n) = C_{n}^{k} p^{k} (1-p)^{n-k} $$

Then

Proof
$$ P(\mathcal{H} | \mathcal{D}) = \frac{P(\mathcal{D} | \mathcal{H}) P(\mathcal{H})}{P(\mathcal{D})} $$

$$ P(\mathcal{D} | \mathcal{H}) P(\mathcal{H}) = \biggl ( C_{n}^{k} \cdot p^{k} (1-p)^{n-k} \biggr ) \cdot \biggl ( \mathbf{\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}} \cdot p^{\alpha-1} (1-p)^{\beta-1} \biggr ) = C \cdot p^{\alpha+k-1}(1-p)^{\beta+n-k-1} $$

$$ P(\mathcal{D}) = \int_{0}^{1} P(\mathcal{D} | \mathcal{H}) P(\mathcal{H}) dp = C \cdot \int_{0}^{1} p^{\alpha+k-1} (1-p)^{\beta+n-k-1} dp = C \cdot B(\alpha+k, \beta+n-k)$$

Consequently combining those two equations, The Binomial Constant and the Prior Beta Constant are completely canceled out

$$ P(\mathcal{H} | \mathcal{D}) = \frac{1}{B(\alpha+k, \beta+n-k)} p^{\alpha+k-1}(1-p)^{\beta+n-k-1} = {Beta}(p; \alpha + k, \beta + n - k) $$

End of Proof

In case of non-informative prior \$Beta(p; 1, 1) \$ what basically given Uniform distribution of the prior - Beta function may be presented with a short binomial coefficient formula

$$ B(k+1, n-k+1) = \frac{Г(k+1)Г(n-k+1)}{Г(n+2)} = \frac{k!(n-k)!}{(n+1)!} = \frac{1}{(n+1)\; C_{n}^{k}} $$

In our case the posterior distribution is \$ {Beta}(p; k + 1, n - k + 1) \$

Hence we can build the predictive interval using this Beta distribution moments:

$$ \mu = \frac{\alpha}{\alpha + \beta} = \frac{k+1}{n+2} $$

This formula for \$ \$ is also used as the Laplace sequence rule, which requires adding one positive and one negative observation to estimate the posterior probability distribution for a random sample.

Adding second moment

$$ \sigma^2 = {\frac {\alpha \beta }{(\alpha +\beta )^{2}(\alpha +\beta +1)}} = \frac{(k+1)(n-k+1)}{(n+2)^2(n+3)} $$

``` python
mu = 11 / 12
sigma = (11 / 12 ** 2 / 13) ** 0.5

print(f"Hence 2 sigma predictive interval for 10/10 successful flips is {mu:.2%} ± {2 * sigma :.2%}")
print(f"Alternatively as a pair of bounds: {mu - 2 * sigma:.2%} - {min(1, mu + 2 * sigma):.2%}")
```

    Hence 2 sigma predictive interval for 10/10 successful flips is 91.67% ± 15.33%
    Alternatively as a pair of bounds: 76.34% - 100.00%

But it's not a Normal distribution, so it's not 95% confidence interval, we need to take Beta distribution quantile instead or calculate it precisely. It resembles the approximation that we get using normal distribution quantiles.

``` python
l, r = sts.beta.ppf([0.05, 1.0], a=11, b=1)
print(f"Beta predictive interval {l:.2%} - {r:.2%}")
```

    Beta predictive interval 76.16% - 100.00%

Another way is get that number analytically from the integral equation that fully coincides with the value from stats package.

$$ \int_{p_{crit}}^{1} (n+1) \cdot C_n^n \cdot p^n (1-p)^0 dp = 0.95 $$

$$ p_{crit} = \sqrt[n + 1]{0.05}$$

``` python
print(f"What makes it an easy computation: p is from {0.05 ** (1/11):.2%} - 100.00%")
```

    What makes it an easy computation: p is from 76.16% - 100.00%

Criterion that is used for analytical model decision making in A/B experiment

$$ P(\lambda_B > \lambda_A) = \int_{p_B > p_A} P(p_A, p_B | \text{Data}) \, dp_A \, dp_B = \sum_{i=0}^{\alpha_B-1} \frac{B(\alpha_A+i, \beta_A+\beta_B)}{(\beta_B+i)B(1+i, \beta_B)B(\alpha_A, \beta_A)} $$

The formula is the result of applying a well-known mathematical identity that allows the cumulative probability of one Beta variable being less than another Beta variable to be expressed as a finite sum of terms involving the Beta function, rather than requiring complex numerical integration. This is why this formula is computationally efficient and preferred for exact Bayesian A/B calculations.

### Rule of three: when no successes are observed

The rule of three is used to provide a simple way of stating an approximate 95% confidence interval in the special case that no successes have been observed - $(0, 3/n)$, alternatively by symmetry, in case of only successes \$(1 - 3/n, 1) \$.

On the other hand mathematically, if all conversions are zero, then we simply may build an equation for upper bound

$(1-p)^n \geq \alpha$, where $\alpha = .05$ and hence $n \leq log_{.95}(.05)$

For example - how many trials needed to challenge the null hypothesis that the success probability is zero?

``` python
print("Approximate N:", int(3 / 0.05))
print("Exact N:", 1 + int((np.log(0.05) / np.log(0.95))))
```

    Approximate N: 60
    Exact N: 59

**Quick Tip for Zero Successes**: If you launch a test and get zero conversions, don't panic. Use the Rule of Three: your approximate 95% upper bound is simply 3/n. It's a "pretty decent" (and fast) estimate without needing a calculator.

## 😱 Part 4: The Plot Twist (Peeking)

Here is the controversial bit: Bayesian A/B testing is NOT immune to peeking. The Myth: "I can check my Bayesian results whenever I want, and it's always valid!" The Reality: Mathematically, the posterior is valid. BUT, if you use a fixed rule like "Stop when Probability \> 95%," you will inflate your False Positive Rate over time. You are essentially fishing for significance. If you stop early just because you crossed a line, you are falling into the same trap as frequentists, but but first things first.

To compare stopping rules and power, the notebook includes a Monte Carlo harness. It simulates repeated experiments, applies frequentist sequential z-tests and several Bayesian stopping rules (naive PoS threshold, expected-loss stopping (OLF), and HDI & PoS combinations), and compares false positive rates and average stopping sample sizes.

<details>
<summary>Code</summary>

``` python
from typing import Callable

n_iterations = 1000

def min_sample_size(mde, mu, sigma, alpha=0.05, power=0.80) -> int:
    """
    Defines superiority one-side z-test sample size

    Args:
        mde: Relative uplift
        mu: Expected Value
        sigma: Square root of variance
        alpha: False Positive Rate, default = 0.05
        power: Experiment power, default = 0.80

    Returns:
        Required sample size to achieve the power
    """
    effect_size = abs(mde) * mu / sigma
    return int(((sts.norm.ppf(1 - alpha) + sts.norm.ppf(power)) / effect_size) ** 2)


def stops_at(is_significant: np.ndarray, sample_size: np.ndarray) -> int:
    """
    Determines the stopping sample size.

    This function identifies the first instance where the input
    condition is True and returns the corresponding sample size.

    Args:
        is_significant: A boolean array of the stop condition for each size
        sample_size: An array of sample sizes.

    Returns:
        The stopping sample size.

    Example:
        >>> stops_at([False, False, True, True], [50, 100, 150, 200])
        150
    """
    if len(is_significant) != len(sample_size):
        raise ValueError("Input arrays must have the same length.")
    w = np.where(is_significant)[0]
    return np.nan if len(w) == 0 else sample_size[w[0]]

def monte_carlo(
    bayesian_stop_rule,
    effect_size: float=0.10,
    aa_test: bool=True,
    alpha: float=0.05,
    peeks: int = 1,
) -> None:

    result = {
        'Frequentist': [],
        'Bayesian': [],
    }

    p = 0.20
    sigma = (p * (1 - p)) ** 0.5
    relative_effect = 0 if aa_test else effect_size

    N = min_sample_size(mde=effect_size, mu=p, sigma=sigma, alpha=alpha)
    n = int(N / peeks)

    print(f"Running {n_iterations} simulations with total sample size {N} that is achieved in {peeks} iterations of {n} size each")

    for seed in tqdm(range(n_iterations)):

        rng = np.random.default_rng(seed)
        binomial_samples = rng.binomial(n=n, p=p*(1+relative_effect), size=peeks)

        sizes = np.arange(n, N + 1, n)
        conversions = np.cumsum(binomial_samples)

        z_scores = [(success / trials - p) / np.sqrt(sigma ** 2 / trials) for success, trials in zip(conversions, sizes)]
        is_prob_high_enough = [
            bayesian_stop_rule(
                success=success,
                trials=trials,
                alpha=alpha,
                p=p,
                effect_size=effect_size
            ) for success, trials in zip(conversions, sizes)
        ]

        result['Frequentist'].append(stops_at(z_scores > sts.norm.ppf(1 - alpha), sizes))
        result['Bayesian'].append(stops_at(is_prob_high_enough, sizes))

    result = "\n".join([
        f"Frequentist Rejected Rate: {np.mean(~np.isnan(result['Frequentist']))}",
        f"Frequentist Required Sample Size: {int(np.nanmean(result['Frequentist']))}",
        f"Bayesian Rejected Rate: {np.mean(~np.isnan(result['Bayesian']))}",
        f"Bayesian Required Sample Size: {int(np.nanmean(result['Bayesian']))}",
    ])

    print(result)

    return

def POS(success: int, trials: int, alpha: float, p: float, **kwargs) -> bool:
    """ Probability of Superiority decision rule """
    return sts.beta.cdf(p, a = 1 + success, b = 1 + trials - success) < alpha
```

</details>

### Correctness

``` python
monte_carlo(bayesian_stop_rule=POS, peeks=1, aa_test=True)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 1 iterations of 2473 size each
    Frequentist Rejected Rate: 0.06
    Frequentist Required Sample Size: 2473
    Bayesian Rejected Rate: 0.06
    Bayesian Required Sample Size: 2473

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"44b0480eba1e40c9b338530f3f80ec4a","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

``` python
monte_carlo(bayesian_stop_rule=POS, peeks=5, aa_test=True)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 5 iterations of 494 size each
    Frequentist Rejected Rate: 0.126
    Frequentist Required Sample Size: 1015
    Bayesian Rejected Rate: 0.126
    Bayesian Required Sample Size: 1015

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"b64b5f4cc2af46028d5703ada98c4528","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

### A/B design power

``` python
monte_carlo(bayesian_stop_rule=POS, peeks=5, aa_test=False)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 5 iterations of 494 size each
    Frequentist Rejected Rate: 0.855
    Frequentist Required Sample Size: 1146
    Bayesian Rejected Rate: 0.855
    Bayesian Required Sample Size: 1146

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"4648418bceed45a5b8d3434c04e36104","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

That is a crucial observation and it points to a common misunderstanding about Bayesian A/B testing:

No, the standard Bayesian approach does not handle peeking (optional stopping) correctly by default if your goal is to control the frequentist Type I Error Rate (False Positive Rate).

While the Bayesian interpretation of results remains valid at any time, using a fixed threshold (e.g., stopping when $P(A > B) > 95\%$) and checking repeatedly will lead to an inflated False Positive Rate over many hypothetical experiments, just like in the frequentist approach.

What is NOT Affected (The Bayesian Advantage) The Posterior Distribution and the Probability of Superiority --- is always valid, regardless of when you look at the data.

What IS Affected (The Peeking Problem) The problem arises when you use a fixed decision rule (like the $95\%$ threshold) to stop the test prematurely, based on the outcome.

The Myth: The common claim that "Bayesian testing is immune to peeking" is overstated. It is only immune in the sense that the posterior is always mathematically correct. It is not immune in the sense that it prevents the inflation of the frequentist Type I Error Rate when using a simple, fixed stopping threshold

## 🛡️ Part 5: How to Peek Safely

### How Bayesian Methods Truly Handle Peeking

To safely peek and stop early in a Bayesian framework, you need to base your decision on a metric that incorporates the cost of a wrong decision, not just the probability of a difference.

The correct Bayesian decision procedure is to stop when:
- Expected Loss (EL) is Minimized: You stop the test when the Expected Loss of choosing the suboptimal variant falls below a commercially acceptable threshold $\epsilon$. This naturally accounts for uncertainty.13 If the posterior distributions are still wide (high uncertainty), the loss will be high, and you won't stop.

$$ E[L](p_a > p_b) = \int_0^1\int_0^1L(p_a,p_b, p_a > p_b)P(p_a|a,b,n_a,k_a)P(p_b|a,b,n_b,k_b)dp_adp_b $$

-   Sequential Designs (Like Multi-Armed Bandits): Techniques like Thompson Sampling are inherently Bayesian and sequential. They don't have a stopping rule based on error rates; they simply choose the best variant to show next based on the posterior, which naturally directs more traffic to the likely winner, making the experiment efficient without needing a fixed sample size.

### Example of Loss function applied

Define the Opportunity Loss Function $L(p_A, p_B)$, which is the regret you incur by choosing a variant that is not the best.If we Choose B, the loss only happens if $p_A > p_B$: $$L(\text{Choose B}) = \max(0, p_A - p_B)$$ If we Choose A, the loss only happens if $p_B > p_A$: $$L(\text{Choose A}) = \max(0, p_B - p_A)$$

Using known properties and identities related to the Beta function, this complex double integral for Opportunity Loss function can be transformed into a closed-form summation:

$$ EL_A = \sum_{i=0}^{\alpha_B-1} \frac{\alpha_A \cdot B(\alpha_A+i+1, \beta_A+\beta_B)}{\beta_A \cdot B(i+1, \beta_B) \cdot B(\alpha_A, \beta_A)} - \sum_{i=0}^{\alpha_B-1} \frac{\alpha_B \cdot B(\alpha_A+i, \beta_A+\beta_B+1)}{(\beta_B+i) \cdot B(i+1, \beta_B) \cdot B(\alpha_A, \beta_A)} $$

The Formula for one-sample test is against benchmark $\lambda_0$:

$$ EL_A = \lambda_0 \cdot I_{\lambda_0}(\alpha, \beta) - \frac{\alpha}{\alpha+\beta} \cdot I_{\lambda_0}(\alpha+1, \beta) $$

Python Implementation note that `betainc` $(\alpha, \beta, \lambda)$ calculates $I_{\lambda}(\alpha, \beta)$, that is an equivalent of `sts.beta.cdf`$(\lambda, \alpha, \beta)$

<details>
<summary>Code</summary>

``` python
from scipy.special import betainc

def calculate_opportunity_loss_one_sample(
    k: int,
    n: int,
    lambda_0: float,
    prior_alpha: int = 1,
    prior_beta: int = 1,
) -> float:
    """
    Calculates the Expected Loss of choosing the observed variant against
    a benchmark using the analytical formula. Absolute value of conversion loss.
    Non-informative conjugate prior is used by Default.

    Args:
        k (int): Observed successes (conversions).
        n (int): Observed trials (sizes).
        lambda_0 (float): The fixed conversion rate benchmark.
        prior_alpha (int): Prior alpha hyper-parameter.
        prior_beta (int): Prior beta hyper-parameter.

    Returns:
        float: The Expected Loss of choosing the observed variant.
    """

    # Calculate Posterior Parameters (alpha and beta)
    alpha = k + prior_alpha
    beta = n - k + prior_beta

    # --- Term 1: Probability of Loss ---
    # Calculates I_lambda_bench(alpha, beta) = P(lambda_obs < lambda_bench)
    # The term is: lambda_bench * P(lambda_obs < lambda_bench)
    term1 = lambda_0 * betainc(alpha, beta, lambda_0)

    # --- Term 2: Weighted Expected Value ---
    # The fraction part: alpha / (alpha + beta) is the mean of Beta(alpha, beta)
    term2 = alpha / (alpha + beta) * betainc(alpha + 1, beta, lambda_0)

    return term1 - term2
```

</details>

Frequentist: We must collect $N$ samples to have an $1-\beta$ chance of detecting a $MDE$ difference with $\alpha$ error.

Bayesian: We will stop the test when the average potential loss incurred by choosing the sub-optimal variant is less than $\epsilon$ percentage points. By setting a small $\epsilon$, you ensure that the test continues until the potential future regret (loss) is extremely low, thus ensuring a high degree of confidence in the final decision while retaining the ability to peek safely

Adjust Monte Carlo procedure with another Bayesian stopping rule

<details>
<summary>Code</summary>

``` python
def OLF(success: int, trials: int, p: float, effect_size: float, **kwargs) -> bool:
    """Opportunity Loss Function stopping rule. Epsilon is usually set to a fraction of MDE"""
    fraction = 1 / 100
    epsilon = fraction * effect_size * p
    return calculate_opportunity_loss_one_sample(success, trials, lambda_0=p) < epsilon
```

</details>

``` python
monte_carlo(bayesian_stop_rule=OLF, peeks=30, aa_test=True)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 30 iterations of 82 size each
    Frequentist Rejected Rate: 0.249
    Frequentist Required Sample Size: 590
    Bayesian Rejected Rate: 0.21
    Bayesian Required Sample Size: 839

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"c8ce647a050042cb9f76ffdb377c8e33","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

So, Loss Function doesn't save from Peeking problem, it's even more vulnerable than well-known p-value approach

We need another piece of the puzzle ...

### Here comes the sun 🌤️ ... and HDI

Highest Density Interval (sometimes called Highest Posterior Density Interval).

Definition: The $X\%$ HDI is the narrowest interval that contains $X\%$ of the probability mass of the posterior distribution.

Purpose: It is the Bayesian equivalent of the frequentist Confidence Interval (CI), but unlike the CI, you can state that there is an $X\%$ probability that the true parameter value (e.g., the true conversion rate) lies within the HDI.

The width of the HDI is simply (Upper Bound - Lower Bound). It is a direct and intuitive measure of the remaining uncertainty. A wide HDI means your posterior is flat and uncertain; a narrow HDI means your posterior is sharply peaked and confident.

<details>
<summary>Code</summary>

``` python
import arviz as az

def calculate_beta_hdi_width(alpha: float, beta: float, hdi_prob=0.95, num_samples=10_000) -> float:
    """
    Calculates the Highest Density Interval (HDI) for a Beta distribution
    using Monte Carlo sampling and the arviz library.

    The calculation of the HDI is an iterative process that must find
    the interval boundary points where the probability density is equal,
    while the area between the points equals the target probability.

    Since Beta distribution is generally not symmetrical, the HDI bounds are not
    the same as the quantiles, which is why a specialized function is needed.

    Args:
        alpha (float): The posterior alpha parameter (k_obs + prior_alpha).
        beta (float): The posterior beta parameter (n_obs - k_obs + prior_beta).
        hdi_prob (float): The target probability mass (e.g., 0.95 for 95% HDI).
        num_samples (int): Number of samples to draw for Monte Carlo calculation.

    Returns:
        tuple: (lower_bound, upper_bound, width)
    """
    posterior_samples = sts.beta.rvs(a=alpha, b=beta, size=num_samples)

    # designed to work on posterior samples
    hdi_interval = az.hdi(posterior_samples, hdi_prob=hdi_prob)

    lower_bound = hdi_interval[0]
    upper_bound = hdi_interval[1]

    return upper_bound - lower_bound
```

</details>

Let's update Monte Carlo once again with a combination of PoS and HDI stopping what represent business and statistical robustness respectively, shall we? - Note that HDI density affects the inference vastly and you'd better experiment to pick up a good for for the data.

<details>
<summary>Code</summary>

``` python
print(f"Width is multiplied by {round(calculate_beta_hdi_width(100, 100, .99) / calculate_beta_hdi_width(100, 100, 0.8), 2)} when increase required density from 0.8 to 0.99")
```

</details>

    Width is multiplied by 1.98 when increase required density from 0.8 to 0.99

<details>
<summary>Code</summary>

``` python
def HDI(success: int, trials: int, alpha: float, p: float, effect_size: float) -> bool:
    """ PoS combined with 95% HDI stopping rule """
    return (
        # checks if 95% of posterior distribution is narrow enough and lays in ± MDE
        (calculate_beta_hdi_width(1 + success, 1 + trials - success) < 2 * effect_size * p)
        # checks if posterior distribution by 95% chance better
        and POS(success, trials, alpha, p)
    )
```

</details>

#### Correctness

``` python
monte_carlo(bayesian_stop_rule=OLF, peeks=1, aa_test=True)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 1 iterations of 2473 size each
    Frequentist Rejected Rate: 0.06
    Frequentist Required Sample Size: 2473
    Bayesian Rejected Rate: 0.069
    Bayesian Required Sample Size: 2473

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"bb0d04680b55411c97e9aa6e2e52ca2d","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

``` python
monte_carlo(bayesian_stop_rule=HDI, peeks=10, aa_test=True)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 10 iterations of 247 size each
    Frequentist Rejected Rate: 0.193
    Frequentist Required Sample Size: 876
    Bayesian Rejected Rate: 0.1
    Bayesian Required Sample Size: 1924

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"9f64a18492b04c9a89a896b63ecf0b6a","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

#### Power

``` python
monte_carlo(bayesian_stop_rule=HDI, peeks=5, aa_test=False)
```

    Running 1000 simulations with total sample size 2473 that is achieved in 5 iterations of 494 size each
    Frequentist Rejected Rate: 0.855
    Frequentist Required Sample Size: 1146
    Bayesian Rejected Rate: 0.818
    Bayesian Required Sample Size: 2035

<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"5dbeef9db69f4f3696d4f7f85db7b7cb","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>

HDI accompanied by Probability of Superiority is a good criterion, although very strict if you increase the required density for HDI from 80% above 95% and it hence requires bigger sample size than Frequentist approach.

Combination of HDI and PoS checks makes the criterion less sensitive to peeking, however it's yet not fully immune.

### Stopping Rules Overview

This table breaks down three common criteria used in Bayesian A/B testing, highlighting their function and their **robustness against peeking** (stopping a test too early based on transient results).

| Criterion                            | Function / Definition                                                                                                       | Robustness Against Peeking                                                                                                                                                  |
|:-----------------------|:-----------------------|:-----------------------|
| **Probability of Superiority (PoS)** | Measures how often the posterior probability of $\lambda_B$ is greater than $\lambda_A$ (i.e., $P(\lambda_B > \lambda_A)$). | **Low Robustness.** The threshold can be quickly and spuriously crossed by early noise or transient fluctuations.                                                           |
| **Expected Loss (EL)**               | Measures the average **Cost or Regret** of selecting the inferior variant.                                                  | **High Robustness.** Requires the posterior distribution to be **tight enough** that the potential loss (regret) is small, preventing premature stopping.                   |
| **HDI Width**                        | Measures the **Precision** or **Uncertainty** of the posterior distribution (e.g., the width of the 95% credible interval). | **High Robustness.** Forces the test to continue until the uncertainty is **low** (the HDI is narrow), regardless of the posterior mean, ensuring adequate data collection. |

## Conclusions & Practical Recommendations

-   Use the posterior (and PoS) to *interpret* results, but prefer decision-theoretic stopping when making business choices: stop when the expected loss of choosing the sub-optimal variant is below a tolerated threshold.
-   . Combine HDI (precision) with PoS (direction) for a conservative, safe stopping rule - but higher density HDI thresholds require larger samples.
-   When many variants or small groups are present, hierarchical models provide safer estimates via partial pooling.
-   If your goal is to guarantee frequentist properties (e.g., Type I control under peeking), design the sequential procedure explicitly: group sequential testing with alpha-spending function or always valid inference approach - [Sequential Testing Guide](https://npodlozhniy.github.io/posts/sequential-testing/) will let you know all you need

Further experiments to try:
- Replace uniform priors with domain-informed priors when available.
- Explore Thompson Sampling for continuous allocation instead of fixed-sample stopping.
- Visualize posterior trajectories and stopping-rule trade-offs across simulated peeks.

Thanks for reading, feel free to fork this notebook and go forth, experiment with your own traffic and loss thresholds as well as react to the post below, and may your posteriors always be narrow!

<script type=application/vnd.jupyter.widget-state+json>
{"state":{"0181731c3e7e4f5886dd418d811ebd9e":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"0554d93f7dfc4c33b270fed6a8c731b1":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"05924aeff1f54bf58cafc79de6f4cd1c":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"07aefbc23958493cadf61a3f7d2ff449":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"0962f908e5ac479d9ac335cb7fb1d3bc":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_fbdcd143a2d8422b854acebbb0836f4d","placeholder":"​","style":"IPY_MODEL_0bacea67b70b47678da4114c670622eb","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:00&lt;00:00, 1571.80it/s]"}},"097ea0be9e6540fea1f6e37723965b49":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_b0780cbcf67543e4abec37637e0b9e29","placeholder":"​","style":"IPY_MODEL_e9a4b7f06616455784758576ea78ca50","tabbable":null,"tooltip":null,"value":"100%"}},"0bacea67b70b47678da4114c670622eb":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"0e4900c9d34448cc9300f7b4916479a7":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_188043e6369142cfaa1b38fcd1fb521e","placeholder":"​","style":"IPY_MODEL_a98e46b8237c463e9fe8a9eacbe5fe64","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:16&lt;00:00, 59.57it/s]"}},"12b7239b009d469a86a2682da9e62f10":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"1661914f8d204fc0b4cef9f0f5cd39a3":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"1747c39aeb1c4b8385942369bbec3ad7":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"188043e6369142cfaa1b38fcd1fb521e":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"210c2eefb8d74134a10a25821824da79":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"35e2327e3b684962a1d3fc681b715f9c":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_210c2eefb8d74134a10a25821824da79","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_83fbbfa362a948359151ec0e17a0399d","tabbable":null,"tooltip":null,"value":1000}},"3a83457f00be47ef8fc554c7f7b8517b":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"3c21633b81d44236a7df49cde47eca47":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"3c42a4d930324bd88406ae5388ddeb0a":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_3a83457f00be47ef8fc554c7f7b8517b","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_8ab523655e834570af9b70fa6d1183dd","tabbable":null,"tooltip":null,"value":1000}},"3f1594d73f1f49e4a38ec80e9622e820":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"3f306533888f4810a2576ca380f5cd51":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"44b0480eba1e40c9b338530f3f80ec4a":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_f61f94b740a54a48b61f483eb9300723","IPY_MODEL_ab9afaf12fee40c1a21c02b59f40bd41","IPY_MODEL_d1c458e4ae4644608d5096c26d093061"],"layout":"IPY_MODEL_4a6d6934fb8d4396b0acaa493989c358","tabbable":null,"tooltip":null}},"4648418bceed45a5b8d3434c04e36104":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_bcc95361668d4390bb257a7b843e4ed9","IPY_MODEL_35e2327e3b684962a1d3fc681b715f9c","IPY_MODEL_da97c3b36cd74cafb0a46f1ba8966fff"],"layout":"IPY_MODEL_4d2bd273d5c340d79d6853ac4e39b63c","tabbable":null,"tooltip":null}},"49d1a84abc9b4807967d5109322735db":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"4a6d6934fb8d4396b0acaa493989c358":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"4d2bd273d5c340d79d6853ac4e39b63c":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"4f16c40cea8b4f5890d36a17cc525cac":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_5ef167be804746fdbbf7b62c01253249","placeholder":"​","style":"IPY_MODEL_a8d4a21db8c14e139850f9375e826814","tabbable":null,"tooltip":null,"value":"100%"}},"54ab74155fff46768e46e7d042c82bb1":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"54deea47056043d3a45590c3857fffd4":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"5d2b15f2353f4ab7884060d588a68800":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_12b7239b009d469a86a2682da9e62f10","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_bc5772f963a6403cad16a9acb93e1dce","tabbable":null,"tooltip":null,"value":1000}},"5dbeef9db69f4f3696d4f7f85db7b7cb":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_097ea0be9e6540fea1f6e37723965b49","IPY_MODEL_9ff46d370eef4689a6efe1e63fcccb96","IPY_MODEL_9f40db484b314e66a32b9ba0efce260b"],"layout":"IPY_MODEL_0181731c3e7e4f5886dd418d811ebd9e","tabbable":null,"tooltip":null}},"5ef167be804746fdbbf7b62c01253249":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"6574bc8f963740b8929eb2a0e770ab7a":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_a5db07c447e24909b4b0dce01e17ab40","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_3f306533888f4810a2576ca380f5cd51","tabbable":null,"tooltip":null,"value":1000}},"6825e9a3aa7d44fda10d8ec64acf056d":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_c9ee3fe4ac904e4f838321673281b3d4","placeholder":"​","style":"IPY_MODEL_f2a2496263bb4b9dacbfabaf91f5d9a2","tabbable":null,"tooltip":null,"value":"100%"}},"6a970a6dd66d45c0927d614586daf792":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_49d1a84abc9b4807967d5109322735db","placeholder":"​","style":"IPY_MODEL_3f1594d73f1f49e4a38ec80e9622e820","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:00&lt;00:00, 1555.70it/s]"}},"83fbbfa362a948359151ec0e17a0399d":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"8ab523655e834570af9b70fa6d1183dd":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"9c4649fe4645490795bf4709c74bae66":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_f899e84351a147edbd2be712c214197b","placeholder":"​","style":"IPY_MODEL_a7e642821b9546fab95641336bb47695","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:00&lt;00:00, 1175.91it/s]"}},"9d98642b7d27440788e99d1915949fc8":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"9dd589fcc9ea45ca8cf36a4716ea632d":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"9ed474f75820407f9bfa7468ce7bbb04":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"9f40db484b314e66a32b9ba0efce260b":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_baeeae25db614d0a8d8b79a07f3c3c46","placeholder":"​","style":"IPY_MODEL_07aefbc23958493cadf61a3f7d2ff449","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:08&lt;00:00, 116.80it/s]"}},"9f64a18492b04c9a89a896b63ecf0b6a":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_fa391050a9914997b77bb38b97d756e5","IPY_MODEL_6574bc8f963740b8929eb2a0e770ab7a","IPY_MODEL_0e4900c9d34448cc9300f7b4916479a7"],"layout":"IPY_MODEL_9ed474f75820407f9bfa7468ce7bbb04","tabbable":null,"tooltip":null}},"9ff46d370eef4689a6efe1e63fcccb96":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_54ab74155fff46768e46e7d042c82bb1","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_c4a1ec09f9404fafbe02066da169514d","tabbable":null,"tooltip":null,"value":1000}},"a5db07c447e24909b4b0dce01e17ab40":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"a6451f818dc048aa842b021e8e94a5ad":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"a7e642821b9546fab95641336bb47695":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"a8d4a21db8c14e139850f9375e826814":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"a98e46b8237c463e9fe8a9eacbe5fe64":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"ab9afaf12fee40c1a21c02b59f40bd41":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_d2a993f8a6964c4a81c829cedf79b5cd","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_e6742189ef95428ba32b4f3c017802c7","tabbable":null,"tooltip":null,"value":1000}},"aeb31422aa1046728e09396a8bd29f91":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"b0780cbcf67543e4abec37637e0b9e29":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"b64b5f4cc2af46028d5703ada98c4528":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_6825e9a3aa7d44fda10d8ec64acf056d","IPY_MODEL_5d2b15f2353f4ab7884060d588a68800","IPY_MODEL_9c4649fe4645490795bf4709c74bae66"],"layout":"IPY_MODEL_f0adcdd3da4847b18e4bcdc20a845078","tabbable":null,"tooltip":null}},"b913916eee934b66940d06a8e48457b6":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"ba80dad4887445dca7aa63ebca2e3974":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"baeeae25db614d0a8d8b79a07f3c3c46":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"bb0d04680b55411c97e9aa6e2e52ca2d":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_e66faca9c6a34064801a112eb58cf80f","IPY_MODEL_e4fd4829079644148b0a2dd17b5efba6","IPY_MODEL_6a970a6dd66d45c0927d614586daf792"],"layout":"IPY_MODEL_54deea47056043d3a45590c3857fffd4","tabbable":null,"tooltip":null}},"bc5772f963a6403cad16a9acb93e1dce":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"bcc95361668d4390bb257a7b843e4ed9":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_1747c39aeb1c4b8385942369bbec3ad7","placeholder":"​","style":"IPY_MODEL_3c21633b81d44236a7df49cde47eca47","tabbable":null,"tooltip":null,"value":"100%"}},"bda4a8aa7b894945be5c9e33fb15a431":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"c4a1ec09f9404fafbe02066da169514d":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"c8ce647a050042cb9f76ffdb377c8e33":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HBoxModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HBoxModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HBoxView","box_style":"","children":["IPY_MODEL_4f16c40cea8b4f5890d36a17cc525cac","IPY_MODEL_3c42a4d930324bd88406ae5388ddeb0a","IPY_MODEL_0962f908e5ac479d9ac335cb7fb1d3bc"],"layout":"IPY_MODEL_bda4a8aa7b894945be5c9e33fb15a431","tabbable":null,"tooltip":null}},"c9ee3fe4ac904e4f838321673281b3d4":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"d1c458e4ae4644608d5096c26d093061":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_9d98642b7d27440788e99d1915949fc8","placeholder":"​","style":"IPY_MODEL_ee863cd54d4d4fabb5f616ec3e63f748","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:00&lt;00:00, 1697.57it/s]"}},"d2a993f8a6964c4a81c829cedf79b5cd":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"d56634c72c7445c58808a1c8bd83387c":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"da97c3b36cd74cafb0a46f1ba8966fff":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_1661914f8d204fc0b4cef9f0f5cd39a3","placeholder":"​","style":"IPY_MODEL_ba80dad4887445dca7aa63ebca2e3974","tabbable":null,"tooltip":null,"value":" 1000/1000 [00:00&lt;00:00, 984.02it/s]"}},"e4fd4829079644148b0a2dd17b5efba6":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"FloatProgressModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"FloatProgressModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"ProgressView","bar_style":"success","description":"","description_allow_html":false,"layout":"IPY_MODEL_9dd589fcc9ea45ca8cf36a4716ea632d","max":1000,"min":0,"orientation":"horizontal","style":"IPY_MODEL_e6a437b6eea94b6ca0205463a1447f86","tabbable":null,"tooltip":null,"value":1000}},"e66faca9c6a34064801a112eb58cf80f":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_b913916eee934b66940d06a8e48457b6","placeholder":"​","style":"IPY_MODEL_a6451f818dc048aa842b021e8e94a5ad","tabbable":null,"tooltip":null,"value":"100%"}},"e6742189ef95428ba32b4f3c017802c7":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"e6a437b6eea94b6ca0205463a1447f86":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"ProgressStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"ProgressStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","bar_color":null,"description_width":""}},"e9a4b7f06616455784758576ea78ca50":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"ee863cd54d4d4fabb5f616ec3e63f748":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"f0adcdd3da4847b18e4bcdc20a845078":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"f2a2496263bb4b9dacbfabaf91f5d9a2":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLStyleModel","state":{"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLStyleModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"StyleView","background":null,"description_width":"","font_size":null,"text_color":null}},"f61f94b740a54a48b61f483eb9300723":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_aeb31422aa1046728e09396a8bd29f91","placeholder":"​","style":"IPY_MODEL_05924aeff1f54bf58cafc79de6f4cd1c","tabbable":null,"tooltip":null,"value":"100%"}},"f899e84351a147edbd2be712c214197b":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}},"fa391050a9914997b77bb38b97d756e5":{"model_module":"@jupyter-widgets/controls","model_module_version":"2.0.0","model_name":"HTMLModel","state":{"_dom_classes":[],"_model_module":"@jupyter-widgets/controls","_model_module_version":"2.0.0","_model_name":"HTMLModel","_view_count":null,"_view_module":"@jupyter-widgets/controls","_view_module_version":"2.0.0","_view_name":"HTMLView","description":"","description_allow_html":false,"layout":"IPY_MODEL_d56634c72c7445c58808a1c8bd83387c","placeholder":"​","style":"IPY_MODEL_0554d93f7dfc4c33b270fed6a8c731b1","tabbable":null,"tooltip":null,"value":"100%"}},"fbdcd143a2d8422b854acebbb0836f4d":{"model_module":"@jupyter-widgets/base","model_module_version":"2.0.0","model_name":"LayoutModel","state":{"_model_module":"@jupyter-widgets/base","_model_module_version":"2.0.0","_model_name":"LayoutModel","_view_count":null,"_view_module":"@jupyter-widgets/base","_view_module_version":"2.0.0","_view_name":"LayoutView","align_content":null,"align_items":null,"align_self":null,"border_bottom":null,"border_left":null,"border_right":null,"border_top":null,"bottom":null,"display":null,"flex":null,"flex_flow":null,"grid_area":null,"grid_auto_columns":null,"grid_auto_flow":null,"grid_auto_rows":null,"grid_column":null,"grid_gap":null,"grid_row":null,"grid_template_areas":null,"grid_template_columns":null,"grid_template_rows":null,"height":null,"justify_content":null,"justify_items":null,"left":null,"margin":null,"max_height":null,"max_width":null,"min_height":null,"min_width":null,"object_fit":null,"object_position":null,"order":null,"overflow":null,"padding":null,"right":null,"top":null,"visibility":null,"width":null}}},"version_major":2,"version_minor":0}
</script>
