---
title: "Bayesian A/B Testing — PyMC examples, HDI, and Safe Stopping"
author: "Nikita Podlozhniy"
summary: "Practical guide: PyMC examples, hierarchical models, loss functions, HDI, and safe Bayesian stopping rules."
date: "2025-11-29"
format:
    hugo-md:
jupyter: python3
execute:
    enabled: true
    cache: true
tags:
    - bayesian
    - ab-testing
    - pymc
    - sequential-testing
categories:
    - posts
language: en
draft: false
---

**Intro**

This post collects practical Bayesian A/B testing recipes I use in production experiments. It contains short runnable PyMC examples, hierarchical models for multiple hypotheses, analytic Bayesian updates, Monte Carlo comparisons against frequentist procedures, and recommendations for stopping rules that are safer under optional stopping.

---

**Prerequisites**

- Python 3.11+ recommended
- Main packages used in the notebook: `pymc`, `numpy`, `scipy`, `arviz`, `seaborn`, `matplotlib`.
- If you render the notebook with Quarto, set `QUARTO_PYTHON` to the Python interpreter used for the notebook (see `README.md`).

---

**1. Simple A/B with PyMC**

This minimal example shows how to fit two Bernoulli conversion rates with weak (uniform) priors and compute the posterior difference (delta). The posterior samples let you compute the probability that variant A is better than B.

```python
from pymc import Uniform, Bernoulli
import pymc as pm
import numpy as np

# synthetic data
a_default, b_default = 0.06, 0.04
a_count, b_count = 200, 150
rng = np.random.default_rng(seed=42)

a_bernoulli_samples = rng.binomial(n=1, p=a_default, size=a_count)
b_bernoulli_samples = rng.binomial(n=1, p=b_default, size=b_count)

with pm.Model() as my_model:
    A_prior = Uniform('A_prior', lower=0, upper=1)
    B_prior = Uniform('B_prior', lower=0, upper=1)

    A_observed = Bernoulli('A_observed', p=A_prior, observed=a_bernoulli_samples)
    B_observed = Bernoulli('B_observed', p=B_prior, observed=b_bernoulli_samples)

    delta = pm.Deterministic('delta', A_prior - B_prior)
    idata = pm.sample(draws=5000, tune=1000, cores=2)

# Posterior: probability A > B can be computed from samples
print('P(A > B):', (idata.posterior['delta'].values > 0).mean())
```

Notes:
- Use `idata` (ArviZ `InferenceData`) to inspect posterior means, credible intervals, and to plot densities.

---

**2. Hierarchical models (multiple hypotheses)**

When you have several related experiments or many variants, hierarchical (multilevel) models help by partially pooling information. A typical construction for binary outcomes models group-level probabilities as draws from a common Beta(a, b) prior whose hyperparameters are estimated from the data (this induces shrinkage).

```python
trials = np.array([842, 854, 862, 821, 839])
successes = np.array([27, 47, 69, 52, 35])

with pm.Model() as hierarchical_model:
    a = pm.Uniform('a', lower=0.01, upper=100)
    b = pm.Uniform('b', lower=0.01, upper=100)
    # custom weak prior on total precision
    pm.Potential('beta_precision_potential', -2.5 * np.log(a + b))
    occurances = pm.Beta('occurances', alpha=a, beta=b, shape=len(trials))
    likelihood = pm.Binomial('likelihood', n=trials, p=occurances, observed=successes)
    idata = pm.sample(draws=5000, tune=1000, cores=2, random_seed=42)

# Inspect posterior of group probabilities and pairwise differences
```

Why this helps:
- Small groups will be pulled toward the global mean (shrinkage), producing more stable estimates and fewer spurious extremes.

---

**3. Analytic Bayesian update (Beta-Binomial)**

The Beta prior is conjugate to the Binomial likelihood. If prior is Beta(alpha, beta) and data is k successes out of n trials, the posterior is Beta(alpha + k, beta + n - k). For a uniform prior Beta(1,1), the posterior mean is (k+1)/(n+2) (Laplace rule).

Use Beta quantiles for credible and predictive intervals instead of normal approximations when counts are small.

---

**4. Probability of Superiority (PoS)**

PoS = P(p_A > p_B) can be computed either by Monte Carlo from posterior draws or using closed-form finite-sum identities for Beta posteriors. The latter is exact and fast.

---

**5. Monte Carlo experiments & stopping criteria**

I include a Monte Carlo harness in the original notebook to compare frequentist (z-test) procedures and Bayesian stopping rules. The main rules considered:

- Frequentist sequential stopping based on one-sided z-scores.
- Naive Bayesian stopping when PoS > threshold (e.g., 95%). Important: this inflates frequentist Type I error under repeated peeking.
- Decision-theoretic stopping based on Expected Loss: stop when expected regret of choosing the suboptimal variant drops below business threshold epsilon.
- Uncertainty-aware stopping: require both a sufficiently narrow HDI (precision) and a high PoS to stop.

Key helper functions used in the notebook (kept verbatim):

```python
from scipy import stats as sts
import numpy as np

def min_sample_size(mde, mu, sigma, alpha=0.05, power=0.80) -> int:
    """Approximate one-sided z-test sample size for desired power."""
    effect_size = abs(mde) * mu / sigma
    return int(((sts.norm.ppf(1 - alpha) + sts.norm.ppf(power)) / effect_size) ** 2)

def stops_at(is_significant: np.ndarray, sample_size: np.ndarray) -> int:
    """Return first sample size where condition is True or NaN if never."""
    w = np.where(is_significant)[0]
    return np.nan if len(w) == 0 else sample_size[w[0]]
```

**Expected Loss (analytic)**

Using Beta function identities we can get a closed-form expression for expected opportunity loss when comparing a variant to a fixed benchmark. This yields a principled decision rule: stop when Expected Loss < epsilon.

**HDI (Highest Density Interval)**

HDI width measures posterior precision. Combining a narrow HDI with a high PoS is more conservative and robust to peeking than PoS alone.

Example HDI width helper (Monte Carlo + ArviZ):

```python
import arviz as az
from scipy import stats as sts

def calculate_beta_hdi_width(alpha, beta, hdi_prob=0.95, num_samples=10_000):
    posterior_samples = sts.beta.rvs(a=alpha, b=beta, size=num_samples)
    hdi_interval = az.hdi(posterior_samples, hdi_prob=hdi_prob)
    return hdi_interval[1] - hdi_interval[0]
```

---

**6. Practical recommendations**

- Don't rely on PoS > 95% alone if you need frequentist Type I error control; it inflates FPR under optional stopping.
- Prefer decision-theoretic thresholds (expected loss) aligned to business impact when possible.
- Combine HDI width (precision) with PoS to avoid premature stopping on noisy early signals.
- For many related comparisons, hierarchical models reduce false discoveries and increase stability.

---

**7. Rendering & local preview**

The notebook `content/posts/bayesian-test/BayesianAB.ipynb` contains runnable cells and is the source of this article. To render and preview locally:

```bash
# (optional) render notebook to markdown with Quarto
quarto render content/posts/bayesian-test/BayesianAB.ipynb

# run Hugo dev server
hugo server -D
```

If you use Quarto's `--execute` flag, ensure `QUARTO_PYTHON` points to the interpreter used for the notebook.

---

**8. Files & next steps**

- The notebook remains in `content/posts/bayesian-test/BayesianAB.ipynb` and this post is available at `content/posts/bayesian-test/BayesianAB.md`.
- I can:
  - run the notebook cells here and embed outputs/figures into the `.md`,
  - run `quarto render` and commit the rendered `.md` with outputs, or
  - add tags/categories/translations if you want a different taxonomy.

Tell me which follow-up you prefer and I will continue.
