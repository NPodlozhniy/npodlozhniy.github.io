---
title: "Mastering Homogeneity hypothesis testing"
summary: "Why Fisher's Exact test is the relic of the past and Chi-squared is the only one you need"
author: "Nikita Podlozhniy"
date: "2024-12-01"
format:
    hugo-md:
        output-file: "contingency-test.md"
        html-math-method: katex
        code-fold: true
jupyter: python3
execute:
    enabled: false
---

## Contingency Tests Overview

**Intro**

In the world of statistical analysis, contingency tests play a crucial role in examining the relationship between two categorical variables. These tests are essential tools for researchers across various disciplines, enabling them to determine whether there is a significant correlation between the variables of interest.

**Real-world Relevance**

To illustrate the practical significance of contingency tests, let's consider a real-world scenario:
imagine a market research team is investigating the relationship between customer satisfaction (a few levels e.g. Satisfied, Neutral, Dissatisfied) and the type of product purchased (there are multiple products) from an online marketplace. They collect data from a limited sample of customers who recently made purchases building a contingency table. By applying a contingency test, such as Fisher's exact or Chi-squared test, researchers can determine whether customer satisfaction and the type of product purchased are connected.

**Focus**

This notebook embarks on a journey to explore the subtleties of Fisher's exact vs. Chi-squared tests application, delving into Fisher's implementation nuances, and performance characteristics. A comparative analysis of two implementations of the Fisher's exact test is covered: one crafted in pure Python and the other leveraging the statistical package of R through the `rpy2` library. Furthermore, we'll scrutinize the performance and accuracy of both approaches, comparing their results and dissecting their respective advantages.

## Background: Fisher's Exact and Chi-Squared

**Contingency Tables and Statistical Independence**

Contingency tables serve as the foundation for these tests, presenting the observed frequencies of different combinations of categories for the two or more variables. By analyzing the distribution of frequencies within the table, contingency tests help to assess whether the observed patterns are likely due to chance or reflect a genuine interconnection between the variables.

<details>
<summary>Code</summary>

``` python
table = [[1, 24, 5], [5, 20, 7], [14, 11, 7], [11, 14, 8], [10, 10, 10], [12, 12, 12]]
```

</details>

### Fisher's Exact

Fisher's Exact Test, a non-parametric statistical test, plays a pivotal role in hypothesis testing for categorical data. This test is particularly valuable when dealing with small sample sizes, where the assumptions of the chi-squared test are violated.

**Derivation of the Hypergeometric Distribution**

The hypergeometric distribution arises from a scenario involving sampling without replacement from a finite population containing two types of objects: "successes" and "failures." In the context of contingency tables, these objects correspond to the different categories of the two variables being analyzed.

Consider a population of size N, containing K objects classified as "successes" and N-K objects classified as "failures." We draw a sample of size n without replacement from this population. The hypergeometric distribution describes the probability of obtaining exactly k successes in the sample.

**The Hypergeometric Distribution Formula**

The probability mass function of the hypergeometric distribution is given by:

$$P(X=k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}$$

where:

-   N: Total population size
-   K: Number of successes in the population
-   n: Sample size
-   k: Number of successes in the sample

**Fisher's Exact Test: Applying the Hypergeometric Distribution**

Fisher's Exact Test leverages the hypergeometric distribution to calculate the exact probability of observing a given contingency table, or one more extreme, assuming the null hypothesis of independence between the variables. This exact probability is then used to assess the statistical significance of the observed association.

### Chi-Squared

The Chi-squared test is another widely used method for analyzing contingency tables to determine whether there is a significant connection between categorical variables. It relies on a statistical approach based on the Chi-squared distribution.

**Chi-Squared Statistic: Measuring the Difference**

The Chi-squared test calculates a test statistic, denoted by $\chi^2$, which quantifies the difference between the observed frequencies in the contingency table and the frequencies expected under the null hypothesis of independence. The null hypothesis assumes that there is no association between the variables, meaning that the observed frequencies should be close to the expected frequencies.

**Testing the Null Hypothesis**

The calculated chi-squared statistic is then compared to the chi-squared distribution with degrees of freedom determined by the dimensions of the contingency table. If the table has $N \times M$ size then degrees of freedom = $(N-1)(M-1)$

**Assumptions and Limitations**

Both of these procedures, like any statistical test, operates under certain assumptions which are crucial for ensuring the validity of the test's results.
Their common requirements are:

1.  **Categorical Data:** The variables being analyzed must be categorical, meaning they can be divided into distinct categories or groups.
2.  **Independent Observations:** The observations in the contingency table should be independent of each other. This means that the outcome of one observation should not influence the outcome of another observation.

In addition each of them has a third extra requirement

-   Fisher: **Fixed Margins**

> The row and column totals in the contingency table are considered fixed. This implies that the sample sizes for each category are predetermined.

-   Chi-squared: **Sample sizes**

> The expected cell frequencies should be sufficiently large for the chi-squared approximation to be valid

Apparently namely the third condition for each is the most challenging.

The third assumption for Fisher is quite strict and is not usually satisfied in practice. There are other representatives of exact test's family that are free of this requirement like Boschloo's or Barnard's tests, although they are much more computationally expensive, as they require the nuisance parameters estimation and it's not feasible to implement them for the tables larger than 2x2. So the performance is the main issue of the exact tests and if it's the case then Chi-squared test is advised to be applied instead.

For Chi-squared violations of the third assumption can lead to inaccurate results. In such cases, Fisher's exact test is often preferred due to its ability to handle small sample sizes and sparse tables where the chi-squared test's approximations may not hold true.

## Fisher's Exact Test Implementation

### Pythonic Fisher NxM

As long as widely used python packages for statistics like `scipy` or `statsmodels` don't furnish Fisher's exact test for tables larger than 2x2, here is the author's pure Pythonic implementation for this procedure, to get more details follow the function documentation below.

<details>
<summary>Code</summary>

``` python
import math


def _pvalue(func: object, shape: tuple=(2,2)) -> str:
    print(f"p-value: {func([row[:shape[1]] for row in table[:shape[0]]]):.5f}")


def NxM_Fisher_exact_test(table: list[list]) -> float:
    """
    Performs Fisher's exact test for a contingency table of an arbitrary size.

    Parameters
    ----------
    table: list[list]
        contigency matrix M x N

    Returns
    -------
    p-value: float
    """
    num_rows = len(table)
    num_cols = len(table[0])

    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[i][j] for i in range(num_rows)) for j in range(num_cols)]

    log_p_constant = (
        sum(math.lgamma(x + 1) for x in row_sums)
        + sum(math.lgamma(y + 1) for y in col_sums)
        - math.lgamma(sum(row_sums) + 1)
    )

    def calculate_log_probability(matrix):
        """
        Calculates the log-probability of a contingency table n x m.

        Fisher's statistic under the truthful null hypothesis has a
        hypergeometric distribution of the numbers in the cells of the table.

        Therefore the probability of the contingency table follows
        hypergeometric probability mass function $C^K_k * C^N-K_n-k / C^N_n$

        So, simplifying it's clear that the probability follows:
        the product of factorials of total row and total columns counts
        divided by the total count factorial and factorials of each cell count.

        row_1! x..x row_n! x col_1! x..x col_m! / (cell_11! x..x cell_nm! x total!)

        1. As the gamma function satisfies: gamma(n + 1) = n!
        and it's computationally more stable- it's used instead of factorials.

        2. Making the computations more stable I'm switching from product to sum
        using logarithmic probability.

        """
        return log_p_constant - sum(
                math.lgamma(cell + 1) for row in matrix for cell in row
        )

    log_p_obs = calculate_log_probability(table)

    p_value = 0

    def dfs(matrix: list[list], row_id, col_id, tol=1e-10):
        """
        Recursive deep-first search function

        Generates all possible contingency tables and calculates their
        log-probability adding up those, that are at least as extreme as
        the observed contingency table, to the total p-value

        Args:
            matrix: A list of lists representing the contingency table
            row_id: Row index up to which the table is already filled
            col_id: Column index up to which the table is already filled
            tol: Maximum absolute log-probability comparison error

        Returns:
            None
        """
        nonlocal p_value

        # Copy is necessary to make recursion working
        table = [row.copy() for row in matrix]

        # Stopping condition - only the last row and column are left
        if row_id == num_rows - 1 and col_id == num_cols - 1:

            for i in range(row_id): # fill last column
                table[i][col_id] = row_sums[i] - sum(table[i][:col_id])
            for j in range(col_id): # fill last row
                table[row_id][j] = col_sums[j] - sum(table[i][j] for i in range(row_id))

            bottom_right_cell = row_sums[row_id] - sum(table[row_id][:col_id])

            if bottom_right_cell < 0:
                # Non-reliable table, all cells must be non-negative
                return

            else:
                table[row_id][col_id] = bottom_right_cell
                log_p = calculate_log_probability(table)

                if log_p <= log_p_obs + tol:
                    p_value += math.exp(log_p)

                return

        # Fill the table until the Stopping condition isn't met
        else:

            remaining_row_sum = row_sums[row_id] - sum(table[row_id])
            remaining_col_sum = col_sums[col_id] - sum(table[i][col_id] for i in range(num_rows))

            for k in range(min(remaining_row_sum, remaining_col_sum) + 1):

                table[row_id][col_id] = k

                if row_id == num_rows - 2 and col_id == num_cols - 2:
                    dfs(table, row_id + 1, col_id + 1, tol=tol)
                elif row_id == num_rows - 2:
                    dfs(table, 0, col_id + 1, tol=tol)
                else:
                    dfs(table, row_id + 1, col_id, tol=tol)

    dfs(matrix=[[0] * num_cols for _ in range(num_rows)], row_id=0, col_id=0)

    return p_value
```

</details>

While this exact test above is a precise solution, it does have limitations related to the computational intensity of the test, especially when dealing with large contingency tables. As the table size increases, the number of possible arrangements of data grows exponentially, making the calculations more time-consuming.

``` python
_pvalue(NxM_Fisher_exact_test, shape=(3, 2))
```

    p-value: 0.00014

### R Fisher NxM

Another option that is to use `rpy` bridge from Python to R, this function works for an arbitrary shape of contingency table and unfortunately doesn't have alternatives in Python.

<details>
<summary>Code</summary>

``` python
import numpy as np

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr


def R_fisher_exact_test(table: list[list]) -> float:
    """
    Performs exact Fisher's test using R

    Parameters
    ----------
    table: list[list]
        contigency matrix M x N

    Returns
    -------
    p-value: float
    """

    # Enable automatic conversion between NumPy and R arrays
    rpy2.robjects.numpy2ri.activate()

    # Import necessary R package
    stats = importr('stats')

    # Perform Fisher's test using the R function with more memory to get p-value
    result = stats.fisher_test(np.array(table), workspace = 2e9)

    # Extract the p-value
    p_value = result[0][0]

    return p_value
```

</details>

Note that the `rpy2` package has native dependencies, what in particular means that, installed R accompanied by the corresponding libraries is required.

``` python
_pvalue(R_fisher_exact_test, shape=(3, 2))
```

    p-value: 0.00014

### Algorithmic Differences: Python vs R

While both the Python and R implementations ultimately calculate the p-value for Fisher's Exact Test, they employ distinct algorithms under the hood, each with its own strengths and weaknesses. Understanding these differences is crucial for selecting the most appropriate implementation for a given scenario.

**Python:** utilizes a recursive algorithm to enumerate all possible contingency tables that could arise under the null hypothesis. This approach, while conceptually straightforward, can become computationally expensive.

**R:** in contrast, it leverages optimized algorithms and data structures that are specifically designed for efficient calculation of Fisher's Exact Test. These algorithms, often implemented in compiled languages, take advantage of advanced numerical techniques and data representations to minimize computational overhead.

### Performance Comparison

**Trade-offs and Considerations:**

The choice between the Python and R implementations depends on the specific needs of the analysis. For smaller tables, the Python implementation may suffice, offering ease of understanding and implementation. However, as the table size increases, the computational advantages of the R implementation become more pronounced.

I suggest we generate random contingency tables with dimensions ranging from 2x2 to 5x5, representing a diverse range of scenarios encountered in real-world applications. For each table size, we will measure the execution time required by both the Python and R implementations to calculate the p-value.

<details>
<summary>Code</summary>

``` python
%%time
_pvalue(NxM_Fisher_exact_test, shape=(3, 2))
_pvalue(NxM_Fisher_exact_test, shape=(4, 2))
_pvalue(NxM_Fisher_exact_test, shape=(3, 3))
```

</details>

    p-value: 0.00014
    p-value: 0.00012
    p-value: 0.00085
    CPU times: user 745 ms, sys: 7.96 ms, total: 753 ms
    Wall time: 756 ms

<details>
<summary>Code</summary>

``` python
%%time
_pvalue(R_fisher_exact_test, shape=(3, 2))
_pvalue(R_fisher_exact_test, shape=(4, 2))
_pvalue(R_fisher_exact_test, shape=(3, 3))
```

</details>

    p-value: 0.00014
    p-value: 0.00012
    p-value: 0.00085
    CPU times: user 2.6 s, sys: 219 ms, total: 2.82 s
    Wall time: 2.82 s

<details>
<summary>Code</summary>

``` python
%%time
_pvalue(R_fisher_exact_test, shape=(4, 3))
```

</details>

    p-value: 0.00149
    CPU times: user 1.46 s, sys: 97 ms, total: 1.55 s
    Wall time: 1.88 s

<details>
<summary>Code</summary>

``` python
%%time
_pvalue(NxM_Fisher_exact_test, shape=(4, 3))
```

</details>

    p-value: 0.00149
    CPU times: user 8min, sys: 921 ms, total: 8min 1s
    Wall time: 8min 5s

For sizes more than 2 x 5 and 3 x 3, R package function can significantly outperform the Python counterpart, whereas with tables of less size pure Python function is shining.

The results of the benchmarks revealed a clear trend: Python implementation is beaten in terms of execution time for larger tables. As the table dimensions increased, the performance gap between the two implementations is widened drastically. This observation aligns with the algorithmic differences discussed earlier, where the optimized algorithms and data structures employed by the R implementation proved to be more efficient.

As a rule of thumb I propose to apply R if $N \times M > 10$, otherwise Python is preferable and what is more - it doesn't have any dependencies on non-native packages

### Accuracy Comparison

Along with the performance let's assure that the p-values generated by both methods are equivalent

<details>
<summary>Code</summary>

``` python
from scipy.stats import multinomial

from statsmodels.stats.proportion import proportion_confint

P = np.arange(.1, .35, .05)
n = 40

rv = multinomial(n, P)

np.random.seed(2024)

tol = 1e-5
alpha = 0.05
n_iterations = 100


for shape in [(2, 2), (2, 4), (3, 3)]:

    false_positives = 0

    for i in range(n_iterations):

        contingency_table = rv.rvs(shape[0])

        p_value_py = NxM_Fisher_exact_test([row[:shape[1]] for row in contingency_table])
        p_value_r = R_fisher_exact_test([row[:shape[1]] for row in contingency_table])

        if abs(p_value_py - p_value_r) > tol:
            print(f"Different p-values! Python: {p_value_py}, R: {p_value_r}")
            break
        elif p_value_py <= alpha:
            false_positives += 1

    l, r = proportion_confint(count=false_positives, nobs=n_iterations, alpha=0.10, method='wilson')
    print(f"shape: {shape}, false positives: {false_positives/n_iterations:.3f} ± {(r - l) / 2:.3f}")
```

</details>

    shape: (2, 2), false positives: 0.020 ± 0.026
    shape: (2, 4), false positives: 0.030 ± 0.030
    shape: (3, 3), false positives: 0.030 ± 0.030

So, it's clear from multiple iterations for different tables and sizes that there is no a single case of different p-values, so the equivalence is practically evident. In addition I've checked Type I error level, it's well below the bound of 5%, which means that ideologically the criterions are valid.

### Performance Analysis: A Comparative Benchmark

As the textbooks say, Fisher's Exact Test stands out as a particularly versatile option when dealing with small sample sizes or sparse contingency tables. It provides accurate p-values, even when the assumptions of other commonly used test, such as the chi-squared test, might be violated. This benefit makes Fisher's Exact test an invaluable method when working with limited data or situations where the chi-squared test's approximation is not applicable.

I offer you a procedure to challenge these statements, namely to call the power of the exact test out, when chi-squared assumptions are not satisfied. First we will check the correctness of these two methods and then the power.

<details>
<summary>Code</summary>

``` python
def chi_squared_challenge(
    shape: tuple = (2, 2),
    n_iterations: int=1_000,
    alpha: float=0.05,
    aa_test: bool=True
) -> None:

    for i in range(1 + len(rv.rvs()[0]) - shape[1]):

        fisher_positives = 0
        chi2_positives = 0
        chi2_yates_positives = 0

        zero_expected_count = 0

        less_than_5 = 0
        less_than_10 = 0

        for _ in range(n_iterations):

            contingency_table = rv.rvs(shape[0])[:, i:i+shape[1]]

            if not aa_test:
                contingency_table[0] = contingency_table[0] ** 2

            if np.min(contingency_table) == 0:
                zero_expected_count += 1
                continue

            less_than_5 += np.max(np.array(contingency_table) < 5)
            less_than_10 += np.max(np.array(contingency_table) < 10)

            if shape == (2, 2):
                p_value_fisher = fisher_exact(contingency_table).pvalue
                p_value_chi2_yates = chi2_contingency(contingency_table, correction=True).pvalue
                if p_value_chi2_yates <= alpha:
                    chi2_yates_positives += 1
            else:
                p_value_fisher = NxM_Fisher_exact_test(contingency_table)
            p_value_chi2 = chi2_contingency(contingency_table, correction=False).pvalue


            if p_value_chi2 <= alpha:
                chi2_positives += 1
            if p_value_fisher <= alpha:
                fisher_positives += 1

        valid_tables = n_iterations - zero_expected_count

        print(
            f"\nIf out of {valid_tables} valid {shape[0]}x{shape[1]} tables "
            f"(w/o zero expected count) number of tables with less than:"
            f"\n - 5 elements in any cell is {less_than_5}"
            f"\n - 10 elements in any cell is {less_than_10}"
            f"\n Then p-values are:"
        )

        l, r = proportion_confint(count=fisher_positives, nobs=valid_tables, alpha=0.10, method='wilson')
        print(f"Fisher positives: {fisher_positives/valid_tables:.3f} ± {(r - l) / 2:.3f}")

        l, r = proportion_confint(count=chi2_positives, nobs=valid_tables, alpha=0.10, method='wilson')
        print(f"Chi2 positives: {chi2_positives/valid_tables:.3f} ± {(r - l) / 2:.3f}")

        if not shape == (2, 2):
            continue

        l, r = proportion_confint(count=chi2_yates_positives, nobs=valid_tables, alpha=0.10, method='wilson')
        print(f"Chi2 Yates positives: {chi2_yates_positives/valid_tables:.3f} ± {(r - l) / 2:.3f}")
```

</details>

According to frequently encountered requirements in the literature regarding expected cell counts for chi-squared test application, a common rule is at least 5 (some requires 10) in all cells of 2x2 table, and 5 or more in 80% of cells in larger tables, but no cells with zero expected count. Furthermore, when the assumption for 2x2 table is not met, Yates's correction is applied.

Now, we will check the feasibility of these conditions for 2x2 tables first.

#### Correctness

<details>
<summary>Code</summary>

``` python
np.random.seed(26)

chi_squared_challenge(aa_test=True)
```

</details>


    If out of 969 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 915
     - 10 elements in any cell is 969
     Then p-values are:
    Fisher positives: 0.019 ± 0.007
    Chi2 positives: 0.043 ± 0.011
    Chi2 Yates positives: 0.011 ± 0.006

    If out of 993 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 566
     - 10 elements in any cell is 993
     Then p-values are:
    Fisher positives: 0.024 ± 0.008
    Chi2 positives: 0.050 ± 0.011
    Chi2 Yates positives: 0.015 ± 0.006

    If out of 1000 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 178
     - 10 elements in any cell is 993
     Then p-values are:
    Fisher positives: 0.032 ± 0.009
    Chi2 positives: 0.052 ± 0.012
    Chi2 Yates positives: 0.022 ± 0.008

    If out of 1000 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 27
     - 10 elements in any cell is 831
     Then p-values are:
    Fisher positives: 0.033 ± 0.009
    Chi2 positives: 0.049 ± 0.011
    Chi2 Yates positives: 0.022 ± 0.008

#### Power

<details>
<summary>Code</summary>

``` python
np.random.seed(26)

chi_squared_challenge(aa_test=False)
```

</details>


    If out of 969 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 785
     - 10 elements in any cell is 969
     Then p-values are:
    Fisher positives: 0.300 ± 0.024
    Chi2 positives: 0.359 ± 0.025
    Chi2 Yates positives: 0.265 ± 0.023

    If out of 993 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 373
     - 10 elements in any cell is 987
     Then p-values are:
    Fisher positives: 0.325 ± 0.024
    Chi2 positives: 0.370 ± 0.025
    Chi2 Yates positives: 0.295 ± 0.024

    If out of 1000 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 90
     - 10 elements in any cell is 897
     Then p-values are:
    Fisher positives: 0.334 ± 0.025
    Chi2 positives: 0.363 ± 0.025
    Chi2 Yates positives: 0.305 ± 0.024

    If out of 1000 valid 2x2 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 9
     - 10 elements in any cell is 586
     Then p-values are:
    Fisher positives: 0.351 ± 0.025
    Chi2 positives: 0.387 ± 0.025
    Chi2 Yates positives: 0.327 ± 0.024

Surprise! In this example it's shown there is no need for Yates nor for Fisher's exact test at all! Chi-squared test doesn't inflate the number of Type I errors and keep the power at least as high as it's for the exact test regardless of the number of cells with low frequencies.

JFYI: Some other exact tests might be applied instead of Fisher's test, e.g. Boschloo's test provides higher power but it's a) much slower and b) yet worse than a plain chi-squared, you may prove it on your own as an exercise. Hint: there is a function `boschloo_exact` in `scipy`

Okay, it' clear with 2x2, but what if the table is getting bigger?

<details>
<summary>Code</summary>

``` python
np.random.seed(26)

chi_squared_challenge(shape=(2, 4), aa_test=False)
```

</details>


    If out of 969 valid 2x4 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 804
     - 10 elements in any cell is 969
     Then p-values are:
    Fisher positives: 0.685 ± 0.025
    Chi2 positives: 0.688 ± 0.024

    If out of 993 valid 2x4 tables (w/o zero expected count) number of tables with less than:
     - 5 elements in any cell is 388
     - 10 elements in any cell is 993
     Then p-values are:
    Fisher positives: 0.666 ± 0.025
    Chi2 positives: 0.672 ± 0.024

The power values are not statistically distinguishable, so chi-squared is still the winner as it's much simpler in calculations and for the data model that I specified it seems that it can handle small table sizes - it takes low values at least as good as Fisher's test.

#### Monte-Carlo Simulation Results Interpretation

I'd like to make an extra note that Monte-Carlo simulations can provide valuable insights into the performance and behavior of statistical tests, but it's essential to interpret their results with caution and awareness of their limitations.

While simulations can mimic real-world scenarios, they are inherently limited by the assumptions and parameters used in their design. They may not capture the full complexity of real-world data and may not be generalizable to all situations. Therefore, it's crucial to consider the specific context and limitations of the simulation when interpreting its results.

Saying that I must admit that I don't have an intention to prove that there is no need for exact tests in any experiment design, I'd rather invite you to challenge your data and your experiments set up specifics, as there is a chance that you will find that chi-squared test is all you need for contingency experiments.

### Justification for Fisher's Exact over Chi-Squared

Even when the chi-squared test appears to perform well in Monte-Carlo simulations, there are compelling theoretical and practical reasons to prefer Fisher's Exact Test in specific situations. Understanding these justifications is crucial for making informed decisions about which test to apply.

1.  Sparse Table: Chi-squared test relies on approximations that may not hold true when dealing with sparse contingency tables.
2.  Sample Size: Chi-squared test is based on the asymptotic distribution of the test statistic, which assumes that the sample size is large.
3.  Effect Size: Chi-squared may be less sensitive in detecting the small effect size, because the approximation may not be as accurate.

So, once again: in order to guarantee that you don't have a need for exact tests in your data model setting, you must consciously simulate your data distributions (especially when it comes to sparse tables, small sample sizes and small effect sizes) and then make a decision, the process that I presented here is based on the data my team is exposed to most and hopefully it might be easily simulated with Multinomial distribution.

## General Pipeline

Finally I'd like to offer you a full pipeline on how to organize contingency tests efficiently: when Fisher's exact test shall be applied and when Chi-squared is just enough.

As you know, exact tests could take time and what I want to achieve is to have a control over the time that I allocate to the function execution.

There are a few ways to implement timeouts, my favourite one is leveraging `multiprocessing` capabilities, however it's not always the case that you can run a subprocess under your main process in production, so another concise way to apply timeouts will be shown via `func_timeout` library.

### Concurrent Execution

Simple decorator to pass the output from the subprocess into main process.

<details>
<summary>Code</summary>

``` python
from typing import Optional
from functools import wraps
from multiprocessing import Queue, Process

def subprocess_output(procedure: object, queue: Optional[Queue]=None) -> object:

    @wraps(procedure)
    def wrapper(*args, **kwargs):

        p_value = procedure(*args, **kwargs)
        queue.put(p_value)

        return p_value

    return wrapper


def concurrent_test(
    method: object,
    table: list[list],
    name: str="NxM Fisher's exact test",
    timeout: int=10,
) -> float:
    """
    Runs the given method in a separate process with a timeout.
    If the process takes longer than the timeout, it is terminated.
    The result is returned from the main process.

    Parameters
    ----------
    method: object
        The method to be run in a separate process.
    table: list[list]
        Contingency matrix M x N
    name: str
        Process name
    timeout: int
        Time limit for subprocess execution

    Returns
    -------
    p-value: float
    """
    queue = Queue()
    procedure = subprocess_output(method, queue)
    p = Process(
        target=procedure,
        args=(table,),
        name=name
    )
    p.start()
    p.join(timeout=timeout)
    p.terminate()
    if p.exitcode is not None:
        p_value = queue.get()
        return p_value
```

</details>

### Timeout Execution

Handy function that is a good solution if a timeout is the only thing you want to get from a subprocess

<details>
<summary>Code</summary>

``` python
# pip install func-timeout
from func_timeout import func_timeout, FunctionTimedOut


def timeout_test(
    method: object,
    table: np.ndarray,
    timeout: int=10,
) -> float:

    try:
        p_value = func_timeout(timeout, method, args=(table,))
    except FunctionTimedOut:
        p_value = None

    return p_value
```

</details>

By the way: there is no need for timeout when running `scipy` Fisher's exact test, so it's applied only to those methods analyzed in the chapter above.

<details>
<summary>Code</summary>

``` python
%%time
fisher_exact(np.array([[10000, 4000], [12000, 5000]]))
```

</details>

    CPU times: total: 0 ns
    Wall time: 14 ms

    SignificanceResult(statistic=1.0416666666666667, pvalue=0.10488212218194087)

### Universal procedure

#### Logic

<details>
<summary>Code</summary>

``` python
from scipy.stats import chi2_contingency, fisher_exact


def _contingency_test(table: np.ndarray, criterion: str, timeout: int) -> dict:
    """
    Performs a test of independence of variables in a contingency table

    Parameters
    ----------
    table: np.ndarray
        Contingency matrix M x N
    criterion: str
        Test to be performed
    timeout: int
        Time limit for Fisher's exact test

    Returns
    -------
    dict: {
        "method": str,
        "p-value": float,
        "error": str (optional)
    }
    """
    if criterion not in {"chi-squared", "fisher-exact", "textbook"}:
        raise ValueError(
            "Incorrect type of criterion, "
            "should be one of the following: 'chi-squared', 'fisher-exact', 'textbook'"
        )

    # No timeout if "fisher-exact" criterion is set
    timeout = timeout + 10_000 * (criterion == "fisher-exact")

    result = dict.fromkeys(["method", "p-value"])

    try:
        if criterion == "textbook" and table.shape == (2, 2) and (table >= 10).all():
            result["method"] = "2x2 Pearson's chi-squared test with Yates"
            test = chi2_contingency(table, correction=True)
            result["p-value"] = test.pvalue
        elif criterion != "chi-squared" and table.shape == (2, 2):
            result["method"] = "2x2 Fisher's exact test in Python"
            test = fisher_exact(table)
            result["p-value"] = test.pvalue
        elif criterion == "chi-squared" or (criterion == "textbook" and np.sum(table >= 5) >= np.size(table) * 0.80):
            result["method"] = "NxM Pearson's chi-squared test w/o Yates"
            test = chi2_contingency(table, correction=False)
            result["p-value"] = test.pvalue
        else: # try Exact fisher test if doesn't take too much
            if np.size(table) > 10:
                name = "NxM Fisher's exact test in R"
                p_value = timeout_test(
                    R_fisher_exact_test, table, timeout
                )
            else:
                name = "NxM Fisher's exact test in Python"
                p_value = timeout_test(
                    NxM_Fisher_exact_test, table, timeout
                )
            result["method"] = name
            if p_value:
                result["p-value"] = p_value
            else:
                result["method"] = " ".join([
                    result["method"],
                    "timed out.",
                    "NxM Pearson's chi-squared test approximation applied"
                ])
                test = chi2_contingency(table, correction=False)
                result["p-value"] = test.pvalue

    except Exception as error:
        result["error"] = f"{error}"

    return result
```

</details>

Here is a quick example of how it works with the identified table

<details>
<summary>Code</summary>

``` python
t = np.array([row[:2] for row in table[:5]])

_contingency_test(t, 'textbook', 5)
```

</details>

    {'method': "NxM Pearson's chi-squared test w/o Yates",
     'p-value': 0.00032439327665678783}

#### Input validation

Adding a validation is an important step to prevent end-users from inference the function in the wrong way

<details>
<summary>Code</summary>

``` python
def _validate_input(table: list[list]) -> np.array:

    try:
        array = np.array(table)
    except ValueError:
        raise ValueError(
            "Contingency table's rows must be of equal length."
        )

    try:
        array = array.astype(dtype=int)
    except ValueError:
        raise ValueError(
            "All cells must contain integer numbers."
        )

    if array.ndim != 2:
        raise ValueError(
            "Contigency table must be a 2-dimensional array."
        )

    if (array < 0).any():
        raise ValueError(
            "All cells must contain non-negative numbers."
        )

    if array.shape[0] == 2 and np.max(np.min(array, axis=1)) == 0:
        raise ValueError(
            "There are cells with zero expected count. "
            "Expectations must contain only positive numbers."
        )

    return array
```

</details>

#### Put it all together

<details>
<summary>Code</summary>

``` python
def general_contingency_test(table: list[list], criterion='textbook', timeout: int=10) -> dict:
    """
    Performs a test of independence of variables in a contingency table

    Parameters
    ----------
    table: list[list]
        matrix M x N,
        where M is the number of compared groups and N is the set of measures
    timeout: int
        Time limit for Fisher's exact test,
        if the calculation takes longer chi-squared test is applied instead

    Returns
    -------
    dict: {
        "method": str,
        "p-value": float,
        "error": str (optional)
    }
    """
    return _contingency_test(_validate_input(table), criterion, timeout)
```

</details>

Now when we have a general procedure, let's take a look at a few examples of the inference

<details>
<summary>Code</summary>

``` python
general_contingency_test(table, criterion='chi-squared')
```

</details>

    {'method': "NxM Pearson's chi-squared test w/o Yates",
     'p-value': 0.0020940807559433087}

<details>
<summary>Code</summary>

``` python
general_contingency_test([[1, 2, 3, 5, 6, 100, 2000], [4, 5, 6, 7, 8, 150, 1000]], timeout=10)
```

</details>

    {'method': "NxM Fisher's exact test in R timed out. NxM Pearson's chi-squared test approximation applied",
     'p-value': 5.060441099877772e-17}

## The logic wrapped into the library

Good news for you, you don't need to repeat all the code that was shared above as it's already a part of the public Python package `podlozhnyy_module` that comes hande every time you have data analysis assignments at work.

**Key Features of the Library:**

-   **Automatic Test Selection:** By default, the library automatically selects the most appropriate test based on textbook rules, considering factors such as sample size and expected cell frequencies. This intelligent selection process ensures the validity and accuracy of the results.
-   **Flexibility and Control:** Users have the flexibility to override the automatic selection and force the library to apply a specific test if desired. This feature is particularly useful when researchers have prior knowledge or preferences regarding the test to be used.
-   **User-Friendly Interface:** The library's interface is designed to be intuitive and easy to use, enabling researchers to effortlessly perform contingency tests without extensive coding or technical expertise.

``` python
# !pip install podlozhnyy-module==2.6-alpha

import podlozhnyy_module as pm
```

Library's application is as simple as the following command

``` python
pm.contingency.general_contingency_test(table[:5], criterion='fisher-exact', timeout=1)
```

    {'method': "NxM Fisher's exact test in R", 'p-value': 0.0011288225617825118}

``` python
pm.contingency.general_contingency_test(table[:3], criterion='fisher-exact', timeout=1)
```

    {'method': "NxM Fisher's exact test in Python",
     'p-value': 0.0008451539443552633}

## Conclusion: A Unified Framework for Contingency Tests

This notebook has explored various aspects of contingency tests, with a particular focus on Fisher's Exact Test comparison to Chi-squared test. As a part of the journey we have contrasted the Python and R implementations of the Fisher's test, highlighting their strengths and weaknesses in terms of performance, accuracy, and computational considerations.

The powerful framework of Monte-Carlo simulations is provided to enable the readers to simulate their data and ultimately apply proper testing techniques in their field providing the accurate guidance to the business.

**podlozhnyy_module** library: a flexible solution

The code presented in this notebook has been thoughtfully integrated into the `podlozhnyy_module` library, offering a flexible and user-friendly solution for conducting contingency tests. This library empowers users to select the most appropriate test based on textbook rules or to override the default behavior and force the application of a specific test, such as Fisher's Exact or the Chi-squared test.
