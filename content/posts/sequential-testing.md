---
title: "Sequential Testing Guide"
summary: "How GST is often misinterpreted and why its genuine version is way better than AVI?"
author: "Nikita Podlozhniy"
date: "2024-12-25"
format:
    hugo-md:
        output-file: "sequential-testing.md"
        html-math-method: katex
        code-fold: true
jupyter: python3
execute:
    enabled: false
---

<!-- comment it out
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
-->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>


# Intro

Sequential testing designs have gained significant popularity in AB testing due to their ability to potentially reduce the required sample size and experiment duration while maintaining statistical correctness. This approach allows for interim analyses of data as it accumulates, offering the possibility to stop the experiment early if a clear winner emerges, or if it becomes evident that the treatment effect is insufficient to justify continuing (stop for futility)

In this article, we shift our focus away from the theoretical intricacies of the problem and instead delve into a comprehensive exploration of available sequential testing solutions. We will discuss their implementations, compare their performance, and highlight their strengths and weaknesses. By examining these practical aspects, we aim to equip practitioners with the knowledge to make informed decisions when incorporating sequential testing into their AB testing workflows.

# Approach

Among the variety of sequential testing designs, there are basically two broad families of algorithms: Group Sequential Testing (GST) and Always Valid Inference (AVI). These methods represent distinct philosophies in their handling of interim analyses and experiment stopping criteria..

-   GST works with predefined interim analysis points and utilizes predetermined stopping boundaries to decide whether to stop the experiment at each stage.

-   AVI allows continuous monitoring and providing valid confident intervals at any point, making it adaptable for uncertain experiment duration or analysis frequencies.

This article primarily focuses on these two techniques, providing an overview of their methodologies and practical implications.

# Group Sequential Testing

For group sequential testing there is a handy [package in R](https://github.com/cran/ldbounds/blob/master/R/ldBounds.R) and I will show below how to use it, although for those, who prefers Python due to any reason, whether it is the absence of an interpreter, infrastructure limitations or just personal preferences, there is no direct and popular alternative package, so I had to write it on my own and now ready to share with you after careful testing and benchmarking.

## GST in R

If you're ready to use R library there are two options: use R runtime directly or through `rpy2` Python package.
Both options are available for example within Google Colab environment.

Here is an instance of `rpy2` package inference within Colab Notebook, when you run R code from the Python interpreter using an extension.

### iPython Notebook

``` python
%load_ext rpy2.ipython
```

``` python
%%R
R.version.string
install.packages("ldbounds")
```

``` python
%%R
library(ldbounds)
ldBounds(t=seq(1/4, 1, 1/4), iuse=3, phi=1, alpha=0.05, sides=1)$upper.bounds
```

	[1] 2.241403 2.125078 2.018644 1.925452

### Python File

The code above may be rewritten into a simple `.py` file as follows, you are to use created `stats` package as a plain Python package thereafter.

> **⚠️ Caution**
>
> It will not work in Google Colab for instance, as it requires R installed in addition to Python.

<details>
<summary>Code</summary>

``` python
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)

# R package names
packnames = ('ldbounds')

# Selectively install what needs to be install.
names_to_install = [
    package for package in packnames if not rpackages.isinstalled(package)
]

if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

stats = rpackages.importr('ldbounds')
```

</details>

If R is not your cup of tea, or simply there is no option to run it within the scope of the production infrastructure, what is the most common limitation by the way, now begins exactly what you need.

## GST in Python

There is quite popular incorrect implementation powered by Zalando [`expan`](https://github.com/zalando/expan/blob/master/expan/core/early_stopping.py)

It works without probability integration and mistakenly leverage alpha spending function as a critical value at each step, it's common misunderstanding about alpha-spending function approach, there are number of implementations that do it the exact same wrong way, and even world's leading publication for data science, according to their own definition, makes the same mistakes, for instance: [Understanding of Group Sequential Testing published in Towards Data Science](https://towardsdatascience.com/understanding-group-sequential-testing-befb35cec07a)

As it will shown below that approach is statistically incorrect and so it's highly recommended to avoid it.

Instead I propose you to apply the new library [`seqabpy`](https://github.com/NPodlozhniy/seqabpy) that is powerful and accurate and what is more important implemented according to the original papers [Interim analysis:
The alpha spending function approach by K. K. Gordon Lan and David L. DeMets (1983)](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/LanDeMets.pdf) and further related publications, you may find them all mentioned in Reference lines of the methods' docstrings, let's take a look at the functionality

<details>
<summary>Code</summary>

``` python
import numpy as np
import pandas as pd

from scipy.stats import norm
```

</details>

`seabpy` provides Group Sequential Testing in a separate module - `gatsby` which name is an anagram of **G**roup **S**equential **AB T**esting in **PY**thon and below it's shown why `gatsby` is often referred as «The Great Gatsby»

``` python
#!pip install seqabpy
from seqabpy import gatsby
```

### Lan-DeMets

`calculate_sequential_bounds` function implements rigorous approach to calculate confidence bounds in one-sided GST.
In addition to upper bounds, if `beta` is provided it calculates lower bounds to unlock the option of stopping the test for futility, maintaining provided Type II error rate. The algorithm is taken from the article [Group sequential designs using both type I and type II error probability spending functions by Chang MN, Hwang I, Shih WJ. (1998)](https://www.tandfonline.com/doi/abs/10.1080/03610929808832161)

``` python
gatsby.calculate_sequential_bounds(np.linspace(1/10, 1, 10), alpha=0.05, beta=0.2)
```

    Sequential bounds algorithm to stop for futility converged to 0.00064 tolerance in 8 iterations using O'Brien-Fleming spending function.

    (array([-3.02102866, -1.41478896, -0.59632535, -0.05664366,  0.35069266,
             0.68203514,  0.96419224,  1.21185008,  1.43395402,  1.79496377]),
     array([6.08789285, 4.22919942, 3.39632756, 2.90614903, 2.57897214,
            2.34174062, 2.15981329, 2.0146325 , 1.89528829, 1.79496377]))

`ldBounds` function returns the exact same numbers as R package and tailored to have similar interface, in both input and output.

As a subtle benefit it supports more spending functions, take a look at the docstring to know more details.

``` python
gatsby.ldBounds(t=np.linspace(1/4, 1, 4), iuse=3, phi=1)
```

    {'time.points': array([0.25, 0.5 , 0.75, 1.  ]),
     'alpha.spending': array([0.0125, 0.0125, 0.0125, 0.0125]),
     'overall.alpha': 0.05,
     'upper.bounds': array([2.24140273, 2.1251188 , 2.01870509, 1.92553052]),
     'nominal.alpha': array([0.0125    , 0.01678835, 0.02175894, 0.02708151])}

In case of calculation of upper bounds only the algorithm is faster, given that these boundaries shall be defined and fixed offline prior to the experiment start, it's totally sensible performance, when in addition lower aka futility bounds are computed it takes more time

``` python
%%time
gatsby.ldBounds(t=np.linspace(1/10, 1, 10), alpha=0.1)
```

    CPU times: total: 7.73 s
    Wall time: 7.75 s

    {'time.points': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
     'alpha.spending': array([1.97703624e-07, 2.34868089e-04, 2.43757240e-03, 6.62960181e-03,
            1.07070137e-02, 1.37029812e-02, 1.55891351e-02, 1.66134835e-02,
            1.70337597e-02, 1.70513868e-02]),
     'overall.alpha': 0.10000000000000009,
     'upper.bounds': array([5.07115563, 3.4973068 , 2.79510152, 2.38848818, 2.11870064,
            1.92350461, 1.77394818, 1.65463799, 1.55659472, 1.47418673]),
     'nominal.alpha': array([1.97703624e-07, 2.34990500e-04, 2.59417098e-03, 8.45892623e-03,
            1.70578873e-02, 2.72083541e-02, 3.80358613e-02, 4.89989759e-02,
            5.97833696e-02, 7.02156613e-02])}

An example of less popular Haybittle-Peto spending function usage

``` python
gatsby.ldBounds(t=np.linspace(1/4, 1, 4), iuse=5)
```

    {'time.points': array([0.25, 0.5 , 0.75, 1.  ]),
     'alpha.spending': array([0.0013499, 0.       , 0.       , 0.0486501]),
     'overall.alpha': 0.05,
     'upper.bounds': array([3.        , 3.        , 3.        , 1.63391418]),
     'nominal.alpha': array([0.0013499 , 0.0013499 , 0.0013499 , 0.05113844])}

### GST

`GST` is the general function that accounts for various deviations in the experiment design, in other words if the peeking strategy is different the method adjusts the bounds to guarantee the valid statistical approach whenever it's possible.

In particular it's perfect to handle a few changed peeking points when the total number of peeking remains the same, and it works in a best possible way with under- and oversampling, in the latter case the procedure is not fully correct though as we will see in the simulations part.

The idea of implementation is taken from [Group Sequential and Confirmatory Adaptive Designs in Clinical Trials by G. Wassmer and W. Brannath (2016)](https://link.springer.com/book/10.1007/978-3-319-32562-0)

Here are a few examples, that shows how different peeking strategy affects the bounds:

``` python
gatsby.ldBounds(t=np.array([0.3, 0.6, 1.0]), alpha=0.025)
```

    {'time.points': array([0.3, 0.6, 1. ]),
     'alpha.spending': array([4.27257874e-05, 3.76533752e-03, 2.11919367e-02]),
     'overall.alpha': 0.02499999999999991,
     'upper.bounds': array([3.92857254, 2.669972  , 1.98103004]),
     'nominal.alpha': array([4.27257874e-05, 3.79287858e-03, 2.37939526e-02])}

-   in case of under-sampling the last upper bound is lower what reflects that all the rest of alpha volume is spent at this point
-   in case of over-sampling the last upper bound is higher, what helps to control Type I error rate, after the expected sample size is reached

``` python
gatsby.GST(actual=np.array([0.3, 0.6, 0.8]), expected=np.array([0.3, 0.6, 1]), alpha=0.025)
```

    array([3.92857254, 2.669972  , 1.96890411])

``` python
gatsby.GST(actual=np.array([0.3, 0.6, 1.2]), expected=np.array([0.3, 0.6, 1]), alpha=0.025)
```

    array([3.92857254, 2.669972  , 1.98949242])

Under- and over- sampling may also happen in a more natural way, when a few peeking points are added or removed

``` python
gatsby.ldBounds(t=np.array([0.3, 0.6, 0.8, 1.0]), alpha=0.025)
```

    {'time.points': array([0.3, 0.6, 0.8, 1. ]),
     'alpha.spending': array([4.27257874e-05, 3.76533752e-03, 8.40372704e-03, 1.27882097e-02]),
     'overall.alpha': 0.02499999999999991,
     'upper.bounds': array([3.92857254, 2.669972  , 2.28886308, 2.03074404]),
     'nominal.alpha': array([4.27257874e-05, 3.79287858e-03, 1.10436544e-02, 2.11404831e-02])}

``` python
gatsby.GST(actual=np.array([0.3, 0.6, 0.8]), expected=np.array([0.3, 0.6, 0.8, 1]), alpha=0.025)
```

    array([3.92857254, 2.669972  , 2.15083427])

``` python
gatsby.GST(actual=np.array([0.3, 0.6, 1, 1.2]), expected=np.array([0.3, 0.6, 1]), alpha=0.025)
```

    array([3.92857254, 2.669972  , 1.98102292, 2.0375539 ])

`GST` also supports `int` as an input, if peeking point are distributed uniformly it's what you should use in sake of convenience

``` python
gatsby.GST(7, 7)
```

    array([5.05481268, 3.48557771, 2.78550934, 2.38021304, 2.1113425 ,
           1.9168349 , 1.76778516])

While the method comes handy in most of scenarios, it doesn't support all the possible deviations: the beginning of the expected and actual peeking strategies must be the same: so it's either over- or under- sampling or the change in peeking points when their number remains equal

Needless to say that the application of `GST` and other functions mentioned above apparently is not limited to one-sided hypotheses, in order to test two-sided alternative: just set $\alpha$ to half of the value, like `0.025` if you want to challenge two-sided hypothesis at `0.95` confidence level, and define lower bounds symmetrically about zero, so they would be the same in absolute values, but negative.

``` python
gatsby.GST(actual=np.array([0.3, 0.6, 0.9, 1.2]), expected=np.array([0.3, 0.6, 0.8, 1]), alpha=0.025)

# the following will not work
# gatsby.GST(actual=np.array([0.3, 0.6]), expected=np.array([0.8, 1]))
# gatsby.GST(
#     actual=np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
#     expected=np.array([0.3, 0.6, 0.8, 1])
# )
```

    array([3.92857254, 2.669972  , 2.30467653, 2.05129976])

# Always Valid Inference

While AVI is becoming increasingly popular in the field, bypassing GST, it's worth noting that there are currently no widely adopted, comprehensive Python or R packages that focus solely on this approach.

There is one recent package `savvi` appeared this year, but it's still in `v0..` version and have not been yet fully acknowledged by the community. What is more it focuses only on the publications of Lindon et al. from [2022](https://openreview.net/pdf?id=a4zg0jiuVi) and [2024](https://arxiv.org/pdf/2210.08589), while there are other notable authors like [Zhao et al.](https://arxiv.org/pdf/1905.10493) and [Howard et al.](https://arxiv.org/abs/1810.08240) whose approach will be challenged in addition to Lindon's work

``` python
from seqabpy import gavi
```

`seqabpy` provides Always Valid Inference functionality in `gavi` module where as of now, `AlwaysValidInference` is a main class that implements confidence intervals valid at any point.
While intervals and namely their continuous comparison to the current z-score provides the apparatus that is just enough for practical decisions, `p-values` are to be released later as well, to complete experiment analysis picture.

`AlwaysValidInference` an array of sample sizes when the peeking happens along with the metric variance and the result point difference.
Multiple supported properties comprise different algorithms (the detailed description may be found in each docsrting) that return a boolean array indicating whether the null hypothesis is rejected in favour of one- or two- sided alternative for each size.

``` python
avi = gavi.AlwaysValidInference(size=np.arange(10, 100, 10), sigma2=1, estimate=1)
```

`GAVI` is the method proposed by Howard et al. and widely adopted in tech by Eppo

``` python
avi.GAVI(50)
```

    array([False,  True,  True,  True,  True,  True,  True,  True,  True])

`mSPRT` is the approach proposed by M. Lindon in his article and is leveraged by Netflix

``` python
avi.mSPRT(0.08)
```

    array([False, False,  True,  True,  True,  True,  True,  True,  True])

`StatSig_SPRT` is the variation proposed by Zhao et al. and as it comes from the name used currently by StatSig

``` python
avi.StatSig_SPRT()
```

    array([False,  True,  True,  True,  True,  True,  True,  True,  True])

The last and, this time, indeed least heavily criticized `statsig_alpha_corrected_v1` approach, which was their first attempt to furnish their platform with a sequential testing framework. It's mainly added for the reference to show how sequential testing must not work like

``` python
avi.statsig_alpha_corrected_v1(100)
```

    array([False, False, False,  True,  True,  True,  True,  True,  True])

# Simulations

For those who have visited my blog before, there is nothing new in how we will conduct testing, it is good old Monte Carlo. For more details checkout my previous posts like [Dunnett's Correction for ABC testing](https://npodlozhniy.github.io/posts/dunnett-correction/#canonical-ab-test)

We will measure False and True positive rates for two kinds of the target metric: a continuous variable and a conversion. Furthermore we will learn how tolerant are different methods to under- and over- sampling.

<details>
<summary>Code</summary>

``` python
# Global simulation settings
N = 500
alpha = 0.05
n_iterations = 100_000


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
        >>> detN([False, False, True, True], [50, 100, 150, 200])
        150
    """
    if len(is_significant) != len(sample_size):
        raise ValueError("Input arrays must have the same length.")
    w = np.where(is_significant)[0]
    return None if len(w) == 0 else sample_size[w[0]]
```

</details>

One thing about GST is that incredible freedom in spending function choice what makes it possible to experiment and find the best fit for your data.
For demonstration purposes I suggest using Kim-DeMets spending function with different values of the power $\phi$: the higher $\phi$ the more strict the function is at the beginning of the experiment.

<details>
<summary>Code</summary>

``` python
gst_linear = gatsby.GST(actual=10, expected=10, iuse=3, phi=1, alpha=alpha)
gst_quadratic = gatsby.GST(actual=10, expected=10, iuse=3, phi=2, alpha=alpha)
gst_cubic = gatsby.GST(actual=10, expected=10, iuse=3, phi=3, alpha=alpha)
```

</details>

You can play around the trade-off: would you like to spend more $\alpha$ at the start, detecting faster if there are greater uplifts in you experiment group or to preserve the major part of alpha until the end keeping maximum power to reject the hypothesis when the expected sample size is reached.

> **💡 Tip**
>
> If the title or the legend items are not visible to you - double click to one of legend items and it will make the chart rendered properly. It may happen due to LaTeX usage.

<details>
<summary>Code</summary>

``` python
import plotly.express as px
import plotly.graph_objs as go

def hex2rgba(hex, alpha):
    """
    Convert plotly hex colors to rgb and enables transparency adjustment
    """
    col_hex = hex.lstrip('#')
    col_rgb = tuple(int(col_hex[i : i + 2], 16) for i in (0, 2, 4))
    col_rgb += (alpha,)
    return 'rgba' + str(col_rgb)

def get_new_color(colors):
    while True:
        for color in colors:
            yield color

colors_list = px.colors.qualitative.Plotly
rgba_colors = [hex2rgba(color, alpha=0.5) for color in colors_list]
palette = get_new_color(rgba_colors)

def add_chart(figure, data, title=None):

    x = np.arange(1, len(data) + 1) / len(data)
    color = next(palette)

    figure.add_trace(
        go.Scatter(
            name=title,
            x=x,
            y=data,
            mode='lines',
            line=dict(color=color, width=4, dash='solid'),
            hovertemplate="%{y:.3f}"
        ),
    )


figure = go.Figure()

add_chart(figure, gst_linear, r"$\text{Linear: } \phi = 1$")
add_chart(figure, gst_quadratic, r"$\text{Quadratic: } \phi = 2$")
add_chart(figure, gst_cubic, r"$\text{Cubic: } \phi = 3$")

figure.update_xaxes(
    title_text="Peeking moments"
)

figure.update_layout(
    yaxis_title="Critical value for z-score",
    title={
        "x": 0.5,
        "text": r"$\text{Kim-DeMets spending function: } \alpha \cdot t^{\phi} \text{ differences}$",
    },
    hovermode="x",
    template="plotly_dark",
)

figure.write_json("alpha-spending-functions-comparison.json")
figure.show()
```

</details>

{{< plotly obj=alpha-spending-functions-comparison >}}

## Expan Flaw

Remember I promised to show, that `expan` way to determine boundaries is wrong, so here is a quick proof: the code is taken without changes from their GitHub: [zalando/expan/early_stopping](https://github.com/zalando/expan/blob/master/expan/core/early_stopping.py)

<details>
<summary>Code</summary>

``` python
from statsmodels.stats.proportion import proportion_confint


def sample_size(x):
    """ Calculates valid sample size given the data.

    :param x: sample to calculate the sample size
    :type  x: pd.Series or list (array-like)

    :return: sample size of the sample excluding nans
    :rtype: int
    """
    # cast into a dummy numpy array to infer the dtype
    x_as_array = np.array(x)

    if np.issubdtype(x_as_array.dtype, np.number):
        _x = np.array(x, dtype=float)
        x_nan = np.isnan(_x).sum()
    # assuming categorical sample
    elif isinstance(x, pd.core.series.Series):
        x_nan = x.str.contains('NA').sum()
    else:
        x_nan = list(x).count('NA')

    return int(len(x) - x_nan)

def obrien_fleming(information_fraction, alpha=0.05):
    """ Calculate an approximation of the O'Brien-Fleming alpha spending function.

    :param information_fraction: share of the information  amount at the point of evaluation,
                                 e.g. the share of the maximum sample size
    :type  information_fraction: float
    :param alpha: type-I error rate
    :type  alpha: float

    :return: redistributed alpha value at the time point with the given information fraction
    :rtype:  float
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


def group_sequential(x, y, spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    """ Group sequential method to determine whether to stop early.

    :param x: sample of a treatment group
    :type  x: pd.Series or array-like
    :param y: sample of a control group
    :type  y: pd.Series or array-like
    :param spending_function: name of the alpha spending function, currently supports only 'obrien_fleming'.
    :type  spending_function: str
    :param estimated_sample_size: sample size to be achieved towards the end of experiment
    :type  estimated_sample_size: int
    :param alpha: type-I error rate
    :type  alpha: float
    :param cap: upper bound of the adapted z-score
    :type  cap: int

    :return: results of type EarlyStoppingTestStatistics
    :rtype:  EarlyStoppingTestStatistics
    """

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)

    n_x = sample_size(_x)
    n_y = sample_size(_y)

    if not estimated_sample_size:
        information_fraction = 1.0
    else:
        information_fraction = min(1.0, (n_x + n_y) / estimated_sample_size)

    # alpha spending function
    if spending_function in ('obrien_fleming'):
        func = eval(spending_function)
    else:
        raise NotImplementedError
    alpha_new = func(information_fraction, alpha=alpha)

    # calculate the z-score bound
    bound = norm.ppf(1 - alpha_new / 2)
    # replace potential inf with an upper bound
    if bound == np.inf:
        bound = cap

    mu_x = np.nanmean(_x)
    mu_y = np.nanmean(_y)
    sigma_x = np.nanstd(_x)
    sigma_y = np.nanstd(_y)
    z = (mu_x - mu_y) / np.sqrt(sigma_x ** 2 / n_x + sigma_y ** 2 / n_y)

    if z > bound or z < -bound:
        stop = True
    else:
        stop = False

    return stop

fpr = 0

for r in range(n_iterations):

    x = np.random.normal(1, 1, N)
    y = np.random.normal(1, 1, N)

    for current_size in np.linspace(N/10, N, 10).astype(int):
        stopping = group_sequential(x[:current_size], y[:current_size], estimated_sample_size=2*N, alpha=0.05)
        if stopping:
            fpr += 1
            break

l, r = proportion_confint(count=fpr, nobs=n_iterations, alpha=0.10, method='wilson')
print(f"false positives: {fpr/n_iterations:.3f} ± {(r - l) / 2:.3f} is significantly higher than {alpha}")
```

</details>

	false positives: 0.070 ± 0.001 is significantly higher than 0.05

So, as was said above, it doesn't control FPR as it should according to Group Sequential Testing problem design and hence this myth of the direct application of alpha spending function have to be dispelled: it doesn't work this way and further you will see that it's not much better than custom ad-hoc corrections.

> **⚠️ Warning**
>
> Please, do not use `expan` for sequential testing as their implementation is wrong.

## Monte Carlo

<details>
<summary>Code</summary>

``` python
from collections import defaultdict


def monte_carlo(
    metric: str="normal",
    sampling: str="accurate",
    effect_size: float=0.10,
    aa_test: bool=True,
    N: int = N,
) -> pd.DataFrame:

    result = defaultdict(list)
    eff = 0 if aa_test else effect_size

    if metric == "normal":
        mu, sigma = 1, 1
    else:
        p = 0.10
        sigma = (p * (1 - p)) ** 0.5
        # for bernoulli rv sigma is less than for normal
        # so it's better to increase N to get similar power
        N *= int((sigma / p) ** 2)

    for _ in range(n_iterations):
        if metric == "normal":
            x = np.random.normal(mu, sigma, N)
            y = np.random.normal(mu+eff, sigma, N)
        else:
            x = np.random.choice(a=[0, 1], size=N, replace=True, p=[1 - p, p])
            y = np.random.choice(a=[0, 1], size=N, replace=True, p=[1 - p*(1+eff), p*(1+eff)])

        size = np.arange(1, N + 1)
        diff = (np.cumsum(y) / size) - (np.cumsum(x) / size)

        test = gavi.AlwaysValidInference(size=size, sigma2=sigma**2, estimate=diff, alpha=alpha)

        itermittent_analyses = np.linspace(N/10, N, 10).astype(int) - 1
        z_score = diff[itermittent_analyses] / np.sqrt(2 * sigma ** 2 / size[itermittent_analyses])

        result['No_Seq'].append(N if z_score[-1] > norm.ppf(1 - alpha) else None)

        if sampling == "accurate":

            result['GAVI'].append(stops_at(test.GAVI(), size))
            result['mSPRT'].append(stops_at(test.mSPRT(), size))
            result['StatSig_SPRT'].append(stops_at(test.StatSig_SPRT(), size))
            result['StatSig_v1'].append(stops_at(test.statsig_alpha_corrected_v1(), size))

            result['GST_linear'].append(stops_at(z_score > gst_linear, size[itermittent_analyses]))
            result['GST_quadratic'].append(stops_at(z_score > gst_quadratic, size[itermittent_analyses]))
            result['GST_cubic'].append(stops_at(z_score > gst_cubic, size[itermittent_analyses]))

        elif sampling == "undersampled":

            result['GAVI'].append(stops_at(test.GAVI(phi=N*7/5), size))
            # undersampling is the case, when the effect is larger than expected
            # so let's say effect ~ 7/5 times larger, 4 * (5/7)^2 ~ 2
            result['mSPRT'].append(stops_at(test.mSPRT(phi=2 * sigma**2 / diff**2), size))
            result['StatSig_SPRT'].append(stops_at(test.StatSig_SPRT(), size))
            result['StatSig_v1'].append(stops_at(test.statsig_alpha_corrected_v1(N=N*7/5), size))

            result['GST_linear'].append(stops_at(z_score > gst_linear_undersampled, size[itermittent_analyses]))
            result['GST_quadratic'].append(stops_at(z_score > gst_quadratic_undersampled, size[itermittent_analyses]))
            result['GST_cubic'].append(stops_at(z_score > gst_cubic_undersampled, size[itermittent_analyses]))

        elif sampling == "oversampled":

            result['GAVI'].append(stops_at(test.GAVI(phi=N*7/10), size))
            # oversmapling is the case, when the effect is lower than expected
            # so let's say effect ~ 7/10 times lower, 4 * (7/10)^2 ~ 8
            result['mSPRT'].append(stops_at(test.mSPRT(phi=8 * sigma**2 / diff**2), size))
            result['StatSig_SPRT'].append(stops_at(test.StatSig_SPRT(), size))
            result['StatSig_v1'].append(stops_at(test.statsig_alpha_corrected_v1(N=N*7/10), size))

            result['GST_linear'].append(stops_at(z_score > gst_linear_oversampled, size[itermittent_analyses]))
            result['GST_quadratic'].append(stops_at(z_score > gst_quadratic_oversampled, size[itermittent_analyses]))
            result['GST_cubic'].append(stops_at(z_score > gst_cubic_oversampled, size[itermittent_analyses]))

        else:
            raise ValueError("Unknown sampling method")

    # remove StatSig_v1 from Power comparison
    if not aa_test:
        result.pop('StatSig_v1')

    df = pd.DataFrame(result).agg(["count", "median"]).T.assign(
          PositiveRate=lambda x: (x["count"] / n_iterations).round(3)
        ).assign(
            SampleSize=lambda x: x["median"].astype(int)
        )[["PositiveRate", "SampleSize"]]

    return df


def plot_positive_rate(
    df: pd.DataFrame,
    aa_test: bool=True,
    sampling: str=None
):

    fig = go.Figure()

    if aa_test:
        error_const = round(3 * (alpha * (1 - alpha) / n_iterations) ** 0.5, 3)
    else:
        error_array = round(3 * (df["PositiveRate"] * (1 - df["PositiveRate"]) / n_iterations) ** 0.5, 3)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["PositiveRate"],
        marker_color=next(palette),
        error_y=dict(type='constant', value=error_const) if aa_test else dict(type='data', array=error_array),
    ))

    if aa_test:
        fig.add_hline(
            y=0.05,
            line_dash="dot",
            annotation_text="designed Type I error rate",
            annotation_position="top right"
        )

    title = (
        f"{'Correctness' if aa_test else 'Power'} of"
        f"{' ' + sampling if sampling else ''} Sequential Testing Design"
    )
    
    fig.update_layout(
        yaxis_title=f"{str(not aa_test)} Positive Rate",
        title={
            "x": 0.5,
            "text": title,
        },
        hovermode="x",
        template="plotly_dark",
    )

    fig.write_json(f"{title.replace(' ', '-').lower()}.json")

    fig.show()
```

</details>

### Continuous Variable

As you can see for GST bounds are pre-calculated for the necessary intermittent analyses number that were expected to and in fact take place.
We calculate bounds for 10 intermittent analyses scenario, in addition considering over- and under- sampling designs.

<details>
<summary>Code</summary>

``` python
gst_linear_undersampled = gatsby.GST(actual=10, expected=14, iuse=3, phi=1, alpha=alpha)
gst_quadratic_undersampled = gatsby.GST(actual=10, expected=14, iuse=3, phi=2, alpha=alpha)
gst_cubic_undersampled = gatsby.GST(actual=10, expected=14, iuse=3, phi=3, alpha=alpha)

gst_linear_oversampled = gatsby.GST(actual=10, expected=7, iuse=3, phi=1, alpha=alpha)
gst_quadratic_oversampled = gatsby.GST(actual=10, expected=7, iuse=3, phi=2, alpha=alpha)
gst_cubic_oversampled = gatsby.GST(actual=10, expected=7, iuse=3, phi=3, alpha=alpha)
```

</details>

#### False Positives

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=True)
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.051        | 500        |
| GAVI          | 0.018        | 221        |
| mSPRT         | 0.048        | 38         |
| StatSig_SPRT  | 0.026        | 43         |
| StatSig_v1    | 0.074        | 421        |
| GST_linear    | 0.051        | 300        |
| GST_quadratic | 0.050        | 400        |
| GST_cubic     | 0.051        | 400        |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=True)
```

</details>

{{< plotly obj=correctness-of-sequential-testing-design >}}

As it immediately comes clear: StatSig v1 correction was a flaw, all the other methods are targeting $\alpha$ as needed, however out of AVI it's only mSPRT that gives high enough level, the rest of them make fewer false positives what usually is a sign of lower statistical power, we will see it later.

<details>
<summary>Code</summary>

``` python
monte_carlo(aa_test=True, sampling="undersampled")
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.050        | 500        |
| GAVI          | 0.014        | 253        |
| mSPRT         | 0.043        | 38         |
| StatSig_SPRT  | 0.026        | 41         |
| StatSig_v1    | 0.015        | 451        |
| GST_linear    | 0.047        | 350        |
| GST_quadratic | 0.046        | 500        |
| GST_cubic     | 0.045        | 500        |

</div>
<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=True, sampling="oversampled")
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.051        | 500        |
| GAVI          | 0.021        | 189        |
| mSPRT         | 0.046        | 35         |
| StatSig_SPRT  | 0.027        | 36         |
| StatSig_v1    | 0.187        | 378        |
| GST_linear    | 0.065        | 250        |
| GST_quadratic | 0.072        | 300        |
| GST_cubic     | 0.077        | 350        |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=True, sampling="oversampled")
```

</details>

{{< plotly obj=correctness-of-oversampled-sequential-testing-design >}}

1.  Over-sampling is a tough cookie for `GST`, in such a case GST doesn't work correctly, it inflates Type I error, so it's important to note the difference here between `AVI` and `GST`, the latter one is not designed to handle over-sampling
2.  StatSig distinguished itself: their v1 version suffers more than any other method form both under- and over- sampling, while on the other flip their SPRT implementation is totally resistant to under- and over- sampling and if identifies the positive, it does it quickly, most likely it will be underpowered though.
3.  As of now mSPRT seems to be the best choice as it identifies the differences so fast and just a little less often than it should.

#### True Positives

It's time to compare the power of different methods, I'm not going to consider StatSig Alpha corrected version anymore as it's not a valid procedure

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=False)
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.474        | 500        |
| GAVI          | 0.222        | 285        |
| mSPRT         | 0.268        | 202        |
| StatSig_SPRT  | 0.188        | 230        |
| GST_linear    | 0.409        | 300        |
| GST_quadratic | 0.445        | 350        |
| GST_cubic     | 0.459        | 400        |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=False)
```

</details>

{{< plotly obj=power-of-sequential-testing-design >}}

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=False, sampling="undersampled")
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.478        | 500        |
| GAVI          | 0.211        | 306        |
| mSPRT         | 0.253        | 210        |
| StatSig_SPRT  | 0.190        | 233        |
| GST_linear    | 0.425        | 350        |
| GST_quadratic | 0.449        | 450        |
| GST_cubic     | 0.455        | 500        |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=False, sampling="undersampled")
```

</details>

{{< plotly obj=power-of-undersampled-sequential-testing-design >}}

<details>
<summary>Code</summary>

``` python
monte_carlo(aa_test=False, sampling="oversampled")
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.476        | 500        |
| GAVI          | 0.236        | 268        |
| mSPRT         | 0.263        | 208        |
| StatSig_SPRT  | 0.190        | 233        |
| GST_linear    | 0.443        | 300        |
| GST_quadratic | 0.496        | 300        |
| GST_cubic     | 0.519        | 350        |

</div>

So, as it comes from bar chart and tables:

1.  all `AVI` (including over-sampled options) are way weaker than even under-sampled GST, so power-wise `GST` is an unconditional winner
2.  it's appealing that for under-sampled `GST` the power has just a subtle decline, and even then only for strict spending function (Cubic), providing an increase for permissive spending function (Linear)
3.  Although if `AVI` rejects null hypothesis it does quicker (the required Sample Size is smaller) than `GST` on average

### Conversion Rate

In addition to continuous measure, let's consider ratio variable, how the methods work with conversions

#### False Positives

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=True, metric="choice")
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.050        | 4500       |
| GAVI          | 0.018        | 1913       |
| mSPRT         | 0.072        | 88         |
| StatSig_SPRT  | 0.044        | 60         |
| StatSig_v1    | 0.074        | 3796       |
| GST_linear    | 0.050        | 2700       |
| GST_quadratic | 0.049        | 3600       |
| GST_cubic     | 0.049        | 3600       |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=True, sampling="ratio")
```

</details>

{{< plotly obj=correctness-of-ratio-sequential-testing-design >}}

As you see, in addition to StatSig v1 Alpha Correction which is again an outsider, mSPRT approximation is not good enough for Bernoulli random variable, for conversions it's another approach that shall be applied, `savvi` package might come handy here as it the main purpose the that library - to work with inhomogeneous Bernoulli or Poisson process.
Alternatively, you may use `sequential_p_value` function from `gavi` module of `seqabpy`, it's a valid procedure following the algorithm defined by M. Lindon and A. Malek in [Anytime-Valid Inference For Multinomial Count Data (2022)](https://openreview.net/pdf?id=a4zg0jiuVi), could be a little less powerful though than `savvi` implementation that follows even more recent articles.

<details>
<summary>Code</summary>

``` python
expected_probs = [0.5, 0.5]

# it's an asymptotic algorithm, so only numerators are compared
# assuming the denominators of convesrion are similar like in fair A/B test
actual_counts = [156, 212]

print(f"AVI p-value for Conversion: {gavi.sequential_p_value(actual_counts, expected_probs):.3f}")
```

</details>

    AVI p-value for Conversion: 0.075

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=True, metric="choice", sampling="oversampled")
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.051        | 4500       |
| GAVI          | 0.022        | 1620       |
| mSPRT         | 0.069        | 85         |
| StatSig_SPRT  | 0.045        | 61         |
| StatSig_v1    | 0.189        | 3377       |
| GST_linear    | 0.064        | 2250       |
| GST_quadratic | 0.072        | 2700       |
| GST_cubic     | 0.076        | 3150       |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=True, sampling="oversampled ratio")
```

</details>

{{< plotly obj=correctness-of-oversampled-ratio-sequential-testing-design >}}

This chart above is just to assure you, that for conversions over-sampled `GST` doesn't work neither, I can't help but prove that `GST` in oversampling design is a flaw, while yet much better than Statsig v1 Alpha Corrections.

#### True Positives

Let's take a brief look at the power comparison for a couple different effect sizes

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=False, metric="choice", effect_size=0.10)
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.480        | 4500       |
| GAVI          | 0.240        | 2482       |
| mSPRT         | 0.310        | 1441       |
| StatSig_SPRT  | 0.226        | 1702       |
| GST_linear    | 0.420        | 2700       |
| GST_quadratic | 0.449        | 3150       |
| GST_cubic     | 0.460        | 3600       |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=False, sampling="ratio")
```

</details>

{{< plotly obj=power-of-ratio-sequential-testing-design >}}

<details>
<summary>Code</summary>

``` python
df = monte_carlo(aa_test=False, metric="choice", effect_size=0.2)
df
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

|               | PositiveRate | SampleSize |
|---------------|--------------|------------|
| No_Seq        | 0.930        | 4500       |
| GAVI          | 0.767        | 2127       |
| mSPRT         | 0.796        | 1593       |
| StatSig_SPRT  | 0.718        | 1923       |
| GST_linear    | 0.898        | 2250       |
| GST_quadratic | 0.915        | 2250       |
| GST_cubic     | 0.921        | 2700       |

</div>
<details>
<summary>Code</summary>

``` python
plot_positive_rate(df, aa_test=False, sampling="strong effect")
```

</details>

{{< plotly obj=power-of-strong-effect-sequential-testing-design >}}

With the growing effect size the relative difference in power is getting lower, but you can check that with any kind of reasonable effect size, `GST` outperforms `AVI` and what is more even for conversion variable, where `mSPRT` method doesn't really control Type I error rate, it's less powerful than `GST` after all.

# Conclusion

Generally speaking, I'd rather say that `GST` is yet the best framework for sequential testing, despite all the recent publications on cutting-edge `AVI` variations.

However, I have to make a clause: while `AVI` is noticeably less powerful, it's perfect to work in a streaming manner for guardrail metrics, while GST is better for target metrics within you AB test.

> **💡 Practical Tip**
>
> Combining these methodologies you may set up robust Sequential Testing framework, gaining from both: quick detection of major deterioration in your product with `AVI` and reliable uplifts discoveries in your decisive metrics with the most powerful `GST` procedure.

Another important point is to be conscious about the choice of the specific version of the algorithms that you will use.

For instance running an under-sampled experiments where `GST` with strict alpha spending functions, like Cubic, applied is less preferable, under-sampling works better with permissive spending functions, as well as over-sampling with Cubic spending is worse as it inflates $\alpha$ more.

> **💡 General Rule**
>
> The more permissive spending function is the faster effect is identified, but the less power at the end of experiment is achieved, what is especially striking for less substantial effect sizes.

Rounding this extensive blog-post up, here are the recommendations on choosing the sequential testing framework wrapped up into a single decision tree:

<!-- draw.io diagram -->
<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;lightbox&quot;:false,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;ac.draw.io\&quot; agent=\&quot;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36\&quot; version=\&quot;25.0.3\&quot;&gt;\n  &lt;diagram name=\&quot;Page-1\&quot; id=\&quot;10a91c8b-09ff-31b1-d368-03940ed4cc9e\&quot;&gt;\n    &lt;mxGraphModel dx=\&quot;1985\&quot; dy=\&quot;1050\&quot; grid=\&quot;1\&quot; gridSize=\&quot;10\&quot; guides=\&quot;1\&quot; tooltips=\&quot;1\&quot; connect=\&quot;1\&quot; arrows=\&quot;1\&quot; fold=\&quot;1\&quot; page=\&quot;1\&quot; pageScale=\&quot;1\&quot; pageWidth=\&quot;1100\&quot; pageHeight=\&quot;850\&quot; background=\&quot;none\&quot; math=\&quot;0\&quot; shadow=\&quot;0\&quot;&gt;\n      &lt;root&gt;\n        &lt;mxCell id=\&quot;0\&quot; /&gt;\n        &lt;mxCell id=\&quot;1\&quot; parent=\&quot;0\&quot; /&gt;\n        &lt;mxCell id=\&quot;BCZGtb4ARAkY1Pyahnf2-1\&quot; value=\&quot;\&quot; style=\&quot;rounded=1;whiteSpace=wrap;html=1;strokeColor=#9B9C9D;glass=0;shadow=0;fillColor=#9B9C9D;\&quot; vertex=\&quot;1\&quot; parent=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;60\&quot; y=\&quot;10\&quot; width=\&quot;950\&quot; height=\&quot;830\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-1\&quot; value=\&quot;Data is supplied in a streaming manner\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;590\&quot; y=\&quot;30\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-2\&quot; value=\&quot;Sample size unknown\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;350\&quot; y=\&quot;140\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-5\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0.108;exitY=1;exitDx=0;exitDy=0;exitPerimeter=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-1\&quot; target=\&quot;62893188c0fa7362-2\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.2051\&quot; y=\&quot;-17\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-8\&quot; value=\&quot;Only upper bound for observations number is known\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;110\&quot; y=\&quot;410\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-9\&quot; value=\&quot;There are many intermittent analyses (&amp;amp;gt;30)&amp;amp;nbsp;\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;590\&quot; y=\&quot;250\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-10\&quot; value=\&quot;&amp;lt;span&amp;gt;Always Valid Inference (AVI)&amp;lt;/span&amp;gt;\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=0;shadow=1;labelBackgroundColor=none;strokeWidth=2;fontFamily=Verdana;fontSize=12;align=center;fillColor=#dae8fc;strokeColor=#6c8ebf;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;850\&quot; y=\&quot;140\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-13\&quot; value=\&quot;Large-scale effect is expected from the experiment\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;110\&quot; y=\&quot;570\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-14\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;eBBfYNGfGk0e3oGPA11c-2\&quot; target=\&quot;62893188c0fa7362-8\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;0.1111\&quot; y=\&quot;-20\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-15\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=1;exitY=1;exitDx=0;exitDy=0;entryX=0;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-2\&quot; target=\&quot;62893188c0fa7362-9\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.2\&quot; y=\&quot;14\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-16\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;endArrow=classic;endFill=1;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0.933;exitY=1.017;exitDx=0;exitDy=0;exitPerimeter=0;strokeColor=default;entryX=0;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-1\&quot; target=\&quot;62893188c0fa7362-10\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.0325\&quot; y=\&quot;11\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint x=\&quot;1\&quot; as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;810\&quot; y=\&quot;220\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n            &lt;mxPoint x=\&quot;880\&quot; y=\&quot;110\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-19\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-8\&quot; target=\&quot;62893188c0fa7362-13\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.2857\&quot; y=\&quot;-20\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;190\&quot; y=\&quot;570\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-21\&quot; value=\&quot;&amp;lt;span&amp;gt;Classic GST with permissive alpha spending function&amp;lt;/span&amp;gt;\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=0;shadow=1;labelBackgroundColor=none;strokeWidth=2;fontFamily=Verdana;fontSize=12;align=center;fillColor=#d5e8d4;strokeColor=#82b366;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;390\&quot; y=\&quot;490\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-22\&quot; value=\&quot;Classic GST&amp;lt;br&amp;gt;with strict alpha spending function\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=0;shadow=1;labelBackgroundColor=none;strokeWidth=2;fontFamily=Verdana;fontSize=12;align=center;fillColor=#d5e8d4;strokeColor=#82b366;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;110\&quot; y=\&quot;730\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-25\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-8\&quot; target=\&quot;62893188c0fa7362-21\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;0.1373\&quot; y=\&quot;16\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;245.71428571428578\&quot; y=\&quot;605\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-26\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-13\&quot; target=\&quot;62893188c0fa7362-22\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;0.0033\&quot; y=\&quot;-12\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;62893188c0fa7362-27\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;entryX=0;entryY=1;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-13\&quot; target=\&quot;62893188c0fa7362-21\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;0.1373\&quot; y=\&quot;16\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;250\&quot; y=\&quot;600\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;f_bxW6br7S2beZE4IyPl-2\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;entryX=0;entryY=1;entryDx=0;entryDy=0;exitX=1.025;exitY=0.133;exitDx=0;exitDy=0;exitPerimeter=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-9\&quot; target=\&quot;62893188c0fa7362-10\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.0325\&quot; y=\&quot;11\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint x=\&quot;1\&quot; as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;710\&quot; y=\&quot;270\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n            &lt;mxPoint x=\&quot;780\&quot; y=\&quot;300\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;f_bxW6br7S2beZE4IyPl-4\&quot; value=\&quot;&amp;lt;span&amp;gt;Simple Bonferroni Correction&amp;lt;/span&amp;gt;\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=0;shadow=1;labelBackgroundColor=none;strokeWidth=2;fontFamily=Verdana;fontSize=12;align=center;fillColor=#e1d5e7;strokeColor=#9673a6;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;850\&quot; y=\&quot;360\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;f_bxW6br7S2beZE4IyPl-5\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=1;exitY=1;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-9\&quot; target=\&quot;f_bxW6br7S2beZE4IyPl-4\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.2\&quot; y=\&quot;-14\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;740\&quot; y=\&quot;400\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n            &lt;mxPoint x=\&quot;670\&quot; y=\&quot;470\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;eBBfYNGfGk0e3oGPA11c-2\&quot; value=\&quot;Is there a need to run a test for an arbitrary long time\&quot; style=\&quot;whiteSpace=wrap;html=1;rounded=1;shadow=1;labelBackgroundColor=none;strokeWidth=1;fontFamily=Verdana;fontSize=12;align=center;\&quot; parent=\&quot;1\&quot; vertex=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;110\&quot; y=\&quot;250\&quot; width=\&quot;120\&quot; height=\&quot;60\&quot; as=\&quot;geometry\&quot; /&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;eBBfYNGfGk0e3oGPA11c-3\&quot; value=\&quot;No\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=0;exitY=1;exitDx=0;exitDy=0;entryX=1;entryY=0;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;62893188c0fa7362-2\&quot; target=\&quot;eBBfYNGfGk0e3oGPA11c-2\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.2\&quot; y=\&quot;-14\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;470\&quot; y=\&quot;245\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n            &lt;mxPoint x=\&quot;340\&quot; y=\&quot;460\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n        &lt;mxCell id=\&quot;eBBfYNGfGk0e3oGPA11c-4\&quot; value=\&quot;Yes\&quot; style=\&quot;rounded=0;html=1;labelBackgroundColor=none;startArrow=none;startFill=0;startSize=5;endArrow=classic;endFill=1;endSize=5;jettySize=auto;orthogonalLoop=1;strokeWidth=1;fontFamily=Verdana;fontSize=12;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;\&quot; parent=\&quot;1\&quot; source=\&quot;eBBfYNGfGk0e3oGPA11c-2\&quot; target=\&quot;62893188c0fa7362-9\&quot; edge=\&quot;1\&quot;&gt;\n          &lt;mxGeometry x=\&quot;-0.027\&quot; y=\&quot;20\&quot; relative=\&quot;1\&quot; as=\&quot;geometry\&quot;&gt;\n            &lt;mxPoint as=\&quot;offset\&quot; /&gt;\n            &lt;mxPoint x=\&quot;480\&quot; y=\&quot;210\&quot; as=\&quot;sourcePoint\&quot; /&gt;\n            &lt;mxPoint x=\&quot;600\&quot; y=\&quot;275\&quot; as=\&quot;targetPoint\&quot; /&gt;\n          &lt;/mxGeometry&gt;\n        &lt;/mxCell&gt;\n      &lt;/root&gt;\n    &lt;/mxGraphModel&gt;\n  &lt;/diagram&gt;\n&lt;/mxfile&gt;\n&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

# References

 [`seqabpy`](https://github.com/NPodlozhniy/seqabpy) is an open source library that is perfect for `GST` and `AVI` in Python. In addition to implemented functionality it contains all the referenced original papers in functions' docstrings, so you may get acquainted with the original works.

There were similar posts made by Booking and Spotify, however they do not share implementation details, hence you may read it to deepen the understanding, but barely can apply it in practice:

 - [Choosing a Sequential Testing Framework by Spotify](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions/)
 - [Sequential Testing at Booking](https://booking.ai/sequential-testing-at-booking-com-650954a569c7)
 