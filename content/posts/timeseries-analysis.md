---
title: "Time Series Analysis in Python"
author: "Nikita Podlozhniy"
date: "2023-09-02"
format:
    hugo-md:
        output-file: "timeseries-analysis.md"
        html-math-method: katex
        code-fold: true
jupyter: python3
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>
<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.25.2.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>



# Intro

It's a short yet handy guide on how to analyze time series data in a few clicks using modern Python libraries.

Tech stack for this guide consists of
- `duckdb` and `pandas` packages for data processing
- Facebook's `prophet` for data exploration and forecasting
- `plotly` for data visualization

<details>
<summary>Code</summary>

``` python
import duckdb as db
import pandas as pd
import numpy as np

orders = pd.read_csv('orders_data.csv')
orders.sample()
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

|       | Created Date | Country  | City   | Restaurant ID | Restaurant Name       | Order State | Cancel Reason | Cuisine       | Platform | Products in Order | Order Value € (Gross) | Delivery Fee | Delivery Time | Order ID  |
|-------|--------------|----------|--------|---------------|-----------------------|-------------|---------------|---------------|----------|-------------------|-----------------------|--------------|---------------|-----------|
| 44255 | 10.01.2020   | Portugal | Lisbon | 6167          | H3 Armazéns do Chiado | delivered   | NaN           | Mediterranean | ios      | 1                 | €9.55                 | 0.0          | 14.7          | 379143844 |

</div>

## Are there any discernible patterns or seasonality present?

Usually it's good to start any time series analysis with tackling four key questions:

-   Is there any trend?
-   Is there seasonality, if so which type: additive or multiplicative?
-   Is there any explicit change points, which we should consider?
-   Are there outliers? - Get rid of them!

### Data

If you never experienced `duckdb` before, I totally recommend you to try as it boosts your efficiency allowing simultaneous usage of best features from Python and SQL

<details>
<summary>Code</summary>

``` python
df = db.query("""
    SELECT strptime("Created Date", '%d.%m.%Y') as date
        ,"City" as city
        ,count(*) as orders
    FROM
            orders o
    WHERE "Order State" = 'delivered'
    GROUP BY 1, 2
""").to_df()

pivot = df.pivot(index='date', columns='city', values='orders')
pivot.head()
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

| city       | Accra | Lisbon |
|------------|-------|--------|
| date       |       |        |
| 2020-01-01 | 124   | 882    |
| 2020-01-02 | 126   | 1104   |
| 2020-01-03 | 76    | 1110   |
| 2020-01-04 | 169   | 784    |
| 2020-01-05 | 198   | 788    |

</div>
<details>
<summary>Code</summary>

``` python
print(f"We have data for {pivot.shape[0]} consecutive days")
```

</details>

    We have data for 59 consecutive days

### Libraries

I'm using my own time series model wrappers to explore and forecast the data, they are based upon popular libraries from big tech companies, in this particular case, on [Facebook Prophet](https://facebook.github.io/prophet/)

<details>
<summary>Code</summary>

``` python
import datetime
import re

from typing import Optional, List, Tuple

from plotly import express, graph_objects
from plotly.subplots import make_subplots

from prophet import Prophet as FP
from prophet.utilities import regressor_coefficients

from sklearn.preprocessing import MinMaxScaler
```

</details>
<details>
<summary>Code</summary>

``` python
class ModelWrapper:
    """
    Custom Wrapper for popular models for TimeSeries analysis, including but not limited to:
        - Google Causal Impact https://pypi.org/project/causalimpact/
        - Facebook Prophet https://facebook.github.io/prophet/

    Parameters
    ----------
    df: DataFrame containg target variable with `y` name and time frame variable with `date` name
    start_date: the start date of experiment with the format YYYY-MM-DD
    test_days: the number of last days in which the experiment was run
    end_date: the end date of experiment with the format YYYY-MM-DD
    date: timeframe column name
    y: target column name

    Attributes
    ----------
    explore: the chart exploring pre-experiment correlations
    show: visualization of the trained model, please notet that you need to run the model first
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        test_days: Optional[int] = None,
        end_date: Optional[str] = None,
        date: str = "date",
        y: str = "y"
    ) -> None:

        self.df = df.copy()
        self.date = date
        self.y = y

        if start_date:
            self.start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        elif test_days:
            if pd.core.dtypes.common.is_datetime64_dtype(self.df[self.date].dtype):
                self.start_dt = self.df[self.date].max().date() - datetime.timedelta(days=test_days)
            else:
                self.start_dt = datetime.datetime.strptime(self.df[self.date].max(), "%Y-%m-%d").date() - datetime.timedelta(days=test_days)
        else:
            raise ValueError("You must specify start_date or test_days variable")

        self.end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
        self.x = list(self.df.columns.difference([self.date, self.y]).values)
        self._preprocess()

    def _preprocess(self):
        self.df[self.date] = pd.to_datetime(self.df[self.date]).dt.date
        self.df = self.df[[self.y, self.date] + self.x].set_index(self.date).sort_index()
        for column in self.df.columns[self.df.dtypes == "object"]:
            try:
                self.df[column] = self.df[column].astype(float)
            except ValueError:
                raise ValueError("All DataFrame columns except Date must be numeric")

    @staticmethod
    def _save(figure, title: str) -> None:
        figure.write_html(
            re.sub('[-:<>|\/\*\?\"\\\\ ]+', '_',  title.lower()) + ".html"
        )
    
    def explore(
        self,
        scale: bool = True,
        title: str = "pre-experiment correlation",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> None:
        """
        Plot the dynamic of pre-experiment correlation

        Parameters
        ----------
        scale: whether the samples should be scaled to [0, 1] interval
        title: the title of the chart
        x_label: label for X axis
        y_label: label for Y axis
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """

        data = self.df.copy()

        if scale:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[data.columns] = scaler.fit_transform(data)
        
        corr = data[data.index < self.start_dt].corr().iloc[0, 1:]
        
        data = pd.melt(
            data,
            value_vars=list(data.columns),
            var_name="variable",
            value_name="value",
            ignore_index=False,
        )

        chart_title = title + f"""<br><sup>{', '.join(
            [f"{feature}: {round(value, 2)}"
            for feature, value in zip(self.x, corr)]
        )}</sup></br>"""

        figure = express.line(
            data,
            x=data.index,
            y="value",
            color="variable",
            height=height,
            width=width,
            title=chart_title,
        )

        figure.add_vline(x=self.start_dt, line_width=2, line_dash="dash", line_color="white")
        if self.end_dt:
            figure.add_vline(x=self.end_dt, line_width=2, line_dash="dash", line_color="white")
        
        figure.update_traces(hovertemplate="%{y}")
        
        figure.update_xaxes(title_text=x_label if x_label else "Date")
        figure.update_yaxes(title_text=y_label if y_label else "Scaled Axis" if scale else "Original Axis")
        
        figure.update_layout(
            title={
                "x": 0.5,
            },
            legend={
                "x": 0.05,
                "y": 1.05,
                "orientation": "h",
                "title": None
            },
            hovermode="x",
            template="plotly_dark",
            xaxis=dict(hoverformat="%a, %b %d, %Y"),
        )

        if save:
            self._save(figure, title)     
            
        figure.show()

    @staticmethod
    def _add_chart(
        figure,
        data: pd.DataFrame,
        titles: list,
        y: str,
        name: str,
        row: int,
        actual: bool = False,
        y_label: Optional[str] = None,
    ) -> None:
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y],
                name=name,
                hovertemplate="%{y}",
                line={"color": "white"},
                legendgroup=f"{row}",
                legendgrouptitle={"text": titles[row]},
                connectgaps=True,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y + "_upper"],
                name="Upper bound",
                hovertemplate="%{y}",
                line={"color": "deepskyblue", "width": 0.5},
                legendgroup=f"{row}",
                connectgaps=True,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y + "_lower"],
                name="Lower bound",
                hovertemplate="%{y}",
                fill="tonexty",
                line={"color": "deepskyblue", "width": 0.5},
                legendgroup=f"{row}",
                connectgaps=True,
            ),
            row=row,
            col=1,
        )

        figure.update_yaxes(title_text="" if not y_label else "% Effect" if row == 4 else y_label)

        if actual:
            figure.add_trace(
                graph_objects.Scatter(
                    x=data.index,
                    y=data["response"],
                    name="Actual",
                    hovertemplate="%{y}",
                    line={"color": "red"},
                    legendgroup=f"{row}",
                    connectgaps=True,
                ),
                row=row,
                col=1,
            )
        else:
            figure.add_hline(y=0, line_width=1, line_color="white", row=row, col=1)        

    def show(
        self,
        keep_n_prior_days: Optional[int] = None,
        title: str = "Causal Impact",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> None:
        """
        Plot the trained model results

        Parameters
        ----------
        keep_n_prior_days: specify the exact number of pre-experiment days you want to keep, skip if you want to show all the time frame
        title: the title of the chart
        x_label: label for X axis
        y_label: label for Y axis
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """
        try:
            if keep_n_prior_days:
                data = self.result[
                    self.result.index > self.start_dt - datetime.timedelta(days=keep_n_prior_days)
                ]
            else:
                data = self.result.iloc[1:]
        except AttributeError:
            raise AttributeError("To show the results run the model first, use .run() method")
        
        titles = [
            "Model Overview",
            "Actual vs Expected",
            "Effect Size: Actual - Expected",
            "Cumulative Effect",
            "Effect Relative to Expected",
        ]

        if isinstance(self, CausalImpact):
            figure = make_subplots(
                rows=4, cols=1, shared_xaxes=True, subplot_titles=titles[1:]
            )
            for y, name, row in zip(
                ["point_pred", "point_effect", "cum_effect", "rel_effect"],
                ["Expected Values", "Effect Size", "Cumulative Effect", "Relative Effect"],
                range(1, 5)
            ):
                self._add_chart(
                    figure,
                    data,
                    titles,
                    y=y,
                    name=name,
                    row=row,
                    actual=(row == 1),
                    y_label=y_label
                )
        elif isinstance(self, Prophet):
            row = 1
            figure = make_subplots()
            self._add_chart(
                figure,
                data,
                titles,
                y="yhat",
                name="Expected Value",
                row=row,
                actual=True,
                y_label=y_label
            )
        
        figure.update_xaxes(title_text=x_label if x_label else "Date", row=row, col=1)
    
        figure.add_vline(x=self.start_dt, line_width=2, line_dash="dash", line_color="white")
        if self.end_dt:
            figure.add_vline(x=self.end_dt, line_width=2, line_dash="dash", line_color="white")

        figure.update_layout(
            title={
                "x": 0.5,
                "text": title,
            },
            width=width,
            height=height,
            hovermode="x",
            template="plotly_dark",
            legend={
                "x": 0.0,
                "y": -0.2,
                "orientation": "h",
                "groupclick": "toggleitem",
                "traceorder": "grouped",
            },
            xaxis=dict(hoverformat="%a, %b %d, %Y"),
        )
        
        if save:
            self._save(figure, title)
        
        figure.show()
```

</details>
<details>
<summary>Code</summary>

``` python
class CausalImpact(ModelWrapper):

    def run(
        self,
        nseasons: int = 7,
        season_duration: int = 1,
        alpha: float = 0.05,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run causal impact analysis

        Parameters
        ----------
        nseasons: Period of the seasonal components.
            In order to include a seasonal component, set this to a whole number greater than 1.
            For example, if the data represent daily observations, use 7 for a day-of-week component.
            This interface currently only supports up to one seasonal component.
        season_duration: Duration of each season, i.e., number of data points each season spans.
            For example, to add a day-of-week component to data with daily granularity, use model_args = list(nseasons = 7, season_duration = 1).
            To add a day-of-week component to data with hourly granularity, set model_args = list(nseasons = 7, season_duration = 24).
        alpha : Desired tail-area probability for posterior intervals. Defaults to 0.05, which will produce central 95% intervals

        Other Parameters
        ----------------
        **kwargs : model_args variables, available options:
            ndraws: number of MCMC samples to draw.
                More samples lead to more accurate inferences. Defaults to 1000.
            nburn: number of burn in samples.
                This specifies how many of the initial samples will be discarded. defaults to 10% of ndraws.
            standardize_data: whether to standardize all columns of the data before fitting the model.
                This is equivalent to an empirical Bayes approach to setting the priors.
                It ensures that results are invariant to linear transformations of the data.
            prior_level_sd: prior standard deviation of the Gaussian random walk of the local level.
                Expressed in terms of data standard deviations. Defaults to 0.01.
                A typical choice for well-behaved and stable datasets with low residual volatility after regressing out known predictors.
                When in doubt, a safer option is to use 0.1, as validated on synthetic data,
                although this may sometimes give rise to unrealistically wide prediction intervals.
            dynamic_regression: whether to include time-varying regression coefficients.
                In combination with a time-varying local trend or even a time-varying local level,
                this often leads to overspecification, in which case a static regression is safer. Defaults to FALSE.
        """
        
        data = self.df.copy().reset_index()
        prior = [
            data.index.min(),
            int(data[data[self.date] < self.start_dt].index.max())
        ]
        posterior = [
            int(data[data[self.date] >= (self.end_dt if self.end_dt else self.start_dt)].index.min()),
            data.index.max()
        ]
        data.drop(columns=[self.date], inplace=True)
        self.ci = CI(
            data,
            prior,
            posterior,
            model_args={
                "nseasons": nseasons,
                "season_duration": season_duration,
                **kwargs
            },
            alpha=alpha,
        )
        self.ci.run()

    def summary(self, format: str = "summary") -> None:
        """
        Print the summary for Causal Impact Analysis model

        Parameters
        ----------
        format: can be 'summary' to return a table or 'report' to return a natural language description
        """
        try:
            self.ci.summary(format)
        except AttributeError:
            raise AttributeError("To get the summary run the model first, use .run() method")
        self.result = self.ci.inferences.set_index(self.df.index)
        for suffix in ["", "_lower", "_upper"]:
            self.result["rel_effect" + suffix] = self.result["point_effect" + suffix] / self.result["point_pred"]
```

</details>
<details>
<summary>Code</summary>

``` python
class Prophet(ModelWrapper):

    def run(
        self,
        growth: str = "linear",
        weekly_seasonality: bool = True,
        monthly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "additive",
        country_holidays: Optional[str] = None,
        outliers: Optional[List[Tuple[str]]] = None,
        floor: Optional[int] = None,
        cap: Optional[int] = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run time-series forecasting

        Parameters
        ----------
        growth: String 'linear', 'logistic' or 'flat' to specify a linear, logistic or flat trend.
        weekly_seasonality: Fit weekly seasonality.
        monthly_seasonality: Fit monthly seasonality.
        yearly_seasonality: Fit yearly seasonality.
        seasonality_mode: 'additive' (default) or 'multiplicative'.
        country_holidays: country code (e.g. 'RU') of the country whose holiday are to be considered
        outliers: list of time intervals (start date, end date) with the format YYYY-MM-DD where there are outliers
        floor: minimum allowed value for the target variable. It's particulary useful with "logistic" growth type
        cap: maximum allowed value for the target variable. It's particulary useful with "logistic" growth type
        alpha: 1 - width of the uncertainty intervals provided for the forecast.

        Other Parameters
        ----------------
        **kwargs : model_args variables, to reveal the whole list follow the Prophet documentation, for example:
            n_changepoints: Number of potential changepoints to include. Not used
                if input `changepoints` is supplied. If `changepoints` is not supplied,
                then n_changepoints potential changepoints are selected uniformly from
                the first `changepoint_range` proportion of the history.
            changepoint_range: Proportion of history in which trend changepoints will
                be estimated. Defaults to 0.8 for the first 80%. Not used if
                `changepoints` is specified.
            changepoint_prior_scale: Parameter modulating the flexibility of the
                automatic changepoint selection. Large values will allow many
                changepoints, small values will allow few changepoints.
        """

        data = self.df.copy().reset_index().rename(columns={self.date: "ds", self.y: "y"}).sort_values("ds")

        if cap:
            data["cap"] = cap
        if floor:
            data["floor"] = floor

        train, test = data[data["ds"] < self.start_dt], data[data["ds"] >= self.start_dt]

        self.model = FP(
            growth=growth,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            interval_width=1-alpha,
            **kwargs,
        )
        
        if monthly_seasonality:
            self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            
        if country_holidays:
            self.model.add_country_holidays(country_name=country_holidays)

        if outliers:
            for pair in outliers:
                train.loc[
                    (train["ds"] > datetime.datetime.strptime(pair[0], "%Y-%m-%d").date()) & 
                    (train["ds"] < datetime.datetime.strptime(pair[1], "%Y-%m-%d").date()),
                "y"] = None

        for feature in self.x:
            self.model.add_regressor(feature)

        self.model.fit(train)

        future = self.model.make_future_dataframe(periods=test.shape[0])
        
        self.result = self.model.predict(future.set_index("ds").join(data.set_index("ds")).reset_index())
        self.result["ds"] = self.result["ds"].dt.date
        self.result = self.result.set_index("ds").join(data[["ds", "y"]].set_index("ds")).rename(columns={"y": "response"})

    def summary(
        self,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> pd.DataFrame:
        """
        Plot the regressors statistics: Coefficients, Impact and Impact Share

        The estimated beta coefficient for each regressor roughly represents the increase
        in prediction value for a unit increase in the regressor value.
        Note that the coefficients returned are always on the scale of the original data
        In addition the credible interval for each coefficient is also returned,
        which can help identify whether each regressor is “statistically significant”.

        On the basis of `seasonality_mode` the model looks like:
            Additive: y(t) ~ trend(t) + seasonality(t) + beta * regressor(t)
            Multiplicative: y(t) ~ trend(t) * ( 1 + seasonality(t) + beta * regressor(t) )

        Therefore, the incremental impact are:
            Additive: increasing the value of the regressor by a unit leads to an increase in y(t) by beta units
            Multiplicative: increasing the value of the regressor by a unit leads to increase in y(t) by beta * trend(t) units

        The Impact is the product of incremental impact(t) * regressor(t) and finally, Share is the percentage of absolute Impact

        Parameters
        ----------
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """

        try:
            data = regressor_coefficients(self.model)
        except AttributeError:
            raise AttributeError("To get the summary run the model first, use .run() method")
        
        last_day_data, last_day_result = self.df.iloc[-1, :], self.result.iloc[-1, :]

        data["incremental_impact"] = data["coef"] * data["regressor_mode"].apply(lambda x: last_day_result["trend"] if x == "multiplicative" else 1)
        data["impact"] = data.apply(lambda x: x["incremental_impact"] * last_day_data[x["regressor"]], axis=1)
        data["share"] = round(100 * np.abs(data["impact"]) / np.sum(np.abs(data["impact"])), 2)

        def plot_bar(data, y, title):
            figure = express.bar(
                data,
                x="regressor",
                y=y,
                color="regressor",
                color_discrete_sequence=express.colors.sequential.Jet,
            )
            figure.add_hline(y=0, line_width=1, line_color="white")
            figure.update_xaxes(title_text=None)
            figure.update_layout(
                title={
                    "x": 0.5,
                    "text": title,
                },
                width=width,
                height=height,
                hovermode="x",
                template="plotly_dark",
                showlegend=False,
            )

            if save:
                self._save(figure, title)
            
            figure.show()

        for y, title in zip(["coef", "impact"], ["Coefficients", "Impact"]):
            plot_bar(data.sort_values(by="coef", ascending=False), y, title)

        pie = express.pie(
            data.sort_values(by="coef", ascending=False),
            color_discrete_sequence=express.colors.sequential.Jet,
            values="share",
            names="regressor",
            color="regressor",
        )

        pie.update_layout(
            title={
                "x": 0.5,
                "text": "Impact Share",
            },
            width=width,
            height=height,
            hovermode="x",
            template="plotly_dark",
            showlegend=False,
        )

        pie.update_traces(
            marker=dict(
                line=dict(color="black", width=3)
            )
        )

        if save:
            self._save(pie, "Impact Share")

        pie.show()
```

</details>

### Model

Sure, you could spend more time trying different models like moving averages and seasonal arima to evaluate and choose better one eventually, but here it's a short way to get pretty good base model, which is fast and robust enough.

<details>
<summary>Code</summary>

``` python
model = Prophet(pivot.reset_index(), y="Lisbon", date="date", test_days=7)
model.explore(scale=True, title="Correlation: Lisbon vs Accra", save=True)
```

</details>

{{< plotly obj=correlation_lisbon_vs_accra >}}

The default implemented strategy saves `plotly` figures as `html` code to be viewable aside of the iPython environment.
However if one is interested in more uniform way of storage, here is the function to save `html` as `json` without building Plotly figure from scratch.

<details>
<summary>Code</summary>

``` python
import plotly
import json
import re

def plotly_html_to_json(name: str) -> None:
    with open(f"{name}.html", "r", encoding="utf-8") as file:
        html = file.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-2**16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    figure = plotly.io.from_json(json.dumps(plotly_json))
    figure.write_json(f"{name}.json")
```

</details>

Key findings:

-   There is a steadily growing trend for both cities
-   There is different seasonality for cities:
    -   additive weekly seasonality for Lisbon (Thursday, Friday - max, Sunday - min)
    -   for Accra it's no so explicit, and the pattern is different (the maximum is often reached on weekends, and Friday is min)
-   There are different change points for cities:
    -   For Lisbon since the beginning of February the increasing trend was replaced by flatten one
    -   For Accra the trend is more subtle, but given the difference in absolute number the confidence is such a trend much lower
-   There are outliers:
    -   For Lisbon Feb 26 is enormously higher, the reason could be the national holiday - carnival on Feb 25
    -   In addition Feb 14 - not a public holiday but a nice time to have a romantic dinner at home by candlelight
    -   For Accra Feb 7 is enormously lower, could be the object for detailed analysis (a bit covered below)
    -   Moreover for Accra Feb 26 is the maximum for Ghana as well, may be some promo took place for both cities
    -   And the last point of data should be the minimum for Accra, but anyway looks too low, might be the dataset doesn't include all the orders until the end of the day

### Outliers Investigation

To interpret outliers one of the possible options is to explore Delivery Success rate

<details>
<summary>Code</summary>

``` python
db.query("""
WITH temp AS (
    SELECT "Created Date"
        ,"City"
        ,"Order State"
        ,COUNT("Order ID") as "Orders Count"
    FROM
            orders o
    WHERE "City" in ('Accra', 'Lisbon')
        AND "Created Date" IN (
            '06.02.2020',
            '07.02.2020',
            '08.02.2020',
            '26.02.2020',
            '27.02.2020',
            '28.02.2020'
        )
    GROUP BY ALL
),

totals AS (
    SELECT "Created Date"
        ,"City"
        ,SUM("Orders Count") AS "Orders Total"
    FROM temp
    GROUP BY 1, 2
)
SELECT temp.*, 100 * ROUND("Orders Count"/"Orders Total", 4) AS "Share"
FROM temp
JOIN totals
USING ("Created Date", "City")
ORDER BY ALL
""").to_df()
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

|     | Created Date | City   | Order State | Orders Count | Share |
|-----|--------------|--------|-------------|--------------|-------|
| 0   | 06.02.2020   | Accra  | delivered   | 229          | 91.60 |
| 1   | 06.02.2020   | Accra  | failed      | 11           | 4.40  |
| 2   | 06.02.2020   | Accra  | rejected    | 10           | 4.00  |
| 3   | 06.02.2020   | Lisbon | delivered   | 1855         | 98.78 |
| 4   | 06.02.2020   | Lisbon | failed      | 1            | 0.05  |
| 5   | 06.02.2020   | Lisbon | rejected    | 22           | 1.17  |
| 6   | 07.02.2020   | Accra  | delivered   | 161          | 86.10 |
| 7   | 07.02.2020   | Accra  | failed      | 23           | 12.30 |
| 8   | 07.02.2020   | Accra  | rejected    | 3            | 1.60  |
| 9   | 07.02.2020   | Lisbon | delivered   | 1725         | 99.19 |
| 10  | 07.02.2020   | Lisbon | failed      | 4            | 0.23  |
| 11  | 07.02.2020   | Lisbon | rejected    | 10           | 0.58  |
| 12  | 08.02.2020   | Accra  | delivered   | 264          | 95.65 |
| 13  | 08.02.2020   | Accra  | failed      | 4            | 1.45  |
| 14  | 08.02.2020   | Accra  | rejected    | 8            | 2.90  |
| 15  | 08.02.2020   | Lisbon | delivered   | 1246         | 99.12 |
| 16  | 08.02.2020   | Lisbon | failed      | 6            | 0.48  |
| 17  | 08.02.2020   | Lisbon | rejected    | 5            | 0.40  |
| 18  | 26.02.2020   | Accra  | delivered   | 355          | 96.47 |
| 19  | 26.02.2020   | Accra  | failed      | 6            | 1.63  |
| 20  | 26.02.2020   | Accra  | rejected    | 7            | 1.90  |
| 21  | 26.02.2020   | Lisbon | delivered   | 2307         | 98.84 |
| 22  | 26.02.2020   | Lisbon | failed      | 8            | 0.34  |
| 23  | 26.02.2020   | Lisbon | rejected    | 19           | 0.81  |
| 24  | 27.02.2020   | Accra  | delivered   | 300          | 92.31 |
| 25  | 27.02.2020   | Accra  | failed      | 13           | 4.00  |
| 26  | 27.02.2020   | Accra  | rejected    | 12           | 3.69  |
| 27  | 27.02.2020   | Lisbon | delivered   | 2201         | 98.79 |
| 28  | 27.02.2020   | Lisbon | failed      | 6            | 0.27  |
| 29  | 27.02.2020   | Lisbon | rejected    | 21           | 0.94  |
| 30  | 28.02.2020   | Accra  | delivered   | 206          | 85.83 |
| 31  | 28.02.2020   | Accra  | failed      | 17           | 7.08  |
| 32  | 28.02.2020   | Accra  | rejected    | 17           | 7.08  |
| 33  | 28.02.2020   | Lisbon | delivered   | 2030         | 98.88 |
| 34  | 28.02.2020   | Lisbon | failed      | 7            | 0.34  |
| 35  | 28.02.2020   | Lisbon | rejected    | 16           | 0.78  |

</div>

If you're a fan of `pandas` of course it may be easier to apply similar operations within familiar package, and it even may seem as more Pythonic way to do so, although believe me, once you start using `duckdb`, it turns out that some operations are just more native to SQL like complex joins and window functions and you naturally perform them faster in SQL; work smarter, not harder - if you're inventing a bicycle writing SQL alike transformations in Pandas, nobody beyond your internal perfectionist will reward it.

#### Pandas alternative

<details>
<summary>Code</summary>

``` python
temp = orders[
    (
        (orders["City"] == "Accra") | 
        (orders["City"] == "Lisbon")
    ) &
    (
        (orders["Created Date"] == '06.02.2020') |
        (orders["Created Date"] == '07.02.2020') |
        (orders["Created Date"] == '08.02.2020') |
        (orders["Created Date"] == '26.02.2020') |
        (orders["Created Date"] == '27.02.2020') |
        (orders["Created Date"] == '28.02.2020') 
    )
].groupby(["Created Date", "City", "Order State"]).agg({"Order ID": "count"})

temp.join(
    temp.groupby(["Created Date", "City"]).sum(), on=["Created Date", "City"], rsuffix=" Total"
).apply(
    lambda x: 100 * round(x["Order ID"] / x["Order ID Total"], 4), axis=1
)
```

</details>

**Insight**: Quick analysis for Feb 7 and for Feb 28 shows that the percentage of delivered orders in Accra is lower than usual while for Lisbon there is no such a drop, that is one of the potential reason of outliers

## Make a forecast of the number of orders expected in the following 4 weeks

The correlation between two cities is not that high (Pearson coef is less than 0.5) to leverage it, providing the other country as a predictor

### Portugal (Lisbon)

#### Fit

<details>
<summary>Code</summary>

``` python
model = Prophet(pivot["Lisbon"].reset_index(), y="Lisbon", date="date", test_days=7)
```

</details>

Incorporate all the logic described within the previous point:
- linear trend
- weekly additive seasonality
- Carnival, Valentine Day as outliers

Last week as a validation time frame

<details>
<summary>Code</summary>

``` python
model.run(
    growth='linear',
    yearly_seasonality=False,
    monthly_seasonality=False,
    seasonality_mode='additive',
    country_holidays='PT',
    outliers=[("2020-02-13", "2020-02-15")]
)

model.show(
    title="Lisbon",
    y_label="Orders, #",
    save=True,
)
```

</details>

{{< plotly obj=lisbon >}}

**Summary**: model fits well, the last week expectedly exceeds the upper bounds due to the holiday

#### Predict

<details>
<summary>Code</summary>

``` python
libon = model.model.make_future_dataframe(periods=36).join(pivot["Lisbon"], on="ds")

model = Prophet(libon, y="Lisbon", date="ds", test_days=28)

model.run(
    growth='linear',
    yearly_seasonality=False,
    monthly_seasonality=False,
    seasonality_mode='additive',
    country_holidays='PT',
    outliers=[("2020-02-13", "2020-02-15")]
)

model.show(
    title="Lisbon Prognosis",
    y_label="Orders, #",
    save=True,
)
```

</details>

{{< plotly obj=lisbon_prognosis >}}

### Ghana (Accra)

The same algorithm for Accra

#### Fit

<details>
<summary>Code</summary>

``` python
model = Prophet(pivot["Accra"].reset_index(), y="Accra", date="date", test_days=7)
```

</details>

Incorporate all the logic described within the previous point:
- linear trend
- weekly additive seasonality
- Feb 7 & Feb 28 as outliers

Last week as a validation time frame

<details>
<summary>Code</summary>

``` python
model.run(
    growth='linear',
    yearly_seasonality=False,
    monthly_seasonality=False,
    seasonality_mode='additive',
    outliers=[("2020-02-06", "2020-02-08"), ("2020-02-27", "2020-02-29")]
)

model.show(
    title="Accra",
    y_label="Orders, #",
    save=True,
)
```

</details>

{{< plotly obj=accra >}}

**Summary**: model fits well, but the last week exceeds the lower bounds of CI, as it was explored below one of the reason is the drop in the delivery success rate, anyway let's take into account this week for the forecast for March.

#### Predict

<details>
<summary>Code</summary>

``` python
accra = model.model.make_future_dataframe(periods=36).join(pivot["Accra"], on="ds")

model = Prophet(accra, y="Accra", date="ds", test_days=28)

model.run(
    growth='linear',
    yearly_seasonality=False,
    monthly_seasonality=False,
    seasonality_mode='additive',
    outliers=[("2020-02-06", "2020-02-08"), ("2020-02-27", "2020-02-29")]
)

model.show(
    title="Accra Prognosis",
    y_label="Orders, #",
    save=True,
)
```

</details>

{{< plotly obj=accra_prognosis >}}

Several Notes:

1.  Lisbon:
    1.  **Valentine Day**: Feb 14 was unusual day, my initial idea seems to be right based upon the cuisine which was specific to this date. The burgers which is the main cuisine were not so popular this day, but there were spikes for the meals for the group of people and "romantic" food, namely Japanese (Sushi) & Italian (Pizza) as well as Breakfast, Ice Cream, Desserts
    2.  **Average Check**: During the top weekdays in addition to orders number the average order values grows, not only more people order, but each person order more
2.  Accra:
    1.  **Delivered Percentage**: first of all the delivered percentage for Accra is less than for Lisbon in general (statistical significance is checked below) and the second point is more accurate look at the dynamic, indeed there were some problems on Feb 7 (driver: ios failed) & Feb 28 (driver: android rejected)
    2.  **Delivery Time**: there is a positive trend in delivery time, one of the reasons can be the launch of the new suppliers, but it's impossible to check precisely using the given time frame only

### Conclusion

**Instead of Summary**: If you don't have enough time, to go too much in detail (it doesn't mean that one can solve only well formulated problems, sometimes it only means that a person indeed doesn't have much time) use this high-level approach to explore you data and get immediate insights about your data structure; because after all arbitrarily amount of time can be spent on researching the data, but business never wants to wait.
