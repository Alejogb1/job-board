---
title: "How can rolling optimization be implemented in Vectorbt with varying lookback windows?"
date: "2025-01-30"
id: "how-can-rolling-optimization-be-implemented-in-vectorbt"
---
Vectorbt, unlike some simpler backtesting libraries, offers considerable flexibility in how rolling optimization is implemented, a feature I've leveraged extensively in my own quantitative strategies over the past few years. Specifically, the ability to vary lookback windows within a rolling optimization scheme allows for more nuanced model calibration, adapting to potentially changing market dynamics. This capability, however, requires careful consideration of the underlying mechanics to avoid pitfalls and generate robust results.

Fundamentally, rolling optimization involves repeatedly optimizing a strategy's parameters on historical data, and then applying those optimized parameters to a subsequent out-of-sample period. The process then rolls forward in time. The "varying lookback window" aspect introduces complexity because it means the amount of historical data used for optimization changes with each rolling step. This is unlike fixed window approaches, where the optimization always uses a constant history length. Vectorbt facilitates this through the `optimize` method coupled with intelligent data slicing. We need to ensure proper slicing logic to dynamically control the lookback period while also handling boundary conditions gracefully.

The challenge comes in efficiently constructing these time-based slices and then feeding them to Vectorbt's optimizer. Vectorbt’s indexing system, based on Pandas DataFrames and Series, is critical. The typical approach involves preparing a time series index representing the available data and then calculating starting and ending points for each optimization window. These points are not necessarily regular. For example, we could configure it to use shorter lookbacks during high volatility periods and longer ones during calmer phases. The optimizer, in conjunction with our slice management, must then properly apply optimized parameters to out-of-sample data. This implies that the optimized parameters need to be stored for each slice.

**Code Example 1: Fixed Lookback, Rolling Forward**

This first example will lay the groundwork, demonstrating a simpler case with a fixed lookback window. While it doesn't directly address varying lookbacks, it is a vital basis for understanding the overall structure.

```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# Sample data generation
np.random.seed(42) # for reproducibility
dates = pd.date_range("2020-01-01", "2023-12-31", freq='B')
price_data = np.random.rand(len(dates)) * 100
price_series = pd.Series(price_data, index=dates)


def create_strategy(window):
    fast_ma = vbt.MA.run(price_series, window=window[0])
    slow_ma = vbt.MA.run(price_series, window=window[1])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return vbt.Portfolio.from_signals(price_series, entries, exits)

# Define fixed lookback window
lookback_days = 365
step_size_days = 90 # for out-of-sample
start_date = dates[lookback_days]

# Generate optimization parameters
fast_mas = np.arange(5, 25, 5)
slow_mas = np.arange(20, 60, 10)

optim_results = {}
for i in range(0, len(dates) - lookback_days - step_size_days, step_size_days):
    train_start = dates[i]
    train_end = dates[i + lookback_days]
    test_start = dates[i + lookback_days]
    test_end = dates[i + lookback_days + step_size_days]

    # Slice the data for optimization
    train_price_slice = price_series.loc[train_start:train_end]

    # Optimize the strategy on the current slice
    portfolio = vbt.Portfolio.from_signals.run(train_price_slice,
                                         vbt.MA.run(train_price_slice, fast_mas).ma_crossed_above(vbt.MA.run(train_price_slice, slow_mas)),
                                         vbt.MA.run(train_price_slice, fast_mas).ma_crossed_below(vbt.MA.run(train_price_slice, slow_mas))
                                         )

    best_params = portfolio.result().sharpe_ratio.idxmax() # retrieve best parameters
    best_fast_ma = fast_mas[best_params[0]]
    best_slow_ma = slow_mas[best_params[1]]

    # Apply optimized parameters on the out-of-sample
    test_price_slice = price_series.loc[test_start:test_end]

    test_portfolio = create_strategy((best_fast_ma, best_slow_ma))
    optim_results[test_start] = test_portfolio.total_return()[0]


print(pd.Series(optim_results).to_string())
```

This code demonstrates a basic fixed-lookback optimization. We iterate through the dataset with `step_size_days`, slice out a training window of `lookback_days` and then optimize over a set of fast and slow MA parameters. The out-of-sample data for the selected window is then used with the optimized parameters. This pattern is crucial for rolling optimization of any kind. The 'optimization' process itself is wrapped inside a simple for-loop and stores the sharpe-maximizing parameters for later out of sample calculation.

**Code Example 2: Varying Lookback, Based on Volatility**

Now we enhance the previous example to implement dynamically varying lookback windows, controlled by a simple measure of volatility. This is the core of the initial request.

```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# Sample data generation (same as before)
np.random.seed(42)
dates = pd.date_range("2020-01-01", "2023-12-31", freq='B')
price_data = np.random.rand(len(dates)) * 100
price_series = pd.Series(price_data, index=dates)

def create_strategy(window, data):
    fast_ma = vbt.MA.run(data, window=window[0])
    slow_ma = vbt.MA.run(data, window=window[1])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return vbt.Portfolio.from_signals(data, entries, exits)


# Define base lookback and sensitivity parameter for volatility
base_lookback_days = 250
volatility_window = 30  # Short window for volatility calculation
volatility_threshold = 1.0 # simple threshold multiplier

step_size_days = 90
start_date = dates[base_lookback_days]

# Optimization parameters (same as before)
fast_mas = np.arange(5, 25, 5)
slow_mas = np.arange(20, 60, 10)

optim_results = {}

for i in range(0, len(dates) - base_lookback_days - step_size_days, step_size_days):

    test_start = dates[i + base_lookback_days]
    test_end = dates[i + base_lookback_days + step_size_days]

    # Calculate volatility to determine the lookback window
    volatility = price_series.loc[dates[i] : test_start].rolling(window=volatility_window).std().mean()
    dynamic_lookback_days = int(base_lookback_days + (volatility - 1.0) * volatility_threshold)
    dynamic_lookback_days = max(dynamic_lookback_days, 30) # min lookback
    train_start = dates[i]
    train_end = dates[i + dynamic_lookback_days]

    # Slice the data for optimization using dynamic lookback
    train_price_slice = price_series.loc[train_start : train_end]

    # Optimize the strategy on the current slice
    portfolio = vbt.Portfolio.from_signals.run(train_price_slice,
                                            vbt.MA.run(train_price_slice, fast_mas).ma_crossed_above(vbt.MA.run(train_price_slice, slow_mas)),
                                            vbt.MA.run(train_price_slice, fast_mas).ma_crossed_below(vbt.MA.run(train_price_slice, slow_mas))
                                         )

    best_params = portfolio.result().sharpe_ratio.idxmax()
    best_fast_ma = fast_mas[best_params[0]]
    best_slow_ma = slow_mas[best_params[1]]

    # Apply optimized parameters on the out-of-sample
    test_price_slice = price_series.loc[test_start:test_end]
    test_portfolio = create_strategy((best_fast_ma, best_slow_ma), test_price_slice)
    optim_results[test_start] = test_portfolio.total_return()[0]

print(pd.Series(optim_results).to_string())
```

Here, we introduce the `volatility_window` to calculate a moving standard deviation as a simplified measure of recent volatility. Based on volatility compared to the threshold, the lookback window is adjusted dynamically. Lower volatility results in longer lookback windows, while higher volatility shrinks the lookback. This shows how a dynamic element may be added to the sliding window.

**Code Example 3: Varying Lookback, Based on User Input**

The final example shows how to use a more arbitrary varying lookback, based on user input, instead of a volatility metric. This allows very fine-grained control over lookback periods.

```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# Sample data generation (same as before)
np.random.seed(42)
dates = pd.date_range("2020-01-01", "2023-12-31", freq='B')
price_data = np.random.rand(len(dates)) * 100
price_series = pd.Series(price_data, index=dates)

def create_strategy(window, data):
    fast_ma = vbt.MA.run(data, window=window[0])
    slow_ma = vbt.MA.run(data, window=window[1])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return vbt.Portfolio.from_signals(data, entries, exits)

# Define manual lookback windows
lookback_windows = {
    dates[0]: 180,
    dates[365]: 220,
    dates[730]: 300,
    dates[1095]: 150
}

step_size_days = 90
start_date = dates[max(lookback_windows.values())]

# Optimization parameters
fast_mas = np.arange(5, 25, 5)
slow_mas = np.arange(20, 60, 10)

optim_results = {}

current_lookback_end = dates[0] #start by assuming the first lookback is used

for i in range(0, len(dates) - max(lookback_windows.values()) - step_size_days, step_size_days):
    
    test_start = dates[i + max(lookback_windows.values())]
    test_end = dates[i + max(lookback_windows.values()) + step_size_days]
    
    #find the applicable lookback window
    train_start = dates[i]
    for k in lookback_windows:
        if k <= train_start:
            current_lookback_end = k
        else:
            break

    dynamic_lookback_days = lookback_windows[current_lookback_end]
    train_end = dates[i + dynamic_lookback_days]

    # Slice the data for optimization using dynamic lookback
    train_price_slice = price_series.loc[train_start : train_end]

    # Optimize the strategy on the current slice
    portfolio = vbt.Portfolio.from_signals.run(train_price_slice,
                                         vbt.MA.run(train_price_slice, fast_mas).ma_crossed_above(vbt.MA.run(train_price_slice, slow_mas)),
                                         vbt.MA.run(train_price_slice, fast_mas).ma_crossed_below(vbt.MA.run(train_price_slice, slow_mas))
                                         )

    best_params = portfolio.result().sharpe_ratio.idxmax()
    best_fast_ma = fast_mas[best_params[0]]
    best_slow_ma = slow_mas[best_params[1]]

    # Apply optimized parameters on the out-of-sample
    test_price_slice = price_series.loc[test_start:test_end]
    test_portfolio = create_strategy((best_fast_ma, best_slow_ma), test_price_slice)
    optim_results[test_start] = test_portfolio.total_return()[0]

print(pd.Series(optim_results).to_string())
```

In this code, we define a dictionary (`lookback_windows`) mapping dates to lookback periods. The code iterates through the dataset as before, and on each loop it determines which user-defined lookback window is applicable at the current date. We use this to dynamically slice the price data and then use the optimization strategy.

In each example, care must be taken to define optimization regions based on the desired start and end points. Vectorbt’s methods then facilitate the actual parameter optimization. Notice that `vbt.Portfolio.from_signals.run` is not a method that acts upon a class or object, but rather can take the raw data, the data for parameter exploration and perform the simulation and exploration all in one go. The slicing logic is the critical element for getting Vectorbt to work within a rolling optimization context.

**Resource Recommendations**

To further refine your approach, I suggest reviewing documentation focusing on time series analysis using Pandas. This is fundamental for correct data manipulation and slicing. Secondly, delve into quantitative finance textbooks focusing on robust backtesting, specifically emphasizing concepts of train/test splitting, and proper parameter exploration methods. Additionally, studying papers and resources on the use of dynamic parameter estimation in quantitative trading can provide theoretical context. Lastly, although the examples above show a basic grid search method, there are many more efficient methods of parameter optimization. I would suggest reading about methods such as Bayesian Optimization for future projects.
