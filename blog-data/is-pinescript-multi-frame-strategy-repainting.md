---
title: "Is PineScript multi-frame strategy repainting?"
date: "2024-12-23"
id: "is-pinescript-multi-frame-strategy-repainting"
---

, let's tackle this, as it's a question that's popped up in forums and trading circles for as long as I can remember – and I’ve spent a good chunk of my career dealing with these types of issues. The short answer regarding Pine Script multi-frame strategy repainting is, yes, *it can*. But, like most things in programming and algorithmic trading, it's not a simple binary 'yes' or 'no'. It depends heavily on how you implement your multi-frame logic, and it's essential to understand the underlying mechanics of how Pine Script handles different timeframes.

Let me start with an experience. Years back, while working on a proprietary trading system, I distinctly recall a very frustrating bug. A backtest showed remarkably consistent gains using multiple timeframes. In reality, it was a mirage. The strategy looked like a beautifully choreographed dance when simulated but, live, it performed like someone tripping over their own feet. The root cause? Repainting stemming from how we had coded our multi-frame data retrieval.

The crux of the issue lies in how Pine Script calculates data on higher timeframes. When you request a higher timeframe’s data (say, daily bars within a 1-hour chart), Pine Script *doesn't* instantaneously load historical daily data point by point from the start of the chart. It uses a ‘look ahead’ mechanism. Instead, it initially calculates only the most recent values of the higher timeframe, and then, as the current bar on the lower timeframe moves forward, those higher timeframe values are *potentially adjusted* or *repainted* based on new lower timeframe information. This is what leads to the illusion of profitability in backtests but produces a significantly different result in live trading. The key here is the *potential* adjustment. It does not always happen.

To understand it better, let's explore some code examples. Consider this first, simplified case:

```pinescript
//@version=5
indicator("Multi-Timeframe Example 1", overlay=true)

daily_close = request.security(syminfo.tickerid, "D", close)
plot(daily_close, color=color.blue, title="Daily Close")
```

In this example, we're just plotting the daily closing price on a chart with a lower timeframe. There is nothing that would cause repainting here, because the close value for the day is only known after the day closes. This example is *not* subject to repainting. But, it can be misleading because if, instead, we tried to derive more complex calculations that are only finalized at daily close, then the repainting can appear in the lower timeframe even if it did not appear in the daily timeframe.

Now, let’s consider a case where repainting can happen. Suppose we are trying to calculate an average price over multiple days and use that average within a higher timeframe's timeframe:

```pinescript
//@version=5
indicator("Multi-Timeframe Example 2", overlay=true)

daily_avg_price = request.security(syminfo.tickerid, "D", avg(high, low))
plot(daily_avg_price, color=color.red, title="Daily Average Price")

if (daily_avg_price > daily_avg_price[1])
    strategy.entry("Long", strategy.long)
if (daily_avg_price < daily_avg_price[1])
    strategy.entry("Short", strategy.short)
```

This example appears innocent, but introduces a critical aspect: it computes a derivative of the higher timeframe's data *based on an event happening on the lower timeframe*. Even when referencing the higher timeframe's average price it is influenced by the lower timeframe's bar. The average daily price is, in this scenario, calculated *only for the current lower timeframe bar*. This is a case where repainting can occur, because that higher timeframe value changes over time depending on the current bar of the lower timeframe. This example will, most likely, repaint in a lower timeframe. The reason is, each bar in the lower timeframe will have different values for "daily_avg_price" depending on when it falls in relation to the higher timeframe's bar.

Let’s take it a step further, focusing on how strategy entries can be affected, to make the repainting more obvious:

```pinescript
//@version=5
strategy("Multi-Timeframe Example 3", overlay=true)

daily_close = request.security(syminfo.tickerid, "D", close)
daily_sma = ta.sma(daily_close, 10)


if (close > daily_sma and close[1] < daily_sma[1])
    strategy.entry("Long", strategy.long)

if (close < daily_sma and close[1] > daily_sma[1])
    strategy.entry("Short", strategy.short)

plot(daily_sma, color=color.green, title="Daily SMA")
```

Here, we're using the daily simple moving average (sma) of closing prices, within the lower timeframe chart. Critically, the code is not directly using the lower timeframe's calculations, but comparing the higher timeframe's value to the current price in order to decide when to enter or exit. This creates an effect similar to using `lookahead=true`. As the lower timeframe progresses, the calculation of the daily sma will change, meaning the entry and exit conditions can trigger retroactively. For example, a long entry may look like it triggered at bar 'x', when the strategy calculation actually only happened at bar 'y' and because the higher timeframe value has been adjusted on bar 'y' the entry condition at 'x' looks like it was correct, when it might not have been at that point in the time series. This can lead to severely optimistic backtests, which will fall apart in real trading.

To avoid these issues, it's crucial to understand how to properly utilize `request.security` and consider alternative approaches. Specifically, the `lookahead=barmerge.lookahead_on` argument can be helpful in some scenarios. The documentation from TradingView is the canonical reference for the specific `lookahead` argument. Understanding it fully is essential. However, it is *not* the solution to every repainting situation. More broadly, you should be thinking about whether or not your calculation is only completed when the higher timeframe's bar is fully closed (for example, is the high of a daily bar finalized during the day?).

Additionally, you can explore other techniques, such as using the `array` and `matrix` functions in Pine Script to explicitly store and manipulate historical data, rather than relying solely on the implicit data provided through `request.security`. While it requires more coding, it also gives you a greater understanding of the data, and more granular control of how values are calculated and when calculations are applied.

The topic of ‘lookahead bias’ is critical to understand as well, and it’s related to repainting in a very significant way. Read "Quantitative Trading: How to Build Your Own Algorithmic Trading Business" by Ernie Chan. It dedicates a chapter to the subject, that is critical to understanding what causes it and how to avoid it.

In summary, Pine Script can introduce multi-frame repainting if the calculations depend on higher timeframe values which are not yet finalized in the higher timeframe’s bar and how the calculations are used in the lower timeframe. The issue arises from how Pine Script handles higher timeframe data, with the potential for the higher timeframe calculations to be adjusted as the current lower timeframe bar progresses. Avoiding this requires careful coding, understanding `request.security`, and exploring more explicit data handling methods. It's not a limitation of the language itself, but a challenge related to how complex timeframe interaction has to be dealt with. My experiences taught me that thorough understanding and testing is crucial when combining multiple timeframes to avoid backtest over-optimization and poor results in live trading.
