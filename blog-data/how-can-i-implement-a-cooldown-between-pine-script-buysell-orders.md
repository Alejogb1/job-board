---
title: "How can I implement a cooldown between Pine Script buy/sell orders?"
date: "2024-12-23"
id: "how-can-i-implement-a-cooldown-between-pine-script-buysell-orders"
---

,  I've seen this particular challenge pop up more times than I can count, especially when people are getting into automated trading systems with Pine Script. It’s a critical aspect of preventing over-trading and, quite frankly, blowing up your account. The core issue is introducing a temporal delay, or 'cooldown', between order executions. This isn't directly available as a built-in Pine Script function, so we need to implement it ourselves, typically by leveraging persistent variables and conditional logic.

In the past, during my early days implementing a momentum-based strategy, I encountered this issue head-on. I had a great set of entry conditions, but the script was firing off orders like crazy whenever price momentarily crossed a threshold, leading to incredibly high transaction costs and very little profit. It quickly became apparent that a simple buy or sell signal was not enough; I needed to implement a cooldown period to prevent continuous, rapid-fire trading based on minor fluctuations.

The approach involves essentially creating a persistent timer that keeps track of the last time an order was placed. When a new signal triggers, we first check if the cooldown period has elapsed. If it hasn't, we ignore the signal. If it has, we execute the order and update the timer. This needs to work consistently across bar updates, which is where the persistent nature of Pine Script variables comes into play.

Let me demonstrate this using three different scenarios, progressively more refined:

**Example 1: Basic Cooldown with Simple Timestamp**

This initial example introduces the fundamental logic. We'll use the `var` keyword to declare variables that persist between bars and employ the `time` built-in variable to track timestamps. The cooldown is expressed in milliseconds.

```pinescript
//@version=5
indicator("Basic Cooldown", overlay=true)

var int lastBuyTime = 0
var int lastSellTime = 0
cooldownPeriod = 30000 // 30 seconds in milliseconds

longCondition = ta.crossover(close, ta.sma(close, 10))
shortCondition = ta.crossunder(close, ta.sma(close, 10))

canBuy = time - lastBuyTime >= cooldownPeriod
canSell = time - lastSellTime >= cooldownPeriod


if (longCondition and canBuy)
    strategy.entry("Long", strategy.long)
    lastBuyTime := time

if (shortCondition and canSell)
    strategy.entry("Short", strategy.short)
    lastSellTime := time
```

This first example is fairly straightforward. We declare two `var` variables: `lastBuyTime` and `lastSellTime`, both initialized to zero. The `cooldownPeriod` is set to 30,000 milliseconds, which corresponds to 30 seconds. We have two basic conditions for long and short entries, using a simple moving average crossover. The core logic lies in the checks `canBuy` and `canSell`, ensuring the elapsed time since the last trade is greater than or equal to the `cooldownPeriod`. If these conditions are met *and* a trade signal is present, an order is placed, and the respective timestamp is updated.

While functional, this approach has a limitation. It doesn’t take into account potential slippage or delayed order executions. If an order is filled slightly later than expected, the timestamp will be updated based on when the *signal* was generated, not when the trade actually occurred, which can sometimes cause a slightly faster trading rhythm than intended. We address this limitation in the next example.

**Example 2: Cooldown Based on Order Execution**

In this improved version, we shift the timestamp update to the *after* an order is successfully executed, or at least when the Pine Script strategy object indicates it has registered the order, using the `strategy.position_size` variable to track whether a position has just been opened. This is a more robust approach since it aligns the cooldown with the true execution of the order.

```pinescript
//@version=5
indicator("Cooldown On Order Execution", overlay=true)

var int lastBuyTime = 0
var int lastSellTime = 0
cooldownPeriod = 30000 // 30 seconds in milliseconds

longCondition = ta.crossover(close, ta.sma(close, 10))
shortCondition = ta.crossunder(close, ta.sma(close, 10))

canBuy = time - lastBuyTime >= cooldownPeriod
canSell = time - lastSellTime >= cooldownPeriod

if (longCondition and canBuy)
    strategy.entry("Long", strategy.long)

if (shortCondition and canSell)
    strategy.entry("Short", strategy.short)


if (strategy.position_size > strategy.position_size[1]) // Check if a long order was just filled
    lastBuyTime := time
if (strategy.position_size < strategy.position_size[1]) //Check if a short order was just filled
    lastSellTime := time
```

Here, the fundamental structure is similar, but the crucial difference is how we update the timestamps. We now check for changes in the `strategy.position_size`. If a long position has *just* been opened (current `strategy.position_size` is greater than the previous bar’s value), we update `lastBuyTime`. Similarly, if a short position has *just* been opened (current `strategy.position_size` is less than the previous bar's value), we update `lastSellTime`. This method ensures the cooldown period is triggered exactly when an order is filled (or at least registered), rather than when the signal was simply generated.

However, this implementation is still reliant on `strategy.position_size`, which might not behave perfectly in very volatile markets when multiple trades are possible on the same bar. While this occurrence is rare, it's worth considering in a production system. We can also start looking into other methods to track the time from an order execution.

**Example 3: Cooldown with a Timer Function**

For a more encapsulated approach, let's incorporate a reusable 'timer' function. This makes the code cleaner and modular, and allows for different cooldown periods for different kinds of signals.

```pinescript
//@version=5
indicator("Cooldown with Timer Function", overlay=true)

cooldownTimer(duration) =>
    var int lastTime = 0
    time - lastTime >= duration ? (lastTime := time, true) : false

longCooldown = 30000
shortCooldown = 60000

longCondition = ta.crossover(close, ta.sma(close, 10))
shortCondition = ta.crossunder(close, ta.sma(close, 10))


if longCondition and cooldownTimer(longCooldown)
    strategy.entry("Long", strategy.long)

if shortCondition and cooldownTimer(shortCooldown)
    strategy.entry("Short", strategy.short)

```
In this third example, we have a function named `cooldownTimer`. This function takes a duration (in milliseconds) as input. Inside it, it maintains a `lastTime` variable. The expression `time - lastTime >= duration` checks if the cooldown period has elapsed. If it has, it updates `lastTime` to the current time and returns `true`. If not, it returns `false`. Using this function makes it easy to have different cooldown periods for long and short positions or different trading strategies, enhancing reusability and readability.

Implementing cooldown logic, as you can see, isn’t inherently complicated, but it requires careful management of persistent variables and proper timing. Remember to always test these strategies meticulously on backtesting tools and with paper trading before committing real capital.

For a deeper dive into Pine Script’s strategy capabilities and how to handle variables, I recommend you review the official Pine Script documentation.  You should also consider checking books like "Trading in the Zone" by Mark Douglas for better trading psychology and risk management techniques. "Mastering Trading: Proven Techniques for Profiting from Intraday and Swing Trading Setups" by Tony Oz will provide more insights into how professional traders often structure their own systems. While not specifically related to Pine Script, concepts explained in these trading books are incredibly beneficial. Additionally, academic papers exploring time series analysis will sharpen your understanding of price action in the market.
By carefully implementing a cooldown mechanism, you’ll be on your way to more disciplined and less frantic trading.
