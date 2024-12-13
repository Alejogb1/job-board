---
title: "stata error factor variables and time series operators not allowed?"
date: "2024-12-13"
id: "stata-error-factor-variables-and-time-series-operators-not-allowed"
---

Alright so this is a classic one I've banged my head against this a few times back in the day especially when I was first getting into econometrics stuff with Stata Feels like yesterday I was pulling my hair out over the same damn error message "factor variables and time series operators not allowed" Man I thought I knew Stata I was wrong so very very wrong

Okay so here’s the deal what's happening when you get this error is that you're trying to use things like lag operators or lead operators or sometimes even interacting categorical variables inside a place where Stata doesn't allow it Stata has specific syntax rules about where those commands can be used and they're often not allowed together it really does take some getting used to

The most common culprit is using factor variables and time series operators directly within a regression model when you have a panel data set the thing is Stata often needs to do a lot of internal work to handle time series operators like lag L or lead F these operators change the order of the data which can conflict with factor variable expansion or when you just mix them up without Stata knowing what to do first. Factor variables which are just categorical variables that Stata expands into a bunch of binary indicators they're great for modeling but they add a layer of complexity that can confuse Stata when used with time series operators

Okay first let me say that there is no way for me to give you a universal solution without knowing the exact data and specific model setup you are working with. However I can give you some general ideas of what could be going on and how to potentially fix your problem based on my past experiences.

So let’s say you have a dataset which tracks some companies over time and you're trying to figure out some model with yearly data and a lagged value of x variable and you also want to have a model using sector and year as dummies.

You might try something like this which is a common mistake I myself made once:

```stata
regress y L.x i.sector i.year
```

This code looks perfectly reasonable at first glance right? However, this will throw the error we are discussing because Stata gets confused with what it needs to solve first. Is it the time series aspect or the factor variables? This is a problem.

Now let's break down how we can fix this. The most common way to address this is usually by creating the lag variable manually and then we use that variable in the regression.

Here's a code example that creates the lag of the variable x:

```stata
tsset company year
generate lag_x = L.x
regress y lag_x i.sector i.year
```

See what I did there? It seems trivial right? But in practice this is one of the most common problems I have seen in other people's code and in my own in the past.

First I use tsset to tell Stata what the panel identifier is and the time index and that's super crucial for the time series operators. Then, I explicitly generate the lagged variable lag_x using L.x and after that, the regression is just straightforward Stata knows exactly what to do. This is because the time series operator is not in the regression itself, and Stata has no conflict as to what to do with the expression in the regression line.

Now what happens if we wanted to do some interaction between year and sectors? Again, you would fall back to a similar idea of manual manipulation. Let's do a slightly more complex example where I include the lag of x and also an interaction of sector with year.

```stata
tsset company year
generate lag_x = L.x
gen year_sector =  year * sector
regress y lag_x i.sector i.year year_sector
```

Now there are ways to make the year_sector as factor variables using the hash tag operator but let's just avoid that for now as it will introduce even more complexity than needed for this specific problem.

Okay so one more thing. Sometimes you might think that the problem comes from the tsset command, but the real issue isn’t that. What happens if you forget the tsset command? you get something like this error message "time series operators not allowed without time variable" now that sounds kind of similar right? but that is a different problem altogether you also need to make sure you have a time series defined for the lag operator L, F.

So let's recap the common problems.

1) Using L or F directly inside a regression involving factor variables.

2) Forgetting tsset.

3) Trying to use factor variables inside time series commands.

The first problem is the most frequent and the best way to solve it is to manually create the lag variable as I showed in the code examples above and then to simply use the generated variable in the regression.

There isn't an easy fix to the other problem without knowing what exactly you are trying to achieve. However it has been my experience that carefully crafting each step and thinking what is Stata trying to do is the way to go and that is why I always say its better to "be explicit" with Stata.

Now I will do my best to not give you "magic tricks" but rather give you solid advice based on my experience, but I won’t lie, sometimes it really feels like you need to cast a spell to make Stata cooperate. I have a funny story about that one time… oh well let's leave that for another day.

Okay let's talk resources cause this is an area where you need some solid foundations if you want to get better. Stata has very good documentation so I always start there but a good place to start would be the [Stata manual on time series analysis](https://www.stata.com/features/overview/time-series/) it has a good overview of the tsset command and its limitations.

If you want to dive deeper and have a better understanding of the underlying econometrics I suggest reading books such as “Econometric Analysis” by Greene it's a classic for a reason or “Introductory Econometrics: A Modern Approach” by Wooldridge it gives you a good understanding of the fundamentals. Those textbooks go much more in depth about time series and panel data which is what you are dealing with.

And lastly remember one thing: be precise when coding for statistics. Stata is very picky with syntax but once you understand the principles that is when the real magic can begin.
