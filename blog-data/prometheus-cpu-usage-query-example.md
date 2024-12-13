---
title: "prometheus cpu usage query example?"
date: "2024-12-13"
id: "prometheus-cpu-usage-query-example"
---

Okay so someone's asking about Prometheus CPU usage queries right Been there done that got the t-shirt and probably burned a few CPUs along the way trying to get this right Over the years I've wrestled more than my fair share of wonky Prometheus setups so let's break it down for ya real simple and clear

First off just a quick reminder Prometheus works by scraping metrics from your target systems which are these little time series databases Each metric has a name some labels and values over time So when we're talking about CPU usage we are not just dealing with one number we're actually dealing with a flow of data that we've gotta manipulate to make sense of it all

The most basic thing you wanna know usually is just like what's the raw CPU usage currently right? The way to do that is using `rate()` which gives us the change in a counter over time. Now CPU usage is cumulative so it grows over time. `rate()` will calculate the rate of increase per second. If you have multiple cores you're going to need `sum()` and sometimes `by()` to sum up the usage per core.

Let's take a look at my go to query to do just that:

```promql
sum(rate(process_cpu_seconds_total[5m])) by (instance)
```

Okay so let's break this down for the newbies `process_cpu_seconds_total` is a common metric that exporters use for recording cumulative CPU time. Then I use `[5m]` which means I’m looking at CPU usage over the last five minutes `rate()` calculates the increase per second over that five minute window. and last `sum() by (instance)` is going to sum the CPU usage across all cores and then group it by instance label for each node I mean that's it pretty straightforward

I recall this one time when I was first starting out and I forgot the `sum()` part and I was getting all these crazy graphs and just weird data. Yeah that’s how you get a PhD in Prometheus troubleshooting the hard way. Believe me after that I never forget the sum unless I am trying to diagnose some really low level issue with the core architecture of the system itself. It's amazing how such a tiny detail can throw everything into chaos right?

Now if you want a percentage instead of just the CPU time spent we need to factor in the total time available which is the number of cores multiplied by seconds

We can do that with a slight variation of what I wrote earlier.

```promql
sum(rate(process_cpu_seconds_total[5m])) by (instance) /
count(count(process_cpu_seconds_total) by (cpu) ) by (instance)
```

Now this looks a bit more hairy right Let's make it crystal clear
`sum(rate(process_cpu_seconds_total[5m])) by (instance)` is the same thing we just did now we are dividing that by a number that represent the number of available CPU cores.
`count(count(process_cpu_seconds_total) by (cpu) ) by (instance)` will give the number of cores and then group that by instance. Now you know if you multiply this whole thing by 100 you get the CPU usage percentage.

There's more fun to be had if you want to drill down to specific types of cpu time like user time system time and so on each of those will have their own metrics depending on what your exporter reports. You'll see metrics like `process_cpu_user_seconds_total` or `process_cpu_system_seconds_total` you just gotta substitute that to see which one you want to analyze.

Let's show an example of that to illustrate a case where you might want to analyze these two different times.

```promql
sum(rate(process_cpu_user_seconds_total[5m])) by (instance) /
count(count(process_cpu_user_seconds_total) by (cpu) ) by (instance)
```
```promql
sum(rate(process_cpu_system_seconds_total[5m])) by (instance) /
count(count(process_cpu_system_seconds_total) by (cpu) ) by (instance)
```
In this code snippet we can see that i used user time in the first query and system time in the second one. This is incredibly useful when you want to pinpoint what is actually causing high cpu usage.

I can never remember which is which and I always find myself checking my own notes. The last thing you want is to go around telling people that the application is the problem when in fact your kernel is going nuts because of some faulty driver right? (it happened to me once and I am still ashamed to talk about it)

Now you might be wondering what about host metrics I've been talking about process metrics so far. For host CPU usage you can use something like `node_cpu_seconds_total` from node exporter. Remember they're pretty much the same just with different names.

For instance I'll show you a query that sums up the total CPU usage and the idle CPU of the host. You can calculate the actual usage from it as I showed before.

```promql
sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance)
sum(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)
```

So yeah that's pretty much it. At the end of the day the best approach really depends on your specific needs. You can get pretty crazy with these queries and build some really powerful visualizations. Just remember to start simple work your way up from the basics and understand what is happening. Read the Prometheus documentation and do not just copy and paste. It will bite you later I guarantee it.

I know I am usually not this serious but with Prometheus you just gotta be because it can be complex.

Now for some resources because I don't like people using random blogs I always recommend reading the source.

For a good understanding of time series data concepts go read "Time Series Analysis" by James D. Hamilton. It is a bit dense but really worth it.

For a practical guide to Prometheus itself the "Prometheus: Up & Running" book by Brian Brazil is really great.

Also, the official Prometheus documentation is also excellent. It's always the first place to look if you need something clarified so go there first.

Okay I hope this helps you out. I am done for the day see you around.
