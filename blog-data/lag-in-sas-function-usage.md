---
title: "lag in sas function usage?"
date: "2024-12-13"
id: "lag-in-sas-function-usage"
---

Okay so lag in SAS function usage huh been there done that got the t-shirt or more accurately the slightly frayed keyboard I’ve smashed in frustration over slow SAS jobs So yeah I get it you’re running some SAS code and the LAG function is making things feel like molasses in January it’s a classic problem and let me tell you I’ve wrestled with this beast more times than I care to admit

First things first when we say lag we’re talking about accessing values from previous rows right It’s a common need in time series analysis or anytime you need to compare a current observation to what came before But the way SAS handles it sometimes well it’s not exactly a speed demon especially if you’re dealing with large datasets or complex logic and the problems start there so let me take you back to the dark ages of my early SAS days I was building this complex financial model for a client a monster of a dataset millions of records you know the type and I naively thought hey LAG should do the trick right

Well oh boy I was wrong the whole process was taking hours to run and when I actually sat down and looked at the logs I saw that most of the time was spent just in that little LAG statement and I was like no way this can't be right so I dove deep trying to understand how it actually works and it turns out the simple apparent simplicity of the LAG function hides some serious overhead and that overhead gets multiplied when the data grows So my mistake was not considering how SAS actually does the lookup it’s not exactly a fast pointer access as we’d like in other languages it’s more like a sequential comparison on the row number I was not happy with this revelation

So what did I do? I started experimenting with different ways of achieving the same goal and that’s where I started learning about other techniques and I’ll share some of those with you

First thing let's talk about the simplest case calculating the difference between adjacent values a very classic usage of LAG we might see something like this

```SAS
data want;
  set have;
  prev_value = lag(value);
  diff = value - prev_value;
run;
```

This looks innocuous enough and for a small dataset it works okay but as the dataset grows it really starts to show the limitations of the LAG function. The issue is every time the LAG function is called the sas process has to keep tracking which row it was which increases the operation cost exponentially. In fact it is so bad that I suspect the SAS developers thought no one would ever run this on huge datasets (but here we are right?).

So one thing I discovered is sometimes we might not actually need the LAG function directly we can use some data step logic to achieve the same result. What do I mean by that? Well think about how you usually implement loops in code in other languages. You don’t have to call a lag function each time you can just save the previous value to some variable then reuse it right? well in SAS it is the same thing.

This is what it might look like:

```SAS
data want;
  set have;
  retain prev_value;
  if _N_ > 1 then do;
    diff = value - prev_value;
  end;
  prev_value = value;
run;
```

Here what I am doing is using a `retain` statement to keep the previous value then doing the difference between current and previous value without ever calling lag. In general this pattern is far more performant than calling `lag`. I mean for small datasets it won’t matter much but for anything sizable it will be a life changer believe me I know. This was actually the first big lightbulb moment I had with SAS performance.

And this was just the beginning of my SAS optimization journey. Now I have another case where I was using the `lag` on sorted data using `by` groups that’s where the problems really compounded and the whole thing became a real slow burner. This is the common case of having some ID grouping data then using the `lag` function on each group for some kind of time serie analysis.

Suppose we have data like this a simple dataset

```
ID Value Timestamp
A 10 2024-01-01
A 15 2024-01-02
A 12 2024-01-03
B 20 2024-01-01
B 25 2024-01-02
B 22 2024-01-03

```

If we wanted to use the lag inside the by group we would usually do something like this

```SAS
proc sort data=have out=sorted_have;
    by ID Timestamp;
run;
data want;
  set sorted_have;
  by ID;
  if first.ID then prev_value=.;
    else prev_value = lag(value);
  diff = value - prev_value;
run;
```

This code is the common solution if you ask someone who is not really proficient in sas. And this code will work but the performance is not great for big datasets and you will not want to be running this in production. Now let me tell you the faster better and cooler approach. It’s basically the same `retain` approach but inside the by group

```SAS
proc sort data=have out=sorted_have;
    by ID Timestamp;
run;
data want;
  set sorted_have;
  by ID;
  retain prev_value;
   if first.ID then prev_value=.;
    else diff = value - prev_value;
  prev_value = value;
run;
```

Notice the complete absence of the `lag` function inside the data step. This is the real deal when it comes to performance it will feel like running in a different program compared to the previous code. So yeah I've lost many nights sleep because of this lag thing and I’m sharing this with you so you don’t have to repeat my mistakes

Now I know what you’re thinking “that's great and all but I need something more concrete not just code I can copy and paste” and I agree. I never was a big fan of copy and pasting code without understanding what’s happening in the background. That’s why I suggest you check some good books on SAS performance optimization. Don't rely on blog posts or random online tutorials they are often wrong or incomplete or both!

Specifically I really benefited from books like “SAS System for Statistical Analysis” by Rudolf Freund and “Carpenter's Complete Guide to the SAS Macro Language” by Art Carpenter even though the latter is not directly related to the specific problem I was having it gives a deeper insight of SAS in general which is a requirement for performance optimization in general. I think it also has an important part about macro variables that are useful when your data are dynamically loaded so it's an important skill to develop. Also the "SAS Certified Specialist Prep Guide: Base Programming Using SAS 9.4" is another good book that shows how SAS works at its core.

And while we’re on the topic of learning how SAS works under the hood did you ever wonder why the heck SAS takes so long to do anything? I once tried to explain the intricacies of SAS I/O to a java dev and I swear I could see the life draining from his eyes... they just don't do it the same way in other languages.

So look I know the struggle is real lag is just one of the many things in SAS that seems simple on the surface but can become a real performance bottleneck. Remember to think about how SAS processes data under the hood and try the `retain` approach instead of `lag` whenever you can it’s one of the best tricks that I learned. Don’t be afraid to experiment and dig into the documentation and more importantly don't ever just copy and paste code without understanding it. And that’s how I think the user should approach these kind of problems

Happy coding!
