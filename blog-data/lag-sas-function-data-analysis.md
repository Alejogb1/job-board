---
title: "lag sas function data analysis?"
date: "2024-12-13"
id: "lag-sas-function-data-analysis"
---

Okay so you're dealing with lag in SAS functions during data analysis I've been there man let me tell you been there done that more times than I care to admit This is a classic headache I've spent way too many late nights debugging this kind of thing

First off what do I mean by lag lets get super clear on the terms here I'm talking about getting values from a previous row in your dataset while processing the current row You know like needing the price from yesterday to calculate todays price change or the previous week's sales to compare against current sales Stuff like that

Now SAS has this function LAG which seems straightforward but its got gotchas and we'll dig into those lets start with a basic example I'll assume you've got some time series data

Let's say you have a dataset called `daily_sales` with variables `date` and `sales` and you want to create a new variable `previous_sales` with sales from the prior day

```sas
data daily_sales_with_lag;
  set daily_sales;
  previous_sales = lag(sales);
run;
```

This looks simple right and it works sometimes like a charm but only if your data is perfectly sorted by date and there are no gaps This is the first place people screw up and I've been guilty of it myself you see me do that too many times with some data from the old company where my responsibilities were around doing analysis of the monthly sales per regional area my manager wanted me to always show the differences from the previous month and I had some problems at the begining I even remember me asking in a forum at that time (not this one) and people gave me the same solution as I'm showing you here but no one told me that I need to sort the data first in most of the cases its a basic stuff but sometimes when you are under pressure you can just easily miss something like this

The problem is that if your data is not sorted or has missing dates you will get garbage values in `previous_sales` If for example you sort the data by sales instead of date which I have also seen by some of my colleagues you will get some weird lag values which do not make any sense at all It's a common oversight and something you gotta be super diligent about because SAS will not throw any error it will just give you a wrong result If you got a dataset like this and you don't sort before applying lag then you will have a bad result

So rule number one always verify your sort order

Now lets make this a bit more real because you will usually not have such clean data lets say your data is not always daily maybe some days are missing or some rows are duplicate it can be a pain dealing with this lets fix it

```sas
proc sort data=daily_sales;
  by date;
run;

data daily_sales_with_lag;
  set daily_sales;
  by date;
  if first.date then previous_sales = .;
  else previous_sales = lag(sales);
run;
```

Okay so what did we do here First we make sure the data is properly sorted by `date` with proc sort and the `by date` instruction. That is super important if you don't you'll have a bad result as we discussed before

Then in the data step we use another `by date` statement which is super important since it tells SAS that you're operating within a group of records that have the same `date` and that you want to see the difference of the current date with the previous one In many cases if you do not have the by variable SAS may generate bad values and incorrect calculations. If you do not have the `by` instruction SAS will just calculate the lag in all of the values and not just with values of the same `date`

We also use `if first.date then previous_sales = .` This is crucial this means if the current record is the first record in each of the groups defined in by then set the lag value to missing this is to avoid having a lag from the previous row that is not from the current date. It ensures that we dont get previous sales from different days that is going to lead to an incorrect result

Now this works pretty well for basic lag calculations but what if you need a more complex lag calculation like a lag of two or three days you can extend the use of lag like this

```sas
data daily_sales_with_multiple_lag;
  set daily_sales;
  by date;
  previous_sales_1 = lag(sales);
  previous_sales_2 = lag(sales, 2);
  previous_sales_3 = lag(sales, 3);
run;
```

Here we are calculating the previous sales using `lag` with different lags this means that `previous_sales_1` will have the lag of one day `previous_sales_2` will be the sales of two days before and `previous_sales_3` will have the sales three days before the current one

This will help you a lot but you need to consider that lag functions always use values from a row before the current row. That is its main feature. There are other options in SAS that you can use but mostly you will use `lag` and `diff` functions

Now a funny thing I've seen with lag is that sometimes you think that lag will do something that you want but it is not quite what you expected. I remember one of the old interns I worked with thought that lag would go back to any previous row but it actually does a calculation based on the immediately previous one. He told me "I thought that it would lag to a specific date no matter what". He was very confused and I had to explain it to him and show him examples

Also another detail to consider is that sometimes you want to apply the lag function based on a grouping variable. Lets say that we have different stores and we want to compute the lag per store. In this case we need to add store as a variable in the by clause to obtain a correct result

The key here is not just slapping `lag` onto your code like you think you are a pro you really need to think about your data structure your desired results and how to apply it correctly

Also be aware of implicit loops and when SAS keeps the value by default you can have multiple variables changing and influencing your results if you do not understand the internal workings of how SAS processes data

For deeper understanding I would strongly suggest reading SAS documentation specifically the sections on data step processing the `set` statement and the `by` statement and lag functions and also read on implicit loops and kept values This documentation is super important and there are tons of examples there so you can play around

There is this book "The little SAS book" by Delwiche and Slaughter its a really easy read and a very good starting point

I also recommend reading papers on time series analysis and data manipulation because those help you understand how these functions work conceptually and you can apply this knowledge to different contexts

And remember always test test test with smaller datasets to understand how it works before using it in your million record datasets If you have to deal with a lot of data you may need to consider using other SAS functionalities such as hash tables or arrays to get more optimized results but if you start with a good understanding of `lag` you'll be in a good spot

So that's my two cents (or maybe more like two dollars) on handling lags in SAS Its a very fundamental technique and it takes a while to master and the gotchas can get you if you are not careful. Always pay attention to your data and make sure you have a solid understanding on what you are doing. You can use the lag function to do some very interesting stuff in data analysis just be careful with its implementation
