---
title: "float sparql filter query usage?"
date: "2024-12-13"
id: "float-sparql-filter-query-usage"
---

Okay so you're asking about using `FILTER` with `float` values in SPARQL queries yeah I've been down that rabbit hole more times than I care to admit. It's one of those things that seems simple on the surface but can lead to some head scratching real quick if you're not careful let me break it down from my experience having wrestled with this a bunch of times

Right off the bat the most basic thing is you're dealing with floats so you need to think about precision and how SPARQL treats number literals and comparisons. They aren't always what you might expect if you are coming from pure programming background.

First let’s nail down a basic example say you've got some RDF data like this stored somewhere it's a simplified version but its to explain the issue

```turtle
@prefix ex: <http://example.org/> .

ex:item1 ex:value 1.23 .
ex:item2 ex:value 1.230 .
ex:item3 ex:value 1.234 .
ex:item4 ex:value 1.2 .
ex:item5 ex:value 1.2300000001 .
```

Now lets say you want to find all items that have a value greater than 1.23 what you probably think is a simple query might not actually work how you think.

```sparql
SELECT ?item ?value
WHERE {
  ?item ex:value ?value .
  FILTER (?value > 1.23)
}
```

This should work fine right Well mostly it depends on your triple store implementation and how it handles floating point representations and comparison. It also assumes that the ?value is correctly interpreted as a floating point. The problem stems from the fact that in RDF floats and decimals may or may not have explicit data types. That means when you compare a `xsd:double` or a `xsd:decimal` to just a number it can produce inconsistent results. RDF is not like your usual code system.

In my early days working with sensor data I had this exact problem. I was getting seemingly random filtering results because some sensors would output values with slightly different levels of decimal precision. It was a mess that took me ages to debug and thats where i started to use the function `datatype` to make sure I was actually comparing the proper types.

For instance I was trying to filter data from a temperature sensor and the returned sensor data had 5 decimal places. My filter only used 2 decimal places for filtering. The results were erratic so I made sure to always compare equal types.

Here is a better approach:

```sparql
SELECT ?item ?value
WHERE {
  ?item ex:value ?value .
  FILTER (datatype(?value) = xsd:double && ?value > 1.23)
}
```

This makes it explicit that you're comparing a float to a float and not a lexical value with an unkown type making sure your queries are consistent. Using `datatype()` makes sure your query is less prone to data type inconsistency.

Also keep in mind that `1.23` is interpreted by some implementations as `xsd:decimal` which is a different type than `xsd:float` or `xsd:double`. This can lead to unexpected behavior. Also the implementation of SPARQL may not have the same support as your programming language. SPARQL is mostly meant for querying data not for processing it the same way your program does. So it is important to make sure your comparisons always have equal types.

Now let’s talk about another common issue. Sometimes you need to filter based on a range of values say between 1.2 and 1.3.

Here is how not to do it the naive approach that will cause you a headache.

```sparql
SELECT ?item ?value
WHERE {
  ?item ex:value ?value .
  FILTER (?value >= 1.2 && ?value <= 1.3)
}
```

Again this *might* work and probably it will work but its a gamble as there are floating point representation problems that you might encounter and it will not solve edge cases. This is especially true if you are working with data from different sources where small variations in representation can affect the outcomes.

Here is the good solution the one you should be using if you really want to avoid headaches

```sparql
SELECT ?item ?value
WHERE {
  ?item ex:value ?value .
  FILTER (datatype(?value) = xsd:double && ?value >= 1.2 && ?value <= 1.3)
}
```

This explicitly checks the type and makes sure that we are really comparing floats against each other and you are comparing the values not the lexical representations. I can't stress this enough having the right type will save you a lot of headaches. I once spent a whole day debugging a query because I forgot to add the `datatype` call and I can tell you I was not happy at all to say the least.

Also I see you were talking about float values so always consider that floating point numbers are often approximations. Floating point are not really meant for exact math. There are floating point errors that need to be considered when using floats in filters. The same number can have a slight variation in representation that affects the comparison. It’s not a bug; its just how computers store numbers. To handle edge cases you might want to add a tolerance when you are doing comparisons. I wish SPARQL had a built-in tolerance comparison.

If your data contains literals with different types you can also use `STRDT` and specify the type. Like `FILTER (STRDT(1.23, xsd:double) = ?value)` also there are also casting functions in SPARQL but I would recommend avoiding them as using them can make your query less readable unless needed. I prefer using the `datatype` function and explicit literals instead.

Another thing to watch out for is how your specific triple store handles indexing of float values. Some stores might have optimizations or limitations in this area and knowing those limitations will help you write efficient queries. Because if your triple store has no index for `xsd:double` it will have to scan the whole graph which makes the query useless.

Oh and a small tip from my experience if you find that your filters are not working or are not performant you might want to check the `EXPLAIN` output of your query. The query execution plans are actually really useful to figure out which part of your query is actually making it slow.

I have been doing this for quite a while so I can tell you that dealing with floating point filters in SPARQL is more complex than it looks. If you think you have a complex query and it should be working but it does not then chances are that it's a floating point data type issue. I've seen many developers make the mistake that numbers are always the same no matter how you compare them. They are not always the same so if you do not want to be scratching your head at 3 am make sure your types are always correct.

To learn more about this I would recommend having a good read of the SPARQL specification itself it is a good source of truth. There are also some great resources on floating point arithmetic and computer representations like "What Every Computer Scientist Should Know About Floating-Point Arithmetic" it's an old paper but is still very relevant and will give you a much better understanding of how floats are actually represented in computers. Now if you excuse me i need to grab another coffee i just had an flashback about a triple store incident I had years ago where a floating point issue took me 3 days to fix.

Remember always type check and you will have a much more easier life doing this.
