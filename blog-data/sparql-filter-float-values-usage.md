---
title: "sparql filter float values usage?"
date: "2024-12-13"
id: "sparql-filter-float-values-usage"
---

Okay so you're asking about filtering float values in SPARQL right I get it This is one of those things that looks simple on the surface but then you dive in and you're like "wait what's going on" I've been there trust me So yeah SPARQL and floating point numbers are not always best friends I mean floats in general are tricky enough as it is without adding a query language to the mix

Let's break it down based on what I've seen and what I've personally banged my head against over the years I swear I once spent a full day debugging a SPARQL query where the issue was a tiny floating point difference between what I expected and what the data actually held good times

First the basic stuff most common filter usage I see people stumble on this early on

**Basic Filtering**

You've probably got some triples with float values attached and you want to only grab the ones where that value fits a certain condition The usual suspects are `>` `<` `=` `>=`, `<=`, and `!=` Right? Like this simple example

```sparql
PREFIX ex: <http://example.org/>

SELECT ?subject ?value
WHERE {
  ?subject ex:hasValue ?value .
  FILTER (?value > 3.14)
}
```

This looks straightforward it finds all `?subject` resources which have `ex:hasValue` that is greater than 3.14 Nothing too wild right The problem appears when you start needing more precise filtering Because floats are not always exact representations of decimal numbers we have to be cautious here

**The Floating Point Problem**

This is the core of the issue Float representations have a limited precision Meaning sometimes 3.14 won't exactly be 3.14 it may be more like 3.14000000001 or 3.13999999999 And that's where those `>` `<` `=` comparisons can bite you because what you expect to match and what actually matches might not be the same for example I was once working with geographic data longitude latitude things and even a difference at the 7th decimal place meant my query would not include a datapoint on the edge of a border and that took way too much time to find as it was never the data itself but me using "=" with floats

For instance that above filter even if it would look to work most of the time consider this case I've seen this exact issue occur in production I shit you not

```sparql
PREFIX ex: <http://example.org/>

INSERT DATA {
  ex:item1 ex:hasValue 3.14 .
  ex:item2 ex:hasValue 3.140000000001 .
  ex:item3 ex:hasValue 3.13999999999 .
}
```

If we run the previous `SELECT` query with that data only `ex:item2` will show up even though `ex:item1` looks like it should match right? So the problem is that strict equality comparison `>` in this case which will exclude similar looking values

What do we do then? Well it depends on the need really If you can round your data before inserting it that's ideal or sometimes you need the precision you cannot alter the source data in this case you should have a slightly different approach

**Using Tolerance When Filtering**

If you have very precise data and cannot alter it and also not losing results is crucial you'll need to handle this floating point issue with tolerance Instead of strict equality and inequality checks I use a range check based on a small "epsilon" value This is where you can specify how far a number can be to match your filter

```sparql
PREFIX ex: <http://example.org/>

SELECT ?subject ?value
WHERE {
  ?subject ex:hasValue ?value .
  FILTER(?value >= 3.13999 && ?value <= 3.14001)
}
```

Here I'm using a tolerance of `0.00001` this range check will include `ex:item1` `ex:item2` and `ex:item3` in the results depending on the data

And just to be clear you need to adjust `0.00001` based on the precision required on your specific problem If you are using 2 decimals precision a tolerance of 0.01 would be more than enough. If you are using a different measure you should adjust accordingly I usually look at my data with a script to see what precision is the minimal required

There is no one-size-fits-all `epsilon` You really need to know the data and what level of error is okay for your application I mean if you are using currency then 2 decimals for the most part is enough unless you are dealing with high amounts

**Type casting**

Another approach that people try which can lead to problems is type casting floats This is where the query engine needs to re-evaluate the float precision based on the casted type

For example if you have strings representing numbers in your data you might think it's smart to cast them to floats for comparisons and it is fine if your string data contains always the same precision or have a precision that is lower than the float type you want to cast to. Here is an example of type casting to compare numbers represented in strings this can be beneficial for many use cases including data cleaning

```sparql
PREFIX ex: <http://example.org/>

SELECT ?subject ?valueString
WHERE {
  ?subject ex:hasStringValue ?valueString .
  BIND(xsd:float(?valueString) AS ?valueFloat) .
  FILTER (?valueFloat > 3.14)
}

```

This should be used with caution because while it is useful for data cleaning it introduces additional processing and can cause issues if the string representation does not represent the same amount of precision as the float you want to use. So I'd use `FILTER` like the previous examples with epsilon values when possible and do more thorough data cleaning before any SPARQL queries

**Resources**

If you want to dive deeper I suggest taking a look at these:

*   **"What Every Computer Scientist Should Know About Floating-Point Arithmetic"**: This is a classic paper by David Goldberg it really lays out the nitty-gritty of floating-point representation. You will understand exactly why these problems happen
*   **"Numerical Recipes"**: This is a more practical book. It has some sections on how to deal with numerical issues in code that can give you a better sense of how to handle floats in general. It is an old book but still very relevant and very useful

Also here is a joke for you:
Why are floating-point numbers so bad at relationships? Because they're always drifting apart.

Anyway back to the topic it's important to always be very conscious about how floats are being used and what the tolerance should be because it's one of those things that is always unexpected until you have experience using it. I've lost countless hours over these things

I hope that helps you avoid some of the head-banging I've gone through Happy querying
