---
title: "postgresql round real number implementation?"
date: "2024-12-13"
id: "postgresql-round-real-number-implementation"
---

Okay so you wanna talk about rounding real numbers in PostgreSQL right Been there done that Got the scars to prove it Seriously this is one of those things that sounds simple on paper but then you dive in and it's like finding a whole new level of SQL weirdness

I remember back when I was working on this high frequency trading system for a startup Yeah a startup those days were rough We had this insane amount of financial data pouring in and we needed to round prices to a specific number of decimal places before storing them in the database Why the rounding Because float arithmetic is a nightmare and you get those tiny little fractional discrepancies that can wreak havoc when you're dealing with thousands of transactions per second

So first thought simple right PostgreSQL has a bunch of built-in math functions surely there's one that does exactly what I need Right? Wrong Well partially wrong actually There are the usual suspects like `round()` and `trunc()` They look great at first glance but they're not always what you want especially when you get into the weeds of floating point representation

Here’s what I found early on in my debugging process a quick and dirty try with the round() function

```sql
SELECT
    round(123.45678, 2);
```

That's the typical use case simple as it can be that will give you 123.46 as a result It rounds to two decimal places and most of the time this is fine but then comes the fun part with more decimal places and the fun that comes with floating point values

Then came the situation where you want to trunc to a specific amount of decimal places which is a bit different than rounding to them truncation simply cuts the value at the specific decimal place you request

```sql
SELECT
    trunc(123.45678, 2);
```

The above sql query returns the value 123.45 this is the truncated version of the number and this will be important in a moment

But the problem is not with those simple cases it's with the more complex cases the ones that always seem to come out of nowhere When you start dealing with floating point values that have a lot of decimal places or when you want rounding behavior that is not simple half-up rounding That's when the standard SQL functions start showing their limitations You might find that certain numbers don't round as expected due to the way floating point numbers are stored internally its kinda a known fact of working with computers

For example we had cases where we needed bankers rounding or sometimes also known as round-to-even rounding which the standard round() function doesn't do Its a specific type of rounding that prevents bias when rounding multiple numbers and that can be super important in statistical and financial computations I was young and didn't know what that was so I tried the simple function and it was not correct the first time

So after a lot of digging reading postgresql documentation and reading some of the papers that deal with numerical computations I found that there's no straightforward built-in way to do bankers rounding in PostgreSQL But there is a way to make it happen by a combination of functions and clever tricks Here's the basic idea you can’t use the built-in function so you have to build your own implementation

Here is one example using a custom implementation of round-to-even that I built that also takes an arbitrary amount of decimal places

```sql
CREATE OR REPLACE FUNCTION round_to_even(value NUMERIC, decimals INT)
RETURNS NUMERIC AS $$
DECLARE
    multiplier NUMERIC;
    rounded_value NUMERIC;
    fractional_part NUMERIC;
    even_check NUMERIC;
BEGIN
    multiplier := 10^decimals;
    rounded_value := TRUNC(value * multiplier);
    fractional_part := (value * multiplier) - rounded_value;
    even_check := MOD(rounded_value, 2);
    IF fractional_part > 0.5 THEN
       rounded_value := rounded_value + 1;
    ELSIF fractional_part = 0.5 THEN
    IF  even_check <> 0 THEN
       rounded_value := rounded_value + 1;
       END IF;
    END IF;
    RETURN rounded_value / multiplier;
END;
$$ LANGUAGE plpgsql;
```
This is a custom implementation using plpgsql where we are taking the number we want to round and the number of decimal places to round to The approach is to multiply the number by a power of 10 then we truncate it then if the value is in the halfway we check if it's even if not round up and finally divide by the power of 10 to get the final rounded number

Then to use it you call it like a normal function passing the value and the decimal places

```sql
SELECT round_to_even(123.455, 2)
```

This custom implementation does banker's rounding as expected It’s not a built-in function but it gets the job done I've used this for different types of numerical rounding in different contexts of the application I was working on

It’s always fun dealing with corner cases right?

I’ve seen a lot of people trying to use string manipulation to achieve this rounding It's a classic "if you have a hammer everything looks like a nail" approach String manipulation can work if you're doing really simple stuff but it’s super brittle for actual serious numerical work Avoid at all costs it will bite you sooner or later when you change the precision

Another thing to consider is the data type itself You can use real or double precision for your values or numeric which are different things Real and double precision are floating-point numbers which have their usual issues Numeric is a fixed point representation which is preferred when you're dealing with money because its exact in representation and it's something you might want to use when you want to store money values without error It is also more accurate and does not have the rounding problems that floating point has The choice of data type will impact the results you get when you use rounding functions

The key takeaway is this: PostgreSQL's rounding functions are useful but you need to understand their limitations You might need to use custom implementations like my function above for complex rounding requirements such as bankers rounding There are some specific cases that need to be taken into account when building more specific round functions such as ceiling and floor that are out of this scope

I highly recommend reading the PostgreSQL documentation on numeric types and the `round` function and its related functions Then try to look for academic papers on computer arithmetic and also on the representation of floating-point numbers Its a topic that requires some good research

Also be very careful when working with financial systems especially with float values. Use numeric data types for storing values that represent money and custom rounding if needed.

Always test with many different values and corner cases to avoid problems in production You are never done with this testing process until you know its rock solid and this can take a considerable amount of time to be perfect But perfection is something that is important in these cases

Hope this helps you with your rounding problem It's not a fun one but at least I can say I’ve solved it
