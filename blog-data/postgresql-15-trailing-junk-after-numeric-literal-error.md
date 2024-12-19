---
title: "postgresql 15 trailing junk after numeric literal error?"
date: "2024-12-13"
id: "postgresql-15-trailing-junk-after-numeric-literal-error"
---

Alright so you're getting the "trailing junk after numeric literal" error in Postgresql 15 right Been there done that Let me tell you this isn't a fun error to chase down mostly because it's not always obvious why its happening at least initially I've seen this pop up in so many different contexts over the years I almost have a PTSD trigger response when I see it

First off lets be clear what's going on Postgres is a bit pedantic when it comes to data types It wants everything to be in its place and playing by the rules When it sees something it thinks is a number it expects that number to be complete no extra bits of letters or symbols clinging onto it like barnacles on a ship Now if you happen to throw something at it like "123abc" it's gonna scream "trailing junk after numeric literal" That `abc` part is the unwanted trailing junk

Okay so what causes this in the real world from my experience I'd say there are a few common culprits One of them is data cleaning issues think CSV or Excel imports often times data isn't super clean you might have extra spaces or letters hanging around after what you think is a number Maybe someone used a 'k' or 'm' to indicate thousand or million which is completely understandable for a human but the SQL parser is definitely not a human In such cases the parser goes haywire as it sees the characters after what it interprets as numeric literal and thus starts the yelling game

Another big one is dynamic SQL generation especially when string concatenation is involved I’ve personally spent long nights staring at code where a query is being built on the fly and one silly little string literal got mixed into a number and caused this error You see the number string getting appended to some text and suddenly boom postgres throws its tantrums In this case you don't actually pass "123abc" to Postgres as a literal it does its due to a concatenation of different string and variables so debugging this case can be much more time consuming

Then there are also more obscure cases with type casting sometimes things just don't cast the way you expect or some library you are using to connect to Postgres may be doing weird things behind the scenes and that may produce the issue for example in the past I once had this type issue because of a ORM mapping of JSON string to a numeric type this caused me to spend about half a day debugging my ORM and then I ended up giving up on using that particular mapping for that case so I just ended up dealing with JSON and then doing the parsing myself

Now lets dig into some practical scenarios and examples Lets assume you have a table with data like this

```sql
CREATE TABLE data_table (
    id SERIAL PRIMARY KEY,
    value_col TEXT
);

INSERT INTO data_table (value_col) VALUES
('123'), ('456  '), ('789x'), ('   1011'), ('invalid');
```
Simple right Then you have a simple query

```sql
SELECT * FROM data_table where CAST(value_col AS NUMERIC) > 100;
```
This query will break because of course you are trying to convert values like '789x' or even `invalid` to numeric This error is obvious in that example but imagine this in a big codebase and in a production environment good luck debugging that In general it's very important to have good logging and error handling in place otherwise you might waste your time in similar scenarios

Now lets see some examples and how to fix them or try to bypass this error

**Example 1: Trimming Whitespace**
A very common case is when you have spaces before or after the numbers In such case you need to get rid of the extra white spaces using the `TRIM()` function

```sql
SELECT * FROM data_table WHERE CAST(TRIM(value_col) AS NUMERIC) > 100;
```

This will work because the `TRIM` function removes leading and trailing whitespaces before the numeric cast occurs

**Example 2: Using Regular Expressions**

Now lets say you have actual non numeric stuff in your data and that you just want the numbers In this case we can use regular expressions to extract only the numbers

```sql
SELECT *
FROM data_table
WHERE CAST(regexp_replace(value_col, '[^0-9.]', '', 'g') AS NUMERIC) > 100;
```
This uses `regexp_replace` function with pattern `[^0-9.]` which means everything that is not number or a dot will be replaced with an empty string effectively stripping the non numeric characters from the original string before converting it to numeric

**Example 3: Conditional Casting**
If your data is really messy and you want to avoid exceptions in the first place you can use conditional expressions and avoid casting in the first place It’s good for data exploration or when you are unsure of the data that you have

```sql
SELECT
    id,
    value_col,
    CASE
        WHEN value_col ~ '^[0-9]+[.]?[0-9]*$'
        THEN CAST(value_col AS NUMERIC)
        ELSE NULL
    END as numeric_value
FROM data_table
WHERE
    CASE
        WHEN value_col ~ '^[0-9]+[.]?[0-9]*$'
        THEN CAST(value_col AS NUMERIC)
        ELSE NULL
    END > 100;
```
This is the most robust example in my opinion In this case we are using a regular expression to check if the string contains only numbers before we are casting to numeric type If the string does not contain only numeric characters then `NULL` value will be returned by the case statement This ensures that no exception is triggered and the SQL can run successfully of course this can also return `NULL` values and will need to be handled properly if it was necessary for the problem at hand

When dealing with these kind of errors you need to be paranoid assume that your data is dirty and someone has put something unexpected in your data that will break the assumptions made by the code So having lots of sanitization checks and guard conditions is a good practice In general I follow the principle of ‘defensive programming’ where I try to protect my code against bad data as much as possible

Oh by the way here is a joke why was the SQL query so sad Because it had too many joins and no one was able to `SELECT` its happiness

Okay okay back to the technical stuff For more resources on data cleaning and SQL I would suggest checking the book "SQL and Relational Theory" by C.J. Date its a good theoretical base for relational database and "Effective SQL" by John L Viescas is more practical and contains lots of best practice tips and tricks and also some debugging strategies that may be useful for this situation. And if you want to go deeper into Regular Expressions in PostgreSQL I suggest reading the Postgresql manual directly there is a specific chapter on Pattern Matching.

So yeah to recap the "trailing junk" error in Postgresql is quite common but it usually stems from data cleaning issues or type mismatches Use the methods I have suggested `TRIM()` regular expressions and conditional casting to try and resolve such issues. Remember good error handling and defensive programming are your best friends in these cases and good logging can help you to track down the error in the first place Good luck debugging.
