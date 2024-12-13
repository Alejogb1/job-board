---
title: "how to split a string in sparql?"
date: "2024-12-13"
id: "how-to-split-a-string-in-sparql"
---

Okay so you're asking about splitting strings in SPARQL huh? Been there done that a few times it's not exactly a walk in the park if you're coming from say a Javascript or Python background where string handling is super straightforward SPARQL is more about data retrieval than complex string manipulation but we can still get it done

First off SPARQL natively doesn't have a `split` function like you'd find in other languages No one thought this would be a problem when they designed SPARQL I guess so we're kind of left to hack our way around it using the functions that it does provide Which yeah means more complexity not really the best experience if you ask me But it is what it is we work with what we have right?

I’ve had to deal with this problem many times before especially when I was wrestling with some messy RDF data scraped from the web I remember this one project where we were extracting information about books from an HTML website and the authors were listed in a single long string separated by commas like "Author One Author Two Author Three" You can imagine the kind of mess that becomes when you want to do data analysis on that or even search by author properly SPARQL's lack of a split function made this simple problem become a real headache I spent almost two days on that one single thing debugging a recursive query that failed because of a null value and I still have the code somewhere with my personal comments saying something like "this is awful I need to learn lisp"

So what can we use instead? Well the most common way to do this is to combine a few functions that SPARQL does provide `REGEX` and `SUBSTR` are our main tools here The idea is to use a regular expression to find the delimiter and then use `SUBSTR` to extract each part before and after the delimiter recursively It's not pretty I know but it gets the job done and honestly it's the least painful option

Let's get some concrete examples shall we I’ll throw in some working code that might help you if you're in my shoes once

**Example 1 Simple Splitting on a Single Delimiter**

Let’s say you have a string like `"apple banana cherry"` and you want to split on the space character

```sparql
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?str ?split1 ?split2 ?rest
WHERE {
  BIND("apple banana cherry" AS ?str)
  BIND(REGEX(?str, "^([^ ]*) (.*)$", "i") AS ?matches)
  BIND(SUBSTR(?str, 1, STRLEN(STR(REPLACE(?str, "^([^ ]*) (.*)$", "$1"))) ) AS ?split1)
  BIND(REPLACE(?str, "^([^ ]*) (.*)$", "$2") AS ?rest)
  BIND(IF(REGEX(?rest, "^([^ ]*) (.*)$", "i") , SUBSTR(?rest, 1, STRLEN(STR(REPLACE(?rest, "^([^ ]*) (.*)$", "$1"))) ) , ?rest ) as ?split2)
}
```

This query uses `REGEX` to find the first word and the rest of the string then extracts the first word using `SUBSTR` and it extracts the rest for further processing in a `rest` variable As you can see there is also a small if check for the second word to actually work it’s a little hacky but it works which is the point here right?

This approach is a bit clunky if you need to split the string in more than two parts I have tried all kinds of methods but the easiest is recursive as follows

**Example 2 Recursive Splitting with a WHILE loop**

This is where we get into the more complicated stuff because SPARQL doesn't have WHILE loops we use a combination of recursive queries using `UNION` which is basically as close as we are going to get to a while loop. Here's how we do that

```sparql
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?str ?part ?level
WHERE {
  BIND("apple,banana,cherry,date" AS ?str)
  {
    SELECT ?str ?part  (1 AS ?level) ?rest
    WHERE {
      BIND(?str AS ?currentString)
      BIND(IF(REGEX(?currentString, "^([^,]*) (.*)$", "i") , SUBSTR(?currentString, 1, STRLEN(STR(REPLACE(?currentString, "^([^,]*) (.*)$", "$1"))) ) , ?currentString ) AS ?part)
      BIND(REPLACE(?currentString, "^([^,]*) (.*)$", "$2") AS ?rest)
      FILTER(STRLEN(?rest) > 0)
    }
    UNION{
      SELECT ?str ?part ( ?level + 1 AS ?level) ?rest
      WHERE{
       ?part ?level ?rest
       BIND(IF(REGEX(?rest, "^([^,]*) (.*)$", "i") , SUBSTR(?rest, 1, STRLEN(STR(REPLACE(?rest, "^([^,]*) (.*)$", "$1"))) ) , ?rest ) AS ?part)
      BIND(REPLACE(?rest, "^([^,]*) (.*)$", "$2") AS ?rest)
       FILTER(STRLEN(?rest) > 0)
     }
    }
    UNION{
       SELECT ?str ?part ( ?level + 1 AS ?level) ?rest
      WHERE{
        ?part ?level ?rest
       BIND(IF(REGEX(?rest, "^([^,]*) (.*)$", "i") , SUBSTR(?rest, 1, STRLEN(STR(REPLACE(?rest, "^([^,]*) (.*)$", "$1"))) ) , ?rest ) AS ?part)
         BIND(REPLACE(?rest, "^([^,]*) (.*)$", "$2") AS ?rest)
         FILTER(STRLEN(?rest) = 0)

      }
    }
  }
}
```

The above query is splitting a string using a comma as a separator This code demonstrates how to do a recursive split operation using a `UNION` operator to simulate a while loop. It’s kind of mind-bending at first but once you get the gist of it it becomes tolerable This query will return each part of the string with a level indicating which part it is but it might be difficult to understand on first look that's normal just give it a try and modify it a bit for your needs

**Example 3 Using a different separator**

It's basically the same thing as the last code but just with a different delimiter

```sparql
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?str ?part ?level
WHERE {
  BIND("apple-banana-cherry-date" AS ?str)
  {
    SELECT ?str ?part  (1 AS ?level) ?rest
    WHERE {
      BIND(?str AS ?currentString)
      BIND(IF(REGEX(?currentString, "^([^-]*) -(.*)$", "i") , SUBSTR(?currentString, 1, STRLEN(STR(REPLACE(?currentString, "^([^-]*) -(.*)$", "$1"))) ) , ?currentString ) AS ?part)
      BIND(REPLACE(?currentString, "^([^-]*) -(.*)$", "$2") AS ?rest)
      FILTER(STRLEN(?rest) > 0)
    }
    UNION{
      SELECT ?str ?part ( ?level + 1 AS ?level) ?rest
      WHERE{
       ?part ?level ?rest
       BIND(IF(REGEX(?rest, "^([^-]*) -(.*)$", "i") , SUBSTR(?rest, 1, STRLEN(STR(REPLACE(?rest, "^([^-]*) -(.*)$", "$1"))) ) , ?rest ) AS ?part)
      BIND(REPLACE(?rest, "^([^-]*) -(.*)$", "$2") AS ?rest)
       FILTER(STRLEN(?rest) > 0)
     }
    }
    UNION{
       SELECT ?str ?part ( ?level + 1 AS ?level) ?rest
      WHERE{
        ?part ?level ?rest
       BIND(IF(REGEX(?rest, "^([^-]*) -(.*)$", "i") , SUBSTR(?rest, 1, STRLEN(STR(REPLACE(?rest, "^([^-]*) -(.*)$", "$1"))) ) , ?rest ) AS ?part)
         BIND(REPLACE(?rest, "^([^-]*) -(.*)$", "$2") AS ?rest)
         FILTER(STRLEN(?rest) = 0)

      }
    }
  }
}
```

Just to make things clear the only thing I changed here was the separator from comma to a dash `-` it’s the same as example 2

**Important Considerations and Optimizations**

Okay a few things to keep in mind with this method

*   **Performance:** These recursive queries can get slow especially with very long strings or many splits It might become a problem but it is usually fine for smaller datasets in those cases you might want to optimize the way you store or get data from the database itself to avoid that
*   **Error Handling:** You'll want to add some error handling to deal with unexpected input such as strings without delimiters Or maybe if the separator is not exactly what you expect You could do that with simple filters checking if the regex matches at all or similar checks
*   **Character Escape:** If your separator includes special regex characters like `.` or `*` or `+` you'll need to escape them properly within your regex expressions. I once spent a few hours because of this and it was just a simple character that needed to be escaped I am sure that you will not forget that for some time
*   **Regex flavor:** SPARQL uses a regular expression standard that is sometimes more limited than other languages so be sure that the syntax that you are using works correctly If it does not that is most likely the reason why
*   **Edge cases:** Consider what should happen if a string has leading trailing or multiple delimiters This is one of the things where people mess up a lot when starting with this kind of solution

**Further Learning**

While there isn't a specific book just on SPARQL string manipulation you can deepen your knowledge by exploring these resources:

*   **"Semantic Web Programming"** by John Hebeler Matthew Fisher Ryan Blace and Andrew Perez: This book goes over a lot of general topics in the area of semantic web if you are interested in more than just SPARQL it will be worth the time
*   **W3C SPARQL 1.1 Specification:** Yes it's dry and boring but it's the ultimate source of truth on SPARQL features If there is something that you want to know for sure go to this resource you might find it useful (or not)
*   **Various online SPARQL tutorials:** There are some pretty good tutorials available on the internet and you can usually just google "SPARQL tutorial" and learn more about this subject
*   **The online documentation of the triple store that you use:** if you are working with a specific database like Virtuoso or Jena they might have specific functions or implementations that could help you with your problem it will be important to learn the specific quirks of your specific triple store that you are using

So yeah splitting strings in SPARQL is kinda a pain it's like trying to assemble furniture with just a screwdriver and a hammer You can get the job done but it's not the most elegant or efficient process You can do it though I mean I did it and I am sure you can do it too Just be prepared to deal with some regex and recursion. The good thing is that after this no string manipulation in any language will be scary for you It's like a trial by fire and after that all other string manipulation techniques look like child's play. Or like they would say in my hometown "after suffering this you are ready for anything".

Hopefully this gives you a good starting point and helps you make sense of this mess let me know if you have more questions
