---
title: "cobol level 88 data type?"
date: "2024-12-13"
id: "cobol-level-88-data-type"
---

 so you're asking about COBOL level 88 data types right I get it been there done that more times than I care to remember Level 88s it's a COBOL thing and honestly if you haven't grappled with them you haven't really lived the COBOL experience let me tell you about it

So first up these aren't really *data types* in the way you'd think of like INT or STRING in other languages Instead level 88s are more like condition names associated with specific values of a regular COBOL data item Think of them as named booleans tied to specific data ranges or values within another variable

Back in the day I was knee deep in a legacy system you know the kind that runs on fumes and a prayer This particular beast had this huge data file with customer records and they had this field called `CUSTOMER-STATUS` yeah real original The thing is `CUSTOMER-STATUS` wasn't just like "active" or "inactive" it was this weird set of codes numbers you know like 1 for good standing 2 for delinquent 3 for suspended and so on

Now my problem was every time we had to check the status we ended up with these monstrous `IF` statements you know like

```cobol
       IF CUSTOMER-STATUS = 1
           PERFORM PROCESS-GOOD-STANDING-CUSTOMER.
       ELSE IF CUSTOMER-STATUS = 2
           PERFORM PROCESS-DELINQUENT-CUSTOMER.
       ELSE IF CUSTOMER-STATUS = 3
           PERFORM PROCESS-SUSPENDED-CUSTOMER.
       ...
```

Imagine that going on for like 10 different status codes it was a nightmare to read and debug plus every time they added a new status we had to go back and change all these `IF` statements talk about brittle code right Then my old mentor you know the guy who probably wrote the system in the first place showed me the light level 88s

What you do is first define the regular data item you're working with just as you normally would So let's say our customer status thing looks like this

```cobol
       05 CUSTOMER-STATUS  PIC 9.
```

That's a simple one-digit numeric field Right simple enough Now this is where the magic happens we can define level 88 condition names like this

```cobol
       88 GOOD-STANDING      VALUE 1.
       88 DELINQUENT        VALUE 2.
       88 SUSPENDED         VALUE 3.
```

See what's going on now Instead of that clunky `IF` statement I can now write

```cobol
       IF GOOD-STANDING
           PERFORM PROCESS-GOOD-STANDING-CUSTOMER.
       ELSE IF DELINQUENT
           PERFORM PROCESS-DELINQUENT-CUSTOMER.
       ELSE IF SUSPENDED
           PERFORM PROCESS-SUSPENDED-CUSTOMER.
       ...
```

Way cleaner right and way easier to read and understand Each level 88 is linked to a particular value in the `CUSTOMER-STATUS` variable When the value is equal to that value the condition is evaluated to true

And you know that's it at a fundamental level You could also define condition names for ranges like this imagine your customer credit score is stored like this

```cobol
   05 CREDIT-SCORE       PIC 9(3).
   88 LOW-RISK   VALUE 750 THRU 999
   88 MEDIUM-RISK   VALUE 650 THRU 749
   88 HIGH-RISK   VALUE 300 THRU 649
```

So now you can say

```cobol
   IF LOW-RISK
       PERFORM PROCESS-LOW-RISK
   ELSE IF MEDIUM-RISK
       PERFORM PROCESS-MEDIUM-RISK
   ELSE IF HIGH-RISK
       PERFORM PROCESS-HIGH-RISK
```

 I think you get the gist this reduces the amount of literal values you have in your code and makes things more readable and easier to maintain But it doesn't stop there Level 88 conditions can have multiple values like this

```cobol
       88 ACTIVE-CUSTOMER    VALUE 1 2 3.
```

Then if the status is 1 or 2 or 3 the `ACTIVE-CUSTOMER` condition will evaluate to true

Now some things that I've learned the hard way about level 88s you know from those long nights debugging COBOL code First when the same field has multiple level 88 conditions they are mutually exclusive In other words they can never be true at the same time for the same variable so beware overlapping conditions you need to structure your condition carefully

Second while they do make code readable they're only as good as the naming convention you use. Don't use arbitrary name use something meaningful that describes what the condition actually represents

Another tricky part is that you can actually use level 88s in the `SET` statement which you should not use often so imagine this situation

```cobol
       SET DELINQUENT TO TRUE.
```

That actually changes the value of `CUSTOMER-STATUS` to `2` since `DELINQUENT` is associated with the `VALUE 2` it is something that can be useful in specific cases but it's a little bit less clear you are assigning a condition variable instead of the variable that stores data be careful here

And another thing to keep in mind is that level 88 conditions are associated with a specific variable You cannot reuse them across different data items if you have a different status flag that needs conditions define new level 88 condition to that specific variable otherwise code can become hard to follow

One more gotcha and this one got me more times than I’d like to admit it's that level 88 conditions are dependent on the order of the data definition You cannot define a level 88 condition name for a variable that hasn't been defined yet so be mindful of declaration order in your data division.

I remember once I spent almost a day tracking down an issue because I mixed up the level 88 order in a copybook that was loaded in multiple programs a real headache

 time for a joke my boss kept telling me that my COBOL code was too complex I asked him “is this the best code you’ve ever seen” he said “Yes it's the best code I've ever seen” and we laughed because its so bad it’s the best I get it you had to be there Anyway I hope you see how level 88s are good now they improve readability make your code much less likely to be buggy reduce the amount of literal values in your code and just make COBOL a bit less painful to work with

If you want to dive deeper I’d recommend picking up a copy of “COBOL for Mainframes” by Grauer and Crawford or even the older “Structured COBOL” by Stern and Stern they are good resources and while old the principles still hold true for COBOL the books give a complete picture about COBOL and level 88s and many tricks that COBOL developers use in day to day activities. Also the official IBM COBOL documentation is a really good resource but it can be hard to navigate sometimes, and the book “COBOL Programming” by Paul Noll and Michael Murach gives a more hands on approach that I used during my early COBOL coding days.

So yeah that’s level 88s in a nutshell nothing too complicated once you get your head around the concept. Just remember readability maintainability and avoiding those monstrous `IF` statements and you'll be set You got this
