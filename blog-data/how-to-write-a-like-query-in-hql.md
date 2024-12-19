---
title: "how to write a like query in hql?"
date: "2024-12-13"
id: "how-to-write-a-like-query-in-hql"
---

Alright so you're hitting that HQL like query wall right Been there man been burned by it too Let me tell you I've wrestled with Hibernate and its quirks more times than I care to admit specifically this very like query thing back in my Java EE days oh god the pain

Let's get this straight first HQL ain't SQL It's a query language that speaks in object terms not table rows So thinking SQL-like `%` wildcards directly in HQL is gonna give you headaches and potentially some very very long debugging sessions I've spent hours staring at logs for that kind of mistake so i know what i'm saying Trust me on this one I learned the hard way

See back in 2009 when I was building this online book store platform (yeah I'm that old) we had to implement search functionality and we started with the naive approach direct SQL like queries thinking we would be heroes oh boy was that naive But the performance was terrible and the maintenance became a total nightmare every time schema changed so we needed to rewrite a lot of our queries which involved refactoring tons of code It got to the point where I could smell the JBoss server burning just from a simple user search I swear That's when I discovered the joy that is HQL and its very very specific way to handle string comparisons

The fundamental issue is that HQL doesn't understand `%` directly you have to use the `like` keyword which is fine but the wildcarding is done using SQL wildcards in a bit different way than direct SQL For example if you want to find all users with a name that starts with "John" you can't just write `WHERE user.name LIKE 'John%'` in HQL you use the `like` keyword with single quotes and `%` is still the wildcard symbol but you also have to use it in single quotes

```hql
FROM User user WHERE user.name like 'John%'
```

This is the basic way to do it It is simple and direct nothing fancy And this should be the go to method for simple cases

But then came the problem of case insensitivity remember the good old days where databases where case sensitive and developers had to deal with it? We had users entering their names with inconsistent case some with all caps some with all lower case and some mixed cases The initial naive way of querying with this was very bad and we were getting a lot of support tickets which meant for me and my team it meant fixing the query every two or three hours. so i was in panic for most of the day. So I learned we needed something better than this so after some good research I found out that HQL provides some help by using the `lower()` function in HQL for both sides of the query. This allows you to make case-insensitive searches pretty easily

```hql
FROM User user WHERE lower(user.name) like lower('john%')
```
Now we can search for "john" "John" "JOHN" and so on all at once this was a good day for me indeed I felt like i had won an olympic gold medal for this it was an incredible feeling

Now for the real tricky stuff what if the pattern you want to match includes the wildcard character itself? I mean what if the user wants to search for name with a literal `%` sign in it? Back in my retail software days i had to deal with a data set that had percentage of discounts in its description. I learned to not assume anything and that users are unpredictable. At first I thought this was easy peasy like just adding a literal wildcard to the search well it was not! Hibernate would interpret the `%` as a wildcard character and it would not work I remember i spent a whole weekend debugging just to figure it out

So the HQL solution for this is to use an escape character. You need to tell HQL to interpret that wildcard as just a literal and not a wildcard. The default escape character is backslash but you can define your own as well if you want. So if I want to search for any string containing for example `10% off` I have to add a backslash before the percent sign.

```hql
FROM Product product WHERE product.description like '%10\%%' ESCAPE '\'
```
This is when I realized that sometimes software engineering feels like fighting a hydra each head you cut another one grows This specific case was so annoying to debug because the queries were working as expected and the only thing wrong with it was that the literal `%` was not escaping.

Now let me give you some food for thought as well just because you can use `like` all the time it doesn't mean that you should If you have full text search requirements you should be considering a full-text search engine like Lucene or Elasticsearch that can integrate well with Hibernate as well. I remember the day that we migrated our book store application to Elasticsearch it was a totally different experience performance wise the searches became instantaneous and the full-text search capabilities were so so powerful so please don't think about like queries as a golden hammer for every search requirement

I also recommend you delve into the Hibernate documentation its the bible for Hibernate developers and there are excellent resources like "Java Persistence with Hibernate" by Christian Bauer and Gavin King it's a lifesaver for understanding all the inner workings of Hibernate including HQL and also "Hibernate in Action" by Christian Bauer and Gavin King another essential read for deepening your knowledge in hibernate and also you should really go through the official Hibernate website they have all the things that you need to know about Hibernate.

And a little joke for you why was the java developer always broke? Because he used up all his cache.

I hope this helps You've now seen my battle scars from the HQL trenches. Go forth and query responsibly my friend and remember to log everything it will save your life one day I promise
