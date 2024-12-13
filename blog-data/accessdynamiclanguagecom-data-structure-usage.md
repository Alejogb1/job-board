---
title: "accessdynamiclanguage.com data structure usage?"
date: "2024-12-13"
id: "accessdynamiclanguagecom-data-structure-usage"
---

Okay so you're asking about using data structures with dynamic languages right I've been there man trust me it's a wild west out there when you jump from a statically typed world into something like Python or JavaScript or Ruby I spent a few years early in my career battling this stuff trying to optimize things in a Python project that was just exploding with data and slow query times

Let's get real first a dynamic language's flexible nature is both its blessing and its curse You can throw anything anywhere but that freedom can bite you hard when it comes to performance or maintainability because unlike say in Java or C++ the data types are not checked at compile time it's all runtime which means issues pop up only after you run the darn thing it's like debugging in the dark sometimes

My journey with this began with a huge data ingestion pipeline for a startup that shall not be named We were collecting tons of user data lots of nested dictionaries and lists basically JSON hell It was a data scientist's dream and an engineer's nightmare We started using lists for pretty much everything which looked simple initially but oh boy that was not a good plan

I remember one specific case where we had a list of user sessions and each session had a bunch of actions It looked something like this initially

```python
user_sessions = [
    [
        {'action': 'login', 'timestamp': 1678886400},
        {'action': 'view_product', 'timestamp': 1678886430, 'product_id': 123},
        {'action': 'add_to_cart', 'timestamp': 1678886460, 'product_id': 123}
    ],
    [
      {'action': 'login','timestamp': 1678888400},
       {'action':'search', 'timestamp': 1678888430, 'term': "shoes"}
    ]
]
```

The problems started showing up when we tried to filter these sessions based on user actions We had to loop through the entire list then loop through each session which was an N squared operation a pure nightmare for our server load time It was getting so slow we started seeing timeouts for some queries the app started failing all over the place and that was not a good Friday afternoon at all

So here's what I learned from all the pain and suffering

First off lists are your go-to for ordered collections but their O(n) lookup can kill you especially if you're searching based on non-index values We're not dealing with small amount of records here we are talking about millions of them Second and maybe even more importantly always consider what you are doing with the data are you searching are you adding are you checking for membership this will guide you into choosing the right tools for the job

For quick lookups dictionaries are your best friend in dynamic languages They offer O(1) average time complexity for lookups which is orders of magnitude faster than looping through a list So we refactored our user sessions data so that we could index it based on let's say user ID it would be like this

```python
user_sessions = {
    'user123': [
        {'action': 'login', 'timestamp': 1678886400},
        {'action': 'view_product', 'timestamp': 1678886430, 'product_id': 123},
        {'action': 'add_to_cart', 'timestamp': 1678886460, 'product_id': 123}
    ],
    'user456': [
      {'action': 'login','timestamp': 1678888400},
       {'action':'search', 'timestamp': 1678888430, 'term': "shoes"}
    ]
}
```

This change alone drastically improved query times we could just go `user_sessions['user123']` and get the session data immediately instead of having to crawl through every session object in a list. The data scientists were happier because their reports were being generated quicker and the users were happier because the app stopped lagging as much it was all smiles

Another common mistake I've seen is overusing lists when a set would be way more efficient for example if you're just trying to check if an element exists you could do it with lists but that is inefficient if the list is large instead use sets they have O(1) membership checks like this

```python
user_ids = {'user123', 'user456', 'user789'} # set

if 'user123' in user_ids: # fast membership check
    print("user 123 exists")
```

Sets are great for things like finding unique items or checking for membership super quick in general so avoid using lists for such operations it might look simple at first but in the long run its not the ideal choice

Now let's talk about nested data structures because they are another common source of performance problems in dynamic languages I had to deal with that specifically when we started having users with lots of interactions and the user object itself became a beast

We were initially using nested dictionaries for user data something like this

```python
user_data = {
    'user_id': 'user123',
    'profile': {
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'location': {
            'city': 'New York',
            'country': 'USA'
        }
    },
    'interactions': [
        {'type': 'view', 'timestamp': 1678886400, 'product_id': 123},
        {'type': 'add_to_cart', 'timestamp': 1678886460, 'product_id': 123}
    ]
}

```
The key thing I learned with these nested structures is to avoid deep nesting like that whenever possible It can make your code harder to read and manipulate and if you need to access things you end up with long paths like `user_data['profile']['location']['city']` which is not efficient at all in the long run and hard to maintain instead I'd recommend splitting the data into separate structures or normalizing data using flat structures if needed

Another trick is to use data classes or named tuples to define the structure of your objects which in dynamic languages still gives you some clarity when accessing attributes it's not a full static type safety but it is better than nothing at all

Now for a bit of advice and a little bad joke I had to do some code golfing when I was refactoring one of the data structures and my colleague said "are you refactoring the code or just doing some magic" i said well they are the same thing aren't they

Now for some useful resources to learn more I'd recommend "Introduction to Algorithms" by Thomas H Cormen et al this book is your go to bible when dealing with data structure algorithms and complexity issues and another great book is "High Performance Python" by Micha Gorelick and Ian Ozsvald this one is all about using Python efficiently especially when dealing with data this book helped me a lot so check them out

Okay so to summarize choose the right data structure for the job think about your operations if you are doing lots of lookups use dictionaries or sets if you want order then use lists use data classes or named tuples to structure your data avoid deep nesting when possible and remember that optimizing is always a process and you might have to refactor your data as your project grows because it probably will so you should expect that but with the proper data structure things will be way simpler
