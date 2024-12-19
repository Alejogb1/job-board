---
title: "How to resolve `member method 'java.lang.Object, add/1' not found` in painless?"
date: "2024-12-15"
id: "how-to-resolve-member-method-javalangobject-add1-not-found-in-painless"
---

alright, so, you're hitting the `member method [java.lang.Object, add/1] not found` error in painless, right? i've been there, trust me. this particular error pops up when you're trying to use the `add()` method on a java object that doesn't actually have it. sounds straightforward, but it can get tricky when you're deep into painless scripts inside elasticsearch. let me break down what's probably happening and how to get around it.

first off, painless is designed to be a safe, sandboxed scripting language. it doesn't give you the full, unrestricted access to all java classes that java itself does. it’s got its guardrails. that said, this error generally surfaces when we're attempting to invoke `add()` on something that's not a list or a set (like, maybe, a plain old `java.lang.object`). in other words, the error message is quite literal. it's telling you exactly what it can’t find: a method named `add` that takes one argument on an object that is a plain old java object instance.

from the look of it you are probably doing something similar to the following:

```painless
def my_object = new Object();
my_object.add("some_value"); // this will cause the error!
return my_object;
```

this code will throw the error you mentioned precisely because java `object` class does not have that method, the error is telling you the code is trying to call `add` on `java.lang.object` which doesn't exist.

now, i’ve seen this crop up in various scenarios. one time, i was working on a complex ingest pipeline for log data. i was attempting to accumulate values from several nested fields into a single list to later be used on aggregations. my initial (and incorrect) approach was to dynamically create a plain java object and attempt to append the values using the add function. it failed just like your example.

it took me a while, a few debugging sessions (and many cups of coffee) to figure out that i was misusing the base `java.lang.object`.

what you should be doing, and i highly recommend you do this. you should explicitly use a list or set data structure if you need to collect values. painless gives you access to `java.util.arraylist` and `java.util.hashset`, which are more suited for that purpose. if you need a list for storing values (and it does not need to be unique) go for the arraylist and if you want to store unique values use the hashset. i have found that is the most common case when one hit that error.

here’s how you'd properly use `arraylist`:

```painless
def my_list = new ArrayList();
my_list.add("value1");
my_list.add("value2");
my_list.add(123);
return my_list;
```
this will create a new list and append different types of elements to it.

or use a `hashset` for collecting unique values:
```painless
def my_set = new HashSet();
my_set.add("value1");
my_set.add("value1"); // this value is a duplicate it will not be added again.
my_set.add("value2");
my_set.add(456);

return my_set;
```

this code will create a new set, where duplicate values are not allowed. this is good to be used when you want only unique values.

note that the error also occurs when trying to call `add` on a specific type that doesn't have the method, like an `integer`. just keep in mind what you are dealing with and which type you are working with.

the key takeaway here is that you need to be aware of the type you're working with. don't assume that every java object can use the `add()` method just because you are used to other programming languages. painless is strict about type safety, which is a good thing because it prevents lots of errors.

now, let's talk about resources. there aren’t many books that focus exclusively on painless (i wish there were more). however, the official elasticsearch documentation is your best friend. the "painless scripting language" section within the elasticsearch reference docs is incredibly detailed and should be your primary source. it has detailed documentation on data structures you can use within painless. pay particular attention to the parts that talk about allowed classes and methods. there are also examples, which can be quite handy. also, another book i recommend, not exclusively about painless scripting, but is the book "elasticsearch: the definitive guide" is an invaluable resource for learning about elasticsearch as a whole and how scripting fits within it. you should definitely look at the book for deeper understanding of the whole ecosystem of elasticsearch.

i spent way too long trying to figure out how to extract particular values from nested maps to build dynamic aggregations and got tripped by this error, because i was not reading carefully the docs about what methods i could call on the data i was extracting from the json objects. then i decided to go through the official painless documentation and i found that the elasticsearch team had already thought of all these edge cases and had a method already prepared. i felt a bit stupid at first, but we all learn from our mistakes. it’s part of the job, right? you may even laugh about it later, i did! it's like the time i was trying to fix a bug that turned out to be a missing semicolon... that was one intense debugging session for something so simple.

anyway, that’s pretty much it. you need to use the right data structures to store your values, which means using lists or sets. be mindful of the types and the methods they have. consult the documentation to know which methods are available to the type you are using. and i think you will be good to go.
