---
title: "How to do Query criteria in a subclass on Grails 3?"
date: "2024-12-14"
id: "how-to-do-query-criteria-in-a-subclass-on-grails-3"
---

alright, so, you're asking about query criteria within a subclass using grails 3, yeah? i’ve been there, more times than i care to count. it can be a bit fiddly, especially when you’re moving from simpler domain models to more complex inheritance hierarchies. let me break down what i’ve learned over the years, and share some code snippets that might help you out.

basically, the challenge arises from how grails, or more precisely gorm, handles inheritance in its criteria queries. when you have a superclass and a subclass, and you try to use `createCriteria()` directly on the subclass, it can sometimes lead to unexpected behavior, specifically around the filtering applied to the superclass properties. the criteria often seem to "leak" down or not be applied correctly. i recall a particularly nasty incident in version 2.x, back when i was working on that legacy project, it caused a bug that took me a whole day to find and fix that, believe me, taught me a few things about domain models and inheritance.

the core issue comes from the fact that gorm, by default, uses table-per-class inheritance strategy. this means that a single table usually holds all the properties from the superclass and the subclass and grails will use a discriminator column (usually `dtype` by default) to differentiate between the different types of records in that table. now, when you create a criteria query on the subclass, you're essentially asking gorm to select rows that correspond to that specific `dtype` value but also potentially apply conditions on the superclass columns. if your criteria are not constructed correctly, it can be confusing and frustrating to deal with.

let's get to some examples. suppose you have a superclass `animal` and a subclass `dog`.

```groovy
class animal {
    string name
    string color
    static constraints = {
        name blank:false
        color blank:false
    }
}

class dog extends animal {
    string breed
    static constraints = {
        breed blank:false
    }
}
```

now, imagine you want to find all dogs that are black. a naive approach might look like this:

```groovy
def blackDogs = dog.createCriteria().list {
    eq('color', 'black')
}
```

this looks simple and for some simple use cases, it might work. but if you need more complex criteria, things can get tricky. this approach directly queries the dog table and includes the subclass’s implicit type discriminator. the discriminator is important because the database uses it to identify whether the returned result is actually a dog object. this helps gorm instantiate the correct subclass from the database results. it's implicit and automatic most of the time.

for more complex cases, you could benefit from using hql or gorm projections, and you might want to add conditions on superclass and subclass columns simultaneously. let's try a more detailed example, say, you want to find all black dogs of a specific breed. the criteria approach can be a bit verbose.

```groovy
def specificDogs = dog.createCriteria().list{
    and {
        eq('color','black')
        eq('breed','golden retriever')
    }
}
```

this example uses an explicit and clause to apply both conditions, one for a superclass property and the other on the subclass property. it works but is more verbose. in my projects i used to prefer to use gorm projections for these kind of specific situations, specifically when i had to add aggregations and filtering using projections because they are less verbose and more powerful in my experience, although they can be a bit complex to learn at first.

another common situation arises when you're dealing with many-to-one or one-to-many relationships and you are trying to filter based on those associated objects from the superclass to a sub class. for example, let's add a new domain class called `owner` to our example and a `many-to-one` association with the `animal` domain class:

```groovy
class owner{
    string name
    static hasMany = [animals: animal]
    static constraints = {
        name blank:false
    }
}
```

and add it to our animal and dog domain class, so we have a many-to-one association:

```groovy
class animal {
    string name
    string color
    owner owner
    static belongsTo = [owner]
    static constraints = {
        name blank:false
        color blank:false
        owner nullable:false
    }
}

class dog extends animal {
    string breed
    static constraints = {
        breed blank:false
    }
}

```

now imagine that you need to find all golden retriever dogs that are owned by an owner named "john". you will now need to do join queries, and the criteria builder can get very verbose.

```groovy
def specificDogsWithOwners = dog.createCriteria().list{
    owner{
         eq('name','john')
    }
    eq('breed','golden retriever')
}
```

again, this will work, but if you try to do aggregations on the result, or maybe filtering by an associated object of the `owner` domain class then your code will become more and more difficult to maintain because of verbosity. you could always use more complex hql queries or use gorm projections to make these queries cleaner and easier to maintain. for example the previous query using a projection would look like this.

```groovy
def specificDogsWithOwnersProjection = dog.withCriteria{
    createAlias('owner', 'ownerAlias')
    projections {
        property("id")
    }
    eq('breed','golden retriever')
    eq('ownerAlias.name', 'john')
}.list()
```

as you can see, the projection allows you to be more explicit about what information you want and how to apply your filtering conditions. it can also simplify queries when using aggregates like sum, min, max, etc. also, the use of `createAlias` makes filtering on the associated `owner` class easier to express and it's easier to manage. this example returns just a list of id's but it can be easily modified to return more complex or simpler results, for example domain objects if you replace the property projection with `root()`.

remember also that gorm and hibernate will optimize your queries by doing things such as adding indexes automatically on your database columns, so don't worry too much about performance and focus on making the code clean and easy to maintain.

regarding resources, i'd suggest checking out the official grails documentation, especially the sections on gorm and querying. it is usually very well written and updated. the hibernate documentation is also very useful, since gorm is built on top of hibernate. the book "grails in action" also has some good chapters on this. i know books may feel a little old school nowadays, but they often explain the concepts and theory very thoroughly. another good resource is the "hibernate in action" book since it goes deep into the hibernate internals and how the query language is handled by the library. and always remember to check the release notes of each grails and gorm version, because features, or how they work, often change.

i know that these domain modeling situations sometimes can be a pain. there was this one time that i spent a whole day figuring out why my criteria was returning more results than expected because of some silly mistake i made when configuring a domain class relationship. i remember having to spend the whole night debugging. and this reminded me of the time i went to a tech conference, and the speaker was doing a live demo and his code crashed because he misspelled a property name, it was very embarrassing and funny, it happens to the best of us, hahaha.

so yeah, remember that persistence and querying can be a tricky process sometimes. keep an eye on your gorm and hibernate configurations. and always write tests, that's very important to avoid this kind of situations. i hope this helps you to understand how grails 3 handles query criteria in subclasses. feel free to ask if you have any more questions.
