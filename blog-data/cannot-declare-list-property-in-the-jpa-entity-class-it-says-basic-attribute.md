---
title: "cannot declare list property in the jpa entity class it says basic attribute?"
date: "2024-12-13"
id: "cannot-declare-list-property-in-the-jpa-entity-class-it-says-basic-attribute"
---

 I've been down this road more times than I care to remember you're staring at a JPA entity class trying to cram a list in there and JPA is just throwing a fit screaming "basic attribute" it's like it thinks you're trying to store a single integer instead of the whole damn collection I've lost count of the hours debugging that exact issue so let's dive in and try to sort this out for you

First off what's happening here is JPA specifically the implementation you're likely using like Hibernate is a bit particular about what it considers a "basic" type in your entities Think of it like this JPA by default knows how to map simple things like ints strings dates directly to database columns it's got built-in converters and all that good stuff A `List` on the other hand is not so simple it's a collection it can have varying numbers of elements and JPA needs to know how to translate that into a relational database world that world of tables columns rows

JPA doesn't handle collections as a direct basic column type out of the box it needs you to define a relationship basically a pointer to other data or other table I mean So when you declare a field like `@Entity class MyEntity { List<String> items; }` it has no idea what to do its default is trying to treat it as a single simple value that's what basic attribute means.

Now lets get to the real fix There are a few ways to handle this Each of them has pros and cons Let's start with the most common one a one-to-many relationship lets imagine this is your case you have an Entity like `Order` and you need a list of Order Items each item represents some specific item in your order.

You can create a new entity for order items then declare a one-to-many relationship between the Order and OrderItem and in this case JPA has a very good idea how to map this. Lets see a code snippet

```java
@Entity
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    //other order properties here

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderItem> items;
    //Getters and setters
}
```
```java
@Entity
public class OrderItem {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "order_id")
    private Order order;

    //other properties here
    //Getters and setters
}
```
What is happening here
* We are defining an `Order` entity that has a list of `OrderItem` entities using the `@OneToMany` annotation the `mappedBy` attribute specifies the field name that owns the relationship in the other entity side
* We are defining `OrderItem` entity that is mapped to an `Order` entity using the `@ManyToOne` annotation and `@JoinColumn` defines the foreign key column

**Pros**
* It's the most relational design and is often the ideal way to represent a list of related entities in a database with all its power of querying joining etc.
*  It's very clear and expressive the data model is directly reflected in the entities

**Cons**
* Can result in more database tables and potentially more complex queries.
*  You may need to be very careful with cascade types if you are not aware of how they work they can lead to unexpected data modifications.

Let's say you just need to save the list of basic type String or Integers in database table for this you have multiple options one is a `ElementCollection`.

This can be useful when you don't have a full entity for each item in the list but just simple values. We are going to use the `Order` example again.

```java
@Entity
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ElementCollection
    @CollectionTable(name = "order_item_names", joinColumns = @JoinColumn(name = "order_id"))
    @Column(name = "item_name")
    private List<String> itemNames;

    //Getters and setters
}
```
* The `@ElementCollection` annotation tells JPA that we are mapping a collection of basic types
* The `@CollectionTable` annotation specifies the name of the table that stores the collection. The `joinColumns` defines how to get back to the `Order` table
* The `@Column` annotation specifies the name of the column in the collection table that contains the values.

**Pros**
* Simpler to model and may not require a separate entity.
* Can be more efficient for simple collections if they are not frequently accessed

**Cons**
* The collection is not a separate entity you cannot query it directly using JPA criteria.
* If the collection needs more properties we cannot add it directly here.
* It can be a bit harder to track changes in a detached entity and requires more data loading and persistence.

There is one more way to do it and it is by using a JSON type but only if your database supports it and JPA provider allows it for example Hibernate does support this if you are using PostgreSQL or MySQL or other databases that support a JSON field

```java
@Entity
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Type(JsonType.class)
    @Column(columnDefinition = "jsonb")
    private List<String> itemNames;

    //Getters and setters
}
```
*  We are using hibernate type to map a `List` to a jsonb database type.
* The `@Column` annotation specify the database column type and it is jsonb.
* If you do not specify the database type it will save as a text.

**Pros**
* Simplest option when there is no other choice.
* Very flexible and you can store more complex json objects

**Cons**
* It is not relational in the database and you cannot query the json content directly in the database unless you use jsonb operators.
* Data might not be as normalized as with using an entity.

I remember once I spent a whole afternoon trying to figure this one out It turns out I was trying to use a list of custom objects without defining any relation or embeddable type and JPA was just giving a very vague error message it was not obvious at all and I was basically copy pasting trying every annotation I found until the error disappeared. I felt like a monkey typing on a keyboard and accidentally wrote a poem (a horrible poem).

So the key takeaway is JPA wants to know how your collections map to the underlying database be it another table an element collection or even a json column

For further study I suggest the following resources
1.  **Java Persistence with Hibernate** by Christian Bauer and Gavin King This book will explain everything in great detail not just the collections but JPA as a whole I often go back to this book to be honest.
2.  **Pro JPA 2 Mastering the Java™ Persistence API** by Mike Keith and Merrick Schincariol Another excellent book about JPA this is the one I used a lot back in the day to understand the fundamentals.
3.  **Official JPA documentation**. This is like the bible to JPA if you have doubts always refer to the specification.

Hopefully this helps and prevents you from losing hours as I have on this very topic Good luck and happy coding.
