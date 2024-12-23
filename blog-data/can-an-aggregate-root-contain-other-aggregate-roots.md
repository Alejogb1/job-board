---
title: "Can an aggregate root contain other aggregate roots?"
date: "2024-12-23"
id: "can-an-aggregate-root-contain-other-aggregate-roots"
---

Okay, let’s dive into the somewhat thorny question of whether an aggregate root can contain other aggregate roots. From my experience, this is a point where a lot of domain-driven design (ddd) implementations can go off the rails, and it's crucial to get this concept solid.

Let’s start with a crucial clarification: the very purpose of an aggregate root is to define a transactional consistency boundary. It's the entry point for accessing and modifying entities within that boundary, enforcing invariants and ensuring data integrity within its scope. When you hear “aggregate root,” think of it as the *single* entity in a given aggregate that can be directly referenced from *outside* that aggregate. It’s the guardian, the gatekeeper, if you will.

Now, the idea that one aggregate root might contain another – directly, that is – starts to break down the very purpose of the pattern. The problem stems from creating a tight coupling and potentially cascading updates that can violate the integrity of the contained aggregate. If root *a* contains root *b*, and you want to modify something in *b*, you need to go through *a*. This creates a dependency chain that makes reasoning about data changes and transactional consistency much more difficult. It’s a recipe for accidental data corruption.

I remember a project a few years back, working on an e-commerce platform. We initially modeled our *order* aggregate root to *contain* a *customer* aggregate root. The thinking was, an order directly relates to a customer, so why not embed it? We soon learned that when updating customer details, we had to go through every order referencing that customer. This led to messy update cascades and a significant performance penalty. It became a nightmare to maintain, and worse, the transaction was becoming too large and difficult to control, making rollbacks problematic.

So, the answer is generally a strong *no*. An aggregate root should not directly contain another aggregate root. Instead, it should reference another aggregate root by its identifier. This way, the integrity of each aggregate is preserved, and modifications within one don't unnecessarily ripple into another. The two aggregates still relate to one another, but the relationship is managed through ids, not direct object containment.

Let me illustrate with some code snippets, first, the way that *would* cause problems:

```java
// problematic code: direct containment of aggregate roots
public class Order {
    private OrderId orderId;
    private Customer customer; // direct reference to an aggregate root - bad!
    private List<OrderItem> orderItems;
    // constructor, etc
}

public class Customer {
    private CustomerId customerId;
    private String name;
    private String email;
    // constructor, etc
}

```

The above example has a *Customer* object residing directly within *Order*. If we were to change the customer's name, we would somehow need to find each order associated with the customer, and then update the customer object inside that order - a massive operational headache that makes reasoning about the system incredibly difficult. Transactionally, this is dangerous too.

The correct approach, and the one I favor, involves referencing by id:

```java
// preferred approach: referencing by id
public class Order {
    private OrderId orderId;
    private CustomerId customerId; // referencing by id is good!
    private List<OrderItem> orderItems;
    // constructor, etc
}

public class Customer {
    private CustomerId customerId;
    private String name;
    private String email;
    // constructor, etc
}

```

Here, the *Order* aggregate root references the *Customer* aggregate root using its `customerId`. When we need to access the customer associated with the order, we'd use the `customerId` to fetch the *Customer* from a customer repository. This keeps the aggregates independent and the transaction boundaries clear.

Here’s another example to make this clearer. Think about a blogging system:

```java
// problematic code: direct containment of aggregate roots
public class BlogPost {
    private PostId postId;
    private Author author;  // direct reference to another aggregate root
    private String title;
    private String content;
    // ...
}

public class Author {
  private AuthorId authorId;
  private String authorName;
  // ...
}

```

In this problematic setup, *BlogPost* directly holds an *Author* instance. It means that any operation that potentially modifies the author information would require going through *BlogPost*. If we want to modify author’s name and the author is associated with multiple blog posts, then we would require updating each instance. Which is not ideal.

The preferred and correct implementation is to simply hold author's id:

```java
// preferred approach: referencing by id
public class BlogPost {
    private PostId postId;
    private AuthorId authorId;  // referencing by id
    private String title;
    private String content;
    // ...
}
public class Author {
  private AuthorId authorId;
  private String authorName;
  // ...
}

```

Here, the *BlogPost* aggregate references the *Author* aggregate by its *authorId*. Changes to the author’s name happen within the *Author* aggregate, and the *BlogPost* remains consistent without cascading updates. The *BlogPost* can then use the *authorId* to look up the author from a repository when needed.

This is not to say you can’t have any relationships between aggregate roots. Relationships are fundamental to any complex system. The key is not *how* you have a relationship, but *how you manage it*. You should avoid *object containment* of other aggregate roots and instead favor holding identifiers. You then use those identifiers to retrieve the related aggregates via dedicated repositories.

For further in-depth study of these concepts, I'd strongly recommend reading Eric Evans’ "Domain-Driven Design: Tackling Complexity in the Heart of Software." It's foundational for understanding ddd principles. Additionally, “Implementing Domain-Driven Design” by Vaughn Vernon provides more practical examples and implementation details. A good paper to read is "Aggregate Patterns" by Martin Fowler, which dives deep into the reasoning behind aggregate roots. These resources will significantly clarify these, and many other, complex concepts related to DDD.
