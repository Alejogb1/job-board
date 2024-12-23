---
title: "How do layered architecture with dependency inversion differ from hexagonal architecture?"
date: "2024-12-23"
id: "how-do-layered-architecture-with-dependency-inversion-differ-from-hexagonal-architecture"
---

, let's talk about architectural patterns, specifically the nuanced differences between layered architecture with dependency inversion and hexagonal architecture. It's a question I've navigated multiple times in my career, and the distinction, while subtle, often dictates the maintainability and testability of a system. I recall a particularly complex project involving a large e-commerce platform, where the initial design was a rather rigid layered architecture. We hit a wall when the business logic, deeply intertwined with database access, became nearly impossible to evolve without cascading changes. That experience taught me the power of proper decoupling and the value of these patterns.

So, let's break it down.

Layered architecture, at its core, divides an application into distinct layers, each with specific responsibilities. A classic example might be a presentation layer (ui), an application layer (business logic), a domain layer (core business rules), and a data access layer (database interaction). The traditional approach has each layer depending on the layer beneath it. So, the ui depends on the application layer, which depends on the domain layer, and so forth. This creates a waterfall of dependencies.

Dependency inversion, or the 'd' in SOLID principles, flips this conventional relationship on its head. Instead of high-level modules depending on low-level modules, both depend on abstractions. These abstractions are typically interfaces. So, the application layer doesn't directly use the concrete database classes; instead, it relies on an interface specifying the needed data access operations. Concrete implementations then implement these abstractions in lower layers. This reduces coupling, making the system more adaptable to changes. It also facilitates unit testing, since you can now easily mock the database for testing the application logic.

Here’s a simplified code example in Java, showing this:

```java
// Interface defining data access operations
interface UserRepository {
    User getUserById(int id);
}

// Concrete implementation for database access
class PostgresUserRepository implements UserRepository {
    @Override
    public User getUserById(int id) {
        // actual database interaction
        return new User("example");
    }
}

// Application layer using the abstraction
class UserService {
    private UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User getUser(int id) {
       return userRepository.getUserById(id);
    }
}
```
In this example, `UserService` doesn't depend directly on `PostgresUserRepository`, but on the `UserRepository` interface. This inversion of control provides flexibility.

Hexagonal architecture, often referred to as ports and adapters, takes this decoupling concept further and focuses on isolating the application core (the domain layer) from the external world. The domain remains completely independent of any specific technology, such as ui frameworks or data storage. The boundaries between the domain and the external world are defined by "ports," which are interfaces specifying required interactions. Adapters are the actual implementations that bridge between these ports and the external world. It's conceptually similar to dependency inversion, but it elevates the domain as the core, surrounded by the framework and infrastrucutre. Think of it as an onion, with the core at the very centre.

Here's a breakdown:

*   **Core (Domain):** Contains the business logic and is unaware of any external concerns.
*   **Ports:** Interfaces that define how the core interacts with the external world. They specify the operations that the core requires and provides.
*   **Adapters:** Implementations of the ports, bridging between the core and external technologies (e.g., ui, database, message queue). Input adapters transform external requests into something the core understands, and output adapters translate core responses for the outside.

The real differentiator here is that hexagonal architecture is less about layers and more about *isolating the domain*, treating everything else as interchangeable peripheral elements. The goal isn't simply to invert dependencies, but to explicitly define the contract between the core application and its environment. This makes it very easy to switch out entire components (like databases) without needing core logic modifications.

Let’s illustrate that with simplified Java examples:

```java
// Port for user retrieval
interface UserRetrievalPort {
    User retrieveUser(int id);
}

// Core domain logic
class UserManagement {
   private UserRetrievalPort userRetrievalPort;
   public UserManagement(UserRetrievalPort userRetrievalPort){
      this.userRetrievalPort = userRetrievalPort;
   }

   public User getUserById(int id) {
    return this.userRetrievalPort.retrieveUser(id);
   }
}


// Adapter for user retrieval using a database
class DatabaseUserAdapter implements UserRetrievalPort {

  // Data access logic here (e.g., using JPA)
  @Override
  public User retrieveUser(int id) {
    // ... database interaction ...
    return new User("example");
  }
}

// Adapter for user retrieval from a mock
class MockUserAdapter implements UserRetrievalPort{
  @Override
  public User retrieveUser(int id) {
    //mock logic here
    return new User("mocked");
  }
}
```

In this example, `UserManagement` is our core, depending only on the `UserRetrievalPort`. The actual database access is handled by the `DatabaseUserAdapter`, and can easily be swapped for a `MockUserAdapter`, for example, allowing the testing of the business logic in isolation from database concerns.

A critical point is that while layered architecture with dependency inversion *can* be a part of hexagonal architecture (in the infrastructure layer, for example), hexagonal architecture goes a step further by explicitly separating core logic and making the entire system driven by ports and adapters, emphasizing the core domain’s isolation and flexibility.

Let’s also look at a real-world example of this. Imagine a system processing financial transactions. Using a traditional layered approach, if you change the persistence method from SQL to NoSQL, you'd likely need to modify multiple classes in the data access and potentially even the application layer. With hexagonal architecture, you would create a new adapter adhering to the existing persistence port, leaving your core business logic untouched.

Here’s another snippet for a quick comparison illustrating it:

```java
// Another example using a message queue
interface MessagePublisher {
    void publishMessage(String message);
}

class CoreService {
    private MessagePublisher messagePublisher;

    public CoreService(MessagePublisher publisher) {
        this.messagePublisher = publisher;
    }

    public void doSomething(String data) {
        // core logic here
       messagePublisher.publishMessage("data processed:" + data);

    }
}

class RabbitMQAdapter implements MessagePublisher {
    // Rabbitmq connection here
    @Override
     public void publishMessage(String message) {
        //implementation using rabbitMQ
        System.out.println("rabbitmq: " + message);
    }

}

class InMemoryMessageAdapter implements MessagePublisher{
  @Override
  public void publishMessage(String message){
    System.out.println("memory: " + message);
  }
}
```

The `CoreService` interacts with the abstraction, the `MessagePublisher`, not the concrete message queue. The type of message queue is abstracted away, and can easily be changed by injecting another implementation, e.g. `InMemoryMessageAdapter`.

For those interested in delving deeper, I strongly recommend reading "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans for understanding domain-driven design concepts, which are closely related to hexagonal architecture. Also, “Clean Architecture” by Robert C. Martin provides excellent insight into both layered and hexagonal architectural patterns, including practical guidance on dependency inversion. For a more practical guide, I’d suggest “Patterns of Enterprise Application Architecture” by Martin Fowler, which contains a wealth of information on various patterns and their applications. And finally, you can explore “Agile Software Development, Principles, Patterns, and Practices” by Robert C Martin, again. This will clarify all the SOLID principles that influence both of these architecture patterns.

In conclusion, while layered architecture with dependency inversion improves coupling and makes testing easier within traditional architecture, hexagonal architecture is a more explicit approach focusing on completely isolating the domain logic and making the external world a mere plugin. Choosing between them depends on the complexity of the project, but if there's a need for high adaptability and domain isolation, hexagonal architecture would be the preferred path, in my experience. It offers superior testability, maintainability and flexibility compared to a traditional layered architecture.
