---
title: "Which DDD pattern, shared kernel or published language, is more suitable for strategic design?"
date: "2025-01-30"
id: "which-ddd-pattern-shared-kernel-or-published-language"
---
The fundamental distinction between the Shared Kernel and Published Language patterns in Domain-Driven Design (DDD) lies in their approach to managing consistency and autonomy between bounded contexts.  My experience working on large-scale financial transaction processing systems has repeatedly highlighted the crucial trade-off between shared understanding and independent evolution, a trade-off directly impacting the strategic design choices between these two patterns.  While both address inter-context communication, the Published Language offers superior scalability and maintainability for complex systems, particularly when considering long-term strategic considerations.  This is primarily because it prioritizes explicit communication over implicit shared understanding.

**1. Clear Explanation:**

A Shared Kernel involves a physically shared codebase representing the core domain logic and models across multiple bounded contexts.  This approach fosters a strong degree of consistency. However, its inherent tight coupling presents significant challenges as the contexts evolve.  Any change to the shared kernel requires coordination across all consuming contexts, creating a bottleneck and increasing the risk of unintended side effects. This becomes increasingly problematic as the number of bounded contexts and teams grows.  Deployment becomes more intricate, requiring careful orchestration to ensure compatibility across all contexts.

In contrast, a Published Language employs a clearly defined, explicit language (typically a formal specification like an API or a messaging protocol) for communication between bounded contexts.  Each context maintains its own independent model and codebase, interacting solely through this defined interface.  This architectural approach significantly reduces coupling, enabling independent development, deployment, and evolution.  While achieving complete consistency across contexts is more challenging with a Published Language, the gain in autonomy and reduced risk of disruptive changes often outweighs this concern, particularly in large, complex systems with diverse teams.

The choice between these patterns is not always straightforward.  A Shared Kernel might be appropriate for tightly integrated systems with a small number of closely aligned contexts, where maintaining absolute consistency is paramount and the evolution is anticipated to be slow and coordinated.  However, for large-scale, complex, evolving systems, a Published Language represents a more robust and sustainable strategic design choice.  The decoupling enabled by the Published Language promotes agility, resilience to change, and ultimately lowers the total cost of ownership.

**2. Code Examples:**

**Example 1: Shared Kernel (Illustrative – impractical for realistic scenarios)**

This simplified example depicts a shared `Customer` entity used by both an `Order` and `Account` context.  The tight coupling is evident.

```java
// Shared Kernel
public class Customer {
    private String id;
    private String name;
    // ... other attributes ...

    // Getters and setters
}

// Order Context
public class Order {
    private Customer customer;
    // ...
}

// Account Context
public class Account {
    private Customer customer;
    // ...
}
```

**Commentary:** Any change to the `Customer` class directly impacts both `Order` and `Account` contexts, demanding coordinated updates and thorough regression testing across all contexts. This is a significant impediment to independent evolution.


**Example 2: Published Language (API-based)**

This example demonstrates a simplified REST API defining a `Customer` resource used for communication between the `Order` and `Account` contexts.  Each context interacts with the API using its own custom model.

```java
// Order Context (Client)
public class OrderService {
    public void createOrder(String customerId, /* ... */) {
        // Make API call to Customer service to retrieve customer details.
        // ...
    }
}

// Account Context (Client)
public class AccountService {
    public void updateAccount(String customerId, /* ... */) {
        // Make API call to Customer service to retrieve customer details.
        // ...
    }

// Customer Context (API Provider)
// ... REST API endpoints for Customer resource ...
```

**Commentary:**  The contexts are loosely coupled.  Changes to the internal representation of `Customer` within the `Customer` context do not directly affect the `Order` and `Account` contexts, as long as the API contract remains unchanged. This approach supports independent evolution and deployment.


**Example 3: Published Language (Event-driven Architecture)**

This showcases an event-driven approach using a message broker. Changes to the customer are communicated asynchronously through events.

```java
// Customer Context (Event Publisher)
public class Customer {
    public void updateName(String newName) {
        // ... update internal state ...
        eventBus.publish(new CustomerNameUpdatedEvent(this.id, newName));
    }
}

// Order Context (Event Subscriber)
public class OrderEventHandler {
    @EventHandler
    public void handleCustomerNameUpdated(CustomerNameUpdatedEvent event) {
        // Update customer name in order if necessary, based on event data.
    }
}

// Account Context (Event Subscriber)
// Similar EventHandler for Account context
```

**Commentary:** This illustrates asynchronous communication, enhancing decoupling further.  Changes in one context are communicated to others without direct coupling, improving robustness and scalability.  The message broker acts as a central point of communication, abstracting the communication details from individual contexts.


**3. Resource Recommendations:**

*   Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software"
*   Vaughn Vernon's "Implementing Domain-Driven Design"
*   Greg Young's various publications and presentations on DDD
*   Alberto Brandolini's work on Event Storming
*   DDD community resources (e.g., conferences, meetups, online forums).  Focusing on resources specifically addressing strategic design will be particularly valuable.


In conclusion, based on extensive experience, I strongly advocate for the Published Language pattern as a more robust and scalable solution for strategic DDD design in complex, evolving systems. The reduced coupling, enhanced autonomy, and increased agility significantly outweigh the slight overhead in achieving consistency.  The Shared Kernel pattern, while applicable in limited scenarios, lacks the long-term scalability and maintainability necessary for systems of significant scope and complexity.
