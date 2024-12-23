---
title: "How should DDD deal with distributed status across domains?"
date: "2024-12-23"
id: "how-should-ddd-deal-with-distributed-status-across-domains"
---

Okay, let's tackle this. Distributed status management across domains within a domain-driven design (ddd) context is a challenge I've seen trip up many teams, including one I led back in '08 when we were migrating a monolithic billing system to microservices. The crux of the issue, as i see it, lies in how you maintain consistency and accuracy of state information when that state is relevant to multiple bounded contexts, especially when those contexts are physically separated and operating asynchronously. Simply put, shared databases are an anti-pattern, and we need a more elegant solution.

My experience showed me that the fundamental principle here is that each domain should own its data and manage its state independently. When status information needs to be shared, it should happen via well-defined events or queries, rather than direct database access. You're looking at a kind of eventual consistency model here, not a real-time transactional guarantee. This isn't a flaw, but rather a feature of distributed systems.

Let’s look at a practical scenario. Imagine a simplified e-commerce setup with two core domains: "Order" and "Inventory". When an order is placed, the 'Order' domain needs to know if the required inventory is available from the ‘Inventory’ domain. We can't just have the 'Order' domain directly query the 'Inventory' database - that would lead to all sorts of coupling and dependency issues. Instead, we employ an event-driven approach. The 'Order' domain initially records a pending order. Then, it sends out an "OrderPlaced" event. The 'Inventory' domain, subscribed to this event, receives it and performs its logic to check for availability. It does *not* respond directly to the 'Order' domain; instead it raises its own event, for instance "InventoryReserved" or "InventoryUnavailable", which the order context may react to, in turn updating its order status.

Let’s put some code snippets into this. We'll use pseudo-code as the underlying logic is what is important not the syntax. The first snippet is a representation of the Order domain emitting the `OrderPlaced` event.

```pseudocode
// Inside the Order Domain
class OrderService {
    private eventBus: EventBus; // Assume we have a publishing mechanism

    public function placeOrder(order: Order) {
        // other validation code etc..

        order.status = "Pending";
        this.eventBus.publish("OrderPlaced", { orderId: order.id, items: order.items });
        // Here we publish an event that the order is now placed
       // we update this entity to reflect the pending status internally.
    }
}
```

In the above, you can see the first part of the asynchronous interaction happening - a signal is sent by one domain that an action has taken place. The order object internally is updated, and so is its internal state.

Next, let's consider the 'Inventory' domain logic which handles the `OrderPlaced` event.

```pseudocode
// Inside the Inventory Domain
class InventoryService {
    private eventBus: EventBus; // Assume we have an event subscriber

    public function onOrderPlaced(event: OrderPlacedEvent) {
      //we receive the order placed event and inspect our inventory
      const { orderId, items } = event;

      const available = this.checkInventory(items);
      if(available) {
            this.reserveInventory(items);
            this.eventBus.publish("InventoryReserved", { orderId: orderId, items: items });
       }else{
          this.eventBus.publish("InventoryUnavailable", { orderId: orderId, items: items });
      }
    }
  private function checkInventory(items) {
    //business logic to look up stock and determine if the items are available
  }

  private function reserveInventory(items) {
    // business logic to reserve the stock
  }
}

```

This snippet illustrates that the inventory domain listens for the “OrderPlaced” event, reacts to it, carries out it's business logic, and, again, publishes its own event to reflect its internal state.

Finally, let's consider the final handler in the order domain.

```pseudocode
//Inside the Order Domain
class OrderService {
    private eventBus: EventBus; // Assume we have an event subscriber
    public function onInventoryReserved(event: InventoryReservedEvent) {
        // receive an inventory reserved event
        const { orderId, items } = event
        const order = this.loadOrder(orderId);
        order.status = "Inventory Reserved";
        // update the order status to reflect the change from pending, persist order
    }

    public function onInventoryUnavailable(event: InventoryUnavailableEvent) {
        // receive an inventory reserved event
        const { orderId, items } = event
        const order = this.loadOrder(orderId);
        order.status = "Order Rejected - Insufficient Inventory";
        // update the order status to reflect the change from pending, persist order
    }
}
```

This final snippet demonstrates the order domain reacting to the event from the inventory domain, completing the communication loop. This mechanism of publish and subscribe of events allows for eventual consistency within the context of DDD, as opposed to strong transactional consistency that a single database would provide.

What's crucial here is that the 'Order' domain doesn’t care *how* the 'Inventory' domain reserves inventory. It only cares about the outcome, communicated via events. Each domain maintains its own state and reacts to external changes asynchronously.

Now, you might ask, what about edge cases and failures? What if the event is lost? That’s where patterns like sagas come in. A saga is essentially a sequence of local transactions coordinated by an orchestrator (or using a choreography style) ensuring that if one transaction fails, all the previous ones are rolled back, or compensating transactions are run. This is crucial for maintaining data consistency in distributed systems. You wouldn't implement this in every interaction, but when you have a complex series of steps across domains, such as a multi-step order process, it becomes essential. Think of a saga as a workflow management system for a distributed context.

For more information, i’d recommend a couple of key texts that have significantly influenced how I approach this kind of problem: "Patterns of Enterprise Application Architecture" by Martin Fowler, which, while not focused purely on DDD, lays a crucial foundation in understanding distributed application architectures. Additionally, "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans is the seminal work on DDD. While not specifically addressing distributed scenarios, it offers the mindset for how to encapsulate your domain and manage state within bounded contexts. Furthermore, look into the concept of “eventual consistency” which will lead you down the path of research into the patterns necessary to support a distributed system. A good text on this is “Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems” by Martin Kleppmann, which explores the architectural patterns behind modern data-centric applications and will help you grasp the reasoning behind these approaches. Also, any resources on the saga pattern will help considerably in dealing with complex workflows that require transactions across multiple domains.

In summary, managing distributed state in DDD is about embracing eventual consistency, relying on events for communication, respecting domain boundaries, and, when necessary, employing sagas for complex workflows. It's not about finding a way to force a single transactional model on distributed components, but rather about understanding the inherent nature of asynchronous systems and building robust solutions around it. It’s a subtle shift in mindset, but once you grasp the principles, it transforms how you design microservices, making them far more resilient and maintainable in the long run. My experience has taught me that focusing on loose coupling and clear communication through events is absolutely key to success in this space.
