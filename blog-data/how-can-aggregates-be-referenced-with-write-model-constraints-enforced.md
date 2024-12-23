---
title: "How can aggregates be referenced with write model constraints enforced?"
date: "2024-12-23"
id: "how-can-aggregates-be-referenced-with-write-model-constraints-enforced"
---

Okay, let's tackle this. Referencing aggregates while ensuring write model constraints are strictly enforced is a challenge I've bumped into quite a few times, especially back in my days working on that sprawling e-commerce platform. It’s a fundamental aspect of domain-driven design (ddd) and maintaining data integrity. It's less about finding a silver bullet and more about choosing the right tool for the job within the context of your specific application. There's no single, universally applicable answer, but rather a set of patterns and techniques that, when combined thoughtfully, can yield a robust solution.

My experience has shown that the core problem usually boils down to maintaining consistency when modifying related aggregates. In a purely relational database world, you might be inclined to reach for foreign keys, cascades, and maybe some clever triggers. But when working with ddd, especially in a microservices environment where aggregates are often self-contained transactional units, these tools can become more of a liability than an asset. They can lead to unintended coupling, making changes across different areas of the application significantly harder.

So, how do we solve this? We need to think about references differently. Instead of relying on direct database-level relationships, we're looking at domain-level references, typically using the aggregate’s id. When an aggregate needs to reference another aggregate, it shouldn't directly hold a complete instance of that aggregate. Instead, it should hold a unique identifier (usually the primary key) and rely on a repository to load the referenced aggregate when needed, following read models for queries. But importantly, for write operations, we do not fetch the entire aggregate for validation if possible, opting for lighter-weight checks instead, and always respecting the consistency boundaries of each aggregate, validating them within the aggregate's scope.

Let’s explore some approaches with code examples, using a hypothetical `Order` and `Customer` aggregate:

**Example 1: Lightweight Validation and Precondition Checks**

In this example, consider that the `Order` aggregate needs to ensure a customer exists before an order can be placed. Rather than pulling the whole `Customer` aggregate, which could be large, we can rely on an interface that provides us with the needed information that the customer exists, without requiring a full load. We make sure that our repository knows how to check for the existence of a customer with a certain id. This keeps coupling minimal and is generally more efficient.

```python
class Order:
    def __init__(self, order_id, customer_id, items):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.status = "Pending"

    def place_order(self, customer_repository):
        if not customer_repository.exists(self.customer_id):
           raise ValueError("Customer does not exist")
        self.status = "Placed"

class CustomerRepository:
   def __init__(self, storage_layer): # assuming a storage layer object
       self.storage = storage_layer

   def exists(self, customer_id):
       return self.storage.check_customer_exists(customer_id)


class Storage:
    def __init__(self, store):
       self.store = store
    
    def check_customer_exists(self, customer_id):
       return customer_id in self.store # assumes self.store is a dictionary or equivalent for simplicity

# Example Usage:
store = {"customer1": {"name": "john doe"}}
storage = Storage(store)
customer_repo = CustomerRepository(storage)

order = Order("order1", "customer1", ["item1", "item2"])
order.place_order(customer_repo) # this should work

invalid_order = Order("order2", "customer2", ["item1"])
try:
    invalid_order.place_order(customer_repo) # should raise ValueError
except ValueError as e:
    print(f"error: {e}")
```

Here, the `Order` aggregate doesn't need to know the `Customer`'s details to enforce its constraint; it just needs a confirmation on the existence. This avoids loading the customer object, which can become a performance bottleneck if customer data is large or needs to be retrieved from a remote database. This example showcases how we can validate preconditions before mutations, while respecting aggregate boundaries. The repository abstraction allows us to use different storage implementation without impacting our domain logic.

**Example 2: Using Domain Services for Complex Validation**

Sometimes, validation is not as simple as a quick existence check, for example, you might want to check that a customer has enough credit to make a purchase. In these cases, it is often better to delegate to a domain service, keeping your aggregate cleaner. Domain services usually sit between the application and domain layers and are where you can implement business logic that involves multiple aggregates. They will coordinate data consistency across the aggregates without direct coupling.

```python
class Order:
    def __init__(self, order_id, customer_id, items, total):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.total = total
        self.status = "Pending"

    def place_order(self, order_service):
       if not order_service.can_place_order(self.customer_id, self.total):
           raise ValueError("Cannot place order due to customer credit limit.")
       self.status = "Placed"


class CustomerRepository: # same as in previous example
   def __init__(self, storage_layer):
       self.storage = storage_layer

   def exists(self, customer_id):
       return self.storage.check_customer_exists(customer_id)

   def get_credit(self, customer_id):
       return self.storage.get_customer_credit(customer_id)


class Storage: # same as in previous example
    def __init__(self, store):
       self.store = store
    
    def check_customer_exists(self, customer_id):
       return customer_id in self.store
    
    def get_customer_credit(self, customer_id):
        if customer_id in self.store:
           return self.store[customer_id]['credit']
        return 0


class OrderService:
    def __init__(self, customer_repository):
        self.customer_repository = customer_repository

    def can_place_order(self, customer_id, total):
       if not self.customer_repository.exists(customer_id):
            return False
       customer_credit = self.customer_repository.get_credit(customer_id)
       return customer_credit >= total

#Example Usage:
store = {"customer1": {"name": "john doe", "credit": 100}}
storage = Storage(store)
customer_repo = CustomerRepository(storage)
order_service = OrderService(customer_repo)

order1 = Order("order1", "customer1", ["item1"], 50)
order1.place_order(order_service)  # Should succeed

order2 = Order("order2", "customer1", ["item2"], 150)
try:
    order2.place_order(order_service) #Should fail
except ValueError as e:
   print(f"error: {e}")

```

Here the `Order` aggregate is only responsible for itself; the more complex logic related to the customer’s credit is delegated to the `OrderService`. The `OrderService` orchestrates a conversation with other aggregates via their repositories, but importantly, the `Order` aggregate itself remains focused on what's specific to its own internal consistency, which is the key point.

**Example 3: Eventual Consistency and Background Processes**

For scenarios where immediate consistency is not essential and high availability is a priority, eventual consistency patterns can be useful. You can place the order and then trigger an asynchronous event that can perform additional validations. For instance, after the order is placed, an event might trigger a process to check the customer’s credit, and if there is a problem, the order can be marked as cancelled after the fact.

```python
class Order:
    def __init__(self, order_id, customer_id, items, total):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.total = total
        self.status = "Pending"

    def place_order(self, event_publisher):
        self.status = "Placed"
        event_publisher.publish_event("order_placed", {"order_id": self.order_id, "customer_id": self.customer_id, "total": self.total})

    def mark_cancelled(self):
        self.status = "Cancelled"


class EventPublisher:
   def __init__(self, event_queue):
       self.event_queue = event_queue

   def publish_event(self, event_name, payload):
        self.event_queue.append({"name": event_name, "payload": payload})

class OrderService:
   def __init__(self, customer_repository, event_publisher, order_repository):
      self.customer_repository = customer_repository
      self.event_publisher = event_publisher
      self.order_repository = order_repository

   def handle_order_placed(self, event):
       customer_id = event['customer_id']
       total = event['total']
       order_id = event['order_id']
       if not self.customer_repository.exists(customer_id):
           order = self.order_repository.get_order(order_id)
           if order:
             order.mark_cancelled()
             print(f"order {order_id} was cancelled due to invalid customer")
             return

       customer_credit = self.customer_repository.get_credit(customer_id)
       if customer_credit < total:
           order = self.order_repository.get_order(order_id)
           if order:
             order.mark_cancelled()
             print(f"order {order_id} was cancelled due to insufficient credit")

class OrderRepository:
    def __init__(self, storage_layer):
      self.storage = storage_layer

    def get_order(self, order_id):
        if order_id in self.storage.store:
           return self.storage.store[order_id]
        return None

#Example usage
store = {"customer1": {"name": "john doe", "credit": 100}}
storage = Storage(store)

order_repo = OrderRepository(storage)
event_queue = []
event_publisher = EventPublisher(event_queue)
customer_repo = CustomerRepository(storage)
order_service = OrderService(customer_repo, event_publisher, order_repo)

order1 = Order("order1", "customer1", ["item1"], 50)
order1.place_order(event_publisher) # order will be placed
storage.store["order1"] = order1
order_service.handle_order_placed(event_queue.pop())

order2 = Order("order2", "customer1", ["item2"], 150)
order2.place_order(event_publisher) # order will be placed
storage.store["order2"] = order2
order_service.handle_order_placed(event_queue.pop()) # but then cancelled

order3 = Order("order3", "customer2", ["item2"], 10)
order3.place_order(event_publisher)
storage.store["order3"] = order3
order_service.handle_order_placed(event_queue.pop()) # will be cancelled due to invalid customer

```
This example showcases asynchronous validation. The order is first placed. It then publishes an event, which the order service listens to and performs credit checks and cancels order if needed. This pattern can provide the flexibility you need for different consistency needs and is ideal in systems where microservices do not have a central datastore.

**Recommended Resources**

To delve deeper into these concepts, I highly recommend exploring Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software," which is essentially the bible for DDD. For a more practical perspective, "Implementing Domain-Driven Design" by Vaughn Vernon offers a solid approach. Also, Martin Fowler’s work, particularly his articles on patterns such as aggregates, eventual consistency, and domain services on his website, will be invaluable. Finally, the online resources from Microsoft on Azure architecture, specifically regarding microservices and data patterns are good resources that show how these patterns are applied in real-world applications.

In essence, referencing aggregates with write model constraints isn't about enforcing database-level relationships. It's about making informed decisions based on the immediate consistency requirements, aggregate boundaries, and the complexity of business logic, and relying on domain services and event-driven architecture where necessary. I’ve found these approaches to provide the best balance between consistency and flexibility in most real-world scenarios.
