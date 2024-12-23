---
title: "Should I use multiple databases or a single source of truth in a Domain-Driven Design?"
date: "2024-12-23"
id: "should-i-use-multiple-databases-or-a-single-source-of-truth-in-a-domain-driven-design"
---

, let's talk databases in a ddd context, a topic I’ve definitely spent my fair share of nights pondering over. You’re asking a question that cuts to the core of system architecture and, honestly, there isn't a single, definitive answer applicable to all scenarios. The short response is: it depends on your domain, its complexity, and your architectural priorities. But let’s unpack that a bit.

Early in my career, I was involved in developing a large-scale e-commerce platform. We initially opted for a monolithic approach with a single, massive relational database serving all domain contexts – customer management, inventory, orders, payments, the whole nine yards. While seemingly simple at first, this quickly became a maintenance and performance bottleneck as the system grew. The schema was a tangled web, deployments became risky and slow, and performance suffered. Changes in one area often had unintended consequences elsewhere. This experience, among others, made me acutely aware of the tradeoffs involved in database choices within a domain-driven design (ddd).

The core idea behind ddd is to model software around the business domain, and within a ddd context, a ‘single source of truth’ (ssot) often refers to a conceptual, domain-specific model, rather than a physical database instance. The domain model should be our primary source of truth and the database implementations, should be a persistence mechanism. In my experience, trying to make a single database the *only* source of truth for the entire application domain often leads to tightly coupled systems and data models that don't reflect business needs accurately. It quickly introduces issues around performance, data integrity, and scalability.

When you approach the question with multiple databases versus a single one, you are essentially confronting the dilemma of ‘one big database’ against ‘polyglot persistence’, a concept popularized by Martin Fowler, where different data storage technologies are used based on the specific needs of the domain. In ddd terminology, we generally align these choices around *bounded contexts*. If you can identify distinct bounded contexts within your domain that have limited interaction and data dependencies, having separate databases (or even data stores – could be a document store, graph db, etc.) can be a highly advantageous strategy.

The benefit lies in having each database optimized for the specific access patterns and requirements within its bounded context. For example, an inventory context might be well-suited to a relational database with strong consistency, while a user profile context might favor a document database due to flexible schema and eventual consistency. You isolate the data and operations within a context, preventing other contexts from inadvertently influencing or corrupting that domain's data. This isolation promotes autonomy between teams as well, as each team can manage the data model and technology stack that fits their bounded context best.

However, this approach isn’t without its challenges. Data synchronization between contexts becomes necessary and that requires careful thinking, as you no longer have the implicit joins of a single database. We often employ strategies such as event-driven architectures where changes in one context publish events that other contexts can subscribe to, allowing for data to be synchronized asynchronously. This helps avoid tight coupling at the application level.

Let's look at three conceptual examples, with some simplified code to underscore the concepts:

**Example 1: Order Management (Single Database)**

```python
# A highly simplified model, showing relationships
class Customer:
    def __init__(self, customer_id, name, email):
        self.customer_id = customer_id
        self.name = name
        self.email = email

class Product:
    def __init__(self, product_id, name, price, stock_level):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock_level = stock_level

class Order:
    def __init__(self, order_id, customer, order_items, order_date):
        self.order_id = order_id
        self.customer = customer
        self.order_items = order_items
        self.order_date = order_date

class OrderItem:
    def __init__(self, product, quantity):
        self.product = product
        self.quantity = quantity

# Hypothetical data access code:
class OrderRepository:
    def save_order(self, order):
        # Persist the order, its customer, and its items
        pass # would use a db connection

    def get_order(self, order_id):
      # retrieve and compose an order with all its dependencies
      pass # would query from db

# In a monolithic architecture, it would all be persisted within a single database.

```

Here, we can imagine all these entities and their relationships (customers, products, order items, and orders) existing within a single database. Simple, in that you can do relational joins. But, imagine this scaled to enterprise levels with tens of tables, and the database grows to gargantuan size. Performance starts to degrade, and you have teams from different domains having to change the same tables.

**Example 2: Order Management (Multiple Databases)**

```python
# Context specific models

# OrderContext Models
class Order:
    def __init__(self, order_id, customer_id, order_items, order_date):
      self.order_id = order_id
      self.customer_id = customer_id # just a foreign key
      self.order_items = order_items
      self.order_date = order_date


class OrderItem:
    def __init__(self, product_id, quantity):
      self.product_id = product_id
      self.quantity = quantity


# CustomerContext Models
class Customer:
    def __init__(self, customer_id, name, email):
        self.customer_id = customer_id
        self.name = name
        self.email = email

# ProductCatalog Model
class Product:
   def __init__(self, product_id, name, price):
     self.product_id = product_id
     self.name = name
     self.price = price

# Hypothetical data access code:
class OrderRepository:
    def save_order(self, order):
      # persist data to order database
      pass

    def get_order(self, order_id):
       # retrieve from order database
       pass

class CustomerRepository:
     def get_customer(self, customer_id):
        # retrieve customer info from the customer database
        pass

class ProductRepository:
    def get_product(self, product_id):
        # retrieve product info from product catalog database
        pass

# Orders reside in their own db, along with items.
# Customer info lives in its own database
# Product info lives in its product catalog database

```

In this second scenario, the customer, order, and product contexts are persisted in their respective databases (or even different data store technologies entirely). Notice how the `Order` object now uses `customer_id` instead of a full customer object, and only a `product_id` instead of the product object. This means when displaying an order you would have to make calls to the customer context to pull back the full customer info. This introduces some complexity in how data is retrieved, but overall it gives better modularity and scalability. We no longer have the same database contention or performance issues. We've made the system more decoupled and each database can be scaled, optimized, and maintained independently.

**Example 3: Event-Driven Synchronization**

```python
# Assume the repositories and models are similar to example 2.

# Hypothetical Event Publisher
class EventPublisher:
  def publish(self, event_name, payload):
        # would send the event over messaging queue
        print(f"publishing event {event_name} with payload: {payload}")
        pass


# Hypothetical Event Listener
class CustomerChangeListener:
    def handle_customer_changed_event(self, customer_id, new_email):
       # Update cached customer info in the orders context
       # Or trigger a full data sync if needed
        print(f"Handling customer update for {customer_id}, new email: {new_email}")
        pass

# Example: When a customer is updated, publish an event
customer = Customer(customer_id="123", name="John Doe", email="john.doe@old.com")
customer.email = "john.doe@new.com"

event_publisher = EventPublisher()
event_publisher.publish("customer.changed", {"customer_id": customer.customer_id, "new_email": customer.email})


customer_change_listener = CustomerChangeListener()
customer_change_listener.handle_customer_changed_event(customer.customer_id, customer.email)
```

Here, we demonstrate how data can be synchronized between bounded contexts using an event-driven approach. When a customer updates their email, an event is published. The Order context (or others) can subscribe to these events and perform necessary data updates, perhaps in a cached store or by triggering full data resynchronization, as it is appropriate to its need. This approach allows for asynchronous communication between contexts and further reduces coupling.

In conclusion, my experience leans towards using multiple databases aligned with your bounded contexts, each optimized for the specific domain needs. While it introduces synchronization challenges, these can be managed with appropriate techniques such as event-driven architectures. It promotes more maintainable, scalable, and adaptable systems. You should think of the 'single source of truth' as the domain model, not necessarily a single database.

For further research, I’d recommend looking into Martin Fowler's “Patterns of Enterprise Application Architecture,” Eric Evans' "Domain-Driven Design," and Vaughn Vernon's “Implementing Domain-Driven Design.” These texts offer in-depth insights into the various patterns and strategies for designing large-scale systems. Additionally, studying papers on eventual consistency and distributed transactions would be invaluable when dealing with multiple data sources. Remember, the ‘right’ solution is situational, but understanding the tradeoffs is paramount.
