---
title: "How can distributed systems be modeled using domain-driven design?"
date: "2024-12-23"
id: "how-can-distributed-systems-be-modeled-using-domain-driven-design"
---

Okay, let’s tackle this. I remember a particularly challenging project back in '18, integrating three previously siloed legacy systems for a financial institution – a true distributed monolith, if you will. We spent weeks untangling spaghetti code and trying to reconcile inconsistent data, until we finally leaned into domain-driven design (ddd). It dramatically changed our approach, providing not only a structure for the software but also a clear model of the business problem we were addressing.

So, how do we use ddd to model distributed systems? The core principle, as always with ddd, is to start with the domain. Forget the technology for now; think about the actual business. A distributed system, inherently, deals with multiple interconnected services, each often responsible for a specific part of that business. This is where the concept of ‘bounded contexts’ becomes critical. These are not simply microservices, although they often map closely. Instead, they represent a semantic boundary within the business domain where a particular model is consistent and meaningful.

Let’s think of an e-commerce platform. Instead of one gigantic 'ecommerce' monolith, we might identify several bounded contexts: 'catalog', 'ordering', 'payments', and 'shipping'. Within 'catalog', for instance, our model will be about products, categories, and inventory. In 'ordering,' we are concerned with orders, customers, and shopping carts. Each context has its own model, language (ubiquitous language), and rules, and is a domain onto itself.

The next critical piece is how these contexts communicate. In a monolithic application, objects interact directly in memory, often through well-defined interfaces. In a distributed environment, however, we're talking about inter-process communication across a network, leading to inherent latency and potential failures. DDD acknowledges this and provides patterns for it:

1.  **Context mapping:** This is all about defining relationships between contexts. Do they share data directly? Is one context a consumer of information provided by another? Are there upstream and downstream relationships? Is the relationship between the contexts collaborative or customer-supplier? For example, the 'ordering' context might *depend on* the 'catalog' context for product information, but the relationship may also be customer-supplier, where the 'catalog' doesn't necessarily need to know about orders, but 'ordering' heavily relies on its data.
2.  **Anti-corruption layers (acls):** When two contexts have different models, and you need to translate between them, you employ an acl. Think of it as a translator between two different languages. The acl sits between the two bounded contexts, translating from the upstream context's model into a model understandable by the downstream context. This protects the downstream context from changes in the upstream context and reduces coupling, making each context easier to maintain independently.
3.  **Event-driven architectures:** Instead of tightly coupled synchronous interactions, we can employ asynchronous messaging via events. When something happens in one context, it publishes an event that other interested contexts can subscribe to. This pattern enables loose coupling and greater resilience. For instance, when an order is placed in the 'ordering' context, it publishes an event ('order-placed') which the 'shipping' context subscribes to in order to schedule the shipment.

The important thing is that ddd guides us to focus on modeling the *business* first, and then implement that model through our distributed architecture. It’s not about retrofitting ddd onto an existing architecture; it’s about using the business domain to *drive* how the system is structured.

Now, let's look at some practical examples. I’ll show how these concepts can translate into code snippets using a simplified representation in python, focusing on high level representation, omitting error handling or specific tech stacks for brevity.

**Example 1: Bounded Contexts and Anti-Corruption Layer**

Imagine the 'catalog' context has a `product` class that stores information including `sku` (stock keeping unit), `name`, `price`, and `available_quantity`. In contrast, the ‘ordering’ context, might model a `line_item` class which includes `item_id`, `name`, `unit_price` and `quantity`. The model itself has a different semantic value within the context. Instead of directly accessing the `product` object and being coupled to it, we introduce an anti-corruption layer (acl):

```python
# catalog context representation
class Product:
    def __init__(self, sku, name, price, available_quantity):
        self.sku = sku
        self.name = name
        self.price = price
        self.available_quantity = available_quantity

# ordering context representation
class LineItem:
    def __init__(self, item_id, name, unit_price, quantity):
        self.item_id = item_id
        self.name = name
        self.unit_price = unit_price
        self.quantity = quantity

#acl implementation
class CatalogToOrderingAcl:
    def to_line_item(self, product, quantity):
        return LineItem(
            item_id=product.sku,
            name=product.name,
            unit_price=product.price,
            quantity=quantity,
        )

#usage
catalog_product = Product("abc-123", "Laptop", 1200.00, 50)
acl = CatalogToOrderingAcl()
line_item = acl.to_line_item(catalog_product, 2)
print(f"Line Item: Item ID: {line_item.item_id}, Name: {line_item.name}, Price: {line_item.unit_price}, Quantity: {line_item.quantity}")
```

Here, the `CatalogToOrderingAcl` translates from a `product` in the 'catalog' context to a `line_item` in the 'ordering' context. If the `product` model changes (e.g., if a product description is added), the 'ordering' context doesn't need to be modified immediately, it depends on the acl, shielding it from the catalog's model.

**Example 2: Event-Driven Architecture**

Let’s illustrate how events work with a simplified publisher-subscriber pattern:

```python
# event mechanism (simplified)
class Event:
    def __init__(self, type, data):
        self.type = type
        self.data = data

class EventBus:
    def __init__(self):
        self.subscribers = {}

    def publish(self, event):
       if event.type in self.subscribers:
          for subscriber in self.subscribers[event.type]:
              subscriber(event)

    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

# ordering context publisher
def publish_order_placed(bus, order_id, customer_id):
  event= Event("order-placed", {"order_id": order_id, "customer_id": customer_id})
  bus.publish(event)

# shipping context subscriber
def handle_order_placed(event):
    print(f"Shipping context received: order-placed event: {event.data}")

#usage
bus = EventBus()
bus.subscribe("order-placed", handle_order_placed)
publish_order_placed(bus, "order-123", "customer-456")
```

In this example, the 'ordering' context publishes an 'order-placed' event, and the 'shipping' context is subscribed to listen for this event to trigger the shipping process. This is a basic example, but it demonstrates how we decouple these domains through event-based interactions.

**Example 3: Context Mapping**

This example shows the distinction between collaborative and customer-supplier relationships, further contextualizing bounded context relationships. Suppose we have a "user management" context which needs to update "user profile" information.

```python
#user management context

class User:
   def __init__(self, id, name):
       self.id = id
       self.name = name

   def update_name(self, name):
       self.name = name

#user profile context representation
class UserProfile:
   def __init__(self, user_id, profile_data):
        self.user_id=user_id
        self.profile_data = profile_data

#collaborative relationship

class UserProfileCollaborator:
     def update_profile(self, user: User, profile: UserProfile):
       profile.profile_data["name"] = user.name
       print(f"Collaborator: User profile {profile.user_id} updated. {profile.profile_data}")

#customer-supplier relationship

class UserProfileSupplier:
    def update_profile_request(self, user_id, profile_data_update):
      print(f"Supplier: Received request to update profile of user {user_id} with {profile_data_update}")


#usage
user = User("user-1", "John Doe")
profile = UserProfile("user-1", {"name": "Old Name"})
collaborator=UserProfileCollaborator()
collaborator.update_profile(user, profile)

supplier = UserProfileSupplier()
supplier.update_profile_request(user.id, {"new_name": "New Name"})
```
Here, the `UserProfileCollaborator` is actively involved in updating the profile in a collaborative relationship. It accesses `UserProfile` and mutates its state directly based on the `User` entity. On the other hand, the `UserProfileSupplier` represents a customer-supplier relationship where user management requests updates, but the profile context maintains control and simply receives requests for changes. The supplier context doesn't know about the User entity. The 'user management' context depends on profile, but profile doesn’t need to know about user, therefore maintaining a loosely coupled relationship.

To delve deeper, I recommend focusing on the following resources: "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans (the classic work), “Implementing Domain-Driven Design” by Vaughn Vernon (more practical implementation guidance), and various articles on context mapping patterns, which will provide more details on relationship between contexts. Furthermore, consider studying resources that explore event-driven architectures, such as papers on message queues and publish/subscribe systems, and how they can be practically applied in combination with ddd. These resources provide the foundations for understanding ddd and its application in distributed systems.

In my experience, applying ddd to distributed systems is not a quick fix; it’s a journey of constant refinement and iteration. It’s about understanding the business, crafting precise domain models, and structuring your system to reflect those models, especially considering the impact of distribution. It’s a process that emphasizes communication, collaboration, and ultimately, building software that truly solves business problems in a maintainable and scalable way. It’s not just about the code, but about aligning the technical solution with the business reality.
