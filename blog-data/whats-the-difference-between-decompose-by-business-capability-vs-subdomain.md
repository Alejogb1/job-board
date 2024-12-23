---
title: "What's the difference between decompose by business capability vs. subdomain?"
date: "2024-12-16"
id: "whats-the-difference-between-decompose-by-business-capability-vs-subdomain"
---

, let's unpack this. It's a distinction that, while sometimes seemingly subtle, can have profound implications for how you architect and manage complex software systems. I've seen projects get derailed by conflating these two concepts, and conversely, others thrive by carefully delineating their boundaries. My perspective here comes from years of wrestling—*ahem*, carefully managing—large-scale systems where the organization’s structure and software architecture had to work in tandem.

The core difference hinges on perspective. **Decomposition by business capability** is about modeling your system around what your *business* does. Think about it: what are the fundamental activities, the core competencies, that enable your organization to deliver value? These are your capabilities. Examples might include "order management," "customer relationship management," "inventory tracking," or "payment processing." Each capability encapsulates a set of related business functions and processes. These are typically higher-level views that might span across various teams or even multiple departments within a business.

In contrast, **decomposition by subdomain** is about looking *inward* at the technical details. It focuses on identifying specific problem spaces within your broader domain. These subdomains often stem from a more technical perspective. For instance, within the business capability of “order management,” you might have subdomains like “payment gateway integration,” “shipping calculation,” or “inventory management.” Each of these tackles a specific, cohesive area of functionality, often associated with particular datasets, user interfaces, or algorithms. They represent the granular bits of logic that collectively enable a larger business capability.

Essentially, a business capability is a ‘what,’ and a subdomain is a ‘how’. Capabilities are about the business' purpose and the value it creates. Subdomains are about the detailed problem areas that need to be addressed from an implementation perspective. A single business capability may encompass multiple subdomains. Thinking in terms of layers or abstraction, business capabilities sit at a higher level than subdomains.

Now, let's get into some practical examples, and show some code to illustrate how these separations might manifest. Imagine we’re working on an e-commerce platform.

**Example 1: Business Capability - Order Management**

This capability encompasses everything related to a customer placing an order, from adding items to a cart to receiving a confirmation. It involves several subdomains. A simplified Python example illustrating the *interface* of this capability might look something like this:

```python
class OrderManagementService:
    def __init__(self, payment_gateway_service, inventory_service, shipping_service):
        self.payment_gateway_service = payment_gateway_service
        self.inventory_service = inventory_service
        self.shipping_service = shipping_service

    def create_order(self, cart_items, customer_id, shipping_address):
        # Placeholder - Orchestrates various subdomains
        total_amount = self._calculate_total(cart_items)
        if self.payment_gateway_service.process_payment(customer_id, total_amount):
            for item in cart_items:
                self.inventory_service.decrease_stock(item)
            shipping_details = self.shipping_service.calculate_shipping(cart_items, shipping_address)
            return { "success": True, "shipping": shipping_details}
        else:
            return {"success": False, "message": "Payment Failed"}
    
    def _calculate_total(self, cart_items):
        # Simplified total calculation (replace with actual logic)
         return sum(item['price'] for item in cart_items)
```

Here, the `OrderManagementService` is the entry point for the “order management” capability, delegating actions to other *domain* services.

**Example 2: Subdomain - Payment Gateway Integration**

This subdomain handles the interaction with external payment processors. This is a specific area focused on the technicalities of payment transactions. Here’s a simple example of the payment gateway subdomain service:

```python
class PaymentGatewayService:
    def __init__(self, api_credentials):
      self.api_credentials = api_credentials

    def process_payment(self, customer_id, amount):
        # Placeholder for interacting with a real payment gateway api.
        # This will include token handling, error management etc.
        print (f"Processing payment for {customer_id}, amount {amount}") # For demo.
        return True # Simulate success

```

Notice that this service is concerned with technical aspects – API authentication, payment processing logic, and error handling, not the larger business context. It’s a technical sub-area that enables the "order management" capability to work.

**Example 3: Subdomain - Inventory Management**

This subdomain focuses on keeping track of product stock levels. This is a separate, dedicated component.

```python
class InventoryService:

  def __init__(self, initial_stock={}):
      self.stock = initial_stock

  def decrease_stock(self, item):
      if item in self.stock and self.stock[item] > 0:
         self.stock[item] -=1
      else:
        raise ValueError(f"Cannot decrease stock of {item} with current amount : {self.stock.get(item, 0)}")

  def get_stock(self, item):
      return self.stock.get(item, 0)

```
Again, this service handles a specific technical responsibility, separated from the overall order management flow.

So, how do you decide which to apply? It really boils down to what you're trying to achieve. For designing a system that maps well to your business, and facilitates better cross-functional collaboration and ownership, structuring around business capabilities is the way to go. When breaking down complex tasks into smaller manageable components for development and maintenance, focusing on subdomains makes more sense. Ideally, you'll want a combination, using capabilities to organize high-level responsibilities, and subdomains to define the technical implementation within those.

A crucial aspect to avoid is simply mimicking the organizational chart. Although teams sometimes align along subdomains, that shouldn't be the defining factor. The architecture needs to reflect the *problem domain* and not the structure of the company.

To deepen your understanding, I strongly recommend digging into a few foundational resources:

*   **Domain-Driven Design: Tackling Complexity in the Heart of Software** by Eric Evans. This is the seminal work on DDD and provides a thorough explanation of both domains and subdomains, and how to map business needs to architectural components. It’s critical reading for anyone involved in software architecture, especially in the context of complexity.
*   **Implementing Domain-Driven Design** by Vaughn Vernon. This book offers a practical guide on how to implement the concepts from Evans' book, and it dives into specifics of subdomains and bounded contexts, showing real-world techniques.
*   **Building Microservices** by Sam Newman. This book doesn’t focus solely on DDD but discusses domain boundaries in the context of designing microservices and is extremely helpful for understanding how services are logically and technically scoped to enable autonomy and scalability.
*   **Patterns of Enterprise Application Architecture** by Martin Fowler. Although not exclusively focused on DDD, this book provides a wealth of information on common enterprise patterns which can be used to address many of the technical aspects of implementing subdomains, such as data access and transaction handling within a domain.

In my experience, understanding these nuances has been the difference between projects that are agile and adaptable, and those that turn into complex, unmanageable monoliths. It's less about following a rigid rule and more about understanding the underlying principles. The goal is to create a system that mirrors your business, is easy to change, and is ultimately maintainable. That involves a good understanding of what capabilities your system needs, and how they’re implemented as technical subdomains. It’s a journey of continuous refinement that demands critical thinking and an ongoing willingness to adapt.
