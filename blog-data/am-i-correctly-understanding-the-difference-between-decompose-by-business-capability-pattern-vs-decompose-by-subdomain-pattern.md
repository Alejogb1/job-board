---
title: "Am I correctly understanding the difference between decompose by business capability pattern vs decompose by subdomain pattern?"
date: "2024-12-23"
id: "am-i-correctly-understanding-the-difference-between-decompose-by-business-capability-pattern-vs-decompose-by-subdomain-pattern"
---

Alright, let’s tackle this. It’s a common point of confusion, and honestly, I've seen teams trip over this distinction more than once. I recall a particular project about five years ago, a rather large e-commerce platform overhaul. We were transitioning from a monolithic architecture, and initially, the team grappled significantly with this very question: should we decompose by business capability, or by subdomain? We tried both approaches, and the lessons learned were, shall we say, instructive. Let’s break down what I learned, and clarify the differences, using practical examples and insights.

The core idea behind both ‘decompose by business capability’ and ‘decompose by subdomain’ is to break down a large, complex system into smaller, more manageable units. This reduces complexity, improves maintainability, and allows teams to specialize more effectively. However, the criteria used for this decomposition—the ‘what’ we use to divide the system—is different. This difference is crucial and determines how the team is structured and ultimately, how the overall system evolves.

**Decompose by Business Capability:**

When decomposing by business capability, you're focusing on *what the business does*. Each capability represents a core activity, a self-contained unit of business logic that provides specific value to the customer or the business. Think of it as a verb, a ‘doing’ or an action that the business performs. Examples would include ‘order fulfillment,’ ‘customer management,’ ‘payment processing,’ or ‘product catalog management’. The key characteristics are:

*   **Business-centric:** The boundaries of each component are defined by business functions, not by technical considerations.
*   **End-to-end process:** A business capability often encapsulates a complete process from beginning to end. For example, 'order fulfillment' would include order intake, payment authorization, shipping, and tracking.
*   **Cross-domain:** A single business capability can often span multiple technical subdomains. It’s common to find elements from ‘database,’ ‘messaging,’ and ‘ui’ within a business capability component.
*   **Team structure:** Usually, a team is structured around one or more business capabilities, reflecting the business’s org structure and promoting end-to-end ownership.

**Decompose by Subdomain:**

Decomposition by subdomain, in contrast, focuses on *what the system *is* composed of. You're breaking down the system into distinct areas of the overall business domain. These areas often represent types of data or specific parts of the business. Think of it as a noun, or a ‘thing’ the business has. Examples include ‘product data,’ ‘customer accounts,’ ‘inventory levels,’ or ‘shipping locations.’ The defining factors are:

*   **Domain-centric:** The decomposition is based on the business domain itself, dividing it into natural, cohesive areas.
*   **Specialization:** Subdomains often allow for the development of specialist teams with deep expertise in specific business areas.
*   **Technical alignment:** Subdomains often tend to naturally align with underlying data models and technological choices. For example, the 'inventory subdomain' may rely more on inventory management systems and optimized databases.
*   **Potential for overlap:** Since subdomains are not tied to complete processes, a single workflow may require multiple subdomains to work together, potentially needing integration patterns.

**The Key Difference in Practice:**

Imagine a simple e-commerce system:

*   **Business Capability Example:** You might have a service called 'Manage Orders.' It would encapsulate the entire order process, interacting with the inventory subdomain, the payment subdomain, and the customer subdomain. It’s a complete business process.
*   **Subdomain Example:** You would have separate services for ‘Product Catalog,’ ‘Customer Management,’ ‘Payment,’ and ‘Inventory.’ Each service focuses on managing data related to that specific subdomain. A process like placing an order would involve interactions between these various subdomains.

Here are three working code snippets to help illustrate this:

**1. Decompose by Business Capability (Python):**

```python
# Business Capability: Manage Orders

class OrderManager:
    def __init__(self, inventory_svc, payment_svc, customer_svc):
        self.inventory_svc = inventory_svc
        self.payment_svc = payment_svc
        self.customer_svc = customer_svc

    def place_order(self, customer_id, product_ids):
        # 1. Check Inventory
        if not self.inventory_svc.check_availability(product_ids):
            raise Exception("Insufficient stock")

        # 2. Process Payment
        self.payment_svc.process_payment(customer_id, product_ids)

        # 3. Update Inventory
        self.inventory_svc.reserve_items(product_ids)

        # 4. Create Order Record
        self.customer_svc.create_order_record(customer_id, product_ids)

        print("Order placed successfully")


class InventoryService:
    def check_availability(self, product_ids):
       # Simplified check - usually connects to a real database
        return True # Assume availability
    def reserve_items(self, product_ids):
         print("Items reserved.")
class PaymentService:
    def process_payment(self, customer_id, product_ids):
       print("Payment Processed")

class CustomerService:
    def create_order_record(self, customer_id, product_ids):
       print("Order record created.")


inventory_service = InventoryService()
payment_service = PaymentService()
customer_service = CustomerService()
order_manager = OrderManager(inventory_service, payment_service, customer_service)
order_manager.place_order("user123", ["product1", "product2"])

```

In this case, `OrderManager` is the focal point, encapsulating the end-to-end flow of order management. It orchestrates interactions with the 'inventory,' 'payment,' and 'customer' systems.

**2. Decompose by Subdomain (Python):**

```python
# Subdomains: Inventory, Payment, Customer
class InventoryService:
    def get_product_availability(self, product_id):
         # Returns availability info from the database
        return True # simplified

    def reserve_inventory(self, product_id):
        print("Product reserved")

class PaymentService:
    def process_payment(self, customer_id, amount):
        print("Payment processed")

class CustomerService:
    def get_customer_details(self, customer_id):
        return {"customer_name": "John Doe"} # simplified

    def create_order_record(self, customer_id, product_ids):
        print("Order created")

# Workflow Implementation Outside of the Subdomains

inventory_svc = InventoryService()
payment_svc = PaymentService()
customer_svc = CustomerService()

customer_id = "user123"
product_ids = ["product1", "product2"]
total_amount = 100

if inventory_svc.get_product_availability(product_ids[0]) and inventory_svc.get_product_availability(product_ids[1]):
    payment_svc.process_payment(customer_id, total_amount)
    inventory_svc.reserve_inventory(product_ids[0])
    inventory_svc.reserve_inventory(product_ids[1])
    customer_svc.create_order_record(customer_id, product_ids)
    print("Order placed successfully.")
else:
    print("Items not available.")
```

Here, each service focuses purely on its subdomain's data and operations. The logic to place an order needs to be implemented outside the services, using each of them as independent components.

**3. Hybrid Approach (Python):**

```python
# Hybrid Approach - Combining Business Capability with Subdomains

class OrderProcessor: # Business Capability Level
    def __init__(self, inventory_svc, payment_svc, customer_svc):
        self.inventory_svc = inventory_svc
        self.payment_svc = payment_svc
        self.customer_svc = customer_svc

    def process_order(self, customer_id, product_ids):

        # 1. check availability via the inventory subdomain
        if not self.inventory_svc.check_availability(product_ids):
            raise Exception("Insufficient stock")
        # 2. payment via the payment subdomain
        self.payment_svc.process_payment(customer_id, product_ids)
        # 3. update inventory via the inventory subdomain
        self.inventory_svc.reserve_items(product_ids)
        #4. Create order via customer subdomain
        self.customer_svc.create_order(customer_id, product_ids)

        print("Order processed successfully")

class InventoryService: # Subdomain Level
    def check_availability(self, product_ids):
        # Simplified Check, connects to database
        return True
    def reserve_items(self, product_ids):
         print("Items reserved")

class PaymentService: # Subdomain Level
    def process_payment(self, customer_id, product_ids):
      print("Payment Processed")

class CustomerService: # Subdomain Level
    def create_order(self, customer_id, product_ids):
        print("Order created")


inventory_service = InventoryService()
payment_service = PaymentService()
customer_service = CustomerService()
order_processor = OrderProcessor(inventory_service, payment_service, customer_service)
order_processor.process_order("user123", ["product1", "product2"])

```
This hybrid approach utilizes a higher-level `OrderProcessor` to orchestrate the flow, but the actual data management resides within the subdomain services.

**Choosing The Correct Pattern:**

Which approach is better depends entirely on your context.

*   **Business Capability** is beneficial when you need end-to-end ownership, want to align with your organization, and need clear responsibility boundaries. This aligns well with domain-driven design and microservices patterns where each service owns a particular business responsibility. However, it can potentially lead to tight coupling with business processes.
*   **Subdomain** provides better technical independence, allowing specialization and potentially reducing duplication. However, it can create integration complexity, and it may blur ownership boundaries without a proper orchestration layer.

In practice, it is common to see teams utilize a *hybrid* approach, combining aspects of both patterns to fit their specific needs. A business capability may, internally, be composed of different subdomains working together. There is rarely a hard-and-fast ‘correct’ answer.

**For Further Reading:**

For a deeper dive into these concepts, I recommend the following resources:

*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans:** This book lays the foundational concepts for domain modeling and how it influences software architecture. It explains the idea of bounded contexts which map well to subdomains.
*   **"Building Microservices: Designing Fine-Grained Systems" by Sam Newman:** This is a practical guide to designing and implementing microservices, offering a great look at practical considerations in applying these patterns.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** While broader in scope, this book provides invaluable context to many architectural concepts, including the different options for component decomposition.

So, were you understanding the difference? I hope this detailed explanation based on my past experiences helps clarify things. It’s not uncommon to need to iterate on these choices during a project, and the most important thing is to make sure you're consciously making that decision based on your specific needs and constraints.
