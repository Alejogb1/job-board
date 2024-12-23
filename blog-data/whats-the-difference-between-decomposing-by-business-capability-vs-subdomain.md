---
title: "What's the difference between decomposing by business capability vs. subdomain?"
date: "2024-12-23"
id: "whats-the-difference-between-decomposing-by-business-capability-vs-subdomain"
---

Alright, let's unpack this. I’ve seen this distinction cause more than a few headaches on projects, and understanding it is truly fundamental when architecting anything of reasonable complexity. The difference between decomposing by business capability versus subdomain isn't just a semantic quirk; it's a strategic decision that impacts how your systems evolve, how your teams organize, and frankly, how much future pain you’re likely to encounter.

In my experience, particularly back when I was working on a rather large e-commerce platform, we initially leaned heavily towards a subdomain-focused architecture. We had ‘Inventory,’ ‘Payment,’ ‘User Management,’ and so on. Seemed logical at the time. The trouble started when changes in one of these subdomains, specifically regarding new payment methods, rippled outwards unexpectedly and impacted the shopping cart functionality, which we'd neatly categorized separately. The coupling was far tighter than we anticipated. This experience taught me a brutal, but valuable lesson: subdomains, while useful for aligning with technical domains or existing organizational structures, often don’t capture the *what* of the business, only the *how*.

Business capabilities, on the other hand, represent *what* a business does, not *how* it does it. They are high-level functions that deliver value directly to the customer or the business itself. Think of a business capability as an activity that the business performs— things like ‘Order Fulfillment,’ ‘Customer Relationship Management,’ or ‘Product Catalog Management.’ These are the business activities that persist even if the underlying technology or organizational structures change. They are the “why” behind the software.

Now, it's critical to understand they’re not mutually exclusive, and the most effective architectures often employ a hybrid approach, drawing on the strengths of each. We eventually refactored that e-commerce platform to a capability-driven design, which wasn't easy, but it paid off in the long run.

So, let's break this down with some examples. Consider a basic scenario: an online bookstore.

**Example 1: Subdomain-focused Decomposition**

Here’s how you might define subdomains for an online bookstore:

```python
class BookDatabase:
    def add_book(self, book_details):
        # Logic to add a book to the database
        print(f"Book added: {book_details['title']}")

    def get_book(self, book_id):
        # Logic to retrieve a book from the database
        print(f"Book retrieved: {book_id}")

class UserManagement:
    def create_user(self, user_details):
        # Logic to create a new user
        print(f"User created: {user_details['username']}")

    def get_user(self, user_id):
        # Logic to get user details
        print(f"User retrieved: {user_id}")

class PaymentGateway:
    def process_payment(self, payment_details):
        # Logic to process payments
        print(f"Payment processed: {payment_details['amount']}")

    def refund_payment(self, payment_id):
       # Logic to process refunds
       print(f"Payment refunded: {payment_id}")

```

Here, we have distinct subdomains: `BookDatabase`, `UserManagement`, and `PaymentGateway`. Each is responsible for its internal domain functionality. However, this approach risks creating silos, and changes in one subdomain often affect other subdomains. Imagine changing the logic of payments (say adding a new processor) – you'd potentially have to touch both the `PaymentGateway` and the shopping cart logic, which might sit outside these subdomains.

**Example 2: Capability-focused Decomposition**

Now, let's reframe this using business capabilities:

```python
class ProductCatalogManagement:
    def add_product(self, product_details):
        # Logic to add a product
        print(f"Product added to catalog: {product_details['title']}")

    def retrieve_product_details(self, product_id):
        # Logic to retrieve product details
        print(f"Product details retrieved: {product_id}")


class OrderProcessing:
    def create_order(self, order_details):
        # Logic to create a new order
        print(f"Order created: {order_details['order_id']}")

    def process_order_payment(self, order_id, payment_details):
        # Logic to handle the processing of payment
        print(f"Payment processed for order {order_id}: {payment_details['amount']}")
        #This method now encapsulates both order and payment concerns

    def fulfil_order(self, order_id):
        # Logic for fulfilling an order
        print(f"Order fulfilled: {order_id}")

class CustomerAccountManagement:
   def create_account(self, user_details):
        # Logic for creating new accounts
       print(f"Account Created for {user_details['username']}")

   def get_account_details(self, user_id):
        # Logic for retrieving an accounts details
       print(f"Account details retrieved for user: {user_id}")

```

Here, the capabilities are `ProductCatalogManagement`, `OrderProcessing`, and `CustomerAccountManagement`.  Observe how `OrderProcessing` now owns the entire process of taking an order and processing payments.  This is a key distinction. Instead of payments being a separate concern, it’s part of the larger capability of “Order Processing.”  This approach reduces the chances of unexpected coupling.

**Example 3: A Hybrid Approach**

The best approach, often, sits in the middle:

```python
class InventoryService:
   def update_stock(self, product_id, quantity_change):
      # Logic for updating stock based on sale or return
      print(f"Stock Updated for {product_id}, change {quantity_change}")

class CatalogService:
   def get_book_details(self, book_id):
       # Logic to retrieve detailed information of a book
       print(f"Book Details Retrived: {book_id}")

class OrderManagement:
   def create_order(self, order_details):
        # Order creation logic
        print(f"Order Created: {order_details['order_id']}")

   def process_order(self, order_id, payment_details):
       #Order payment logic
       print(f"Payment Processed for Order {order_id}: {payment_details['amount']}")
       InventoryService().update_stock(order_details['product_id'], -order_details['quantity'])

   def fulfil_order(self, order_id):
       #Logic to complete an order
       print(f"Order Fulfilled: {order_id}")


```

In this hybrid example, you'll see a `CatalogService` and `InventoryService`. These *could* be considered subdomains, but they're scoped to service and support the high-level capability of `OrderManagement`.  The capability is the driving force, and the subdomains exist to support its functionality. When `OrderManagement` calls to `process_order`, it also triggers a call to the `InventoryService` to reduce stock. This helps to delineate responsibility, but not isolate functionality. This approach allows for future adjustments to the `InventoryService` (say, adding an api) without impacting the wider ordering functionality.

**The Crucial Considerations**

Choosing between a subdomain- or capability-focused approach, or a hybrid, requires some introspection on your end.

*   **Business Context:**  How rapidly is the business changing? Capability-focused designs tend to be more resilient to shifts in the market because they’re tied to the core *what* rather than the evolving *how*.
*   **Team Structure:**  How are your teams organized? Do your teams align more closely with the technical subdomains or the business capabilities? It often makes sense for teams to align to the boundaries you've defined with your architectural approach.
*   **Communication Patterns:**  Do your subdomains have strong interdependencies? Capability-based decomposition can help isolate those concerns and prevent unintended side effects.
*   **Scalability:** How will this design scale in the long run? Are there potential bottlenecks in the current design?  Capabilities can scale more gracefully since they naturally encapsulate related functions.

For further reading on the topic, I would highly recommend two specific sources. First, Eric Evans' book *Domain-Driven Design: Tackling Complexity in the Heart of Software*. This book provides an excellent foundation for understanding both subdomains and capabilities within a broader design context. Second, check out *Building Microservices* by Sam Newman. This work gives valuable practical insights on how these concepts play out in a microservice-oriented architecture, which is where this kind of thinking really shines.

In conclusion, the choice between decomposing by business capability versus subdomain isn't a binary one; it's about understanding the trade-offs and selecting the approach that aligns with the specific needs of the business and your technical constraints. I've found that capabilities lead to more cohesive and adaptable systems, while a hybrid approach often provides the best of both worlds by leveraging subdomains as useful elements within a capability-centric design. The key is to remain flexible and adaptable, and to consistently evaluate your architecture as your business and technology landscape evolves.
