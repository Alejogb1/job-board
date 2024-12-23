---
title: "How can domain be identified in DDD?"
date: "2024-12-23"
id: "how-can-domain-be-identified-in-ddd"
---

Alright, let's tackle this domain identification question. It's one I’ve had to grapple with more than a few times, particularly back in the early days of moving away from monolithic applications. The shift to domain-driven design (ddd) can feel a bit nebulous at first, especially when you're used to thinking in terms of technical layers instead of business concepts. Identifying the domain isn’t a purely theoretical exercise; it's a crucial step that shapes how you structure your software.

My experience has taught me that there's no single, magical algorithm, rather a careful, iterative process. It’s less about finding a pre-existing domain carved in stone, and more about actively *defining* the domain through collaboration and exploration. The central idea revolves around understanding the business and its core activities—the very heart of what the organization does. We’re not building software for the sake of it; we’re building software to support the business, and therefore, we need to mirror its realities in our code.

One of the first things I learned the hard way is the importance of talking directly to the domain experts—those people who are deeply involved in the day-to-day operations. These experts usually aren't technical and might struggle with translating their understanding of the business to software jargon, but they possess the crucial knowledge. It’s our job to bridge that gap. We can't just rely on project managers or business analysts; we need to engage directly with the people who actually perform the work. These early conversations often reveal hidden complexities and tacit knowledge that you won’t find documented anywhere.

To begin identifying the domain, I typically start with a few key steps. First, I try to identify what are the main *business capabilities*—what can the business actually *do*? For a fictional e-commerce system, these could include order processing, inventory management, customer support, and so forth. These capabilities are typically quite high-level.

Then, we need to explore the bounded contexts. A bounded context represents a specific area of the domain, where particular language and concepts hold true. It’s like a conceptual boundary that helps us manage complexity. Different bounded contexts may use the same terms, but with subtly different meanings, and this is key. For instance, in a product ordering system, “product” in the “catalog” context has different properties and lifecycle requirements from the “product” in the “inventory” context. This subtle distinction makes a big difference in how you design each context. We should strive for context boundaries that align with the team structure as well to promote better collaboration.

Finally, we're looking for core subdomains and supporting subdomains. Core subdomains represent the aspects of the business that differentiate it from competitors. These should be your investment focus and deserve the best people. Supporting subdomains are needed, but don't provide a unique advantage. You should not over-engineer them as they typically don't provide a competitive advantage. Generic subdomains are things that could easily be implemented with off-the-shelf tools or outsourced.

Now, let's explore this with some code. Consider a simplified e-commerce system. First, we could define a very basic `Order` entity, but we need to consider the context of which this is used:

```python
# In the 'Ordering' bounded context
class Order:
    def __init__(self, order_id, customer_id, items, order_date):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = items
        self.order_date = order_date

    def calculate_total(self):
      total = 0
      for item in self.items:
        total += item.price * item.quantity
      return total

class OrderItem:
   def __init__(self, product_id, price, quantity):
      self.product_id = product_id
      self.price = price
      self.quantity = quantity


# Sample Usage
order_item1 = OrderItem(product_id="prod123", price=25.00, quantity=2)
order_item2 = OrderItem(product_id="prod456", price=10.00, quantity=1)

order1 = Order(order_id="order001", customer_id="cust001",
              items=[order_item1,order_item2], order_date="2024-03-08")

print(f"Total cost: ${order1.calculate_total()}")
```

In this initial example, `Order` is part of the 'ordering' context. The order processing logic is central here. But now let's say we have another aspect of the business which is the inventory.

```python
# In the 'Inventory' bounded context
class Product:
    def __init__(self, product_id, sku, name, current_stock, unit_price):
        self.product_id = product_id
        self.sku = sku
        self.name = name
        self.current_stock = current_stock
        self.unit_price = unit_price

    def adjust_stock(self, quantity_change):
        self.current_stock += quantity_change

# Sample Usage
product = Product(product_id="prod123", sku="SKU123", name="Laptop", current_stock=100, unit_price=25.00)
print(f"Initial stock: {product.current_stock}")
product.adjust_stock(-2)
print(f"Stock after sale: {product.current_stock}")
```

Notice that the `Product` object is used in a different way than `OrderItem`. In the 'inventory' context it represents a product and its stock information, while in the 'ordering' context it is only a representation of the item in an order, with its price. Both represent a ‘product’, but the meaning is slightly different depending on the context, and there are different operations allowed depending on the context. In the ‘inventory’ context, operations regarding stock level are important. In the ‘ordering’ context, knowing the cost of the item is what’s important.

Finally, let's examine a potential 'customer' context:

```python
# In the 'Customer' bounded context
class Customer:
    def __init__(self, customer_id, name, email, shipping_address):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.shipping_address = shipping_address

    def update_address(self, new_address):
        self.shipping_address = new_address

# Sample Usage
customer = Customer(customer_id="cust001", name="John Doe", email="john.doe@example.com", shipping_address="123 Main St")
print(f"Old address: {customer.shipping_address}")
customer.update_address("456 New Ave")
print(f"New address: {customer.shipping_address}")
```

Here, the customer is a domain entity focused on customer information and associated actions. The concept of customer is only loosely related to the concept of ‘order’. This is again intentional as a customer can place multiple orders, and the customer information is not just about specific orders, but the customer as a whole.

The code examples illustrate how the same concept can vary slightly depending on the context, and that the code has to evolve to match how the business operates. This isn't an abstract exercise. These represent distinct units of the business. It’s crucial that we understand the business in order to decompose the domain in a way that reflects it. Each context will likely have its own model, terminology, and even technologies. For example, the 'inventory' context might involve a time-series database while the 'customer' context might interact with a crm system.

To further solidify your understanding, I would recommend exploring Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software," which is foundational. Another good resource is Vaughn Vernon's "Implementing Domain-Driven Design" which is more practical in nature and deals with implementation patterns. Also, Alberto Brandolini's "Introducing Event Storming" is invaluable for facilitating collaborative domain exploration. These resources provide the theoretical background combined with practical techniques to build a successful DDD approach. They emphasize the importance of ubiquitous language, which is the shared vocabulary used by both domain experts and developers.

In closing, identifying the domain in ddd is a continuous process of learning, adaptation, and refinement, not a single event. It's a conversation. It's a collaboration. It's a journey that deeply reflects how the business actually works. It’s not easy, but the payoff is significantly improved business-aligned software.
