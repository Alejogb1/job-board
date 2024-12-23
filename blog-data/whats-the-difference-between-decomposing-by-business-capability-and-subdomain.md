---
title: "What's the difference between decomposing by business capability and subdomain?"
date: "2024-12-23"
id: "whats-the-difference-between-decomposing-by-business-capability-and-subdomain"
---

Alright,  I've seen this distinction cause more than a few late nights on projects, particularly when teams are transitioning towards more modular architectures. It's a subtle but crucial difference, and honestly, it can make or break how effectively you can adapt to changing requirements. The crux of the issue lies in how you're defining the boundaries of your system's responsibilities and, ultimately, how that reflects the organizational structure and goals.

To illustrate, think back to my stint at a mid-sized e-commerce platform. We initially had a monolith that did everything, including inventory management, order processing, and user authentication. We soon realized we were suffocating under our own weight—small changes in one area often had unpredictable ripple effects elsewhere. That’s when we started exploring decomposition, and that’s where this whole business capability versus subdomain debate came into sharp focus.

Decomposition by business capability focuses on what the business *does*, not necessarily how it does it internally. Think of them as broad, high-level functions that the business needs to perform to operate. These are usually aligned with the core activities that contribute directly to the business's value proposition. In our e-commerce context, business capabilities could include things like "managing the product catalog," "processing customer orders," "handling payments," or "providing customer support." Each of these represents a complete functional area that delivers business value. The important thing here is that it's business-driven and technology-agnostic to a large extent. The internal implementation might change over time, but the capability remains consistent as long as that business function is still required.

On the other hand, decomposition by subdomain delves into the details of how those capabilities are supported at a more granular level. A subdomain is a focused area within the business that has a specific set of requirements and challenges. Subdomains often have different technical needs or specialized expertise associated with them. They are more technical in nature and might vary drastically in size and complexity. For example, under the “managing the product catalog” capability, you might find subdomains like "product data management," "image processing," "pricing engine," and "search indexing." These are distinct areas of specialization that require different development approaches and skill sets.

The crucial point is that subdomains are usually contained within business capabilities, and one capability might span multiple subdomains, and vice-versa. Think of business capabilities as the 'what' and subdomains as the 'how'. The capabilities provide the broad context and the subdomains provide the specific areas.

Let's move into some examples that I've found helpful when trying to clarify this distinction with teams:

**Example 1: Inventory Management**

*   **Business Capability:** *Manage Inventory*. The overall capability of tracking and controlling stock levels across all distribution channels.
*   **Subdomains:**
    *   *Warehouse Management:* Manages stock levels, locations within warehouses, and optimizes storage.
    *   *Real-time Inventory Tracking:* Handles data feeds from various sources, and updates inventory levels.
    *   *Forecasting & Replenishment:* Utilizes statistical models to predict future demand and trigger replenishment processes.

Here's a simplified code snippet showing a potential structure where inventory tracking might be handled. This is obviously just illustrative:

```python
class InventoryTracker:
    def __init__(self):
        self.stock = {}

    def update_stock(self, product_id, quantity_change):
        if product_id in self.stock:
            self.stock[product_id] += quantity_change
        else:
            self.stock[product_id] = quantity_change

    def get_stock_level(self, product_id):
       return self.stock.get(product_id, 0)

class WarehouseManager:
    def __init__(self, inventory_tracker):
        self.inventory_tracker = inventory_tracker

    def process_incoming_stock(self, product_id, quantity):
        self.inventory_tracker.update_stock(product_id, quantity)
        print(f"Received {quantity} of {product_id}.")

    def get_current_stock(self, product_id):
        return self.inventory_tracker.get_stock_level(product_id)


# example of usage
tracker = InventoryTracker()
warehouse = WarehouseManager(tracker)
warehouse.process_incoming_stock("product_123", 100)
print(f"Current stock for product_123: {warehouse.get_current_stock('product_123')}")

```

In this example `InventoryTracker` and `WarehouseManager` can be considered part of different subdomains under `Manage Inventory`. The `InventoryTracker` could be part of the `Real-time Inventory Tracking` subdomain, and the `WarehouseManager` could be part of the `Warehouse Management` subdomain. The `Manage Inventory` capability is the overall functional area, whereas the subdomains focus on specifics.

**Example 2: Order Processing**

*   **Business Capability:** *Process Orders*. The full flow from order placement to completion.
*   **Subdomains:**
    *   *Order Capture & Validation:* Takes in new orders, validates data, and ensures a valid request.
    *   *Payment Processing:* Handles the financial transactions related to the order.
    *   *Order Fulfillment:* Manages picking, packing, and shipping the ordered goods.

Here, a code snippet focusing on payment processing might look like this:

```python
class PaymentProcessor:

  def __init__(self, gateway):
    self.gateway = gateway # assume that gateway handles specific API calls

  def process_payment(self, order_id, amount, payment_method):
    if self.gateway.charge(order_id, amount, payment_method):
      print(f"Payment of {amount} for order {order_id} processed successfully.")
      return True
    else:
      print(f"Payment failed for order {order_id}.")
      return False

class PaymentGateway:
    def charge(self, order_id, amount, payment_method):
        # Implementation that would interact with external payment providers
        # Example of fake success/fail behavior
        if payment_method == "credit_card":
             return amount > 0 #Assume for credit card we accept if amount > 0
        return False


# example of usage
gateway = PaymentGateway()
processor = PaymentProcessor(gateway)
processor.process_payment("order_567", 100, "credit_card")
processor.process_payment("order_568", 0, "credit_card")
processor.process_payment("order_569", 100, "paypal")
```

This demonstrates part of the `Payment Processing` subdomain, with the `PaymentProcessor` acting on behalf of that subdomain within the higher level capability of `Process Orders`.

**Example 3: Customer Support**

*   **Business Capability:** *Provide Customer Support*. The ability to assist customers with their queries and issues.
*   **Subdomains:**
    *   *Ticket Management:* Handles incoming customer requests.
    *   *Knowledge Base:* Provides self-service documentation and articles.
    *   *Live Chat Support:* Manages real-time interactions with customers.

A simplified example of live chat support could be:

```python

class ChatSession:

    def __init__(self, agent_id, customer_id):
        self.agent_id = agent_id
        self.customer_id = customer_id
        self.messages = []
        print(f"Chat started with agent: {agent_id}")


    def add_message(self, sender, message):
        self.messages.append({"sender": sender, "message":message})

    def get_last_message(self):
        if self.messages:
            return self.messages[-1]
        return None

class ChatManager:
    def __init__(self):
      self.sessions = {}

    def start_session(self, agent_id, customer_id):
        session = ChatSession(agent_id, customer_id)
        self.sessions[(agent_id, customer_id)] = session
        return session

    def end_session(self, agent_id, customer_id):
      if (agent_id, customer_id) in self.sessions:
        del self.sessions[(agent_id, customer_id)]
      else:
        print(f"No active chat session for agent {agent_id} and customer {customer_id}")


#example of usage

manager = ChatManager()
chat = manager.start_session("agent123", "customer456")
chat.add_message("customer456", "Hi, i have a question")
print(f"Last message: {chat.get_last_message()['message']}")
manager.end_session("agent123", "customer456")
```

In this case, `ChatSession` and `ChatManager` would fall into the `Live Chat Support` subdomain.

So, when making the decision between business capabilities and subdomains, consider the overall business value and the level of technical granularity you need to focus on. Starting with business capabilities helps you to map the core business functions, and then dive into subdomains as needed based on the required level of isolation and specialization. For further reading, I'd recommend Eric Evans' "Domain-Driven Design" – it's a classic, and while dense, it lays a strong theoretical foundation for understanding this concept. Also, "Building Microservices" by Sam Newman provides practical guidance on implementing these patterns in real-world systems. Additionally, Martin Fowler's website and related publications often explore various architectural styles, including considerations for decomposition.

In practice, I've found that starting with business capabilities often makes more sense, as these align with how the business itself functions. This helps to build better communication and collaboration between business and tech teams. Then, when you need to go deeper, subdomains can then provide the focus and technical detail you need for more granular decomposition. The key takeaway is that both are tools, and using them effectively is a matter of aligning them with the organizational and technical needs at hand. It’s a continuous process of learning and refinement, and it seldom is perfectly “right” the first time around. It’s about iteration and improvement.
