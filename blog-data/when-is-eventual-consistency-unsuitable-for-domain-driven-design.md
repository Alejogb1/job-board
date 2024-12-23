---
title: "When is eventual consistency unsuitable for Domain-Driven Design?"
date: "2024-12-23"
id: "when-is-eventual-consistency-unsuitable-for-domain-driven-design"
---

Alright,  I've seen my share of distributed systems, and the eventual consistency vs. strong consistency debate is one I've been through multiple times, often with teams trying to fit a square peg into a round hole. Domain-Driven Design (DDD) adds another layer of complexity here. It isn't a one-size-fits-all scenario, and that’s what makes it interesting. So, let's dive into when eventual consistency is a poor choice within the DDD context, pulling from my experiences.

First off, it's crucial to understand that eventual consistency, at its core, means data will become consistent *eventually*, but not necessarily instantaneously. This delay, however short it may be in practice, introduces a window of inconsistency. Now, when we are talking about strategic design in DDD, specifically focusing on aggregates, this delay can become problematic if not handled carefully. An aggregate, in DDD, is the consistency boundary. All operations within an aggregate should happen transactionally, meaning they must all succeed or none should. Using eventual consistency *within* an aggregate's boundary is essentially an architectural anti-pattern, as the core idea of an aggregate is to enforce invariants and maintain consistency within it.

I recall a project from years back, an e-commerce platform. We were naively trying to use a message queue to handle order updates. We'd have a user "place order," then throw a "order placed" message on the bus, which then asynchronously updated inventory. The problem, of course, was that we could potentially over-sell items if two orders came in almost simultaneously while inventory was still being updated based on the first order's message. The domain clearly required strong consistency on the inventory counts related to an order aggregate. We were trying to apply eventual consistency at the aggregate level, which led to many headaches and frantic bug fixes before we realized the mistake and refactored to maintain transactional operations within the order aggregate itself.

This brings us to the core point. Eventual consistency is ill-suited for scenarios within a bounded context (another DDD concept) where strong, immediate consistency is mandatory for business rules and data integrity. These are often contexts where a transaction is expected to either fully complete or fully fail. When that type of immediate and all-or-nothing consistency is required, eventual consistency can lead to inconsistent states, race conditions, and erroneous business decisions, often manifested as data corruption. These situations often occur within the core domain of a system. The core domain typically contains the key business logic and entities, and any data inconsistencies there have massive and immediate repercussions.

Let's look at some specific examples with code to illustrate these points. Here’s an example in a hypothetical Python context, assuming we are using a naive message queue approach that does not allow for atomic transactions:

```python
# Example 1: Incorrect use of eventual consistency inside an aggregate.

class InventoryItem:
    def __init__(self, sku, quantity):
        self.sku = sku
        self.quantity = quantity

    def decrease_stock(self, quantity):
        if self.quantity >= quantity:
            self.quantity -= quantity
            return True
        return False

class Order:
    def __init__(self, order_id, items):
        self.order_id = order_id
        self.items = items

    def place_order(self, inventory_service):
        for item_data in self.items:
            sku = item_data['sku']
            quantity = item_data['quantity']
            if inventory_service.decrease_stock_async(sku, quantity) is False:
              raise Exception("Insufficient stock for some items")
        print("Order Placed", self.order_id)
        return True

class InventoryService:

    def __init__(self):
        self.inventory = {}
    
    def initialize_inventory(self, initial_inventory):
        self.inventory = initial_inventory

    def decrease_stock_async(self, sku, quantity):
      if sku in self.inventory:
        item = self.inventory[sku]
        if item.decrease_stock(quantity):
             print(f"Decreased stock for {sku} by {quantity}")
             return True
        else:
            return False
      else:
          return False

inventory_data = {
    "sku1": InventoryItem(sku="sku1", quantity=10),
    "sku2": InventoryItem(sku="sku2", quantity=5),
}

inventory_service = InventoryService()
inventory_service.initialize_inventory(inventory_data)

order1 = Order(order_id="order1", items=[{'sku': 'sku1', 'quantity': 3}, {'sku': 'sku2', 'quantity': 1}])
order2 = Order(order_id="order2", items=[{'sku': 'sku1', 'quantity': 8}])


try:
    order1.place_order(inventory_service)
    order2.place_order(inventory_service)
except Exception as e:
  print(f"Error placing orders: {e}")
```
In this simplified example, if the `decrease_stock_async` method, is a message on a bus or asynchronous call and several orders for the same item are placed nearly simultaneously, one could bypass the checks within a given `InventoryItem` causing potential over-sales, which is a violation of aggregate consistency.

Now, here's how it *should* look within an aggregate using an in-memory representation, assuming a singular process:

```python
# Example 2: Correct use of strong consistency inside an aggregate.

class InventoryItem:
    def __init__(self, sku, quantity):
        self.sku = sku
        self.quantity = quantity

    def decrease_stock(self, quantity):
        if self.quantity >= quantity:
            self.quantity -= quantity
            return True
        return False

class Order:
    def __init__(self, order_id, items):
        self.order_id = order_id
        self.items = items

    def place_order(self, inventory_service):
        if inventory_service.validate_and_decrease_stock(self.items):
            print("Order Placed", self.order_id)
            return True
        else:
            raise Exception("Insufficient stock")

class InventoryService:

    def __init__(self):
        self.inventory = {}
    
    def initialize_inventory(self, initial_inventory):
        self.inventory = initial_inventory

    def validate_and_decrease_stock(self, items):
        # Attempt to decrease inventory for all items transactionally
        # If any stock is insufficient, rollback
      
        for item_data in items:
            sku = item_data['sku']
            quantity = item_data['quantity']
            if sku in self.inventory:
                if self.inventory[sku].quantity < quantity:
                  return False
            else:
                return False
        for item_data in items:
            sku = item_data['sku']
            quantity = item_data['quantity']
            self.inventory[sku].decrease_stock(quantity)
            print(f"Decreased stock for {sku} by {quantity}")
        return True


inventory_data = {
    "sku1": InventoryItem(sku="sku1", quantity=10),
    "sku2": InventoryItem(sku="sku2", quantity=5),
}

inventory_service = InventoryService()
inventory_service.initialize_inventory(inventory_data)


order1 = Order(order_id="order1", items=[{'sku': 'sku1', 'quantity': 3}, {'sku': 'sku2', 'quantity': 1}])
order2 = Order(order_id="order2", items=[{'sku': 'sku1', 'quantity': 8}])


try:
    order1.place_order(inventory_service)
    order2.place_order(inventory_service)
except Exception as e:
  print(f"Error placing orders: {e}")
```

This second example, when within a single process, makes sure that either all items for an order are decremented transactionally or, if one item lacks the required stock, none of the order items are decremented.

Now, to really drive the point home, let’s consider another example, where we're using event sourcing in a bank application. Consider our aggregate is an account. Using eventual consistency *within* the account aggregate would be disastrous.

```python
# Example 3: Event sourcing without a transactional approach in place.
import uuid
class Account:
  def __init__(self, account_id):
    self.account_id = account_id
    self.balance = 0
    self.events = []

  def deposit(self, amount):
      self.balance += amount
      event = {
            "event_id": uuid.uuid4(),
            "event_type": "Deposit",
            "amount": amount
        }
      self.events.append(event)

  def withdraw(self, amount):
      if self.balance >= amount:
          self.balance -= amount
          event = {
              "event_id": uuid.uuid4(),
              "event_type": "Withdrawal",
              "amount": amount
          }
          self.events.append(event)
      else:
        raise Exception("Insufficient funds")

  def get_balance(self):
     return self.balance

account = Account("123")
account.deposit(100)
account.withdraw(20)
print(f"Balance: {account.get_balance()} Events: {account.events}")


account2 = Account("124")
account2.deposit(100)
#What if deposit event, or withdraw event got lost?
```

In this example using event sourcing with in-memory representation, we have strong consistency within the `Account` aggregate. However, in a distributed system, it may be tempting to emit the `deposit` or `withdraw` events asynchronously to some database or message queue. If these writes aren’t atomic, we could lose an event, thus leading to a data inconsistency.

So, when *is* eventual consistency suitable? It is perfectly fine, and often preferred, for situations *between* bounded contexts or aggregates. For instance, you might have a "read model" that updates based on events published from the core domain, often a view of the data optimized for a different context. The delay in updating these read models is typically acceptable, as they’re usually not directly involved in core business transactions.

In essence, think of eventual consistency as a tool best suited for looser, less critical data operations and *never* inside a consistency boundary within a DDD aggregate. Within a bounded context, where invariants need to be maintained, strong consistency is paramount.

For further reading and a deeper technical understanding, I strongly recommend "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans as your starting point, followed by "Implementing Domain-Driven Design" by Vaughn Vernon for practical guidance. For a better understanding of consistency models in distributed systems, "Designing Data-Intensive Applications" by Martin Kleppmann provides an excellent in-depth exploration of various consistency models and their trade-offs.

In summary, eventual consistency within the core domain or aggregates of a system built using DDD is a pitfall. Understanding where to draw the consistency boundary is not merely a technical issue, it is a core aspect of strategic domain design and is crucial for building robust and correct applications.
