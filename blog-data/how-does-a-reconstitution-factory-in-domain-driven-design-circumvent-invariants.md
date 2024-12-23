---
title: "How does a reconstitution factory in Domain-Driven Design circumvent invariants?"
date: "2024-12-23"
id: "how-does-a-reconstitution-factory-in-domain-driven-design-circumvent-invariants"
---

, let's dive into this. I remember a particularly tricky project a few years back involving a complex logistics system. We were deep in domain-driven design territory, aiming for a microservices architecture, and the challenge of maintaining aggregate invariants across service boundaries came up repeatedly. One of the most interesting challenges was, precisely as the question poses, how a “reconstitution factory” impacted invariant maintenance. It's not always intuitive, so let's get to it.

The core problem stems from the fact that aggregates, in DDD, are intended to be units of consistency. They encapsulate data and logic, enforcing rules about how their state can change. These rules—the invariants—are critically important to the integrity of the domain. Now, a “reconstitution factory” isn't a standard term in the canonical DDD literature like Eric Evans’ book, but it’s a concept we often use in practice. Think of it as a piece of code responsible for reconstructing an aggregate from a persisted representation, often for operations beyond the aggregate’s own domain logic. Consider this: a typical aggregate load would go through the repository, which ensures that the aggregate is in a consistent state after the load based on its constructor. The reconstitution factory, in contrast, may have slightly less strict requirements as long as the integrity of the aggregate is intact once all operations are performed. However, this “less strict” area is where problems can arise concerning invariants.

The common scenario where you need this factory is when performing migrations, bulk data processing, and other operations that require the manipulation of an aggregate’s underlying state outside of its usual transactional context. It's like putting Humpty Dumpty back together again. Each piece might individually look correct, but the whole might not be if the reassembly doesn't adhere to all the rules.

The way we tackled this, and I've seen it applied effectively in several other contexts, involves a three-pronged approach: first, carefully defining the necessary reconstitution parameters that respect the invariant constraints; second, utilizing a combination of static checks during the factory’s reconstitution process; and finally, using defensive coding to validate the reconstituted aggregate's integrity prior to performing any further operation. Let's break it down with a few code examples. I'll use Python for its readability, but the principles apply across languages.

**Example 1: Basic Invariant Check during Reconstitution**

Imagine we have an `Order` aggregate with a simple invariant: the `order_total` must be greater than zero if there are any items in the order.

```python
class OrderItem:
    def __init__(self, product_id, quantity, unit_price):
      self.product_id = product_id
      self.quantity = quantity
      self.unit_price = unit_price

class Order:
    def __init__(self, order_id, items=None):
      self.order_id = order_id
      self.items = items if items else []
      self.order_total = self._calculate_total()

    def add_item(self, item):
        self.items.append(item)
        self.order_total = self._calculate_total()

    def _calculate_total(self):
      return sum(item.quantity * item.unit_price for item in self.items)

    def is_valid(self):
        return not self.items or self.order_total > 0


class OrderReconstitutionFactory:

    @staticmethod
    def reconstitute(order_id, items):
      # We recreate an order with pre-existing items
      order = Order(order_id, items)
      if not order.is_valid():
          raise ValueError("Order reconstitution failed: Invariant violated")
      return order

# Usage example
items = [OrderItem(product_id='123', quantity=2, unit_price=10), OrderItem(product_id='456', quantity=1, unit_price=20)]
try:
    reconstituted_order = OrderReconstitutionFactory.reconstitute('order-1', items)
    print(f"Order '{reconstituted_order.order_id}' was successfully reconstituted with a total of {reconstituted_order.order_total}")
except ValueError as e:
    print(f"Error: {e}")

# Example of a violation
try:
  reconstituted_order = OrderReconstitutionFactory.reconstitute('order-2', []) #items are empty but the order is still loaded.
  print(f"Order '{reconstituted_order.order_id}' was successfully reconstituted with a total of {reconstituted_order.order_total}")
except ValueError as e:
  print(f"Error: {e}")
```

In this example, the `OrderReconstitutionFactory.reconstitute` method explicitly checks the validity of the `Order` after its reconstruction. If the invariant isn't satisfied, it throws an exception, preventing the creation of an invalid object.

**Example 2: Complex Invariants and Partial Reconstitution**

Let’s consider a more intricate scenario. Suppose the `Order` aggregate also has a rule that specifies a discount can only be applied if there is a specific customer segment. This might involve a customer aggregate which may or may not be available during reconstitution. We can't assume it's always there and we may need to reconstitute the order without this information and handle this separately.

```python
class Customer:
  def __init__(self, customer_id, segment):
    self.customer_id = customer_id
    self.segment = segment

class Order:
    def __init__(self, order_id, items=None, customer=None, discount=0):
      self.order_id = order_id
      self.items = items if items else []
      self.customer = customer
      self.discount = discount
      self.order_total = self._calculate_total()

    def add_item(self, item):
        self.items.append(item)
        self.order_total = self._calculate_total()

    def _calculate_total(self):
      total = sum(item.quantity * item.unit_price for item in self.items)
      return total * (1 - self.discount)

    def apply_discount(self, discount):
        if self.customer and self.customer.segment == 'premium':
          self.discount = discount
          self.order_total = self._calculate_total()
        else:
           raise ValueError("Discount cannot be applied for non premium customers")

    def is_valid(self):
        if not self.items and self.order_total > 0:
            return False #Total can not be > 0 when there are no items
        if self.customer and self.discount > 0 and self.customer.segment != 'premium':
            return False
        return True


class OrderReconstitutionFactory:
    @staticmethod
    def reconstitute(order_id, items, discount, customer_segment = None):
      # We recreate an order with pre-existing items
      customer = None
      if customer_segment:
          customer = Customer(customer_id = "customer1", segment = customer_segment)
      order = Order(order_id, items, customer, discount)

      if not order.is_valid():
          raise ValueError("Order reconstitution failed: Invariant violated")
      return order

# Usage example
items = [OrderItem(product_id='123', quantity=2, unit_price=10), OrderItem(product_id='456', quantity=1, unit_price=20)]
try:
    reconstituted_order = OrderReconstitutionFactory.reconstitute('order-1', items, discount = 0.1, customer_segment='premium')
    print(f"Order '{reconstituted_order.order_id}' was successfully reconstituted with a total of {reconstituted_order.order_total}")
except ValueError as e:
    print(f"Error: {e}")

# Example of a violation
try:
  reconstituted_order = OrderReconstitutionFactory.reconstitute('order-2', items, discount = 0.1) # Discount applied without a customer
  print(f"Order '{reconstituted_order.order_id}' was successfully reconstituted with a total of {reconstituted_order.order_total}")
except ValueError as e:
  print(f"Error: {e}")

try:
  reconstituted_order = OrderReconstitutionFactory.reconstitute('order-3', items, discount = 0.1, customer_segment='non-premium')
  print(f"Order '{reconstituted_order.order_id}' was successfully reconstituted with a total of {reconstituted_order.order_total}")
except ValueError as e:
  print(f"Error: {e}")

```

Here, the factory allows for reconstituting an order without complete customer context. If a customer segment is provided, the factory validates that the discount aligns with the segment. It demonstrates how we manage scenarios with partial information but maintain invariant rules. Notice that I have added a check for the order total, this is an example of a situation where the factory makes sure the aggregate is in a correct state, even if the data is being reconstituted outside of the original bounded context.

**Example 3: Post-Reconstitution Validation and Repair**

Sometimes, even after a careful reconstitution, minor discrepancies might exist due to edge cases in historical data, or incomplete migration mapping. Instead of making the factory too complex, we can leverage a validation step post-reconstitution and attempt to repair the aggregate.

```python

class Order:
    def __init__(self, order_id, items=None, customer=None, discount=0, order_total=0): # total is now supplied on reconstitution
      self.order_id = order_id
      self.items = items if items else []
      self.customer = customer
      self.discount = discount
      self.order_total = order_total

    def add_item(self, item):
        self.items.append(item)
        self.order_total = self._calculate_total()

    def _calculate_total(self):
      total = sum(item.quantity * item.unit_price for item in self.items)
      return total * (1 - self.discount)

    def apply_discount(self, discount):
        if self.customer and self.customer.segment == 'premium':
          self.discount = discount
          self.order_total = self._calculate_total()
        else:
           raise ValueError("Discount cannot be applied for non premium customers")

    def validate(self):
        calculated_total = self._calculate_total()
        if not self.items and self.order_total > 0:
            return False #Total can not be > 0 when there are no items
        if self.customer and self.discount > 0 and self.customer.segment != 'premium':
            return False

        if abs(self.order_total - calculated_total) > 0.001: # handle floating point imprecision
            self.order_total = calculated_total
            return False # repaired but not valid
        return True

    def repair(self):
        self.order_total = self._calculate_total()

class OrderReconstitutionFactory:
    @staticmethod
    def reconstitute(order_id, items, discount, order_total, customer_segment = None):
        customer = None
        if customer_segment:
          customer = Customer(customer_id = "customer1", segment = customer_segment)
        order = Order(order_id, items, customer, discount, order_total)
        return order

# Usage example

items = [OrderItem(product_id='123', quantity=2, unit_price=10), OrderItem(product_id='456', quantity=1, unit_price=20)]

reconstituted_order = OrderReconstitutionFactory.reconstitute('order-1', items, discount = 0.1, order_total=30, customer_segment='premium')

if reconstituted_order.validate():
    print(f"Order '{reconstituted_order.order_id}' is valid and total is {reconstituted_order.order_total}")
else:
    print(f"Order '{reconstituted_order.order_id}' required repair")
    reconstituted_order.repair()
    if reconstituted_order.validate():
        print(f"Order '{reconstituted_order.order_id}' is valid after repair and total is now {reconstituted_order.order_total}")
    else:
         print(f"Order '{reconstituted_order.order_id}' is still invalid after repair and total is {reconstituted_order.order_total}")

#Example of invalid total
reconstituted_order = OrderReconstitutionFactory.reconstitute('order-2', items, discount = 0.1, order_total=10, customer_segment='premium')

if reconstituted_order.validate():
    print(f"Order '{reconstituted_order.order_id}' is valid and total is {reconstituted_order.order_total}")
else:
    print(f"Order '{reconstituted_order.order_id}' required repair")
    reconstituted_order.repair()
    if reconstituted_order.validate():
        print(f"Order '{reconstituted_order.order_id}' is valid after repair and total is now {reconstituted_order.order_total}")
    else:
         print(f"Order '{reconstituted_order.order_id}' is still invalid after repair and total is {reconstituted_order.order_total}")


```

In this scenario, we see that the reconstitution factory creates the aggregate with a total passed in as input, this could be data sourced from an external system. The validate method can then be called after the aggregate is created to ensure all invariants are valid, if not, it can try to repair the object. The repair method in this case can simply recalculate the total based on the items and discount.

This approach gives you more flexibility to handle legacy data discrepancies without complicating the core logic or factory itself.

**Further Reading**

For a deeper understanding of these principles, I'd recommend:

*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans:** This is the foundational text on DDD.
*   **"Implementing Domain-Driven Design" by Vaughn Vernon:** This book provides practical examples and implementation details.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** While not strictly DDD, this book provides a lot of insights into the construction of robust software systems including many relevant patterns.

In conclusion, a reconstitution factory can introduce potential violations of aggregate invariants if not implemented with care. We can mitigate this by carefully designing the factory parameters, performing static checks during the creation process, and having a post-reconstitution validation step combined with potential repair. These techniques ensure that, even when reconstituting an aggregate outside its usual lifecycle, the core integrity of our domain model is preserved. I've applied these principles across several projects and they have held up quite well even in the face of complex domain rules and data challenges.
