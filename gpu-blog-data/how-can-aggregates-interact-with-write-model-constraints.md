---
title: "How can aggregates interact with write model constraints?"
date: "2025-01-26"
id: "how-can-aggregates-interact-with-write-model-constraints"
---

Aggregates, in the context of Domain-Driven Design (DDD), represent transactional boundaries and consistency units, demanding that any interaction modifying their state adheres strictly to their defined invariants. This directly implies that write model constraints, the business rules dictating permissible state transitions, must be enforced at the aggregate level. Failing this introduces the possibility of data inconsistencies and breaks down the integrity of the domain model.

The relationship between aggregates and write model constraints is primarily about managing concurrency and ensuring data integrity within those boundaries. An aggregate is responsible for maintaining its own consistency; it will not allow itself to be placed into an invalid state by outside forces. When a command is directed toward an aggregate, the first action undertaken is to validate if this command can be fulfilled based on current state and its internal constraints. If the validation fails, the write operation should be rejected, often resulting in an exception being returned to the calling service. Only when the proposed state transition is considered valid should the aggregate update its internal data structure. This entire process is, in essence, constraint enforcement at the aggregate level.

Consider a simplified example of an `Order` aggregate.  This order might have a constraint specifying that it cannot be modified once it has been marked as ‘Shipped’. If an attempt is made to add a new `OrderItem` to an order that has already been shipped, this constraint must raise an error, preventing the modification. This constraint, an integral part of the order aggregate, acts as the gatekeeper for all write operations within the order’s context. The logic to check the `Shipped` state is not stored elsewhere.

To further illustrate this, consider three code examples representing diverse scenarios of constraint enforcement.

**Example 1: Basic State-Based Constraint**

This example demonstrates the constraint described previously, where modification to a shipped order is prohibited. This is a common state-based rule. The `Order` class holds the state, and attempts to add to items when shipped will raise an exception.

```python
class Order:
    def __init__(self, order_id):
        self.order_id = order_id
        self.items = []
        self.shipped = False

    def add_item(self, item):
      if self.shipped:
        raise OrderShippedError("Cannot add items to a shipped order.")
      self.items.append(item)

    def mark_shipped(self):
      self.shipped = True

class OrderShippedError(Exception):
    pass

# Usage
order = Order("123")
order.add_item("Widget")
order.mark_shipped()
try:
  order.add_item("Gasket") # Raises OrderShippedError
except OrderShippedError as e:
    print(f"Error: {e}")

```

*Commentary:* Here, `add_item` first checks if `shipped` is `True`. If so, the `OrderShippedError` is raised. This effectively enforces a direct constraint based on the internal state of the order aggregate. The error provides direct information about the specific business rule that was violated. This approach encapsulates the constraint logic directly within the aggregate, promoting clarity and reducing opportunities for external rule bypass. It also provides a consistent way to signal that modification is prohibited based on current state.

**Example 2: Constraint Based on Aggregate Data and external value**

In this example, the aggregate has a maximum number of seats. It has a capacity, and an operation to add a member. The constraint is that no more than the maximum capacity can be reached.

```python
class Concert:
    def __init__(self, concert_id, capacity):
        self.concert_id = concert_id
        self.capacity = capacity
        self.members = []

    def add_member(self, member_id):
        if len(self.members) >= self.capacity:
            raise ConcertFullError("Cannot add another member. Concert capacity reached.")
        self.members.append(member_id)

    def get_member_count(self):
        return len(self.members)

class ConcertFullError(Exception):
  pass


# Usage
concert = Concert("RockShow1", 5)

concert.add_member("user1")
concert.add_member("user2")
concert.add_member("user3")
concert.add_member("user4")
concert.add_member("user5")

try:
  concert.add_member("user6") # Raises ConcertFullError
except ConcertFullError as e:
  print(f"Error: {e}")

print(f"Number of members: {concert.get_member_count()}")
```

*Commentary:* This example expands upon the first, using a combination of internal state (`members`) and a configured value (`capacity`). The check for capacity is done *within* the `add_member` method and is not handled externally. It highlights that constraint enforcement can consider data within the aggregate and that it can use pre-configured values to decide on its state transitions. This demonstrates how constraints can be derived from multiple factors, still being directly managed by the aggregate itself.

**Example 3: Constraint Based on External Service (Indirectly)**

This example considers a constraint requiring an external check, not a state within the aggregate.  In this case, we have a loyalty program that can only grant points to users if they have been active within the past month.

```python
import datetime

class LoyaltyProgram:
    def __init__(self):
        self.last_active = {}
        # pretend this is an external service

    def is_user_active(self, user_id):
        if user_id not in self.last_active:
            return False
        time_delta = datetime.datetime.now() - self.last_active[user_id]
        return time_delta < datetime.timedelta(days = 30)

    def record_activity(self, user_id):
        self.last_active[user_id] = datetime.datetime.now()

class UserAccount:
    def __init__(self, user_id, loyalty_program):
        self.user_id = user_id
        self.points = 0
        self.loyalty_program = loyalty_program

    def add_loyalty_points(self, points):
       if not self.loyalty_program.is_user_active(self.user_id):
         raise InactiveUserError("User not active in last month.")
       self.points += points

class InactiveUserError(Exception):
  pass

# Usage
loyalty = LoyaltyProgram()
user = UserAccount("user123", loyalty)

loyalty.record_activity("user123")
user.add_loyalty_points(100)
print(f"User points: {user.points}")

try:
    user2 = UserAccount("user456", loyalty)
    user2.add_loyalty_points(200) # Raises InactiveUserError
except InactiveUserError as e:
    print(f"Error: {e}")

```
*Commentary:* Here, the `UserAccount` aggregate enforces a constraint based on the result from `LoyaltyProgram`. The `UserAccount` aggregate itself does not store the last active date. This constraint is indirect, relying on an external service. However, it is *still* enforced as part of the `add_loyalty_points` operation. It emphasizes that even checks on external conditions should be part of the aggregate's operations, maintaining its consistency and responsibility. This pattern is particularly important for ensuring data validity even when reliant on external factors.

The above examples illustrate varied scenarios of how aggregates interact with write model constraints, reinforcing the notion of aggregates as self-contained units responsible for maintaining their own validity through business rule enforcement. It is essential to keep these constraints simple and directly related to the aggregate’s responsibility. Overly complex constraints or constraints involving many external calls should be examined as potential design smells or opportunities to restructure aggregate boundaries.

Further resources to gain a deeper understanding of aggregates, their boundaries, and consistency constraints are Domain-Driven Design by Eric Evans, Implementing Domain-Driven Design by Vaughn Vernon, and Patterns of Enterprise Application Architecture by Martin Fowler. These texts provide the theoretical foundations and practical guidance needed to build robust domain models where aggregates are respected as the primary unit of transaction and consistency, ensuring the integrity of write operations.  These sources detail the importance of clear aggregate boundaries,  the impact of transaction scopes, and the critical role that constraints play in guaranteeing valid states.
