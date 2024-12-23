---
title: "How does DDD deal with distributed status across domains?"
date: "2024-12-23"
id: "how-does-ddd-deal-with-distributed-status-across-domains"
---

, let's unpack the complexities of handling distributed status in a Domain-Driven Design (DDD) context. This is something I’ve grappled with firsthand in several large-scale projects, and it's where the rubber truly meets the road with DDD. It’s not uncommon to find yourself in a situation where different bounded contexts need to be aware of changes happening in others, without creating tight coupling. Ignoring this can quickly lead to a monolithic mess, defeating the very purpose of DDD.

The fundamental challenge here revolves around the concept of eventual consistency. In a distributed environment, assuming immediate, synchronous consistency across all domains is often impractical and can severely impact performance and availability. Instead, DDD emphasizes designing for systems where updates propagate across domain boundaries eventually, allowing each domain to operate autonomously and handle its specific data consistency needs.

One of the primary techniques for dealing with this is via domain events. When a significant state change occurs within a bounded context, that context publishes a domain event. Other bounded contexts that care about this change subscribe to and process these events. This event-driven architecture is a cornerstone of handling distributed status, promoting loose coupling and asynchronous communication. This means, for example, an order placed in the ‘Sales’ context generates an ‘OrderPlaced’ event. The ‘Inventory’ context, upon receiving this event, reduces its stock accordingly. This mechanism allows the ‘Inventory’ context to maintain its ‘view’ of the ‘Sales’ domain, without the need for direct, synchronous calls.

Let’s look at a simplified Python snippet to illustrate:

```python
import uuid
import datetime
from dataclasses import dataclass

@dataclass
class DomainEvent:
    event_id: uuid.UUID
    event_type: str
    event_time: datetime.datetime
    payload: dict

class EventBus:
    def __init__(self):
        self._handlers = {}

    def subscribe(self, event_type, handler):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def publish(self, event: DomainEvent):
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                handler(event)

# Example Event Implementation
def order_placed_event(order_id):
    return DomainEvent(
        event_id=uuid.uuid4(),
        event_type="OrderPlaced",
        event_time=datetime.datetime.now(),
        payload={"order_id": order_id}
    )

# Example subscriber implementation: Inventory Adjustment
def adjust_inventory(event: DomainEvent):
    print(f"Inventory adjusting based on event: {event.event_type} - Order ID: {event.payload.get('order_id')}")

# Setup the Bus
event_bus = EventBus()
event_bus.subscribe("OrderPlaced", adjust_inventory)

# Publish an Event
placed_order_event = order_placed_event(uuid.uuid4())
event_bus.publish(placed_order_event)
```

This simplified example showcases how a domain event (e.g., `OrderPlaced`) is published and then consumed by a subscriber that handles the change (adjusting inventory). Note the use of `dataclasses` in Python to easily create event instances, and the simple `EventBus` which serves as an intermediary.

However, relying solely on naive pub/sub can lead to issues. What if an event fails to be processed? This is where strategies like message queues and sagas become valuable. A message queue adds a layer of durability and resilience – events are persisted, ensuring they aren't lost during communication issues, which allows systems to retry later. A saga, on the other hand, provides a way to manage long-running transactions that span across multiple contexts. Think of it as a sequence of local transactions, where each transaction updates a specific domain, and then an event is published that triggers the next. A failure in the saga typically involves a rollback to ensure eventual consistency.

Now, let's delve into a basic Saga pattern example, focusing on how we might manage the 'OrderPlacement' saga:

```python
class SagaStep:
    def __init__(self, action, compensate=None):
        self.action = action
        self.compensate = compensate

def process_order(order_id):
    print(f"Processing order {order_id}")
    # Simulate actual processing like validations and checks
    if not order_id:
      raise Exception('Invalid order')


def reserve_inventory(order_id):
    print(f"Reserving inventory for order {order_id}")
    # Simulate reserving inventory
    if not order_id:
        raise Exception('Inventory reservation failed')

def cancel_reservation(order_id):
     print(f"Canceling inventory reservation for order {order_id}")
    # Simulate cancelling the reservation

def dispatch_order(order_id):
    print(f"Dispatching order {order_id}")

order_placement_saga = [
    SagaStep(action=process_order),
    SagaStep(action=reserve_inventory, compensate=cancel_reservation),
    SagaStep(action=dispatch_order)
]

def execute_saga(order_id, saga):
    for step in saga:
        try:
            step.action(order_id)
        except Exception as e:
            print(f"Action failed, initiating compensation: {e}")
            if step.compensate:
                for compensate_step in reversed(saga[:saga.index(step)]):
                   if compensate_step.compensate:
                       compensate_step.compensate(order_id)
            return False
    return True

order_id = uuid.uuid4()

if execute_saga(order_id, order_placement_saga):
     print(f"Saga completed successfully for order: {order_id}")
else:
    print(f"Saga failed for order: {order_id}")

```

Here, the `order_placement_saga` list defines the sequence of actions. If the `reserve_inventory` step fails, then any previous compensatory actions will be run in reverse order to rollback any state changes.

A critical aspect often overlooked is the design of your aggregates. In DDD, aggregates act as consistency boundaries within a domain. To maintain this consistency in a distributed environment, it's essential to carefully consider the size and scope of your aggregates. Ideally, they should be relatively small and focused on a cohesive set of operations, limiting the impact of changes across domains. We avoid large, sprawling aggregates which end up being hard to maintain and require heavy transactions to maintain consistency.

Let’s illustrate this with a scenario where a product catalog aggregates data across multiple sources, but it's built in a domain-driven way. We have a simplified representation of a `Product` aggregate with an embedded `InventoryInformation` object. The key is that these are treated as a single consistency boundary:

```python
from dataclasses import dataclass

@dataclass
class InventoryInformation:
    stock_level: int
    reorder_threshold: int

@dataclass
class Product:
    product_id: str
    name: str
    description: str
    inventory: InventoryInformation

    def update_inventory(self, new_stock_level):
        self.inventory.stock_level = new_stock_level

# Example usage
product = Product(
    product_id="PROD123",
    name="Sample Product",
    description="A test product",
    inventory=InventoryInformation(stock_level=100, reorder_threshold=20)
)

product.update_inventory(90) # All changes within the aggregate

print(f"Product: {product.name} ,Stock Level: {product.inventory.stock_level}")
```

Note that any updates to the inventory must go through the `Product` aggregate root. This is crucial because the aggregate ensures that all changes to the `InventoryInformation` are consistent with the overall state of the `Product`. In a distributed scenario, the persistence of this aggregate might involve communicating with separate databases or services. This communication must ensure transactional consistency within the aggregate's boundary, often achieved by using patterns such as unit-of-work.

Finally, it’s crucial to embrace eventual consistency and design your system accordingly. Avoid striving for transactional consistency across all domains, as this leads to significant complexities. Instead, accept that updates may propagate over time and design your read models to cope with this eventual consistency. Implement compensatory actions for potential failures. Techniques like CQRS (Command Query Responsibility Segregation) can help achieve more robust, scalable systems when coupled with these patterns.

For further exploration, I highly recommend consulting "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans, which provides a comprehensive understanding of DDD principles. “Patterns of Enterprise Application Architecture” by Martin Fowler offers additional insight into architectural patterns, and for advanced event-driven patterns, “Designing Event-Driven Systems” by Ben Stopford will be very valuable. These resources cover both the theoretical and practical aspects and have been invaluable in my own experience tackling such challenges. Understanding the principles in these texts can set a solid foundation for addressing the challenges of distributed state in your projects.
