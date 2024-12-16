---
title: "How should DDD handle distributed status across domains?"
date: "2024-12-16"
id: "how-should-ddd-handle-distributed-status-across-domains"
---

Okay, let's tackle this. I've been in the trenches with distributed systems for what feels like an eternity, and handling status across domain boundaries within a distributed domain-driven design (ddd) architecture is definitely a recurring challenge, one that I've seen go sideways many times. We're essentially dealing with the inherent tension between domain autonomy and the need for coordination. Let me walk you through how I typically approach this, drawing on my experiences.

The crux of the problem, as i see it, is that we can't let one domain reach into another to directly interrogate its state. That violates the fundamental principle of domain encapsulation. But the reality is that domains often need to react to changes in other domains, which brings us to the crux of distributed status. It's not about having one central authority that dictates the "true" status, but rather, allowing each domain to maintain its own consistent view of the world, often in relation to other domains.

My preferred approach revolves around the concept of *eventual consistency*, achieved through asynchronous communication, typically using message brokers. Specifically, we would utilize domain events to broadcast relevant status changes. This means when a domain undergoes a state transition that might be of interest to others, it emits a domain event. These events are descriptive, immutable records of "something that happened" within that domain. Another domain, subscribed to these events, can then independently update its local data to reflect these changes, according to its own domain logic. Crucially, we are not asking the emitting domain for its current status; rather, we are reacting to its past state changes.

It is absolutely vital to understand that this model does not guarantee instantaneous consistency. there is a propagation delay inherent in this asynchronous mechanism. This is acceptable – in fact, it's desired – in most DDD contexts, particularly when we're dealing with multiple services.

Let's make this concrete. Consider an e-commerce system with two domains: `order` and `inventory`. When an order is placed in the `order` domain, we would *not* directly query the `inventory` domain to check stock. Instead, the `order` domain emits an `orderplaced` event. The `inventory` domain, subscribed to this event, receives it and, based on its logic, updates its stock levels and may emit further events like `stockreserved` or `stockdepleted`. Here is a rough example in a python-like syntax of what this could look like:

```python
# inside the 'order' domain

class Order:
    def __init__(self, order_id, items, ...):
        self.order_id = order_id
        self.items = items
        self.status = "pending"
        # ... other order properties

    def place(self):
        if self.status != "pending":
            raise InvalidOperation("Order already processed")
        self.status = "placed"
        event_bus.publish(OrderPlaced(order_id=self.order_id, items=self.items))

class OrderPlaced:
    def __init__(self, order_id, items):
       self.order_id = order_id
       self.items = items
```
And then in the `inventory` domain

```python
# Inside the 'inventory' domain

class Inventory:
    def __init__(self, product_id, stock_level):
        self.product_id = product_id
        self.stock_level = stock_level

    def reserve_stock(self, quantity):
        if self.stock_level < quantity:
            raise InsufficientStock("Not enough stock")
        self.stock_level -= quantity
        event_bus.publish(StockReserved(product_id = self.product_id, quantity = quantity))

def handle_order_placed(event):
  for item in event.items:
      try:
          inventory = get_inventory_by_product_id(item.product_id)
          inventory.reserve_stock(item.quantity)
      except InsufficientStock:
          # Handle scenario when not enough inventory exists
          # this logic can vary a lot
          event_bus.publish(OrderFailed(order_id=event.order_id, reason='insufficient stock'))

event_bus.subscribe('OrderPlaced', handle_order_placed)

class StockReserved:
    def __init__(self, product_id, quantity):
       self.product_id = product_id
       self.quantity = quantity

class OrderFailed:
    def __init__(self, order_id, reason):
        self.order_id = order_id
        self.reason = reason

```
You can see how the `order` domain doesn't directly know or care about the inventory level. it simply emits an `orderplaced` event. This allows for loose coupling and independent deployments of the two domains. the `inventory` domain, using the handler `handle_order_placed` has received the event and then acts on the data. Crucially note that the `order` domain never directly queries the `inventory` domain; everything happens asynchronously through the event bus.

Another practical example involves a user management domain and a notification service domain. When a user changes their email preference in the user domain, it should not directly trigger an update in the notification service. Instead, the user domain would emit a `useremailpreferenceschanged` event. The notification service would then receive the event and update its internal representation of user preferences, allowing it to send notifications according to the user's settings. Here is a snippet:

```python
# Inside user domain

class User:
    def __init__(self, user_id, email_preferences):
        self.user_id = user_id
        self.email_preferences = email_preferences

    def update_email_preferences(self, new_preferences):
        self.email_preferences = new_preferences
        event_bus.publish(UserEmailPreferencesChanged(user_id=self.user_id, email_preferences=self.email_preferences))

class UserEmailPreferencesChanged:
    def __init__(self, user_id, email_preferences):
        self.user_id = user_id
        self.email_preferences = email_preferences

# Inside notification service
def handle_email_preferences_changed(event):
   #Logic to store preferences locally

event_bus.subscribe('UserEmailPreferencesChanged', handle_email_preferences_changed)
```

We don't want the user domain to be aware of the notification service's specific needs, or even the fact that such a service exists. The separation is clean.

Furthermore, consider a content management system (cms) with two domains: `article` and `publishing`. When an article is published in the `article` domain, it emits an `articlepublished` event. the `publishing` domain receives this event and then creates a copy of the article for public consumption, or it updates its index. We are, yet again, avoiding any sort of direct interrogation.

```python
# inside the article domain
class Article:
  def __init__(self, article_id, content, status="draft"):
    self.article_id = article_id
    self.content = content
    self.status = status

  def publish(self):
      if self.status != "draft":
         raise InvalidOperation("Article not in draft status")
      self.status = "published"
      event_bus.publish(ArticlePublished(article_id=self.article_id, content=self.content))


class ArticlePublished:
    def __init__(self, article_id, content):
        self.article_id = article_id
        self.content = content


# Inside the publishing domain

def handle_article_published(event):
  # Logic to create a published copy of the article
event_bus.subscribe('ArticlePublished', handle_article_published)
```

These examples illustrate a core pattern: domains communicate their status changes via events, not by directly exposing their current state. the consuming domain is then responsible for maintaining its consistency, often by creating a local projection of the data relevant to its domain.

Now, where do you go to dive deeper into this? I strongly recommend reading *domain-driven design: tackling complexity in the heart of software* by eric evans, which lays out all these core principles. For more practical insights into distributed systems patterns, *building microservices* by sam newman is a great resource. *designing data-intensive applications* by martin kleppmann will also provide context on the challenges inherent in distributed architectures and strategies for managing data consistency and integrity.

One important note; there isn't *one* correct way to manage distributed status – context is king. The optimal solution will often depend on the level of consistency required by the business, the complexity of the system, and the trade-offs that teams are comfortable with making. My personal experiences suggest that adopting a pattern of event-driven architecture, along with careful domain boundary definition, is one of the most pragmatic ways to tame the complexities of managing status across domains in a distributed system.
