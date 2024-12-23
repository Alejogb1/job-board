---
title: "How do ubiquitous and published languages differ in Domain-Driven Design?"
date: "2024-12-23"
id: "how-do-ubiquitous-and-published-languages-differ-in-domain-driven-design"
---

Let's explore this. I’ve encountered the complexities of domain-driven design (DDD) in several projects, specifically those where the gap between technical implementation and business understanding was a constant challenge. The nuances of ubiquitous language versus published language are critical, and failing to properly distinguish between them can lead to substantial rework and communication breakdowns. It's a topic I've spent quite a bit of time untangling, and hopefully, this will provide some clarity.

Essentially, the ubiquitous language, in the context of DDD, is the shared language developed by the *entire* team, both technical and business, to discuss and understand the domain model. This language is not merely a set of terms; it embodies the conceptual understanding and rules that govern the domain. It’s a living, breathing vocabulary that evolves as the team's comprehension deepens. When I’m working, I think of it as the code’s vocabulary, the names of classes, methods, and even comments — all directly reflect the concepts of the business. The goal is complete clarity; if a business expert uses a term, the developers understand exactly what they mean and vice-versa. When this clicks, coding gets exponentially smoother, because ambiguity vanishes. I’ve seen teams where a term had different implications in business meetings compared to design discussions, and the cost of translation and misinterpretation was crippling.

Now, published language, on the other hand, is the formally documented language used in official communication, requirements documents, and often in contractual agreements. It is generally more precise and less flexible than ubiquitous language. It tends to be more static and is often used to define the scope of the system, the deliverables, or the business rules. It is not necessarily a working language, in the same way the ubiquitous language is. The published language is for communication outside the core team to other stakeholders. The focus here is on precision, clarity for external consumption, and often legal implications.

The critical difference isn’t just that one is spoken and the other is written. The difference is in their purpose and their flexibility. The ubiquitous language is a conversational tool for refining shared understanding, constantly shaped by daily interactions. The published language is a more formal, less mutable set of terms created for broader external use and formal specifications.

Let’s illustrate this with a few code examples and how this separation is handled. Imagine a team building a system for a shipping company.

**Example 1: Simple Shipping Calculation**

Let’s say we're dealing with calculating shipping costs. Initially, during a discussion, the business might refer to a “parcel” as a general term, not distinguishing between different sizes or types. The initial code (using a fictional language close to Python) might reflect this:

```python
class Parcel:
    def __init__(self, weight, distance):
        self.weight = weight
        self.distance = distance

    def calculate_shipping_cost(self):
        return self.weight * 0.1 + self.distance * 0.05 # a simplistic calculation
```

However, as the team dives deeper, the business clarifies that different parcel *types* (e.g., "small parcel," "large parcel", or "pallet") are subject to different cost models. This new understanding directly informs the ubiquitous language. We're not just talking about “parcels” anymore; we have *specific types* of parcels with different characteristics.

The updated code, reflecting the evolved ubiquitous language, looks like this:

```python
class SmallParcel:
    def __init__(self, weight, distance):
        self.weight = weight
        self.distance = distance

    def calculate_shipping_cost(self):
        return self.weight * 0.1 + self.distance * 0.05

class LargeParcel:
    def __init__(self, weight, distance, volume):
        self.weight = weight
        self.distance = distance
        self.volume = volume

    def calculate_shipping_cost(self):
      return self.weight * 0.2 + self.distance * 0.1 + self.volume * 0.01
```

Here, the ubiquitous language evolved to include `SmallParcel` and `LargeParcel`, and so the code adapted. In the published language (e.g., a functional specification), we might have "All parcels with dimensions under 20x20x20cm are classified as 'Small Parcel,' anything above is considered 'Large Parcel'". This is the formal published definition; however, our code and daily discussions revolve around the actual domain objects: `SmallParcel` and `LargeParcel`.

**Example 2: Order Fulfillment**

Consider another scenario where the business uses the term “order status”. The initial conversations might reveal a very simplistic view where there are only two states: ‘Pending’ and ‘Shipped.’ Our initial code might look something like this:

```python
class Order:
  def __init__(self, order_id, items):
    self.order_id = order_id
    self.items = items
    self.status = "Pending"

  def mark_shipped(self):
    self.status = "Shipped"
```

However, further discussions with the operations team reveal more complexity. In reality, the fulfillment process has states such as “Processing”, “Ready to Ship”, "In Transit", "Delivered". These are not simply details; they’re essential states within the business process. Our ubiquitous language now expands to include these, directly influencing the code:

```python
class Order:
    def __init__(self, order_id, items):
      self.order_id = order_id
      self.items = items
      self.status = "Pending" # Initial state

    def process_order(self):
        self.status = "Processing"

    def mark_ready_to_ship(self):
        self.status = "Ready to Ship"

    def mark_in_transit(self):
        self.status = "In Transit"

    def mark_delivered(self):
        self.status = "Delivered"

```

The published documentation may still summarize this as "Order status transitions through a series of steps from initiation to delivery," but that is too high level for actual design and development. Our code (ubiquitous language) incorporates every discrete step, each being a meaningful part of the business flow.

**Example 3: Customer Address Validation**

Initially, address validation might just involve checking for a non-empty street field. The initial code in our project may reflect this very simple check:

```python
class Address:
    def __init__(self, street, city, postal_code):
        self.street = street
        self.city = city
        self.postal_code = postal_code

    def is_valid(self):
        return len(self.street) > 0
```

However, in the real world, address validation can involve complex rules such as country-specific formats, postal code validations, and even geo-location checks. So, during our domain discussions, we discover that a generic ‘address’ is insufficient. Instead, it requires a structured address specific to different countries. Our code and our language need to reflect that. The ubiquitous language now includes concepts like `USAddress`, `UKAddress`, each with its own validation rules:

```python
class USAddress:
    def __init__(self, street, city, state, postal_code):
      self.street = street
      self.city = city
      self.state = state
      self.postal_code = postal_code

    def is_valid(self):
        # More complex US-specific validation
        return len(self.street) > 0 and len(self.postal_code) == 5 # Simplified

class UKAddress:
    def __init__(self, street, city, postal_code):
      self.street = street
      self.city = city
      self.postal_code = postal_code

    def is_valid(self):
         # More complex UK-specific validation
         return len(self.street) > 0 and len(self.postal_code) > 2 # Simplified

```

The published requirements might contain statements such as “Address validation must comply with local postal guidelines,” but the actual code now deals with `USAddress` and `UKAddress` instances, a more specific reflection of the language and understanding we developed.

In summary, the ubiquitous language is the *dynamic* language used inside the team, while the published language is *static* and for broader audiences. It’s vital to understand that the code should always reflect the ubiquitous language – that’s our goal. We continually shape the code and our shared understanding together. Confusing or conflating the two will lead to systems that may satisfy published requirements but don’t necessarily capture the real intent and understanding of those actually working within the domain.

For further exploration of these concepts, I recommend reviewing Eric Evans's “Domain-Driven Design: Tackling Complexity in the Heart of Software”. It is foundational to understanding the importance of language. Also, "Implementing Domain-Driven Design" by Vaughn Vernon is very good. Another useful resource would be the various writings by Martin Fowler on domain-specific languages and modeling. Lastly, make sure to also spend time with the architectural patterns explored in “Patterns of Enterprise Application Architecture,” also by Martin Fowler, as they often impact how we design our classes to capture this language. Each of these resources can offer additional context and practical advice on navigating these complexities. These have been indispensable resources to me over the years.
