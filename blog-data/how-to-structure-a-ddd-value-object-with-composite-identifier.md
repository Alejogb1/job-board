---
title: "How to structure a DDD value object with composite identifier?"
date: "2024-12-23"
id: "how-to-structure-a-ddd-value-object-with-composite-identifier"
---

Okay, let's tackle this. The question of structuring a domain-driven design (DDD) value object with a composite identifier is one that I've bumped into more than a few times in my career. I recall one particularly gnarly project involving a distributed inventory system where product identifiers had to accommodate both internal system codes and external vendor SKUs – a perfect storm for composite key considerations. It’s more common than you might think, and it warrants a considered approach.

The core principle here is that a value object should represent a concept in the domain that doesn't have identity over time; instead, its identity is derived entirely from its attributes. When you add a composite identifier into the mix, you're essentially creating a value object whose identity is determined by a combination of multiple fields, each having its own meaning within the system. If those combined fields are not directly related to business meaning as a single concept, but rather act as an identifier for some larger entity, then it might not be a proper value object, but perhaps an entity identifier instead. We need to be very clear on that line.

First, let's define our terms a little better. When I talk about a "composite identifier," I’m referencing a unique key composed of multiple attributes. Imagine a product's identifier not just as a simple integer, but as a combination of, say, a manufacturer code and a product code. Each on its own might be meaningless within your specific domain, but together they form a unique and meaningful identifier. The challenge lies in correctly encapsulating that composite identifier inside your value object, ensuring immutability and proper equality checking.

My personal experience has taught me that treating the composite identifier as a single immutable unit within the value object is critical. Avoid making it mutable. That's a recipe for disaster, leading to unintended consequences with caching, comparisons, and state management across systems. We want to model the immutable concept as close as possible, reflecting domain constraints rather than simply database constraints. Also, don't let the database model guide your domain model too much. Instead, let the domain model guide the database implementation.

Let's move on to practical implementations. I'll showcase these with some basic code examples, leaning towards python, as it provides a clear syntax that keeps things readable.

**Example 1: Basic Composite Identifier Value Object**

Here's a straightforward example where we are defining a `ProductIdentifier` value object with a simple composite identifier combining `manufacturer_code` and `product_code` both of which are strings.

```python
from typing import NamedTuple

class ProductIdentifier(NamedTuple):
    manufacturer_code: str
    product_code: str

    def __str__(self):
      return f"{self.manufacturer_code}-{self.product_code}"

    def __hash__(self):
       return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, ProductIdentifier):
            return False
        return str(self) == str(other)


# example usage
product_id1 = ProductIdentifier("ABC", "123")
product_id2 = ProductIdentifier("ABC", "123")
product_id3 = ProductIdentifier("DEF", "456")

print(f"product_id1: {product_id1}")
print(f"product_id2: {product_id2}")
print(f"product_id3: {product_id3}")
print(f"product_id1 == product_id2 : {product_id1 == product_id2}")
print(f"product_id1 == product_id3 : {product_id1 == product_id3}")
```

In this case, I've used `NamedTuple`, a standard Python feature that automatically takes care of creating the constructor and immutability, as well as the basic equality checking which we override to be based on a string representation. The crucial point is the overridden `__eq__` and `__hash__` methods. These are fundamental for value objects. The `__eq__` method ensures that two value objects with the same composite identifier are considered equal, not just identical instances in memory. The `__hash__` method, which is based on a stable string representation, means that the value object can function as a key in dictionaries and sets correctly. This is critical if you use these value objects in any kind of collection where identity matters.

**Example 2: Composite Identifier with Validation**

Now, let's make it a little more complex. Often, real-world identifiers have constraints. What if, for instance, our manufacturer codes must adhere to a particular pattern?

```python
from typing import NamedTuple
import re

class ProductIdentifier(NamedTuple):
    manufacturer_code: str
    product_code: str

    def __post_init__(self):
      if not re.match(r"^[A-Z]{3}$", self.manufacturer_code):
        raise ValueError("Invalid manufacturer code")

    def __str__(self):
      return f"{self.manufacturer_code}-{self.product_code}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, ProductIdentifier):
            return False
        return str(self) == str(other)


# Example Usage
try:
    product_id4 = ProductIdentifier("ab", "123") # Invalid manufaturer code
except ValueError as e:
    print(f"Error creating product id: {e}")

product_id5 = ProductIdentifier("ABC", "456")
print(f"product_id5: {product_id5}")
```

Here, I've added a validation rule to `__post_init__`.  The pattern `^[A-Z]{3}$` means that manufacturer codes must be three uppercase letters. This prevents creating an invalid object and allows us to implement domain rules at the value object level. This is essential when building robust systems; validate at the boundary.

**Example 3: Composite Identifier with Different Data Types**

Let's say we now have an identifier composed of a manufacturer code, a product code, and a revision number (integer). The composite identifier should combine different data types:

```python
from typing import NamedTuple

class ProductIdentifier(NamedTuple):
    manufacturer_code: str
    product_code: str
    revision_number: int

    def __str__(self):
        return f"{self.manufacturer_code}-{self.product_code}-{self.revision_number}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, ProductIdentifier):
            return False
        return str(self) == str(other)


# Example Usage
product_id6 = ProductIdentifier("ABC", "123", 1)
product_id7 = ProductIdentifier("ABC", "123", 2)

print(f"product_id6: {product_id6}")
print(f"product_id7: {product_id7}")
print(f"product_id6 == product_id7 : {product_id6 == product_id7}")

```

This example demonstrates that a composite identifier isn't limited to only strings. Including a revision number here allows you to track changes, something quite useful in several domains. The critical point still remains: the composite identifier is treated as a single unit within the value object, and the `__str__` method creates a unified representation of the combined fields.

**Further Considerations and Recommended Resources**

When tackling complex systems, relying solely on basic principles isn't enough. I’d highly recommend studying Martin Fowler’s "Patterns of Enterprise Application Architecture" which gives you solid base knowledge on patterns and how to design domain models. Also, Eric Evans’ “Domain-Driven Design” book is foundational for understanding the overall rationale and methodology behind working with DDD. It's also worthwhile looking into "Implementing Domain-Driven Design" by Vaughn Vernon for practical, real-world implementation insights. These resources will enhance your understanding and ability to tackle more complex domain modelling challenges.

You should also think about scenarios where the composite identifier needs to be serialized or deserialized. Having a canonical string representation is helpful, but you might need custom serialization methods to handle complex data formats, especially if your identifier is a combination of nested value objects. Moreover, consider testing your value objects thoroughly, especially regarding equality and hashing, as these form the basis of their correct usage within your domain. And be sure to align your value objects with the ubiquitous language of your domain. It will make the system a lot easier to understand and maintain in the long run.

In summary, structuring a value object with a composite identifier is about maintaining immutability, ensuring proper equality checks, and encapsulating the underlying complexity of a multi-part identity within a cohesive unit. The focus should be on reflecting domain semantics and not merely mimicking database models. This approach leads to more robust, understandable, and maintainable code.
