---
title: "Why does my custom Python class exhibit unexpected behavior with __eq__ and __contains__?"
date: "2025-01-30"
id: "why-does-my-custom-python-class-exhibit-unexpected"
---
The interaction between `__eq__` and `__contains__` in custom Python classes, particularly within data structures like sets or lists where identity and equality are crucial, often manifests unexpectedly when the `__hash__` method is either missing or inconsistent with `__eq__`. I've encountered this in several projects, most notably a complex simulation involving dynamic entity management, where failing to manage this relationship accurately led to subtle, yet significant, logical errors.

At its core, Python's built-in containers rely on two key methods for object handling: `__eq__` (equality check) and `__hash__` (hash generation). When you use the `in` operator, which underpins `__contains__`, or attempt to store instances of your class in hash-based structures like sets or dictionaries, Python uses the hash value of your objects for efficient lookup. If `__eq__` determines two objects are equal, their hash values *must* also be equal. If this rule is violated, or if `__hash__` isn't implemented at all when `__eq__` is, hash-based lookups may return inconsistent or unexpected results. This is the root cause of the 'unexpected behavior' with `__contains__`.

The problem stems from how Python's default implementation of `__hash__`, which is based on object identity (i.e., the memory address), operates if your class lacks a specific `__hash__` method. In cases where you implement `__eq__` to compare based on object content rather than identity, you're essentially asserting that objects may be considered equal even if they are not the same instance in memory. If a hash value is not generated consistently with this notion of equality, Python might consider two objects to be distinct even if `__eq__` says they are equivalent, breaking the contract and leading to anomalies.

Consider, for instance, a simple `Point` class representing a 2D coordinate. We want two `Point` objects with identical `x` and `y` values to be considered equal.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


p1 = Point(1, 2)
p2 = Point(1, 2)

print(p1 == p2)  # Output: True

points_set = {p1}
print(p2 in points_set) # Output: False
```

In the above code, although `p1` and `p2` are considered equal by `__eq__`,  `p2 in points_set` returns `False`.  This is because a `__hash__` method isn’t defined, and the implicit object identity based hashing results in different hash values for `p1` and `p2`. As the hash of `p2` is not equal to that of `p1` (the only element in the set), the `in` check fails. This directly illustrates the disconnect and resulting incorrect `__contains__` behavior.

To fix this, you *must* add a `__hash__` method that’s consistent with your `__eq__` implementation. The safest and generally correct way to do this is to generate a hash based on the attributes considered in `__eq__`. Here's an example of how to resolve the issue with the `Point` class by adding `__hash__`:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


p1 = Point(1, 2)
p2 = Point(1, 2)

print(p1 == p2) # Output: True

points_set = {p1}
print(p2 in points_set) # Output: True
```

By adding `__hash__` where the hash is derived from the same attributes used in `__eq__`, the container now correctly identifies that `p2` is considered an existing member. The hash values for `p1` and `p2` are now equivalent and the set lookup functions as intended.

It is critical to be careful when using mutable objects in hashable contexts. If an object’s attributes change after its hash value has been calculated, it may become irretrievable from the hash-based data structure. A common error pattern is making a mutable property participate in the calculation of the hash value. If the property’s value changes, the hash changes, but the set/dictionary does not re-index using the updated hash. As such the object will no longer be retrievable. To illustrate this issue, consider the following flawed example:

```python
class MutablePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, MutablePoint):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"MutablePoint({self.x}, {self.y})"

mutable_p = MutablePoint(3,4)
mutableset = {mutable_p}

mutable_p.x = 5

print(mutable_p in mutableset) # Output: False
print(mutableset) # Output: {MutablePoint(5, 4)}
```

Here, even though `__eq__` would return `True` if a `MutablePoint(5,4)` were compared to `mutable_p`, because `mutable_p`’s hash has changed since it was added to the set, and the set doesn't update its index, lookups using `in` will fail. This example shows the dangers of including mutable attributes in `__hash__` without additional protections and care. If you must have mutable properties included in equality checks, the underlying container should ideally be a list and equality should be implemented using linear search rather than relying on hash based lookups.

In summary, unexpected `__contains__` behavior often arises due to the relationship between `__eq__` and `__hash__`. If you override `__eq__` to provide custom equality logic, you *must* implement `__hash__` to be consistent; objects that `__eq__` claims are equal *must* produce identical hash values. Furthermore, avoid mutable properties in the hash function to maintain set/dict integrity, and understand the performance implications of not relying on hash based lookups. If your objects are mutable, it may be better to use lists or other container types instead of relying on hashing semantics.

For further information on these topics, I recommend reviewing the official Python documentation, especially regarding object models and data structures.  Additionally, consult sources that detail best practices for implementing equality and hashing, specifically focusing on custom class design and immutability. Resources detailing common pitfalls and error patterns in Python object behavior will be beneficial. Finally, textbooks discussing algorithm design and data structures are indispensable.
