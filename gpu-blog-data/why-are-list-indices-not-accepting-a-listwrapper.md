---
title: "Why are list indices not accepting a ListWrapper?"
date: "2025-01-30"
id: "why-are-list-indices-not-accepting-a-listwrapper"
---
List indices in Python, unlike languages that might allow arbitrary objects for indexing, rely on the integer type for direct memory access within the underlying data structure. I've repeatedly encountered this while optimizing large data processing pipelines, where I attempted to use a custom wrapper object, a `ListWrapper`, to index standard Python lists, only to receive a `TypeError`.

The reason a `ListWrapper`, or any custom class for that matter, cannot be directly employed as a list index boils down to Python's design and implementation of its sequence types, particularly lists. Lists, in their essence, are dynamic arrays stored contiguously in memory. Accessing an element at a specific index, like `my_list[5]`, involves directly calculating the memory address where the fifth element (starting from 0) is located. This calculation is predicated on knowing the fixed size of each list item and starting memory address of the entire list. To make that calculation fast and efficient, the index must be an integer. When you provide something that's *not* an integer, Python raises a `TypeError` because it cannot directly perform this address calculation.

The index access is routed through the `__getitem__` method of the list object, but even this method internally requires a valid integer index. While Python's dynamic nature permits overriding `__getitem__` for custom classes, the `list` class itself is implemented in C and expects integer indices. Trying to pass a `ListWrapper` instance directly bypasses the standard integer indexing pathway and cannot be interpreted correctly as a memory offset. The standard implementation simply does not handle non-integer indexing. This design principle is fundamental to achieving fast element access in lists. Allowing arbitrary objects for indexing would require significant overhead with complicated mapping lookups and type checks, defeating the purpose of fast sequential list access. Instead, Python forces users to convert those complex lookup patterns to integers.

Consider, for example, the need to have a list that can be indexed via user-defined keys. The correct approach is to map these keys to integers through a dictionary. That way, accessing a list is a fast integer-based lookup, and user logic is delegated to external classes. We can provide the wrapper with some intelligence to translate more complex data access patterns to list indexes, but that needs to be implemented separately, not through the indexing mechanism of a list.

Let's look at some concrete code examples to illustrate this.

**Example 1: Basic `TypeError`**

This example demonstrates the fundamental problem: attempting to use a custom class for indexing.

```python
class ListWrapper:
    def __init__(self, value):
        self.value = value

my_list = [10, 20, 30, 40, 50]
wrapper = ListWrapper(2)

try:
    element = my_list[wrapper] # Attempt to access list with ListWrapper
except TypeError as e:
    print(f"Error: {e}")  # Expected output: Error: list indices must be integers or slices, not ListWrapper
```

In this code, we define a `ListWrapper` class and initialize it with a value. We attempt to use an instance of `ListWrapper` to index `my_list`. This results in a `TypeError` with the message "list indices must be integers or slices, not ListWrapper". This error confirms that the `list` object is rejecting our attempt to use a non-integer object as an index. The list implementation explicitly checks for integer indices before proceeding to the memory lookup.

**Example 2: Implementing `__int__` for implicit casting**

This example introduces the `__int__` method to implicitly cast the `ListWrapper` object into an integer, which allows the list to accept it as index.

```python
class ListWrapper:
    def __init__(self, value):
        self.value = value
    def __int__(self):
      return self.value

my_list = [10, 20, 30, 40, 50]
wrapper = ListWrapper(2)

try:
    element = my_list[wrapper]
    print(f"Element: {element}") # Output: Element: 30
except TypeError as e:
    print(f"Error: {e}")
```

By implementing the `__int__` method within the `ListWrapper`, we provide a way for Python to implicitly convert our custom class object into an integer. When we now use `wrapper` as a list index, Python invokes this method first, retrieving the integer value and then using it to access the list as a standard integer index would, meaning our wrapper object behaves like a standard index. This implicit casting is what allows our `ListWrapper` to be indirectly accepted as an integer by the standard list implementation. The `__int__` method has allowed us to bypass the error, but not because it is using the `ListWrapper` directly, but because it’s now using an integer.

**Example 3: Using a method for Index Retrieval**

Here we show how to correctly use `ListWrapper` as an intermediate object to access list elements.

```python
class ListWrapper:
    def __init__(self, value):
        self.value = value
    def get_index(self):
      return self.value

my_list = [10, 20, 30, 40, 50]
wrapper = ListWrapper(2)

try:
    element = my_list[wrapper.get_index()] # Access via an index retrieval method
    print(f"Element: {element}") # Output: Element: 30
except TypeError as e:
    print(f"Error: {e}")
```

This approach demonstrates that rather than trying to make the `ListWrapper` directly work as an index, we can use it to *retrieve* the integer index that is then used to access the list. Instead of relying on implicit casting, we now have a clear and explicit `get_index()` method. This separates the index logic from the direct usage of the class. The `my_list` is still getting an integer index, which is compliant with Python's design principles, and the custom object is now acting as a data store with specific access logic. This is the correct pattern when wanting to use intermediary objects to access lists.

To deepen understanding of these concepts, I recommend reviewing the official Python documentation on sequence types and data models, specifically focusing on the `__getitem__` method and the implicit type conversion rules. Additionally, exploring resources that explain Python's C implementation of lists can provide further insight into the underlying memory structures. Discussions on the design of Python's data structures in various books and online articles dedicated to Python's core design principles can provide a broader understanding of why certain choices were made.
