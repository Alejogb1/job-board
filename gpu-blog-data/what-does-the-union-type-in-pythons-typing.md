---
title: "What does the `Union` type in Python's `typing` module do?"
date: "2025-01-30"
id: "what-does-the-union-type-in-pythons-typing"
---
Type hinting within Python, particularly using the `typing` module, becomes essential when developing larger, more complex applications. Specifically, the `Union` type allows a developer to denote that a variable, function parameter, or function return value can accept or be of one of several different types. This addresses situations where data might legitimately be of varying types, which Python's dynamic nature often permits, but this flexibility can introduce ambiguity and hinder static analysis.

Before the introduction of `Union` and related constructs, one common way to document such situations was through docstrings, detailing that a function, for example, could accept an integer or a string. However, these were essentially comments, not enforceable at runtime, and therefore of limited use to static analysis tools like MyPy. The `Union` type bridges this gap, providing a way to formally specify acceptable types, permitting both the flexibility of dynamic typing while providing the advantages of static type checking at development time.

Here’s how I’ve used it in practice. During the development of a data processing library, a particular function required handling either a single file path (string) or a list of file paths (list of strings). Initial attempts without using `Union` were problematic: static analysis consistently flagged potential type errors, and the code felt fragile. This is a typical scenario where using `Union` clarifies the expected input significantly.

The basic syntax for a union type uses `Union[Type1, Type2, ..., TypeN]`. The order of types does not generally matter unless you are using the union within a context where a more specific type should be preferred if a value matches multiple types, a use-case addressed later. It indicates that the entity being typed can legitimately hold values of *any* of the specified types.

Let's illustrate with code. Consider a function designed to accept either a single string representing a user ID or a numerical ID. Without type hinting, the function's behavior isn't immediately clear from its signature.

```python
def process_id(id_value):
    if isinstance(id_value, str):
        print(f"Processing user ID: {id_value}")
    elif isinstance(id_value, int):
        print(f"Processing numerical ID: {id_value}")
    else:
        print("Invalid ID type.")

process_id("user123") # Output: Processing user ID: user123
process_id(456)      # Output: Processing numerical ID: 456
process_id(1.23)     # Output: Invalid ID type.
```

This code, while functional, does not leverage the benefits of the `typing` module. If, using a tool like MyPy to perform static analysis, the types are not explicitly given, errors or warnings will be triggered. Refactor this same function, using `Union`:

```python
from typing import Union

def process_id(id_value: Union[str, int]):
    if isinstance(id_value, str):
        print(f"Processing user ID: {id_value}")
    elif isinstance(id_value, int):
        print(f"Processing numerical ID: {id_value}")
    else:
        print("Invalid ID type.")


process_id("user123")
process_id(456)
# MyPy would catch the following error as it is not a union of int or str
# process_id(1.23)
```

By adding `id_value: Union[str, int]`, the function signature now clearly indicates the expected type, and static analysis tools can verify that only strings or integers are passed. If you try to pass an incompatible type, like a float, a type error will be detected by tools such as MyPy *before* runtime. This makes errors more visible during development.

`Union` is also often useful when handling return types. For example, a function might return a result value, or, in the case of an error, return `None`. Consider a function that attempts to fetch user data, returning a dictionary containing the data if successful and None otherwise:

```python
from typing import Union, Dict, Optional

def fetch_user_data(user_id: str) -> Union[Dict, None]:
    # Placeholder: Simulate database fetch operation
    if user_id == "valid_user":
       return {"id": "valid_user", "name": "Jane Doe"}
    else:
        return None


user_data = fetch_user_data("valid_user")
if user_data:
   print(f"User data: {user_data['name']}")

invalid_user_data = fetch_user_data("invalid")
if invalid_user_data is None:
   print("User not found.")

# A more concise and readable way to type None is Optional
def fetch_user_data_optional(user_id: str) -> Optional[Dict]:
    # Placeholder: Simulate database fetch operation
    if user_id == "valid_user":
       return {"id": "valid_user", "name": "Jane Doe"}
    else:
        return None
```

The original function signature `-> Union[Dict, None]` uses `Union`, stating that the return value could be a dictionary or `None`. While correct, there is syntactic sugar for this specific case. As `None` is a common return type for functions that might fail or not find data, the `typing` module provides `Optional[T]`, which is equivalent to `Union[T, None]`. Thus, `-> Optional[Dict]` is semantically equivalent to `-> Union[Dict, None]` and improves readability.

In practice, when defining unions, it's generally better to be more specific rather than allowing for generic types such as `Any` unless absolutely necessary. For example, consider a function dealing with configurations, where certain configuration values might be a string, or a boolean, or an integer. While `Union[str, bool, int]` would work, you can often be more precise. Perhaps a configuration value which is a string should be the filename, where the filename should have an extension, or the boolean must be associated with a flag such as "enable" and "disable." Defining a new type using `TypeAlias` ( introduced in Python 3.10 and available using backports) to specify `Filename = str` or defining a class to specify a `Flag` can give more context and improve readability of the code. Specificity leads to clearer, more maintainable, and less error-prone code.

The `typing` module contains more advanced constructs, but `Union` often forms the basis for many other type hints, including generics and more complex type structures. As the need for robust typing increases, especially when collaborating on large projects, familiarity with `Union` and the wider `typing` module is crucial.

To further solidify your understanding, I recommend consulting the official Python documentation for the `typing` module. Additionally, exploring resources that delve into static analysis with MyPy and other similar tools can help demonstrate the tangible benefits of employing type hints. Furthermore, examining open-source projects that utilize type hinting can provide real-world examples of how these features can improve code quality. While external websites offer tutorials and examples, initially focusing on the core documentation and well-structured books or articles is often the best approach. These resources provide the theoretical foundation which can then be applied to practice.
