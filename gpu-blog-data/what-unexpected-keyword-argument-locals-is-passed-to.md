---
title: "What unexpected keyword argument 'locals' is passed to __init__()?"
date: "2025-01-30"
id: "what-unexpected-keyword-argument-locals-is-passed-to"
---
The unexpected keyword argument `locals` passed to `__init__()` within a class definition typically arises from a misunderstanding of how class decorators, particularly those that dynamically generate attributes or methods, interact with the creation process of class instances. Over the years, I've encountered this during framework development when dealing with declarative programming patterns and meta-classes that were heavily modified by decorators. The core issue is that decorators, when wrapping a class definition, can unintentionally pass their local variables into the class's constructor. Let's break down the mechanics of this.

Class decorators in Python are effectively syntax sugar for function application. When a decorator `@decorator_name` appears above a class definition, like:

```python
@my_decorator
class MyClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b
```

it is equivalent to:

```python
class MyClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

MyClass = my_decorator(MyClass)
```

The decorator `my_decorator` receives the class `MyClass` as its argument, performs some operations, and then typically returns the modified class. Now, if this decorator happens to have a local variable named `locals` (which is unfortunately also a built-in function in Python), it will be present in the scope where the wrapped class is constructed. When that wrapped class is later instantiated, the decorator's local scope, including that accidental 'locals' variable, can be forwarded along as keyword arguments during the construction process, leading to the `__init__()` method inadvertently receiving it.

The `__init__` method of a class only expects arguments defined in its signature. Any additional keyword arguments not specified in the `__init__` definition will lead to the `TypeError: __init__() got an unexpected keyword argument 'locals'`. This issue doesn't stem from the class itself, but rather from the surrounding context created by the decorator.

Here are three examples demonstrating this behavior with commentary:

**Example 1: The Basic Case**

```python
def problematic_decorator(cls):
    locals = 123  # Shadowing the built-in locals
    return cls

@problematic_decorator
class MyClass1:
    def __init__(self, x, y):
        self.x = x
        self.y = y

try:
    instance1 = MyClass1(x=1, y=2) # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```

In this basic example, `problematic_decorator` defines a local variable named `locals` which shadows the built-in. Although this is simply a number, when the class `MyClass1` is instantiated, the python interpreter effectively attempts to pass `locals=123` as a keyword argument to `__init__`. Since `__init__` does not accept a `locals` parameter, a `TypeError` is raised.

**Example 2: A Function within a Decorator**

```python
def more_complex_decorator(cls):
    def inner_func():
       locals = { 'foo': 'bar' } # Another local scope
       return cls
    return inner_func()

@more_complex_decorator
class MyClass2:
    def __init__(self, name):
        self.name = name

try:
    instance2 = MyClass2(name="example") # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```

This example showcases that the issue doesn't have to occur directly within the decorator function but can also arise within nested scopes. `more_complex_decorator` returns the result of `inner_func()`, and that's when the class construction is finalized. Since `inner_func` has `locals` as a local variable, it is passed as an unexpected keyword argument to `__init__`.  It highlights that any local scope within the decorator’s call stack can trigger the `TypeError`.

**Example 3: Decorator with Arbitrary Keyword Arguments**

```python
def flexible_decorator(**kwargs):
    def decorator(cls):
        return cls

    return decorator

@flexible_decorator(locals = {"a":1, "b":2})
class MyClass3:
    def __init__(self, data):
        self.data = data


try:
    instance3 = MyClass3(data="hello") # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```
This example demonstrates that even if the decorator uses a `**kwargs` syntax, it can also lead to the same TypeError because the `kwargs` dictionary will be passed along with the class constructor arguments. In `flexible_decorator`, the `locals` key is present within `kwargs` and, like the other examples, becomes an accidental keyword argument during instantiation.

**Mitigation Strategies**

The core solution lies in carefully avoiding the accidental use of `locals` as a local variable name within decorators. Employing better naming conventions will circumvent this problem. Moreover, understanding the execution scope of class decorators and how they interact with class instantiation is fundamental. Another technique, when you absolutely need to collect arbitrary arguments, is to use `*args, **kwargs` parameters and perform careful filtering/validation before passing arguments to the class constructor.

**Resource Recommendations**

For deepening your comprehension of these intricacies, I recommend studying the following areas through their corresponding Python documentation and other educational resources:

*   **Decorators:** Examining the underlying mechanics of decorator application and how they transform function or class definitions. This includes the execution order of decorators and the scope they generate.

*   **Class Creation Process:** Exploring the details of the Python metaclass machinery and how it creates class objects and instances. This aids in understanding what happens "behind the scenes" during class declaration and instantiation.

*   **Scoping and Closures:** Developing a robust understanding of Python's variable scoping rules and how they apply to nested functions and closures. This is essential for grasping how local variables in a decorator's scope affect class instantiation.

*   **Magic Methods:** Examining `__init__` and the other "dunder" methods in detail to understand their invocation semantics and the parameters they accept. It is important to consider which methods will be called during the class creation and instantiation lifecycle.

By focusing on these core concepts, one can effectively diagnose and prevent unexpected keyword arguments like `locals` from causing runtime errors, especially when dealing with complex class decorators.
