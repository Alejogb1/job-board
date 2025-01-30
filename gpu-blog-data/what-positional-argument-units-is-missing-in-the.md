---
title: "What positional argument 'units' is missing in the __init__() method causing a TypeError?"
date: "2025-01-30"
id: "what-positional-argument-units-is-missing-in-the"
---
The `TypeError` arising from a missing positional argument 'units' in the `__init__()` method usually indicates a mismatch between the constructor’s expected parameters and the arguments provided during object instantiation. This occurs most often in custom classes when the `__init__` method, which is Python’s constructor, is designed to receive specific parameters that are then not supplied, or supplied incorrectly, when a new object is created. I’ve encountered this frequently when developing simulations and data processing pipelines that make heavy use of user-defined classes, specifically those representing physical quantities that need associated units.

The core issue is that Python, being dynamically typed, relies on the `__init__` method’s signature to know what values to expect during object creation. If this signature mandates a parameter named `units`, but that parameter is absent in the actual instantiation, a `TypeError` is raised. Essentially, the interpreter says "I was expecting an argument called `units`, but I didn't receive it," resulting in the program's crash. Let’s break this down. The `__init__` method defines the initial state of an object; all required state-related attributes should be passed as arguments. When a new object is created, the arguments supplied to the class constructor call are forwarded to the `__init__` method.

Consider a scenario where you’re defining a custom class called `Measurement` to represent physical measurements with an associated unit, common when dealing with scientific or engineering data. This class might store the numerical value of a measurement and its unit. The constructor (`__init__`) should, ideally, accept these two values: the value itself and its unit. The most straightforward method to fix the error is to either supply the `units` argument when creating an instance of the `Measurement` class, or re-engineer your class design to accommodate defaults or less strict initialization if appropriate to the design.

Here’s a basic example demonstrating the error and subsequent fixes:

**Example 1: Demonstrating the TypeError**

```python
class Measurement:
    def __init__(self, value, units):
        self.value = value
        self.units = units

# Intended to create a Measurement object, but throws an error
try:
  measurement_a = Measurement(10)
except TypeError as e:
  print(f"Error: {e}")
```
The above code snippet illustrates the root of the problem. The `Measurement` class's `__init__` method is defined with two parameters, `value` and `units`. However, the instantiation attempt provides only the `value` parameter, resulting in a `TypeError`. The error message will indicate that one required positional argument, `units`, is missing. The `try-except` block here catches the error, which allows the program to proceed without outright failure, printing the specific error message to the console. In this instance, `Error: __init__() missing 1 required positional argument: 'units'` would be outputted.

**Example 2: Correcting the TypeError by Supplying the Missing Argument**

```python
class Measurement:
    def __init__(self, value, units):
        self.value = value
        self.units = units


measurement_b = Measurement(10, "meters")
print(f"Value: {measurement_b.value}, Units: {measurement_b.units}")

```

This corrected example shows the most direct solution: when instantiating the object, we supply both the `value` (10) and the `units` (“meters”), satisfying the `__init__` method’s required parameters. The program now successfully creates the `Measurement` object, initializes its state, and the print statement confirms the expected output.

**Example 3: Correcting the TypeError with Default Argument**

```python
class Measurement:
    def __init__(self, value, units=""):
        self.value = value
        self.units = units


measurement_c = Measurement(10)
print(f"Value: {measurement_c.value}, Units: {measurement_c.units}")

measurement_d = Measurement(15, "kilograms")
print(f"Value: {measurement_d.value}, Units: {measurement_d.units}")
```

In the third example, the code has been modified to provide a default value for the units parameter. This example implements a less stringent approach by setting a default value for units in the class’s constructor. In this scenario, the `units` argument is now optional; if a unit is not explicitly specified when a `Measurement` object is created, the units will default to an empty string, as seen in the instantiation of `measurement_c`. If units are supplied as with `measurement_d`, the specified units are properly assigned. This strategy offers flexibility in cases where units may not always be immediately available, while still enabling explicit unit specification when needed.

These examples demonstrate the crucial role of matching the instantiation arguments with the `__init__` method's parameters. The resolution depends entirely on the intended usage and design of the class. Sometimes, providing defaults is acceptable, especially if a default unit is standard, or the program can handle uninitialized units as necessary.

Several strategies exist to mitigate such errors, beyond the basic solutions, including:

* **Type Hinting**: Introducing type hints into your `__init__` method’s parameters, e.g., `def __init__(self, value: float, units: str):`, though they don’t prevent the `TypeError` at runtime, improve code clarity and enable static analysis tools to catch such errors before runtime.
* **Keyword Arguments:** Employing keyword arguments in instantiation can make the code more readable and less error-prone, i.e. `Measurement(value=10, units="meters")`.
* **Argument Validation:** Include specific checks in your `__init__` method to validate that argument types are correct, and raising more informative `ValueError` exceptions if validation fails.

To further explore this concept, I recommend examining resources on object-oriented programming in Python, focusing specifically on class constructors. Python's official documentation on classes is an indispensable resource. Additionally, I’d suggest any book that dives into advanced object-oriented design patterns within the context of Python. Exploring how libraries like `pint` or `astropy` handle units can also provide practical examples and insights. Lastly, working through online tutorials covering object creation and initialization helps solidify the concepts. These resources, used in combination with practical implementation, contribute to a more thorough understanding and prevention of `TypeError` related to missing positional arguments in `__init__` methods.
