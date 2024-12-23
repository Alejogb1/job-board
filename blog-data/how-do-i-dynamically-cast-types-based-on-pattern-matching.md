---
title: "How do I dynamically cast types based on pattern matching?"
date: "2024-12-23"
id: "how-do-i-dynamically-cast-types-based-on-pattern-matching"
---

Alright, let's dive into dynamic type casting via pattern matching. This is something that's come up quite a few times in my career, especially when dealing with complex data structures or integrating diverse systems. One particular instance that springs to mind was when I was working on a data ingestion pipeline where the incoming data format could vary considerably depending on the source. We weren’t dealing with simple json every time; often, it was a mix of proprietary formats that evolved rather rapidly. So, the usual static type hinting wasn't going to cut it. We needed a mechanism to inspect the incoming data at runtime and convert it to the correct type before further processing.

The core idea, of course, is to use pattern matching to determine the structure and, based on that, perform the appropriate type conversion. This approach allows for a flexible and robust system that can handle various data shapes. I often see folks stumbling into overly complex conditional statements when dealing with these kinds of scenarios; this is where pattern matching shines, offering a cleaner, more declarative way of achieving the same goal.

The foundational concept here relies on the ability of a language (or a library within a language) to deconstruct structures and evaluate them against predefined patterns. Once a pattern matches, we can extract the relevant parts and use them to construct a new object of the desired type. This usually involves some kind of type conversion logic, which could range from basic casts to more elaborate instantiation procedures.

Now, let's get into some code. I'll be using python for examples, but the general principle applies across many languages with decent pattern-matching capabilities (like scala or haskell). Let’s assume we have to handle incoming data representing either a `Point` object, or a `Rectangle` object or even just a number. Here is how we might approach this.

```python
from dataclasses import dataclass
from typing import Union

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point

def dynamic_cast(data: Union[dict, int]) -> Union[Point, Rectangle, int, None]:
    match data:
        case {"x": int(x), "y": int(y)}:
            return Point(x=x, y=y)
        case {"top_left": {"x": int(tlx), "y": int(tly)}, "bottom_right": {"x": int(brx), "y": int(bry)}}:
            return Rectangle(top_left=Point(x=tlx, y=tly), bottom_right=Point(x=brx, y=bry))
        case int(value):
            return value
        case _:
           return None

# Example usage:
point_data = {"x": 5, "y": 10}
rectangle_data = {"top_left": {"x": 1, "y": 2}, "bottom_right": {"x": 5, "y": 6}}
number_data = 42
invalid_data = "not data I would expect"

print(dynamic_cast(point_data)) # Output: Point(x=5, y=10)
print(dynamic_cast(rectangle_data)) # Output: Rectangle(top_left=Point(x=1, y=2), bottom_right=Point(x=5, y=6))
print(dynamic_cast(number_data)) # Output: 42
print(dynamic_cast(invalid_data)) #Output: None
```

In this first example, we're using Python’s `match` statement, introduced in version 3.10. We are explicitly declaring the shape of a `Point` as a dictionary with keys "x" and "y", both having integer values. Same for rectangle with nested points. The `int(x)` syntax within the pattern does not force a conversion. It checks if the value is an integer; if so, it assigns it to the variable x. This ensures we don't proceed with object instantiation if the data is not structurally sound. If none of the patterns match, we return `None`. This approach allows you to convert from a potentially untyped data (e.g., json) to strongly typed objects. This greatly simplifies downstream usage, as type checkers are happy with specific type now.

Now, let's look at a scenario where we're dealing with potentially nested lists and we want to convert them to tuples, for example when working with numerical arrays of different dimensions, received as lists. This time, we want to convert lists to tuples, ensuring that each element is a float.

```python
from typing import List, Tuple, Union

def list_to_tuple(data: Union[List, float]) -> Union[Tuple, float, None]:
    match data:
        case [float(x), float(y), float(z)]:
            return (x, y, z)
        case [float(x), float(y)]:
            return (x,y)
        case float(value):
            return value
        case _:
            return None


# Example Usage
list_3d = [1.0, 2.0, 3.0]
list_2d = [1.1, 2.2]
float_val = 3.14
invalid_list = ["a", "b", "c"]


print(list_to_tuple(list_3d)) # Output: (1.0, 2.0, 3.0)
print(list_to_tuple(list_2d)) # Output: (1.1, 2.2)
print(list_to_tuple(float_val)) # Output: 3.14
print(list_to_tuple(invalid_list)) # Output: None

```

In this second example, we’re using pattern matching to handle different list lengths. The `float(x)` pattern ensures each list element can be converted to a float. This is particularly useful when dealing with numerical data that might be represented as strings or integers, and you want the certainty that everything is in float format. If the data doesn't conform to our expectations, it returns `None` as a signal to caller that data is not in the shape we expected.

Let's examine a slightly more elaborate scenario. Imagine a function that takes a general "shape" definition, where shapes can be circle or a rectangle, using a dictionary-like syntax.

```python
from typing import Dict, Union
import math

@dataclass
class Circle:
    center_x: float
    center_y: float
    radius: float

@dataclass
class Rectangle:
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float


def parse_shape(shape_data: Dict[str, Union[str, float, dict]]) -> Union[Circle, Rectangle, None]:
    match shape_data:
        case {"type": "circle", "center": {"x": float(cx), "y": float(cy)}, "radius": float(r)}:
            return Circle(center_x=cx, center_y=cy, radius=r)
        case {"type": "rectangle", "top_left": {"x": float(tlx), "y": float(tly)},
                                 "bottom_right": {"x": float(brx), "y": float(bry)}}:
            return Rectangle(top_left_x=tlx, top_left_y=tly, bottom_right_x=brx, bottom_right_y=bry)
        case _:
            return None

def calculate_area(shape: Union[Circle, Rectangle, None]) -> Union[float, None]:
  match shape:
    case Circle(center_x, center_y, radius):
      return math.pi * radius * radius
    case Rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y):
      return abs(bottom_right_x-top_left_x) * abs(bottom_right_y-top_left_y)
    case _:
        return None

# Example usage:
circle_data = {"type": "circle", "center": {"x": 1.0, "y": 2.0}, "radius": 3.0}
rectangle_data = {"type": "rectangle", "top_left": {"x": 1.0, "y": 1.0}, "bottom_right": {"x": 5.0, "y": 5.0}}
invalid_shape = {"type":"triangle", "points": [0,0, 1,1, 2,2]}

circle = parse_shape(circle_data)
rectangle = parse_shape(rectangle_data)
invalid = parse_shape(invalid_shape)

print(f'Area of circle:{calculate_area(circle)}') # Output: 28.274333882308138
print(f'Area of rectangle:{calculate_area(rectangle)}') # Output: 16.0
print(f'Area of invalid shape:{calculate_area(invalid)}') # Output: None
```

Here, we’re not only converting to new types but are also using nested patterns to extract the shape-specific data. Moreover, in `calculate_area`, we are matching the shape type to perform specific computations. As you can observe, the process becomes very straightforward and easy to understand and maintain.

When implementing dynamic type casting like this, remember to handle edge cases gracefully. Logging and error handling are paramount, particularly in production scenarios where you might not have control over the format of incoming data.

For a deeper understanding of pattern matching, I would suggest diving into papers on functional programming concepts. Specifically, looking into research on algebraic data types (ADTs) and their implementation in various languages is very helpful. "Programming in Haskell" by Graham Hutton provides a rigorous treatment of ADTs, while "Structure and Interpretation of Computer Programs" by Abelson and Sussman covers foundational computer science concepts, including data abstraction. The documentation for languages like OCaml, Haskell, and Scala also provide a great source of knowledge on pattern matching. I found papers discussing ML’s pattern matching particularly illuminating.

Finally, a good grasp of type systems in general is also useful, a book such as “Types and Programming Languages” by Benjamin C. Pierce is a great starting point.

Dynamic type casting through pattern matching can be a very powerful tool when employed judiciously. It offers flexibility and clarity that procedural approaches often lack. However, avoid using it when there are good static typing solutions available or to mask poor design decisions. Use it when dealing with truly dynamic systems where schema is not known beforehand.
