---
title: "How can I combine existing functions to create a new, desired function?"
date: "2025-01-30"
id: "how-can-i-combine-existing-functions-to-create"
---
The cornerstone of effective software development lies not just in crafting individual functions, but in composing them into larger, more complex functionalities. This process, often termed function composition, allows for modularity, reusability, and a reduction in code duplication. My experience over the years has shown that mastering function composition is a critical step in progressing beyond basic scripting to building maintainable and scalable applications.

The core idea behind function composition is to take the output of one function and feed it as the input to another. This approach creates a chain of operations, where each function transforms the data in a specific way. This differs significantly from nested function calls, which can quickly become difficult to read and debug. Instead, function composition emphasizes a more declarative style, making the flow of data and transformations explicit. The goal is to build complex behaviors from simple, well-defined components. This can be applied across programming paradigms but is often associated with functional programming.

When composing functions, consider the signature of each function involved. The output type of the first function in the sequence must be compatible with the input type of the second function, and so on, for all the intermediate steps.  Carefully planning the transformations and intermediate data representations is vital to create a composite function that correctly processes data.

Here are three examples that illustrate the concept:

**Example 1: Basic String Manipulation**

Imagine I have two existing functions, `trim_string` and `capitalize_string`, responsible for cleaning string data. `trim_string` removes leading and trailing whitespace and `capitalize_string` converts a string to title case (first letter of each word capitalized).

```python
def trim_string(input_string: str) -> str:
    """Removes leading and trailing whitespaces from a string."""
    return input_string.strip()

def capitalize_string(input_string: str) -> str:
    """Converts a string to title case."""
    return " ".join(word.capitalize() for word in input_string.split())

def process_string(input_string: str) -> str:
  """Composes trim and capitalize functions."""
  trimmed = trim_string(input_string)
  capitalized = capitalize_string(trimmed)
  return capitalized


#Example usage
dirty_string = "  hello world  "
cleaned_string = process_string(dirty_string)
print(cleaned_string) #Output: Hello World
```

Here, `process_string` embodies function composition. First, the input string is passed to `trim_string`, and its result is directly passed as the argument to `capitalize_string`. This clear flow makes the logic easy to follow. While simple, it demonstrates how individual operations can be combined. It is also important to note that in practice, composing `trim_string` and `capitalize_string` may be more efficient using libraries with optimized implementations of these functionalities. However, this example illustrates the general principle of function composition.

**Example 2: Applying a Function Chain to Lists**

My work often requires processing data stored in lists. Suppose I have functions named `filter_positive` that selects positive numbers from a list, `square_number` that squares a number, and the goal is to square all positive numbers in a list:

```python
from typing import List

def filter_positive(numbers: List[int]) -> List[int]:
    """Returns a new list containing only positive numbers from input."""
    return [number for number in numbers if number > 0]

def square_number(number: int) -> int:
    """Squares a single number."""
    return number * number

def square_positives(numbers: List[int]) -> List[int]:
  """Composes filter and map functions."""
  positive_numbers = filter_positive(numbers)
  squared_numbers = list(map(square_number,positive_numbers))
  return squared_numbers

# Example usage
numbers_list = [-2, 1, 4, -5, 6]
squared_positives = square_positives(numbers_list)
print(squared_positives)  # Output: [1, 16, 36]
```
In `square_positives`, the output of `filter_positive` is used as the input for `map`, which applies `square_number` to each positive number. The `map` function is a classic example of a higher-order function, allowing you to apply an operation to a collection.  Note that `map` returns a map object (an iterator), so we convert it to a list for printing. This illustrates how functions that operate on single elements can be combined with functions operating on collections. It also shows how higher-order functions can be included in the sequence of operations.

**Example 3: Composing functions with an intermediate data structure**

Consider the scenario where data requires multiple transformations involving an intermediate data structure. Let's assume functions named `extract_information`,  `convert_to_dict` and `filter_by_key`: `extract_information` extracts specific fields from a raw text (assume the raw text represents a record) , `convert_to_dict` transforms the extracted fields into a dictionary and `filter_by_key` filters the dictionary based on a given key-value pair.
```python
from typing import List, Dict, Any

def extract_information(raw_text:str) -> List[str]:
  """Parses raw text and returns a list of fields."""
  # Assume that the text is delimited by commas for simplicity
  return raw_text.split(',')


def convert_to_dict(fields:List[str], keys:List[str]) -> Dict[str,str]:
  """Converts a list of fields to a dictionary."""
  if len(fields) != len(keys):
    raise ValueError("Number of fields must match the number of keys")
  return dict(zip(keys,fields))

def filter_by_key(data:Dict[str,str], key:str, value:str) -> Dict[str,str]:
  """Filters dictionary based on a given key-value pair."""
  if data.get(key) == value:
    return data
  else:
    return {}

def process_data(raw_text:str, keys:List[str], filter_key:str, filter_value:str) -> Dict[str,str]:
  """Composes extract, convert and filter functions."""
  fields = extract_information(raw_text)
  record_dict = convert_to_dict(fields, keys)
  filtered_record = filter_by_key(record_dict,filter_key,filter_value)
  return filtered_record

# Example usage
raw_data = "John,30,New York"
keys = ["name","age","city"]
filtered_data = process_data(raw_data,keys, "city", "New York")
print(filtered_data) #Output: {'name': 'John', 'age': '30', 'city': 'New York'}

raw_data_filtered = "John,30,London"
filtered_data_filtered = process_data(raw_data_filtered,keys, "city", "New York")
print(filtered_data_filtered) #Output: {}

```

In `process_data`, the flow is as follows: raw text is extracted to a list, then transformed into a dictionary and finally filtered. The intermediate data structure (the list and then the dictionary) allows to achieve this by passing information from one function to the next.

These examples illustrate how to combine existing functions to achieve complex behaviors. When designing functions, aim for each one to have a singular, well-defined purpose. This increases the flexibility and reusability when composing larger operations. A key advantage of this approach is its testability; each component can be tested independently, making the overall system more robust.

For further exploration of this topic, consider the book "Structure and Interpretation of Computer Programs" which deeply explores functional programming concepts. Similarly, "Clean Code" provides guidance on writing modular and understandable code, closely related to effective function composition. Additionally, studying the principles of functional programming, including concepts like currying and function composition operators (such as the `compose` function often used in functional languages),  will provide additional techniques for function composition.  Lastly, reading through design pattern literature will reveal how function composition is crucial for building robust and modular software.
