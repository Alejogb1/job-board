---
title: "How can I create sub-dictionaries from a Python dictionary?"
date: "2025-01-30"
id: "how-can-i-create-sub-dictionaries-from-a-python"
---
Dictionary manipulation is a frequent task in data processing; specifically, the need to create sub-dictionaries from an existing dictionary arises often when structuring data for further analysis or API interactions. My experience over the last decade working on systems that heavily rely on data transformations has shown the need for flexible sub-dictionary generation, and there are a few efficient approaches to accomplish this.

The fundamental principle involves extracting key-value pairs from a parent dictionary based on specific criteria, then using those pairs to construct the sub-dictionary. This process might involve selecting keys explicitly, utilizing key patterns, or employing value-based filtering. The choice of technique depends on the context of the data and the desired structure of the resulting sub-dictionaries.

Let us examine a scenario. I was working on an e-commerce backend where product data was initially stored in a flat dictionary, such as `{'product_id': 123, 'name': 'Laptop X', 'price': 1200, 'category': 'Electronics', 'manufacturer': 'TechCorp', 'description': 'A powerful laptop...', 'stock_level': 50}`. However, for different API endpoints, we required this data to be organized into sub-dictionaries, such as one for product details and another for stock management. Here are some methods I successfully used:

**1. Explicit Key Selection:**

This technique is straightforward and well-suited when specific keys need to be grouped into a sub-dictionary. I often used this when preparing data for distinct microservices. This involves creating the sub-dictionary by assigning values corresponding to pre-defined keys from the parent dictionary. The code demonstrates this approach:

```python
def create_sub_dict_explicit(parent_dict, keys):
    """Creates a sub-dictionary using explicitly specified keys."""
    sub_dict = {}
    for key in keys:
        if key in parent_dict:
            sub_dict[key] = parent_dict[key]
    return sub_dict

product_data = {'product_id': 123, 'name': 'Laptop X', 'price': 1200, 'category': 'Electronics',
                'manufacturer': 'TechCorp', 'description': 'A powerful laptop...', 'stock_level': 50}

product_details_keys = ['product_id', 'name', 'price', 'category', 'manufacturer']
product_details = create_sub_dict_explicit(product_data, product_details_keys)

stock_management_keys = ['product_id', 'stock_level']
stock_management = create_sub_dict_explicit(product_data, stock_management_keys)

print("Product Details:", product_details)
print("Stock Management:", stock_management)
```

In this example, the `create_sub_dict_explicit` function iterates through the specified keys. If the key exists in the `parent_dict`, the corresponding key-value pair is added to the `sub_dict`. We then apply this function to `product_data`, creating the `product_details` and `stock_management` sub-dictionaries using pre-defined lists of keys. This provides precise control over the sub-dictionary's composition. The inclusion of the check `if key in parent_dict:` is crucial, as it prevents `KeyError` exceptions if a specific key is missing from the source data, ensuring robustness, an important consideration in production systems.

**2. Filtering Based on Key Patterns:**

When dealing with dictionaries containing numerous similar keys (for instance, keys with a common prefix), explicit key selection can become cumbersome. I have encountered such cases in configurations where parameters are named according to a convention. In these scenarios, using regular expressions or string pattern matching proves more efficient. The code shows pattern-based filtering in action:

```python
import re

def create_sub_dict_pattern(parent_dict, key_pattern):
    """Creates a sub-dictionary using key pattern matching."""
    sub_dict = {}
    for key, value in parent_dict.items():
        if re.match(key_pattern, key):
            sub_dict[key] = value
    return sub_dict

sensor_data = {'sensor1_temp': 25.5, 'sensor1_humidity': 60.2, 'sensor2_temp': 26.0, 'sensor2_humidity': 61.5, 'sensor1_status': 'OK', 'location': 'Lab'}

sensor1_pattern = r'^sensor1_.*'
sensor1_data = create_sub_dict_pattern(sensor_data, sensor1_pattern)

sensor2_pattern = r'^sensor2_.*'
sensor2_data = create_sub_dict_pattern(sensor_data, sensor2_pattern)

print("Sensor 1 Data:", sensor1_data)
print("Sensor 2 Data:", sensor2_data)
```

The `create_sub_dict_pattern` function utilizes the `re.match` method from the `re` module to compare each key against the given regular expression pattern. In this specific case, the pattern `r'^sensor1_.*'` selects keys starting with `sensor1_` followed by any characters. This approach makes creating sub-dictionaries for sensor readings quite efficient without explicitly listing every sensor key, and it's a pattern I find myself using fairly often. The use of the raw string `r` avoids backslash interpretation issues with regular expressions.

**3. Filtering Based on Value Conditions:**

In certain instances, the creation of sub-dictionaries relies not on keys but on values. Consider log files, where you want to filter entries based on severity or status codes. I have needed this when creating alerts for error conditions. Here’s a method to achieve this:

```python
def create_sub_dict_value_filter(parent_dict, condition_func):
    """Creates a sub-dictionary based on value conditions."""
    sub_dict = {}
    for key, value in parent_dict.items():
        if condition_func(value):
            sub_dict[key] = value
    return sub_dict


log_data = {'log1': 'INFO: Server started', 'log2': 'ERROR: Database connection failed', 'log3': 'WARNING: Disk space low',
             'log4': 'INFO: User logged in', 'log5': 'ERROR: Invalid user credentials'}


def error_condition(value):
   return 'ERROR' in value

error_logs = create_sub_dict_value_filter(log_data, error_condition)


def warning_condition(value):
   return 'WARNING' in value

warning_logs = create_sub_dict_value_filter(log_data, warning_condition)

print("Error Logs:", error_logs)
print("Warning Logs:", warning_logs)
```

The `create_sub_dict_value_filter` function uses a callable `condition_func` to evaluate each value in the dictionary. It includes key-value pairs in the sub-dictionary only if the `condition_func` returns True. Here, we have defined `error_condition` and `warning_condition` functions which check for the substrings 'ERROR' and 'WARNING' respectively within the log message strings. This approach is extremely useful for categorizing data based on value-based conditions. The use of a function as a condition makes the filtering more extensible than a fixed expression and adaptable to different conditions, and it's quite beneficial in data pipelines.

**Resource Recommendations**

For expanding knowledge in these areas, I recommend consulting the official Python documentation for dictionary operations, string manipulations, and the `re` (regular expression) module. Books such as "Fluent Python" by Luciano Ramalho offer profound insights into Pythonic ways of handling collections, and texts focusing on data structures and algorithms often cover strategies for efficient data manipulation. Numerous online coding platforms and courses also contain lessons on effective Python programming techniques. A good foundation in general programming principles and problem-solving will greatly assist in applying these techniques. It is also advantageous to delve into concepts regarding data transformation and data pipelines for a larger understanding of context within which these operations are employed.
