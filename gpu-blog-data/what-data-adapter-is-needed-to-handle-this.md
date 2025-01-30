---
title: "What data adapter is needed to handle this input?"
date: "2025-01-30"
id: "what-data-adapter-is-needed-to-handle-this"
---
**Question:** What data adapter is needed to handle this input?

The need for a robust data adapter arises from the common scenario where the format of incoming data does not directly match the expected structure of the consuming application or service.  I've encountered this countless times in the development of large-scale applications, particularly when integrating disparate systems.  Specifically, let's consider situations where external data feeds, API responses, or legacy databases present data in a format that requires transformation before it can be efficiently processed, persisted, or displayed.

The input requiring adaptation isn’t defined here, but based on experience, it frequently presents in one of a few common variations requiring different adapter strategies. These include:

1. **Inconsistent Naming Conventions:** The input might use different key names, casing, or abbreviations than our internal data model. For example, “customer_id” in the incoming data might need to become “customerId” in our application.
2. **Data Type Mismatches:** The input might represent numerical values as strings or dates in a format that requires parsing.
3. **Hierarchical Structure Differences:** The input might represent nested objects in a structure that does not fit our desired data model; often flattening or reshaping such structures is crucial.
4. **Data Transformation Requirements:** The need to extract or combine data from multiple fields, perform validation, or enrichment of data.
5. **Handling Optional Data:**  The input might contain optional fields that require conditional logic or default values.

Given these common issues, a data adapter acts as an intermediary layer, translating the incoming data into a consumable form by implementing mapping, transformation, and validation routines. The precise adapter approach depends on the nature and complexity of the transformation required.

**Explanation of Common Data Adapter Approaches:**

A primary design consideration is determining whether a simple mapping is sufficient or if more intricate logic is necessary. In simpler scenarios, a direct mapping approach using a configuration file or a straightforward conversion function can suffice. I've used this pattern frequently when integrating APIs that adhere to standardized schemas but use different field names than our own data models. For more complex scenarios where the shape, type, or content of the data must be altered significantly, a more structured approach employing techniques of transformation and validation is crucial.

In cases where input data is inconsistently named or structured, I've found leveraging mapping configuration to be highly beneficial. This involves creating a configuration file (typically JSON or YAML) that defines the mapping between incoming and desired data fields. The adapter uses this configuration file to perform the necessary transformations. This approach is both flexible and maintainable, as you can alter mapping rules without modifying the underlying adapter logic.

When dealing with data type mismatches, a different set of procedures come into play. These typically involve data parsing, formatting and conversion. For example, string representations of numbers must be cast into appropriate numerical types, and date strings parsed into date objects using the target system’s date format.  Failure to address these incompatibilities leads to application failures, data corruption and ultimately, business disruption.

For more advanced operations that involve conditional logic, data validation, or more sophisticated transformations, a builder pattern or an abstract factory method might be employed,  resulting in more robust and complex data adapter classes, but also in more structured and reliable results.  This approach allows a high level of customization while preserving encapsulation and testability.

The choice of data adapter also depends on the volume and throughput requirements. For high-throughput systems, I've found it crucial to optimize for performance, for example, by minimizing object creation and string manipulation, and by avoiding complex reflection.

**Code Examples with Commentary:**

Here are three code examples illustrating different adapter strategies:

**Example 1: Simple Mapping Adapter**

This example demonstrates the use of a simple mapping configuration to rename incoming keys. This is commonly used for integrating APIs that have different naming conventions than your system. I have implemented this kind of pattern for a variety of external system integrations, including sales order platforms and inventory management systems.

```python
import json

def simple_map_adapter(input_data, mapping):
    """Adapts input data using a provided key mapping."""
    output_data = {}
    for key, value in input_data.items():
        if key in mapping:
             output_data[mapping[key]] = value
        else:
             output_data[key] = value

    return output_data

# Example Usage:
input_data = {
    "customer_id": 123,
    "first_name": "Alice",
    "last_name": "Smith",
}
mapping = {
   "customer_id": "customerId",
   "first_name": "firstName",
   "last_name": "lastName"
}
output_data = simple_map_adapter(input_data, mapping)
print(json.dumps(output_data, indent = 4))
# Expected output:
# {
#    "customerId": 123,
#    "firstName": "Alice",
#    "lastName": "Smith"
# }

```
**Commentary:**

* The `simple_map_adapter` function takes `input_data` (a dictionary) and a `mapping` dictionary.
* The function iterates through each key-value pair in the input data, and checks if the key exists within the mapping. If a mapping exists, the new key is utilized, and if not, the original key is retained.
* This example assumes that input and output data are dictionaries and are readily interchangeable, which is often the case when working with JSON payloads from APIs.

**Example 2: Data Type Conversion Adapter**

This example demonstrates how to handle data type conversions, specifically when parsing string values for dates and numeric data, an issue I've dealt with multiple times when integrating legacy systems with modern software.

```python
from datetime import datetime

def type_conversion_adapter(input_data):
    """Adapts data, parsing dates and converting numeric strings to numbers."""

    output_data = {}
    for key, value in input_data.items():
        if key == 'orderDate':
            output_data['orderDate'] = datetime.strptime(value, '%Y-%m-%d')
        elif key == 'amount' and isinstance(value, str):
              output_data['amount'] = float(value)
        else:
            output_data[key] = value

    return output_data


# Example usage:
input_data = {
    "orderId": "abc123xyz",
    "orderDate": "2023-10-26",
    "amount": "123.45",
    "product": "Widget"
}

output_data = type_conversion_adapter(input_data)
print(json.dumps(output_data, default=str, indent=4))
# Expected output:
# {
#   "orderId": "abc123xyz",
#   "orderDate": "2023-10-26 00:00:00",
#   "amount": 123.45,
#   "product": "Widget"
# }
```

**Commentary:**

* The `type_conversion_adapter` function receives raw input data and outputs an adjusted dictionary.
* It specifically targets `orderDate` and attempts to convert the value to a datetime object based on the ISO 8601 format of "YYYY-MM-DD".
* Similarly, it targets the field `amount` and checks if it’s a string, and if so, parses it to a float.
* This code includes a simple conversion function of stringified dates to datetime objects and stringified numeric data to numeric values. This conversion process can be more complex if the data input is not standardized.

**Example 3: Advanced Transformation Adapter**

This example represents a more advanced adapter scenario, combining mapping with some transformation logic. I have had to implement these types of transforms on numerous occasions to combine and restructure data from multiple sources into unified data objects.

```python
def transform_adapter(input_data):
    """Transforms the data, combining name fields and standardizing address structure."""
    output_data = {
        "fullName": f"{input_data['firstName']} {input_data['lastName']}",
         "address": {
            "street": input_data['streetAddress'],
            "city": input_data['city'],
            "postalCode": input_data['zipCode']
       }
    }

    return output_data


# Example usage
input_data = {
    "firstName": "Jane",
    "lastName": "Doe",
    "streetAddress": "123 Main St",
    "city": "Anytown",
    "zipCode": "12345"
}

output_data = transform_adapter(input_data)
print(json.dumps(output_data, indent = 4))
# Expected Output:
# {
#   "fullName": "Jane Doe",
#   "address": {
#         "street": "123 Main St",
#         "city": "Anytown",
#         "postalCode": "12345"
#    }
# }
```

**Commentary:**
*  The function `transform_adapter` takes input data and creates a restructured object.
*  It creates a “fullName” field by concatenating the input “firstName” and “lastName” fields.
*  It also creates an “address” nested object using a transformation of the input street, city, and zip data.

**Resource Recommendations:**

For gaining a deeper understanding of data adapter design, I recommend consulting texts and materials focusing on:
    * **Design Patterns:**  Specifically look into adapter, factory, and builder patterns.
    * **Data Mapping and Transformation:** Explore techniques used for ETL processes.
    * **Data Validation:**  Learn about strategies for validating incoming data.
    * **Data Parsing and Formatting:** Explore techniques related to processing dates, numbers, and complex strings.
    * **System Integration:** Materials focusing on data interchange best practices are valuable.

These resources provide a foundational understanding of the concepts discussed and enable the reader to design and implement effective data adapters for a wide variety of scenarios. My practical experience, coupled with the knowledge gained from the above resources, has consistently helped me produce maintainable and robust applications.
