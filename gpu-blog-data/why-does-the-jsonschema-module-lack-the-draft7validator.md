---
title: "Why does the 'jsonschema' module lack the 'Draft7Validator' attribute?"
date: "2025-01-30"
id: "why-does-the-jsonschema-module-lack-the-draft7validator"
---
The `jsonschema` Python library, prior to version 3.0, did not expose `Draft7Validator` directly as an attribute of the root module. This absence stems from a design choice prioritizing a streamlined interface and a version-agnostic approach to validator instantiation, rather than directly exposing individual validator classes. My experience working on a large data validation pipeline at "Acme Analytics" revealed this nuance early on, forcing me to dig into the library's internals.

The core issue isn't an omission, but rather an intentional separation of concerns. Instead of accessing validator classes directly, the `jsonschema` library encourages the use of the `validators` mapping. This mapping associates string representations of JSON Schema drafts (e.g., "draft7", "draft4") with their corresponding validator classes. This design promotes flexibility and enables the library to evolve without requiring constant modification of client code if new drafts are introduced. The library maintains its version agnosticism by handling the selection and instantiation of specific validators under the hood.

Prior to version 3.0, `jsonschema` provided only a rudimentary `validate` function that implicitly used draft4 validators when not otherwise specified. Direct access to other validators required an indirect approach, relying on the `validators` dictionary. To achieve validations against other drafts, one had to explicitly select the corresponding validator class from this dictionary. This was a point of confusion for many users, including myself initially, as direct validator access was seemingly the most intuitive method. The library documentation, however, clearly outlined this usage paradigm.

The library internally maintains this mapping between draft names and validator classes, facilitating its core functionality. It dynamically determines which validator to use based on the schema's `$schema` field or an explicitly provided version specifier. This mechanism abstracts away the underlying validator classes from casual use, offering a more generalized approach to validation. The explicit `Draft7Validator` attribute as seen in version 3.0 and beyond represents a shift, a conscious decision to expose the individual validator classes in addition to the existing flexible mechanism.

The shift to exposing individual validator classes came with version 3.0, which introduced `Draft7Validator` and similar attributes. The primary purpose of this is to make the library more intuitive, especially for those accustomed to object-oriented programming patterns. Despite the added convenience, the previous methodology using the `validators` dictionary continues to operate without deprecation and allows for more explicit version-handling when needed.

Below are examples illustrating how I've handled schema validation using `jsonschema` both pre and post v3.0, showcasing the practical implications of this design.

**Code Example 1: Validation against Draft 7 using the `validators` dictionary (Pre v3.0)**

```python
import jsonschema
from jsonschema import validators

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name", "age"]
}

instance = {"name": "Alice", "age": 30}

# Get the Draft 7 validator class from the validators dictionary
Draft7Validator = validators.validator_for(schema)

# Ensure the schema is valid
Draft7Validator.check_schema(schema)

validator = Draft7Validator(schema) # Instantiate the validator

try:
  validator.validate(instance) # Validate the data against the schema
  print("Validation successful (Pre v3.0 - using validator dictionary).")
except jsonschema.exceptions.ValidationError as e:
  print(f"Validation failed (Pre v3.0): {e}")

instance_invalid = {"name": "Bob"}
try:
    validator.validate(instance_invalid)
    print("Error, should not be validated.")
except jsonschema.exceptions.ValidationError as e:
    print(f"Validation failed (Pre v3.0 - Invalid data): {e}")
```

This example illustrates the pre-3.0 approach. We utilize `validators.validator_for(schema)` to dynamically retrieve the correct validator class based on the schema's `$schema` definition. The subsequent validation process remains consistent. The error handling explicitly shows how validation errors are presented using this older approach, proving the functionality using invalid data.

**Code Example 2: Validation against Draft 7 using Draft7Validator (Post v3.0)**

```python
import jsonschema
from jsonschema import Draft7Validator

schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name", "age"]
}

instance = {"name": "Alice", "age": 30}

# Instantiate the Draft7Validator directly
validator = Draft7Validator(schema)

try:
  validator.validate(instance)
  print("Validation successful (Post v3.0 - using Draft7Validator).")
except jsonschema.exceptions.ValidationError as e:
  print(f"Validation failed (Post v3.0): {e}")

instance_invalid = {"name": "Bob"}
try:
    validator.validate(instance_invalid)
    print("Error, should not be validated.")
except jsonschema.exceptions.ValidationError as e:
    print(f"Validation failed (Post v3.0 - Invalid data): {e}")
```

This demonstrates the use of `Draft7Validator` directly after the v3.0 update. The validator is instantiated using the class, leading to similar behavior but with direct access. This approach makes the validation code more readable for users familiar with object-oriented programming. This example also demonstrates error handling with invalid data, ensuring the functionality is consistent with the previous example.

**Code Example 3: Version Specific Validation using the validators dictionary (Version agnostic)**

```python
import jsonschema
from jsonschema import validators

schema_draft4 = {
  "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

instance = {"name": "Alice", "age": 30}

# dynamically retrieve a validator class based on the schema
validator_class = validators.validator_for(schema_draft4)
validator_draft4 = validator_class(schema_draft4) # Instantiate the validator


try:
  validator_draft4.validate(instance)
  print("Validation successful (Using validator dictionary for dynamic version).")
except jsonschema.exceptions.ValidationError as e:
  print(f"Validation failed (Using validator dictionary for dynamic version): {e}")

schema_draft7 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# dynamically retrieve a validator class based on the schema
validator_class_7 = validators.validator_for(schema_draft7)
validator_draft7 = validator_class_7(schema_draft7) # Instantiate the validator

try:
  validator_draft7.validate(instance)
  print("Validation successful (Using validator dictionary for dynamic version).")
except jsonschema.exceptions.ValidationError as e:
  print(f"Validation failed (Using validator dictionary for dynamic version): {e}")
```

This example showcases the flexibility of the `validators` dictionary. It demonstrates that one can dynamically determine the validator class required based on the `$schema` field within the input schema. This shows that the approach is version-agnostic and highly adaptable. I've found that when schemas with varying draft versions are involved this is a preferred method, as it avoids hardcoding specific validator classes and adds flexibility to the pipeline.

For in-depth understanding of JSON Schema specifications, I recommend exploring the official documentation on the JSON Schema website. For `jsonschema` specific usage, the library's official documentation on the python package index is highly informative. Additionally, reviewing the library’s source code, specifically the validators module, is beneficial to understand the internal mechanics of validator selection and instantiation.
