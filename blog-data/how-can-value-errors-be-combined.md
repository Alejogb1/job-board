---
title: "How can value errors be combined?"
date: "2024-12-23"
id: "how-can-value-errors-be-combined"
---

Let's explore how we can effectively combine value errors, a situation I've encountered more times than I'd prefer over my career. It's a common challenge, particularly when dealing with complex systems or user inputs where invalid data is inevitable. The crux of the problem isn't so much about encountering a single error, but managing a collection of them and presenting them in a way that provides actionable feedback without overwhelming the user or making debugging a nightmare. We’re not just talking about catching errors; we're talking about orchestrating their collective behavior to improve the system's resilience and user experience.

My approach, developed through years of trial and (mostly) error, focuses on moving beyond simple boolean flags and into structured error handling. Instead of thinking in terms of ‘an error occurred,’ I tend to think in terms of ‘*which* errors occurred, and what information can we extract from them.’ This subtle shift in perspective makes all the difference.

The traditional method, and one I often saw junior developers gravitate towards early in my career, involves using boolean flags scattered throughout the code. Something like:

```python
def process_data(data):
    valid_input = True
    if not data:
        valid_input = False
        print("Error: No data provided.")

    if not isinstance(data, list):
        valid_input = False
        print("Error: Input must be a list.")

    if valid_input:
       # Process data
       print("Processing successful")
    else:
        print("Processing failed due to errors.")
    return valid_input
```

This approach, while seemingly straightforward, suffers from several crucial flaws. First, it obscures the specific nature of the errors. The user, or the person debugging the code, only receives a generic 'processing failed' message, and the print statements are generally unsuitable for any form of automated logging or sophisticated user feedback. Second, this method lacks flexibility. What if I needed to return *all* the specific errors? Or perhaps provide different messages based on the errors detected? Handling such scenarios becomes incredibly convoluted with a basic boolean flag system.

The improved path involves aggregating errors into a collection – a list, a set, or even a custom class, depending on the complexity of your scenario. I usually prefer a dictionary structure with error codes to ensure programmatical access to the errors for logging and further processing, which brings me to my first concrete example.

```python
def validate_data_v1(data):
    errors = {}
    if not data:
        errors["empty_data"] = "Error: No data provided."

    if not isinstance(data, list):
        errors["invalid_type"] = "Error: Input must be a list."

    return errors

def process_data_v1(data):
    validation_errors = validate_data_v1(data)

    if validation_errors:
        print("Processing failed with the following errors:")
        for error_code, message in validation_errors.items():
            print(f"- {error_code}: {message}")
    else:
        print("Processing successful.")

    return validation_errors
```

This version provides significantly improved error reporting. I am now able to easily examine what exactly went wrong. This version also demonstrates the use of error codes, `empty_data`, and `invalid_type`. I frequently rely on such identifiers as they allow me to automate responses such as error handling, logging, and user feedback customization. Each error is stored within a dictionary, preserving both the code and the associated message. The calling function can then leverage this detailed information for more sophisticated error processing.

Now, imagine a more complex scenario where you need to handle errors across multiple interdependent components. It wouldn’t be sensible to check for errors locally within each function, you’d want something to pass all the errors through. This is where a structured, accumulative approach shines. Consider a situation where you’re processing user profiles and need to check for errors in username, email, and other user details, and then aggregate all of them for a comprehensive response.

```python
def validate_username(username):
    errors = []
    if not username:
        errors.append("username_missing")
    if len(username) < 5:
        errors.append("username_short")
    return errors

def validate_email(email):
    errors = []
    if not email:
        errors.append("email_missing")
    if "@" not in email:
        errors.append("email_invalid_format")
    return errors

def validate_profile(username, email):
  all_errors = {}
  username_errors = validate_username(username)
  email_errors = validate_email(email)
  if username_errors:
     all_errors["username"] = username_errors
  if email_errors:
     all_errors["email"] = email_errors

  return all_errors

def process_profile(username, email):
  validation_results = validate_profile(username,email)

  if validation_results:
    print("Profile validation failed with the following errors:")
    for field, error_codes in validation_results.items():
      print(f"- {field}: {', '.join(error_codes)}")
  else:
    print("Profile validated successfully")

  return validation_results

```

This example builds upon the previous one by using lists instead of the dictionary to further group error codes for specific fields, with the dictionary containing all the errors for all the fields. This allows us to pass the errors up the chain of function calls while maintaining a clear and structured representation. It’s no longer just about *if* there is an error, but *where* the errors are, and *which* specific errors exist. These structures can be modified to fit your needs, and can be designed to be hierarchical or simply flattened.

The third, and final example, is a very flexible approach where you accumulate errors from multiple potentially failing operations using a function that will take an error aggregator as input. This is an approach I often find myself using in complex business logic that involves many function calls, and one that reduces code duplication.

```python
class ErrorAggregator:
  def __init__(self):
    self.errors = {}

  def add_error(self, code, message):
     self.errors[code] = message

  def has_errors(self):
     return bool(self.errors)

  def get_all_errors(self):
    return self.errors

def validate_with_aggregator(data, error_aggregator):
  if not data:
        error_aggregator.add_error("no_data","No data provided")
  if not isinstance(data, dict):
       error_aggregator.add_error("data_invalid","Input must be a dictionary")

def operation_one(data, error_aggregator):
  if "field1" not in data:
        error_aggregator.add_error("field1_missing", "Field1 missing from input")
  if isinstance(data.get("field1"), int):
     if data["field1"] < 0:
         error_aggregator.add_error("field1_invalid", "Field1 cannot be negative")
  else:
        error_aggregator.add_error("field1_invalid_type", "Field1 must be an integer")


def operation_two(data, error_aggregator):
  if "field2" not in data:
      error_aggregator.add_error("field2_missing", "Field2 missing from input")

def perform_operations(data):
    error_aggregator = ErrorAggregator()
    validate_with_aggregator(data, error_aggregator)
    if not error_aggregator.has_errors():
        operation_one(data, error_aggregator)
        operation_two(data, error_aggregator)


    if error_aggregator.has_errors():
      print("Operation failed with the following errors:")
      for error_code, message in error_aggregator.get_all_errors().items():
        print(f"- {error_code}: {message}")
    else:
       print("All operations successful.")

    return error_aggregator
```

The above code showcases how a class can be used to accumulate errors within potentially many different functions in a very flexible manner. The `ErrorAggregator` is passed around into various functions, which add error codes and messages to it. The code has a clear distinction between validation and operation code. This demonstrates the flexibility with which to accumulate and handle errors.

For those seeking to dive deeper into robust error handling, I highly recommend exploring the "Domain-Driven Design" book by Eric Evans, as it provides a foundational understanding of how to model and manage errors in a complex system. Additionally, reading articles on functional programming principles will also provide a perspective on how error handling can be done within monads, which offers another avenue to explore. Understanding exception handling mechanisms, as described in "Effective Java" by Joshua Bloch, and how to potentially represent exceptions as data rather than control flow also offers good insight. These resources, combined with practical experience, should give you a solid understanding of how to effectively combine value errors.
