---
title: "regular expression in json data validation?"
date: "2024-12-13"
id: "regular-expression-in-json-data-validation"
---

Okay so yeah regular expressions and json data validation huh I've been there man trust me this isn't my first rodeo with this particular tech rodeo

So you're dealing with JSON data that needs some stricter rules than just is it valid JSON or not Right you want to ensure that specific fields within that JSON follow a specific pattern defined by a regex Okay I get it I've spent way too many late nights wrestling with this exact problem and believe me I've seen all the messy ways it can go sideways

First off let's be clear there's no built in way that standard JSON spec provides for regex validation It's just a data format not a schema language or anything like that So we can't expect JSON to magically understand our regex needs We need to roll our own solution but that's fine that's what we do right

Okay so how I usually approach this is by using a programming language that I'm comfortable with Usually Python because hey it's Python and it's pretty straightforward and has libraries that make working with both JSON and regex a breeze But the concepts should translate to any language

What we're going to do is to load the JSON data into a suitable data structure then iterate through the fields and perform our regex check on each one This approach gives us fine grain control over the process and lets us decide exactly which fields we want to validate and how Let's look at a concrete example

Lets say we have this JSON data

```json
{
  "user_id": "user123",
  "email": "test@example.com",
  "phone_number": "555-123-4567",
    "order_ids": ["order1" , "order2"]
}
```

And we want to ensure that for example `user_id` field is a string with only alphanumerical characters, that the `email` field is actually a valid email and that `phone_number` follows a pattern of 3 digits dash 3 digits dash 4 digits I know I know phone number formats are a pain but that's beside the point

Here's how I'd tackle that with Python

```python
import json
import re

def validate_json_with_regex(json_data, schema):
    """
    Validates JSON data against a provided schema with regex patterns.

    Args:
        json_data: The JSON data as a string.
        schema: A dictionary defining the fields and their regex patterns.

    Returns:
         True if the data matches the schema and False otherwise

    Raises:
        ValueError: If the input is not valid json or if a key to be validated is missing
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    
    for field, regex in schema.items():
        if field not in data:
          raise ValueError(f"Missing field {field}")
        value = data[field]
        if not isinstance(value,str):
           value = str(value)
        if not re.match(regex, value):
            return False
    return True

json_string = '''
{
  "user_id": "user123",
  "email": "test@example.com",
  "phone_number": "555-123-4567",
    "order_ids": ["order1" , "order2"]
}
'''

schema = {
    "user_id": r"^[a-zA-Z0-9]+$",
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone_number": r"^\d{3}-\d{3}-\d{4}$"
}

if validate_json_with_regex(json_string, schema):
    print("JSON is valid according to the schema")
else:
    print("JSON does not validate")
```

This is a pretty simple example but it outlines the core idea basically what the function does is to first parse the json data and then it iterates through the keys of our schema dictionary and check if the keys exists in the data provided then checks if the values under those keys match the regular expression in the schema If they dont match it means data is invalid And we exit at the first failed match else we say the whole data is valid

Now the schema we are passing is a dictionary and that is something that can be easily modified and extended so you are not limited to just 3 keys that I provided.

One thing that made me scratch my head back in the day when I first approached this problem is that you need to make sure that your JSON data matches the schema that you are passing because if the key that you are trying to access in JSON does not exist well you are going to get an exception This was a subtle error that cost me several hours

I mean like really several hours.

Okay so lets say we have a case where you have a nested JSON object and you want to apply regex checks there Well that can be a bit more challenging but the same principle applies Here is a possible solution

```python
import json
import re

def validate_nested_json_with_regex(json_data, schema):
    """
    Validates nested JSON data against a provided schema with regex patterns.

    Args:
        json_data: The JSON data as a string.
        schema: A dictionary defining the fields and their regex patterns.
                Nested fields are represented as a string with "." delimiter.

    Returns:
         True if the data matches the schema and False otherwise

    Raises:
        ValueError: If the input is not valid json or if a key to be validated is missing
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    
    def _get_nested_value(obj, key_path):
      parts = key_path.split('.')
      current = obj
      for part in parts:
        if isinstance(current, dict) and part in current:
           current = current[part]
        else:
           return None #Key not found
      return current

    for field, regex in schema.items():
        value = _get_nested_value(data,field)
        if value is None:
          raise ValueError(f"Missing field {field}")
        if not isinstance(value,str):
           value = str(value)
        if not re.match(regex,value):
            return False
    return True

json_string = '''
{
  "user": {
    "profile":{
         "user_id": "user123",
         "email": "test@example.com"
      },
    "contact": {
      "phone_number": "555-123-4567"
     }
   }
}
'''

schema = {
    "user.profile.user_id": r"^[a-zA-Z0-9]+$",
    "user.profile.email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "user.contact.phone_number": r"^\d{3}-\d{3}-\d{4}$"
}

if validate_nested_json_with_regex(json_string, schema):
    print("JSON is valid according to the schema")
else:
    print("JSON does not validate")
```

So in this example we have nested JSON and we use a utility function `_get_nested_value` to get the value at a particular nested key Using a dot `.` notation we represent nested keys

And yes I had my share of bugs using this pattern because sometimes the JSON structure has lists or objects that don't exist so I needed to add some checks for that This is how I learn from my mistakes I swear

Now you might be wondering what if I have an array and I want to validate each element inside the array well that is also something that is possible to do here is how

```python
import json
import re

def validate_json_array_elements_with_regex(json_data, schema):
    """
    Validates JSON data that has an array against a provided schema with regex patterns.
    Applies the regex check to each element of the array field

    Args:
        json_data: The JSON data as a string.
        schema: A dictionary defining the fields and their regex patterns.

    Returns:
         True if the data matches the schema and False otherwise

    Raises:
        ValueError: If the input is not valid json or if a key to be validated is missing or if the type of the value is not an array
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    
    for field, regex in schema.items():
        if field not in data:
          raise ValueError(f"Missing field {field}")
        value = data[field]
        if not isinstance(value,list):
          raise ValueError(f"Field {field} is not an array")
        for item in value:
            if not isinstance(item,str):
               item = str(item)
            if not re.match(regex,item):
                return False
    return True
    

json_string = '''
{
  "order_ids": ["order123","order456","order789"],
  "products_ids" : ["product_A" , "product_B"]
}
'''

schema = {
    "order_ids": r"^[a-zA-Z0-9]+$",
    "products_ids":r"^[a-zA-Z]+[_][a-zA-Z]$"
}

if validate_json_array_elements_with_regex(json_string, schema):
    print("JSON is valid according to the schema")
else:
    print("JSON does not validate")
```

So now we are just iterating through each element of the array and validating against the regex provided in the schema

And of course I've made many mistakes in the past while dealing with edge cases like for example what happens when you have an empty array I mean we have to account for those cases right?

One thing that I'd recommend is to not re-invent the wheel and instead take a look at libraries that already provide robust solutions for schema validation there are some good ones available in all sorts of languages That way you'd avoid writing code like this from the ground up

Okay so a few pointers for you now if you want to dig deeper into this topic

First regex itself its a field that needs its own time to understand it if you really want to know what you are doing so I recommend taking a look at the book "Mastering Regular Expressions" by Jeffrey Friedl it’s a deep dive into the world of regex and it's a very useful book for your daily tech life

Also for understanding more about JSON and schemas in a more general sense I would recommend looking at the official JSON specification and checking the different things it enables

And also take a look at "Understanding JSON Schema" by Kris Zyp it will give you a better idea on how to work with JSON and schemas and it may also help you make your validation logic more robust

And that's pretty much it for now I have provided code examples of validating json data with regex with different use cases so if you have any other doubts feel free to ask I will be around to help you and maybe share some more stories from my past experiences
