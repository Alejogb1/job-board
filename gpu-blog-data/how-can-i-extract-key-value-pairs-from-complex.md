---
title: "How can I extract key-value pairs from complex JSON using jq and map?"
date: "2025-01-30"
id: "how-can-i-extract-key-value-pairs-from-complex"
---
JSON, frequently the lingua franca for data exchange, can often present itself in nested, complex structures, necessitating powerful tools for targeted data extraction. I've repeatedly encountered scenarios where specific key-value pairs deep within JSON hierarchies were required, and while standard text processing tools faltered, `jq` in combination with its `map` function consistently provided an elegant, efficient solution.

The core challenge lies in traversing the intricate tree-like structure of a JSON object to pinpoint specific keys and retrieve their associated values. Simple dot notation access (e.g., `.a.b.c`) is inadequate for variable depths or iterative extraction. This is where `jq`'s functional approach and the `map` filter prove invaluable. `map` allows us to iterate over arrays, transform each element according to a supplied function, and collect the results. When combined with the recursive descent operator (`..`) or specific indexing patterns, it becomes a highly versatile mechanism for key-value pair extraction.

Let's start with a straightforward illustration. Suppose I have a JSON structure representing a collection of customer orders:

```json
{
  "orders": [
    {
      "order_id": "12345",
      "customer": {
        "name": "Alice",
        "address": {
          "city": "New York",
          "zip": "10001"
        }
      },
      "items": [
        {"item_id": "A1", "price": 10},
        {"item_id": "B2", "price": 20}
      ]
    },
      {
      "order_id": "67890",
      "customer": {
        "name": "Bob",
        "address": {
          "city": "Los Angeles",
          "zip": "90001"
        }
      },
      "items": [
        {"item_id": "C3", "price": 15},
         {"item_id": "D4", "price": 25}
      ]
    }
  ]
}

```

My objective is to extract a list of just the customer names. I can achieve this through the following `jq` command:

```bash
jq '.orders | map(.customer.name)' data.json
```

This command first selects the `orders` array using `.orders`. Then, the `map(.customer.name)` applies the expression `.customer.name` to each element in the array, thus accessing the `name` value within each `customer` object. The output would be:

```json
[
  "Alice",
  "Bob"
]
```

This simple example demonstrates the basic usage of `map` for one level of traversal. However, the full power unfolds when combined with the recursive descent operator `..`. Consider another scenario where I have a more deeply nested configuration:

```json
{
  "config": {
    "services": {
      "api1": {
        "endpoint": "/v1/data",
        "settings": {
           "timeout": 30,
          "retries": 3
        }
      },
      "api2": {
        "endpoint": "/v2/users",
        "settings": {
           "timeout": 60,
           "retries": 5
        }
      }
    },
      "logging": {
        "level": "INFO",
        "format": "json"
    }
  }
}
```

My intention now is to retrieve all the timeout values, regardless of their location in the JSON. The `..` operator allows me to recursively descend through the structure. I can then use a select filter to locate all keys with name `timeout`:

```bash
jq '.. | .timeout?' data.json |  select(. != null)
```

Here, `..` iterates through the entire JSON structure. `| .timeout?` attempts to extract the value associated with the `timeout` key. The `?` handles potential absence of such keys in some nodes.  `| select(. != null)` filters out null values, ensuring that only actual timeout values are returned.  This would output:

```json
30
60
```

Note that each `timeout` value is presented on a new line. This can be adjusted to return an array if needed.

For a more complex task involving collecting multiple keys at different levels, I typically employ `map` in conjunction with object construction. For instance, imagine needing to extract both the city and zip code from the previous orders data, while maintaining their association within each customer.

```bash
jq '.orders | map({ name: .customer.name, location: { city: .customer.address.city, zip: .customer.address.zip} })' data.json
```

In this instance, I'm utilizing `map` to iterate through the `orders` array, and for each item, I construct a new object containing the `name` and `location`.  The location itself is constructed from keys `city` and `zip` from each respective `customer.address` object.  The result becomes:

```json
[
  {
    "name": "Alice",
    "location": {
      "city": "New York",
      "zip": "10001"
    }
  },
  {
     "name": "Bob",
    "location": {
      "city": "Los Angeles",
      "zip": "90001"
    }
  }
]

```
This final example highlights the ability to construct structured output during the `map` iteration. I find the combination of nested object construction, `map` and recursive descent exceptionally effective at extracting complex data.

The power of `jq` for this kind of extraction comes from its ability to treat JSON structures as data that can be programmatically transformed using functional techniques like mapping. This contrasts with tools relying on purely string-based pattern matching, which tend to struggle with deeply nested and variably structured data.

For deeper understanding, I highly recommend exploring resources such as “The jq manual” which is the official documentation.  Further, I suggest reviewing example tutorials that demonstrate a range of practical use cases of `jq`. Numerous online articles that demonstrate common data extraction patterns using `jq` and `map` can provide additional learning material. I have found that building a personal library of common `jq` queries by experimenting and adapting them to new scenarios is the most effective strategy for mastering its capabilities.
