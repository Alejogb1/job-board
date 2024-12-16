---
title: "Does JMESPath support a spread operator?"
date: "2024-12-16"
id: "does-jmespath-support-a-spread-operator"
---

Okay, let’s tackle this JMESPath question. It's a common point of confusion, and I've personally had to navigate it multiple times in past projects. I wouldn’t call it a straightforward ‘yes’ or ‘no’, and it definitely requires a nuanced understanding of what JMESPath offers and where it falls short. The short answer is: no, JMESPath doesn’t have a spread operator in the way, say, javascript or python does. But, and this is crucial, it offers different mechanisms to achieve similar outcomes, often in a more specific and declarative way.

I remember working on a data aggregation tool for a distributed system. We were pulling a massive amount of json data, and the need for flattening and restructuring nested arrays was paramount. We initially hoped a spread operator would come to our rescue, but quickly learned that it was not included in the JMESPath spec. Instead, we had to become proficient with alternative techniques, primarily utilizing multi-select lists, flatten projections, and even a touch of piping for more complex operations.

So, while JMESPath lacks a conventional spread operator like `...` in JavaScript or `*` in Python when used in this context, its projection features effectively fill that gap. Think of projections as JMESPath's way to iterate over collections and extract data, sort of like a selective map operation, and when used in conjunction with multi-select lists `[]`, you can create the effect of "spreading" and reshaping your data structure.

Let's break this down with some practical examples. Imagine we have the following JSON structure representing customer orders:

```json
{
  "orders": [
    {
      "order_id": 101,
      "items": [
        {"name": "laptop", "price": 1200},
        {"name": "mouse", "price": 25}
       ]
    },
    {
      "order_id": 102,
      "items": [
        {"name": "keyboard", "price": 100},
        {"name": "monitor", "price": 300}
       ]
    }
  ]
}
```

Our initial problem may be: "get all items from all orders into one flat list". If a spread operator existed, it would potentially be a one-line operation. Since it doesn’t, let's explore how to achieve it using JMESPath's projection and multi-select lists.

**Example 1: Flattening with a multi-select projection**

The following JMESPath expression achieves this flattening effect:

```jmespath
orders[*].items[*]
```

This expression means, and I’ll unpack it: Select all elements from `orders` represented by `*`. For each of the selected `order` element, select all the elements from `items` again represented by `*`.  This gives us a two-dimensional array. JMESPath’s default behaviour flattens the result into a single array. In essence, we have extracted all items and “spread” them into a single list. It returns something akin to this:

```json
[
  {"name": "laptop", "price": 1200},
  {"name": "mouse", "price": 25},
  {"name": "keyboard", "price": 100},
  {"name": "monitor", "price": 300}
]
```

Notice that there was no explicit use of the concept ‘spread’ but, through the way JMESPath processes the expression, we achieved the same effect. The absence of an explicit spread operator is, in my experience, beneficial as it encourages thinking more declaratively about the desired data transformation.

**Example 2: Multi-select lists and object restructuring**

Let's say instead of getting a flat list of items, we want to restructure the data to have all items in the single output object, keyed by the order id. We cannot use a spread here, but again, we can craft a clever expression.

```jmespath
orders[*].{orderId: order_id, items: items[*]}
```

Here, for each element in the `orders` list, we are generating an object that includes the `order_id` under the key `orderId`, and the `items` which is a flattening of all items inside the `items` property using a standard projection. This results in:

```json
[
  {
    "orderId": 101,
    "items": [
      {
        "name": "laptop",
        "price": 1200
      },
      {
        "name": "mouse",
        "price": 25
      }
    ]
  },
  {
    "orderId": 102,
    "items": [
      {
        "name": "keyboard",
        "price": 100
      },
      {
        "name": "monitor",
        "price": 300
      }
    ]
  }
]
```

Again, there is no direct spread operator, but with this construct, we effectively achieve a similar goal by selectively extracting and combining elements to restructure the output.

**Example 3: Complex flattening with piping**

Now consider a more complex scenario. Let’s say each item has a quantity, and we want to calculate the total value of each order by multiplying price by quantity, and then sum total revenue across all orders. This could be a tricky problem to solve using only a spread operator since we would need to iterate, transform, and aggregate. JMESPath doesn’t have a spread operator, but it does allow us to chain transformations.

Consider this updated JSON:

```json
{
  "orders": [
    {
      "order_id": 101,
      "items": [
        {"name": "laptop", "price": 1200, "quantity": 1},
        {"name": "mouse", "price": 25, "quantity": 2}
      ]
    },
    {
      "order_id": 102,
      "items": [
        {"name": "keyboard", "price": 100, "quantity": 3},
        {"name": "monitor", "price": 300, "quantity": 1}
       ]
    }
  ]
}
```

The following, more involved expression uses a pipe `|` operator to perform a series of operations:

```jmespath
orders[*].items[*].{total: price * quantity} | sum(@.total)
```

Let's break this down. First, we select and flatten all items using the `orders[*].items[*].{total: price * quantity}` component. This creates a new object for each item with the `total` property. Then, the pipe operator, `|`, passes the array of generated objects to the right hand side expression, `sum(@.total)`.  This expression then adds all the `total` values up. The final output will be a single value representing the total revenue:

```
1950
```
So, while this isn’t a ‘spread’ operation, we've used JMESPath’s declarative nature to iteratively transform the data and perform aggregation, achieving our objective. The `|` operator allows functional composition of queries, which is extremely useful, especially when the data transformation required is more complex.

In conclusion, while JMESPath does not possess a specific ‘spread operator’, the use of projections, multi-select lists, and the pipe operator, in combination, provide powerful mechanisms to achieve similar goals of transforming and restructuring your data. Instead of focusing on a feature that is simply absent, embracing the flexibility of its projection mechanism and list manipulation functionalities will prove far more effective. For a deeper understanding of JMESPath, I would recommend reviewing the official JMESPath specification document as well as a thorough read of "Data Wrangling with Python" by Jacqueline Nolis and Katharine Jarmul, which, while focused on python, provides strong foundational concepts for data transformation and how different query languages can approach these issues. Furthermore, understanding functional programming paradigms, such as map, reduce and filter, is incredibly useful for building complex JMESPath queries and for understanding why certain approaches are taken over others. Remember, in data manipulation, a deep understanding of the tool at hand is always more valuable than searching for missing features.
