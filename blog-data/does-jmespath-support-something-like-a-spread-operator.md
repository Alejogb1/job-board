---
title: "Does JMESPath support something like a spread operator?"
date: "2024-12-23"
id: "does-jmespath-support-something-like-a-spread-operator"
---

Okay, let's talk about JMESPath and its capabilities, particularly regarding anything resembling a spread operator. It's a frequent question, and one that I've encountered more than a few times in projects where we heavily relied on manipulating json data structures. The short answer is: no, JMESPath doesn't inherently support a spread operator as you might find it in, say, JavaScript or Python. However, the language offers alternative approaches to achieve similar outcomes, albeit with a different syntax and mindset. It's crucial to understand JMESPath is a query language, not a general-purpose programming language, and therefore its priorities lean heavily towards data retrieval and transformation, rather than complex data manipulation involving spread operations.

Instead of a direct spread, JMESPath leverages its core features like projections, list flattening, and object creation to pull apart, combine, or restructure json documents. I’ve seen engineers mistakenly try to shoehorn spread semantics into JMESPath, and that usually leads to unnecessarily complex or outright non-functional queries. Instead, let’s focus on how to get the data you need using the tools it does provide. Think about it this way: instead of asking “how can I *spread* this?”, ask “how can I *project*, *filter* or *transform* this to get what I need?”. That's usually where the solution lies.

The real-world scenarios where a spread operator might feel indispensable often involve cases where you need to extract multiple fields from a list of objects and combine those fields into a new structure. Or, conversely, where you want to take multiple lists and merge them. In my experience, during the development of a service that aggregated data from various APIs, we repeatedly faced such problems, where the need to shape data for consistent consumption was crucial. That’s where understanding JMESPath’s projection, list flattening and, to a lesser degree, object creation became vital.

Let's examine some concrete examples to illustrate these concepts. Consider this JSON document, which represents a hypothetical dataset of product categories, and nested products:

```json
{
  "categories": [
    {
      "name": "Electronics",
      "products": [
        {"id": "1", "name": "Laptop", "price": 1200},
        {"id": "2", "name": "Smartphone", "price": 800}
      ]
    },
    {
      "name": "Books",
      "products": [
        {"id": "3", "name": "Fiction", "price": 20},
        {"id": "4", "name": "Non-fiction", "price": 30}
      ]
    }
  ]
}
```

Now, imagine we need a flat list containing only product ids. A naive approach might tempt us to look for a spread-like operator. However, in JMESPath, the correct way involves a combination of projection and flattening. The following query accomplishes exactly this:

```jmespath
categories[*].products[*].id
```

This query first projects the *products* arrays from each *categories* element. Then it projects the *id* from each product. Lastly, JMESPath automatically flattens the resulting list of lists into a single, flat list containing just the product ids. The result would be `["1", "2", "3", "4"]`. If that isn't enough, and we wanted to keep the product name too, we can adjust it:

```jmespath
categories[*].products[*].[id, name]
```

This will generate a flattened result of `[["1", "Laptop"], ["2", "Smartphone"], ["3", "Fiction"], ["4", "Non-fiction"]]`. This demonstrates that while a spread operator isn't directly available, projection allows you to select and combine fields into structured data.

Now, suppose we require a structure where each product is included, but the *category name* is also associated with every product. Again, this does not require a spread operator but a more thoughtful projection strategy, often involving the construction of new objects directly within the projection. We can do this:

```jmespath
categories[*].products[*].{id: id, name: name, category: ^.name}
```

This query first selects all the products, then for each product, creates a new object. Within the object it selects the product *id*, the product *name* and also, by using the `^` (parent selector), selects the *name* of the category this product belongs to. The result will be a flat list of objects, similar to what a spread operator followed by field assignment could achieve in other languages:

```json
[
    {
        "id": "1",
        "name": "Laptop",
        "category": "Electronics"
    },
    {
        "id": "2",
        "name": "Smartphone",
        "category": "Electronics"
    },
    {
        "id": "3",
        "name": "Fiction",
        "category": "Books"
    },
     {
        "id": "4",
        "name": "Non-fiction",
        "category": "Books"
    }
]
```

It’s this ability to not just select, but reshape data on the fly, that allows JMESPath to avoid the direct need for spread semantics.

Another scenario may require you to merge attributes of different elements within the same document. Consider a slightly modified JSON document where each category has some associated metadata, and you need to attach this metadata to the corresponding products.

```json
{
    "categories": [
      {
        "name": "Electronics",
        "metadata": {"supplier": "TechCo", "shipping": "Fast"},
        "products": [
          {"id": "1", "name": "Laptop", "price": 1200},
          {"id": "2", "name": "Smartphone", "price": 800}
        ]
      },
      {
        "name": "Books",
        "metadata": {"supplier": "PageTurners", "shipping": "Standard"},
        "products": [
          {"id": "3", "name": "Fiction", "price": 20},
          {"id": "4", "name": "Non-fiction", "price": 30}
        ]
      }
    ]
  }
```

While JMESPath doesn't provide a direct way to 'spread' metadata into products in the same way you might in, say, Javascript, we can achieve a similar result by carefully leveraging parent selectors and creating new objects with the required data. Let’s assume we want a flat list of products, where each product also includes the category's metadata:

```jmespath
categories[*].products[*].{id: id, name: name, price: price, supplier: ^.metadata.supplier, shipping: ^.metadata.shipping}
```

This JMESPath query iterates through each category's products, and for each product, constructs a new object that includes the product's id, name, and price, while additionally pulling the supplier and shipping metadata from the parent category object using the `^` selector. The result will be a flat list of objects, each including product attributes combined with the category's metadata:

```json
[
    {
        "id": "1",
        "name": "Laptop",
        "price": 1200,
        "supplier": "TechCo",
        "shipping": "Fast"
    },
    {
        "id": "2",
        "name": "Smartphone",
         "price": 800,
        "supplier": "TechCo",
        "shipping": "Fast"
    },
    {
        "id": "3",
        "name": "Fiction",
        "price": 20,
        "supplier": "PageTurners",
        "shipping": "Standard"
    },
     {
        "id": "4",
        "name": "Non-fiction",
        "price": 30,
        "supplier": "PageTurners",
        "shipping": "Standard"
    }
]
```

These examples highlight the power of JMESPath's projection capabilities. Instead of a spread operator, you’re essentially creating new, custom objects on the fly, while simultaneously restructuring and flattening the input json. This approach, using a mix of projections and object creation, is a more aligned with JMESPath’s design as a query language.

For further depth into JMESPath, I'd recommend consulting the official JMESPath specification. It is available on GitHub and is a precise definition of its syntax and semantics, and there's also a good chapter on JMESPath in *Effective AWS with Python* by Mark C. Veenstra, which provides practical examples in a cloud computing context. For understanding the underlying concepts behind query languages in general, papers on topics like relational algebra, while not directly JMESPath specific, can provide a more fundamental understanding of querying processes which helps understand the design choices.

In conclusion, while JMESPath doesn’t offer a spread operator, its robust selection, projection and object creation capabilities provide powerful and flexible alternatives. Instead of focusing on simulating a spread operator, focusing on mastering the available features like projection, list flattening, and the use of parent selectors will equip you to handle various data transformation needs effectively and efficiently with JMESPath. I find that its declarative approach, once understood, is generally more maintainable and less prone to complexity than manually looping and assembling data structures in imperative programming styles.
