---
title: "How can I translate MongoDB data structures into a domain model?"
date: "2024-12-23"
id: "how-can-i-translate-mongodb-data-structures-into-a-domain-model"
---

Let's tackle this from a practical perspective, shall we? I've encountered this particular conundrum more times than i care to count, especially in the early days of integrating NoSQL databases into projects with pre-existing domain models. It’s not just about data transfer, it’s about crafting a robust bridge between your data storage and your application's logic. Translating mongodb data structures to a domain model effectively involves careful planning and a solid understanding of both paradigms. It's not a one-size-fits-all process; the optimal approach depends greatly on the complexity of your domain and your team's specific needs.

First, the core concept: a domain model represents the conceptual model of your application’s problem domain. It’s not a database schema; it’s a representation of the objects and relationships that matter to the users. MongoDB, being a document database, allows a different level of flexibility in how data is structured. This disparity can lead to impedance mismatches if not handled carefully. We’re not just moving data; we're translating concepts.

My experience leans heavily on a microservices architecture where each service had its own data store. In one instance, a user management service employed mongodb, while other services relied on relational databases. This presented challenges in terms of data consistency and cross-service communication. The naive approach—directly exposing mongodb documents as application objects—led to tight coupling and maintenance nightmares. So, we had to adopt a more principled strategy.

The key is to define clear mapping strategies. We must understand how the inherent structure in mongodb documents (nested objects, arrays, embedded documents) align with our domain model’s entities (classes, properties, relationships). It involves breaking the document's structure down into manageable pieces and then reconstituting those into domain objects.

Here’s a crucial point, the transformation process isn't usually a direct, one-to-one relationship. Sometimes you'll need to flatten nested structures, combine elements from different fields, or even perform simple data transformations, like parsing strings into date objects. I’ve seen projects where failure to handle this led to unnecessary complexity in the application code and performance issues.

Let’s consider an example. Suppose we have the following mongodb document, representing a user:

```json
{
  "_id": ObjectId("64455f1a0d24827411122367"),
  "firstName": "Jane",
  "lastName": "Doe",
  "contactDetails": {
    "email": "jane.doe@example.com",
    "phone": "555-123-4567"
  },
  "addresses": [
    {
      "street": "123 Main St",
      "city": "Anytown",
      "zip": "12345"
    },
    {
      "street": "456 Second Ave",
      "city": "Someplace",
      "zip": "67890"
    }
  ],
   "status" : "active",
    "createdOn" : ISODate("2024-04-23T14:30:00Z"),
    "lastModified" : ISODate("2024-04-23T15:30:00Z")
}
```

Our domain model, however, might have a `User` class and an `Address` class, and we may choose to flatten the 'contactDetails' into user properties:

```python
from datetime import datetime
from bson import ObjectId

class Address:
    def __init__(self, street: str, city: str, zip: str):
        self.street = street
        self.city = city
        self.zip = zip

class User:
    def __init__(self, id: str, firstName: str, lastName: str, email: str, phone: str, addresses: list[Address], status : str, createdOn : datetime, lastModified : datetime):
        self.id = id
        self.firstName = firstName
        self.lastName = lastName
        self.email = email
        self.phone = phone
        self.addresses = addresses
        self.status = status
        self.createdOn = createdOn
        self.lastModified = lastModified

    @staticmethod
    def from_document(document: dict) -> 'User':
        addresses = [Address(a["street"], a["city"], a["zip"]) for a in document.get("addresses", [])]
        return User(
            str(document["_id"]),
            document["firstName"],
            document["lastName"],
            document["contactDetails"]["email"],
            document["contactDetails"]["phone"],
            addresses,
            document["status"],
            document["createdOn"],
            document["lastModified"]
        )
```

In this first code snippet, the static `from_document` method is the key. It receives the mongodb document as input and constructs a `User` instance, handling the mapping and flattening of the contact details and transforming the embedded address documents into Address objects. Notably, we convert the `_id` which is an `ObjectId` into a string. This is a common step to make it more suitable for use outside the MongoDB context.

The next aspect is dealing with more complex nested structures. Consider a scenario involving product catalog data. You may have product documents with nested categories and attributes which need to be represented as separate domain entities.

```json
{
    "_id": ObjectId("64455f1a0d24827411122368"),
    "name": "Laptop",
    "category": {
        "name": "Electronics",
        "subCategory": "Computers",
        "attributes" : ["size", "weight", "processor"]
    },
    "price": 1200,
    "description": "A high-performance laptop",
    "inventory": {
      "stockLevel" : 100,
      "lastStockCheck" : ISODate("2024-04-24T10:00:00Z")
    }
}
```

Here, a reasonable domain model might include separate `Product`, `Category`, and `Inventory` entities.

```python
from datetime import datetime
from bson import ObjectId

class Inventory:
  def __init__(self, stockLevel : int, lastStockCheck : datetime):
    self.stockLevel = stockLevel
    self.lastStockCheck = lastStockCheck

class Category:
    def __init__(self, name: str, subCategory: str, attributes : list):
        self.name = name
        self.subCategory = subCategory
        self.attributes = attributes


class Product:
    def __init__(self, id: str, name: str, category: Category, price: float, description: str, inventory : Inventory):
        self.id = id
        self.name = name
        self.category = category
        self.price = price
        self.description = description
        self.inventory = inventory

    @staticmethod
    def from_document(document: dict) -> 'Product':
        category = Category(
            document["category"]["name"],
            document["category"]["subCategory"],
            document["category"]["attributes"]
        )

        inventory = Inventory(
          document["inventory"]["stockLevel"],
          document["inventory"]["lastStockCheck"]
        )
        return Product(
           str(document["_id"]),
            document["name"],
            category,
            document["price"],
            document["description"],
            inventory
        )
```

Again, the static `from_document` method in the second code snippet handles the transformation, now with an added layer of nested object creation. We map the embedded document into separate objects that reflect the domain concepts. It's also useful to note that if these entities had further relations between each other within your domain model, then you would handle that within this method to create those associations. For instance, a product may belong to one or many categories, or have an associated supplier. These would all need to be built into this transformation layer if not explicitly present in your data model.

Finally, let’s touch on handling array of primitives. Suppose we have an order document:

```json
{
    "_id": ObjectId("64455f1a0d24827411122369"),
    "orderId": "ORD-12345",
    "customerName": "John Smith",
    "productIds": [
      ObjectId("64455f1a0d24827411122367"),
      ObjectId("64455f1a0d24827411122368")
    ],
    "totalAmount": 1300,
    "orderedOn" : ISODate("2024-04-25T09:00:00Z")
}
```

The list of product ids would typically be resolved into a list of `Product` objects by fetching the products by ids in the service layer. In this case, the order domain model would keep the product ids as strings rather than trying to eagerly load all products which could be inefficient if the order had hundreds of productIds. However, if you did want the order object to contain a collection of product objects you would resolve that within this layer.

```python
from datetime import datetime
from bson import ObjectId

class Order:
    def __init__(self, id: str, orderId: str, customerName: str, productIds: list[str], totalAmount: float, orderedOn:datetime):
        self.id = id
        self.orderId = orderId
        self.customerName = customerName
        self.productIds = productIds
        self.totalAmount = totalAmount
        self.orderedOn = orderedOn


    @staticmethod
    def from_document(document: dict) -> 'Order':

        product_ids = [str(id) for id in document.get("productIds", [])]
        return Order(
           str(document["_id"]),
            document["orderId"],
            document["customerName"],
           product_ids,
           document["totalAmount"],
            document["orderedOn"]
        )
```

In this third code snippet, we simply convert each `ObjectId` inside the "productIds" array into a string within the `from_document` method. This keeps the domain object simple and lets the service layer take care of fetching the product entities.

Key principles to remember when translating document data:

1.  **Explicit mapping:** Never assume a one-to-one mapping. The domain model's clarity and maintainability rely on carefully defined conversions.
2.  **Decoupling:** Keep the domain model separate from the data access logic. This allows for flexibility in data storage without affecting the core logic of the application. We've achieved this with the static `from_document` methods.
3.  **Transformation logic:** Handle transformations within the mapper layer, not within the domain model itself. The domain model should contain only essential business logic.
4.  **Id handling:** `ObjectId` instances should be transformed into something more generic, such as a string or a UUID representation when in your domain model.
5.  **Service Layer:** This layer is responsible for orchestrating the data access and transformation, calling the domain model with appropriate objects

For further reading, I recommend exploring Eric Evans' "Domain-Driven Design" for a deeper understanding of domain modeling principles. Martin Fowler’s "Patterns of Enterprise Application Architecture" is also an excellent resource for understanding different architectural approaches and data mapping strategies. Additionally, the official MongoDB documentation is a must, particularly the sections that explain data modeling concepts in mongoDB and the BSON format. These will provide the theoretical and practical knowledge that allows one to create robust application using NoSQL databases. I've found that a sound understanding of these concepts is essential to prevent technical debt down the road.
