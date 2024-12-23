---
title: "What is the difference between decomposition by subdomain vs business capability?"
date: "2024-12-23"
id: "what-is-the-difference-between-decomposition-by-subdomain-vs-business-capability"
---

, let's unpack this. I've seen the confusion around subdomain vs. business capability decomposition quite a bit, especially in organizations moving towards more microservice-oriented architectures. I'd say the core misunderstanding stems from how we define “scope.” We're talking about different lenses through which to analyze an organization’s complexity and, consequently, how we structure our systems. I vividly recall a particularly painful migration project a few years back where we blurred these two concepts, and the resulting spaghetti architecture haunted us for months. It's crucial to get this distinction correct early on.

To put it simply, **subdomain decomposition** is all about dividing your problem space based on the *technical* aspects of the business. It focuses on the 'what' – what are the core activities that enable the business to operate? These subdomains often relate directly to specific business functions or departments, such as marketing, sales, or fulfillment. The emphasis is on domain knowledge and the inherent technical boundaries within that knowledge. For example, in an e-commerce system, you might identify subdomains like *product catalog*, *inventory management*, *order processing*, and *customer accounts*. These are tangible entities with clear operational parameters.

Conversely, **business capability decomposition** examines the 'how' – how does the business achieve its goals? Capabilities are *what the business does*, rather than the departments or functions that do them. They're often cross-functional and require collaboration between multiple subdomains. Think of capabilities as a higher-level abstraction, focusing on the outcomes that matter to the business rather than the underlying mechanics. Examples in an e-commerce system could be *attract new customers*, *manage customer loyalty*, or *fulfill orders efficiently*. These capabilities encompass a wider range of actions and might involve coordination across several of the previously defined subdomains.

Here's the critical difference: subdomains represent concrete areas of expertise and data ownership, whereas capabilities describe desired business outcomes. Subdomains are more about the *nouns* (products, users, orders), while capabilities are more about the *verbs* (attract, manage, fulfill).

Think of it this way: subdomains provide the building blocks – the lego bricks, if you will – and capabilities explain how we assemble those bricks to achieve a specific end goal.

To make this clearer, consider some code examples. Let's start by thinking about an e-commerce context:

**Example 1: Subdomain-Oriented Service**

Let’s say we've identified a *product catalog* subdomain. A simple service for managing this might look like this using Python:

```python
class ProductCatalogService:

    def __init__(self, db_connection):
        self.db = db_connection

    def get_product_by_id(self, product_id):
        query = "SELECT * FROM products WHERE id = %s"
        cursor = self.db.cursor()
        cursor.execute(query, (product_id,))
        result = cursor.fetchone()
        cursor.close()
        return result

    def add_product(self, product_data):
         query = "INSERT INTO products (name, description, price) VALUES (%s, %s, %s)"
         cursor = self.db.cursor()
         cursor.execute(query, (product_data['name'], product_data['description'], product_data['price']))
         self.db.commit()
         cursor.close()
         return True

# Example Usage
# db_conn = get_database_connection()
# catalog_service = ProductCatalogService(db_conn)
# product = catalog_service.get_product_by_id(123)
# success = catalog_service.add_product({"name":"New Gadget","description":"Cool stuff", "price":99.99})
```

This service operates squarely within the boundaries of the *product catalog* subdomain. It handles all operations related to fetching and manipulating product data.

**Example 2: Capability-Oriented Logic**

Now, let's look at the *attract new customers* capability. This capability requires interaction with multiple subdomains. Consider a simplified function in the same system that uses both `ProductCatalogService` and a fictional `MarketingCampaignService`:

```python
class MarketingCampaignService:

     def __init__(self, db_connection):
        self.db = db_connection

     def get_active_campaigns(self):
        query = "SELECT campaign_id FROM campaigns WHERE is_active = true"
        cursor = self.db.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results

def attract_new_customers(product_catalog_service, marketing_campaign_service):
    # Assume the business wants to highlight products in active marketing campaigns.
    active_campaigns = marketing_campaign_service.get_active_campaigns()
    featured_products = []
    for campaign in active_campaigns:
         # Here, we'd query the relevant product IDs from an active campaign
        # Lets say the data is structured as (campaign_id, [product_ids])
        query = f"SELECT product_ids FROM campaign_products WHERE campaign_id = '{campaign[0]}'"
        cursor = marketing_campaign_service.db.cursor()
        cursor.execute(query)
        products = cursor.fetchone()[0]
        cursor.close()

        for product_id in products.split(','): # Assuming product ids are comma separated in DB
             product = product_catalog_service.get_product_by_id(int(product_id))
             if product:
                featured_products.append(product)
    return featured_products

# Example Usage
# featured_items = attract_new_customers(catalog_service, MarketingCampaignService(db_conn))

```
Here, the `attract_new_customers` function pulls information from both the *product catalog* subdomain (via the `ProductCatalogService`) and marketing campaigns data via a `MarketingCampaignService`. It's not tied to a single subdomain; instead, it orchestrates interactions across multiple areas to fulfill the *attract new customers* business capability.

**Example 3: A Slightly More Complex Capability**

Let’s illustrate with another capability, *fulfill orders efficiently*, which often interacts with multiple subdomains like inventory management and shipping. This time we'll use a very basic example showing the flow:

```python

class InventoryManagementService:
  def __init__(self, db_connection):
    self.db = db_connection

  def decrement_inventory(self, product_id, quantity):
    query = "UPDATE inventory SET quantity = quantity - %s WHERE product_id = %s"
    cursor = self.db.cursor()
    cursor.execute(query,(quantity, product_id))
    self.db.commit()
    cursor.close()
    return True

class ShippingService:
    def __init__(self, db_connection):
        self.db = db_connection

    def ship_order(self, order_id, shipping_address):
        query = "INSERT into shipping_details (order_id, shipping_address) values (%s, %s)"
        cursor = self.db.cursor()
        cursor.execute(query, (order_id, shipping_address))
        self.db.commit()
        cursor.close()
        return True


def fulfill_order(order_details, product_catalog_service, inventory_service, shipping_service):
    for item in order_details['items']:
        product_id = item['product_id']
        quantity = item['quantity']

        # Ensure product exists
        product = product_catalog_service.get_product_by_id(product_id)
        if not product:
          print(f"Product with ID: {product_id} does not exist")
          return False # Failure - order can't be fulfilled


        # Decrease inventory:
        inventory_service.decrement_inventory(product_id, quantity)


    shipping_service.ship_order(order_details['order_id'], order_details['shipping_address'])

    return True

# Example usage
# success = fulfill_order(order_details, catalog_service, inventory_service, shipping_service)
```

The `fulfill_order` function is clearly within the context of a business *capability*. It uses services from different subdomains – `ProductCatalogService` to validate products, `InventoryManagementService` to adjust stock, and `ShippingService` to handle the delivery. This emphasizes that capabilities require coordination and collaboration among different parts of the system.

As you can see, thinking in subdomains allows us to encapsulate specific functionality, leading to more cohesive and loosely coupled services. However, these services are merely components. Capabilities on the other hand represent the broader business needs and outcomes, guiding how these subdomains work in concert. When we start with business capabilities, we get a more holistic picture of the business needs and can structure our systems more effectively.

In practice, a hybrid approach is often most effective. Start by identifying the key business capabilities. This gives you a clear understanding of what your business needs to accomplish. Then, decompose your system into subdomains aligned with those capabilities. This ensures that your technical implementation reflects your business’s true requirements.

To deepen your understanding, I recommend looking into "Domain-Driven Design" by Eric Evans. This book provides a detailed explanation of domain decomposition, bounded contexts, and strategic design patterns. For more on the capability side, I'd suggest researching the concept of *business architecture* and the related frameworks, often explored in management science literature and consulting publications from firms like McKinsey and BCG.

The separation is subtle but crucial. Subdomains define *what* components we need; business capabilities dictate *how* these components work together to deliver value. The interplay between these two viewpoints is essential for building adaptable, well-architected systems that are aligned with real business needs.
