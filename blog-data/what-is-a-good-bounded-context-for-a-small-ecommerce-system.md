---
title: "What is a good Bounded Context for a small eCommerce System?"
date: "2024-12-15"
id: "what-is-a-good-bounded-context-for-a-small-ecommerce-system"
---

let's talk about bounded contexts for a small e-commerce system, something i've seen a few times. it's a deceptively simple question, really. the devil is in the details, as they say, or maybe that's just a very verbose way of saying "it's complicated". i've personally had to untangle a few systems where the bounded contexts were… less than ideal, let’s just say. like, imagine a single, monolithic database trying to handle everything from user profiles to shipping logistics. total nightmare. performance tanks, changes become a multi-day project, and the team, well, let’s just say morale isn’t exactly soaring.

anyway, a bounded context, in essence, is about defining clear boundaries within your system. it's like saying, "this part of the application has its own model, its own logic, and its own understanding of the world." it allows teams to work independently, prevents the 'spaghetti code' effect where changes in one place unexpectedly break something else, and generally keeps your sanity intact. it sounds simple enough in theory but, in practice, it requires a good understanding of the business domain. so here's how i'd approach it for a small e-commerce setup, and what i've learned by doing it poorly in the past:

first, let’s ditch the “one big blob” approach. that's a recipe for disaster. for a small e-commerce system, i’d recommend starting with these as the main bounded contexts:

**1. catalog:**

this one is pretty straightforward. it's all about managing products. the product details, categories, pricing, descriptions, images, all that good stuff. this context doesn’t necessarily need to know about user accounts or orders. it just knows about products. i once worked on a system where product attributes were intertwined with order processing logic – it was a mess. updates to product details could break the checkout flow, it was a bad time. keeping catalog separate is key. the data might look something like this:

```json
{
    "product_id": "12345",
    "name": "awesome t-shirt",
    "description": "a really cool t-shirt",
    "price": 25.00,
    "category": "apparel",
    "images": ["image1.jpg", "image2.jpg"]
}
```
notice that i am not adding anything about users or orders here. catalog is self contained.

**2. inventory:**

closely related to the catalog but still distinct. it deals with stock levels, where items are physically stored, and probably has integration with the warehouse or stock systems (if you are dealing with physical products). the reason to separate inventory from the catalog is that inventory changes frequently (sales, returns, restocks) while product details do not. mixing both causes constant updates to the catalog, and constant problems. this data might look like:

```json
{
    "product_id": "12345",
    "location": "warehouse a",
    "stock_level": 50,
     "reserved": 10
}
```

**3. customers:**

this one manages user accounts. it deals with usernames, passwords, addresses, order history, payment methods, all the sensitive user data. customer context is about the user relationship with the shop. this is where you want to enforce strict data privacy rules, so having it separate makes the security auditing process much easier. in my early days, i merged user data with order data and that was not a great idea. security auditing became a nightmare and queries were very hard to write and debug. the data might be something like this:

```json
{
    "user_id": "user123",
    "username": "johndoe",
    "email": "john@example.com",
    "billing_address": {
      "street": "123 main st",
      "city": "anytown"
    }
    "shipping_address": {
      "street": "456 oak ave",
      "city": "anycity"
      }
    "payment_methods": [
      {
        "type": "credit_card",
        "card_number": "xxxx-xxxx-xxxx-1234"
      }
    ]
}

```

**4. orders:**

this deals with the order lifecycle. creating orders, handling payment processing, shipping notifications, order status updates, and all that jazz. this context doesn't need to know the full product details (it just needs the product id), and it doesn't need to know users password. it needs to know the shipping address, and payment information and this information can be duplicated from the user context. that is why bounded contexts communicate in specific ways. this might look like:

```json
{
    "order_id": "order987",
    "user_id": "user123",
    "order_date": "2024-01-02",
    "shipping_address": {
      "street": "456 oak ave",
      "city": "anycity"
      },
    "items": [
    {
       "product_id": "12345",
       "quantity": 1,
       "price": 25.00
    }
   ],
   "total": 25.00,
   "status": "pending"
}
```

**5. payments:**

this is another logical separation which is a sub-domain of the orders context, but because payments deal with sensitive data and are a complex domain it makes sense to treat it as its own context. processing transactions, handling refunds, communicating with payment gateways and recording these transactions. it's a critical piece. my experience says that keeping payments separate is a good move for security and also for clarity. the data might be something like:

```json
{
  "payment_id": "payment777",
  "order_id": "order987",
  "payment_date": "2024-01-02",
  "payment_method": "credit_card",
  "amount": 25.00,
  "status": "success"
}
```

**communication between contexts:**

now, here’s where it gets interesting. these contexts aren't isolated silos. they need to communicate with each other. the key here is to define *how* they communicate. we’re talking about well-defined interfaces. you don't want one context directly manipulating data in another. think of it as each context having a ‘service layer’ that exposes a limited set of operations that can be performed within their boundaries. this can be http apis, message queues, or whatever you choose. for example, the orders context might call the catalog context to get product details when building an order, but it would never change product prices directly. it also might call the payment service to process a transaction, and it might communicate with inventory to mark stock as reserved.

**why this works:**

*   **clear ownership:** each context has a dedicated team (even if it is just you), making it easier to manage and make changes.
*   **reduced complexity:** changes within one context are less likely to break other parts of the system.
*   **scalability:** you can scale different contexts independently based on their resource needs. if the catalog needs more horsepower because of many visits, you can scale that part without scaling the payment processing.
*   **technology diversity:** each context can, in theory, use a different technology stack. although i would recommend starting with the same one for simplicity.

**things to avoid:**

*   **shared database:** don't let different contexts directly read and write to the same database tables. use the interfaces we talked about.
*   **leaky abstractions:** keep context models isolated. don’t let implementation details leak. for example, the customer context should not expose the raw password hash to other contexts.
*   **trying to be too perfect:** bounded contexts are not carved in stone. they might evolve as the system evolves. it is about finding the practical boundaries in your system.

**resources:**

instead of just dropping some links, let me recommend a couple of real books that helped me clarify these concepts (because reading them is usually better than just googling around):

*   "domain-driven design: tackling complexity in the heart of software" by eric evans. this book is the bible for all things ddd. it’s very theoretical but it’s worth its weight in gold.
*   "implementing domain-driven design" by vaughn vernon. this is more hands-on and practical. it really helped me translate the theory from evans book into real software.

i once, and i hate to say it, built a system where the entire application was a single big context. you want to know how many developers it took to change a single css rule? three. one to make the change, one to merge the change (god bless git), and one to revert the change because the initial change broke the database. it is funny in retrospective but it was not funny at the time.

so, yeah, that's my take on bounded contexts for a small e-commerce system. start with clear, well-defined boundaries. focus on the business domain. keep your interfaces simple and clear, and avoid common pitfalls. good luck and happy coding.
