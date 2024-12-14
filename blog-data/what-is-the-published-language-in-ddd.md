---
title: "What is the published language in DDD?"
date: "2024-12-14"
id: "what-is-the-published-language-in-ddd"
---

alright, so the question is about the "published language" in domain-driven design (ddd). i've spent a good chunk of my career elbow-deep in ddd projects, and this concept, the published language, it's absolutely fundamental. it's not about some specific programming language like python or java, instead it’s about the structured, shared vocabulary the team uses to communicate about the domain. it's the language we use in discussions, in code, in documentation, and in user stories. basically, it's how we talk and think about the problem at hand.

think of it like this: we're not just writing code, we are actually encoding our understanding of a specific problem space, and that understanding must be unified in the way we use language. if one person calls a user 'customer' and another calls it 'client', we have a problem. inconsistencies creep in, bugs appear, and we end up wasting time untangling stuff that shouldn't even be tangled in the first place.

now, from my experience on several projects, the published language is never just given to you. it's a discovery process. on one occasion i recall working for a bank on their payment system, we initially started using very generic terms like ‘transaction’ for everything. it turned out we had several types of transactions each with very different behavior, like internal transfers, credit card payments, and direct debits, etc. we ended up needing a much more specific terminology, including things like 'credit_transfer', 'card_payment' and 'direct_debit'. that was a real eye opener and it also slowed down the project due to the miscommunication. so the lesson i learned from this project, is that building a published language is not a one-time thing, it is a gradual iterative process that keeps evolving as the team's understanding of the domain deepens.

so, let’s be concrete. there isn't a formal specification for *how* to create a published language, it is more of a practice. however, there are very specific things we try to achieve with it. we want a language that is:

*   **ubiquitous:** everyone on the team, from the developers to the product owner, uses the same terms. no exceptions.
*   **precise:** each term has a very specific meaning and no room for ambiguity. we should be able to point to our domain model and see a one-to-one mapping with our language.
*   **consistent:** terms are used the same way across different parts of the system. if a customer is defined as an object with certain attributes in one part of the code, it is the same in all parts of the code.
*   **expressive:** it captures the richness and nuance of the domain. it should feel natural to use when discussing the domain, not some stilted set of terms.
*  **simple:** easy to use and understand, but simple doesn't mean generic. It is specific, yet easy to understand by anyone in the team, even the new joiners.

let me give you an example of how the published language might look in code. imagine an e-commerce domain. here's a simple python snippet using some core concepts:

```python
class Customer:
    def __init__(self, customer_id, name, email):
        self.customer_id = customer_id
        self.name = name
        self.email = email

class Product:
    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        self.price = price

class OrderItem:
    def __init__(self, product, quantity):
        self.product = product
        self.quantity = quantity

class Order:
    def __init__(self, order_id, customer, items):
        self.order_id = order_id
        self.customer = customer
        self.items = items
```

in this simplified code, we've got 'customer', 'product', 'order', and 'orderitem'. these terms should be exactly what the team uses, from discussions to the database schema. if someone says 'client', that would mean that they are not using the published language and the first thing that should come to mind is, why is this person not using it? and then to have a discussion with the entire team about it.

let’s take it further with another example. imagine we have a domain for a library system. here are some classes:

```python
class Book:
    def __init__(self, isbn, title, author):
        self.isbn = isbn
        self.title = title
        self.author = author

class Member:
    def __init__(self, member_id, name, address):
        self.member_id = member_id
        self.name = name
        self.address = address

class Loan:
    def __init__(self, book, member, loan_date, due_date):
        self.book = book
        self.member = member
        self.loan_date = loan_date
        self.due_date = due_date
```

here, 'book', 'member', and 'loan' are part of the published language. we are not using something like 'user' instead of 'member' or 'material' instead of 'book'. if we are consistently doing it this way, we are less prone to making errors and misunderstandings, at the design level and at the coding level.

another key point is that the published language isn't just about nouns (like 'customer' or 'product'). it includes actions too. if we are using an event driven architecture and our domain event is named 'order_placed' it is a part of the published language. if we use 'new_order_created' then we should know it right away we are inconsistent with the language. it also includes concepts like 'aggregate root' and 'value object' and all this ddd concepts that help us to structure our domain model.

let's add another example, this time showing how actions as a part of the published language could be used:

```python
class OrderService:
    def __init__(self):
        pass

    def place_order(self, customer, items):
      # some logic for placing an order
        order = Order(order_id = "some_id", customer=customer, items=items)
        print(f"order placed for {customer.name}")
        return order

    def fulfill_order(self, order_id):
        #some logic to fulfill the order
        print(f"order fulfilled for id {order_id}")


service = OrderService()

customer = Customer(customer_id=1, name="john doe", email="john.doe@mail.com")
product1 = Product(product_id=1, name="product a", price = 100)
order_item1 = OrderItem(product1, 2)

service.place_order(customer=customer, items=[order_item1])
service.fulfill_order("some_id")
```

here, `place_order` and `fulfill_order` are actions and, therefore, they are an essential part of the published language. this is a simplified example, but in real-world scenarios, these actions can be more complex and require a lot more domain knowledge. the key thing is to ensure that the actions, their names, and what they mean to the business is agreed by everyone.

one of the things that i found hard in the beginning of my ddd journey was getting stakeholders to use the published language. many times, i found them saying things like “i just want this functionality done”, and while that is valid, it is not very useful to establish a common ground. so, when this happened, i used to try to translate what they were saying to the published language. for example if a stakeholder says “i need a way to finalize the shopping cart” i would translate it to “we need a way to place an order”. so it is a continuous effort to get everyone in the same page.

in short, the published language is the shared vocabulary that allows us to effectively communicate about the domain and the model we're building. it's a crucial element of ddd, and not something that should be overlooked.

if you want to really go deep into this topic, there are a few resources that i can highly recommend:

*   **"domain-driven design: tackling complexity in the heart of software" by eric evans:** this is the bible of ddd. it covers the concept of the published language, or ubiquitous language as eric evans calls it, extensively. i keep going back to this book time and time again. it's a difficult book, i will be honest, but worth the effort.
*   **"implementing domain-driven design" by vaughn vernon:** this book provides a more practical guide on how to apply ddd principles, including the importance of the published language. it's a great companion to eric evans' book.
*   **various articles on domain-driven design:** there are a lot of good articles and blog posts that discuss aspects of ddd, including the published language, usually in a more concise format. i often use these to refresh the concepts and to get different perspectives on the topic. and last but not least, remember that the practice of ddd is not a sprint it is a marathon.

one last bit of advice i can give is to always iterate on your published language. it’s like a good wine it matures with time. as your domain understanding improves so should the published language. don't expect to get it perfect the first time. it's a continuous feedback cycle. and with that, i think i have covered most of what i wanted to say, except for the joke, i guess it was that last bit of advice, i mean, who expected a good wine analogy? i certainly didn't.

hope it helps.
