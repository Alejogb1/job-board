---
title: "domain class software design pattern?"
date: "2024-12-13"
id: "domain-class-software-design-pattern"
---

Alright so domain class software design pattern right yeah I've wrestled with that beast more times than I care to remember Let me tell you it's not just about slapping some classes together and calling it a day It's about crafting a structure that's not only technically sound but also reflects the actual business domain you're working with I'm thinking back to this one project a CRM system I had to build from scratch a few years back We started off with what we thought was a pretty good object-oriented design you know entities and services and the whole nine yards But things quickly got messy It became a tangled mess of code updates to one thing would ripple through seemingly unrelated parts of the system it was awful debugging felt like a guessing game And we were dealing with data like customers products orders you know the usual suspects But it was scattered all over the place like a digital explosion that's when we realized we hadn't really understood our domain fully We were treating everything like general-purpose data instead of specific domain concepts Like an address wasn't just a string it had its own rules for formatting validation it wasn't just a column in the customers table

So here's the lowdown when you talk about domain class software design patterns you're basically talking about crafting classes that represent the core concepts of your problem space These aren't just data containers they are active agents in your system each with its own responsibilities and behaviours They encapsulate the logic and rules related to that specific concept think of them as mini experts in their own domain That CRM I mentioned yeah we ended up refactoring it quite a bit adopting a more domain-centric approach we introduced specific classes for `Customer` `Product` `Order` and `Address` etc Each of these classes contained their respective logic for example `Customer` had methods for changing addresses and adding preferences all business logic related to customers that wasn't just getters and setters

Now for the technical nitty-gritty The key is to identify entities value objects and aggregates that are relevant to your domain Entities are objects that have an identity meaning two entities are different even if their properties are the same for example two customers are different people even if they live in the same address and have the same name Value objects don't have an identity they only care about the value itself an example is address an address is the same address if it contains the same state city street number etc Aggregates are a collection of entities that act as a unit for example a product with different product options can be an aggregate and these things are not just academic terms it's how you represent your domain

Let's dive into some examples using Python because I've been using it lately:

First example a simple `Product` class:

```python
class Product:
    def __init__(self, product_id, name, price, description):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.description = description

    def apply_discount(self, discount_percentage):
        if 0 <= discount_percentage <= 100:
            self.price = self.price * (1 - discount_percentage/100)
        else:
            raise ValueError("Discount percentage must be between 0 and 100")

    def __str__(self):
       return f"Product ID: {self.product_id} Product Name: {self.name} Price: {self.price}"


product = Product(123,"Laptop", 1200.00, "Gaming Laptop")
print (product)
product.apply_discount(10)
print (product)
```

This is a basic entity with basic methods for discounts and representation notice that the discount method takes care of applying that to the object and not just return a value it is encapsulating its behaviour as well

Second Example lets say an `Address` Value Object:

```python
class Address:
    def __init__(self, street, city, state, zip_code):
        self.street = street
        self.city = city
        self.state = state
        self.zip_code = zip_code

    def __eq__(self, other):
        if isinstance(other, Address):
            return (self.street == other.street and
                    self.city == other.city and
                    self.state == other.state and
                    self.zip_code == other.zip_code)
        return False

    def __hash__(self):
        return hash((self.street, self.city, self.state, self.zip_code))

    def __str__(self):
        return f"{self.street} {self.city} {self.state} {self.zip_code}"


address1 = Address("123 Main St", "Anytown", "CA", "91234")
address2 = Address("123 Main St", "Anytown", "CA", "91234")
address3 = Address("456 Oak Ave", "Otherville", "NY", "54321")

print(f"Address 1: {address1}")
print(f"Address 2: {address2}")
print(f"Address 3: {address3}")
print (address1 == address2)
print (address1 == address3)
```

This has a more complex implementation and is different from the product class this is because it represents a value object notice that the `__eq__` and `__hash__` methods are overridden to provide equality based on value not object identity

Third example lets say a `Customer` Entity that uses the previous `Address` value object:

```python
class Customer:
    def __init__(self, customer_id, name, email, shipping_address):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.shipping_address = shipping_address

    def update_shipping_address(self, new_address):
       if not isinstance(new_address,Address):
           raise TypeError("New address must be an Address object")
       self.shipping_address = new_address

    def __str__(self):
        return f"Customer ID: {self.customer_id} Customer Name: {self.name} Email: {self.email} Address: {self.shipping_address}"

customer = Customer(1, "John Doe", "john.doe@email.com", address1)
print (customer)
customer.update_shipping_address(address3)
print (customer)

```
This shows an entity and how it uses a value object and encapsulates logic and its not just getter and setters like I've mentioned before

The whole point of domain-driven design DDD principles is to shift the focus from purely technical concerns to the actual business problem you are trying to solve This has a big advantage that the domain classes are reusable in different applications if needed it's about creating a model of the domain that is easy to understand for the business people not just developers because after all they are the ones that know the ins and outs of the system and not the developer always

I've found that when you are using a layered architecture with domain objects you tend to end up with a cleaner design where the user interface layer only knows about the service or application layer which interacts with the domain layer and the data persistence layer interacts with the domain layer as well. This decoupling makes the code easier to maintain and test It also allows a modular design which promotes reuse and prevents unexpected ripple effects when changing code

Oh and I almost forgot there was this one time where my domain classes were so tightly coupled that changing the way a customer address was formatted required me to rewrite about half my codebase it was really bad I was just thinking "Well this is just great". Don't be like me learn from my mistakes

So if you are looking for resources besides the usual searches I would recommend the classic "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans it's like the bible for this stuff also "Implementing Domain-Driven Design" by Vaughn Vernon is also a good choice for more practical stuff and also a good read

To summarise keep it simple model your real world problem not an abstract model and encapsulate behaviour in your domain objects and remember to keep your classes cohesive and decoupled you don't want to end up rewriting half your codebase because of some silly error it’s not fun trust me I've been there and I'm sure there are others out there that also know the feeling of rewriting a massive codebase because of a simple oversight

Well that's it for me I've been using this for quite a while now and it has saved me a lot of debugging time I hope that helps I’m out
