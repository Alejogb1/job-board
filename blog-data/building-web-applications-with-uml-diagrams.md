---
title: "building web applications with uml diagrams?"
date: "2024-12-13"
id: "building-web-applications-with-uml-diagrams"
---

 so you wanna build web apps using UML diagrams right I’ve been there done that and got the T-shirt and maybe even a few scars so let's break this down in a way that hopefully makes sense.

First off I’m not gonna sugarcoat it using UML directly to build web apps isn't the most common path nowadays it's more of a design tool than a code generation tool and in my experience it can be kinda clunky if you try to make it do more than it's meant to but don't get me wrong it has its place mainly in the initial planning stage I guess

I remember this one project back in the day when I was at this startup doing some e-commerce site we had a bunch of backend guys who were all into UML and they went hog wild designing the whole database schema class diagrams sequence diagrams all that jazz at the start of the project It was beautiful on paper looked like a work of art but then it was a nightmare to translate it to actual code everyone interpreted the diagrams differently which led to a lot of time wasted trying to fix these misunderstandings in design and in implementation phase we basically ended up spending a couple of weeks trying to reverse engineer code from UML diagrams it was not fun believe me

So here's my take on how you might actually approach this now in 2024

**1 Design with UML but don't become too obsessed**

UML is great for visualizing your application's structure especially when you are working on more complex projects Class diagrams for object modeling are super helpful you know when defining your entities and their relationships like users products shopping carts things like that and sequence diagrams they're invaluable to show the flow of actions in your application when designing API calls or database interactions I usually use some tools like diagrams net its a good open source online tool for diagrams

**Example UML Class Diagram Fragment (Not Code but a representation of design)**

```text
   +---------------------+       +---------------------+
   |       User         |       |       Product       |
   +---------------------+       +---------------------+
   | - userId: int      |       | - productId: int   |
   | - username: string |       | - name: string      |
   | - email: string    |       | - price: float     |
   +---------------------+       +---------------------+
         |                  1..*        |
         |   user-orders    -------->    |
         +---------------------+       +---------------------+
         |      Order        |       |   CartItem       |
         +---------------------+       +---------------------+
         | - orderId: int     |       | - itemId: int      |
         | - orderDate: date  |       | - quantity: int    |
         | - total: float     |       | - product: Product |
         +---------------------+       +---------------------+

```

So what I'm saying is the diagram above should illustrate what you want to achieve with the code and should be used to translate that to the code for example the User class should map into a class in the code like this

```python
#Example code for backend in python
class User:
    def __init__(self, user_id, username, email):
        self.user_id = user_id
        self.username = username
        self.email = email

class Product:
    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        self.price = price

class Order:
    def __init__(self, order_id, order_date, total, user: User):
        self.order_id = order_id
        self.order_date = order_date
        self.total = total
        self.user = user

class CartItem:
    def __init__(self, item_id, quantity, product:Product):
        self.item_id = item_id
        self.quantity = quantity
        self.product = product


```

Now this python code shows how a representation of the UML is translated into code that we can use in our web application

**2 Translate UML to Code by hand**

The real work starts when you move from these nice diagrams to actual code I always tend to use them as a base not as the definitive source of truth that should generate all my code Instead take each component and map it to your specific needs. I've noticed that it's far more useful to do some mental work thinking more like a software engineer instead of just translating UML to code word for word.

For the front end its different but similar use UML class diagrams to design components and interactions its not really different than the backend at all

Here's an example of a front end component and its representation in React

```jsx
//Example code for frontend in react
import React, { useState } from 'react';

function ProductCard({ product }) {
    const [quantity, setQuantity] = useState(1);

    const handleAddToCart = () => {
        // Logic to add item to cart, you know, API call stuff
        console.log(`Added ${quantity} of ${product.name} to cart`);
    };

    return (
        <div className="product-card">
            <h3>{product.name}</h3>
            <p>Price: ${product.price.toFixed(2)}</p>
            <input
                type="number"
                value={quantity}
                min="1"
                onChange={(e) => setQuantity(parseInt(e.target.value))}
            />
            <button onClick={handleAddToCart}>Add to Cart</button>
        </div>
    );
}

export default ProductCard;
```

The `ProductCard` component would probably be mapped as a UML class in the class diagram and its logic with API calls would translate into the sequence diagram its not that difficult its just a little time consuming

**3 Don't try to generate code from UML**

 here's where I draw the line and this is where some developers mess up because they believe in this idea so much, there are some UML tools that promise code generation and in my experience they often fall short they are not smart enough to map your UML to the complexities of the web frameworks the real world uses like Spring Boot or React you know there's a lot of stuff you need to take care like security and persistence and those frameworks have all that so if you try to do this with code generated from UML you're probably going to have a hard time.

I mean I once saw a guy try to generate a complex Spring application from a UML model. Let's just say it didn't end well. He spent more time debugging the generated code than it would have taken to write it from scratch. It's better to write the code yourself I mean that's why you became a programmer right?

**4 Learn Frameworks and Patterns**

Don’t rely on UML as the definitive method to define your code learn frameworks like Spring Boot for backend React or Vuejs for the front end and also learn good design patterns like MVC or microservices those are more important than a bunch of UML diagrams because frameworks and patterns will give you a more structured approach than raw UML diagrams to building the architecture of your web application believe me.

So in this journey of using UML to build web applications you should consider UML as a tool to give you some guidance not the way to do everything. Think of it as a way to brainstorm your idea. I’d also recommend some good resources for that you know there are some good books out there

*   "Head First Design Patterns" this is a very good and easy to read book that can give a very good start to learn about patterns.
*   "Domain-Driven Design: Tackling Complexity in the Heart of Software" this will give you some insights into how to structure your application in a good way and this book is really useful to learn how to actually use the UML in a software design perspective.
*   "Clean Architecture" its a very good book to structure your web applications in a more logical way and more framework agnostic way.

To sum it up UML is great for initial design but you have to know when to transition to code and how to write code properly use UML as a base you know to avoid going blindly.

One last thing I’ve learned over the years is this “Good coders code great developers solve problems” so you have to see the big picture and not get stuck in the details. I hope that helps you get started and have fun building your application and good luck
