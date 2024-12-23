---
title: "How can I call methods from multiple Ruby classes?"
date: "2024-12-23"
id: "how-can-i-call-methods-from-multiple-ruby-classes"
---

Okay, let's explore this. Having dealt with my fair share of complex Ruby projects over the years, the need to invoke methods across various classes pops up frequently. It's a core aspect of object-oriented design, and Ruby provides several elegant mechanisms to achieve this. I remember one particularly challenging project, a complex e-commerce platform where disparate functionalities had to interoperate smoothly; that's where I really solidified my understanding of this topic. We had classes managing user accounts, product inventory, and order processing – all independent but needing to communicate through method calls. There's no single 'best' way; the approach often depends on the specifics of your problem and how tightly coupled you need the classes to be. Let’s delve into some common strategies, and I’ll illustrate each with code.

Firstly, the most straightforward way is direct method invocation, often facilitated through object instantiation and calling methods directly on those instances. This is suitable when you have a clear relationship between classes, maybe a "has-a" relationship. For example, let's say you have an `Order` class that utilizes a `PaymentProcessor`:

```ruby
class PaymentProcessor
  def process_payment(amount)
    puts "Processing payment of $#{amount}"
    true  # Simulate successful payment
  end
end

class Order
  def initialize(total_amount)
    @total_amount = total_amount
    @payment_processor = PaymentProcessor.new
  end

  def checkout
      if @payment_processor.process_payment(@total_amount)
        puts "Order completed successfully."
      else
        puts "Payment failed."
      end
  end
end

order = Order.new(100)
order.checkout
```

Here, the `Order` class creates an instance of `PaymentProcessor` directly within its constructor and calls the `process_payment` method. This approach works well for clearly defined dependencies where you have a class actively using another. The downside is tight coupling; the `Order` is explicitly bound to `PaymentProcessor`. Changing the payment logic would require altering `Order`, which might not be ideal for larger projects.

Now, let’s consider a slightly more decoupled scenario using dependency injection. This involves passing dependencies into a class instead of creating them internally. This is significantly more flexible. Let's modify our example:

```ruby
class PaymentProcessor
  def process_payment(amount)
    puts "Processing payment of $#{amount}"
    true # Simulate success
  end
end

class Order
  def initialize(total_amount, payment_processor)
    @total_amount = total_amount
    @payment_processor = payment_processor
  end

  def checkout
      if @payment_processor.process_payment(@total_amount)
        puts "Order completed successfully."
      else
        puts "Payment failed."
      end
  end
end


payment_processor = PaymentProcessor.new
order = Order.new(100, payment_processor)
order.checkout
```

Notice the change: now the `Order` class receives a `PaymentProcessor` object as an argument to its constructor. This is dependency injection. The `Order` is no longer creating the payment processor internally, and you can swap it out easily by injecting a different implementation if necessary. This loose coupling is vital for testability and maintainability. For a deep dive into dependency injection principles, Martin Fowler's articles on this subject are a solid resource; he offers profound and practical explanations. His work, like “Inversion of Control Containers and the Dependency Injection pattern,” is quite seminal in this area.

The third common approach uses modules and mixins to add functionality to classes from outside. Mixins allow a class to adopt methods from a module, giving it features it wouldn’t otherwise have. I used this a lot when needing to share common functionalities across otherwise unrelated classes. Here's an illustration:

```ruby
module Logging
    def log_message(message)
        puts "[LOG] #{Time.now}: #{message}"
    end
end

class User
    include Logging
    def register(username)
        log_message("User #{username} registered")
        puts "User #{username} registered successfully."
    end
end

class Product
    include Logging
    def add_to_inventory(product_name, quantity)
        log_message("Product #{product_name} added to inventory - Quantity: #{quantity}")
        puts "#{quantity} #{product_name} added to inventory."
    end
end

user = User.new
user.register("johndoe")
product = Product.new
product.add_to_inventory("Laptop", 10)
```

Here, both the `User` and `Product` classes include the `Logging` module. Each then gains the `log_message` method, showcasing how modules enable shared behavior across classes. Mixins are potent tools for code reuse and modularity. If you want a more in-depth understanding of Ruby’s module system and metaprogramming, I recommend reading "Programming Ruby 1.9 & 2.0: The Pragmatic Programmers' Guide," specifically the sections on modules and mixins. That book was instrumental when I was getting to grips with more advanced Ruby techniques.

It is critical to note that the correct method call pattern largely depends on the specifics of your application. Direct invocation is acceptable in tightly controlled, small-scale projects. But as your codebase grows, employing techniques like dependency injection and mixins becomes increasingly crucial for creating testable, maintainable, and extensible software. Over years, I’ve learned that thinking about object relationships upfront saves substantial refactoring time later on. While these three examples cover the basic approaches, there are additional patterns such as using callbacks or observer pattern for loosely coupled communication between objects, but these generally add more complexity. In most situations, using direct method invocation with proper object relationships (as exemplified), dependency injection for greater flexibility, or mixin based shared functionality as demonstrated, will cover the majority of required methods calls between Ruby classes.

These techniques are not just abstract theoretical concepts; they are practical solutions you will use every day as your project matures. The code examples I've provided, along with the resources suggested, should provide you with a comprehensive understanding. The focus remains that choosing the best approach is context-dependent, and a solid grasp of these techniques will serve you well when tackling complex projects, enabling you to write efficient and scalable Ruby applications.
