---
title: "How can implementation and fields be shared?"
date: "2024-12-23"
id: "how-can-implementation-and-fields-be-shared"
---

Alright, let’s talk about sharing implementations and fields, a topic that’s often more complex than it initially appears. I’ve seen this trip up even the most seasoned developers, usually when projects start scaling and the initial design choices, which seemed perfectly reasonable at the time, suddenly become painful bottlenecks. There are several ways to approach this, and the "best" method really depends on the specific context of your project. I’ll break down a few techniques I've employed successfully over the years, avoiding the pitfalls that I've personally stumbled into.

One common approach, and where many start, involves object composition and interfaces. Think of it like lego bricks. Each object, or 'brick', should have its own encapsulated data and behavior, and then the application is built by combining these pieces. Interfaces then define a contract, dictating what each brick can do, rather than how. I distinctly remember a project several years back where we were building a system for processing financial transactions. We had several different transaction types (credit, debit, transfers), and each needed a unique validation procedure. Rather than repeat the validation logic in each class, we extracted an `IValidatable` interface with a single `validate()` method. Then, each transaction type had a custom validator object that implemented this interface. This allowed us to reuse the core validation structure without the coupling.

Here’s how that might look in code. Let’s take C# for example:

```csharp
public interface IValidatable
{
  bool Validate();
}

public class CreditTransactionValidator : IValidatable
{
   private readonly CreditTransaction _transaction;

   public CreditTransactionValidator(CreditTransaction transaction)
   {
      _transaction = transaction;
   }

   public bool Validate()
   {
        // Credit specific validation logic.
        if(_transaction.Amount <= 0) return false;
        //additional checks...
        return true;

   }
}

public class DebitTransactionValidator : IValidatable
{
   private readonly DebitTransaction _transaction;

   public DebitTransactionValidator(DebitTransaction transaction)
    {
       _transaction = transaction;
    }
   public bool Validate()
    {
       // Debit specific validation logic.
        if(_transaction.Amount > _transaction.AccountBalance) return false;
        //additional checks...
        return true;
    }
}

public class CreditTransaction
{
    public decimal Amount { get; set;}
}

public class DebitTransaction
{
    public decimal Amount { get; set;}
    public decimal AccountBalance {get; set;}
}
```
This snippet shows two different validator classes implementing the same interface, each operating on different types and with different validation logic. The crucial aspect here is that other parts of the system can operate on *any* object implementing `IValidatable`, without knowing the specifics of *how* the validation is done. This demonstrates that we're sharing the functionality (the 'validate' action) through an interface and not the underlying implementation, which is unique per transaction. This reduces duplication and makes the system more extensible.

Another approach to sharing implementation is through mixins or traits, although this mechanism is language-dependent. Some languages, such as Python, support mixins directly. Other languages, while not having direct support, can achieve similar results through patterns such as extension methods or composition. Mixins allow you to "inject" behavior into classes, which is very beneficial when you have code that you want to share between multiple classes but inheritance hierarchies become problematic. Think of a logging functionality that you may want to add to almost any class within your application. You wouldn’t necessarily want all of your classes inheriting from a ‘loggable’ class; this is where mixins shine.

I once worked on a system for managing complex business rules where logging each step was crucial for debugging and auditing. Using mixins, we could easily add consistent logging to any class that needed it, avoiding a tangled mess of inheritance or excessive code duplication.

Let's explore a simplified Python example using mixins:

```python
class Loggable:
    def log(self, message):
        print(f"Log: {message}")

class Order(Loggable):
    def __init__(self, order_id, customer_id):
      self.order_id = order_id
      self.customer_id = customer_id

    def process_order(self):
        self.log(f"Processing order {self.order_id} for customer {self.customer_id}")
        # Actual order processing logic here

class Customer(Loggable):
    def __init__(self, customer_id, name):
        self.customer_id = customer_id
        self.name = name
    def update_address(self, address):
        self.log(f"Updating address for customer {self.customer_id}")
        # Address updating logic here

order = Order(123,"customer_a")
order.process_order()
customer = Customer(456, "customer_b")
customer.update_address("New Address")
```

Here, both the `Order` and `Customer` classes gain the logging functionality by inheriting from the `Loggable` mixin, showcasing how shared functionality is introduced without requiring these classes to have a common root class.

Lastly, a third approach involves the use of utility classes or static methods. These classes can contain common algorithms or functions that are shared among multiple classes. This is particularly useful for stateless operations. For example, date manipulation, string formatting or calculations that are independent of a class's internal state are good candidates for static methods or utility class functions. When I was building a data analysis pipeline, I remember we had a great deal of data cleansing required across several modules. We opted to create a static ‘DataCleaner’ class where we put all the string parsing, and format functions, enabling us to call these operations consistently across modules.

Here's an example illustrating a static utility class:

```java
public class StringUtilities {
    public static String formatPhoneNumber(String phoneNumber) {
      //Implement a phone number formatting logic, such as stripping any characters apart from digits
        return phoneNumber.replaceAll("[^0-9]", "");
    }

    public static String capitalizeFirstLetter(String input) {
        if (input == null || input.isEmpty()) {
            return input;
        }
        return input.substring(0, 1).toUpperCase() + input.substring(1);
    }
}

public class User {
   private String _phone;
   private String _name;
   public User(String name,String phone){
     _name = name;
     _phone = phone;
   }

   public String getFormattedPhone(){
      return StringUtilities.formatPhoneNumber(_phone);
   }

   public String getFormattedName(){
      return StringUtilities.capitalizeFirstLetter(_name);
   }

}
```

In this Java example, `StringUtilities` holds two static methods used by the `User` class to standardize phone numbers and names. This illustrates how these static utilities can be easily invoked without an instantiated object of the utility class. This promotes code reuse without introducing the complexities of inheritance or dependency injection.

To delve deeper into these concepts, I'd strongly recommend exploring the Gang of Four's book, "Design Patterns: Elements of Reusable Object-Oriented Software", which offers an in-depth look at design principles and patterns, particularly regarding composition and interface-based design. For those interested in mixins and traits, reading up on object-oriented programming in languages like Python, Ruby, or Scala is beneficial. “Programming in Scala” by Martin Odersky, Lex Spoon, and Bill Venners provides detailed insights into trait usage. Regarding utility classes and static methods, the "Effective Java" series by Joshua Bloch offers solid guidelines for proper usage.

In summary, sharing implementation and fields effectively isn't about finding a single "silver bullet" solution. Instead, it's about choosing the right approach based on your specific needs and understanding the tradeoffs of each. Interfaces, mixins, and utility classes each offer a unique way of sharing functionality, each with its particular strengths. Through judicious design and a clear understanding of these techniques, one can build robust, maintainable and scalable applications. My experiences, both successes and failures, have solidified the importance of this foundational concept in good software architecture.
