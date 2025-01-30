---
title: "How can .NET types be shared across multiple owned types?"
date: "2025-01-30"
id: "how-can-net-types-be-shared-across-multiple"
---
The challenge of sharing .NET types across multiple owned types frequently surfaces when designing complex object models where direct inheritance isn't suitable or desired. This scenario requires careful consideration of type accessibility and dependency management. A core concept to address this is the strategic use of interfaces and composition. Interfaces define contracts that multiple classes can implement, allowing these classes to share a common set of behaviors and properties without being bound by a single inheritance hierarchy. Composition, on the other hand, involves creating classes that hold instances of other classes as members. This approach fosters flexibility and reduces tight coupling.

In my previous project involving a large-scale inventory management system, we grappled with precisely this issue. We had several distinct entities, such as 'Product,' 'Location,' and 'Order,' all requiring common properties like timestamps for creation and modification. Initially, we considered inheritance. However, creating a base class for all of these entities felt inappropriate and quickly led to an unwieldy class hierarchy, violating the Single Responsibility Principle. We needed a way to share specific functionalities without forcing classes into an unsuitable inheritance model. That's when we shifted our focus to interfaces and composition.

Specifically, we defined an interface, `IHasTimestamps`, which declared the `CreationDate` and `ModificationDate` properties. Any entity requiring these properties could then implement this interface. This is significantly different from inheritance where the implementation is passed to the child class. Instead, each class implementing the interface is responsible for its specific implementation of those properties, providing the needed flexibility.

The implementation of an interface in a class declares a contract. This means the implementer must satisfy that contract; the contract is not satisfied for the implementer automatically. The implementation of the properties and methods are left to the class, which provides maximum flexibility for different implementations. A class that implements an interface and provides an implementation of the methods and properties is said to fulfill or to satisfy the interface.

Here’s the first code example demonstrating the usage of the `IHasTimestamps` interface:

```csharp
public interface IHasTimestamps
{
    DateTime CreationDate { get; set; }
    DateTime ModificationDate { get; set; }
}

public class Product : IHasTimestamps
{
    public string Name { get; set; }
    public decimal Price { get; set; }
    public DateTime CreationDate { get; set; }
    public DateTime ModificationDate { get; set; }
}

public class Location : IHasTimestamps
{
    public string Address { get; set; }
    public string City { get; set; }
    public DateTime CreationDate { get; set; }
    public DateTime ModificationDate { get; set; }
}

public class Order
{
  public string OrderNumber { get; set; }
  public DateTime OrderDate { get; set; }
}
```

In this example, both `Product` and `Location` classes implement the `IHasTimestamps` interface, providing their concrete implementation of the properties defined in the interface. The `Order` class does not implement `IHasTimestamps`. They now share that functionality, with complete independence in all other aspects. We have not had to create a base class or to use inheritance at all.

Now, let's move to how we addressed the handling of common functionality, particularly the setting of modification timestamps. To avoid repetitive code in the classes themselves, we created a helper class that used the interface as a type constraint. This demonstrates the value of a shared behavior that can be applied to multiple, different concrete classes.

```csharp
public static class TimestampHelper
{
  public static void UpdateModificationDate<T>(T entity) where T : IHasTimestamps
  {
     entity.ModificationDate = DateTime.UtcNow;
  }

  public static void SetCreationDate<T>(T entity) where T : IHasTimestamps
  {
     entity.CreationDate = DateTime.UtcNow;
     entity.ModificationDate = DateTime.UtcNow; //Set both the first time
  }
}
```

The `TimestampHelper` class offers static methods that operate on any type `T` that implements `IHasTimestamps`. This generic constraint prevents this helper class from being incorrectly used with a class that does not implement the contract. I have found this approach to be incredibly helpful, promoting both code reuse and testability. This demonstrates how interfaces allow for common actions to be taken on different data types.

We also tackled more complex scenarios where a collection of owned objects required access to shared logic or properties of a parent class. In this case, composition became our go-to. Imagine a scenario with a `Customer` class owning multiple `Order` objects. The `Order` objects might need to access information or functionalities residing in the `Customer` class but should not inherit from it, as this would again create tight coupling.

Here's the approach:

```csharp
public class Customer
{
    public string CustomerId { get; set; }
    public string CustomerName { get; set; }
    public List<Order> Orders { get; } = new List<Order>();

  public void AddOrder(Order newOrder)
  {
    newOrder.SetCustomer(this); //Pass the parent to the child.
    Orders.Add(newOrder);
  }

}

public class Order : IHasTimestamps
{
    public string OrderNumber { get; set; }
    public DateTime OrderDate { get; set; }
    public Customer Customer {get; private set;}
    public DateTime CreationDate { get; set; }
    public DateTime ModificationDate { get; set; }

    public void SetCustomer(Customer customer)
    {
      this.Customer = customer;
    }
  }
```

In this example, the `Order` class holds a reference to its owning `Customer` object. The `Customer` class provides a method to create the `Order`, thus allowing the `Order` to be aware of its parent. This allows the `Order` object to, for example, access customer-specific configuration data without creating a direct coupling between the two classes. We passed the instance to the owned class to avoid tight coupling and to maintain a sense of ownership. This promotes a more modular and easily testable design.

The choice between interfaces and composition depends heavily on the specific requirements of the system being built. Interfaces are excellent for defining contracts that multiple, unrelated types can implement, promoting polymorphism and decoupling. Composition is the preferred choice for situations where a class needs to access functionalities or data from its parent or owner, fostering modular design and maintainability. Avoid passing dependencies unnecessarily; only do so if there is a strong reason for this relationship to exist.

My experience has shown that strategically using both interfaces and composition allows for significant flexibility when dealing with common types and behaviors in .NET. By adhering to principles like the Single Responsibility Principle, and carefully considering the responsibilities of each class, developers can build more maintainable, scalable, and robust software.

For further study, I would recommend exploring resources related to object-oriented design principles, particularly those dealing with interface segregation, dependency injection, and composition vs. inheritance. In the book "Patterns of Enterprise Application Architecture," Martin Fowler presents several patterns to address these needs. The book "Design Patterns: Elements of Reusable Object-Oriented Software" is another valuable resource. Furthermore, I highly advise studying the documentation for dependency injection frameworks, as they provide a practical implementation of these concepts, allowing classes to be decoupled. Understanding these fundamental design elements is paramount for building robust and maintainable systems.
