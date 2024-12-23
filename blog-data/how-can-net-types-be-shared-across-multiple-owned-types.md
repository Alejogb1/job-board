---
title: "How can .NET types be shared across multiple owned types?"
date: "2024-12-23"
id: "how-can-net-types-be-shared-across-multiple-owned-types"
---

Alright, let’s tackle this. I’ve run into this situation quite a few times in my career, usually when dealing with complex domain models or systems that have grown organically over time, demanding a more granular, yet still cohesive design. Sharing types across multiple owned types in .NET isn't always straightforward, primarily because we need to be very clear about ownership, lifecycle, and mutability. It's not enough to just pass references around; that's a recipe for debugging nightmares. Let’s break down how I approach this, drawing from past experiences where things went sideways before I settled on these strategies.

The core challenge arises from the fact that in object-oriented programming, especially in the .NET framework, ownership implies certain responsibilities. If a class *owns* another, it usually controls its lifecycle – creating it, managing it, and disposing of it. But what happens when multiple classes should conceptually 'own' the same type or, rather, need access to it without full ownership? That’s where we need to think about shared access rather than full ownership.

One of the most common, and often most misused, solutions is simple property exposure. Let’s illustrate this. I recall working on an e-commerce system where we had `Order` and `Customer` classes. Initially, both classes needed access to `Address` data. We naively made the `Address` an owned property of both. The problem? Changes in one `Order`'s `Address` would unintentionally modify the `Customer`'s `Address` if we were not extremely careful with defensive copying, which of course, we weren't everywhere.

Here's a simplified (and problematic) representation of that situation:

```csharp
public class Address
{
    public string Street { get; set; }
    public string City { get; set; }
}

public class Customer
{
    public Address Address { get; set; }
    public Customer() {
      Address = new Address();
    }
}

public class Order
{
    public Address Address { get; set; }
    public Order() {
      Address = new Address();
    }
}

// This was the problem:
Customer customer = new Customer();
Order order = new Order();
customer.Address.Street = "Main Street";
order.Address = customer.Address; // Bad Idea! They share the same address.
order.Address.City = "New City";
// Now the customer's address city is also "New City".
```

This resulted in data corruption and was painful to debug. The "solution" to this problem was, and often is, defensive copying which can become a maintenance problem. It does not solve the underlying issue that different logical entities had the same logical entity as a private property.

The better approach involves decoupling ownership through a shared context. Instead of each type owning the object, we leverage composition with a shared service or data structure that manages the shared type. This is where patterns such as an aggregate root or even a simple service class come into play.

Let’s look at how I refactored that scenario. I moved the `Address` management to an `AddressService`:

```csharp
public class Address
{
    public string Street { get; set; }
    public string City { get; set; }
    // we've added a unique ID in order to look it up
    public Guid Id { get; set; }
}

public class AddressService
{
    private Dictionary<Guid, Address> _addresses = new Dictionary<Guid, Address>();

    public Address CreateAddress(string street, string city) {
        var address = new Address {
            Street = street,
            City = city,
            Id = Guid.NewGuid()
        };
        _addresses.Add(address.Id, address);
        return address;
    }

     public Address GetAddress(Guid id){
         if(_addresses.TryGetValue(id, out var address)){
             return address;
         }
        return null; // Or handle not found case appropriately
     }
}

public class Customer
{
    public Guid AddressId { get; set; }
    // customer would hold just the ID instead of Address object.

    public Customer(Address address){
        AddressId = address.Id;
    }
}

public class Order
{
    public Guid AddressId { get; set; }
    // order would hold just the ID instead of Address object

    public Order(Address address){
       AddressId = address.Id;
    }
}

// Usage now becomes:
var addressService = new AddressService();
var address1 = addressService.CreateAddress("Old Street", "Old City");
var customer = new Customer(address1);
var order = new Order(address1);
var retrievedAddressForCustomer = addressService.GetAddress(customer.AddressId);
var retrievedAddressForOrder = addressService.GetAddress(order.AddressId);

// Now we can modify the customer's or order's address, without affecting each other because they are using a copy
retrievedAddressForCustomer.City = "Updated Customer City";
// order Address doesn't change
```

Now, both `Customer` and `Order` classes reference the `Address` via its ID, and any updates are managed through the service. This approach decouples the types and ensures changes in one don't affect the other. It also centralizes address creation and retrieval, which is a definite benefit. In my experience, using a service class for this has helped me avoid a lot of potential issues and enforce consistency.

Another effective technique, especially in scenarios where the "owned" type is relatively small and immutable, is using a value object. If an address, for example, can be fully described by its properties and its identity is based entirely on its data rather than a separate identifier, we can treat it as a value object. This strategy removes the need for tracking identity and shared ownership becomes very simple: it's simply creating another instance with the same properties as required.

Here's a modified example using `record` which is ideal for a value type:

```csharp
public record Address(string Street, string City); // Address is now an immutable value object

public class Customer
{
    public Address Address { get; set; }
     public Customer(Address address)
     {
          Address = address;
     }

}

public class Order
{
    public Address Address { get; set; }
    public Order(Address address){
        Address = address;
    }
}

// And we use it like this:
var address1 = new Address("Main Street", "Old City");
var customer = new Customer(address1);
var order = new Order(address1);
var modifiedAddress = address1 with { City = "New City"}; // creating a new immutable address with City modified.
customer.Address = modifiedAddress;
// order's address is not modified because Address is a value type.
```

The key difference here is that `Address` is a `record`, making it immutable. When a new `Address` is needed, we create a new object with the relevant values; modifications aren’t applied in-place, preventing the shared object problem.

It's crucial to choose the correct method based on the needs of your specific application. For complex shared data with independent lifecycles, a service to manage that shared data is a robust solution. For small, immutable pieces of information, value objects offer a great advantage. Misusing these strategies can still lead to problems, so careful analysis of requirements is paramount.

For further exploration, I'd highly recommend reading Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software," particularly the chapters on entities, value objects, and aggregates. Also, Martin Fowler's "Patterns of Enterprise Application Architecture" offers incredibly useful insights into patterns for data management in complex systems. Finally, for a deeper understanding of how value objects are implemented and their benefits, consult the book "Implementing Domain Driven Design" by Vaughn Vernon; it offers a comprehensive discussion on this subject. These resources have been invaluable in my own practice, and I have a strong hunch that they’ll be useful for you too.
