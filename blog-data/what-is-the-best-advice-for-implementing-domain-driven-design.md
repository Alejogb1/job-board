---
title: "What is the best advice for implementing Domain Driven Design?"
date: "2024-12-23"
id: "what-is-the-best-advice-for-implementing-domain-driven-design"
---

Alright, let's unpack this domain-driven design (ddd) question. It's not a one-size-fits-all kind of deal, and the "best" advice really hinges on context, team dynamics, and the problem at hand. I’ve seen teams try to apply ddd dogmatically and end up creating more complexity than they solved, so let's try to be nuanced about it. I've personally had my share of scrapes with this over the years, from startups where we were basically sprinting from idea to deployment, to larger enterprises where the organizational inertia made even the simplest changes feel like moving mountains.

The first, and perhaps most crucial, piece of advice I can give you is: don't start with code. I know, it sounds counterintuitive in our line of work. But ddd isn't primarily about creating classes and methods; it's about understanding the business domain. Before you even think about a `repository` or a `service`, you need a solid grasp of what the business actually *does*. I remember working on an early project for a logistics company. We started with the technical architecture and it ended in a total mess. We were building something theoretically sound, but had fundamentally misunderstood the core process of how packages moved. It wasn't until we had a series of intensive workshops with the actual dispatchers, managers, and drivers, and diagrammed the whole thing on whiteboards, that things started to click. That means talking with subject matter experts, drawing out process flows, and identifying core concepts in the business.

Once you’ve gained a solid grasp of the domain, the next key step is identifying your *bounded contexts*. A bounded context essentially represents a specific area of the application where a particular model holds true and consistent meaning. Don’t think of your application as a monolithic entity. Break it down. This is where domain experts become invaluable. For example, in that logistics project, we identified separate bounded contexts for ‘inventory management,’ ‘dispatch management,’ ‘billing,’ and ‘customer service,’ each with their own language and models. Trying to force all of these into one single, god-like context results in a confusing, coupled mess. The goal here is to delineate clearly where concepts from one context do *not* directly translate into another. This reduces cognitive load and allows for more focused development.

Now, for those code snippets. Let’s focus on implementing aggregate roots and value objects, which are critical for modeling our bounded contexts.

**Snippet 1: Value Object Example**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Address:
    street: str
    city: str
    zip_code: str
    country: str

    def __str__(self):
        return f"{self.street}, {self.city}, {self.zip_code}, {self.country}"

    # Note: Value objects are immutable, meaning operations return a *new* instance if data needs modification
    def with_updated_street(self, new_street):
        return Address(new_street, self.city, self.zip_code, self.country)

# Usage
address1 = Address("123 Main St", "Anytown", "12345", "USA")
address2 = address1.with_updated_street("456 Oak Ave") # Creates new instance, address1 remains unchanged
print(address1)
print(address2)

```

This is a simple Python example, but the concept applies across languages. The `Address` object is a value object; it's identified by its attributes, not its identity. It's also immutable, meaning that once created, it can't be changed. If you need a slightly different address, you create a *new* address object instead. This immutability promotes easier reasoning about your code and better concurrency handling.

**Snippet 2: Aggregate Root Example**

```java
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

class Order {
    private UUID orderId;
    private Customer customer; // Aggregate reference
    private List<OrderItem> orderItems;
    private String status;


    public Order(Customer customer) {
        this.orderId = UUID.randomUUID();
        this.customer = customer;
        this.orderItems = new ArrayList<>();
        this.status = "PENDING";
    }

    public void addOrderItem(OrderItem item) {
        this.orderItems.add(item);
        // other important domain logic might go here
    }

    public void completeOrder() {
        if(status.equals("PENDING")) {
             this.status = "COMPLETED";
        }else {
             throw new IllegalStateException("Cannot complete order in state: " + status);
        }
    }
    public String getStatus(){
      return status;
    }

    public List<OrderItem> getOrderItems() {
       return this.orderItems;
    }
    public UUID getOrderId(){
        return orderId;
    }
}

class OrderItem {
    private String itemName;
    private int quantity;
    private double price;

    public OrderItem(String itemName, int quantity, double price){
        this.itemName = itemName;
        this.quantity = quantity;
        this.price = price;
    }

    public String getItemName(){
        return itemName;
    }
    public int getQuantity(){
        return quantity;
    }
    public double getPrice(){
        return price;
    }

}

class Customer {
  private UUID customerId;
    private String name;
  public Customer(String name){
    this.customerId = UUID.randomUUID();
    this.name = name;
  }
  public UUID getCustomerId(){
    return customerId;
  }
  public String getName(){
    return name;
  }
}

// Usage
public class Main {
 public static void main(String[] args) {
    Customer customer = new Customer("Jane Doe");
    Order order = new Order(customer);
    order.addOrderItem(new OrderItem("Laptop", 1, 1200.00));
    order.addOrderItem(new OrderItem("Mouse", 1, 25.00));
    System.out.println("Order status: " + order.getStatus());
    order.completeOrder();
    System.out.println("Order status: " + order.getStatus());

   System.out.println("Order items: "+ order.getOrderItems().size() );
  }
}


```

Here, the `Order` is an aggregate root. It controls access to its internal state and manages the lifecycle of `OrderItem` entities, which don’t have a standalone existence in the system. You notice the `Customer` is an entity, referenced but not owned by the `Order`. This maintains proper boundaries and ensures that the state transitions within an `Order` remain consistent, making the aggregate a transactional boundary. You wouldn't manipulate the `orderItems` from outside the `Order` class. This is important for maintaining data integrity and consistency.

**Snippet 3: Domain Service Example**
```csharp
using System;
using System.Collections.Generic;

public class ShippingAddress
{
    public string Street { get; set; }
    public string City { get; set; }
    public string ZipCode { get; set; }
}
public class Customer{
   public Guid CustomerId {get;set;}
  public string Name {get;set;}
  public ShippingAddress ShippingAddress {get;set;}
}
public class Order
{
    public Guid OrderId { get; set; }
    public Customer Customer {get;set;}
    public List<OrderItem> OrderItems { get; set; }

    public string Status { get; set; }

    public Order(Customer customer)
    {
        this.OrderId = Guid.NewGuid();
        this.Customer = customer;
        this.OrderItems = new List<OrderItem>();
        this.Status = "PENDING";
    }
}
public class OrderItem{
  public string ItemName{get;set;}
  public int Quantity {get;set;}
  public decimal Price {get;set;}
}

public class OrderProcessingService
{
  private IShippingCalculator _shippingCalculator;
  public OrderProcessingService(IShippingCalculator shippingCalculator){
    _shippingCalculator = shippingCalculator;
  }

  public void ProcessOrder(Order order)
  {
       var shippingCost = _shippingCalculator.CalculateShipping(order.Customer.ShippingAddress);
       Console.WriteLine($"Shipping cost for order {order.OrderId}: ${shippingCost}");
       // some other processing like payment
    order.Status = "PROCESSED";
    Console.WriteLine($"Order {order.OrderId} processed");
  }
}
public interface IShippingCalculator{
   decimal CalculateShipping(ShippingAddress address);
}
public class BasicShippingCalculator: IShippingCalculator {
    public decimal CalculateShipping(ShippingAddress address){
    //Basic logic just adds flat shipping fee for now
      return 5.00m;
    }
}
// Usage
public class Program
{
    public static void Main(string[] args)
    {
       ShippingAddress address = new ShippingAddress(){
         Street="123 main",
         City="Anytown",
         ZipCode="12345"
       };
        Customer customer = new Customer(){
          CustomerId = Guid.NewGuid(),
          Name = "John Doe",
          ShippingAddress=address
        };
        Order order = new Order(customer);
        order.OrderItems.Add(new OrderItem(){ItemName="Laptop", Price=1200.00m, Quantity=1});
        order.OrderItems.Add(new OrderItem(){ItemName="Mouse", Price=25.00m, Quantity=1});
        IShippingCalculator shippingCalculator = new BasicShippingCalculator();
        OrderProcessingService orderProcessingService = new OrderProcessingService(shippingCalculator);
         orderProcessingService.ProcessOrder(order);
    }
}
```
In this c# example, `OrderProcessingService` is a domain service since `ProcessOrder` doesn't naturally belong to any particular entity or aggregate. It orchestrates interactions across aggregates, like fetching shipping cost before processing the payment. Note that domain service does not hold domain state; its behavior should operate based on domain objects' values, and the domain logic is injected into service via interface abstraction. This aligns with ddd principles because it keeps domain logic separate from technical considerations like how we compute a shipping cost.

Finally, constant communication with domain experts is paramount. DDD is a collaborative process, not a solitary coding exercise. Your model should be a reflection of their mental model. Don’t be afraid to refactor and iterate. Ddd is an adaptive process; your understanding of the domain will evolve, and your code should be able to accommodate those changes without causing a major rework.

For further reading, I recommend Eric Evans’ *Domain-Driven Design: Tackling Complexity in the Heart of Software*, which is a foundational text for DDD. Also, *Implementing Domain-Driven Design* by Vaughn Vernon provides practical guidance and examples. These should provide a solid foundation for understanding and applying DDD effectively. Remember that ddd is a journey, not a destination, and applying these principles in a sensible, context-aware manner will ultimately lead to better software.
