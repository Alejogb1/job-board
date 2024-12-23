---
title: "How to extract data from Domain-Driven Design entities/aggregates?"
date: "2024-12-23"
id: "how-to-extract-data-from-domain-driven-design-entitiesaggregates"
---

, let's tackle data extraction from DDD entities and aggregates. I've seen this trip up many teams, particularly when they're transitioning to a more domain-centric approach. The heart of the issue isn't the data itself, but how we conceptualize its access in relation to the domain model. It's about preserving the integrity of your aggregates and avoiding the dreaded "anemic domain model". Over my years, I've encountered projects that fell into this pitfall, and I've developed some patterns that tend to work pretty reliably.

The fundamental principle here is to avoid directly exposing the internal state of your entities. Remember, your entities and aggregates are meant to encapsulate business logic and enforce domain invariants. Directly accessing their fields from outside breaks this encapsulation, leading to inconsistent states and brittle code. So, the challenge is to extract the data *without* compromising the encapsulation that DDD strives for. We need to think of it as exposing a carefully considered view, not a raw data dump.

One approach I’ve found effective is using *accessor methods*. These aren't just simple getters, though. They are *domain-specific* methods that return only the relevant pieces of data for a particular use case. The key is to tailor these methods to the specific contexts where you need data, rather than exposing everything all at once. For instance, consider an `Order` aggregate. Instead of exposing all internal fields, you might have an `getOrderSummary()` method, which returns a DTO containing only order ID, customer details, and total amount. This way, you avoid direct access to things like the internal list of order items, which should be managed through domain methods like `addItem(OrderItem item)`. This approach is akin to having a well-defined interface, not just a raw data structure. This separation helps maintain the integrity of the business rules associated with order management and avoids accidentally corrupting the state of an order.

Here's a simple java example illustrating this concept:

```java
import java.util.List;
import java.util.UUID;

class Order {
    private UUID orderId;
    private String customerId;
    private List<OrderItem> items;
    private OrderStatus status;

    public Order(UUID orderId, String customerId) {
      this.orderId = orderId;
      this.customerId = customerId;
      this.status = OrderStatus.PENDING;
      this.items = new java.util.ArrayList<>();
    }


    public void addItem(OrderItem item) {
        this.items.add(item);
        // Other business logic involving adding an item
    }

    public OrderSummary getOrderSummary() {
      return new OrderSummary(orderId, customerId, calculateTotal());
    }

    private double calculateTotal() {
      double total = 0.0;
      for (OrderItem item: this.items) {
        total += item.getPrice() * item.getQuantity();
      }
      return total;
    }

    // Internal enum for Order status
    private enum OrderStatus {
      PENDING,
      SHIPPED,
      COMPLETED
    }

    public static class OrderItem {
      private double price;
      private int quantity;

      public OrderItem(double price, int quantity) {
        this.price = price;
        this.quantity = quantity;
      }
      public double getPrice() {
        return price;
      }
      public int getQuantity() {
        return quantity;
      }

    }

    public static class OrderSummary {
        private UUID orderId;
        private String customerId;
        private double totalAmount;

        public OrderSummary(UUID orderId, String customerId, double totalAmount){
          this.orderId = orderId;
          this.customerId = customerId;
          this.totalAmount = totalAmount;
        }

      public UUID getOrderId() {
        return orderId;
      }

      public String getCustomerId() {
        return customerId;
      }

      public double getTotalAmount() {
        return totalAmount;
      }
    }
}
```

In this example, the `getOrderSummary()` method constructs an `OrderSummary` DTO, exposing only the necessary information, while keeping the internal `items` list encapsulated within the `Order` aggregate. You'll notice that the `Order` does not have public getter methods for its internal fields. Instead, it exposes a `addItem()` method to modify the aggregate's internal state and `getOrderSummary` method for data extraction.

Another situation I've often encountered involves querying. You might need to retrieve collections of aggregates based on certain criteria. Directly exposing methods that return an entire collection of entities can quickly become a performance bottleneck, especially with large datasets. Here, employing a *read model* becomes invaluable. Read models are denormalized views of your data specifically designed for read operations, often stored in a different database optimized for querying. For example, instead of querying through an aggregate, you can query a read model that already contains a pre-calculated representation of the data.

Let me illustrate with a simplified scenario and code example. Let's assume you need to display a list of all active users in your application with their last login timestamp. Instead of directly querying user aggregates, you can project this information into a read model. The following C# example demonstrates this approach:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

public class User
{
    public Guid Id { get; }
    public string Username { get; }
    public bool IsActive { get; private set; }
    public DateTime LastLogin { get; private set; }


    public User(Guid id, string username)
    {
        Id = id;
        Username = username;
        IsActive = true;
        LastLogin = DateTime.MinValue;
    }

    public void LogIn()
    {
       LastLogin = DateTime.UtcNow;
    }

    public void Deactivate()
    {
      IsActive = false;
    }
}

public class UserReadModel
{
    public Guid Id { get; }
    public string Username { get; }
    public DateTime LastLogin { get; }

    public UserReadModel(Guid id, string username, DateTime lastLogin)
    {
        Id = id;
        Username = username;
        LastLogin = lastLogin;
    }
}

public class UserRepository
{
  private List<User> _users = new List<User>();

  public void Add(User user){
     _users.Add(user);
  }

  public User GetById(Guid id) {
     return _users.FirstOrDefault(user => user.Id == id);
  }

  public List<User> GetActiveUsers() {
    return _users.Where(user => user.IsActive).ToList();
  }


}

public class UserReadModelProjector
{
    public List<UserReadModel> Project(List<User> users)
    {
        return users.Where(user => user.IsActive)
                    .Select(user => new UserReadModel(user.Id, user.Username, user.LastLogin))
                    .ToList();
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        // Sample usage
        var userRepository = new UserRepository();
        var userProjector = new UserReadModelProjector();


        var user1 = new User(Guid.NewGuid(), "user1");
        var user2 = new User(Guid.NewGuid(), "user2");
        var user3 = new User(Guid.NewGuid(), "user3");
        user3.Deactivate();
        user1.LogIn();
        userRepository.Add(user1);
        userRepository.Add(user2);
        userRepository.Add(user3);

        var activeUsers = userRepository.GetActiveUsers();
        var userReadModels = userProjector.Project(activeUsers);

        foreach (var userReadModel in userReadModels)
        {
            Console.WriteLine($"User: {userReadModel.Username}, Last Login: {userReadModel.LastLogin}");
        }

    }
}
```

Here, `UserReadModelProjector` transforms `User` aggregates into `UserReadModel` instances, tailored for querying. The `UserRepository` is responsible for fetching the `User` aggregates, but the projection of the required data is handled separately. This separates the query responsibility from the aggregate. This approach decouples the write side (your domain aggregates) from the read side, and is a foundational aspect of CQRS (Command Query Responsibility Segregation).

Finally, let’s consider reporting scenarios. Often, you need to create analytical reports based on the data within your aggregates. Directly querying your write database for these reports can lead to performance problems and contention. For such cases, *materialized views* or data warehouses are often a more suitable choice. These mechanisms create pre-calculated data sets optimized for reporting, eliminating the need to perform complex computations each time a report is generated. I had an e-commerce system where we used materialized views to pre-aggregate sales data for the dashboard. This allowed us to load the dashboard extremely fast and without burdening the main order processing system.

Here's a simplified python example to give you a sense of how one might structure a function to transform and create such data for downstream analysis, although not a full fledged data warehouse.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Product:
  id: int
  name: str
  price: float

@dataclass
class OrderItem:
  product: Product
  quantity: int

@dataclass
class Order:
    id: int
    order_date: datetime
    items: list[OrderItem]

def calculate_sales_data(orders):
    sales_data = {}
    for order in orders:
        for item in order.items:
            product_id = item.product.id
            if product_id not in sales_data:
                sales_data[product_id] = {
                    "product_name": item.product.name,
                    "total_quantity": 0,
                    "total_revenue": 0,
                }

            sales_data[product_id]["total_quantity"] += item.quantity
            sales_data[product_id]["total_revenue"] += item.product.price * item.quantity
    return list(sales_data.values())


# Sample usage

products = {
    1: Product(id=1, name="Laptop", price=1200.00),
    2: Product(id=2, name="Mouse", price=25.00),
    3: Product(id=3, name="Keyboard", price=75.00)
}

order1 = Order(id=1, order_date=datetime.now(), items=[OrderItem(product=products[1], quantity=1), OrderItem(product=products[2],quantity=2)])
order2 = Order(id=2, order_date=datetime.now(), items=[OrderItem(product=products[2], quantity=1), OrderItem(product=products[3], quantity=3)])
orders = [order1, order2]
sales_report = calculate_sales_data(orders)
for item in sales_report:
    print(f"Product:{item['product_name']}, Total Quantity: {item['total_quantity']}, Total Revenue: {item['total_revenue']}" )

```

Here `calculate_sales_data()` does not access the underlying aggregate directly, it takes a list of `Order` objects and processes them into a more report-friendly format. This type of approach precomputes and structures data for reporting purposes and this data could be fed into a more traditional data warehousing approach.

In summary, extracting data from DDD entities and aggregates isn't about circumventing the domain model; it’s about providing a controlled, intentional, and context-aware view into the underlying data. Prioritize domain specific accessor methods for specific data requirements. Embrace read models for efficient querying.  Use materialized views or data warehouses for your complex reporting needs. For a deeper understanding of these patterns, I would suggest exploring the work of Eric Evans in "Domain-Driven Design: Tackling Complexity in the Heart of Software," specifically the chapters on Aggregates and Repositories. Also, consider reading "Implementing Domain-Driven Design" by Vaughn Vernon for practical implementation guidance. This will provide a more robust understanding of the why behind these strategies, and not just the how. By adopting these patterns, you’ll build a system that's not only functional but also maintainable and scalable in the long run.
