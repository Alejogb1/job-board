---
title: "What are the best practices for using Entity Framework Core with Domain-Driven Design?"
date: "2024-12-23"
id: "what-are-the-best-practices-for-using-entity-framework-core-with-domain-driven-design"
---

Okay, let's tackle this. I’ve seen my share of developers, myself included initially, stumble when trying to marry the elegance of domain-driven design (ddd) with the practicality of entity framework core (ef core). It's a powerful combination, but it requires careful consideration to avoid common pitfalls. I recall a particularly frustrating project a few years back where we tried to shoehorn our entire domain model into ef core entities directly. Let's just say the performance suffered, and the maintenance became a nightmare. That experience really drove home the need for clear separation and well-defined boundaries.

At its core, the challenge lies in the fact that ddd focuses on modeling the *business* domain, whereas ef core is an *infrastructure* concern, specifically around data persistence. Therefore, best practices center around preventing persistence concerns from leaking into your domain model. Here are the key areas where I've found a solid approach to be indispensable:

**1. Domain Model Isolation:** This is paramount. Your domain model should be completely ignorant of ef core, or any other persistence mechanism. It should be composed of plain c# objects (poco), domain entities, value objects, domain services, and aggregates. These should represent your business logic and rules, not database tables. The domain model should not have any attributes or methods that are directly related to ef core (such as navigation properties for database relationships).

The first mistake I often see is direct association of ef core's attributes within domain entities. Here's a contrasting example showing 'what *not* to do' and 'a better approach':

```csharp
// Incorrect - Domain entity directly influenced by EF Core
public class IncorrectOrder
{
    public int OrderId { get; set; } //database id leaking to domain
    public int CustomerId { get; set; } // foreign key directly exposed
    public DateTime OrderDate { get; set; }
   public List<IncorrectOrderItem> OrderItems { get; set; } //navigation property.
}

public class IncorrectOrderItem {
    public int OrderItemId { get; set; }
    public int OrderId { get; set; }
    public int ProductId { get; set; }
    public int Quantity { get; set; }
}

// Correct - Pure Domain entity

public class Order
{
  public OrderId Id { get; private set; } // domain id
    public CustomerId CustomerId { get; private set; }
    public OrderDate OrderDate { get; private set;}
     private readonly List<OrderItem> _orderItems = new List<OrderItem>();
    public IReadOnlyCollection<OrderItem> OrderItems => _orderItems.AsReadOnly();
  public Order (OrderId id, CustomerId customerId, OrderDate orderDate)
    {
        Id = id;
        CustomerId = customerId;
        OrderDate = orderDate;
    }
  public void AddOrderItem(OrderItem item)
    {
    _orderItems.Add(item);
    }
    // other domain logic
}

public class OrderItem
{
    public ProductId ProductId { get; private set;}
    public int Quantity { get; private set;}
     public OrderItem(ProductId productId, int quantity)
    {
      ProductId = productId;
        Quantity = quantity;
    }
    // domain logic here
}
public record OrderId (Guid Value); // value object pattern for IDs
public record CustomerId (Guid Value);
public record ProductId (Guid Value);
public record OrderDate(DateTime Value);
```

Notice how the 'Correct' example doesn't expose database ids directly but uses value objects like `OrderId` and `ProductId`. Also it uses IReadOnlyCollection to prevent modification from external classes which enforces better domain control. The domain model is focused solely on business logic, completely decoupled from the underlying data storage. The 'Incorrect' example directly maps to how a database table might look, directly leaking persistence concerns.

**2. Persistence Ignorance with Repository Pattern:** To bridge the gap between your domain and ef core, implement the repository pattern. Repositories should expose an interface defined within your *domain* layer and the implementation lives within your *infrastructure* layer (where ef core is used). This allows the application layer to interact with data through the abstraction defined by the interface, without knowing about the specifics of data access.

This snippet shows what an interface for an order repository would look like in the domain layer and its implementation using ef core in the infrastructure layer.

```csharp
//Domain layer
public interface IOrderRepository
{
    Task<Order> GetByIdAsync(OrderId id);
    Task AddAsync(Order order);
    Task UpdateAsync(Order order);
    Task DeleteAsync(OrderId id);
}

//infrastructure layer - using ef core

public class EfOrderRepository : IOrderRepository
{
 private readonly ApplicationDbContext _context;

   public EfOrderRepository(ApplicationDbContext context)
    {
      _context = context;
    }

    public async Task<Order> GetByIdAsync(OrderId id)
    {
         var efOrder = await _context.Orders.Include(o => o.OrderItems).FirstOrDefaultAsync(e => e.Id == id.Value);
       if(efOrder is null)
          return null;

        // Map database object to domain object.
       var orderItems = efOrder.OrderItems.Select(i => new OrderItem(new ProductId(i.ProductId), i.Quantity)).ToList();
       var order = new Order(new OrderId(efOrder.Id), new CustomerId(efOrder.CustomerId), new OrderDate(efOrder.OrderDate));
       foreach(var orderItem in orderItems)
       {
          order.AddOrderItem(orderItem);
       }
        return order;

    }

    public async Task AddAsync(Order order)
    {

        var efOrder = new EfOrder()
        {
          Id = order.Id.Value,
            CustomerId = order.CustomerId.Value,
            OrderDate = order.OrderDate.Value,

        };
       var efOrderItems = order.OrderItems.Select(item => new EfOrderItem
       {
          ProductId = item.ProductId.Value,
          Quantity = item.Quantity,
          OrderId = order.Id.Value
       }).ToList();
       efOrder.OrderItems = efOrderItems;

        _context.Orders.Add(efOrder);

        await _context.SaveChangesAsync();
    }
    public async Task UpdateAsync(Order order)
    {

       var efOrder = _context.Orders.Include(o => o.OrderItems).FirstOrDefault(e=> e.Id == order.Id.Value);
       if (efOrder == null)
          return;

        efOrder.OrderDate = order.OrderDate.Value;
      var efOrderItems = order.OrderItems.Select(item => new EfOrderItem
      {
        ProductId = item.ProductId.Value,
        Quantity = item.Quantity,
        OrderId = order.Id.Value
      }).ToList();

      //simple update strategy.
      _context.OrderItems.RemoveRange(efOrder.OrderItems);
        efOrder.OrderItems = efOrderItems;

      await _context.SaveChangesAsync();

    }

     public async Task DeleteAsync(OrderId id)
    {
         var efOrder = await _context.Orders.FindAsync(id.Value);
       if(efOrder is null)
          return;

        _context.Orders.Remove(efOrder);
        await _context.SaveChangesAsync();
    }


}


 // EF Core Entities - these should ideally be in a different project/assembly than the domain

 public class EfOrder
{
    public Guid Id { get; set; }
    public Guid CustomerId { get; set; }
    public DateTime OrderDate { get; set; }
     public List<EfOrderItem> OrderItems { get; set; }
}

 public class EfOrderItem
{
    public int Id { get; set; }
    public Guid ProductId { get; set; }
    public int Quantity { get; set; }
   public Guid OrderId { get; set; }
}


```

Here, `EfOrderRepository` is specific to ef core, handling the actual data persistence. The `IOrderRepository` interface in the domain layer defines the contract for data access. Notice the use of ef core's `Include` to eager load related data, which could be a performance issue if used excessively. I tend to prefer lazy loading or using a separate query to load navigation properties. This also demonstrates the essential mapping of ef core entities (e.g. `EfOrder`) to domain entities (`Order`).

**3. Mapping between Entities and Persistence Models:** Since your domain entities shouldn't be ef core entities directly, you’ll need a mapping strategy. This usually involves creating separate ef core entities (like `EfOrder` and `EfOrderItem` in my example), which represent your database structure. Then, you'll need mapping code to transform between domain and persistence entities when retrieving or storing data. I find this a crucial step because failing to do this often results in persistence concerns bleeding into the domain model, which causes issues in the long run. This mapping is usually handled within repository implementation.

This final snippet shows how to handle mapping of related entities.

```csharp
// Inside EfOrderRepository - see previous snippet
// mapping of ef order to domain order
        var efOrder = await _context.Orders.Include(o => o.OrderItems).FirstOrDefaultAsync(e => e.Id == id.Value);
       if(efOrder is null)
          return null;

        // Map database object to domain object.
       var orderItems = efOrder.OrderItems.Select(i => new OrderItem(new ProductId(i.ProductId), i.Quantity)).ToList();
       var order = new Order(new OrderId(efOrder.Id), new CustomerId(efOrder.CustomerId), new OrderDate(efOrder.OrderDate));
       foreach(var orderItem in orderItems)
       {
          order.AddOrderItem(orderItem);
       }
        return order;
 //mapping domain order to ef order
 var efOrder = new EfOrder()
        {
          Id = order.Id.Value,
            CustomerId = order.CustomerId.Value,
            OrderDate = order.OrderDate.Value,

        };
       var efOrderItems = order.OrderItems.Select(item => new EfOrderItem
       {
          ProductId = item.ProductId.Value,
          Quantity = item.Quantity,
          OrderId = order.Id.Value
       }).ToList();
       efOrder.OrderItems = efOrderItems;

```

This example shows the mapping in both directions. Notice how domain value objects are transformed to database types (such as guid from `OrderId`) and back again.

**Recommended Resources**

For a deeper dive, I strongly recommend exploring the following resources:

*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans:** This is the foundational text on ddd and a must-read for anyone implementing ddd principles.
*   **"Implementing Domain-Driven Design" by Vaughn Vernon:** A practical guide that expands upon Evans' concepts and includes code examples to understand the practical implementations of DDD.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** Offers valuable insight into various architectural patterns, including the repository pattern, which plays a crucial role in achieving separation of concerns.
*  **Microsoft Documentation on Entity Framework Core:** Keep up to date with changes to EF Core, as knowing the ins and outs of its functionality is essential.

By adhering to these practices – particularly the strict separation of concerns, use of the repository pattern, and careful mapping – you can create robust, maintainable applications that leverage the strengths of both ddd and ef core, just as I have learnt to do from my experiences. These principles have served me well in numerous projects, and I hope they prove useful for you too. Remember that the goal is to make your domain model reflect your business, not your database.
