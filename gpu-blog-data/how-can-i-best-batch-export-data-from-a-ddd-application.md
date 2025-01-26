---
title: "How can I best batch export data from a DDD application?"
date: "2025-01-26"
id: "how-can-i-best-batch-export-data-from-a-ddd-application"
---

The inherent challenge in batch exporting data from a Domain-Driven Design (DDD) application lies in balancing performance and domain integrity. Large-scale data extraction, especially when numerous aggregates are involved, can easily overwhelm resources and bypass carefully crafted domain logic, potentially leading to inconsistencies. My experience working on a large e-commerce platform highlighted these issues acutely, and the solutions I implemented centered around a combination of architectural patterns and technology choices.

The core issue stems from the impedance mismatch between how data is typically structured in a DDD application, with its focus on rich domain objects and aggregates, and the flattened, normalized structures often required for efficient exports. Trying to directly map aggregates to CSV or JSON formats, especially with nested relationships, results in inefficient queries and costly transformations. Therefore, a dedicated read model, separate from the write model, is essential for optimized export operations.

My initial approach, using the primary repository directly for export, resulted in severe performance degradation during peak hours. I was retrieving large sets of `Order` aggregates, each potentially pulling associated `OrderItem` and `ShippingAddress` entities, leading to N+1 query problems and excessive database load. The subsequent data transformations, mapping these rich domain objects into a more flattened structure, further compounded the inefficiency. It became apparent that the read model for export needed to be optimized for this particular use case.

The first architectural pattern I implemented was the **Command Query Responsibility Segregation (CQRS)**, specifically separating the read side from the write side at the data access layer. This allowed us to build read models that were specifically tailored to the needs of the data export process, independent of the write model. These read models were denormalized views, designed for fast querying with minimal transformations necessary for export. For example, instead of fetching all associated entities of an `Order` aggregate, the read model for order export contained only the necessary attributes of these entities, flattened into a single structure. This reduced the number of joins and related queries drastically, resulting in faster data retrieval.

Next, I incorporated the **Data Transfer Object (DTO)** pattern to further abstract the export process from the underlying database schema. While the read models provided an optimized query interface, the data they returned was still bound to the specifics of the database structure. I introduced DTOs which were simple objects containing only the data necessary for the export format. This separation allowed us to easily adapt our export formats without having to modify the read models. It also provided a natural place to implement any necessary value conversions or specific formatting, ensuring that the exported data was exactly as required.

Finally, I tackled the challenge of batch processing. Instead of exporting the entire dataset at once, I implemented a **cursor-based pagination** system on the read model query. This allowed the export process to fetch data in manageable chunks, preventing memory exhaustion and database overloads, especially with large datasets. Each fetch would return a limited number of DTOs, along with a cursor indicating the position for the next chunk. These chunks were then passed to a service responsible for writing the output, allowing the export to proceed incrementally and reliably.

Here are three code examples illustrating these points:

**1. Read Model Interface and Implementation:**

```csharp
public interface IOrderExportReadModel
{
    IEnumerable<OrderExportDto> GetOrdersForExport(int batchSize, string cursor);
}

public class OrderExportReadModel : IOrderExportReadModel
{
    private readonly IDbConnection _connection;
    public OrderExportReadModel(IDbConnection connection)
    {
        _connection = connection;
    }

    public IEnumerable<OrderExportDto> GetOrdersForExport(int batchSize, string cursor)
    {
      // SQL query with cursor pagination for optimized performance
      // Instead of selecting all at once, we select a batch of OrderExportDto based on cursor
      string query = $@"SELECT
                       o.OrderId,
                       o.OrderDate,
                       c.CustomerId,
                       c.Email,
                       SUM(oi.Quantity * oi.Price) as TotalAmount
                       FROM Orders o
                       JOIN Customers c ON o.CustomerId = c.CustomerId
                       JOIN OrderItems oi ON o.OrderId = oi.OrderId
                       WHERE o.OrderId > @Cursor
                       GROUP BY o.OrderId, o.OrderDate, c.CustomerId, c.Email
                       ORDER BY o.OrderId
                       LIMIT @BatchSize";
        using var connection = _connection;
        return connection.Query<OrderExportDto>(query, new { BatchSize = batchSize, Cursor = cursor}).ToList();
    }
}
```
*This example demonstrates a read model interface and a concrete implementation that leverages optimized SQL queries with cursor-based pagination. This approach avoids loading all data into memory at once.*

**2. Data Transfer Object (DTO) Definition:**

```csharp
public class OrderExportDto
{
    public int OrderId { get; set; }
    public DateTime OrderDate { get; set; }
    public string CustomerId { get; set; }
    public string Email { get; set; }
    public decimal TotalAmount { get; set; }
}
```
*This shows a simple DTO tailored for exporting order data. It's decoupled from database entities and the Domain Model, focusing solely on the data required for export.*

**3. Batch Export Service with cursor-based pagination:**

```csharp
public class OrderExportService
{
    private readonly IOrderExportReadModel _orderReadModel;
    private readonly IExportWriter _exportWriter;

    public OrderExportService(IOrderExportReadModel orderReadModel, IExportWriter exportWriter)
    {
        _orderReadModel = orderReadModel;
        _exportWriter = exportWriter;
    }

    public void ExportOrders(int batchSize)
    {
         string currentCursor = "0";
         bool hasMoreData = true;

         while (hasMoreData)
         {
            var batch = _orderReadModel.GetOrdersForExport(batchSize, currentCursor);
            if (!batch.Any())
            {
               hasMoreData = false;
               continue;
            }

            _exportWriter.Write(batch);
            currentCursor = batch.Last().OrderId.ToString();
         }
    }
}
```
*This example demonstrates a service using the read model and a writer service. It shows how cursor pagination is handled, and how the process can incrementally write data in chunks.*

For further study, I would recommend exploring the principles outlined in *Patterns of Enterprise Application Architecture* by Martin Fowler. Understanding the use of mapping patterns and database access strategies outlined within is invaluable for implementing efficient data export. For further reading on CQRS, explore the resources provided by Greg Young. Also, gaining familiarity with database query optimization specific to your chosen database engine (SQL Server, MySQL, PostgreSQL, etc.) is crucial for refining the performance of read models. Finally, understanding principles of bulk processing, such as in the *Enterprise Integration Patterns* book by Hohpe and Woolf is helpful in architecting this type of application. By focusing on separation of concerns and optimizing for read performance with these approaches, achieving efficient and domain-consistent data exports from a DDD application becomes significantly more achievable.
