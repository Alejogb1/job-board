---
title: "How can I model direct SQL Server interactions in a DDD context?"
date: "2025-01-30"
id: "how-can-i-model-direct-sql-server-interactions"
---
A common challenge in Domain-Driven Design (DDD) arises when dealing with legacy systems or requirements that necessitate direct SQL Server interactions, bypassing the typical abstraction layers favored in a purely DDD approach. While DDD emphasizes domain logic over persistence concerns, practical situations often demand a pragmatic blend. I’ve encountered this issue several times in my career, particularly when optimizing performance-critical data operations. We can model these direct interactions by treating them as infrastructure concerns, separated from the core domain, and accessed through carefully defined interfaces.

The crucial aspect is to maintain the integrity of the domain model while acknowledging the reality of database-specific operations. In a strict DDD application, the domain layer should be completely unaware of the persistence mechanism – be it SQL Server, NoSQL databases, or even in-memory storage. However, direct SQL interactions often require knowledge of specific database features, such as stored procedures, functions, or query optimization techniques. We can address this by defining interfaces that abstract the data access behavior and implementing those interfaces with concrete classes that directly interact with SQL Server. The key separation is that the domain layer uses the interfaces (defined within the domain or application layers), never the concrete SQL Server implementation classes.

This design maintains the core DDD principle of keeping the domain pure and independent of external concerns, and it is achievable through a layered architecture. The typical application layer orchestrates domain interactions and utilizes infrastructure implementations behind an interface. This allows the domain model to remain unaware of the specific database being utilized. The infrastructure layer houses the implementations with direct SQL interaction. The application layer uses dependency injection to pass concrete implementations that act on the domain.

Let’s examine this approach with three code examples based on a simplified e-commerce context using C#. I'll focus on implementing SQL Server interactions for a `Product` entity. First, consider the domain layer. Here, the `Product` entity is a pure domain object with only properties relevant to the business. We define an interface `IProductRepository` to abstract the persistence of this domain object:

```csharp
// Domain Layer

public class Product
{
  public int ProductId { get; }
  public string Name { get; }
  public decimal Price { get; }

  public Product(int productId, string name, decimal price)
  {
      ProductId = productId;
      Name = name;
      Price = price;
  }
}

public interface IProductRepository
{
  Product GetById(int productId);
  void Save(Product product);
  IEnumerable<Product> GetAllByPriceRange(decimal minPrice, decimal maxPrice);
}
```

The domain layer is completely unaware of SQL. Now, let's move to the infrastructure layer and see how we implement the `IProductRepository` using direct SQL Server interactions:

```csharp
// Infrastructure Layer (SQL Server Implementation)

using System.Data.SqlClient;

public class SqlProductRepository : IProductRepository
{
  private readonly string _connectionString;

  public SqlProductRepository(string connectionString)
  {
      _connectionString = connectionString;
  }

  public Product GetById(int productId)
  {
    using (var connection = new SqlConnection(_connectionString))
    {
      connection.Open();
      using (var command = new SqlCommand("SELECT ProductId, Name, Price FROM Products WHERE ProductId = @ProductId", connection))
      {
        command.Parameters.AddWithValue("@ProductId", productId);
        using (var reader = command.ExecuteReader())
        {
          if (reader.Read())
          {
              return new Product(
                  (int)reader["ProductId"],
                  (string)reader["Name"],
                  (decimal)reader["Price"]);
          }
          return null;
        }
      }
    }
  }

  public void Save(Product product)
  {
    using (var connection = new SqlConnection(_connectionString))
    {
        connection.Open();
        using (var command = new SqlCommand("INSERT INTO Products (ProductId, Name, Price) VALUES (@ProductId, @Name, @Price)", connection))
        {
            command.Parameters.AddWithValue("@ProductId", product.ProductId);
            command.Parameters.AddWithValue("@Name", product.Name);
            command.Parameters.AddWithValue("@Price", product.Price);

            command.ExecuteNonQuery();
      }
    }
  }

  public IEnumerable<Product> GetAllByPriceRange(decimal minPrice, decimal maxPrice)
    {
      List<Product> products = new List<Product>();
      using (var connection = new SqlConnection(_connectionString))
      {
          connection.Open();
          using (var command = new SqlCommand("SELECT ProductId, Name, Price FROM Products WHERE Price >= @MinPrice AND Price <= @MaxPrice", connection))
          {
              command.Parameters.AddWithValue("@MinPrice", minPrice);
              command.Parameters.AddWithValue("@MaxPrice", maxPrice);
              using (var reader = command.ExecuteReader())
              {
                  while(reader.Read())
                  {
                      products.Add(new Product((int)reader["ProductId"], (string)reader["Name"], (decimal)reader["Price"]));
                  }
              }
          }
      }

        return products;
    }
}

```

In this example, the `SqlProductRepository` is a concrete implementation of the `IProductRepository` interface. It directly interacts with SQL Server using `SqlConnection` and `SqlCommand`, executing queries using ADO.NET. Crucially, the repository is responsible for mapping between the domain entity (`Product`) and the database representation.  The domain remains oblivious to these database details. Notice also how the database query details are handled within this concrete implementation, such as the actual SQL queries used, the parameterization strategy, and more complex database-specific code.

Finally, let’s consider how the application layer utilizes these components:

```csharp
// Application Layer

public class ProductService
{
  private readonly IProductRepository _productRepository;

  public ProductService(IProductRepository productRepository)
  {
    _productRepository = productRepository;
  }

  public Product GetProduct(int productId)
  {
    return _productRepository.GetById(productId);
  }

  public void CreateProduct(int productId, string name, decimal price)
  {
      var product = new Product(productId, name, price);
      _productRepository.Save(product);
  }

  public IEnumerable<Product> FindProductsByPrice(decimal minPrice, decimal maxPrice)
    {
        return _productRepository.GetAllByPriceRange(minPrice, maxPrice);
    }
}
```

In the `ProductService`, the dependency on a concrete implementation is resolved through dependency injection.  The service does not directly create the `SqlProductRepository`; it receives an `IProductRepository` implementation (which happens to be the SQL Server specific one in this case). This separation through interfaces allows the domain to remain agnostic and independent of the underlying database technology, and also enables easy replacement or testing with different persistence mechanisms. The `ProductService` focuses purely on business logic.

Key to this approach is the role of dependency injection to decouple the concrete repository from its use. The DI container is responsible for injecting the appropriate implementation of `IProductRepository` (e.g., the `SqlProductRepository`) into the `ProductService`. This makes the application highly testable. You can replace the SQL repository with an in-memory implementation for unit tests without any changes to the `ProductService`.

Some general recommendations to further enhance this pattern:

1. **Repository Pattern:** Strictly follow the repository pattern, focusing on abstracting persistence logic. This can involve creating generic repository interfaces that use domain entities and their IDs as parameters, allowing your business logic to be completely ignorant of persistence mechanisms.
2.  **Data Transfer Objects (DTOs):**  Consider using DTOs when data retrieved from the database does not precisely map to the domain entity. This pattern is helpful when projections or transformations are required before exposing data to the domain layer. This can also minimize exposing more database-centric details to the domain.
3. **Stored Procedures and Functions:** Wrap complex SQL operations within stored procedures or functions within the database. Call these from the repository implementations. This moves complicated queries and logic to the database server, taking advantage of its optimization capabilities, and making the .NET repository classes shorter and easier to understand.
4. **Connection Pooling:** Implement connection pooling strategies to optimize SQL Server interactions within the repository classes. This can have a performance impact when scaling.
5. **Error Handling:** Implement detailed error handling for any SQL Server exceptions, ensuring the application does not directly expose database specifics or break when SQL Server errors occur.

For further study on this topic, I recommend looking at resources covering advanced Domain-Driven Design patterns, focusing on repositories and persistence ignorance, as well as patterns like the use of DTOs. Furthermore, examining the SOLID principles, particularly the Dependency Inversion Principle, will help solidify an understanding of how to design these types of applications well. Books detailing best practices for using entity frameworks, or a focus on micro-ORM solutions such as Dapper, can also be very useful in achieving these goals effectively. While Entity Framework is more an ORM and not for direct SQL queries, they do work very well together, as you can use the entity framework for much of the data access, but also implement direct SQL for specific situations as described in this text.

This strategy allows me to effectively leverage direct SQL server interactions within a DDD context. By maintaining clear separation of concerns through layered architecture, interfaces, and dependency injection, the system preserves domain integrity while achieving the required performance and database specific features.
