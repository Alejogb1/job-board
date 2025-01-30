---
title: "How can I avoid code duplication of identical models across multiple .NET EF Core contexts using Clean Architecture?"
date: "2025-01-30"
id: "how-can-i-avoid-code-duplication-of-identical"
---
The core issue with identical model duplication across multiple .NET EF Core contexts within a Clean Architecture stems from a violation of the Dependency Inversion Principle.  Specifically, the persistence layer is directly coupled to the domain model, making reuse challenging and introducing unnecessary maintenance overhead.  My experience resolving this in large-scale projects involved decoupling the domain model from specific EF Core contexts using a repository pattern and a shared library containing the core entities.

**1. Clear Explanation: Decoupling the Domain Model**

The solution hinges on separating the domain model – the representation of your business objects – from its persistence mechanism.  EF Core contexts are specifically tied to data access; they should not dictate the structure of your domain entities. Instead, create a separate library, a "Shared Kernel" if you will,  containing the pure domain models. These models should be devoid of any database-specific annotations or properties.  Only business logic and validation should reside within these classes.

Each EF Core context then interacts with the domain model through an abstraction layer – a repository.  The repository provides a contract for interacting with persistent data, hiding the specific implementation details. This allows you to switch persistence mechanisms (e.g., from EF Core to another ORM) without impacting the rest of your application.  Consequently, your contexts remain thin wrappers around database access, primarily responsible for managing connections and transactions, leaving the complex interactions with your domain models to the repository.

This approach adheres to Clean Architecture principles by clearly separating concerns. The domain layer (shared kernel) remains independent and can be reused across various contexts and projects. The infrastructure layer (containing EF Core contexts and repositories) becomes a specialized component that adapts the domain model to the persistent storage.

**2. Code Examples with Commentary**

**Example 1: Shared Kernel Model (Domain)**

```csharp
// SharedKernel.csproj  -  This project contains only the domain models.

namespace MyCompany.SharedKernel
{
    public class Product
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public decimal Price { get; set; }

        public Product() { } // Required for EF Core, despite being in the domain

        public Product(string name, decimal price)
        {
            Name = name;
            Price = price;
            // Add domain logic validation here.
        }
    }
}
```

This example shows a simple `Product` model residing in a separate shared kernel project.  Notice the absence of EF Core annotations like `[Key]` or `[DatabaseGenerated]`. The constructor demonstrates encapsulation, enforcing domain rules.


**Example 2: Repository Interface and Implementation**

```csharp
// Infrastructure.csproj -  This project houses the EF Core contexts and repositories

namespace MyCompany.Infrastructure.Repositories
{
    public interface IProductRepository : IRepository<Product> // Generic Repository Interface
    {
        // Add repository-specific methods if needed.  For example:
        Task<Product> FindByNameAsync(string name);
    }

    public class ProductRepository : IProductRepository
    {
        private readonly MyDbContext _context; // EF Core Context

        public ProductRepository(MyDbContext context)
        {
            _context = context;
        }

        public async Task<IEnumerable<Product>> GetAllAsync() => await _context.Products.ToListAsync();
        // ...other implementation details...
    }
}


// Generic Repository interface (can be placed in SharedKernel)

namespace MyCompany.SharedKernel
{
    public interface IRepository<T> where T : class
    {
        Task<IEnumerable<T>> GetAllAsync();
        Task<T> GetByIdAsync(int id);
        Task AddAsync(T entity);
        // ... Other CRUD methods ...
    }
}

```

This demonstrates a generic repository interface and a concrete implementation using EF Core. The repository handles the persistence logic, abstracting it away from the calling contexts.  The `IRepository<T>`  interface enhances reusability across different entity types.


**Example 3: EF Core Context Usage**

```csharp
// Application.csproj - This project contains the application layer interacting with the repositories

namespace MyCompany.Application.Contexts
{
    public class MyDbContext : DbContext
    {
        public DbSet<Product> Products { get; set; }

        public MyDbContext(DbContextOptions<MyDbContext> options) : base(options) { }
    }
}

// Example Controller (or other application service)

namespace MyCompany.Application.Controllers
{
    public class ProductController : Controller
    {
        private readonly IProductRepository _productRepository;

        public ProductController(IProductRepository productRepository)
        {
            _productRepository = productRepository;
        }

        public async Task<IActionResult> Index()
        {
            var products = await _productRepository.GetAllAsync();
            return View(products);
        }
    }
}
```

The `MyDbContext` is a simple wrapper around the database connection.  It's injected into the `ProductRepository`, illustrating dependency injection. The controller utilizes the repository to interact with the data, completely unaware of the underlying EF Core context. This facilitates easy switching of contexts or even persistence mechanisms.


**3. Resource Recommendations**

*  **Clean Architecture: A Craftsman's Guide to Software Structure and Design** by Robert C. Martin
*  **Implementing Domain-Driven Design** by Vaughn Vernon
*  **Entity Framework Core in Action** (Focus on advanced topics like repository patterns and unit testing within the context of EF Core)
*  Documentation for Microsoft's Dependency Injection container.


This layered approach ensures that identical models aren't duplicated. Any changes to the domain model propagate effortlessly across all using contexts. This significantly reduces maintenance and improves the overall maintainability and scalability of the application.  Furthermore, this architecture fosters better testability, as you can easily mock repositories for unit testing of the application and domain logic without needing to involve a database. My own experiences have proven the effectiveness of this strategy in large-scale, long-lived projects.  Remember to carefully consider the specific needs of your application when implementing this approach, potentially tailoring the generic repository interface to meet its demands.
