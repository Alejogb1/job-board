---
title: "How can I implement a DbContext interface in DDD?"
date: "2024-12-23"
id: "how-can-i-implement-a-dbcontext-interface-in-ddd"
---

, let's get into it. There's a lot of conceptual overlap—and sometimes confusion—when bridging the principles of domain-driven design (ddd) and concrete technologies like entity framework core's `dbcontext`. The key is to realize that `dbcontext` itself isn't your domain model, but rather an infrastructure concern. My own journey, particularly when I was scaling a financial reporting platform a few years back, forced me to confront this head-on, leading to some important architectural decisions that I'll share.

The core idea is that the domain layer, following ddd principles, should be agnostic of any specific data access mechanism. It deals with pure business logic, entities, value objects, aggregates, and domain services. `dbcontext`, on the other hand, belongs squarely to the infrastructure layer – it’s the *how* not the *what*. We need to carefully abstract the data persistence mechanism so that our domain model is not coupled to it.

The classic approach is to introduce a repository pattern, and that’s where an abstracted `dbcontext` interface can fit in, though it's not the only solution. Think of the interface as a contract that dictates *what* operations you need to perform on the data store from the perspective of the domain, without revealing *how* it is actually being done. The concrete implementation of this interface will utilize the `dbcontext`.

Essentially, you’ll create an interface – let's call it, for the sake of clarity, `iunitofwork` (although sometimes you'll see `irepositorycontext`) – which exposes methods that are relevant to your aggregate roots, not to the `dbcontext` directly. You can then wrap the actual `dbcontext` within the implementation of this interface. This approach avoids leaking infrastructure concerns into your domain. This avoids the problem of having a change in your database technology forcing a change across your whole code base.

Here's how you might achieve this in practice:

```csharp
public interface IUnitOfWork : IDisposable
{
  Task<int> SaveChangesAsync(CancellationToken cancellationToken = default);

  IRepository<T> GetRepository<T>() where T : class;

  Task BeginTransactionAsync(CancellationToken cancellationToken = default);
  Task CommitTransactionAsync(CancellationToken cancellationToken = default);
  Task RollbackTransactionAsync(CancellationToken cancellationToken = default);

}


public interface IRepository<T> where T : class
{
  Task<T> GetByIdAsync(Guid id, CancellationToken cancellationToken = default);
  Task<IEnumerable<T>> GetAllAsync(CancellationToken cancellationToken = default);
  Task AddAsync(T entity, CancellationToken cancellationToken = default);
  Task UpdateAsync(T entity, CancellationToken cancellationToken = default);
  Task DeleteAsync(T entity, CancellationToken cancellationToken = default);
  Task<bool> ExistsAsync(Guid id, CancellationToken cancellationToken = default);
  // Other common operations
}

```

Then, you'd have a concrete implementation of both, something like this:

```csharp
public class EfUnitOfWork : IUnitOfWork
{
    private readonly MyDbContext _context;
    private readonly Dictionary<Type, object> _repositories = new Dictionary<Type, object>();
    private DbTransaction _transaction;
    public EfUnitOfWork(MyDbContext context)
    {
      _context = context;
    }
    public async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
      return await _context.SaveChangesAsync(cancellationToken);
    }

    public IRepository<T> GetRepository<T>() where T : class
    {
      if (!_repositories.ContainsKey(typeof(T)))
      {
        _repositories[typeof(T)] = new EfRepository<T>(_context);
      }
      return (IRepository<T>)_repositories[typeof(T)];
    }
    public async Task BeginTransactionAsync(CancellationToken cancellationToken = default)
    {
        if (_transaction != null)
            throw new InvalidOperationException("Transaction is already opened");

        _transaction = await _context.Database.BeginTransactionAsync(cancellationToken);
    }

    public async Task CommitTransactionAsync(CancellationToken cancellationToken = default)
    {
        if (_transaction == null)
        {
            throw new InvalidOperationException("No transaction is open to commit.");
        }
        try
        {
            await _transaction.CommitAsync();
        }
        finally
        {
          await _transaction.DisposeAsync();
          _transaction = null;
        }
    }


    public async Task RollbackTransactionAsync(CancellationToken cancellationToken = default)
    {
        if (_transaction == null)
        {
            throw new InvalidOperationException("No transaction is open to rollback.");
        }
        try
        {
            await _transaction.RollbackAsync();
        }
         finally
        {
          await _transaction.DisposeAsync();
          _transaction = null;
        }
    }

    public void Dispose()
    {
        _context.Dispose();
    }
}


public class EfRepository<T> : IRepository<T> where T : class
{
    private readonly MyDbContext _context;

    public EfRepository(MyDbContext context)
    {
        _context = context;
    }

    public async Task<T> GetByIdAsync(Guid id, CancellationToken cancellationToken = default)
    {
        return await _context.Set<T>().FindAsync(new object[] { id }, cancellationToken);
    }

    public async Task<IEnumerable<T>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        return await _context.Set<T>().ToListAsync(cancellationToken);
    }

    public async Task AddAsync(T entity, CancellationToken cancellationToken = default)
    {
        await _context.Set<T>().AddAsync(entity, cancellationToken);
    }

    public async Task UpdateAsync(T entity, CancellationToken cancellationToken = default)
    {
        _context.Set<T>().Update(entity);
    }
    public async Task DeleteAsync(T entity, CancellationToken cancellationToken = default)
    {
        _context.Set<T>().Remove(entity);
    }

    public async Task<bool> ExistsAsync(Guid id, CancellationToken cancellationToken = default)
    {
        return await _context.Set<T>().AnyAsync(e => (Guid)(typeof(T).GetProperty("Id")?.GetValue(e) ?? Guid.Empty) == id, cancellationToken);
    }
    // Other common operations
}
```

In this pattern, the `EfUnitOfWork` class manages the lifecycle of a `MyDbContext` instance (your actual `dbcontext`) and any potential transactions. It also provides a generic repository (`EfRepository`) implementation that handles basic crud operations, using the passed context. You'd typically register the concrete implementations (both `EfUnitOfWork` and `EfRepository`) with your dependency injection framework and expose `iunitofwork` and `irepository<t>` to the domain and application services. The application service would then access repositories through unit of work. The domain service would use the repository to access data through the repository interface.

This allows the domain to focus on business logic and data manipulation, without being aware of how data is actually persisted. For example, a domain service might call a repository to load an aggregate root based on a given ID and execute a method. The concrete implementation of that repository is in the infrastructure layer which actually uses the `dbcontext` behind the scenes.

One final example, just to clarify how a use-case scenario might look, assuming you had an `Order` aggregate root:

```csharp
public class OrderService
{
    private readonly IUnitOfWork _unitOfWork;

    public OrderService(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    public async Task ProcessOrderAsync(Guid orderId)
    {
        var orderRepository = _unitOfWork.GetRepository<Order>();

        var order = await orderRepository.GetByIdAsync(orderId);

        if(order == null)
        {
          //throw domain exception
          throw new Exception("Order does not exists");
        }

        order.MarkAsCompleted();

        await orderRepository.UpdateAsync(order);
        await _unitOfWork.SaveChangesAsync();

    }

      public async Task CreateOrderAsync(Order order)
    {
      var orderRepository = _unitOfWork.GetRepository<Order>();
        await orderRepository.AddAsync(order);

        await _unitOfWork.SaveChangesAsync();
    }
}
```

This example demonstrates how the `OrderService` interacts with the data access layer via the `iunitofwork` interface, making it independent of the `dbcontext`. It utilizes the `irepository<order>` to load and update `order` entities, the unit of work abstraction to persist changes and also handles transactional scope if required.

This abstraction provides testability (you can easily mock the repository) and facilitates a clean separation of concerns. It’s a slightly more work initially but is crucial for maintaining maintainability and adaptability in the long run. For deeper understanding of this pattern I’d recommend studying the book "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans and "Patterns of Enterprise Application Architecture" by Martin Fowler. These are cornerstone texts that provide the underlying principles for ddd and its implementation. Also, reviewing "Implementing Domain-Driven Design" by Vaughn Vernon offers practical examples and insights to reinforce the concepts.
