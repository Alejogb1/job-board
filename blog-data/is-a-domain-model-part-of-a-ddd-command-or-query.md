---
title: "Is a domain model part of a DDD command or query?"
date: "2024-12-23"
id: "is-a-domain-model-part-of-a-ddd-command-or-query"
---

, let's unpack this one. It's a question that often surfaces in DDD discussions, particularly when people start getting hands-on with implementation details. I remember wrestling with this myself several years back while working on a distributed e-commerce platform. We were transitioning from a more traditional layered architecture to a DDD approach, and the boundaries between commands, queries, and the domain model became… well, let's just say ‘murky’ for a while. The short answer is, the domain model isn’t *part* of a command or query in the way you might think. Instead, commands and queries *interact* with the domain model.

Let's break that down. We need to first define what we mean by a domain model in a DDD context. It's not just a database schema, it's a conceptual representation of the real-world business entities and their interactions. Think of it as the heart of your application, encapsulating business logic and rules. In a well-designed system, the domain model is where you express these rules in code, using aggregates, entities, value objects, and domain events.

Now, where do commands and queries fit in? Commands are requests to *change* the system’s state. Examples would be ‘place order,’ ‘cancel shipment,’ or ‘update customer address.’ They represent actions taken within the business domain that can modify persistent data. Queries, conversely, are requests to retrieve data without modifying anything. Examples would include ‘get customer details,’ ‘find all pending orders,’ or ‘calculate total sales for a period.’ Crucially, queries should not have side effects; they should not alter the system state.

So, back to the main question: is the domain model *part* of the command or query? The answer is no, not directly. Commands and queries orchestrate *interactions* with the domain model to achieve their purposes. They use the model's components, such as aggregates, to enforce business rules during state transitions (commands) or to retrieve relevant data (queries). The domain model itself remains a cohesive, independent unit that encapsulates the core business logic. It's not something that’s included inside a specific command or query like a property or argument, rather it's the ‘stage’ on which commands play out and from which queries gather information.

To illustrate, let's look at a command example in a hypothetical e-commerce system. We'll use C# for our examples but the concepts translate to other languages easily.

```csharp
//Example 1: A command handler interacting with the domain model.

public class PlaceOrderCommand
{
    public Guid CustomerId { get; set; }
    public List<OrderItem> Items { get; set; }
    // Other relevant order properties...
}

public class PlaceOrderCommandHandler
{
    private readonly IOrderRepository _orderRepository;
    private readonly ICustomerRepository _customerRepository;

    public PlaceOrderCommandHandler(IOrderRepository orderRepository, ICustomerRepository customerRepository)
    {
        _orderRepository = orderRepository;
        _customerRepository = customerRepository;
    }

    public void Handle(PlaceOrderCommand command)
    {
        var customer = _customerRepository.GetById(command.CustomerId);
        if (customer == null)
           throw new CustomerNotFoundException($"Customer with ID {command.CustomerId} not found.");


        var order = Order.CreateNew(customer, command.Items); // Domain model creating order
        _orderRepository.Save(order);

    }
}

```

In this example, `PlaceOrderCommand` is the command, and `PlaceOrderCommandHandler` orchestrates the operation. The handler retrieves the customer via a repository, then invokes a factory method `Order.CreateNew()` on our domain model (specifically, an aggregate `Order`), which encapsulates the business logic related to order creation. The command handler doesn’t “contain” the model, instead it utilizes the model to enact the command. The repositories themselves are abstract interfaces which abstract the mechanism to persist data, whether it's a database, file, or some other persistent storage, and operate on the domain model directly.

Now, let’s look at a query example:

```csharp
// Example 2: A query handler fetching data from the domain model via repository.

public class GetCustomerDetailsQuery
{
    public Guid CustomerId { get; set; }
}


public class GetCustomerDetailsQueryHandler
{
    private readonly ICustomerRepository _customerRepository;
    public GetCustomerDetailsQueryHandler(ICustomerRepository customerRepository)
    {
        _customerRepository = customerRepository;
    }

    public CustomerDetailsDto Handle(GetCustomerDetailsQuery query)
    {
        var customer = _customerRepository.GetById(query.CustomerId);
        if (customer == null)
            return null;


        return new CustomerDetailsDto // Mapping to DTO
        {
            Id = customer.Id,
            FirstName = customer.FirstName,
            LastName = customer.LastName,
            // Other relevant properties...
        };
    }
}

public class CustomerDetailsDto
{
    public Guid Id { get; set;}
    public string FirstName { get; set; }
    public string LastName { get; set; }

    // Other properties
}

```

In this case, `GetCustomerDetailsQuery` is the query, and `GetCustomerDetailsQueryHandler` is the handler. The handler uses the repository to retrieve the `Customer` aggregate from the underlying data store. This illustrates how the query is interacting with the domain model indirectly through repositories. The query doesn’t modify the domain model but uses it to form its response. Note the use of a DTO - Data Transfer Object; It's best practice to avoid directly exposing the domain model to the outside world. Mapping from domain entities to DTOs allows for flexibility in the query response.

Finally let’s look at how you might handle complex criteria. While we might think queries always return an aggregate, often we need to work with specific datasets. In cases where you need to tailor the data, a query could involve multiple queries through specialized repositories or views, or utilizing something like specifications. A specification provides a way to abstract out query predicates.

```csharp
//Example 3: A complex query using a specification.

public interface ISpecification<T>
{
    Expression<Func<T, bool>> ToExpression();
}


public class CustomerWithPendingOrders : ISpecification<Customer>
{
    public Expression<Func<Customer, bool>> ToExpression()
    {
            return c => c.Orders.Any(o => o.Status == OrderStatus.Pending);
    }
}


public class GetCustomersWithPendingOrdersQuery
{
}


public class GetCustomersWithPendingOrdersQueryHandler
{
    private readonly ICustomerRepository _customerRepository;

    public GetCustomersWithPendingOrdersQueryHandler(ICustomerRepository customerRepository)
    {
        _customerRepository = customerRepository;
    }

    public List<CustomerDetailsDto> Handle(GetCustomersWithPendingOrdersQuery query)
    {
        var specification = new CustomerWithPendingOrders();
        var customers = _customerRepository.Find(specification); // Using specification

        return customers.Select(customer => new CustomerDetailsDto
            {
                Id = customer.Id,
                FirstName = customer.FirstName,
                LastName = customer.LastName
            }).ToList();

    }

}

```

Here, the specification filters the customer set based on the criteria “customer has pending order”. This decouples the query from knowing *how* to perform a filtering based on the complex requirement; the repository can utilize this specification to perform the required query. The query handler uses the customer repository to retrieve relevant customer data by using the specification. Again, the domain is not directly *in* the query, but the query operates on the domain.

To delve deeper into these concepts, I'd recommend several resources. For a solid foundation in DDD, *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans is the canonical text. It's dense, but indispensable. *Implementing Domain-Driven Design* by Vaughn Vernon offers a more practical approach, walking through implementation strategies. And for understanding command-query separation (CQS), you can explore papers from Martin Fowler and Greg Young; often it’s more fruitful to explore their essays and talks on their personal websites and conference presentations, which are usually available online.

In summary, the domain model isn’t a component of a command or query. Rather, it is a central element that is leveraged *by* commands to implement business logic and change state, and *by* queries to extract data without side effects. The domain model should be a well-defined, cohesive unit that embodies the core business logic, and it should be interacted with, not embedded within, commands and queries. This separation of concerns is vital for building maintainable and scalable applications. From the front-lines, trust me on this one; I've seen first-hand the benefits of keeping these concepts distinct.
