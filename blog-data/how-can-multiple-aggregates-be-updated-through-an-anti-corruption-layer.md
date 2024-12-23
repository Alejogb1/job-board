---
title: "How can multiple aggregates be updated through an anti-corruption layer?"
date: "2024-12-23"
id: "how-can-multiple-aggregates-be-updated-through-an-anti-corruption-layer"
---

Let's dive straight into it, shall we? Handling updates across multiple aggregates through an anti-corruption layer (acl) is a problem I've bumped into more than a few times in my career, especially when dealing with microservice architectures where domain boundaries are strictly enforced. It's not uncommon to find yourself in a situation where a single user action needs to reflect changes in several independent parts of your system, each represented by its own aggregate root. The challenge, of course, is to maintain transactional consistency and avoid creating a tightly coupled mess.

Now, the core issue here isn't just about pushing data around; it's about translating concepts from one bounded context to another. The acl isn't merely a data transfer mechanism; it’s a semantic translator. One aggregate might conceptualize 'customer' differently than another aggregate focused on 'order processing.' The acl’s job is to ensure that information flowing between them is correctly interpreted and doesn't leak the internal implementation details of either domain.

In practice, this often manifests as a need to propagate changes originating from a single source to multiple downstream aggregates. It could be a user updating their profile information which needs to update a user management aggregate, a notification aggregate and even perhaps a separate reporting aggregate.

So how do we achieve this? Let’s explore three common scenarios and how I’ve tackled them, along with example code to clarify each approach:

**1. Event-Driven Propagation with Event Transformation**

This is the approach I've found most versatile. Here's how it typically goes: When a primary aggregate changes, it emits a domain event. The acl subscribes to these events and translates them into appropriate commands or events for the downstream aggregates. This decoupling means aggregates aren't directly aware of each other. It's a bit like a central messaging hub.

Imagine an 'Account' aggregate emits an `AccountNameChanged` event. The acl intercepts this, and transforms it into, say, a `UserNameUpdated` command for a 'User' aggregate and a `CustomerNameUpdated` command for a 'Reporting' aggregate.

Here's a simplified c# example:

```csharp
public class AccountService
{
    public void ChangeAccountName(Guid accountId, string newName)
    {
      // fetch account, update name, save and then:
      var accountNameChangedEvent = new AccountNameChanged(accountId, newName);
      EventPublisher.Publish(accountNameChangedEvent);
    }
}

public class Acl
{
    public Acl(IEventSubscriber eventSubscriber, ICommandDispatcher commandDispatcher)
    {
        eventSubscriber.Subscribe<AccountNameChanged>(HandleAccountNameChanged);
        _commandDispatcher = commandDispatcher;
    }

    private readonly ICommandDispatcher _commandDispatcher;

    private void HandleAccountNameChanged(AccountNameChanged @event)
    {
        var userNameUpdateCommand = new UpdateUserName(@event.AccountId, @event.NewName);
        _commandDispatcher.Dispatch(userNameUpdateCommand, destination: "User");

        var customerNameUpdateCommand = new UpdateCustomerName(@event.AccountId, @event.NewName);
        _commandDispatcher.Dispatch(customerNameUpdateCommand, destination: "Reporting");

    }
}
```

In this scenario, `EventPublisher` and `IEventSubscriber` represent an event bus (e.g., kafka, rabbitmq, or even an in-memory implementation), while `ICommandDispatcher` dispatches commands to their respective handlers within each aggregate's bounded context, ensuring the acl does its work to translate the events emitted from the `AccountService` to the commands appropriate for the `User` and `Reporting` aggregates. This method helps maintain clear boundaries and promotes eventual consistency.

**2. Command Translation through an Orchestration Layer**

In some situations, it might make more sense to orchestrate a series of commands. This is particularly useful when the updates are closely related and should occur more synchronously. The acl sits between the command originator and the destination aggregates, translating commands and ensuring the proper format.

For example, say a user registration process involves creating a user entity and setting up initial permissions. Instead of individual events, we can handle this using command orchestration:

```csharp
public class RegistrationService
{
    private readonly ICommandDispatcher _commandDispatcher;
    public RegistrationService(ICommandDispatcher commandDispatcher){
      _commandDispatcher = commandDispatcher;
    }

    public void RegisterUser(string username, string password, string email)
    {
      var registerUserCommand = new RegisterUserCommand(username, password, email);
      _commandDispatcher.Dispatch(registerUserCommand, destination: "UserManagement");
    }
}


public class Acl
{
  public Acl(ICommandDispatcher commandDispatcher){
    _commandDispatcher = commandDispatcher;
  }

  private readonly ICommandDispatcher _commandDispatcher;
    public void HandleRegisterUserCommand(RegisterUserCommand command)
    {
      var createUserCommand = new CreateUserCommand(command.Username, command.Password, command.Email);
      _commandDispatcher.Dispatch(createUserCommand, destination: "User");

       var createPermissionsCommand = new InitializePermissions(command.Username);
      _commandDispatcher.Dispatch(createPermissionsCommand, destination: "Permission");
    }
}
```

Here, `RegistrationService` does not dispatch a single command that affects multiple aggregates, rather it uses the `_commandDispatcher` to submit the `RegisterUserCommand`. The `Acl` then receives this command, translates it to multiple commands relevant to individual aggregates, and dispatches each command. This process ensures that the registration process involves creating both a user and setting up their initial permissions, all through the acl. It's a controlled, orchestrated sequence within the domain. This approach trades some of the decoupling of event-driven architectures for more immediate consistency within the context of the operation.

**3. Using API Adapters for Data Retrieval and Transformation**

Sometimes, the update isn't a direct command but rather a data retrieval and transformation operation. For example, maybe one aggregate needs to pull data from another and transform it to suit its own context. In such cases, the acl acts as an adapter that encapsulates the data access details of the source aggregate and translates the data into the format required by the destination.

Consider a `Customer` aggregate that needs to obtain user data from the `User` aggregate. The `Customer` aggregate shouldn’t directly call the `User` aggregate's data access layer. Instead, it makes a request through the acl.

```csharp

public class CustomerService
{
  private readonly IUserService _userService;
  public CustomerService(IUserService userService) {
      _userService = userService;
  }

  public void CreateCustomer(string userName)
  {
    var userDto = _userService.GetUserDtoByName(userName); //call the acl to fetch the user
    //create the customer and add user related information as needed
    var customer = new Customer(userDto.Id, userDto.UserName);
  }

}


public interface IUserService{
  UserDto GetUserDtoByName(string userName);
}

public class Acl : IUserService
{
  private readonly IUserRepository _userRepository;

  public Acl(IUserRepository userRepository)
  {
    _userRepository = userRepository;
  }

  public UserDto GetUserDtoByName(string userName)
  {
      var user = _userRepository.GetUserByName(userName); // fetch user from user's data layer
      return new UserDto(user.Id, user.UserName);
  }
}

public class UserDto
{
  public Guid Id { get; set; }
  public string UserName { get; set; }
  public UserDto(Guid id, string userName)
  {
    Id = id;
    UserName = userName;
  }
}

```

Here, `CustomerService` calls the acl (`IUserService`) through an interface. The acl uses a repository pattern to pull user data, transforms it to the `UserDto` object and returns that to the consumer (i.e., `CustomerService`). This encapsulates data access concerns and avoids direct coupling between aggregates. The `UserDto` object in this example represents a data transfer object, designed to be consumed outside of the `User` bounded context, which the acl will convert to the user format needed by the customer service, thus preventing the customer service from needing to know about the details of the user bounded context.

These examples should help visualize the acl's role in propagating updates across aggregates. It's critical to note that the "correct" method depends heavily on the specifics of the domain and system requirements. Generally, the event-driven approach provides the greatest decoupling, whereas orchestration offers more immediate consistency. The api adapters provide a clean way of bridging the differences between models.

For further reading, I'd recommend checking out these resources: *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans is foundational for understanding bounded contexts and aggregates. For a practical approach to microservices, Sam Newman's *Building Microservices* is invaluable. The concepts of event sourcing and cqrs are discussed in Martin Fowler's writings and are crucial to leveraging an event driven architecture for this problem, and the concept of message buses are discussed in *Enterprise Integration Patterns* by Gregor Hohpe. Finally, if you're interested in the technical specifics of implementing acl's in dotnet, the Microsoft documentation on MediatR is a good starting point for command and query handling, and for message buses, such as Azure service bus, or RabbitMq. These will help provide a sound theoretical and practical understanding of how to handle data translation.

Remember, using an anti-corruption layer isn’t about making the system less complex; it's about structuring that complexity in a way that’s manageable and maintainable. It’s about clear boundaries, explicit data transformation, and controlled propagation of changes. It’s a technique that's proved its value many times over.
