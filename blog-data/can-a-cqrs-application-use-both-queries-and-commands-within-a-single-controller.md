---
title: "Can a CQRS application use both queries and commands within a single controller?"
date: "2024-12-23"
id: "can-a-cqrs-application-use-both-queries-and-commands-within-a-single-controller"
---

Let's explore this. In my years building and maintaining complex systems, I've certainly encountered situations where the lines between what *should* be done and what *can* be done get blurred, and the question of combining commands and queries within a single controller in a CQRS-based system definitely falls into that category. While conceptually and strictly speaking, it might appear to violate the core principle of Command Query Responsibility Segregation (CQRS), the reality of software engineering often presents scenarios that push the boundaries.

To answer the direct question, yes, *technically*, you *can* have a controller handle both queries and commands. However, whether you *should* is a different, and arguably more important question. The problem arises not from technical limitations but from architectural cleanliness and maintainability. CQRS is all about separating the read (query) side from the write (command) side, each often optimized for its specific task. A controller handling both starts to dilute that clarity and can introduce several potential issues.

Let's think about a scenario I faced in a past project. We had an e-commerce platform, and initially, we had a straightforward REST controller handling product requests. It fetched product details (a query) and also handled updates like price changes (a command). Everything worked, but as the system scaled, we started noticing performance issues on reads when writes were heavy. The database, which was under stress, was doing too much, too often. Moving to a CQRS pattern where a dedicated read database and a dedicated write database made sense. It improved dramatically.

However, there were instances where we had a controller that needed to check if a product existed before allowing any action (a read) and then, based on certain user actions, perform a write. At that time, we had to grapple with the question of how to avoid tightly coupling everything, and a straight 'one-controller-for-both' solution felt like a step back.

The key is that CQRS isn't an absolute rule; it's a guideline that we adapt to the requirements. Sometimes, a very limited, controlled read in a command-handling context can be justified, if done consciously and cautiously. I'll now illustrate this with several examples, highlighting how, with careful implementation, you can do a little bending, without breaking the underlying intention of CQRS.

**Example 1: Before Write Validation with Read**

Imagine you need to update product information. To prevent incorrect data, you need to check if the product exists *before* applying any changes. In a strict CQRS implementation, a separate read operation would usually be required before sending the update command, possibly making two roundtrips.

Here, within the command handler itself (not the controller), a read can be performed, only for the validation purposes. The controller doesn't directly call a query service. It sends the command which contains the necessary information to perform the read as part of the validation before applying any changes.

```csharp
public class UpdateProductCommand : IRequest<bool>
{
    public Guid ProductId { get; set; }
    public string? NewName { get; set; }
    public decimal? NewPrice { get; set; }
}

public class UpdateProductCommandHandler : IRequestHandler<UpdateProductCommand, bool>
{
    private readonly IProductReadRepository _readRepository;
    private readonly IProductWriteRepository _writeRepository;

    public UpdateProductCommandHandler(IProductReadRepository readRepository, IProductWriteRepository writeRepository)
    {
        _readRepository = readRepository;
        _writeRepository = writeRepository;
    }

    public async Task<bool> Handle(UpdateProductCommand command, CancellationToken cancellationToken)
    {
        var product = await _readRepository.GetProductByIdAsync(command.ProductId);
        if (product == null)
        {
           // Log and handle product not found
            return false;
        }
        // Update fields if available and persist using the _writeRepository
        product.Name = command.NewName ?? product.Name;
        product.Price = command.NewPrice ?? product.Price;
        await _writeRepository.UpdateProductAsync(product);
        return true;
    }
}
```

In this snippet, the read is isolated inside the command handler and isn't part of a broad read operation. The controller only sends a command, maintaining clear separation of concerns.

**Example 2: Limited Read for Command Completion**

Sometimes, after successfully processing a command, a read operation is needed to return an updated entity with correct and current data. For example, consider a scenario where a new user is registered; we'd need to return the user details after successful creation including an auto-generated id, which is generated as part of the command operation, but is not available before the write operation.

```csharp
public class CreateUserCommand : IRequest<UserDto>
{
    public string UserName { get; set; }
    public string Email { get; set; }
}

public class CreateUserCommandHandler : IRequestHandler<CreateUserCommand, UserDto>
{
    private readonly IUserWriteRepository _writeRepository;
    private readonly IUserReadRepository _readRepository;
    public CreateUserCommandHandler(IUserWriteRepository writeRepository, IUserReadRepository readRepository) {
        _writeRepository = writeRepository;
        _readRepository = readRepository;
    }

    public async Task<UserDto> Handle(CreateUserCommand command, CancellationToken cancellationToken)
    {
        var newUser = new User { UserName = command.UserName, Email = command.Email };
        await _writeRepository.AddUserAsync(newUser);
        var createdUser = await _readRepository.GetUserByIdAsync(newUser.Id); // Read after successful create
        return new UserDto { Id = createdUser.Id, UserName = createdUser.UserName, Email = createdUser.Email };
    }
}
```

Again, the read operation is directly tied to the result of the command and is within the scope of the command handler. The controller only interacts with command dispatch.

**Example 3: Controller-Level Read with Predefined DTO**

In a situation where you need to display a confirmation message after a successful command execution, you *could* use a very constrained read at the controller, but only for a simple projection of the data to format the response.

```csharp
[ApiController]
[Route("api/products")]
public class ProductController : ControllerBase
{
    private readonly IMediator _mediator;
    private readonly IProductReadRepository _readRepository;

    public ProductController(IMediator mediator, IProductReadRepository readRepository)
    {
        _mediator = mediator;
        _readRepository = readRepository;
    }
    [HttpPost("update")]
    public async Task<IActionResult> UpdateProduct([FromBody] UpdateProductCommand command)
    {
        var result = await _mediator.Send(command);
        if (!result) return BadRequest();

        // Read a minimal view projection to display successful command response.
        var productView = await _readRepository.GetProductViewByIdAsync(command.ProductId);

        return Ok(new {message = $"Product {productView.Name} updated successfully" });
    }
}

```
Here the `GetProductViewByIdAsync` method within `IProductReadRepository` returns a very specific DTO optimized for that purpose. This pattern limits the read side exposure and reduces risk of inadvertently polluting the command handler.

These examples, based on my experience, show that CQRS isn't necessarily about absolutes. It's more about how you structure your application and maintain a strong separation of concerns. I tend to use a slightly less rigid approach where reading in a command handler (or a very limited read in the controller) is a valid option when very specifically limited to the outcome of a command execution. The most important aspect is that the separation of concerns is not compromised, and queries are not used to modify data or cause a state change. The controller, in my view, should primarily deal with orchestrating command submission and handling responses, and avoid calling query operations directly as far as feasible.

For a thorough understanding of the underlying principles of CQRS, I strongly recommend "Patterns, Principles, and Practices of Domain-Driven Design" by Scott Millett and Nick Tune. Also, the original work by Greg Young is invaluable; his "CQRS Documents" offer a deep dive into the rationale behind the pattern. Furthermore, "Building Microservices" by Sam Newman is a crucial guide for putting CQRS into a larger architectural perspective, specifically when designing microservice architectures. These resources will help you understand that CQRS is a tool, and like any tool, its effective usage lies in its correct application, not strict adherence to rigid interpretation of it.
