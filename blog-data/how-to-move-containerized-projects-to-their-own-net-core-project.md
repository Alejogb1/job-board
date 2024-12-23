---
title: "How to move containerized projects to their own .NET Core project?"
date: "2024-12-23"
id: "how-to-move-containerized-projects-to-their-own-net-core-project"
---

Okay, let's tackle this. I remember a particularly challenging project back in '18 where we had a sprawling monolith of a .net framework application, and the mandate was to containerize and modernize, piece by piece. Part of that process involved breaking specific functionalities into their own, discrete .net core services, each living within its own container. It wasn't a trivial task, but definitely achievable with a structured approach. So, you’re looking at moving containerized projects into their own .net core projects? It's a good move for modularity and scalability. Here's how I’ve approached this, based on my experiences.

The fundamental concept revolves around decoupling. You're essentially taking a piece of your application, which might currently be bundled with everything else inside a container, and giving it its own .net core identity. This requires meticulous planning and a deep understanding of your application's architecture. Here are the core steps:

**1. Identifying the Boundaries:**

The first and most crucial step is identifying which functionalities logically belong to separate services. This isn't always immediately apparent. I've found it useful to look for specific modules or responsibilities within the existing containerized app that:

*   Have distinct input/output boundaries.
*   Have independent data access patterns.
*   Can scale or change independently of the rest of the application.
*   Have different deployment requirements (e.g., CPU vs. memory bound).

Avoid creating microservices that are too granular, which can lead to excessive inter-service communication overhead and added complexity. Conversely, don’t make them too monolithic either; find a balance that provides sufficient independence but keeps the system manageable. The aim here is to create well-defined, cohesive units. Think “single responsibility principle” but applied at the service level. In the past, I often mapped the modules on a large whiteboard, annotating dependencies, and discussing potential bottlenecks with other developers; it's a collaborative process and very beneficial.

**2. Extracting the Code:**

Once you've identified the boundaries, it's time to extract the relevant code. This involves creating a new .net core project (usually a web api or a worker service, depending on the functionality), and carefully porting over the relevant files. This usually includes:

*   Business logic classes.
*   Data access classes (repositories, entities, etc.).
*   Configuration files.
*   Any relevant third-party libraries or dependencies.

I typically start by setting up a clean project template from the .net cli. It's crucial at this stage to ensure that all external dependencies are properly managed using nuget packages. This ensures version consistency and eliminates some of the common pitfalls that can occur during migration. The extraction needs to be performed carefully, so as to not introduce breaking changes in the original container's functionality. Use git branches wisely here; work in a new branch for the extraction, so you can revert quickly if needed.

**3. Defining Communication Interfaces:**

Once extracted, these new projects need a way to communicate with the rest of the application. Avoid tight coupling through direct method calls. Instead, adopt well-defined communication interfaces such as:

*   **REST APIs:** For synchronous request/response patterns.
*   **Message Queues:** For asynchronous communication and event-driven architectures. gRPC is also a valuable contender for high-performance inter-service communication.
*   **Event Buses:** For broadcasting events within the system.

The choice here depends on the specific needs of the service. For instance, if a service needs to return a user's profile details, a REST API would make sense. On the other hand, a background process that notifies users about daily reminders would be better suited for a message queue. Consider frameworks like MassTransit or RabbitMQ for easier integration.

Here's an example of a simplified api controller within a .net core web api project. This illustrates the structure, it's very basic and has been trimmed for brevity.

```csharp
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace UserService.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class UsersController : ControllerBase
    {
        private readonly IUserRepository _userRepository;

        public UsersController(IUserRepository userRepository)
        {
            _userRepository = userRepository;
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetUser(int id)
        {
            var user = await _userRepository.GetUserByIdAsync(id);
            if (user == null)
            {
                return NotFound();
            }
            return Ok(user);
        }
    }
}
```

And here is an extremely basic example of a message consumer in another service:

```csharp
using MassTransit;
using System.Threading.Tasks;

namespace NotificationService.Consumers
{
    public class UserCreatedConsumer : IConsumer<UserCreatedEvent>
    {
        public async Task Consume(ConsumeContext<UserCreatedEvent> context)
        {
            // Send a notification to the user using context.Message data
            // This would involve some form of sending an email or similar.

            System.Console.WriteLine($"Received event user created for UserID : {context.Message.UserId}  Notification sent");
            await Task.CompletedTask;
        }
    }
}
```

And, finally, here is an example of a producer of the above event, it would be in a different service:

```csharp
using MassTransit;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace UserService.Controllers
{

    [ApiController]
    [Route("api/[controller]")]
    public class UserController : ControllerBase
    {
        private readonly IPublishEndpoint _publishEndpoint;

        public UserController(IPublishEndpoint publishEndpoint)
        {
            _publishEndpoint = publishEndpoint;
        }

        [HttpPost]
        public async Task<IActionResult> CreateUser([FromBody] User newUser)
        {
           //Logic to create new user
           var newUserId = 1234; // Example

            var userCreatedEvent = new UserCreatedEvent
            {
                 UserId = newUserId
            };
            await _publishEndpoint.Publish(userCreatedEvent);
            return Ok(new { Message = "User Created" });
        }
    }
}
```

These examples show how separate services can communicate, one over a rest api and the others through a pub/sub pattern. The `IUserRepository` in the api controller would be an abstraction to decouple data access implementation and allow for testability. The `UserCreatedEvent` would be defined in an shared library between the two services.

**4. Containerizing the New Project:**

Once the .net core service is working, it’s time to containerize it. Create a Dockerfile that:

*   Specifies the base image (usually a .net sdk image for building and a runtime image for deployment).
*   Copies the project files.
*   Runs the necessary commands to restore and build the application.
*   Defines the entry point for the container.

This step is very similar to the one used for the original container, but this time it's for your new, smaller service. Tools such as docker-compose or kubernetes are useful for orchestrating these individual containers into a cohesive application.

**5. Testing and Deployment:**

Thorough testing is essential. Implement unit tests and integration tests to ensure the new service functions correctly in isolation and when interacting with other services. It's vital to test the communication interfaces rigorously, verifying both success cases and failure scenarios.

For deployment, consider using cloud platforms (e.g., aws, azure, google cloud) or container orchestration platforms (e.g., kubernetes). Ensure that your deployment pipeline is automated so that deployments can be performed consistently and repeatedly.

**Further Reading and Resources:**

*   **"Building Microservices" by Sam Newman:** A classic book that lays the foundation for understanding microservices architectures, from a theoretical to a practical level.
*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans:** While not directly about containerization, this book is crucial for designing well-bounded contexts that are critical for identifying microservice boundaries.
*   **Microsoft's Documentation on .net Core:** They have excellent guides and tutorials on building web apis and worker services that can be deployed as containers.
*   **Kubernetes in Action by Marko Luksa:** A very comprehensive and practical guide for understanding container orchestration and management using kubernetes.

This process isn't always easy. There might be unexpected dependencies, configuration challenges, or performance issues to address. However, by taking a systematic and disciplined approach, you can transition from a monolithic architecture to a more modular and scalable system based on discrete .net core services. It requires careful planning, thorough testing, and a solid grasp of fundamental software engineering principles, but the benefits in terms of maintainability and scalability are immense. It’s a journey, and iterative improvements are just part of the process.
