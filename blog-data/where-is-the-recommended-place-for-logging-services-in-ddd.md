---
title: "Where is the recommended place for logging services in DDD?"
date: "2024-12-16"
id: "where-is-the-recommended-place-for-logging-services-in-ddd"
---

, let's tackle this one. I've seen this discussion come up more than a few times, and it's a detail that, while seemingly small, can have a significant impact on the maintainability and overall health of a system designed with domain-driven design (ddd). The short answer is: logging is an *infrastructure concern*, not a core domain one. Let's break down why, and how to implement it in a way that doesn't pollute your domain logic.

Let’s begin with the perspective I developed through a particularly challenging migration project about six years ago. We were transitioning a legacy monolith to a microservices architecture underpinned by ddd principles. Initially, the team had scattered logging statements across domain entities, services, and even within value objects. This made debugging a nightmare because we were drowning in irrelevant logs and struggled to extract actionable insights. What appeared as an elegant solution in isolation, soon transformed into a maintenance disaster.

So, the core problem stems from the nature of logging. Domain logic should be concerned with the “what” and not the “how”. Consider a bank transfer operation. The domain is concerned with ensuring the correct amount is debited from one account and credited to another, adhering to rules such as insufficient funds or account lockouts. How this process is logged – using a particular format, routing to a specific data store, or even the severity of the log level – is entirely separate from the core business rules. That's an *infrastructure* decision. Injecting logging directly into domain objects introduces an unnecessary dependency on the logging infrastructure. This breaks the separation of concerns, tightly coupling the domain to specific technical implementations, and making the core logic harder to test and evolve.

When we had this realization in our previous project, we immediately began refactoring towards a structured logging approach. We extracted all logging logic into infrastructure-specific concerns, creating dedicated services or decorators responsible for logging. This yielded significant benefits in terms of code readability, maintainability, and testability.

Here's how you can achieve a clean separation of concerns in a ddd application while implementing effective logging:

**1. Application Services and Logging Decorators**

Application services sit at the boundary between your domain and the outside world. These services coordinate domain operations, and are a great candidate for applying a logging decorator pattern.

Here’s an example in pseudocode using Python, focusing on clarity over exact implementation details:

```python
from abc import ABC, abstractmethod

class AccountTransferService(ABC): # Abstract base class for service, can be replaced with an interface in other languages
    @abstractmethod
    def transfer(self, from_account_id, to_account_id, amount):
        pass

class DefaultAccountTransferService(AccountTransferService):
    def __init__(self, account_repository):
        self.account_repository = account_repository

    def transfer(self, from_account_id, to_account_id, amount):
       from_account = self.account_repository.get_by_id(from_account_id)
       to_account = self.account_repository.get_by_id(to_account_id)
       from_account.debit(amount)
       to_account.credit(amount)
       self.account_repository.save(from_account)
       self.account_repository.save(to_account)
       # No logging here, pure domain logic.

class LoggingAccountTransferService(AccountTransferService):
    def __init__(self, decorated_service, logger):
        self.decorated_service = decorated_service
        self.logger = logger

    def transfer(self, from_account_id, to_account_id, amount):
        self.logger.info(f"Starting transfer from account: {from_account_id} to account: {to_account_id}, amount: {amount}")
        try:
            result = self.decorated_service.transfer(from_account_id, to_account_id, amount)
            self.logger.info(f"Successfully completed transfer from account: {from_account_id} to account: {to_account_id}, amount: {amount}")
            return result
        except Exception as e:
            self.logger.error(f"Failed transfer from account: {from_account_id} to account: {to_account_id}, amount: {amount}, error: {e}")
            raise
```

In this example `DefaultAccountTransferService` implements core domain logic. The `LoggingAccountTransferService` acts as a decorator, wrapping the core service and adding logging logic *around* the domain logic execution. This follows the open/closed principle - we can extend the core behaviour (in this case adding logging) without modifying its source.

**2. Infrastructure Layer Logging Service**

For more complex scenarios, consider a dedicated logging service, located in your infrastructure layer. This allows for centralized logging configuration and consistent handling across the application.

Here is how it might be implemented in Java:

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultBankingLogger {
    private static final Logger logger = LoggerFactory.getLogger(DefaultBankingLogger.class);

    public void logInfo(String message) {
        logger.info(message);
    }

    public void logError(String message, Exception exception) {
       logger.error(message, exception);
    }

    public void logDebug(String message) {
       logger.debug(message);
    }
}

// Sample service that would consume this logger in the application layer
class BankService {
    private DefaultBankingLogger logger;

    public BankService(DefaultBankingLogger logger) {
        this.logger = logger;
    }

    public void performTransaction(String accountId, int amount) {
        logger.logInfo("Initiating transaction for account: " + accountId + ", amount: " + amount);
        try {
            // Execute core business logic here ...
            // ...
            logger.logDebug("Transaction succeeded for account: " + accountId);

        } catch (Exception e) {
            logger.logError("Transaction failed for account: " + accountId + ", Error:" , e);
            throw e;

        }
    }

}

```

Here, the `DefaultBankingLogger` is a concrete implementation using Slf4j. The logging logic is abstracted and can be easily replaced, such as using log4j2, without affecting the business layer. `BankService` then consumes this. Notice that this service doesn’t concern itself with how log output is generated, only with *what* should be logged.

**3. Logging Within Event Handlers**

Event handlers can also benefit from structured logging. They respond to domain events, often triggered by domain operations. Treat them similarly to application services, and wrap them in a logging decorator or use a dedicated logging service, ensuring minimal domain interaction. This might look something like this in C#:

```csharp
using Microsoft.Extensions.Logging;
using System;

public interface IEventHandler<TEvent>
{
    void Handle(TEvent domainEvent);
}

public class AccountCreatedEventHandler : IEventHandler<AccountCreatedEvent>
{
  private readonly ILogger<AccountCreatedEventHandler> _logger;

  public AccountCreatedEventHandler(ILogger<AccountCreatedEventHandler> logger)
  {
      _logger = logger;
  }

  public void Handle(AccountCreatedEvent domainEvent) {
      _logger.LogInformation($"Handling account created event for account {domainEvent.AccountId}");
      try {
           // process logic here
           _logger.LogInformation($"Account created event successfully handled for account {domainEvent.AccountId}");

      } catch(Exception ex){
           _logger.LogError(ex, $"Failed to handle account created event for account {domainEvent.AccountId}");
           throw;
      }
  }
}

// Dummy Event
public class AccountCreatedEvent {
   public Guid AccountId { get; set; }
}
```

Here, `AccountCreatedEventHandler` handles the `AccountCreatedEvent`.  We leverage `ILogger` from Microsoft’s extension library for logging capabilities. Notice that logging is not directly embedded within the event or domain logic, but handled by the infrastructure.

**Resources and Further Reading:**

For a deeper dive into ddd and separation of concerns, I'd recommend *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans. For more on general software architecture and design patterns, I recommend *Patterns of Enterprise Application Architecture* by Martin Fowler. Further, reading about the open/closed principle and general solid programming principles will be beneficial. These are the texts I found most informative when I was building systems with these requirements, and will be particularly helpful for the principles discussed in this response.

In summary, logging is crucial for observability, but it's an infrastructure concern that should be carefully isolated from domain logic. Keep your domain clean and focused on the "what," leaving the "how" of logging to the application and infrastructure layers. This makes your code more modular, testable, and maintainable in the long run.
