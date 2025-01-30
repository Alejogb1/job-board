---
title: "Can read-only repositories in DDD return value objects?"
date: "2025-01-30"
id: "can-read-only-repositories-in-ddd-return-value-objects"
---
The core principle of Domain-Driven Design (DDD) is to model the domain accurately, and a crucial aspect of that modeling involves distinguishing between entities and value objects. Specifically, value objects, defined by their attributes rather than identity, pose an interesting interaction point with read-only repositories. Based on my experience developing systems utilizing a DDD architecture for financial data analysis, I have encountered this exact scenario multiple times, and my conclusion is that read-only repositories *absolutely* can and should return value objects. In fact, it's often a necessity to preserve the domain model's integrity.

My experience leads me to this conclusion based on the understanding that read-only repositories, designed for querying data without modifying it, often need to return aggregated or constructed data. These aggregated or constructed pieces of data frequently fit the description of value objects within the domain. These objects represent a meaningful conceptual whole within the domain, lacking identity and possessing immutability characteristics that define value objects. The read-only nature of the repository does not preclude their usage. Conversely, if the read-only repository returns something other than value objects when suitable, the overall design would become more complex and would likely create leaky abstractions.

A clear explanation here requires emphasizing the fundamental nature of value objects and the limitations of read-only repositories. Value objects, as a pattern, are immutable objects whose equality depends solely on their attribute values. They represent domain concepts, but they do not have a unique identity within the system. This contrasts with entities, which have identity and a lifecycle. Now, read-only repositories are responsible for retrieving information from the persistence layer or any other data source. They do not modify the state of the data they access. In scenarios requiring data aggregation or filtering, the data retrieved by read-only repositories may not exactly match an entity. Therefore, encapsulating the returned data within a value object preserves the domain model.

For instance, consider a scenario in a financial application where you need to retrieve statistical summaries for a customer's accounts. It would be inefficient to have a different kind of repository which provides only statistical reports. Using the existing read-only repository would provide a better abstraction. This statistical summary, which might include values like total balance, average monthly balance, and highest transaction amount, is not a traditional entity; it is dependent on calculations and does not have its own identity. It's instead a value object, representing the meaningful collection of data.

Below, I present three code examples to illustrate the point. The first focuses on simple value object return. The second example demonstrates value object aggregation. The third example demonstrates using a domain service with a read-only repository to compose value objects.

**Example 1: Simple Value Object Return**

```csharp
public record AccountBalance(decimal Balance, string Currency);

public interface IReadOnlyAccountRepository
{
    AccountBalance GetBalance(Guid accountId);
}

public class AccountRepository : IReadOnlyAccountRepository
{
   public AccountBalance GetBalance(Guid accountId)
    {
        // Simulate database query (replace with actual implementation)
        var balance =  GetBalanceFromDatabase(accountId);
        var currency =  GetCurrencyFromDatabase(accountId);

        return new AccountBalance(balance, currency);
    }

   private decimal GetBalanceFromDatabase(Guid accountId) {
            // Imagine a DB call
           return 1234.56m;
    }

  private string GetCurrencyFromDatabase(Guid accountId) {
      // Imagine a DB call
       return "USD";
    }
}
```

In this example, `AccountBalance` is a value object representing an account's balance and its currency. The `IReadOnlyAccountRepository` method `GetBalance` retrieves the balance and currency from some data source, constructs the `AccountBalance` object, and returns it. The repository remains read-only as it does not modify the underlying data.

**Example 2: Value Object Aggregation**

```csharp
public record CustomerSummary(decimal TotalBalance, int AccountCount);

public interface IReadOnlyCustomerRepository
{
    CustomerSummary GetCustomerSummary(Guid customerId);
}

public class CustomerRepository : IReadOnlyCustomerRepository
{
    private readonly IReadOnlyAccountRepository _accountRepository;

    public CustomerRepository(IReadOnlyAccountRepository accountRepository)
    {
        _accountRepository = accountRepository;
    }

    public CustomerSummary GetCustomerSummary(Guid customerId)
    {
        // Simulate database query to get customer accounts (replace with actual implementation)
        var accountIds = GetCustomerAccountIdsFromDatabase(customerId);

        decimal totalBalance = 0;
        foreach (var accountId in accountIds)
        {
             var accountBalance = _accountRepository.GetBalance(accountId);
            totalBalance += accountBalance.Balance;
        }

        return new CustomerSummary(totalBalance, accountIds.Count);
    }
    private List<Guid> GetCustomerAccountIdsFromDatabase(Guid customerId){
           // imagine fetching IDs from DB
           return new List<Guid>(){ Guid.NewGuid(),Guid.NewGuid()};
    }
}

```

Here, `CustomerSummary` is a value object containing the total balance of all customer's accounts and the number of accounts. The `GetCustomerSummary` method on `IReadOnlyCustomerRepository`, aggregates data retrieved using underlying repository, creating and returning the `CustomerSummary` value object. The aggregation logic exists solely within the repository, maintaining a clear separation of concerns. The read-only nature of the repository is maintained, and yet domain concepts are expressed in the correct fashion.

**Example 3: Value Object Composition via Domain Service**

```csharp
public record  ExchangeRate(string FromCurrency, string ToCurrency, decimal Rate);

public interface IReadOnlyExchangeRateRepository
{
    ExchangeRate GetExchangeRate(string fromCurrency, string toCurrency);
}

public class ExchangeRateService
{
    private readonly IReadOnlyExchangeRateRepository _exchangeRateRepository;

    public ExchangeRateService(IReadOnlyExchangeRateRepository exchangeRateRepository)
    {
        _exchangeRateRepository = exchangeRateRepository;
    }

     public decimal Convert(decimal amount, string fromCurrency, string toCurrency)
    {
        var exchangeRate = _exchangeRateRepository.GetExchangeRate(fromCurrency, toCurrency);

        if (exchangeRate == null) {
             // handle
             throw new ArgumentException($"Cannot find exchange rate from {fromCurrency} to {toCurrency}");
        }

         return amount * exchangeRate.Rate;
    }
}

public class ExchangeRateRepository : IReadOnlyExchangeRateRepository
{
    public ExchangeRate GetExchangeRate(string fromCurrency, string toCurrency)
    {
        // Simulate database query to fetch rates
       // in practice a DB query would be implemented here
        return new ExchangeRate(fromCurrency, toCurrency, 1.10m);
    }
}
```

In this instance, the `ExchangeRate` is the value object. The `ExchangeRateService` depends on `IReadOnlyExchangeRateRepository`. In this example, a domain service retrieves a value object using a read-only repository. The service then uses it as part of a larger domain operation. This example shows how value objects retrieved by the repository are used to compose more complex domain behaviors. This maintains separation of concern while allowing the repository to return the most suitable abstraction, which are domain objects.

These examples are designed to be illustrative, but the core ideas are transferable. Returning value objects from read-only repositories does not contradict the purpose of these repositories; rather, it's a necessary aspect of achieving a cohesive domain model. It reduces coupling by correctly abstracting business data and logic within the value object. They provide a flexible, efficient, and highly coherent way to work with domain objects. It improves maintainability and extensibility, since it allows for a central location for querying data, while also adhering to DDD principles.

In terms of resources, I would suggest exploring literature which emphasizes the crucial distinction between value objects and entities. The core concepts of DDD are critical to understanding the rationale behind the design, specifically those regarding aggregate design. Books and articles that focus on design principles of effective repositories should be studied, as they often address the interaction between repositories and domain objects in context of both read and write operations. Consider resources that address read-model implementations in CQRS systems too, as it helps to understand how the data is transformed before arriving at domain layer.
