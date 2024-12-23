---
title: "Should CQRS queries return existence checks alongside data?"
date: "2024-12-23"
id: "should-cqrs-queries-return-existence-checks-alongside-data"
---

Alright, let's talk about CQRS and specifically whether queries should bundle existence checks with their data payloads. I’ve seen this debated more times than I can count, and having personally dealt with the fallout of both approaches on various projects, I feel I have a fairly pragmatic perspective. It’s not a cut-and-dried yes or no, it truly depends on the context of your application and its performance constraints.

The core tenet of Command Query Responsibility Segregation (CQRS) is, of course, separating operations that modify data (commands) from those that read it (queries). This separation is designed to offer significant flexibility and optimization opportunities, particularly in complex systems. When we then look at queries, we're typically focused on retrieving data for presentation, analytics, or other consumption. Throwing an existence check, like a `boolean` or similar indicator, into that payload might seem convenient at first glance. “Did we find it? Oh yes, here’s the data.” But this is a shortcut that can introduce subtle complexities and even performance bottlenecks, particularly when scaling.

First off, consider the scenarios where a straightforward boolean check within a query result is perfectly suitable. Suppose we're dealing with a single user profile lookup. If the query returns a user object or `null` (or an empty object depending on framework and language), embedding a `found` flag isn't adding any significant burden. Here's a simplified example in C#:

```csharp
public class UserProfile
{
    public Guid Id { get; set; }
    public string UserName { get; set; }
    // other user data
}

public class UserQueryResponse
{
  public bool Found { get; set; }
  public UserProfile User { get; set; }
}

public class UserService
{
  public UserQueryResponse GetUserById(Guid id)
  {
      var user =  _userRepo.GetById(id); // Assuming a repository

      if(user != null)
      {
        return new UserQueryResponse {Found = true, User = user};
      }

        return new UserQueryResponse {Found = false};

  }
}
```

In this example, the `UserQueryResponse` carries both the found status and the data. This works nicely for simple cases. However, it begins to fray when you scale up, specifically regarding the granularity of your queries and the complexity of your read models.

The problems arise when this practice is applied universally. The next example involves a scenario where I had to troubleshoot a reporting system where multiple complex filters combined with a large data set, using a database that wasn’t particularly optimized for read-heavy workloads. The initial implementation was returning a flag (alongside data) for each record indicating it passed all filters, but it quickly became a performance bottleneck. Every filter check resulted in not only data processing, but also this additional layer of “did it match” boolean handling and transmission.

Here's a conceptual view of the problem, also in C#, but simplified for readability:

```csharp
public class ReportItem
{
    public int Id { get; set; }
    public string Category { get; set; }
    public DateTime Date { get; set; }
   // other data
}

public class FilteredReportItem
{
  public bool Matched {get; set;}
  public ReportItem ReportItem { get; set;}

}


public class ReportingService
{
  public List<FilteredReportItem> GetFilteredReport(List<string> categories, DateTime fromDate)
  {
      var reportItems = _reportRepo.GetAll();

      var filteredItems = reportItems.Select(ri => new FilteredReportItem {
                                                                      Matched = categories.Contains(ri.Category) && ri.Date > fromDate,
                                                                      ReportItem = ri

                                                                      }).ToList();


    return filteredItems;

  }
}
```

Here, each `ReportItem` is wrapped in a `FilteredReportItem` class, carrying its data and a boolean. It wasn’t a query, but it demonstrates the issue: unnecessary bloat and computational cost, and that this approach scales poorly when you are dealing with larger datasets. The client side application or service would often discard items that ‘did not match’, yet all that data and processing occurred on the server anyway.

Instead, we moved to a pattern where we utilized separate queries or data views specifically tailored for each use case. This reduced the data volume moving around and delegated more filtering work to the database engine, which is usually significantly better optimized for these operations. For simple existence checks, we employed `COUNT` queries instead, which return only a number and incur a smaller performance penalty. The application logic then makes conditional calls to fetch the actual data when needed.

A refined example, again in C#, shows this revised approach:

```csharp
public class ReportingService
{
  public int GetFilteredCount(List<string> categories, DateTime fromDate)
  {
     return _reportRepo.GetFilteredCount(categories, fromDate);
  }

   public List<ReportItem> GetFilteredData(List<string> categories, DateTime fromDate, int skip, int take)
  {
    return _reportRepo.GetFilteredData(categories, fromDate, skip, take);
  }

}
```

In this version, we have a specialized query for the count and another for retrieving the data. The count query could be a simple `SELECT COUNT(*)` on the database, with filters in the `where` clause which provides a performance enhancement. The data query has paging applied using `skip` and `take`. The main point is that `Found` flags are removed completely from data queries, making each function more specific.

Ultimately, whether to include existence checks within queries boils down to the specific context and data being handled. When dealing with simple cases, a combined response might be acceptable. But as complexity increases and your application needs to scale, separating existence checks from data retrievals by using specialized queries is beneficial. Consider techniques like database views, indexing, and specialized queries to optimize data handling. You want to be sure your queries are as focused and fast as possible.

For more on database optimization techniques, "Database System Concepts" by Silberschatz, Korth, and Sudarshan is a classic, offering deep insights into query optimization and indexing. Furthermore, exploring patterns specific to CQRS can benefit from reading "Implementing Domain-Driven Design" by Vaughn Vernon. These resources provide fundamental understanding for building more robust and performant data access layers. These books have helped shaped my thinking over the years, and I'd recommend exploring them further. This, combined with practical application in different projects, has led me to this perspective. In short, be specific with your queries, and don't force a single pattern everywhere, adapt to the situation.
