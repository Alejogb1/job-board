---
title: "How can I execute multiple REST API requests concurrently in C# .NET Core?"
date: "2025-01-30"
id: "how-can-i-execute-multiple-rest-api-requests"
---
Concurrent execution of multiple REST API requests within a C# .NET Core application necessitates careful consideration of threading and asynchronous operations to avoid performance bottlenecks and ensure responsiveness.  My experience developing high-throughput microservices has highlighted the critical importance of leveraging the `async` and `await` keywords, along with appropriate task management techniques.  Incorrectly implemented concurrency can lead to deadlocks, resource starvation, and significantly degraded application performance.  Therefore, selecting the appropriate approach depends heavily on the specific requirements of the application, primarily the degree of dependency between the requests.

**1. Clear Explanation:**

The core challenge involves managing multiple asynchronous operations concurrently without blocking the main thread.  Directly invoking multiple `HttpClient` requests sequentially will result in serial execution, negating the benefits of concurrency.  Instead, we must employ mechanisms to initiate each request asynchronously and then efficiently aggregate their results.  Three prominent approaches exist:  `Task.WhenAll`, `Parallel.ForEach`, and custom asynchronous task scheduling with `SemaphoreSlim`.  The optimal choice depends on whether the requests are independent (can proceed without influencing each other) or dependent (require sequential processing or specific ordering).

For independent requests, `Task.WhenAll` provides a straightforward, efficient solution.  It awaits the completion of all provided tasks, returning an array of results once all are finished.  If any task throws an exception, the `WhenAll` method aggregates the exceptions into an aggregate exception.  This approach is ideal when the order of results is not critical and failure of a single request doesn't necessitate halting others.

For dependent requests or situations requiring granular control over concurrency, `Parallel.ForEach` offers more fine-grained control by allowing for parallel execution with configurable parallelism levels.  However, `Parallel.ForEach` typically manages threads directly, while `Task.WhenAll` works with tasks – the latter offering better integration with the async/await pattern.

Finally, using `SemaphoreSlim` offers superior control over resource allocation, especially in scenarios with limited external resources (like connection pools or API rate limits).  It ensures a specified number of requests are executed concurrently, preventing overload of the target API or resource depletion.

**2. Code Examples with Commentary:**

**Example 1: Using `Task.WhenAll` for Independent Requests**

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

public class ApiCaller
{
    private readonly HttpClient _httpClient;

    public ApiCaller()
    {
        _httpClient = new HttpClient();
    }

    public async Task<string[]> MakeMultipleRequestsAsync(string[] urls)
    {
        var tasks = urls.Select(async url => await _httpClient.GetStringAsync(url));
        return await Task.WhenAll(tasks);
    }
}


//Example usage:
string[] urls = { "http://api.example.com/data1", "http://api.example.com/data2", "http://api.example.com/data3" };
ApiCaller caller = new ApiCaller();
string[] results = await caller.MakeMultipleRequestsAsync(urls);

foreach (string result in results)
{
    Console.WriteLine(result);
}

```

This example demonstrates the simplest approach. `Task.WhenAll` efficiently handles multiple independent API calls, returning an array of strings containing the responses. Error handling is implicit; exceptions from individual tasks will be aggregated into an exception thrown by `Task.WhenAll`.  More robust error handling could be implemented using `try-catch` blocks within the lambda expression.


**Example 2: Using `Parallel.ForEach` for Dependent Requests (with caution)**

```csharp
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;

public class ParallelApiCaller
{
    private readonly HttpClient _httpClient;

    public ParallelApiCaller()
    {
        _httpClient = new HttpClient();
    }

    public async Task<List<string>> MakeDependentRequestsAsync(List<string> urls)
    {
        List<string> results = new List<string>();
        Parallel.ForEach(urls, url =>
        {
            string result = _httpClient.GetStringAsync(url).Result; // Blocking call - avoid if possible
            lock (results)
            {
                results.Add(result);
            }
        });
        return results;
    }
}
```

This example showcases `Parallel.ForEach`. However,  the `.Result` property is used, which is a blocking call and defeats the purpose of asynchronous programming.  This approach is demonstrably less efficient than using async/await.   The lock statement protects the shared `results` list from race conditions during concurrent access.  Ideally, a more sophisticated, asynchronous data structure should be employed to avoid blocking. This is primarily included to illustrate a problematic scenario and should be avoided in favor of the Task-based approaches.



**Example 3:  `SemaphoreSlim` for Rate-Limited APIs**

```csharp
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

public class RateLimitedApiCaller
{
    private readonly HttpClient _httpClient;
    private readonly SemaphoreSlim _semaphore;

    public RateLimitedApiCaller(int maxConcurrentRequests)
    {
        _httpClient = new HttpClient();
        _semaphore = new SemaphoreSlim(maxConcurrentRequests);
    }

    public async Task<List<string>> MakeRequestsAsync(List<string> urls)
    {
        List<string> results = new List<string>();
        List<Task> tasks = new List<Task>();

        foreach (string url in urls)
        {
            tasks.Add(Task.Run(async () =>
            {
                await _semaphore.WaitAsync();
                try
                {
                    string result = await _httpClient.GetStringAsync(url);
                    lock (results)
                    {
                        results.Add(result);
                    }
                }
                finally
                {
                    _semaphore.Release();
                }
            }));
        }
        await Task.WhenAll(tasks);
        return results;
    }
}

//Example usage:
RateLimitedApiCaller rateLimitedCaller = new RateLimitedApiCaller(5); // Allows 5 concurrent requests
string[] urls = { /* ... your URLs ... */ };
List<string> results = await rateLimitedCaller.MakeRequestsAsync(urls.ToList());

```

This example leverages `SemaphoreSlim` to limit concurrent requests to `maxConcurrentRequests`.  The `SemaphoreSlim.WaitAsync()` method acquires a permit before making a request, and `SemaphoreSlim.Release()` releases the permit afterward. This ensures that no more than the specified number of requests are in flight simultaneously, preventing overloading of the target API or exceeding rate limits.  The `try-finally` block guarantees that the permit is always released, even if an exception occurs.


**3. Resource Recommendations:**

*   **Microsoft Docs on Asynchronous Programming:** Provides comprehensive documentation on the `async` and `await` keywords, task management, and related concepts in C#.
*   **"Concurrent Programming on Windows" by Joe Duffy:** A detailed resource exploring concurrency in Windows systems, relevant for understanding the underlying mechanisms.
*   **"CLR via C#" by Jeffrey Richter:** Offers in-depth insights into the Common Language Runtime (CLR) and how it handles threading and asynchronous operations.  It helps understand the underlying mechanics.


These resources provide foundational knowledge and advanced techniques for handling concurrent operations efficiently and safely within C# .NET Core applications. Remember that choosing the right approach requires a thorough understanding of the dependencies between API calls and resource limitations.  Incorrectly implemented concurrency can lead to unforeseen performance issues and application instability.  Thorough testing and careful consideration of error handling are crucial in production environments.
