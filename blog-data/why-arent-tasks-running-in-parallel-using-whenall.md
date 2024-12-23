---
title: "Why aren't tasks running in parallel using .WhenAll?"
date: "2024-12-23"
id: "why-arent-tasks-running-in-parallel-using-whenall"
---

Let’s tackle this. I’ve seen this specific situation arise more than a few times, and it usually boils down to a misunderstanding of how `Task.WhenAll` interacts with its underlying task execution infrastructure, particularly in asynchronous programming. The common misconception stems from the belief that simply wrapping multiple async operations within `Task.WhenAll` magically forces them onto different threads, which is not always the case.

The key is to understand that `Task.WhenAll` itself doesn’t create parallelism. Instead, it's a coordination mechanism. It waits for a collection of tasks to complete, returning a single task that resolves when all its children have resolved. The parallelism, or lack thereof, happens *within* the tasks that are being awaited, not within `Task.WhenAll` itself.

Let's break down a few common culprits why things may appear to run sequentially when you expected parallelism:

**1. The Asynchronous Operations are Not Truly Concurrent:**

This is the most prevalent issue. If the asynchronous operations wrapped in `Task.WhenAll` are not designed to run concurrently, they will execute sequentially regardless of the coordination mechanism. This commonly occurs when the underlying I/O bound operations are handled by the same single thread.

To illustrate this, consider a scenario where I once had to deal with a batch processing system that was fetching data from a database. In the initial naive attempt, the fetching logic used a synchronous database query operation wrapped in a `Task.Run`. While it *looked* async from the surface, it was not truly concurrent. Here’s a simplified representation of what the initial faulty code looked like:

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;

public class DatabaseFetcher {
    public int FetchDataSync(int id) {
       Thread.Sleep(1000);  // Simulate synchronous database query
       Console.WriteLine($"Fetched data for ID: {id} on thread: {Thread.CurrentThread.ManagedThreadId}");
        return id * 2;
    }

   public Task<int> FetchDataAsync(int id) {
       return Task.Run(() => FetchDataSync(id));
   }
}

public static class Example {
    public static async Task Main() {
        var fetcher = new DatabaseFetcher();
        List<Task<int>> tasks = new List<Task<int>>();

        for (int i = 1; i <= 5; i++) {
            tasks.Add(fetcher.FetchDataAsync(i));
        }

        var results = await Task.WhenAll(tasks);

        foreach (var result in results) {
             Console.WriteLine($"Result: {result}");
         }
    }
}
```

In this example, even though `FetchDataAsync` returns a `Task`, the synchronous `Thread.Sleep` within the `FetchDataSync` method block the thread, rendering `Task.WhenAll` unable to run anything concurrently. Each call to `FetchDataAsync` essentially queues a job on the thread pool, but the actual execution is serialized due to the synchronous nature of the database query.

**2. Resource Contention and Bottlenecks:**

Parallelism doesn’t guarantee performance improvements if the system has resource limitations. If the concurrent operations are all trying to access the same resource, such as a single file, database connection, or network resource, you introduce contention that reduces or eliminates parallel execution.

Imagine another scenario from a data processing project where we were trying to concurrently write logs to a single shared file. Initially, it appeared as though the tasks were running together, but the performance was atrocious due to the file access contention. Here’s how the code looked conceptually:

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Threading;

public class LogWriter {
    private readonly string _filePath;

    public LogWriter(string filePath) {
        _filePath = filePath;
    }

    public async Task WriteLogAsync(string message)
    {
          using (StreamWriter writer = new StreamWriter(_filePath, true))
          {
                await writer.WriteLineAsync($"{DateTime.Now}: {message} Thread: {Thread.CurrentThread.ManagedThreadId}");
              await Task.Delay(10); // Simulate some IO delay
            }

        }
}


public static class Example2 {
    public static async Task Main() {
        var logWriter = new LogWriter("logs.txt");
        List<Task> tasks = new List<Task>();

        for (int i = 1; i <= 5; i++)
        {
            tasks.Add(logWriter.WriteLogAsync($"Log message {i}"));
        }

        await Task.WhenAll(tasks);

       Console.WriteLine("Log writes completed.");
    }
}
```

In this case, while the tasks are asynchronous, the file I/O operations are contending for the same resource, leading to substantial serialization. Each task has to acquire a lock on the file to perform the write operation, and this lock contention severely impacts the perceived concurrency.

**3. Improper Use of Asynchronous APIs:**

Sometimes the problem lies in the way the asynchronous APIs are being used. If you are wrapping synchronous code with `Task.Run`, that won't magically make it faster. I've often seen code that has a mix of async and sync calls, with blocking synchronous calls interspersed, or methods returning tasks based on synchronous code when it's avoidable. This leads to inefficiencies and potentially blocking threads from the thread pool, causing poor performance and diminished concurrency.

Here's an illustration where even though an operation *can* be executed async (using an existing API that supports asynchronous execution), a synchronous version was chosen by mistake leading to blocking the current thread and rendering `Task.WhenAll` less efficient:

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Net.Http;
using System.Threading;

public class WebFetcher {

    private readonly HttpClient _httpClient = new HttpClient();


    public string FetchDataSync(string url) {
      Console.WriteLine($"Fetching synchronously from {url} on thread: {Thread.CurrentThread.ManagedThreadId}");
        return _httpClient.GetStringAsync(url).GetAwaiter().GetResult(); //blocking!
    }

    public Task<string> FetchDataAsync(string url) {
        return Task.Run(() => FetchDataSync(url));
    }
}
public static class Example3 {
    public static async Task Main() {
        var webFetcher = new WebFetcher();
        List<Task<string>> tasks = new List<Task<string>>();
        string[] urls = { "https://www.example.com", "https://www.microsoft.com","https://www.google.com" };
        foreach (var url in urls) {
            tasks.Add(webFetcher.FetchDataAsync(url));
        }
        var results = await Task.WhenAll(tasks);
        foreach(var result in results){
            Console.WriteLine($"Result length: {result.Length}");
        }
    }
}
```

Notice the `.GetAwaiter().GetResult()` in the `FetchDataSync` method. This synchronously blocks, making the operation that is supposedly running asynchronously, actually a synchronous one, and impacting the concurrent execution.

**Recommendations and Further Reading**

To effectively leverage parallelism with `Task.WhenAll`, it's crucial to ensure the individual tasks being awaited are truly non-blocking and asynchronous. You must avoid blocking synchronous operations, and manage resource access to prevent contention. Here are some valuable resources that I found helpful in my past experience:

*   **"Concurrency in C# Cookbook" by Stephen Cleary:** This is an excellent, hands-on guide for understanding asynchronous programming concepts and patterns in C#, including detailed explanations of how `async` and `await` work. It's a must-read for any C# developer dealing with concurrency.
*   **"Programming Microsoft Async" by Stephen Toub:** This deep-dive resource from Microsoft is essential for understanding the intricacies of asynchronous programming in the .NET environment, offering insights into the internal workings of the TPL and the asynchronous execution model.
*  **"Patterns of Enterprise Application Architecture" by Martin Fowler:** Though not solely focused on asynchronous programming, this book addresses patterns for concurrency and resource management that are highly relevant when dealing with complex systems where parallel operations might impact shared resources.

In summary, `Task.WhenAll` is a vital tool for managing parallel operations, but it does not magically create concurrency. The real parallelism is determined by how the underlying tasks are implemented and how well their resource usage is handled. If you're seeing tasks running sequentially when they shouldn't be, it's always worth taking a detailed look at the individual operations that make up the tasks, the resources those operations need and to ensure you are using the correct asynchronous implementations of your I/O operations.
