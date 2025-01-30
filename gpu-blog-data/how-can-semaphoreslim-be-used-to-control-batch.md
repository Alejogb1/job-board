---
title: "How can SemaphoreSlim be used to control batch execution?"
date: "2025-01-30"
id: "how-can-semaphoreslim-be-used-to-control-batch"
---
A fundamental challenge in concurrent programming is limiting the number of operations executing simultaneously. In scenarios involving batch processing, where multiple tasks can be grouped for efficiency, controlling the level of parallelism is crucial to prevent resource exhaustion and maintain system stability. SemaphoreSlim, a lightweight synchronization primitive in .NET, offers an effective mechanism to manage this. I've found it particularly useful in scenarios where a large set of data needs processing by a limited number of worker threads.

Specifically, a `SemaphoreSlim` instance maintains a count, and this count represents the number of available resources. A thread attempting to enter a critical section protected by the semaphore will either decrement the count (if it's positive) and proceed, or be blocked until the count becomes positive again through a signal from another thread. This characteristic allows me to control how many concurrent batch processing operations can run at any given time. It's far more efficient than using heavier locking mechanisms such as `Monitor`, particularly when contention is expected to be moderate to high. The crucial distinction with `Semaphore` is its optimized usage with asynchronous code. Using `WaitAsync` and `Release` makes it possible to prevent blocking the current thread, allowing for highly scalable concurrency patterns.

Here’s an example demonstrating how I've used `SemaphoreSlim` to limit the number of concurrent file processing operations:

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

public class FileProcessor
{
    private readonly SemaphoreSlim _semaphore;
    private readonly int _maxConcurrency;

    public FileProcessor(int maxConcurrency)
    {
        _maxConcurrency = maxConcurrency;
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency); // Initial & max = concurrency
    }


    public async Task ProcessFilesAsync(IEnumerable<string> filePaths)
    {
        var tasks = new List<Task>();
        foreach (var filePath in filePaths)
        {
            tasks.Add(ProcessFileWithSemaphoreAsync(filePath));
        }

        await Task.WhenAll(tasks);
    }


    private async Task ProcessFileWithSemaphoreAsync(string filePath)
    {
         await _semaphore.WaitAsync(); // Wait for a resource to become available
        try
        {
            // Simulate file processing
            Console.WriteLine($"Processing: {filePath}, Thread: {Thread.CurrentThread.ManagedThreadId}");
            await Task.Delay(new Random().Next(100, 500));
        }
        finally
        {
            _semaphore.Release(); // Release resource when task is complete
        }

    }
}

public class Program
{
    public static async Task Main(string[] args)
    {
          var filePaths = new List<string>();
          for (int i = 0; i < 20; i++) {
              filePaths.Add($"file_{i}.txt");
          }
        int maxConcurrency = 5;
        var processor = new FileProcessor(maxConcurrency);
        await processor.ProcessFilesAsync(filePaths);
        Console.WriteLine("File processing complete.");
    }
}
```

In this example, the `FileProcessor` class is instantiated with a specified maximum concurrency level. The `SemaphoreSlim` is initialized with the same number of permits as the concurrency limit. The `ProcessFileWithSemaphoreAsync` method first waits on the semaphore. This ensures that the number of concurrent file processing tasks never exceeds the defined maximum concurrency. Once processing is complete (whether successful or failed), the `finally` block releases the semaphore, allowing another waiting task to proceed. This pattern of `WaitAsync` followed by `Release` inside a `try-finally` block is standard to prevent resource leakage in cases of uncaught exceptions. The core concept here is to control access to a limited pool of resources.

Another scenario where I’ve leveraged `SemaphoreSlim` is when needing to throttle access to a third-party API with rate limits. Instead of trying to manually queue requests or perform complex timing calculations, the semaphore acts as a gatekeeper. Consider this code:

```csharp
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

public class ApiClient
{
    private readonly HttpClient _httpClient = new HttpClient();
    private readonly SemaphoreSlim _semaphore;
    private readonly int _maxConcurrentRequests;

    public ApiClient(int maxConcurrentRequests)
    {
        _maxConcurrentRequests = maxConcurrentRequests;
        _semaphore = new SemaphoreSlim(maxConcurrentRequests, maxConcurrentRequests);
    }

    public async Task<string> FetchDataAsync(string apiUrl)
    {
         await _semaphore.WaitAsync();
        try
        {
             Console.WriteLine($"Fetching data from {apiUrl} , Thread: {Thread.CurrentThread.ManagedThreadId} ");
            var response = await _httpClient.GetAsync(apiUrl);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
        finally
        {
             _semaphore.Release();
        }
    }
}

public class Program
{
     public static async Task Main(string[] args)
    {
         var apiEndpoints = new List<string> {
            "https://jsonplaceholder.typicode.com/todos/1",
            "https://jsonplaceholder.typicode.com/todos/2",
            "https://jsonplaceholder.typicode.com/todos/3",
            "https://jsonplaceholder.typicode.com/todos/4",
            "https://jsonplaceholder.typicode.com/todos/5",
            "https://jsonplaceholder.typicode.com/todos/6",
            "https://jsonplaceholder.typicode.com/todos/7",
            "https://jsonplaceholder.typicode.com/todos/8",
            "https://jsonplaceholder.typicode.com/todos/9",
            "https://jsonplaceholder.typicode.com/todos/10",
            "https://jsonplaceholder.typicode.com/todos/11",
             "https://jsonplaceholder.typicode.com/todos/12",
             "https://jsonplaceholder.typicode.com/todos/13",
             "https://jsonplaceholder.typicode.com/todos/14",
             "https://jsonplaceholder.typicode.com/todos/15",
             "https://jsonplaceholder.typicode.com/todos/16",
             "https://jsonplaceholder.typicode.com/todos/17",
             "https://jsonplaceholder.typicode.com/todos/18",
             "https://jsonplaceholder.typicode.com/todos/19",
              "https://jsonplaceholder.typicode.com/todos/20"

          };


          int maxConcurrency = 3; // Limit concurrent API requests
          var apiClient = new ApiClient(maxConcurrency);
          var tasks = new List<Task<string>>();
         foreach(var url in apiEndpoints){
             tasks.Add(apiClient.FetchDataAsync(url));
         }
           await Task.WhenAll(tasks);

        Console.WriteLine("API requests complete.");

    }
}
```

Here, the `ApiClient` limits the number of concurrent requests that can be made to the external API. The `FetchDataAsync` method utilizes the same semaphore pattern: waiting for a permit before performing the API call and releasing it afterwards. This effectively prevents overloading the third-party API, respecting its rate limits. The use of `Task.WhenAll` ensures that the main execution flow is not held up, and the program continues when all tasks have been completed. I usually set the `maxConcurrentRequests` based on the documented API limits or empirical testing to determine the optimal rate.

Finally, consider a data ingestion pipeline. Instead of batching only at the source, controlling the downstream processing concurrently using a semaphore can provide more fine-grained control. Imagine you're reading data from a queue and need to limit the number of records processed concurrently:

```csharp
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

public class DataProcessor
{
   private readonly SemaphoreSlim _semaphore;
    private readonly int _maxConcurrency;
    private readonly BlockingCollection<string> _dataQueue;

    public DataProcessor(int maxConcurrency)
    {
        _maxConcurrency = maxConcurrency;
        _semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
        _dataQueue = new BlockingCollection<string>();
    }

     public void AddDataToQueue(string data)
    {
        _dataQueue.Add(data);
    }

    public async Task ProcessDataAsync()
    {
         var tasks = new List<Task>();
         while (!_dataQueue.IsCompleted)
            {
                if(_dataQueue.TryTake(out var item)){
                    tasks.Add(ProcessDataItemWithSemaphoreAsync(item));
                }
                else{
                    await Task.Delay(100);
                }
            }
        await Task.WhenAll(tasks);
    }


   private async Task ProcessDataItemWithSemaphoreAsync(string data)
    {
        await _semaphore.WaitAsync();
        try
        {
            Console.WriteLine($"Processing data: {data} , Thread: {Thread.CurrentThread.ManagedThreadId}");
            await Task.Delay(new Random().Next(100, 300)); // Simulate processing
        }
        finally
        {
             _semaphore.Release();
        }
    }

    public void CompleteAdding(){
         _dataQueue.CompleteAdding();
    }
}


public class Program
{
    public static async Task Main(string[] args)
    {
         var processor = new DataProcessor(4);

        // Add simulated data to the queue
         for (int i = 0; i < 20; i++)
        {
            processor.AddDataToQueue($"Data Item {i}");
        }
         processor.CompleteAdding(); // Signal no more data to be added
        await processor.ProcessDataAsync();
        Console.WriteLine("Data processing complete.");
    }
}
```

In this scenario, the `DataProcessor` utilizes a `BlockingCollection` to act as a queue. Data items are added, and the `ProcessDataAsync` method continuously checks the queue. If an item is available, it is dequeued and processed using the semaphore. The semaphore limits the concurrent processing of these data items. The `CompleteAdding` method signals the end of adding data to the queue.  This demonstrates that the semaphore can be applied not just to simple lists, but to more complex, producer/consumer scenarios.

For further exploration of this concurrency pattern, I recommend consulting resources such as "Concurrency in C# Cookbook" for patterns involving asynchronous operations and task parallel library (TPL). "Programming with C#" provides a more general background in asynchronous programming, useful in understanding the use case of the SemaphoreSlim. Further understanding of multithreading is useful too. "C# 8.0 and .NET Core 3.0 – Modern Cross-Platform Development" and "CLR via C#" provides a deeper background of CLR execution.
