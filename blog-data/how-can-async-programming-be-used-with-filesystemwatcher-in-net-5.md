---
title: "How can async programming be used with FileSystemWatcher in .NET 5?"
date: "2024-12-23"
id: "how-can-async-programming-be-used-with-filesystemwatcher-in-net-5"
---

Let's dive straight in. I recall a particularly gnarly project back in 2019 – a real-time data ingestion pipeline – where we were constantly battling performance bottlenecks. A key component was watching a directory for new files, processing them, and then moving them to an archive location. Initially, we'd implemented this using the classic synchronous approach with `FileSystemWatcher`, and well, it wasn't pretty. The application became unresponsive during heavy file creation periods, leading to data loss and a generally unhappy user base. That experience underscored for me the necessity of truly understanding and leveraging asynchronous programming, especially when interacting with I/O-bound operations like file system events.

The core problem with a synchronous approach, as most of you probably know, is the blocking nature of operations. When a `FileSystemWatcher` detects a change, the event handler is executed on the thread that raised the event. If that handler performs lengthy operations (like complex data processing), the thread gets tied up, preventing it from receiving further events until the handler completes. This leads to the very issue we encountered: missed events and delayed processing. Async programming elegantly sidesteps this by allowing the thread to continue receiving and reacting to events while a long-running operation is being handled in the background.

The solution in .NET 5, and indeed most .NET versions since, involves several techniques working in tandem. First and foremost is the use of the `async` and `await` keywords, which enable asynchronous execution without relying on complex thread management. Secondly, proper handling of concurrency is paramount, especially when multiple file system events trigger close in time. And finally, using appropriate data structures to manage and schedule these asynchronous operations to prevent race conditions.

Here's how I’d structure an asynchronous `FileSystemWatcher` workflow, demonstrating a robust and scalable approach, coupled with actual code examples:

**Example 1: Basic Asynchronous Event Handling**

This snippet demonstrates how to make a basic event handler asynchronous, making it non-blocking. This approach is suitable for simple scenarios where minimal processing is required.

```csharp
using System;
using System.IO;
using System.Threading.Tasks;

public class AsyncFileSystemWatcher
{
    private FileSystemWatcher _watcher;
    private string _directoryToWatch;

    public AsyncFileSystemWatcher(string directoryToWatch)
    {
        _directoryToWatch = directoryToWatch;
        _watcher = new FileSystemWatcher(_directoryToWatch);
        _watcher.Created += OnFileCreatedAsync;
        _watcher.EnableRaisingEvents = true;
    }

    private async void OnFileCreatedAsync(object sender, FileSystemEventArgs e)
    {
        Console.WriteLine($"File created: {e.FullPath}");
        await ProcessFileAsync(e.FullPath); // Call asynchronous processing method
    }

    private async Task ProcessFileAsync(string filePath)
    {
        await Task.Delay(1000);  // Simulate processing delay
        Console.WriteLine($"Finished processing: {filePath}");
    }

    public void StopWatching()
    {
        _watcher.EnableRaisingEvents = false;
    }

    public static void Main(string[] args)
    {
        string watchedDirectory = "C:\\TestFolder"; // Replace with your test folder path
        if (!Directory.Exists(watchedDirectory))
        {
            Directory.CreateDirectory(watchedDirectory);
        }
        var watcher = new AsyncFileSystemWatcher(watchedDirectory);
        Console.WriteLine("Watching directory, press any key to stop...");
        Console.ReadKey();
        watcher.StopWatching();
    }
}
```

In this example, the `OnFileCreatedAsync` handler uses the `async void` pattern. While convenient, it doesn't allow direct awaiting, making error handling trickier. Note that this is suitable here as we're attaching to event handler that does not return a value. Also, remember that `async void` should be used cautiously and only for event handlers or other top-level asynchronous methods, and the underlying `ProcessFileAsync` method uses `async Task` pattern, allowing for composability.

**Example 2: Using a Concurrent Queue for Event Processing**

For more complex workloads or when events are likely to occur in rapid succession, a concurrent queue combined with a dedicated worker thread proves beneficial. This helps decouple event arrival from actual processing, ensuring that no events are lost.

```csharp
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

public class ConcurrentQueueWatcher
{
    private FileSystemWatcher _watcher;
    private BlockingCollection<FileSystemEventArgs> _eventQueue;
    private string _directoryToWatch;
    private CancellationTokenSource _cts;

    public ConcurrentQueueWatcher(string directoryToWatch)
    {
        _directoryToWatch = directoryToWatch;
        _eventQueue = new BlockingCollection<FileSystemEventArgs>();
        _cts = new CancellationTokenSource();
        _watcher = new FileSystemWatcher(_directoryToWatch);
        _watcher.Created += OnFileCreated;
        _watcher.EnableRaisingEvents = true;
        Task.Run(() => ProcessEventsAsync(_cts.Token));
    }

    private void OnFileCreated(object sender, FileSystemEventArgs e)
    {
        _eventQueue.Add(e);
    }

    private async Task ProcessEventsAsync(CancellationToken cancellationToken)
    {
        foreach (var fileEvent in _eventQueue.GetConsumingEnumerable(cancellationToken))
        {
           Console.WriteLine($"File Created: {fileEvent.FullPath}");
           await ProcessFileAsync(fileEvent.FullPath);
        }
    }


    private async Task ProcessFileAsync(string filePath)
    {
        await Task.Delay(2000);  // Simulate more involved processing
        Console.WriteLine($"Finished processing: {filePath}");
    }

    public void StopWatching()
    {
        _watcher.EnableRaisingEvents = false;
        _cts.Cancel();
        _eventQueue.CompleteAdding();
    }


        public static void Main(string[] args)
    {
        string watchedDirectory = "C:\\TestFolder"; // Replace with your test folder path
       if (!Directory.Exists(watchedDirectory))
       {
           Directory.CreateDirectory(watchedDirectory);
       }
        var watcher = new ConcurrentQueueWatcher(watchedDirectory);
         Console.WriteLine("Watching directory with queue, press any key to stop...");
        Console.ReadKey();
        watcher.StopWatching();

    }
}
```

In this snippet, `BlockingCollection<FileSystemEventArgs>` is used, allowing multiple event threads to add events without blocking, while the consuming task processes them sequentially. This effectively decouples the watcher and the processor, improving robustness and responsiveness. We also introduce the use of cancellation tokens for clean shutdown.

**Example 3: Using Task Parallel Library (TPL) for Parallel Processing**

If each file processing operation is independent, you can use the Task Parallel Library (TPL) to process multiple files concurrently. This improves throughput on multi-core systems, and is beneficial when each file requires a significant amount of independent processing.

```csharp
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;

public class TplWatcher
{
    private FileSystemWatcher _watcher;
    private BlockingCollection<FileSystemEventArgs> _eventQueue;
    private string _directoryToWatch;

    public TplWatcher(string directoryToWatch)
    {
        _directoryToWatch = directoryToWatch;
        _eventQueue = new BlockingCollection<FileSystemEventArgs>();
        _watcher = new FileSystemWatcher(_directoryToWatch);
        _watcher.Created += OnFileCreated;
        _watcher.EnableRaisingEvents = true;
        Task.Run(ProcessEventsAsync);
    }

    private void OnFileCreated(object sender, FileSystemEventArgs e)
    {
        _eventQueue.Add(e);
    }

     private async Task ProcessEventsAsync()
        {
            foreach(var fileEvent in _eventQueue.GetConsumingEnumerable())
            {
                 Console.WriteLine($"File created : {fileEvent.FullPath}");
                 await ProcessFileParallelAsync(fileEvent.FullPath);

            }
        }


    private async Task ProcessFileParallelAsync(string filePath)
    {
       await Task.Run(async () =>
       {
           await Task.Delay(3000);  // Simulate intensive processing
          Console.WriteLine($"Finished processing: {filePath} on thread {System.Threading.Thread.CurrentThread.ManagedThreadId}");
       });
    }



    public void StopWatching()
    {
        _watcher.EnableRaisingEvents = false;
        _eventQueue.CompleteAdding();
    }

        public static void Main(string[] args)
        {
           string watchedDirectory = "C:\\TestFolder"; // Replace with your test folder path
           if (!Directory.Exists(watchedDirectory))
           {
               Directory.CreateDirectory(watchedDirectory);
           }
           var watcher = new TplWatcher(watchedDirectory);
           Console.WriteLine("Watching directory with parallel tasks, press any key to stop...");
           Console.ReadKey();
            watcher.StopWatching();
        }
}
```
In this version, instead of processing files sequentially from the queue, the `ProcessFileParallelAsync` method uses `Task.Run` to offload the work to the thread pool for concurrent execution.

**Recommendations and Key Considerations:**

While async is powerful, it introduces complexity. You need to be careful about issues such as:

*   **Error handling:** You need to implement robust error handling for each level of asynchronous operation. Consider using `try-catch` blocks and logging any exceptions that occur, rather than blindly swallowing errors.
*   **Resource management:** Ensure that you properly dispose of any resources (such as file handles) used in your processing methods using the `using` statement, even when dealing with asynchronous tasks.
*   **Cancellation:** Implement cancellation tokens if you need to be able to stop ongoing tasks if required by the application.
*   **Concurrency Limits:** Even with asynchronous execution, if your processing methods involve I/O operations to the same resource, you could overload them. Consider introducing throttling or limiting the level of concurrency if required.

For those seeking a deeper dive into the topic, I strongly recommend picking up "Concurrency in .NET" by Stephen Cleary, which provides an excellent, detailed explanation of asynchronous programming. "C# 10 and .NET 6 Modern Cross-Platform Development" by Mark J. Price is also a fantastic resource for understanding how asynchronous patterns are integrated into the .NET ecosystem. Microsoft's official documentation on Task Parallel Library (TPL) is also invaluable for mastering the use of tasks, `async` and `await`.

Using asynchronous programming with `FileSystemWatcher` is not just about performance, it's about building robust and responsive applications. It allows for better resource utilization and graceful handling of I/O bound operations which are commonplace in modern software development. The scenarios and code samples provided illustrate just some of the common approaches that I and other developers have used to overcome this challenge. Ultimately, understanding your workload, choosing the correct approach, and implementing it thoughtfully can make all the difference in the reliability of your applications.
