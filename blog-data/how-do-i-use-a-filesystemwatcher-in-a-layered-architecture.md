---
title: "How do I use a FileSystemWatcher in a layered architecture?"
date: "2024-12-23"
id: "how-do-i-use-a-filesystemwatcher-in-a-layered-architecture"
---

Alright,  I've grappled with file system monitoring in layered architectures more times than I care to count, and it's rarely as straightforward as the tutorials suggest. The key challenge isn't just *using* a `FileSystemWatcher`, it's doing it cleanly without creating a tangled mess of dependencies and breaking your carefully crafted separation of concerns. Over the years, I've found that a pragmatic, event-driven approach, coupled with careful encapsulation, generally works best.

The core issue revolves around where the `FileSystemWatcher` lives and how it communicates changes back to the higher layers of your application. If you directly instantiate the watcher in your business logic or presentation layers, you're setting yourself up for tight coupling and testing nightmares. Ideally, the `FileSystemWatcher` itself should be tucked away within an infrastructure layer, acting as a kind of "data source" for your application, much like a database or an external api.

Here's my take, distilled from a few projects that have thankfully made it to the light of day:

**The Core Concepts**

1.  **Infrastructure Layer Responsibility:** The `FileSystemWatcher`, being a technology-specific detail, should reside solely within the infrastructure layer. This layer is responsible for interacting with the external world (in this case, the file system) and converting that interaction into something the application layer can consume.

2.  **Event-Driven Communication:** The best way to communicate file system changes upwards is via events. This decouples the monitoring logic from the processing logic, making both more maintainable and testable. The infrastructure layer raises events when changes occur, and the application layer subscribes to these events to perform actions.

3.  **Data Transfer Objects (DTOs):** Avoid passing raw `FileSystemEventArgs` or `RenamedEventArgs` objects directly up to your application layer. Instead, use DTOs that contain only the necessary information. This prevents your application layer from becoming dependent on the specifics of the file system event arguments.

4.  **Dependency Injection:** Always, always use dependency injection (DI) to provide the watcher implementation to your infrastructure and application layers. This allows for easy swapping of implementations and simplifies testing.

**Example Scenario**

Let’s say we have a scenario where an application monitors a directory for new CSV files, processes them, and then stores the processed data. Here’s how I’d break that down in practice.

**Infrastructure Layer (C# Example):**

```csharp
using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;

namespace FileMonitor.Infrastructure
{
    public class FileSystemEventAggregator
    {
        private readonly FileSystemWatcher _watcher;
        public event EventHandler<FileChangeDto> FileChanged;

        public FileSystemEventAggregator(string directoryPath, string fileFilter)
        {
            _watcher = new FileSystemWatcher(directoryPath, fileFilter)
            {
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite,
                IncludeSubdirectories = false,
                EnableRaisingEvents = true
            };

            _watcher.Created += OnFileCreated;
            _watcher.Changed += OnFileChanged;
            _watcher.Deleted += OnFileDeleted;
            _watcher.Renamed += OnFileRenamed;
        }

        protected virtual void OnFileCreated(object sender, FileSystemEventArgs e)
        {
            FileChanged?.Invoke(this, new FileChangeDto(e.FullPath, "created"));
        }

        protected virtual void OnFileChanged(object sender, FileSystemEventArgs e)
        {
            FileChanged?.Invoke(this, new FileChangeDto(e.FullPath, "changed"));
        }

        protected virtual void OnFileDeleted(object sender, FileSystemEventArgs e)
        {
            FileChanged?.Invoke(this, new FileChangeDto(e.FullPath, "deleted"));
        }

        protected virtual void OnFileRenamed(object sender, RenamedEventArgs e)
        {
            FileChanged?.Invoke(this, new FileChangeDto(e.FullPath, "renamed", e.OldFullPath));
        }
    }

    public class FileChangeDto
    {
        public string FilePath { get; }
        public string ChangeType { get; }
        public string OldFilePath { get; }

        public FileChangeDto(string filePath, string changeType, string oldFilePath = null)
        {
            FilePath = filePath;
            ChangeType = changeType;
            OldFilePath = oldFilePath;
        }
    }
}
```

In this example, `FileSystemEventAggregator` encapsulates the `FileSystemWatcher`. It translates the events into a more generic `FileChangeDto`, making it easier to consume. Note that we’re utilizing `EventHandler` which means that a null reference check (using the `?.`) is required before raising an event, which prevents errors when no subscribers exist.

**Application Layer (C# Example):**

```csharp
using FileMonitor.Infrastructure;
using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace FileMonitor.Application
{
    public class FileProcessor
    {
        private readonly FileSystemEventAggregator _eventAggregator;

        public FileProcessor(FileSystemEventAggregator eventAggregator)
        {
            _eventAggregator = eventAggregator;
            _eventAggregator.FileChanged += HandleFileChange;
        }

        private void HandleFileChange(object sender, FileChangeDto e)
        {
            if (e.ChangeType == "created")
            {
                ProcessFileAsync(e.FilePath);
            }
            else if (e.ChangeType == "changed")
            {
                ProcessChangedFileAsync(e.FilePath);
            }
            else if (e.ChangeType == "deleted")
            {
              Console.WriteLine($"File {e.FilePath} was deleted");
            }
            else if(e.ChangeType == "renamed")
            {
                Console.WriteLine($"File {e.OldFilePath} was renamed to {e.FilePath}");
            }
        }

        private async Task ProcessFileAsync(string filePath)
        {
          // Here would be logic to process a new file

            Console.WriteLine($"Processing new file: {filePath}");
        }
        private async Task ProcessChangedFileAsync(string filePath)
        {
            // Logic to handle changes to an existing file
            Console.WriteLine($"File changed: {filePath}");
        }

        //Other methods to handle processed data.
    }
}

```

Here, the `FileProcessor` class receives an instance of `FileSystemEventAggregator` via its constructor, and subscribes to the `FileChanged` event to process file-related operations. The `HandleFileChange` event handler switches logic based on the `ChangeType` property of our DTO.

**Dependency Injection and Composition:**

This is where DI comes into its own. Assuming your application uses a basic container, you'd set things up something like this (very basic C# example):

```csharp
using FileMonitor.Application;
using FileMonitor.Infrastructure;
using Microsoft.Extensions.DependencyInjection;
using System;

public class Program
{
    public static void Main(string[] args)
    {
        var services = new ServiceCollection();
        services.AddSingleton(provider =>
          new FileSystemEventAggregator("/path/to/watch", "*.csv")); //Configuration path and filter

        services.AddSingleton<FileProcessor>(); // Application

         var serviceProvider = services.BuildServiceProvider();

         var fileProcessor = serviceProvider.GetRequiredService<FileProcessor>();

        Console.WriteLine("File watcher started. Press any key to exit");
        Console.ReadKey();
    }
}
```

In this example, a basic service collection is created, the required classes for infrastructure and application are set up, and `FileProcessor` can then be easily instantiated to handle the logic defined above. The important thing to take away is that `FileProcessor` does not directly depend on the implementation of `FileSystemWatcher`. This makes it easy to swap out different monitoring methods without changing the core logic.

**Further Considerations**

*   **Error Handling:** Robust error handling is crucial, especially when dealing with file system operations. Log exceptions appropriately within the infrastructure layer. I've found that using a logging library that adheres to structured logging practices helps.

*   **Testing:** Because we've decoupled the application logic, we can easily mock `FileSystemEventAggregator` and test `FileProcessor` in isolation without hitting the file system.

*   **Scalability:** If dealing with a very high volume of file system events, consider using a message queue (e.g., RabbitMQ or Kafka) between the infrastructure and application layers to avoid overwhelming the application and also ensure no message is lost. The infrastructure layer publishes change events to the message queue and the application layer consumes these events asynchronously.

**Recommended Resources**

For more comprehensive understanding of these concepts, I'd recommend:

*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software"** by Eric Evans. This is crucial for understanding the importance of separating concerns and modeling your domain appropriately.

*   **"Patterns of Enterprise Application Architecture"** by Martin Fowler. This book gives excellent practical insights and patterns on working with the architecture of complex software.

*   **"Dependency Injection in .NET"** by Mark Seemann. A deep dive into DI best practices with many practical examples.

Using a layered approach with a clean separation of concerns is the most resilient strategy for handling file system monitoring. It makes your code easier to understand, easier to test, and easier to adapt to future changes. This approach has served me well over numerous projects, and I hope it’ll be of help to you too.
