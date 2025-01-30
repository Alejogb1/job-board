---
title: "How can I use `FileSystemWatcher` asynchronously in .NET 5?"
date: "2025-01-30"
id: "how-can-i-use-filesystemwatcher-asynchronously-in-net"
---
The `FileSystemWatcher` class in .NET, by default, operates synchronously, raising events on the thread that invokes the `WaitForChanged` method or, in the case of continuous monitoring, on internal threads managed by the `FileSystemWatcher`. This characteristic presents a challenge when aiming to integrate file system monitoring into responsive user interfaces or long-running background tasks, as these operations should typically not block the main thread. My experience with building a large-scale document processing application required a robust and asynchronous file monitoring solution, prompting me to delve into effective strategies for utilizing `FileSystemWatcher` asynchronously.

The key issue is that while `FileSystemWatcher` triggers events, those events often require further processing that can be I/O bound or computationally intensive. Executing this processing synchronously within the event handler blocks the `FileSystemWatcher`'s internal mechanisms, potentially leading to missed events, unresponsive applications, and overall poor performance. To mitigate these issues, a producer-consumer pattern with asynchronous operations is a suitable and effective approach. This pattern involves collecting events from the `FileSystemWatcher` and dispatching them to a worker thread or a queue for asynchronous processing.

Several approaches exist for achieving this asynchronous behavior. My favored method involves using `System.Threading.Channels` introduced in .NET Core 3.0 and later. These channels provide a robust, thread-safe way to transfer data between a producer (the `FileSystemWatcher` event handler) and a consumer (a background worker). This offers excellent performance characteristics and integrates well with .NET's asynchronous programming model. Alternatively, we could use `TPL Dataflow` blocks for a more complex workflow; however, `Channels` provides simpler, more directly controlled execution. I'll demonstrate this `Channels` pattern using examples with increasing complexity.

**Code Example 1: Basic Asynchronous Event Handling**

This example illustrates the fundamental use of `Channels` to offload event handling.

```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using System.Threading.Channels;

public class FileSystemMonitor
{
    private FileSystemWatcher _watcher;
    private Channel<FileSystemEventArgs> _channel;

    public FileSystemMonitor(string path)
    {
        _channel = Channel.CreateUnbounded<FileSystemEventArgs>();
        _watcher = new FileSystemWatcher(path);
        _watcher.Created += OnFileChanged;
        _watcher.Changed += OnFileChanged;
        _watcher.Deleted += OnFileChanged;
        _watcher.Renamed += OnFileRenamed; // Note the distinct handler
        _watcher.EnableRaisingEvents = true;
    }

    private void OnFileChanged(object sender, FileSystemEventArgs e)
    {
        _channel.Writer.TryWrite(e); // Non-blocking write to channel
    }

    private void OnFileRenamed(object sender, RenamedEventArgs e)
    {
        // Handle RenameEventArgs specifically as it has different properties
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Renamed,Path.GetDirectoryName(e.OldFullPath), Path.GetFileName(e.OldFullPath))); // Simulate a 'Deleted' event on the old filename
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Created,Path.GetDirectoryName(e.FullPath), Path.GetFileName(e.FullPath)));// Simulate a 'Created' event on the new filename
    }

    public async Task StartProcessingAsync()
    {
        await foreach (var eventArgs in _channel.Reader.ReadAllAsync())
        {
             await ProcessFileEventAsync(eventArgs);
        }
    }

   private async Task ProcessFileEventAsync(FileSystemEventArgs e)
    {
          await Task.Delay(100); // Simulate some processing
          Console.WriteLine($"File Event: {e.ChangeType}, {e.FullPath}");
    }

    public void Stop()
    {
        _watcher.EnableRaisingEvents = false;
         _channel.Writer.Complete(); // Signal the reader no further data.
         _watcher.Dispose();

    }
}

public static class Program {
    public static async Task Main(string[] args)
    {
        var monitor = new FileSystemMonitor(@"C:\temp"); // Replace with a valid directory
        Task monitoringTask = monitor.StartProcessingAsync();

        Console.WriteLine("Press any key to stop...");
        Console.ReadKey();

         monitor.Stop();
        await monitoringTask; //Await completion
    }

}
```

In this initial version, a `Channel<FileSystemEventArgs>` facilitates the transfer of event data from the `FileSystemWatcher` to a background process. The `TryWrite` method on the channel's writer prevents blocking the event handler. The `StartProcessingAsync` method uses `ReadAllAsync()` to consume items from the channel asynchronously. `ProcessFileEventAsync` simulates some I/O processing. The `OnFileRenamed` handler shows how a special event argument should be handled by creating a 'delete' and 'create' event to preserve simplicity of the handler.

**Code Example 2: Filtering and Throttling Events**

This example adds event filtering based on file extensions and throttling to prevent excessive processing in rapid change situations.

```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using System.Threading.Channels;
using System.Collections.Generic;
using System.Linq;

public class FileSystemMonitor
{
    private FileSystemWatcher _watcher;
    private Channel<FileSystemEventArgs> _channel;
    private HashSet<string> _allowedExtensions;
    private TimeSpan _throttleInterval;
    private DateTime _lastEventTime;

    public FileSystemMonitor(string path, IEnumerable<string> allowedExtensions, int throttleMilliseconds = 500)
    {
       _allowedExtensions = new HashSet<string>(allowedExtensions.Select(x => x.ToLowerInvariant()));
       _throttleInterval = TimeSpan.FromMilliseconds(throttleMilliseconds);
       _channel = Channel.CreateUnbounded<FileSystemEventArgs>();
       _watcher = new FileSystemWatcher(path);
       _watcher.Created += OnFileChanged;
       _watcher.Changed += OnFileChanged;
       _watcher.Deleted += OnFileChanged;
       _watcher.Renamed += OnFileRenamed;
       _watcher.EnableRaisingEvents = true;
       _lastEventTime = DateTime.MinValue; // Initialize
    }

    private void OnFileChanged(object sender, FileSystemEventArgs e)
    {
        if (_allowedExtensions.Count > 0 && !_allowedExtensions.Contains(Path.GetExtension(e.FullPath).ToLowerInvariant()))
            return;

         if( (DateTime.Now - _lastEventTime) < _throttleInterval )
            return;

         _lastEventTime = DateTime.Now;
         _channel.Writer.TryWrite(e);
    }

    private void OnFileRenamed(object sender, RenamedEventArgs e)
    {
        if(_allowedExtensions.Count > 0 && !_allowedExtensions.Contains(Path.GetExtension(e.OldFullPath).ToLowerInvariant()))
            return;

        if( (DateTime.Now - _lastEventTime) < _throttleInterval )
            return;

         _lastEventTime = DateTime.Now;
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Renamed,Path.GetDirectoryName(e.OldFullPath), Path.GetFileName(e.OldFullPath)));
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Created,Path.GetDirectoryName(e.FullPath), Path.GetFileName(e.FullPath)));
    }


    public async Task StartProcessingAsync()
    {
        await foreach (var eventArgs in _channel.Reader.ReadAllAsync())
        {
             await ProcessFileEventAsync(eventArgs);
        }
    }

   private async Task ProcessFileEventAsync(FileSystemEventArgs e)
    {
          await Task.Delay(100); // Simulate some processing
          Console.WriteLine($"File Event: {e.ChangeType}, {e.FullPath}");
    }

    public void Stop()
    {
        _watcher.EnableRaisingEvents = false;
        _channel.Writer.Complete();
         _watcher.Dispose();

    }
}


public static class Program {
    public static async Task Main(string[] args)
    {
        var monitor = new FileSystemMonitor(@"C:\temp", new[] { ".txt", ".log" }, 250); // Filter only txt and log files, throttle every 250ms
        Task monitoringTask = monitor.StartProcessingAsync();

        Console.WriteLine("Press any key to stop...");
        Console.ReadKey();

         monitor.Stop();
        await monitoringTask;
    }

}
```
This example introduces `_allowedExtensions`, a `HashSet<string>`, to restrict the monitored files based on their extensions. The `_throttleInterval` and `_lastEventTime` members are used to enforce a minimum time interval between events, reducing the load when there are frequent changes. This is especially helpful when dealing with rapid file updates.

**Code Example 3: Batch Processing of Events**

This example demonstrates batch processing, where multiple events are collected before processing, potentially improving efficiency for operations that benefit from handling multiple changes at once.

```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using System.Threading.Channels;
using System.Collections.Generic;
using System.Linq;

public class FileSystemMonitor
{
    private FileSystemWatcher _watcher;
    private Channel<FileSystemEventArgs> _channel;
    private TimeSpan _batchInterval;
    private int _batchSize;
    private DateTime _lastBatchTime;

    public FileSystemMonitor(string path, int batchSize = 5, int batchMilliseconds = 500)
    {
      _batchSize = batchSize;
      _batchInterval = TimeSpan.FromMilliseconds(batchMilliseconds);
       _channel = Channel.CreateUnbounded<FileSystemEventArgs>();
       _watcher = new FileSystemWatcher(path);
       _watcher.Created += OnFileChanged;
       _watcher.Changed += OnFileChanged;
       _watcher.Deleted += OnFileChanged;
       _watcher.Renamed += OnFileRenamed;
       _watcher.EnableRaisingEvents = true;
       _lastBatchTime = DateTime.MinValue; //Initialize
    }

    private void OnFileChanged(object sender, FileSystemEventArgs e)
    {
      _channel.Writer.TryWrite(e); // Write regardless of time. Batch processing does the time based restriction.
    }

    private void OnFileRenamed(object sender, RenamedEventArgs e)
    {
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Renamed,Path.GetDirectoryName(e.OldFullPath), Path.GetFileName(e.OldFullPath)));
        _channel.Writer.TryWrite(new FileSystemEventArgs(WatcherChangeTypes.Created,Path.GetDirectoryName(e.FullPath), Path.GetFileName(e.FullPath)));
    }

    public async Task StartProcessingAsync()
    {
       while(true)
       {
           List<FileSystemEventArgs> batch = await ReadBatchAsync();

           if(batch.Count == 0) //If the channel is empty, just loop. This may require a break clause.
              continue;

          await ProcessFileEventsAsync(batch);

           if(_channel.Reader.Completion.IsCompleted) //Stop if the channel has been completed
              break;
       }
    }
     private async Task<List<FileSystemEventArgs>> ReadBatchAsync()
    {
      var batch = new List<FileSystemEventArgs>();

      while(batch.Count < _batchSize && (DateTime.Now - _lastBatchTime) < _batchInterval)
      {
         if(await _channel.Reader.WaitToReadAsync())
            batch.Add(await _channel.Reader.ReadAsync());
      }

      _lastBatchTime = DateTime.Now;
      return batch;
    }

   private async Task ProcessFileEventsAsync(List<FileSystemEventArgs> e)
    {
         await Task.Delay(100);
         Console.WriteLine($"Batch Events Processed: {e.Count} events.");
        foreach(FileSystemEventArgs f in e)
             Console.WriteLine($"\tFile Event: {f.ChangeType}, {f.FullPath}");
    }

    public void Stop()
    {
        _watcher.EnableRaisingEvents = false;
        _channel.Writer.Complete();
         _watcher.Dispose();

    }
}


public static class Program {
    public static async Task Main(string[] args)
    {
        var monitor = new FileSystemMonitor(@"C:\temp",5,1000); //Batch 5 files and/or wait 1s.
        Task monitoringTask = monitor.StartProcessingAsync();

        Console.WriteLine("Press any key to stop...");
        Console.ReadKey();

         monitor.Stop();
        await monitoringTask;
    }
}
```
This example incorporates a batching strategy using `ReadBatchAsync`. It reads events from the channel until a specific `_batchSize` or timeout defined by `_batchInterval` is reached. Then, the accumulated events are processed collectively by `ProcessFileEventsAsync`. This can be very beneficial when you are looking for aggregate changes.

For further exploration and a deeper understanding, I recommend reviewing the official Microsoft documentation on `System.IO.FileSystemWatcher` and `System.Threading.Channels`. Books such as "Concurrency in C# Cookbook" by Stephen Cleary or "Programming .NET 4" by Jesse Liberty also contain relevant information on asynchronous programming patterns and the Task Parallel Library. Additionally, numerous blog posts and online articles cover these topics in detail; searching for `async programming`, `TPL`, or `producer-consumer pattern` in .NET will yield considerable supplementary material. Each approach to asynchronous event handling with `FileSystemWatcher` must be carefully tailored to meet the specific requirements of the application to ensure responsiveness, reliability, and efficient resource utilization.
