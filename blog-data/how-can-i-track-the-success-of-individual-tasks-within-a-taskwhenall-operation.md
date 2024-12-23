---
title: "How can I track the success of individual tasks within a Task.WhenAll operation?"
date: "2024-12-23"
id: "how-can-i-track-the-success-of-individual-tasks-within-a-taskwhenall-operation"
---

, let's get into this. I've seen this problem pop up more times than I can count, especially in high-throughput systems where monitoring the performance of individual operations is absolutely crucial. Dealing with `Task.WhenAll` is usually straightforward for getting results, but the devil's in the details when you need granular feedback on each task. A naive approach might leave you in the dark as to *which* specific task failed or succeeded and with what kind of metrics. Here’s how I've tackled this in practice, and it moves well beyond the simple fire-and-forget approach.

The essential challenge with `Task.WhenAll` is its inherent abstraction. It treats a collection of tasks as a single unit, returning a task that completes when all of the input tasks complete. This is great for convenience, but it hides individual task outcomes. Simply put, if one task throws an exception, you’ll get an `AggregateException`, but good luck easily pinpointing which task triggered the exception without more effort. Similarly, understanding latency or success rates for each individual task becomes nearly impossible with vanilla `Task.WhenAll`.

My approach revolves around wrapping each individual task, injecting observability and control mechanisms. We'll need to leverage a few key strategies to handle this effectively. The first one is to wrap each task in its own try-catch block to capture any exceptions that might be thrown. That might seem obvious, but the catch needs to include more than just logging. We want to keep track of not just whether it succeeded or failed, but *why* and potentially timing information too.

Let's look at our first code snippet, which implements this basic task wrapping.

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

public static class TaskExtensions
{
    public static async Task<TrackedTaskResult<T>> TrackedExecuteAsync<T>(this Task<T> task, string taskId)
    {
        var stopwatch = Stopwatch.StartNew();
        TrackedTaskResult<T> result = new();
        result.TaskId = taskId;

        try
        {
           result.Result = await task;
           result.IsSuccessful = true;
        }
        catch(Exception ex)
        {
            result.IsSuccessful = false;
            result.Exception = ex;
        }
        finally
        {
            result.ElapsedTime = stopwatch.Elapsed;
        }

        return result;
    }
}

public class TrackedTaskResult<T>
{
    public string TaskId { get; set; }
    public T Result { get; set; }
    public bool IsSuccessful { get; set; }
    public Exception Exception { get; set; }
    public TimeSpan ElapsedTime { get; set; }
}

public class Example1
{
    public static async Task Run()
    {
        var tasks = new List<Task<TrackedTaskResult<int>>>();

        for (int i = 0; i < 5; i++)
        {
            var task = SomeWorkAsync(i);
            tasks.Add(task.TrackedExecuteAsync($"Task_{i}"));
        }

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
           if (result.IsSuccessful)
           {
               Console.WriteLine($"{result.TaskId} completed successfully, result: {result.Result}, time taken: {result.ElapsedTime}");
           } else
           {
               Console.WriteLine($"{result.TaskId} failed, exception: {result.Exception.Message}, time taken: {result.ElapsedTime}");
           }
        }
    }

     public static async Task<int> SomeWorkAsync(int taskId)
    {
        await Task.Delay(100 * (taskId + 1)); // Simulate work
        if (taskId == 3)
            throw new InvalidOperationException("Simulated error.");

        return taskId * 2;
    }
}

// To run this snippet from Main or something similar:
// await Example1.Run();
```

Here, the `TrackedExecuteAsync` extension method wraps the core task. It uses a `Stopwatch` for timing, catches exceptions, and packs everything neatly into a `TrackedTaskResult`. This allows us to iterate through results after `Task.WhenAll` completes and analyze the outcome of each one.

Now, let's move to a more complex scenario that also includes capturing cancellation. In production scenarios, task cancellation is commonplace, and the failure mode of a cancelled task is distinct from an exception. The previous `try-catch` block would catch a `TaskCanceledException`, but it is useful to make it explicit. Our second code snippet demonstrates how we can achieve this:

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

public static class TaskExtensions2
{
    public static async Task<TrackedTaskResultWithCancellation<T>> TrackedExecuteAsync<T>(this Task<T> task, string taskId, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        TrackedTaskResultWithCancellation<T> result = new();
        result.TaskId = taskId;

        try
        {
           result.Result = await task.WithCancellation(cancellationToken);
           result.IsSuccessful = true;
        }
        catch(OperationCanceledException)
        {
           result.IsSuccessful = false;
           result.IsCancelled = true;
        }
         catch(Exception ex)
        {
            result.IsSuccessful = false;
            result.Exception = ex;
        }
        finally
        {
            result.ElapsedTime = stopwatch.Elapsed;
        }

        return result;
    }
}

public class TrackedTaskResultWithCancellation<T>
{
    public string TaskId { get; set; }
    public T Result { get; set; }
    public bool IsSuccessful { get; set; }
    public Exception Exception { get; set; }
    public bool IsCancelled {get; set;}
    public TimeSpan ElapsedTime { get; set; }
}

public static class TaskExtensions3 {
     public static async Task<T> WithCancellation<T>(this Task<T> task, CancellationToken cancellationToken)
    {
        var tcs = new TaskCompletionSource<T>();
        using(cancellationToken.Register(() => tcs.TrySetCanceled(cancellationToken)))
        {
            return await Task.WhenAny(task, tcs.Task).ContinueWith(t => {
                if (t.Result == task)
                {
                    return task.Result;
                } else
                {
                    throw new OperationCanceledException();
                }
            }, TaskContinuationOptions.OnlyOnRanToCompletion);

        }
    }
}


public class Example2
{
   public static async Task Run()
    {
        var tasks = new List<Task<TrackedTaskResultWithCancellation<int>>>();
        var cancellationTokenSource = new CancellationTokenSource();


        for (int i = 0; i < 5; i++)
        {
            var task = SomeWorkAsync(i);
            tasks.Add(task.TrackedExecuteAsync($"Task_{i}", cancellationTokenSource.Token));

            if (i == 2) //cancel after 2 task starts
                cancellationTokenSource.Cancel();
        }

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
           if (result.IsSuccessful && !result.IsCancelled)
           {
               Console.WriteLine($"{result.TaskId} completed successfully, result: {result.Result}, time taken: {result.ElapsedTime}");
           } else if (result.IsCancelled)
           {
              Console.WriteLine($"{result.TaskId} cancelled , time taken: {result.ElapsedTime}");
           } else
           {
               Console.WriteLine($"{result.TaskId} failed, exception: {result.Exception?.Message}, time taken: {result.ElapsedTime}");
           }
        }
    }

     public static async Task<int> SomeWorkAsync(int taskId)
    {
         await Task.Delay(100 * (taskId + 1)); // Simulate work
        if (taskId == 3)
            throw new InvalidOperationException("Simulated error.");

        return taskId * 2;
    }
}
// To run this snippet from Main or something similar:
// await Example2.Run();

```

Here, we include a `CancellationToken`, and we now wrap the task execution using a `WithCancellation` extension method that essentially allows a task to be canceled in a way that throws the correct `OperationCanceledException`. We also add the `IsCancelled` flag to make clear when a cancellation happens.

For my final example, let’s explore collecting metrics, specifically, timing metrics. We'll create a slightly more robust version of our wrapper class that allows for detailed performance analysis, particularly for latency in high-volume operations, and lets us have some configurable logic before and after the task executes, a pattern common in aspects.

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;


public interface IMetricsCollector
{
    void Increment(string name, double value);
}

public class NullMetricsCollector : IMetricsCollector
{
    public void Increment(string name, double value) { }
}

public class ConsoleMetricsCollector : IMetricsCollector
{
    public void Increment(string name, double value)
    {
        Console.WriteLine($"Metric {name}: {value}");
    }
}

public static class TaskExtensions4
{
     public static async Task<TrackedTaskResultWithMetrics<T>> TrackedExecuteAsync<T>(this Func<Task<T>> taskFactory, string taskId, IMetricsCollector metricsCollector, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        TrackedTaskResultWithMetrics<T> result = new();
        result.TaskId = taskId;

        try
        {
            result.IsBeforeExecuteSuccessful = BeforeExecute(taskId, metricsCollector);
            result.Result = await taskFactory().WithCancellation(cancellationToken);
            result.IsSuccessful = true;
        }
        catch(OperationCanceledException)
        {
            result.IsSuccessful = false;
            result.IsCancelled = true;
        }
         catch(Exception ex)
        {
            result.IsSuccessful = false;
            result.Exception = ex;
        }
        finally
        {
            result.ElapsedTime = stopwatch.Elapsed;
           result.IsAfterExecuteSuccessful = AfterExecute(taskId, metricsCollector);
        }

        return result;
    }

     private static bool BeforeExecute(string taskId, IMetricsCollector metricsCollector)
        {
            metricsCollector.Increment($"task_{taskId}_start", 1);
            return true;
        }

     private static bool AfterExecute(string taskId, IMetricsCollector metricsCollector)
        {
             metricsCollector.Increment($"task_{taskId}_end", 1);
             return true;
        }
}



public class TrackedTaskResultWithMetrics<T>
{
    public string TaskId { get; set; }
    public T Result { get; set; }
    public bool IsSuccessful { get; set; }
    public Exception Exception { get; set; }
     public bool IsCancelled {get; set;}
    public TimeSpan ElapsedTime { get; set; }
    public bool IsBeforeExecuteSuccessful {get; set;}
    public bool IsAfterExecuteSuccessful {get; set;}
}

public class Example3
{
   public static async Task Run()
    {
        var tasks = new List<Task<TrackedTaskResultWithMetrics<int>>>();
        var cancellationTokenSource = new CancellationTokenSource();
        var metricsCollector = new ConsoleMetricsCollector(); //You could plug an actual metrics system here

        for (int i = 0; i < 5; i++)
        {
             Func<Task<int>> taskFactory = () => SomeWorkAsync(i); // Wrap the Task creation in a Func
             tasks.Add(taskFactory.TrackedExecuteAsync($"Task_{i}", metricsCollector,cancellationTokenSource.Token));


             if (i == 2) //cancel after 2 task starts
                cancellationTokenSource.Cancel();
        }

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
           if (result.IsSuccessful && !result.IsCancelled)
           {
               Console.WriteLine($"{result.TaskId} completed successfully, result: {result.Result}, time taken: {result.ElapsedTime}");
           } else if (result.IsCancelled)
           {
              Console.WriteLine($"{result.TaskId} cancelled , time taken: {result.ElapsedTime}");
           } else
           {
               Console.WriteLine($"{result.TaskId} failed, exception: {result.Exception?.Message}, time taken: {result.ElapsedTime}");
           }
        }
    }

     public static async Task<int> SomeWorkAsync(int taskId)
    {
         await Task.Delay(100 * (taskId + 1)); // Simulate work
        if (taskId == 3)
            throw new InvalidOperationException("Simulated error.");

        return taskId * 2;
    }
}
// To run this snippet from Main or something similar:
// await Example3.Run();
```

Here, we've introduced an `IMetricsCollector` interface which is easily configurable and injected into the wrapper method. This allows collecting arbitrary metrics before and after task execution, demonstrating a flexible setup for comprehensive task tracking. The task creation is now encapsulated in a `Func<Task<T>>` to allow for capturing work done before task execution, another common use-case for logging.

For further investigation on advanced task handling, I’d strongly recommend exploring "Concurrency in C# Cookbook" by Stephen Cleary. It is an excellent resource for understanding task-based asynchrony and all of the nuances related to managing concurrent operations. Additionally, “Programming in the Large with C#” by Eric Lippert provides a deep dive into the design considerations of software at scale, which has some useful sections about proper usage of async and await. And, to truly grasp the underlying mechanism, the TPL documentation on Microsoft's website provides a comprehensive resource, with a lot of insight on things like the scheduler and the task lifecycle.

These approaches, while adding a bit of initial complexity, make it much easier to debug, monitor, and fine-tune the performance of individual task operations within `Task.WhenAll`. I’ve found that putting in this bit of effort up-front significantly reduces time spent troubleshooting and improves overall system stability.
