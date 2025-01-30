---
title: "Why is C# async/await progress reporting out of order?"
date: "2025-01-30"
id: "why-is-c-asyncawait-progress-reporting-out-of"
---
Understanding asynchronous operations in C# requires acknowledging that `async`/`await` doesn't inherently guarantee strictly sequential execution of progress updates, particularly when multiple asynchronous tasks contribute to the overall process. Progress reporting, typically implemented using an `IProgress<T>` instance, can appear out of order due to the non-deterministic nature of task scheduling and the context in which progress updates are invoked. I've encountered this exact issue while developing a large-scale file processing application, leading to confusing user interfaces that showed seemingly random progress steps.

The core issue resides in the cooperative multitasking nature of `async`/`await`. When an `await` keyword is encountered, the currently executing method yields control back to its caller, allowing other tasks that are ready to execute to use the current thread. This is not a threaded context; tasks don't necessarily execute on different threads unless configured to do so, and asynchronous operations are typically I/O bound. When progress updates are fired within these awaited tasks, the specific order in which those updates reach the `IProgress<T>` handler depends on when those tasks are resumed after awaiting their respective operations. The order they `await` or complete isn’t equivalent to the order the calling code started them. If multiple asynchronous operations are started concurrently using, say, `Task.WhenAll`, their progress callbacks can interleave irregularly if they are reporting to the same shared progress instance. This interleaving is further complicated by how the thread pool allocates execution resources; some tasks may complete faster based on operating system-level task priorities, I/O availability and hardware factors. Progress updates depend not on the order a task *starts* executing, but rather the order the relevant `await` points within the task *complete*.

Consider an example where three files are being processed concurrently. Each processing step generates a progress update. If the first file’s initial processing phase completes quickly, it may report 25% complete before the second file has even reached that stage, even if the second file was "started" prior to it within the calling function. It's not about the order of `StartAsync()` calls, but the order of the inner `await` points across many asynchronous tasks. The order in which `Progress.Report()` is called doesn’t guarantee the order in which your handler executes when various asynchronous operations are underway.

Let's illustrate this with code.

```csharp
public async Task ProcessFilesAsync(string[] filePaths, IProgress<int> progress)
{
    var tasks = filePaths.Select((filePath, index) => ProcessFileAsync(filePath, progress, index+1)).ToList();
    await Task.WhenAll(tasks);
}

private async Task ProcessFileAsync(string filePath, IProgress<int> progress, int fileNumber)
{
    // Simulate some work
    await Task.Delay(new Random().Next(50,200));
    progress.Report(fileNumber * 25);
    await Task.Delay(new Random().Next(50,200));
    progress.Report(fileNumber * 50);
    await Task.Delay(new Random().Next(50,200));
    progress.Report(fileNumber * 75);
    await Task.Delay(new Random().Next(50,200));
    progress.Report(fileNumber * 100);
}
```

In this snippet, `ProcessFilesAsync` iterates through a collection of file paths and starts the `ProcessFileAsync` operation for each file using `Task.WhenAll`. `ProcessFileAsync` simulates work using `Task.Delay` and reports progress at several points. The `progress.Report()` calls are not made sequentially across the files because their respective `Task.Delay()` operations complete at variable, unpredictable times. Thus, the progress reports reaching the handler linked to `progress` instance will most likely interleave as the various asynchronous tasks complete their simulated work.

To demonstrate a more practical situation, imagine we are downloading data chunks concurrently.

```csharp
public async Task DownloadDataAsync(string[] urls, IProgress<int> progress)
{
    var tasks = urls.Select(url => DownloadChunkAsync(url, progress)).ToList();
    await Task.WhenAll(tasks);
}

private async Task DownloadChunkAsync(string url, IProgress<int> progress)
{
    using var client = new HttpClient();
    byte[] data = await client.GetByteArrayAsync(url);
    var fileSize = data.Length;

    // Simulate reporting progress as parts of file download
    for(int i = 0; i < 4; i++)
    {
       await Task.Delay(new Random().Next(100,300));
       progress.Report((int)((float)(i+1)/4 * 100));
    }

}
```

Here, each URL is processed using `DownloadChunkAsync`, and multiple progress updates occur for each download, based on the simulated chunk download progress (again using `Task.Delay` to represent I/O). With multiple concurrent downloads, the updates are unlikely to be orderly if these chunks have different completion times. It's improbable that we’ll see 25%, then 50%, then 75%, then 100% sequentially for each download. Instead, we’re likely to see 25% updates from multiple downloads appear, then perhaps several 50% reports, then more 75%, and so on, with a random interleaving of files.

A more complex example involves hierarchical asynchronous operations:

```csharp
public async Task ProcessComplexWorkflowAsync(IProgress<int> progress)
{
    await PerformStepOneAsync(progress);
    await PerformStepTwoAsync(progress);
}

private async Task PerformStepOneAsync(IProgress<int> progress)
{
    // Initiate several tasks that report progress
    var tasks = Enumerable.Range(0,3).Select( async x =>
    {
      await Task.Delay(new Random().Next(50,200));
      progress.Report(x * 10);
    }).ToList();
    await Task.WhenAll(tasks);
}

private async Task PerformStepTwoAsync(IProgress<int> progress)
{
     // Initiate a single task that reports progress
    await Task.Delay(new Random().Next(300,500));
    progress.Report(50);
    await Task.Delay(new Random().Next(300,500));
    progress.Report(100);
}
```

Here, `ProcessComplexWorkflowAsync` executes two asynchronous steps. Each step may independently generate progress updates through a shared `IProgress<int>`. The issue is that `PerformStepOneAsync` initiates *several* tasks with their own progress reports, but it only reports them internally, and not to an overall external progress metric, before finally moving to `PerformStepTwoAsync`. Hence, we might see progress interleaved between tasks within `PerformStepOneAsync`, but it’s highly unlikely that the reporting of task within step 1, with ranges between 0-20, will neatly precede the updates of step 2 that report 50 and 100. These updates will appear as if they are occurring out of the scope of the overall process. This is not strictly a bug but it leads to confusing progress indicators because the caller expects linear progression.

To mitigate this behavior and ensure orderly progress reporting, several approaches can be taken. You can use a progress aggregator, which collects partial progress from each task and manages a single overall progress value. Or instead of reporting progress directly from within the asynchronous tasks, these could return partial results, with the calling code responsible for aggregating these results and translating them to consistent and sequential progress reports. Furthermore, the structure of asynchronous workflow could be reworked to reduce the level of concurrency when fine-grained progress is required.

For further understanding of asynchronous programming in C#, I recommend exploring the official Microsoft documentation, particularly the sections on `async/await` and `Task` based programming. Books focusing on asynchronous and parallel programming in .NET also prove valuable, providing theoretical background and various examples beyond the immediate context of this answer. Look for material that focuses on cooperative multitasking and task scheduling specifically. Finally, examining examples and source code of popular open-source .NET libraries often clarifies common patterns and solutions.
