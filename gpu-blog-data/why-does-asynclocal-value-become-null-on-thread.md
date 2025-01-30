---
title: "Why does AsyncLocal value become null on thread context change?"
date: "2025-01-30"
id: "why-does-asynclocal-value-become-null-on-thread"
---
The loss of `AsyncLocal<T>` values across thread context switches is a direct consequence of how `AsyncLocal<T>` manages its data and how asynchronous operations in .NET are typically implemented. I've frequently encountered this behavior while debugging complex parallel processing pipelines, particularly those using `Task.Run` or custom thread pool interactions within an asynchronous workflow. This isn't a bug but an inherent characteristic of the design. `AsyncLocal<T>` is fundamentally tied to the *logical execution context*, not the physical thread of execution.

The logical execution context, introduced with asynchronous programming, represents the conceptual flow of an asynchronous operation.  While a single asynchronous operation might be executed across multiple threads, the logical context ensures that contextual information like activity IDs, cancellation tokens, and, crucially, `AsyncLocal<T>` values, are propagated correctly within the intended operation flow. However, this propagation mechanism isn't automatic across all thread changes; specific APIs and techniques are needed to preserve context between logical flows that happen on different threads.

A core tenet of `AsyncLocal<T>` operation is its use of `ExecutionContext` and `SynchronizationContext`. When an asynchronous operation begins, the current `ExecutionContext` is captured. This context contains a snapshot of all `AsyncLocal<T>` values that were set at that point. When an awaited asynchronous operation is resumed (e.g., after an `await` keyword), the captured `ExecutionContext` is restored, effectively carrying over the `AsyncLocal<T>` values to the new continuation. The .NET framework manages these `ExecutionContext` captures and restores using mechanisms that are intrinsic to the design of async/await. If you deviate from this model, you can lose the captured context.

The issue arises prominently when you transition to threads managed outside of the core asynchronous flow framework. This often occurs when using `Task.Run`, which schedules a task on a thread pool thread. When `Task.Run` initiates a new task, it doesn't necessarily inherit the captured `ExecutionContext`. Instead, it often creates a *new* execution context on the new thread unless explicitly specified otherwise. Consequently, any `AsyncLocal<T>` values set in the original logical context are not available within the code running inside the `Task.Run`-generated task. This explains why you frequently observe `AsyncLocal<T>` values becoming `null` on a thread context change introduced using `Task.Run` or similar mechanisms.

Consider an example:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExample
{
    public static AsyncLocal<string> MyValue = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
        MyValue.Value = "Initial Value";
        Console.WriteLine($"Before Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

        await Task.Run(async () =>
        {
            Console.WriteLine($"Inside Task.Run (before await): {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

            await Task.Delay(10); // Simulate some async work

            Console.WriteLine($"Inside Task.Run (after await): {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
        });

        Console.WriteLine($"After Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
    }
}
```

In this example, the value "Initial Value" is set before the `Task.Run` call. Inside the anonymous method passed to `Task.Run`, you'll find that `MyValue.Value` is null (or an empty string if default is set), demonstrating that the original context isn’t directly passed to the `Task.Run` delegate, and it does not automatically flow. Moreover, the awaits will flow properly within the thread started by `Task.Run`.

Here’s a second example where we try to workaround it using `ExecutionContext.Run` and `ExecutionContext.Capture`:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExampleWithContext
{
    public static AsyncLocal<string> MyValue = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
         MyValue.Value = "Initial Value";
        Console.WriteLine($"Before Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

        var context = ExecutionContext.Capture();

        await Task.Run(async () =>
        {
              ExecutionContext.Run(context, _ => {

            Console.WriteLine($"Inside Task.Run (before await): {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

              }, null);

                await Task.Delay(10); // Simulate some async work

                 ExecutionContext.Run(context, _ =>
                {
                    Console.WriteLine($"Inside Task.Run (after await): {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
                }, null);

        });

       Console.WriteLine($"After Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
    }
}

```
Here, `ExecutionContext.Capture()` grabs the current execution context that contains `MyValue`. Inside the `Task.Run`’s anonymous method, we use `ExecutionContext.Run` to explicitly set the context captured before, enabling the `AsyncLocal` value to be preserved. While this works, it adds additional boiler plate to the code. It's also crucial to use `ExecutionContext.Run` before and after each await inside of this context.

To further illustrate a more modern approach, here is an example utilizing `AsyncLocal<T>.CreateCopyOnWrite`:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExampleCopyOnWrite
{
    public static AsyncLocal<string> MyValue = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
         MyValue.Value = "Initial Value";
        Console.WriteLine($"Before Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

        var copy = MyValue.CreateCopyOnWrite();

        await Task.Run(async () =>
        {
             Console.WriteLine($"Inside Task.Run (before await): {copy.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");

                await Task.Delay(10); // Simulate some async work


                Console.WriteLine($"Inside Task.Run (after await): {copy.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
        });

       Console.WriteLine($"After Task.Run: {MyValue.Value}, Thread ID: {Thread.CurrentThread.ManagedThreadId}");
    }
}
```

In this version, we leverage the `CreateCopyOnWrite()` method. This provides a more focused approach for passing the `AsyncLocal<T>` across logical boundaries. It creates a copy of the `AsyncLocal<T>` with the current value, and future modifications to the original won't affect this copied variable. This reduces boilerplate, and is likely the best route if you need to manually move an `AsyncLocal<T>` across thread pool boundaries, as it's more specific and readable than explicitly dealing with the whole execution context.

In summary, `AsyncLocal<T>` relies on the `ExecutionContext` to propagate values across asynchronous operations that are executed using the framework. Thread changes induced by techniques like `Task.Run` often create a new execution context, which doesn't automatically carry over these `AsyncLocal<T>` values. Understanding this mechanism, and utilizing techniques like `ExecutionContext.Run` or `AsyncLocal<T>.CreateCopyOnWrite` appropriately, is critical for effective asynchronous programming with context preservation.

For further information and to solidify understanding, it would be beneficial to study the .NET documentation on `ExecutionContext`, `SynchronizationContext`, and `AsyncLocal<T>`. Examining the source code of asynchronous operations within the BCL can also provide deep insights into how these concepts are implemented under the hood. Exploring advanced threading patterns like custom thread pools and how `ExecutionContext` and `SynchronizationContext` are handled within them can further enhance understanding.
