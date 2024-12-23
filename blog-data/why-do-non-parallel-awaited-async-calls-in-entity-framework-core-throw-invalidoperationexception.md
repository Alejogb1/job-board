---
title: "Why do non-parallel awaited Async calls in Entity Framework Core throw InvalidOperationException?"
date: "2024-12-23"
id: "why-do-non-parallel-awaited-async-calls-in-entity-framework-core-throw-invalidoperationexception"
---

, let’s dive into this. It's a classic pitfall I’ve seen trip up many developers, including myself back in my early days with Entity Framework Core (EF Core). The issue with non-parallel, awaited async calls throwing `InvalidOperationException` when dealing with database operations is not a bug, per se, but a consequence of how EF Core manages its context and its expectations around execution flow. Let me explain.

The core problem stems from the fact that `DbContext`, the heart of EF Core, is not inherently thread-safe. It’s designed to operate within a single, linear execution path. When you introduce asynchronous operations (`async`/`await`) without understanding their implications, you risk violating this expectation. The `await` keyword doesn't magically make your code execute in parallel; instead, it effectively pauses the current method's execution, allows other tasks to proceed, and resumes where it left off once the awaited operation completes. This is crucial to understand.

Imagine a scenario where you have a method that's supposed to, say, retrieve a user, then update it and then retrieve an order associated with the user. If these operations are implemented as async calls without proper care you can introduce unexpected state changes to your `DbContext` instance. Let's say you start fetching user ‘A’ asynchronously, the `DbContext` records this intention internally. However, due to awaiting and the thread pool behaviour, it might return to your method later to complete the operation and at that time another awaiting call could be processing an entirely different data or even the same entity in different state. Therefore, subsequent operations operating on the same instance of `DbContext` could collide with the existing operation in flight and cause a conflict leading to that `InvalidOperationException`. The problem is not the asynchrony, but the potential for interleaved, non-serial access to the context's internal state and tracking.

A typical manifestation of this error is when you attempt to perform another database operation, via `await` call on the same context, before the first has fully completed. EF Core expects operations on the same context to be serialized. Even though you have `await` in between operations, it’s possible for the runtime to bring multiple tasks using the same context, interleaved in a way that the `DbContext` internal operation flow is confused.

This contrasts with scenarios where you *do* have parallel executions via threading. In those scenarios, you are likely to have separate instances of your `DbContext` and therefore the mentioned race condition related to interleaved operations with the same instance is less likely to occur.

To make this more concrete, consider the following example. Let's assume a simplified case where you have a user entity and you're trying to retrieve it, update it and retrieve an order for that user.

```csharp
public async Task ProcessUserAsync(int userId, ApplicationDbContext context)
{
    // Problematic example – non-parallel awaiting
    var user = await context.Users.FindAsync(userId);

    user.LastLogin = DateTime.UtcNow;
    await context.SaveChangesAsync(); // First SaveChangesAsync

    var order = await context.Orders
                             .FirstOrDefaultAsync(o => o.UserId == userId); // Subsequent await call.
        
    if (order != null) {
        Console.WriteLine($"Order ID: {order.Id}");
        }

}
```

In this code, you might *sometimes* not encounter an exception, depending on the timing of the underlying database I/O. However, the potential for the `InvalidOperationException` is certainly there since there is an `await` call after an `await` call using the same `DbContext` and its associated internal state. Although you don’t have multiple calls that are happening at exactly the same time because of the `await`, the problem exists because the context can receive updates out of order if there are multiple tasks interleaving in a way the execution flow is disrupted. The `DbContext` is not designed for this and expects each operation to complete before another starts operating on its internal state.

Now, let’s demonstrate how to *correctly* approach this by making these calls sequentially.

```csharp
public async Task ProcessUserCorrectedAsync(int userId, ApplicationDbContext context)
{
    var user = await context.Users.FindAsync(userId);

    user.LastLogin = DateTime.UtcNow;
    await context.SaveChangesAsync(); // First SaveChangesAsync

    var order = await context.Orders
                                    .FirstOrDefaultAsync(o => o.UserId == userId);
    if (order != null) {
         Console.WriteLine($"Order ID: {order.Id}");
       }
}
```

This snippet functions as intended because each `await` call is executed fully and the subsequent calls on the same `DbContext` can then rely on a consistent state within that same context. There is no interleaved behaviour and the `DbContext` operates as designed.

Finally, here's an illustration of how to introduce true parallelism by creating multiple contexts, and therefore, avoiding the issue entirely. Each task has its own `DbContext` instance to work on.

```csharp
public async Task ProcessUserParallelAsync(int userId, string connectionString)
{
    var task1 = Task.Run(async () =>
    {
        using (var context = new ApplicationDbContext(connectionString))
        {
            var user = await context.Users.FindAsync(userId);
           if(user != null)
            {
                user.LastLogin = DateTime.UtcNow;
                await context.SaveChangesAsync();
            }
            
        }

    });

    var task2 = Task.Run(async () =>
    {
        using (var context = new ApplicationDbContext(connectionString))
        {
            var order = await context.Orders
                .FirstOrDefaultAsync(o => o.UserId == userId);
                if(order != null)
                {
                    Console.WriteLine($"Order ID: {order.Id}");
                }
        }
    });

    await Task.WhenAll(task1, task2);
}

```

Here, we create two separate tasks, each with its own `DbContext` instance, ensuring isolation. This approach enables true parallel execution as each context has its own tracking and state management and the interleaved operations within the same context is avoided by not using a single instance across the different tasks.

To understand the nuances of `async`/`await` and thread pool management, I'd recommend reading Stephen Cleary's "Concurrency in C# Cookbook". Also, for a deeper dive into how `DbContext` works internally and its limitations, the official EF Core documentation is indispensable. Look specifically at the sections on change tracking and managing concurrency issues. For practical advice on managing concurrency in .NET, I found "CLR via C#" by Jeffrey Richter incredibly helpful.

In short, avoid non-parallel `await` calls on the same context instance and create separate contexts when you need true concurrency. If you get that `InvalidOperationException` related to the `DbContext`, that’s often the culprit. Learn the principles of async and how EF Core interacts with it, and it will save you a lot of debugging time in the long run.
