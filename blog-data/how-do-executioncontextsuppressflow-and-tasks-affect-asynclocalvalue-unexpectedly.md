---
title: "How do `ExecutionContext.SuppressFlow` and tasks affect `AsyncLocal.Value` unexpectedly?"
date: "2024-12-23"
id: "how-do-executioncontextsuppressflow-and-tasks-affect-asynclocalvalue-unexpectedly"
---

Alright,  I remember back when we were migrating a complex legacy application to .net core, we hit a really confounding issue involving asynchronous operations and shared state. It took some time, a lot of debugging, and a deep dive into the inner workings of the framework to finally understand why our `AsyncLocal<T>` values were behaving erratically. The issue stemmed, specifically, from the interaction between `ExecutionContext.SuppressFlow` and tasks. Let me break it down for you.

The core of the problem resides in how the .net framework manages the *execution context* across asynchronous operations. The execution context, in a nutshell, is like a snapshot of the current thread's environment. It includes things like security settings, impersonation tokens, and, crucially for our discussion, `AsyncLocal<T>` values. Normally, when an async operation is initiated, the framework captures the current execution context and flows it to the newly created task, ensuring that things like `AsyncLocal.Value` are preserved. This, for the most part, is exactly what you want.

However, there are situations where you might *not* want this flow to occur, and this is where `ExecutionContext.SuppressFlow` comes in. This static method allows you to temporarily prevent the capturing and flowing of the current execution context. The intended use cases are primarily performance optimization or situations where certain context data is not relevant to the child operation. The problem, however, comes in when you use this and try to rely on `AsyncLocal<T>` values within the context that’s suppressing flow.

The key thing to understand is: when `ExecutionContext.SuppressFlow` is active, any `Task.Run`, `Task.StartNew`, or similar operations started inside its scope will *not* inherit the current `AsyncLocal<T>` values. The new task will essentially start with a blank slate. It's not that `AsyncLocal` is being mutated; the value isn’t being explicitly changed; it's simply not being propagated.

Let's illustrate this with some code examples.

**Example 1: The Naive Approach**

This first snippet shows how you might *expect* `AsyncLocal` to work, assuming a naive understanding of asynchronous context flow:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExample
{
    public static AsyncLocal<string> CurrentUser = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
        CurrentUser.Value = "ParentUser";
        Console.WriteLine($"Main:  User is {CurrentUser.Value}");

        await Task.Run(async () => {
            Console.WriteLine($"Task.Run: User is {CurrentUser.Value}");
            await Task.Delay(10);
            Console.WriteLine($"Task.Run after delay: User is {CurrentUser.Value}");
        });

         Console.WriteLine($"Main after Task.Run:  User is {CurrentUser.Value}");
    }
}
```

Running this, you’ll (almost) always see the following:

```
Main:  User is ParentUser
Task.Run: User is ParentUser
Task.Run after delay: User is ParentUser
Main after Task.Run:  User is ParentUser
```

As anticipated, the `AsyncLocal` value flows correctly from the main thread to the task, and it remains consistent through the asynchronous operation.

**Example 2: Suppressed Flow and its Pitfalls**

Now, let's see what happens when we introduce `ExecutionContext.SuppressFlow`:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExample
{
    public static AsyncLocal<string> CurrentUser = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
        CurrentUser.Value = "ParentUser";
        Console.WriteLine($"Main:  User is {CurrentUser.Value}");

       ExecutionContext.SuppressFlow();
       try {
        await Task.Run(async () => {
            Console.WriteLine($"Task.Run: User is {CurrentUser.Value}");
            await Task.Delay(10);
            Console.WriteLine($"Task.Run after delay: User is {CurrentUser.Value}");
        });
       } finally {
        ExecutionContext.RestoreFlow();
       }

       Console.WriteLine($"Main after Task.Run:  User is {CurrentUser.Value}");
    }
}
```

The output is different:

```
Main:  User is ParentUser
Task.Run: User is
Task.Run after delay: User is
Main after Task.Run:  User is ParentUser
```

Notice that within the `Task.Run`, the `CurrentUser.Value` is now `null`, or more precisely, the default value for a `string`. This is because we've suppressed the context flow, and the task does not inherit the parent’s `AsyncLocal` value. It's crucial to understand that the `AsyncLocal` itself is not being modified, its value in the parent thread remains unchanged; it’s just not being passed down to child tasks when suppression is in effect.

**Example 3: A Controlled Solution**

The common way to handle this when `SuppressFlow` is necessary is to manually pass on the context data. This can be achieved by capturing the value *before* the context is suppressed and explicitly passing it to the task, like so:

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncLocalExample
{
    public static AsyncLocal<string> CurrentUser = new AsyncLocal<string>();

    public static async Task Main(string[] args)
    {
        CurrentUser.Value = "ParentUser";
        Console.WriteLine($"Main:  User is {CurrentUser.Value}");

        string capturedUser = CurrentUser.Value;

        ExecutionContext.SuppressFlow();
       try {
        await Task.Run(async () => {
           CurrentUser.Value = capturedUser;
           Console.WriteLine($"Task.Run: User is {CurrentUser.Value}");
            await Task.Delay(10);
            Console.WriteLine($"Task.Run after delay: User is {CurrentUser.Value}");
        });
       } finally {
        ExecutionContext.RestoreFlow();
       }

       Console.WriteLine($"Main after Task.Run:  User is {CurrentUser.Value}");

    }
}
```

The output of this code snippet will be:

```
Main:  User is ParentUser
Task.Run: User is ParentUser
Task.Run after delay: User is ParentUser
Main after Task.Run:  User is ParentUser
```

Now we have successfully flowed the value, even when `SuppressFlow` is used, because we captured the needed value and manually passed it to the worker task. This approach requires careful management of context data, but in performance-critical code paths, the manual approach can be more efficient than relying on the framework’s automatic context flow.

**When to Reach for These Tools:**

As a general rule, I’d advise against routinely using `ExecutionContext.SuppressFlow`. It should only be considered in specific performance-critical situations or when you absolutely need to control the exact context that a task runs under. The automatic context flow is the safest, and least surprising, way to manage shared state in most applications. The "danger" with `SuppressFlow` is that it's easy to introduce subtle bugs if you are not completely aware of what you are doing, as I can attest to from experience. For most use cases, manual context propagation is a better, albeit more explicit, approach.

For a deeper theoretical understanding, I recommend taking a look at “Concurrent Programming on Windows” by Joe Duffy, as well as the detailed documentation on the `ExecutionContext` class provided by Microsoft. Understanding the underlying mechanics of context management is vital to avoid these kinds of subtle issues when writing asynchronous code. I hope this gives you a clear explanation of this complex issue. It’s something that can trip up even seasoned developers, as I found out, so it’s definitely a concept worth digging into.
