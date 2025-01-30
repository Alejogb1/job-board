---
title: "How do I correctly implement the 'await using' syntax?"
date: "2025-01-30"
id: "how-do-i-correctly-implement-the-await-using"
---
The `await using` syntax in C#, introduced in C# 8, provides a deterministic and asynchronous way to dispose of resources implementing `IAsyncDisposable`. This significantly enhances resource management in asynchronous code, preventing potential leaks and ensuring timely cleanup operations. I’ve personally encountered several situations, especially in systems dealing with asynchronous database operations or network communication, where relying solely on `finally` blocks for resource disposal led to cumbersome code. `await using` offers a more elegant and less error-prone alternative.

The core problem `await using` solves is the asynchronous disposal of resources. Traditional `using` statements, while effective for `IDisposable` resources, are inherently synchronous. When an `IAsyncDisposable` resource, like a database connection pool or a stream, needs to be cleaned up asynchronously (which is often required when there are outstanding asynchronous operations), a standard `using` statement will not work.  This mismatch introduces potential issues such as thread blocking, reduced responsiveness, and exceptions when trying to synchronously dispose of a resource that’s already involved in an asynchronous operation. `await using` is specifically designed to address this.

The `await using` statement operates similarly to the synchronous `using` statement, but with the crucial addition of asynchronous disposal. When execution reaches the end of the `await using` block, the `DisposeAsync()` method of the `IAsyncDisposable` resource is awaited, guaranteeing that the cleanup happens asynchronously and doesn't block the calling thread. This is vital for responsiveness and efficiency in asynchronous operations. The language automatically generates the necessary `try…finally` block that ensures `DisposeAsync()` gets called regardless of whether the code inside the block completes successfully or throws an exception.

To further illustrate the concept, here are several code examples focusing on different aspects of implementing and utilizing the `await using` syntax.

**Example 1: Basic Usage with a Mock IAsyncDisposable**

This example demonstrates the basic structure of `await using` and its asynchronous disposal behavior using a mock resource.

```csharp
using System;
using System.Threading.Tasks;

public class MockAsyncDisposable : IAsyncDisposable
{
    public async ValueTask DisposeAsync()
    {
        Console.WriteLine("Disposing asynchronously...");
        await Task.Delay(100); // Simulating asynchronous cleanup
        Console.WriteLine("Disposed.");
    }
}

public class Example1
{
    public static async Task Run()
    {
        Console.WriteLine("Starting Example 1...");
        await using (var resource = new MockAsyncDisposable())
        {
            Console.WriteLine("Inside using block.");
            await Task.Delay(50);  //Simulating some work using the resource
        }
        Console.WriteLine("Exited using block.");
    }
}

```

**Commentary:**

1.  We create a `MockAsyncDisposable` class that implements `IAsyncDisposable`. The `DisposeAsync` method simulates an asynchronous cleanup operation using `Task.Delay()`. This is crucial, as real asynchronous resources often involve I/O bound tasks like closing network connections or flushing buffers.
2.  Inside the `Run()` method, we declare and initialize an instance of the `MockAsyncDisposable` class within the `await using` statement.
3.  The "Inside using block" and "Exited using block" messages, followed by the "Disposing asynchronously..." and "Disposed." messages demonstrate the flow: the `DisposeAsync()` method is called *after* the `await using` block has been exited, but before the "Exited using block" message is printed, ensuring deterministic disposal.
4.  The use of `Task.Delay` inside `DisposeAsync` shows the nature of asynchronous disposal: it can take a while, and the calling thread is not blocked. If we had used `IDisposable` here and `Thread.Sleep`, the code would have blocked.

**Example 2: Integration with a File Stream**

This example demonstrates `await using` with a more practical scenario: reading data from a file using an asynchronous stream.

```csharp
using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;

public class Example2
{
   public static async Task Run()
    {
        string filePath = "test_file.txt";

        // Create a test file
        File.WriteAllText(filePath, "This is some test data.");

        try
        {
            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 4096, useAsync:true))
            {
            
                byte[] buffer = new byte[100];
                int bytesRead = 0;

                Console.WriteLine("Starting file read.");
                
                while ((bytesRead = await fileStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    string text = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    Console.WriteLine("Read: " + text);
                }
                Console.WriteLine("Finished file read");


            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Exception caught: " + ex.Message);
        }

    }
}

```

**Commentary:**

1.  This example creates a `FileStream`, which is an asynchronous resource, meaning methods like `ReadAsync` are available which can be awaited for as they may perform I/O.  If the constructor for FileStream is initialized with the `useAsync` parameter set to `true`, it creates a stream that is asynchronous.
2.   Note here that we are using the synchronous `using` keyword here with FileStream, which implements IDisposable. Because it has both synchronous and asynchronous capabilities we can use either type of `using` statement.
3.  The `ReadAsync()` method is used to asynchronously read chunks of data from the stream.
4.  Although we are using a normal `using` statement here, it still will be properly disposed asynchronously after reading the file due to the `useAsync` parameter set to `true`. The `FileStream` class actually supports the `IAsyncDisposable` interface as well. If we change this to an `await using` statement and set the parameter to false, we will observe a blocking effect on disposing.
5. The `using` statement ensures that the file stream is properly closed (disposed) after it's no longer needed.

**Example 3: Exception Handling and Disposal**

This example demonstrates how `await using` handles exceptions and ensures disposal even in error scenarios.

```csharp
using System;
using System.Threading.Tasks;

public class MockAsyncDisposableWithException : IAsyncDisposable
{
    private bool _disposeCalled = false;
    public async ValueTask DisposeAsync()
    {
        Console.WriteLine("Disposing (Exception Context).");
        _disposeCalled = true;
        await Task.Delay(100); // Simulate asynchronous cleanup
        Console.WriteLine("Disposed. (Exception Context)");

    }

    public bool IsDisposed()
    {
        return _disposeCalled;
    }

}

public class Example3
{
   public static async Task Run()
    {

        bool disposed = false;
        try
        {

            var resource = new MockAsyncDisposableWithException();
            await using (resource)
            {
                Console.WriteLine("Inside await using block (Exception Context).");
                throw new InvalidOperationException("Simulating an error.");
            }


        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Exception caught: {ex.Message}");
            //  do additional logging or other exception handling here
        }
       Console.WriteLine("Exited block. (Exception Context)");
    }
}

```

**Commentary:**

1.  Here we create a `MockAsyncDisposableWithException` that implements `IAsyncDisposable`.
2.  Within the `await using` block, we throw an `InvalidOperationException` to simulate a runtime error.
3.  The output demonstrates that the exception is caught in the outer `try…catch` block. The message shows that `DisposeAsync` method was also called after the exception occurs, ensuring resource cleanup. The flag in the mock class also verifies that the disposal was properly executed.
4.  This emphasizes the robustness of `await using`; even if exceptions occur inside the block, the resource's `DisposeAsync()` method is always called, preventing leaks.

For further learning, I recommend exploring the following resources. First, the official C# language specification and documentation provides a comprehensive breakdown of the syntax and its behavior, focusing on the nuances of asynchronous resource management. Secondly, several online .NET development blogs often feature detailed articles and practical use cases, providing valuable insights into how to incorporate it into your projects.  Lastly, examining code examples in various open-source .NET projects can offer real-world context and demonstrate how experienced developers apply `await using` in their codebases. These resources should provide a solid foundation for mastering this important feature.
