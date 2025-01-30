---
title: "How can I use WriteAsync with a timeout?"
date: "2025-01-30"
id: "how-can-i-use-writeasync-with-a-timeout"
---
The primary challenge in using `WriteAsync` with a timeout lies in the inherent nature of asynchronous operations; while `WriteAsync` itself is non-blocking, its completion isn't guaranteed within a specific timeframe without additional mechanisms. I've encountered scenarios in high-throughput network applications where a slow or unresponsive connection could indefinitely stall a writing operation, leading to resource exhaustion and degraded performance. Therefore, employing timeouts with asynchronous writes is crucial for robust and resilient applications.

The core concept involves combining the asynchronous `WriteAsync` operation with a cancellation token that is linked to a timeout. The `CancellationToken` class and its associated `CancellationTokenSource` enable signaling cancellation of asynchronous tasks. When a timeout expires, the cancellation token is signaled, which then propagates to the `WriteAsync` operation, causing it to terminate (typically by throwing an `OperationCanceledException`).

The basic pattern revolves around these steps:

1. **Create a `CancellationTokenSource` with a timeout.** This source manages the lifetime of the cancellation token and automatically signals it after the specified timeout duration.
2. **Obtain a `CancellationToken` from the `CancellationTokenSource`.** This token will be passed to the `WriteAsync` operation.
3. **Invoke `WriteAsync` with the cancellation token.** The `WriteAsync` method is aware of the cancellation token and will gracefully terminate if it becomes signaled.
4. **Handle the potential `OperationCanceledException`.** This exception indicates that the write operation timed out and allows you to take appropriate corrective actions.
5. **Dispose of the `CancellationTokenSource` when it is no longer needed.** Proper resource management is important to prevent memory leaks.

It's important to understand that the cancellation does not instantaneously halt the underlying write operation at the transport layer; rather, it signals that the operation should be considered failed from the perspective of the application. The actual write might still be in progress depending on the underlying stream implementation, but the application's immediate responsibility is to treat it as a failure.

Here are three code examples illustrating different aspects of using `WriteAsync` with a timeout:

**Example 1: Basic Timeout with Stream Writing**

```csharp
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public static class StreamWritingExample
{
    public static async Task WriteWithTimeout(Stream stream, string data, int timeoutMilliseconds)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(data);
        var cts = new CancellationTokenSource(timeoutMilliseconds);
        try
        {
            await stream.WriteAsync(bytes, 0, bytes.Length, cts.Token);
            Console.WriteLine("Write operation completed successfully.");
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("Write operation timed out.");
            // Implement retry logic, error logging, etc. here.
        }
        finally
        {
            cts.Dispose();
        }
    }

    public static async Task Main()
    {
        // Simulate a slow stream by using a MemoryStream.
         using (MemoryStream ms = new MemoryStream())
        {
            string largeString = new string('A', 1024 * 1024);
            await WriteWithTimeout(ms, largeString, 100); //Short timeout to demonstrate the cancellation.
            
            string smallString = "This should complete quickly";
            await WriteWithTimeout(ms, smallString, 5000);
         }
         Console.ReadLine();
    }
}
```

* **Commentary:** This example demonstrates the fundamental pattern. The `WriteWithTimeout` method encapsulates the logic of creating a `CancellationTokenSource`, passing the `CancellationToken` to `WriteAsync`, and handling the potential `OperationCanceledException`. The example within `Main` showcases both scenarios where the timeout is triggered with a large string and when it completes successfully with a small string. In a real-world scenario, the stream would likely be a network stream, but this memory stream setup allows easy testing of the timeout behavior. This provides a concrete starting point for incorporating timeouts into more complex asynchronous I/O operations. The use of `finally` ensures proper disposal of the `CancellationTokenSource`, critical for avoiding memory leaks. The simulated slow stream helps illustrate the behavior of the timeout during testing.

**Example 2: Timeout with Cancellation and Resource Cleanup**

```csharp
using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public static class NetworkWritingExample
{
     public static async Task WriteWithTimeout(NetworkStream stream, string data, int timeoutMilliseconds)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(data);
        var cts = new CancellationTokenSource(timeoutMilliseconds);
        try
        {
           await stream.WriteAsync(bytes, 0, bytes.Length, cts.Token);
           Console.WriteLine("Network write completed successfully");
        }
        catch (OperationCanceledException)
        {
           Console.WriteLine("Network write operation timed out");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An unexpected error occurred during the write: {ex.Message}");
        }
        finally
        {
            cts.Dispose();
        }
    }

   public static async Task Main()
    {
         try
        {
             using (TcpClient client = new TcpClient())
             {
                 await client.ConnectAsync("google.com", 80);
                 using (NetworkStream stream = client.GetStream())
                 {
                       string httpRequest = "GET / HTTP/1.1\r\nHost: google.com\r\nConnection: close\r\n\r\n";
                      await WriteWithTimeout(stream, httpRequest, 1000);

                       if (client.Connected)
                         {
                          Console.WriteLine("Connection is still open.");
                           } else {
                           Console.WriteLine("Connection is closed");
                           }
                 }
            }

        }
        catch (Exception ex)
        {
           Console.WriteLine($"An error occurred: {ex.Message}");
        }
          Console.ReadLine();
   }
}
```

* **Commentary:** This example demonstrates how timeouts are applicable to a common scenario with a `NetworkStream` (albeit against google.com just as a test server.) The core logic of handling the timeout is identical to the first example, but it highlights the context of network communications. It includes connection establishment and the actual write operation to the network stream along with more comprehensive exception handling (for other potential errors besides timeouts). The use of `using` blocks ensure proper resource cleanup for the TCP client and network stream. It emphasizes that a write timeout does not imply immediate closure of the underlying socket connection, which must be managed separately. It explicitly prints the status of the connection following the write. This illustrates that the `OperationCanceledException` indicates a failure of the asynchronous *operation*, not necessarily the transport, which might have completed asynchronously after cancellation.

**Example 3: Timeout with Multiple Writes and Error Aggregation**

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Text;

public static class MultipleWritesExample
{
    public static async Task WriteMultipleWithTimeout(Stream stream, List<string> dataList, int timeoutMilliseconds)
    {
        var exceptions = new List<Exception>();
         foreach(var data in dataList)
        {
            try
            {
               await WriteWithTimeout(stream, data, timeoutMilliseconds);
            }
            catch(Exception ex)
            {
                exceptions.Add(ex);
             }
        }
        if (exceptions.Count > 0)
        {
          throw new AggregateException("One or more write operations failed.", exceptions);
        }
    }

    private static async Task WriteWithTimeout(Stream stream, string data, int timeoutMilliseconds)
    {
      byte[] bytes = Encoding.UTF8.GetBytes(data);
      var cts = new CancellationTokenSource(timeoutMilliseconds);
     try
     {
         await stream.WriteAsync(bytes, 0, bytes.Length, cts.Token);
          Console.WriteLine("Write completed");

     }
     catch (OperationCanceledException)
     {
       Console.WriteLine("Write timed out.");
       throw;
     }
    finally
     {
       cts.Dispose();
     }
    }

    public static async Task Main()
    {
        using (MemoryStream ms = new MemoryStream())
        {
            var data = new List<string>() {"First Data", new string('B', 1024*1024), "Third Data"}; // second data will timeout due to large size and small timeout.
            try
            {
               await WriteMultipleWithTimeout(ms, data, 100);
            }
             catch (AggregateException ae)
             {
                Console.WriteLine($"Aggregated errors: {ae.Message}");
                foreach (var ex in ae.InnerExceptions)
                    Console.WriteLine($"- {ex.Message}");
            }

        }
        Console.ReadLine();
    }

}
```

* **Commentary:** This example demonstrates how to handle multiple asynchronous write operations with individual timeouts. It introduces `WriteMultipleWithTimeout` which iterates over a list of data and attempts to write each element with its own timeout. The key part is the use of an `AggregateException` to collect and rethrow errors, which is essential when multiple operations might fail. This pattern is useful when dealing with batch operations where the failure of a single write should not prevent processing of the remaining data but should be reported collectively. This also further illustrates how the single `WriteWithTimeout` method can be re-used within different contexts. This offers a more scalable approach to integrating timeouts when processing multiple asynchronous tasks.

For further exploration of this topic, I would suggest reviewing resources focused on:
* Asynchronous programming patterns in C#, specifically using `async`/`await`.
* The documentation for `CancellationToken` and `CancellationTokenSource`.
* Network programming and `System.Net.Sockets` namespace.
* Error handling patterns when working with asynchronous operations, particularly `AggregateException`.
* Best practices for managing resources such as `Stream` objects and disposable classes to avoid memory leaks and resource exhaustion.
 By studying these resources, one can gain a more in depth understanding and application of asynchronous operations with timeouts.
