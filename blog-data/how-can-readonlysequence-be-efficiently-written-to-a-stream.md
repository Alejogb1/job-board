---
title: "How can ReadOnlySequence be efficiently written to a stream?"
date: "2024-12-23"
id: "how-can-readonlysequence-be-efficiently-written-to-a-stream"
---

,  It's a question I've had to address more times than I care to remember, especially when dealing with network protocols and high-performance I/O. The core challenge with `ReadOnlySequence` arises from its segmented nature. Unlike a contiguous `byte[]` or `Memory<byte>`, it might be fragmented across multiple underlying buffers. This means directly writing it to a stream isn't as straightforward as a simple `stream.Write(myByteArray)` call. Efficiency hinges on avoiding unnecessary allocations and copy operations, which can become significant bottlenecks in performance-critical scenarios.

From my experience, particularly during my stint building a custom TCP proxy a few years back, understanding how to efficiently marshal a `ReadOnlySequence` to a stream directly impacted our throughput. We were handling a large volume of data, so any overhead quickly manifested as a significant performance hit. Here’s how I've found it best to approach this problem, along with some illustrative code.

Essentially, the most performant method boils down to iterating through the segments of the `ReadOnlySequence` and writing each segment directly to the stream. The `ReadOnlySequence` provides mechanisms for this; specifically, the `SequenceReader` or the explicit enumeration using `ReadOnlySequence.GetEnumerator()`. While `SequenceReader` is particularly useful for parsing, simple writing can be done efficiently with the enumeration approach. Avoid copying to intermediate buffers whenever possible.

Let’s delve into an example. Suppose we have a `ReadOnlySequence<byte>` and an open `Stream`:

```csharp
using System;
using System.Buffers;
using System.IO;

public static class ReadOnlySequenceExtensions
{
    public static void WriteToStream(this ReadOnlySequence<byte> sequence, Stream stream)
    {
        foreach (var segment in sequence)
        {
            stream.Write(segment.Span);
        }
    }
}
```

Here, we're directly writing the `Span` of each `ReadOnlyMemory<byte>` segment to the stream. Notice there are no intermediate buffer allocations; each segment's underlying memory is directly consumed by the stream's write operation. This is about as efficient as it gets, barring specialized stream implementations that might offer further internal optimizations.

Now, let’s consider a situation where you also need to track the bytes being written. Perhaps you need to perform some form of logging or checksum calculation. We can modify the approach slightly to incorporate that functionality without introducing intermediate buffers, using a `SequenceReader`:

```csharp
using System;
using System.Buffers;
using System.IO;

public static class ReadOnlySequenceExtensions
{
   public static long WriteToStreamWithTracking(this ReadOnlySequence<byte> sequence, Stream stream, Action<int> bytesWrittenCallback)
   {
      long totalBytesWritten = 0;
      SequenceReader<byte> reader = new(sequence);
      while (reader.TryRead(out ReadOnlySpan<byte> span))
      {
         stream.Write(span);
         bytesWrittenCallback(span.Length);
         totalBytesWritten += span.Length;
      }
      return totalBytesWritten;
   }
}

```

In this implementation, the `SequenceReader` gives us access to `ReadOnlySpan<byte>` slices, which we write to the stream. The added `bytesWrittenCallback` allows for tracking the amount written. This keeps the allocation low, as the `SequenceReader` is a struct and works directly with the sequence's internal pointers.

Finally, imagine a more complex scenario where the `Stream` might have blocking operations or you need to implement cancellation. A simple `Write` method might not be ideal; you may need to move to an asynchronous write operation. Here’s how you could adapt it for async operations:

```csharp
using System;
using System.Buffers;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

public static class ReadOnlySequenceExtensions
{
  public static async Task WriteToStreamAsync(this ReadOnlySequence<byte> sequence, Stream stream, CancellationToken cancellationToken = default)
   {
       foreach (var segment in sequence)
       {
            await stream.WriteAsync(segment.Span, cancellationToken).ConfigureAwait(false);
       }
    }
}
```
The transition to `WriteAsync` is fairly direct. We are using the asynchronous counterpart of the stream write method. It uses `ConfigureAwait(false)` to avoid capturing the context, further improving performance in asynchronous setups. The cancellation token allows gracefully ending the operation.

These three examples show various approaches, starting with the most basic and moving to more complex, real-world applications. In my experience, understanding the underlying mechanisms of `ReadOnlySequence`, and specifically the `Span` and `SequenceReader` APIs, is critical to writing efficient, performant code. Premature optimization can be harmful, but in performance-critical areas, avoiding allocations and memory copies can make a significant difference in the overall application efficiency.

For deeper study, I highly recommend diving into the Microsoft documentation on `System.Buffers` and `System.IO`, particularly regarding `ReadOnlySequence`, `Memory<T>`, and `Span<T>`. Books such as “C# 8.0 in a Nutshell” by Joseph Albahari, specifically the chapters covering these areas, and "Concurrent Programming on Windows" by Joe Duffy offer excellent detailed coverage, with insightful performance considerations. Moreover, researching the background and design rationale behind `Span<T>` and `Memory<T>` on various .NET GitHub repositories will greatly improve your understanding. These resources will give you a solid foundation to further investigate and tackle complex performance scenarios in the .NET ecosystem. Don't rely solely on blog posts; the official documentation, reputable books, and exploring the source code itself will provide the most valuable insight.
