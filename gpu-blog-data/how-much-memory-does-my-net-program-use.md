---
title: "How much memory does my .NET program use?"
date: "2025-01-26"
id: "how-much-memory-does-my-net-program-use"
---

The memory usage of a .NET program is not a single, static value; it fluctuates dynamically based on the application's state, the Common Language Runtime's (CLR) behavior, and interactions with the operating system. I've spent considerable time optimizing .NET applications, and understanding the intricacies of memory management is crucial for performance. Monitoring memory, especially for long-running services, requires attention to both the heap and non-heap usage.

A .NET program's memory footprint primarily involves the managed heap, the garbage collector (GC), and unmanaged resources. The managed heap stores the objects you create, like classes and data structures. The garbage collector periodically reclaims memory occupied by objects that are no longer referenced. Unmanaged resources, such as file handles, network sockets, or memory allocated by native libraries through platform invoke (P/Invoke), reside outside of the CLR’s direct control. Monitoring and optimizing memory usage involves understanding how these areas behave and interact. The total memory consumption visible in task manager encompasses not just the managed heap but also all other associated memory.

Several aspects contribute to the apparent complexity of memory consumption. The garbage collector's generations (Gen0, Gen1, and Gen2) influence the frequency and impact of garbage collection. Gen0 collects very frequently, and Gen2 collections, while less common, are more computationally intensive. Further, large object heap (LOH) allocations, for objects over approximately 85,000 bytes, are treated differently and often result in increased memory footprint.  Pinning objects, which prevents the garbage collector from moving them in memory, can lead to heap fragmentation.  Moreover, the allocation and deallocation patterns of the program directly impact the memory pressure.

Let's examine several approaches and tools I utilize to diagnose memory consumption problems.

**Code Example 1: Basic Memory Monitoring with `GC.GetTotalMemory`**

```csharp
using System;
using System.Threading;

public class MemoryMonitor
{
    public static void Main(string[] args)
    {
        Console.WriteLine("Memory Usage Example");
        long startMemory = GC.GetTotalMemory(false);
        Console.WriteLine($"Initial Memory: {startMemory / 1024} KB");

        // Allocate a large array.
        byte[] buffer = new byte[1024 * 1024 * 100]; // 100 MB
        long afterAllocationMemory = GC.GetTotalMemory(false);
        Console.WriteLine($"After Allocation Memory: {afterAllocationMemory / 1024} KB");

       // Force a garbage collection to see the impact
        GC.Collect();
        long afterCollectionMemory = GC.GetTotalMemory(false);
        Console.WriteLine($"After Garbage Collection Memory: {afterCollectionMemory / 1024} KB");

        Thread.Sleep(10000); // Keep process alive

    }
}

```
*Explanation:* This example uses the `GC.GetTotalMemory(false)` method to obtain the current size of the managed heap. The `false` parameter indicates that we do not want to force a collection; we just want to obtain the current value. By capturing the memory usage before and after a large allocation and then after forcing a collection, we can directly observe the heap's behavior under different scenarios.  The `Thread.Sleep` call keeps the process alive long enough to observe the memory in process monitoring tools as well.

**Code Example 2: Exploring Object Lifetimes and Weak References**

```csharp
using System;
using System.Collections.Generic;

public class WeakReferenceExample
{
    private class LargeObject
    {
        public byte[] Data {get;}

        public LargeObject(int sizeInMb) {
            Data = new byte[1024 * 1024 * sizeInMb];
        }

        ~LargeObject() {
           Console.WriteLine("Finalizer called");
        }
    }

    public static void Main(string[] args)
    {
       Console.WriteLine("Weak Reference Example");
        var largeObject = new LargeObject(100); //100 MB
        Console.WriteLine($"Object created. Memory: {GC.GetTotalMemory(false)/ 1024} KB");
        
        var weakRef = new WeakReference<LargeObject>(largeObject);

        largeObject = null; //remove the strong reference

        GC.Collect();
        Console.WriteLine($"After GC. Memory: {GC.GetTotalMemory(false)/ 1024} KB");

         //check if the object is still alive
        if(weakRef.TryGetTarget(out LargeObject? retrievedObject)) {
            Console.WriteLine($"Weak reference still available!");
        } else {
             Console.WriteLine($"Weak reference cleared!");
        }

        GC.Collect(); //Collect again.
        Console.WriteLine($"After 2nd GC. Memory: {GC.GetTotalMemory(false)/ 1024} KB");
       //check if the object is still alive again
        if(weakRef.TryGetTarget(out LargeObject? retrievedObject2)) {
            Console.WriteLine($"Weak reference still available!");
        } else {
             Console.WriteLine($"Weak reference cleared!");
        }
        

        //Console.ReadLine();
    }
}
```
*Explanation:*  This example introduces the concept of weak references, allowing us to reference objects without preventing garbage collection.  The first part of the program creates a large object and a weak reference to it. By setting the original reference to null and forcing a GC, we illustrate that the object does not get immediately collected as the weak reference is still holding onto it. Running a subsequent GC demonstrates that, unless the weak reference is used, the GC will eventually reclaim it. Understanding how the garbage collector interacts with strong and weak references helps manage resource lifetimes effectively.  Note, that finalizer of the object is called only once the memory is reclaimed.

**Code Example 3: Pinning and its Impact**

```csharp
using System;
using System.Runtime.InteropServices;
using System.Threading;

public class PinningExample
{
    public static void Main(string[] args)
    {
        Console.WriteLine("Pinning Example");
       byte[] buffer = new byte[1024 * 1024 * 10]; // 10MB

        GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);

        try {
           Console.WriteLine($"Memory before pinning : {GC.GetTotalMemory(false)/ 1024} KB");
            //Do some work with the pinned buffer
            Thread.Sleep(5000);

           Console.WriteLine($"Memory after pinning : {GC.GetTotalMemory(false)/ 1024} KB");

        } finally
        {
            handle.Free();
            Console.WriteLine($"Memory after unpinning : {GC.GetTotalMemory(false)/ 1024} KB");
             GC.Collect();
             Console.WriteLine($"Memory after 2nd GC : {GC.GetTotalMemory(false)/ 1024} KB");
        }

        Thread.Sleep(10000);
    }
}
```
*Explanation:* This example utilizes `GCHandle` to pin memory in place preventing the garbage collector from moving it.  Pinning is often required when interoperating with unmanaged code through P/Invoke, because the unmanaged code will require a fixed memory location to be used. Pinning, however, can lead to heap fragmentation if overused. This program demonstrates the pinning and unpinning process. Notice that while pinned, this memory is not eligible to be compacted during the garbage collection. Only once the `GCHandle` is freed, can this memory be collected. This highlights a key concept in memory management where explicit operations are needed to ensure memory can be cleaned up.

Beyond these code examples, utilizing tools such as the .NET PerfView tool, or using the built-in memory profilers within Visual Studio, are critical for real-world debugging and optimization. PerfView provides a deep view into the garbage collector's behavior, heap allocations, and overall memory usage patterns. It's invaluable for identifying memory leaks, fragmentation, and other potential issues. Visual Studio’s profiler also allows for similar insights and can be more convenient if you are already using the IDE for development.

For learning more about .NET memory management, I recommend reading the official .NET documentation on garbage collection. Specifically understanding how the generations work, how objects are promoted through the different generations, and how to manage larger objects on the LOH. Further resources on the .NET runtime, CLR internals, and performance best practices are also advisable. Additionally, understanding memory structures like stacks and heaps as well as the concept of virtual address spaces can further aid in comprehending how an application behaves at the process level.

In summary, monitoring and optimizing .NET memory usage involves careful coding, awareness of the GC's behavior, and the intelligent use of diagnostic tools.  It’s a continuous process, especially for high-performance applications. The three examples and the recommended resources can provide a strong foundation for managing your application's memory usage effectively.
