---
title: "How can C# .exe file size be reduced?"
date: "2025-01-30"
id: "how-can-c-exe-file-size-be-reduced"
---
Executable file size in .NET applications, especially those compiled for Windows using C#, often becomes a point of concern, particularly for distribution or resource-constrained environments. My experience, gained from years of developing desktop utilities and smaller embedded applications using .NET, indicates that several effective strategies can significantly reduce the final .exe size. The core problem often isn't the inherent size of .NET itself, but rather the inclusion of unnecessary libraries, debugging symbols, or inefficient packaging.

A major contributor to bloated executable sizes is the inclusion of unused code and dependencies. The .NET runtime and its associated libraries are robust but extensive, encompassing far more functionality than a typical application utilizes. By default, .NET compilers may include large portions of the framework even if only a small fraction is actually invoked by the application's code. A key approach involves meticulously trimming down these inclusions. This is achieved through techniques targeting both the compilation and publishing phases. At compile time, optimizing code structure, reducing unnecessary abstractions, and utilizing .NET's built-in features effectively contributes to more concise intermediate language (IL). However, the most significant gains are typically realized at the publish phase through options that explicitly control the final executable.

**Code Example 1: Linker Trimming with `PublishTrimmed`**

The .NET SDK provides a powerful mechanism for trimming unused code during the publish process, known as linker trimming. This functionality, introduced in .NET 6, allows for analyzing the application's dependencies and removing portions of the framework that are not explicitly referenced. To enable this feature, the project file must be configured. This method has saved me substantial space on countless occasions, particularly when migrating from full framework to .NET core or later.

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <PublishTrimmed>true</PublishTrimmed>
    <TrimMode>link</TrimMode>
    <SelfContained>true</SelfContained>
    <PublishSingleFile>true</PublishSingleFile>
  </PropertyGroup>

</Project>
```

*   **`PublishTrimmed`:** This crucial flag enables the linker, which performs static analysis of the project to determine which parts of the framework are actually used and removes the rest.
*   **`TrimMode`:** While `link` is the standard, consider `copyused` if further testing with `link` fails. `copyused` is less aggressive but can still reduce file sizes while being less likely to introduce runtime errors.
*   **`SelfContained`:** Ensures the .NET runtime is bundled in with the executable, rather than relying on the host machine’s runtime. This makes the .exe larger, but self-contained applications are more robust.
*   **`PublishSingleFile`:** This bundles the application's code, dependencies, and runtime into a single .exe, simplifying distribution but can be larger. I often use it for small utilities.

This combination often results in a much smaller final executable. However, it's essential to test thoroughly after applying these settings. Linker trimming can sometimes remove code that's accessed through reflection or other dynamic means, potentially causing runtime issues. If that happens, additional configurations are needed.

**Code Example 2:  Assembly Embedding and Compression**

Beyond linker trimming, another technique I employ involves embedding and compressing assemblies. Rather than distributing separate .dll files along with the .exe, they can be embedded directly within the .exe itself, creating a self-contained application. Compression further reduces the overall size. This is particularly useful for small-to-medium-sized projects with numerous dependencies. There are third-party tools that I often employ for this purpose, however, a manual approach is achievable.

```csharp
using System;
using System.IO;
using System.IO.Compression;
using System.Reflection;

public static class AssemblyEmbedder
{
    public static void CompressAndEmbedAssembly(string assemblyPath)
    {
        byte[] assemblyBytes = File.ReadAllBytes(assemblyPath);
        byte[] compressedBytes = CompressBytes(assemblyBytes);

        using(Stream resourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream("EmbeddedAssembly.Resources"))
        {
          using(var writer = new BinaryWriter(resourceStream))
            writer.Write(compressedBytes);
        }

    }

    private static byte[] CompressBytes(byte[] data)
    {
        using (var outputStream = new MemoryStream())
        {
            using (var compressionStream = new GZipStream(outputStream, CompressionLevel.Optimal))
            {
                compressionStream.Write(data, 0, data.Length);
            }
            return outputStream.ToArray();
        }
    }

    public static byte[] DecompressBytes(byte[] compressedData)
    {
          using (var inputStream = new MemoryStream(compressedData))
        {
            using (var decompressionStream = new GZipStream(inputStream, CompressionMode.Decompress))
            {
                using (var outputStream = new MemoryStream())
                {
                    decompressionStream.CopyTo(outputStream);
                    return outputStream.ToArray();
                 }
            }
        }
    }
}

// Usage Example
AssemblyEmbedder.CompressAndEmbedAssembly("path/to/your/dependency.dll");
```

*   The `CompressAndEmbedAssembly` method reads an assembly, compresses it using GZip, and then embeds the compressed bytes as a manifest resource in the current assembly.
*   The compression logic is handled in the `CompressBytes` method, and decompression is implemented in `DecompressBytes`.
*   During program initialization (not shown) the application will need to hook into the assembly resolver and decompress and load this embedded assembly.

This example demonstrates the basic principle. The actual implementation requires careful handling of resource loading and assembly resolution. While third-party tools handle this more elegantly, this illustrates the underlying mechanism I utilize in a manual context for fine control.

**Code Example 3: Optimizing Code and Data Structures**

While compilation and publish settings are primary drivers of executable size, the design of the application itself significantly influences the outcome. Unnecessary use of complex data structures, verbose code constructs, and poorly written loops can lead to bloat. Here's an example showing a potential optimization:

```csharp
// Inefficient approach (Illustrative example for comparison)
public class DataProcessor {
  public void Process(List<string> items) {
      foreach(var item in items) {
        if (item.Contains("example")) {
            // do something
           Console.WriteLine($"Found {item}");
        }
      }
   }
}
```
```csharp
// Optimized approach using a simple loop over an array
public class OptimizedDataProcessor {
  public void Process(string[] items) {
      for (int i=0; i< items.Length; i++) {
          if(items[i].Contains("example")) {
              // do something
               Console.WriteLine($"Found {items[i]}");
          }
      }
  }
}
```

*   The initial `DataProcessor` uses a `List<string>` and `foreach` loop. While this is common practice, it incurs a slight overhead.
*   The optimized `OptimizedDataProcessor` uses an array of strings and a basic `for` loop. This is a simple example, but it illustrates how minimizing abstractions and using more direct approaches can contribute to smaller code. In practice, this can mean choosing value types over reference types when appropriate, avoiding unnecessary string manipulations, and careful algorithmic choices.

While the size difference from such a simple example is minor, cumulatively, such optimization efforts across an entire project can result in noticeable reductions in final executable size.

Further techniques, such as using appropriate data structures (dictionaries instead of lists for lookups, for instance), avoiding deep inheritance hierarchies, and profiling code to identify performance bottlenecks (which often correlate with code size), also play critical roles.

In summary, reducing C# .exe file size involves a multifaceted approach encompassing: aggressive linker trimming (`PublishTrimmed` and related settings), assembly embedding and compression (though third-party tools often provide robust implementations), diligent code optimization, and careful library selection. Each of these contributes to a leaner and more efficient final executable.

For further information on optimizing .NET applications, I recommend exploring resources focusing on .NET SDK documentation, particularly sections dealing with publish settings, trimming, and advanced compilation. Official documentation relating to the GZip compression library and associated resource management techniques is also helpful. Additionally, articles and blog posts centered around .NET performance optimization provide valuable insights into efficient code construction and data management practices. Lastly, various resources and communities are available discussing the fine-grained control of assembly linking and manipulation for those seeking deeper levels of optimization. These resources, when consulted alongside hands-on practice, are invaluable in the effort to minimize executable size.
