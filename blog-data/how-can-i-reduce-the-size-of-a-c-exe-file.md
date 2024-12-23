---
title: "How can I reduce the size of a C# .exe file?"
date: "2024-12-23"
id: "how-can-i-reduce-the-size-of-a-c-exe-file"
---

Let's tackle this one; the quest to shrink .exe sizes is a perennial favorite, and I’ve spent more hours than I care to count optimizing binaries over the years. The perceived bloat of compiled executables can often be quite jarring, especially coming from leaner languages, but with a systematic approach, considerable reductions are indeed achievable in C#.

The size of a .net executable, in essence, is determined by several core components. We're not just talking about your meticulously crafted code here; rather, we’re dealing with metadata, intermediate language (il), framework dependencies, and any embedded resources. All of these can contribute significantly to the final file size, and each can be trimmed, albeit through different methods.

One of the initial culprits and often the largest, is unnecessary dependency baggage. The .net framework is powerful, but it brings its share of overhead. Early in my career, I was involved in a project where we were shipping a very small utility; yet, its .exe was several megabytes in size. The bulk of this was due to the full .net framework runtime being bundled with the application.

**Publishing Options: Trim and Self-Contained Deployment**

The first, and often most effective step, is to examine the *publish* process. If you’re using dotnet cli, we want to move beyond the default debug build. We need to carefully consider the *publish* configuration. You'll be using the `dotnet publish` command, and the options you specify here will be critical.

Instead of relying on a framework-dependent deployment (where the target machine needs the .net framework installed), a significant reduction can be achieved with *self-contained deployment* (SCD). In SCD, all necessary framework components are bundled into your executable. Now, this may seem counterintuitive, initially seeming like an *increase* in size but the benefit here is *trimming*. Specifically, we want the *Trimmed* option.

By adding `<PublishTrimmed>true</PublishTrimmed>` to your .csproj file within a `PropertyGroup`, you can tell the publish process to perform an *illinker* step. This illinker examines the actual usage of classes and methods within your application and removes the rest. This often yields substantial size savings. The trade-off is slightly increased build time as this extra step occurs. However, I've consistently found the output to be considerably smaller.

Here is an example, where we modify the .csproj:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <PublishTrimmed>true</PublishTrimmed>
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
  </PropertyGroup>

</Project>

```

Here, the `<SelfContained>true</SelfContained>` ensures the framework is bundled, the `<PublishTrimmed>true</PublishTrimmed>` performs the trimming process and `<PublishSingleFile>true</PublishSingleFile>` outputs to a single executable. Finally, the `<RuntimeIdentifier>win-x64</RuntimeIdentifier>` specifies the target OS and architecture, which is essential for SCD. Remember to switch out the `win-x64` to the appropriate OS (e.g., `linux-x64`, `osx-x64`). After you make these additions, the command for publishing should look something like: `dotnet publish -c Release`.

**Assembly Linking and Native AOT**

Beyond the trimmer, you can also look towards *native aot* compilation. This option compiles your code directly to machine code ahead-of-time, rather than il, completely removing the .net runtime and greatly reducing the size of required files. This provides the smallest possible executable in .net, but is a more involved process. It also introduces limitations in terms of code usage such as reflection and dynamic loading.

The main benefit here isn't the binary size of your primary executable per-se; its more so that no framework dlls need to be included in the application.

To use native aot, you will need to add the appropriate nuget package: `Microsoft.NET.Runtime.NativeAOT`, and then set `<PublishAot>true</PublishAot>` in your project file. Additionally, you’ll have to change your target framework from net to `net8.0-windows`, or similar, depending on your target operating system. Note, this method does not currently support every operating system. I’ve found it essential to understand the associated limitations of aot before committing to it, and have personally had to revert from this approach more than once.

Here is an example of a `.csproj` file using AOT:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0-windows</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <PublishAot>true</PublishAot>
    <PublishSingleFile>true</PublishSingleFile>
    <SelfContained>true</SelfContained>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Runtime.NativeAOT" Version="8.0.0" />
  </ItemGroup>

</Project>
```

Again, you will then run `dotnet publish -c Release`.

**Resource Management and Further Optimizations**

Finally, consider how you handle your resources. Embedded resources, like images or large datasets, often contribute significantly to file sizes. Where feasible, consider loading these resources from external files or servers. This is especially applicable if these resources are not strictly required for initial application startup. Furthermore, for simple text resources consider embedding as simple strings within the source code to avoid the overhead of resource management.

I’ve been in numerous situations where I had overlooked large embedded files. Even if these are compressed, the added overhead can add up rapidly.

As an example, if you have a very small string based text file, instead of embedding it as a resource, you can embed it directly as a string in code. Here’s an example:

```csharp
using System;
public static class StringResource
{
    public static string MyResource = "This is a simple text string";

    public static void PrintResource()
    {
         Console.WriteLine(MyResource);
    }
}
```

While not a huge saving, especially with modern compression algorithms, this demonstrates the point of avoiding embedded resources when the overhead is not needed.

**Recommendations for Further Study**

For deeper understanding, I highly recommend diving into the official .net documentation on publishing and deployment. Specifically, I recommend the ".net deployment documentation", found easily through a quick web search. This is a crucial source and provides the fundamental concepts needed to achieve optimal binary size. Also, look into the documentation on the *illinker*.

Additionally, "C# 10 in a Nutshell" by Joseph Albahari is an excellent reference that gives considerable background on underlying aspects of the compiler and runtime, including the concepts mentioned here. Finally, some research on general executable trimming techniques, and the inner workings of the PE file format can illuminate the deeper aspects of executable size.

In summary, reducing the size of a c# .exe is an iterative process, requiring careful evaluation of publish options, resource management, and potentially more aggressive techniques such as native aot compilation. There's no one-size-fits-all solution, but by systematically addressing these elements, I've seen dramatic reductions in executable sizes across numerous projects, and I’m confident you can too.
