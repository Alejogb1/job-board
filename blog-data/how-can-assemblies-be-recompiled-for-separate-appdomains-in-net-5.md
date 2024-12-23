---
title: "How can assemblies be recompiled for separate appdomains in .NET 5?"
date: "2024-12-23"
id: "how-can-assemblies-be-recompiled-for-separate-appdomains-in-net-5"
---

,  Having navigated the complexities of .net appdomains across several projects, particularly back when we were dealing with a microservices architecture that relied heavily on dynamic code loading (a rather hairy setup, I must say, but educational!), I've got some practical insights on recompiling assemblies for separate appdomains. In .net 5, appdomains themselves have been superseded by isolated processes, which changes the game a bit. We're not strictly dealing with appdomains anymore, but the core challenge of having separate execution contexts with potentially different assembly versions or configurations remains highly relevant. We just achieve it differently. Instead of appdomains, we now leverage separate process spaces or custom `assemblyloadcontext` instances for isolation.

Let's break down how we achieve this, focusing on the core of the original question: recompiling assemblies specifically for different isolated environments. It’s not about actually modifying the dll *file*, but rather about loading it within a different context, possibly using modified code at load time.

The primary hurdle here lies in assembly loading and dependency resolution. When you load an assembly into a .net runtime, it typically searches for dependencies in the application's base directory or other predefined locations, and loads them into its current load context. If you have multiple processes, or different versions of dependencies, you cannot use the default process-wide shared context for different needs. We need a way to direct each isolated context to its own set of assemblies.

For .net 5, we have mainly two options: using separate processes, or using custom `assemblyloadcontext`.

**Option 1: Separate Processes**

This is the most robust isolation technique. Each "appdomain" is, in fact, a completely separate os process. You compile your assemblies as needed, ensuring each process has only the required dependencies in its application directory. When you initiate a process, the os will create a clean environment with only the necessary files. This means you explicitly deploy the required versions within a particular process's executable directory. If we had an "appdomain1" needing `MyLibrary.dll` v1, and "appdomain2" needing `MyLibrary.dll` v2, we'd have two different executables each with their own dedicated local copy.

However, the "recompilation" is more of a “selective deployment" here. We're not changing the assembly *code*, but selecting where the assembly is loaded for different processes. We might use different compiler flags, configurations, or even distinct project setups to generate different versions of assemblies and deploy them separately, if the different versions are not merely configuration-based, but code-based changes.

This method works because the runtime loader for each process operates independently. The downside is, we have the overhead of inter-process communication (ipc) to send data between them.

Here is an example illustrating how to launch processes, and assume two processes may have different dependencies by having different project configurations that create different binaries:

```csharp
// ProcessLauncher.cs - Example of launching two isolated processes
using System;
using System.Diagnostics;
using System.IO;

public class ProcessLauncher
{
    public static void Main(string[] args)
    {
        // Path to the executables, each with a slightly different set of dependencies, potentially
        // due to building from two different project files.
        string app1Path = Path.Combine(Environment.CurrentDirectory, "app1\\app1.exe");
        string app2Path = Path.Combine(Environment.CurrentDirectory, "app2\\app2.exe");

        // Launch app1
        Console.WriteLine("Launching app1...");
        Process.Start(app1Path);

        // Launch app2
        Console.WriteLine("Launching app2...");
        Process.Start(app2Path);


        Console.WriteLine("Processes launched successfully.");
    }
}
```

Each app1.exe and app2.exe are simple .net executable projects built with separate configurations, and they might have different dependent assemblies deployed to their respective directories.

**Option 2: Custom `assemblyloadcontext`**

This is the more nuanced approach, and it gets closer to the idea of "recompiling" in a different context because it allows us to control how assemblies are loaded *within* a single process. We leverage the `system.runtime.loader.assemblyloadcontext` class. This class allows us to define custom resolution logic for assemblies. Instead of using the default loader, we can create an instance of `assemblyloadcontext`, load an assembly into it, and it will use that context's logic when searching for dependent assemblies. Thus we get different isolation even within the same process.

Here's how we might implement that:

```csharp
// MyCustomLoadContext.cs
using System;
using System.Reflection;
using System.Runtime.Loader;

public class MyCustomLoadContext : AssemblyLoadContext
{
    private string _basePath;

    public MyCustomLoadContext(string name, string basePath) : base(name, true)
    {
        _basePath = basePath;
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        string assemblyPath = System.IO.Path.Combine(_basePath, assemblyName.Name + ".dll");
        if (System.IO.File.Exists(assemblyPath))
        {
          Console.WriteLine($"Loading assembly '{assemblyName.Name}' from '{assemblyPath}'.");
          return LoadFromAssemblyPath(assemblyPath);

        }
        Console.WriteLine($"Could not load assembly '{assemblyName.Name}' using custom loader.");
        return null;  // Use the default resolution if we didn't find it in our context's directory.
    }
}

// Program.cs
using System;
using System.IO;
using System.Reflection;

public class Program
{
  public static void Main(string[] args)
  {
     // Paths to separate directories containing different assembly versions
    string context1Path = Path.Combine(Environment.CurrentDirectory, "context1");
    string context2Path = Path.Combine(Environment.CurrentDirectory, "context2");

    // Create two distinct custom load contexts
    var context1 = new MyCustomLoadContext("Context1", context1Path);
    var context2 = new MyCustomLoadContext("Context2", context2Path);


    // Load assemblies into their respective contexts. If they are named the same,
    // but the files in `context1` and `context2` are different versions, they
    // will not conflict with each other.

    var assembly1 = context1.LoadFromAssemblyName(new AssemblyName("MyLibrary"));
    var assembly2 = context2.LoadFromAssemblyName(new AssemblyName("MyLibrary"));

    // Let us assume each assembly has a class named `MyClass` and each defines
    // a method `GetName`, for example, and each will report the version or
    // the specific context they belong to
    if (assembly1 != null && assembly2 != null)
      {
        dynamic instance1 = Activator.CreateInstance(assembly1.GetType("MyClass"));
        dynamic instance2 = Activator.CreateInstance(assembly2.GetType("MyClass"));

        Console.WriteLine($"Instance 1: {instance1.GetName()}");
        Console.WriteLine($"Instance 2: {instance2.GetName()}");
      }
    else
    {
        Console.WriteLine("Failed to load one or more assemblies.");
    }
  }
}
```

In this example, `MyLibrary.dll` could exist in both `context1` and `context2`, but they could be different versions or builds. Each will get loaded by a different custom loader and the conflict will be resolved. The key part is overriding the `load` method on the `assemblyloadcontext` class.

**Considerations and Resources**

When dealing with assembly isolation, you should carefully consider a few factors. First, the overhead of creating a process vs creating a load context. Processes have higher overhead because they are completely separate, which is fine if you need that level of isolation and fault tolerance, but might be overkill if you only need to isolate library versions within a single application domain.

Second, ipc might become a necessity to communicate results between different contexts if you use processes. You can serialize data, or use named pipes, or sockets, among other techniques. With `assemblyloadcontext` the communication will happen within the same process, which reduces communication overhead.

Third, ensure the version of the .net runtime is the same across all processes or load contexts. In general, if you load .net 5 dlls within a .net 6 context or the other way around, you are likely to run into exceptions and other unpredictable behaviors.

For delving deeper, I'd recommend the following resources:

* **"CLR via C#" by Jeffrey Richter**: This book provides an in-depth understanding of the .net common language runtime, which is very useful in understanding how assembly loading works and why we do what we do with the mechanisms above.
* **The official .net documentation on `system.runtime.loader.assemblyloadcontext`**: The microsoft documentation is fairly comprehensive, especially the examples, and it should be one of your go to resources when doing this kind of advanced work with assembly loading.

Working with different assembly contexts is a complex topic, and you will discover, as I did through some painful debugging cycles, that carefully planning your loading strategies is crucial. These approaches will get you very far, and allow you to achieve what used to be done with appdomains but now using modern alternatives like processes or custom load contexts. The main things to keep in mind are isolation, dependency resolution, and communication strategies when dealing with isolated runtimes.
