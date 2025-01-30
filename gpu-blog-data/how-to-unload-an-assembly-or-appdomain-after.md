---
title: "How to unload an assembly or AppDomain after use?"
date: "2025-01-30"
id: "how-to-unload-an-assembly-or-appdomain-after"
---
In .NET Framework, and less directly in .NET Core/.NET, assembly unloading is not a straightforward or guaranteed operation, primarily due to the inherent design of the common language runtime (CLR) and its garbage collection mechanisms. The inability to explicitly “unload” an assembly, especially one loaded into the default `AppDomain`, stems from a complex web of references and dependencies the CLR manages. Attempting to force this can lead to unpredictable application behavior, memory leaks, and unrecoverable exceptions. However, there are techniques to manage the isolation and lifecycle of assemblies.

The fundamental challenge lies in the fact that assemblies loaded into the default `AppDomain` remain in memory for the lifetime of the process. The CLR does not provide a mechanism for unloading them directly. The garbage collector only releases memory held by objects, not assemblies themselves. Assemblies loaded in this context are considered a permanent part of the application’s domain. Therefore, the typical solution to deal with this limitation, particularly when loading third-party plugins or dynamically generated code, is to employ secondary `AppDomain`s.

An `AppDomain`, conceptually similar to a separate process within the same application, provides isolation. When assemblies are loaded within a secondary `AppDomain`, the entire domain can be unloaded, thus releasing the assemblies associated with it. Crucially, this is not "unloading" in the sense of releasing an assembly from system memory, but rather terminating the process space that contained the assembly. The system memory will be available for reuse only after the `AppDomain` and all its resources are reclaimed by the operating system.

My experience working on a plugin architecture for a financial modeling application highlighted this issue. We needed a flexible solution for loading and unloading custom calculation plugins, and naively trying to remove the plugin assembly after usage from the default `AppDomain` proved futile and unstable. The solution, as many before us had discovered, required working with separate `AppDomain`s. This isolates the loaded assembly, and when the need for the plugin ceases, its entire isolated environment could be unloaded cleanly.

Here's how one can create a separate `AppDomain`, load an assembly within it, and subsequently unload it, along with code examples demonstrating the process:

**Code Example 1: Setting up a secondary AppDomain and loading an assembly:**

```csharp
using System;
using System.Reflection;
using System.IO;

public class AssemblyLoader
{
    public void LoadAndExecuteAssembly(string assemblyPath, string typeName, string methodName)
    {
        AppDomain domain = AppDomain.CreateDomain("PluginDomain");
        try
        {
            var assembly = domain.Load(AssemblyName.GetAssemblyName(assemblyPath));
            var type = assembly.GetType(typeName);
            var instance = Activator.CreateInstance(type);
            var method = type.GetMethod(methodName);
            method.Invoke(instance, null);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading/executing assembly: {ex.Message}");
        }
        finally
        {
          AppDomain.Unload(domain);
          Console.WriteLine($"AppDomain '{domain.FriendlyName}' Unloaded successfully.");
        }

    }
}
```

**Commentary on Code Example 1:** This code illustrates the creation of a new `AppDomain` named "PluginDomain".  `AppDomain.CreateDomain` initializes the isolation boundary. The assembly is loaded using `domain.Load` using its `AssemblyName`, which bypasses the `AppDomain.CurrentDomain` and keeps it isolated. The code then dynamically creates an instance of the specified type from the assembly and executes the specified method. After use, the `AppDomain` is unloaded using `AppDomain.Unload(domain)`. Note that the entire try-finally block is crucial: unloading the domain must occur even if an exception is thrown while loading or executing the assembly. This is critical for preventing resources leaking.

**Code Example 2: Structure of a simple plugin assembly**

```csharp
using System;

namespace PluginAssembly
{
    public class SimplePlugin
    {
        public void PerformAction()
        {
            Console.WriteLine("Plugin action performed!");
        }
    }
}
```

**Commentary on Code Example 2:** This shows the basic structure of the plugin assembly that will be loaded and used in the first example. It contains a simple class named `SimplePlugin` with a `PerformAction` method, which will be called through reflection in the primary program. This is a simple example, in real-world plugin architecture, interfaces are commonly used and the class will implement these interfaces. The crucial part is that this `PluginAssembly.dll` resides as a file that can be loaded by the `AssemblyLoader`.

**Code Example 3: Invoking the AssemblyLoader:**

```csharp
public class Program
{
  public static void Main(string[] args)
  {
      var assemblyLoader = new AssemblyLoader();
      string assemblyPath = Path.Combine(Environment.CurrentDirectory, "PluginAssembly.dll");
      assemblyLoader.LoadAndExecuteAssembly(assemblyPath, "PluginAssembly.SimplePlugin", "PerformAction");
      Console.ReadKey();
  }
}
```

**Commentary on Code Example 3:** This shows how to invoke the `AssemblyLoader` class, providing the path to the plugin assembly and the fully qualified type name, along with the name of the method to be called. This encapsulates the whole loading, execution and unloading process in the secondary `AppDomain`, isolating the code. The `Path.Combine` ensures that the path to the assembly is correctly constructed, adapting to various operating systems. After execution, the `Console.ReadKey()` keeps the console open to inspect output.

It’s worth clarifying the practical implications of this isolation. Assemblies within an `AppDomain` cannot directly access objects or types in a different `AppDomain`. Any inter-domain communication requires techniques like remoting or serialization. In my experience developing the aforementioned financial application, we employed interfaces and .NET remoting to facilitate safe data transfer and invocation of methods across domain boundaries. The plugin architecture demanded that both the application and plugins be carefully designed with clear, serializable contracts.

For further study, I would recommend reviewing documentation on `AppDomain`,  specifically focusing on concepts of remoting, cross-domain communication, and security policies associated with `AppDomain` use. Examining examples and articles related to plugin architectures in .NET will also greatly enhance understanding. Resources focusing on garbage collection and memory management within the CLR, while not directly addressing assembly unloading, provide a crucial underpinning for understanding why certain techniques are necessary to manage assembly lifecycles and prevent resource leaks. These resources explain the subtle nuances of object lifetimes and the importance of deterministic resource disposal. The Microsoft documentation itself is a valuable starting point, along with community-authored books and articles. Understanding how CLR internals work significantly improves one's ability to build robust, long-running applications requiring dynamic loading of code. Finally, the proper use of `finally` blocks and the `using` statement is critical for cleanup operations related to resource-heavy assemblies, especially those involving database or network connectivity.
