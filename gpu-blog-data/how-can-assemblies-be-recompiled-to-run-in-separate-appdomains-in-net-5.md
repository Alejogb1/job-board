---
title: "How can assemblies be recompiled to run in separate appdomains in .NET 5?"
date: "2025-01-26"
id: "how-can-assemblies-be-recompiled-to-run-in-separate-appdomains-in-net-5"
---

AppDomain isolation, while no longer the primary focus in modern .NET development due to the dominance of process and container-based isolation, still presents a relevant scenario when dealing with legacy systems or for granular resource management within a single process. Specifically, the traditional `System.AppDomain` class is obsolete starting with .NET Core 3.0 and has no direct analogue in .NET 5 and later versions. However, the fundamental concept of running assemblies in an isolated environment can be achieved through a combination of dependency loading strategies and custom `AssemblyLoadContext` implementations. Therefore, the challenge is not to recompile assemblies to run in *AppDomains* (as they no longer exist), but to load them in distinct, isolated contexts which closely emulate the behavior of former appdomains.

The core issue with attempting direct use of appdomains is the underlying architecture shift. AppDomains provided operating system-level process isolation, but within a single .NET runtime process. This model was replaced by the concept of `AssemblyLoadContext` which operates entirely within the .NET runtime and provides isolated assembly loading and type resolution, but without the process-level guarantees of appdomains. This change is driven by the evolution of .NET towards cross-platform, containerized deployment where the unit of isolation is typically the operating system process, not isolated units within the process.

To achieve the behavior formerly associated with appdomains in .NET 5+, the primary mechanism is to create custom instances of `System.Runtime.Loader.AssemblyLoadContext`. Each `AssemblyLoadContext` instance functions as an isolated unit where loaded assemblies and their dependencies are kept separate from other contexts within the same process. This ensures that different versions of the same assembly, or conflicting dependencies, can coexist without causing conflicts.

To load an assembly into a custom `AssemblyLoadContext`, the following approach is typically followed:

1. **Define a Custom `AssemblyLoadContext`:** Inherit from `System.Runtime.Loader.AssemblyLoadContext`, overriding methods to control assembly resolution. This allows granular control over how dependencies are loaded and resolved.
2. **Create an Instance:** Instantiate the custom `AssemblyLoadContext`.
3. **Load the Assembly:** Use methods on the `AssemblyLoadContext` such as `LoadFromAssemblyPath` to load the target assembly.
4. **Invoke Code:** Once the assembly is loaded, reflection can be used to create instances of types within the assembly and call methods.

Here's the first code example demonstrating the creation of a custom `AssemblyLoadContext`:

```csharp
using System;
using System.Reflection;
using System.Runtime.Loader;
using System.IO;

public class IsolatedLoadContext : AssemblyLoadContext
{
    private AssemblyDependencyResolver _resolver;

    public IsolatedLoadContext(string assemblyPath) : base(isCollectible: true)
    {
        _resolver = new AssemblyDependencyResolver(assemblyPath);
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        string? assemblyPath = _resolver.ResolveAssemblyToPath(assemblyName);
        if (assemblyPath != null)
        {
            return LoadFromAssemblyPath(assemblyPath);
        }

        return null; // Let default load context attempt to resolve.
    }

    protected override IntPtr LoadUnmanagedDll(string unmanagedDllName)
    {
         string? libraryPath = _resolver.ResolveUnmanagedDllToPath(unmanagedDllName);
         if (libraryPath != null)
         {
            return LoadUnmanagedDllFromPath(libraryPath);
         }

        return IntPtr.Zero;
    }
}
```

**Commentary:** This class, `IsolatedLoadContext`, inherits from `AssemblyLoadContext`. The constructor sets up an `AssemblyDependencyResolver` which is used to find assemblies and unmanaged dlls in the same directory as the primary assembly. The `Load` and `LoadUnmanagedDll` methods are overridden to instruct the context how to resolve assembly and native library dependencies respectively. Setting `isCollectible: true` allows the `AssemblyLoadContext` to be garbage collected when no longer in use. This reduces the resource footprint of the application over time.

The second code example showcases how to load an assembly into the custom context and invoke a method.

```csharp
public static class AssemblyLoader
{
     public static void LoadAndExecute(string assemblyPath, string typeName, string methodName, object[] methodArgs)
     {
          IsolatedLoadContext context = new IsolatedLoadContext(assemblyPath);
          Assembly assembly = context.LoadFromAssemblyPath(assemblyPath);

          if(assembly == null)
          {
               throw new Exception($"Failed to load the assembly at {assemblyPath}.");
          }

          Type? type = assembly.GetType(typeName);
          if (type == null)
          {
                throw new Exception($"Type {typeName} not found in assembly.");
          }
          object? instance = Activator.CreateInstance(type);
           if(instance == null)
          {
                throw new Exception($"Could not create an instance of type {typeName}.");
          }

          MethodInfo? method = type.GetMethod(methodName);
          if(method == null)
          {
                throw new Exception($"Method {methodName} not found on type {typeName}.");
          }
        try
        {
            method.Invoke(instance, methodArgs);
        }
        catch (Exception ex)
        {
             Console.WriteLine($"Exception during method invocation {ex}");
        }

        context.Unload();
     }
}
```

**Commentary:** This static `AssemblyLoader` class contains the core logic for loading assemblies into a custom context and then using reflection to create an instance of a given type and invoke a specified method. The `AssemblyLoadContext.Unload()` method is called after execution. This allows the CLR to release resources associated with the context. The `try-catch` block prevents unhandled exceptions during execution.

The third code example illustrates the usage of the loader to invoke method from an assembly in an isolated context.

```csharp
// Main entrypoint
public static class Program
{
      public static void Main(string[] args)
      {
           // Assume target assembly path is in the same directory.
           string targetAssemblyPath = "TargetAssembly.dll"; // Replace with actual assembly.
            try {
               AssemblyLoader.LoadAndExecute(targetAssemblyPath, "TargetAssembly.MyClass", "MyMethod", new object[]{ "Hello from Isolated context." });
               // If another assembly needs to be loaded it can be done in the same fashion but with the other assembly's path.
               // This gives a demonstration of how assemblies can be loaded in separate contexts.
               AssemblyLoader.LoadAndExecute(targetAssemblyPath, "TargetAssembly.MyClass", "MyMethod", new object[]{ "Hello from another isolated context." });
           }
           catch (Exception ex){
               Console.WriteLine($"Error:{ex.Message}");
           }

            Console.ReadLine();
      }
}

//  TargetAssembly.dll Content (Simplified Example)
// namespace TargetAssembly
// {
//     public class MyClass
//     {
//         public void MyMethod(string message)
//         {
//             Console.WriteLine("From TargetAssembly: " + message);
//         }
//     }
// }
```

**Commentary:** In this simplified `Program` class, the `LoadAndExecute` method is used twice to load and execute methods from a hypothetical `TargetAssembly.dll`. This example shows how to use the `AssemblyLoader` from the second example. It is important to note that the `TargetAssembly.dll` would be a separate compiled assembly. Each call to `LoadAndExecute` in the `Program` creates its own `IsolatedLoadContext`, thereby effectively replicating the behavior of different appdomains without relying on the obsolete `System.AppDomain` class. The code also demonstrates how this method can be called multiple times for isolated invocations.

For deeper understanding, resource recommendations should include exploring Microsoft's official documentation on `System.Runtime.Loader.AssemblyLoadContext` and related classes. Additionally, consulting resources on assembly loading, reflection, and dependency resolution within the .NET runtime will prove invaluable. Furthermore, detailed articles on modern .NET application architecture and isolation strategies will help place the concept of context-based isolation in a broader context. Studying these foundational aspects provides a comprehensive understanding of how to achieve isolation in .NET 5+ in a manner that aligns with current best practices. This combination of theoretical and practical knowledge is critical for effectively managing application complexity and avoiding compatibility issues when integrating older components into modern systems.
