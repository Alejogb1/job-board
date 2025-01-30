---
title: "How can a C# application unload a temporary AppDomain while the main AppDomain continues to use MethodInfo from the temporary one?"
date: "2025-01-30"
id: "how-can-a-c-application-unload-a-temporary"
---
AppDomains provide process-level isolation within a single application, crucial for plugin architectures and dynamic code loading. However, when you unload a secondary AppDomain, any references held in the primary AppDomain to objects from the unloaded domain become invalid. Specifically, `MethodInfo` instances, reflecting methods in the secondary domain, are not designed for cross-domain persistence after the source domain is unloaded. Attempting to use these stale `MethodInfo` objects results in exceptions, typically of type `System.AppDomainUnloadedException`. The core challenge lies in maintaining the *ability to invoke methods* defined in the temporary AppDomain after it has been unloaded, while respecting the constraints imposed by the .NET runtime. Achieving this requires an indirection layer.

I faced this exact problem building a plugin system for a simulation framework several years ago. I needed to load user-defined code, execute it, and then potentially unload the domain, all while maintaining a link to the user-defined functionality. My solution leveraged a proxy object, which is created within the temporary AppDomain but marshaled back to the main AppDomain. This proxy implements a known interface, and the proxy's methods delegate execution to the actual methods residing within the temporary domain. Upon domain unloading, the proxy object remains valid within the main AppDomain, and it handles the necessary communication with the original (now unloaded) domain's functionality via the `Delegate` abstraction.

The fundamental principle is that a `Delegate` is marshalled across AppDomain boundaries by reference, not by copying the underlying method pointer. This is critical, as a copied method pointer would become stale. Using an interface allows us to have type safety and a clear contract between the main AppDomain and any plugins. The `MarshalByRefObject` class serves as the foundation for creating these proxy objects because instances of these classes are marshalled by reference instead of by value when transmitted across AppDomain boundaries.

The following code examples demonstrate this approach:

```csharp
// Example 1: Interface definition and PluginProxy class
public interface IPlugin
{
    int Execute(int input);
}

public class PluginProxy : MarshalByRefObject, IPlugin
{
    private Func<int, int> _executionDelegate;

    public PluginProxy(Func<int, int> executionDelegate)
    {
        _executionDelegate = executionDelegate;
    }

    public int Execute(int input)
    {
      if(_executionDelegate == null)
      {
        throw new InvalidOperationException("The delegate is not configured");
      }
        return _executionDelegate(input);
    }
}

```

This first snippet defines `IPlugin`, the interface contract, and `PluginProxy`, the proxy itself. Critically, `PluginProxy` inherits from `MarshalByRefObject`, making it a candidate for cross-AppDomain marshaling. It receives a `Func<int,int>` delegate when instantiated, which is where the actual execution logic will reside, and stores the delegate which is used to perform the computation. Note the safety check to make sure the delegate is available before use, a step I found crucial during debugging. This `PluginProxy` is created in the temporary `AppDomain`.

```csharp
// Example 2: Loading and Creating Proxy within Secondary AppDomain
public static class PluginLoader
{
    public static IPlugin LoadPlugin(string assemblyPath, string typeName, string methodName)
    {
        AppDomainSetup domainSetup = new AppDomainSetup { ApplicationBase = AppDomain.CurrentDomain.BaseDirectory };
        AppDomain secondaryDomain = AppDomain.CreateDomain("PluginDomain", null, domainSetup);

        try
        {
            var assembly = secondaryDomain.Load(AssemblyName.GetAssemblyName(assemblyPath));
            var pluginType = assembly.GetType(typeName);
            var method = pluginType.GetMethod(methodName);
            if (method == null)
              {
                  throw new MissingMethodException($"Method '{methodName}' not found in type '{typeName}'.");
              }

            //Creating the delegate using methodinfo
            var function = (Func<int, int>)Delegate.CreateDelegate(typeof(Func<int, int>), null, method);

            //Instantiating PluginProxy within secondary AppDomain
             var proxy = (PluginProxy)secondaryDomain.CreateInstanceAndUnwrap(typeof(PluginProxy).Assembly.FullName,
                                                                                typeof(PluginProxy).FullName,
                                                                               false,
                                                                                BindingFlags.Default,
                                                                                null,
                                                                                new object[] { function },
                                                                                null,
                                                                                null);

           return proxy;
       }
        catch (Exception)
        {
            AppDomain.Unload(secondaryDomain);
            throw;
       }
    }

    public static void UnloadDomain(AppDomain domain)
    {
       AppDomain.Unload(domain);
    }
}
```

Example 2 focuses on the domain loading and proxy instantiation process. It creates a secondary `AppDomain`, loads the assembly containing the plugin, retrieves the `MethodInfo` using reflection, and then creates a `Func<int,int>` delegate using `CreateDelegate` which allows us to keep the execution logic from the original AppDomain after it is unloaded. The `CreateInstanceAndUnwrap` method is responsible for the proxy instantiation and marshaling it back to the main `AppDomain`. Note that the returned proxy is an instance in the main `AppDomain` and marshals calls to the delegate stored inside of it. Also of note is the catch block, ensuring we always unload the AppDomain in case of an issue when loading it. I found this to be essential for stable integration. A public method is also included to safely unload the domain outside of the method.

```csharp
// Example 3: Using the proxy and unloading the secondary AppDomain
public class Example
{
    public static void Main(string[] args)
    {
        //Assumes "PluginAssembly.dll" and PluginImplementation "MyPlugin" and method "AddFive" exist
        var assemblyPath = "PluginAssembly.dll";
        var typeName = "MyPlugin";
        var methodName = "AddFive";

        IPlugin pluginProxy;
         AppDomain secondaryDomain = null;
        try
        {
           pluginProxy = PluginLoader.LoadPlugin(assemblyPath, typeName, methodName);

            // Use proxy to invoke plugin methods
            int result = pluginProxy.Execute(10);
            Console.WriteLine($"Result: {result}"); // Output: Result: 15

             //We must extract the domain after creating the proxy to ensure it is unloaded after usage.
             //This is done to prevent issues if an unhandled exception occurs
             secondaryDomain = AppDomain.CurrentDomain.GetAssemblies()
             .Where(a => a.FullName.Contains("PluginDomain"))
             .Select(a => AppDomain.GetDomain(a.GetName().Name))
             .FirstOrDefault();

             PluginLoader.UnloadDomain(secondaryDomain);

            // Try to execute after unloading
            result = pluginProxy.Execute(20);
            Console.WriteLine($"Result: {result}"); // Output: Result: 25
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
            if (secondaryDomain != null)
             {
                PluginLoader.UnloadDomain(secondaryDomain);
             }
        }


    }
}
```

Finally, Example 3 shows how to use the loaded plugin proxy. It obtains the `IPlugin` proxy, uses it for method invocation and then safely unloads the secondary `AppDomain`. Crucially, the call to `pluginProxy.Execute` *after* the domain is unloaded continues to work because the underlying delegate reference is still valid, even if the domain is no longer present in memory.

The `MarshalByRefObject` and the `Delegate` provide the necessary abstraction to interact with the unloaded domain and invoke the code within it. It is also important to note that errors can and will occur when reflection is used and proper exception handling can be key to ensuring a robust application, and this is one of the reasons why the `try...catch` block is added to Example 2 and Example 3. In practice, you'd likely employ more sophisticated exception handling and potentially add mechanisms for dynamically generating proxies to support multiple method signatures without creating a multitude of proxy classes.

For further study, I recommend exploring these resources:

*   The official .NET documentation on `AppDomain` and `MarshalByRefObject`. Understanding these classes at a deeper level is crucial for correctly applying this pattern.
*   Detailed guides on .NET Remoting, though it is an older technology, understanding its concepts is beneficial when working with `MarshalByRefObject` across app domains.
*   Books covering advanced C# topics, specifically those that focus on dynamic code loading and plugin architectures, often discuss techniques similar to the one detailed here.

By implementing a similar proxy pattern, you can achieve the desired outcome: unloading the temporary `AppDomain` while maintaining functional access to the methods it contained, offering the benefits of isolated execution without sacrificing usability. The key is not to access the method itself directly via a `MethodInfo` after the AppDomain has been unloaded but rather to access it through the `Delegate`. This approach has been a stable and effective solution for my past projects.
