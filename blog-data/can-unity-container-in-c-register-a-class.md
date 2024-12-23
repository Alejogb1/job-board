---
title: "Can Unity container in C# register a class?"
date: "2024-12-23"
id: "can-unity-container-in-c-register-a-class"
---

Let's tackle this one. I've spent a fair share of time architecting applications, and dependency injection with Unity, particularly in C#, is something I’ve encountered quite frequently. The short answer is, yes, absolutely, Unity can register classes, but the details are where things get interesting. It’s not just about simple registration; there’s a spectrum of possibilities that dictate how the container manages dependencies, lifetime, and abstraction.

Early in my career, I recall a large enterprise application we were developing. We initially took a very manual dependency management approach, which, as you can probably guess, quickly devolved into a chaotic mess of tightly coupled components. Moving to a dependency injection (DI) container like Unity was transformational for us. It streamlined our architecture and improved the testability significantly.

Now, let's unpack this. When we talk about “registering a class” with Unity, we're generally referring to the process of informing the container how to create an instance of that class when a dependency is requested. The most basic form is registering a class against itself. This means, when the container encounters a request for a type, it knows to instantiate that same type using its public constructor.

For instance, consider a simple logging class:

```csharp
public class ConsoleLogger
{
  public void Log(string message)
  {
      Console.WriteLine(message);
  }
}
```

We could register this with Unity like so:

```csharp
using Unity;

public static void RegisterConsoleLogger()
{
  var container = new UnityContainer();
  container.RegisterType<ConsoleLogger>();
  // Further configurations can be placed here.
}
```

Here, `container.RegisterType<ConsoleLogger>();` is instructing Unity that whenever an instance of `ConsoleLogger` is required, it should create one using the parameterless constructor. So, later in your code, you could resolve this:

```csharp
 var logger = container.Resolve<ConsoleLogger>();
 logger.Log("Application started.");
```

This basic registration is fine for simple cases. However, real-world applications rarely operate in such isolation. Usually, you'll need to deal with abstract types such as interfaces. This is where Unity truly shines. Instead of registering concrete classes directly, you typically register mappings between interfaces and concrete implementations. This adheres to the dependency inversion principle, making your code more modular and easier to test.

Let’s assume we have an `ILogger` interface:

```csharp
public interface ILogger
{
    void Log(string message);
}
```

And our `ConsoleLogger` implements it:

```csharp
public class ConsoleLogger : ILogger
{
   public void Log(string message)
   {
       Console.WriteLine(message);
   }
}
```

To register this in Unity, we use the following:

```csharp
using Unity;
using Unity.Lifetime;

public static void RegisterILogger()
{
  var container = new UnityContainer();
  container.RegisterType<ILogger, ConsoleLogger>(new ContainerControlledLifetimeManager());
  // ContainerControlledLifetimeManager dictates that the same instance will always be returned
  // when requesting an ILogger
}
```

Now, anytime a class requests an `ILogger` as a dependency, Unity will provide an instance of `ConsoleLogger`. This decouples the consuming class from a specific implementation, meaning you could, say, switch to logging to a file without modifying the consuming class. Just register a new implementation against `ILogger`.

You’ll notice the `ContainerControlledLifetimeManager` included above, which dictates how instances are created and their lifetimes managed. Unity provides other lifetime managers. Some include:

*   **TransientLifetimeManager:** Creates a new instance each time the dependency is resolved.
*   **HierarchicalLifetimeManager:** Shares an instance within the scope of its current container hierarchy.
*   **PerResolveLifetimeManager:** Creates a new instance per call to the resolve method of the container, and all of its dependencies.

The choice of lifetime manager is crucial for avoiding memory leaks and ensuring your objects behave as expected. If no lifetime manager is defined, by default unity will use a transient manager, which creates new instances for each resolve.

A more intricate case can arise when dealing with classes that have constructor parameters themselves. For instance, suppose we want to have a logger that can filter messages based on severity, which we would pass into the constructor:

```csharp
public class FilteredLogger : ILogger
{
    private readonly string _severity;
    public FilteredLogger(string severity)
    {
      _severity = severity;
    }

    public void Log(string message)
    {
        Console.WriteLine($"[{_severity}]: {message}");
    }
}
```

Here, the constructor of `FilteredLogger` takes a `string` as a parameter. Unity needs to know how to provide this parameter during resolution. We can achieve this by specifying the constructor parameter value upon registration:

```csharp
using Unity;
using Unity.Lifetime;
using Unity.Resolution;

public static void RegisterFilteredLogger()
{
    var container = new UnityContainer();
    container.RegisterType<ILogger, FilteredLogger>(
         new InjectionConstructor(new ResolvedParameter<string>("INFO")),
         new ContainerControlledLifetimeManager());
}
```

In this snippet, `InjectionConstructor` is used to supply the constructor argument. The `ResolvedParameter<string>("INFO")` specifies that the string parameter should always be "INFO" whenever `FilteredLogger` is instantiated. There are multiple ways to achieve this. For example: `InjectionParameter` can be used for concrete values or `InjectionProperty` for public setters.

Furthermore, you can use named registrations when you want to register multiple implementations of the same interface. This could be needed when for instance, in addition to `ConsoleLogger` we also have a `FileLogger`. In such cases, you might want to resolve each of them specifically by name.

While Unity provides a robust set of features, it's always good to check out other dependency injection containers available in the .NET ecosystem. For a deeper understanding of DI patterns and practices, I'd recommend reading "Dependency Injection in .NET" by Mark Seemann. Also, "Patterns of Enterprise Application Architecture" by Martin Fowler is great for understanding broader enterprise application patterns, some of which strongly favor DI.

In conclusion, Unity's container can indeed register classes, and it does so in a highly flexible and configurable manner. The ability to register concrete implementations against interfaces, manage object lifecycles, and handle constructor parameters make it a very versatile tool for modern .NET application development. Understanding these concepts and how to apply them correctly is crucial for building maintainable and testable software.
