---
title: "Why does Unity resolve the same parameterized constructor twice?"
date: "2024-12-23"
id: "why-does-unity-resolve-the-same-parameterized-constructor-twice"
---

Right then, let’s tackle this peculiar behavior regarding Unity’s parameterized constructor resolution. I’ve seen this manifest in a few projects across my career, and it always throws developers for a loop initially. It's less about a bug and more about understanding Unity's injection mechanism and how it interacts with reflection, especially when generics and multiple registrations come into play. It can seem counterintuitive that a constructor seems to be called multiple times, specifically the parameterized one, when you’d naturally expect it to be instantiated only once per request for that type.

Essentially, the issue stems from Unity’s internal workings regarding lifetime management and type registration. When you're asking Unity to resolve a type, particularly one that takes dependencies through its constructor, Unity performs a series of steps. It needs to not only create an instance of your target type but also ensure all of its dependencies are satisfied. If you're using constructor injection, which you should be, Unity will then need to find or create those dependencies using its container. This is where things can become… interesting.

Think of it this way: each *resolve* operation requires Unity to determine which constructor to use. If your class has multiple constructors, or if it has parameterized constructors where it needs to figure out what parameters to inject, Unity goes through a process to find a "best fit." It does this by considering the declared constructor parameters and the types registered in the container.

The core reason you see a parameterized constructor being called multiple times isn't that Unity is creating multiple *instances* of the class. Instead, it's that the constructor resolution process can occur repeatedly within a single resolve call. This is more pronounced with complex object graphs involving generics and multiple type registrations because Unity needs to try different potential solutions to create a fully formed instance. Unity essentially attempts several resolutions during the construction of a complex type, resolving dependencies and sub-dependencies, sometimes trying the constructor multiple times when trying different paths of resolution. This is a key difference—the constructor is run multiple times during the resolution process, but an instance is usually only produced once in the final result.

Consider this scenario. Suppose you have a class `ServiceA` that depends on `Logger`. You also have another class, `ServiceB`, which, in turn, depends on `ServiceA` *and* `Logger`. Let’s say you register these types using `RegisterType` where specific concrete implementations are mapped using an interface, for example `ILogger`. You then try to resolve `ServiceB` directly using Unity.

Here's how that internal process looks conceptually:

1.  Unity attempts to create `ServiceB`. It sees a parameterized constructor that needs a `ServiceA` and an `ILogger`.
2.  Unity resolves `ServiceA`. It sees a parameterized constructor that needs an `ILogger`.
3.  Unity resolves `ILogger`, creates an instance.
4.  Unity calls the `ServiceA` constructor with the injected `ILogger` instance.
5.  Unity is now back to resolving `ServiceB`, it already has `ServiceA`, and knows it needs `ILogger`.
6.  It sees a cached registration for the `ILogger` interface which it had created before and reuses the same `ILogger` instance from before.
7.  Unity calls the `ServiceB` constructor with the resolved `ServiceA` instance and the already injected `ILogger` instance.

Note, the `ILogger` is only ever instantiated once in the examples given. The key point is that the constructor of `ServiceA` is only called when trying to resolve `ServiceA` and it is only used as a step during the resolution of `ServiceB`. The underlying lifetime and resolution mechanism of Unity results in the constructor call happening as part of Unity's resolution process, even if only a single final instance is ever created.

To further solidify this, let’s look at some simplified examples in C#.

**Example 1: Basic Dependency Injection**

```csharp
using Unity;
using System;

public interface ILogger
{
    void Log(string message);
}

public class ConsoleLogger : ILogger
{
    public void Log(string message)
    {
        Console.WriteLine($"Log: {message}");
    }
}

public class ServiceA
{
    private ILogger _logger;
    public ServiceA(ILogger logger)
    {
         _logger = logger;
        Console.WriteLine("ServiceA Constructor Called");
    }
}

public class ServiceB
{
    private ServiceA _serviceA;
    private ILogger _logger;

    public ServiceB(ServiceA serviceA, ILogger logger)
    {
        _serviceA = serviceA;
        _logger = logger;
        Console.WriteLine("ServiceB Constructor Called");
    }
}


public class Example
{
  public static void Main(string[] args)
    {
      var container = new UnityContainer();
      container.RegisterType<ILogger, ConsoleLogger>();
      container.RegisterType<ServiceA>();
      container.RegisterType<ServiceB>();
      var serviceB = container.Resolve<ServiceB>();

      Console.ReadLine();
    }
}
```
In this scenario, `ServiceA` and `ServiceB` both have their constructors called as Unity resolves dependencies. You will see the console output in this order: `ServiceA Constructor Called` and `ServiceB Constructor Called`. Crucially, the `ServiceA` constructor was only called once for the sake of creating a `ServiceB` instance.

**Example 2: Generic Type Registration**
Let’s add a generic layer into the mix to see a deeper behaviour of how Unity resolves constructors, which may help explain why you are seeing it be called twice in your specific case:

```csharp
using Unity;
using System;

public interface IRepository<T> {
  void Save(T entity);
}

public class Entity {
    public int Id { get; set;}
}

public class SqlRepository<T> : IRepository<T>
{
  private ILogger _logger;

    public SqlRepository(ILogger logger)
  {
      _logger = logger;
      Console.WriteLine($"SqlRepository constructor called for type {typeof(T).Name}");
  }

  public void Save(T entity) {
    Console.WriteLine($"Saving {typeof(T).Name}");
  }
}

public class UserService
{
    private IRepository<Entity> _repository;

    public UserService(IRepository<Entity> repository)
    {
      _repository = repository;
      Console.WriteLine($"UserService constructor called");
    }
    public void Save(Entity entity)
    {
      _repository.Save(entity);
    }
}

public class Example
{
  public static void Main(string[] args)
    {
      var container = new UnityContainer();
      container.RegisterType<ILogger, ConsoleLogger>();
      container.RegisterType(typeof(IRepository<>), typeof(SqlRepository<>));
      container.RegisterType<UserService>();
      var userService = container.Resolve<UserService>();
      userService.Save(new Entity() {Id = 1});
      Console.ReadLine();
    }
}
```

In this example, when `UserService` is resolved, Unity uses the generic type definition to create the `SqlRepository<Entity>`, and you will see the constructor called once for the concrete type `Entity`. This is because Unity uses the definition to resolve. Even though we register the generic type definition, Unity still resolves the correct concrete type from the given definition. Note how `SqlRepository` constructor will be called with concrete type `Entity` when resolving the `UserService`.

**Example 3: Multiple Registrations with Conflicts**

```csharp
using Unity;
using System;

public interface IService {
    void DoSomething();
}

public class ServiceImplA : IService
{
  private ILogger _logger;
  public ServiceImplA(ILogger logger)
  {
    _logger = logger;
      Console.WriteLine("ServiceImplA constructor called");
  }

    public void DoSomething()
    {
        Console.WriteLine("Service Impl A is doing something");
    }
}


public class ServiceImplB : IService
{
  private ILogger _logger;
    public ServiceImplB(ILogger logger)
    {
        _logger = logger;
        Console.WriteLine("ServiceImplB constructor called");
    }
    public void DoSomething()
    {
        Console.WriteLine("Service Impl B is doing something");
    }
}


public class Client
{
    private IService _service;
    public Client(IService service)
    {
      _service = service;
      Console.WriteLine("Client constructor called");
    }
    public void DoAction()
    {
        _service.DoSomething();
    }
}


public class Example
{
  public static void Main(string[] args)
    {
      var container = new UnityContainer();
      container.RegisterType<ILogger, ConsoleLogger>();
      container.RegisterType<IService, ServiceImplA>("A");
      container.RegisterType<IService, ServiceImplB>("B");
      container.RegisterType<Client>();

      var client = container.Resolve<Client>();
      client.DoAction();
      Console.ReadLine();

    }
}
```

Here, we register two implementations of `IService`. Unity must pick one of these when resolving the dependencies of `Client`. Unity will try to match the constructor arguments with the registered types to work out what to inject. This is where things can get interesting and may be the cause of what you have been observing.

While you might not see multiple *instances* of `ServiceImplA` or `ServiceImplB`, Unity will internally evaluate multiple constructor calls based on what's defined when looking at the registered dependencies, causing constructor execution multiple times during its internal resolution phase. Again, the constructor execution is a temporary side-effect of resolving the dependency.

To really understand this better, I would recommend checking out the *Dependency Injection in .NET* by Mark Seemann and Steven van Deursen, as it goes deep into the concepts behind container resolution. The *Microsoft Patterns & Practices: Unity Application Block* documentation, though slightly older, offers more insight into the Unity-specific implementation. And for a deeper dive into C# internals and reflection, which Unity heavily leverages, *C# in Depth* by Jon Skeet is a goldmine.

In summary, what you're witnessing isn't necessarily a problem of multiple objects being created, but rather multiple constructor executions as Unity explores the dependency graph during type resolution. This is especially true with multiple registrations, generic types, or complex object graphs. Understanding this resolution behavior is paramount for effective use of Unity as a dependency injection container.
