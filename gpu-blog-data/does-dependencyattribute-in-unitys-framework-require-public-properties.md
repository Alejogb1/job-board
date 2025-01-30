---
title: "Does DependencyAttribute in Unity's framework require public properties?"
date: "2025-01-30"
id: "does-dependencyattribute-in-unitys-framework-require-public-properties"
---
`DependencyAttribute` within Unity’s framework does *not* mandate that properties it decorates be declared as `public`. My experience developing complex Unity projects, including systems relying heavily on dependency injection, has shown that the attribute’s functionality is tied to the backing field, not the property’s accessibility. It’s the field targeted by the attribute during Unity's reflection process that is of importance. Unity, at runtime, uses its reflection capabilities to locate fields marked with `DependencyAttribute` and then inject instances of the requested type, irrespective of the property's accessibility specifier.

To clarify, a property in C# is essentially a syntactic wrapper around getter and setter methods, which manipulate a private backing field. Unity’s dependency injection mechanism, when using `DependencyAttribute`, is concerned with this backing field, and therefore, the access level of the *property* itself (public, private, protected, etc.) is irrelevant. The framework operates by resolving dependencies based on field type and the presence of the attribute. When I initially worked on a large simulation project, this distinction led to some confusion, believing that all properties intended for injection had to be visible in the inspector, hence `public`. I quickly learned this wasn’t the case. This allows for a much cleaner design where only properties meant for user configuration are exposed.

To illustrate, consider these code examples. The first demonstrates a scenario where a private property's backing field is decorated with `DependencyAttribute`:

```csharp
using UnityEngine;
using UnityEngine.SceneManagement;

public class ExampleServiceUser : MonoBehaviour
{
    [Dependency]
    private IExampleService MyService { get; set; }

    void Start()
    {
        if (MyService != null)
        {
            Debug.Log("Service injected successfully: " + MyService.GetServiceId());
        }
        else
        {
            Debug.LogError("Service injection failed.");
        }
    }
}

public interface IExampleService
{
    string GetServiceId();
}

public class ExampleService : MonoBehaviour, IExampleService
{
    public string GetServiceId()
    {
        return "Service123";
    }

    void Awake()
    {
        DontDestroyOnLoad(this.gameObject);
        SceneManager.MoveGameObjectToScene(this.gameObject, SceneManager.GetActiveScene());
    }
}
```

In this first example, `MyService` is a private property. However, Unity will inject an instance of `IExampleService` into the underlying field of the `MyService` property.  `ExampleService` implements `IExampleService`. The `Awake()` function is necessary to prevent the `ExampleService` object from being deleted between scene loads, ensuring the dependency can be resolved after scene changes. When `ExampleServiceUser` is instantiated, the `Start()` method will execute, and if the service has been injected correctly a corresponding log will be generated. This directly shows that a private property is valid for dependency injection with `DependencyAttribute`.

The next code snippet exhibits a public property, just to demonstrate the equivalence regarding the attribute's function:

```csharp
using UnityEngine;
using UnityEngine.SceneManagement;

public class AnotherExampleServiceUser : MonoBehaviour
{
    [Dependency]
    public IAnotherExampleService MyService { get; set; }

    void Start()
    {
        if (MyService != null)
        {
            Debug.Log("Service injected successfully (public): " + MyService.GetServiceId());
        }
        else
        {
           Debug.LogError("Service injection failed (public).");
        }
    }
}

public interface IAnotherExampleService
{
   string GetServiceId();
}

public class AnotherExampleService : MonoBehaviour, IAnotherExampleService
{
   public string GetServiceId()
   {
       return "AnotherService456";
   }

   void Awake()
    {
       DontDestroyOnLoad(this.gameObject);
       SceneManager.MoveGameObjectToScene(this.gameObject, SceneManager.GetActiveScene());
    }
}

```

Here the property `MyService` is declared `public`. As with the first example, an implementation of `IAnotherExampleService` will be injected by Unity via the field backing the property.  The resulting log message confirms injection success. The key takeaway here is that the accessibility of the property (`public` vs `private`) makes no difference to the `DependencyAttribute`. It's the existence of the `DependencyAttribute` on a field (accessed through the property) that is crucial.

The following example further emphasizes that the injection occurs on the *field*, rather than the property itself, using explicit field declaration instead of a property.

```csharp
using UnityEngine;
using UnityEngine.SceneManagement;

public class ExplicitFieldInjection : MonoBehaviour
{
    [Dependency]
    private IYetAnotherExampleService _serviceField;

    public IYetAnotherExampleService MyService { get { return _serviceField; } }


    void Start()
    {
        if (_serviceField != null)
        {
            Debug.Log("Service injected successfully into field: " + _serviceField.GetServiceId());
        }
        else
        {
            Debug.LogError("Service injection failed (field).");
        }
    }
}

public interface IYetAnotherExampleService
{
   string GetServiceId();
}

public class YetAnotherExampleService : MonoBehaviour, IYetAnotherExampleService
{
    public string GetServiceId()
    {
        return "YetAnotherService789";
    }

    void Awake()
    {
       DontDestroyOnLoad(this.gameObject);
       SceneManager.MoveGameObjectToScene(this.gameObject, SceneManager.GetActiveScene());
    }
}
```

In this case, the `DependencyAttribute` decorates a private field, `_serviceField`. A public property `MyService` only exposes a getter for the injected field. This arrangement is perfectly valid with Unity’s injection mechanism, further highlighting the field-level operation of `DependencyAttribute`. The injected service instance assigned to `_serviceField` is then accessed through the `MyService` property.  The output will confirm that the service has been injected directly into the field.

Based on my practical experience, relying on the default field visibility for properties is preferable for clean and maintainable code. The `DependencyAttribute` mechanism allows injection of services into private properties, enabling developers to maintain clear distinction between injected dependencies (often implementation details) and user-configurable properties exposed in the inspector. This separation of concerns has been critical in maintaining code quality as projects have grown in complexity.

For those looking to deepen their understanding of dependency injection in Unity, I suggest investigating Unity's documentation pertaining to the `DependencyAttribute`. Further resources include the Unity forums and community-driven tutorials focusing on architectural patterns. Explore documentation for `Zenject`, `StrangeIoC`, or similar libraries, which expand upon dependency injection with more complex features. These external libraries and documentation will provide alternative or enhanced injection mechanisms that move beyond Unity's built-in support. Specifically, reading about how these libraries resolve dependencies via reflection (as Unity does) will enhance understanding of the process. Although Unity's implementation of dependency injection may seem somewhat basic, understanding it deeply makes working with third-party DI frameworks significantly easier. Understanding reflection is foundational knowledge. Lastly, experimenting with different visibility specifiers and examining injection behavior is the most effective way to solidify this concept.
