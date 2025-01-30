---
title: "How can constructor arguments be added to Ninject bindings?"
date: "2025-01-30"
id: "how-can-constructor-arguments-be-added-to-ninject"
---
Ninject's flexibility extends to enabling constructor argument injection during bindings, allowing for a more nuanced and configurable object graph than simple type-to-type mappings. This functionality is essential when a concrete type's constructor requires specific values beyond what's already registered in the kernel or available through auto-binding. Over my time developing modular applications with Ninject, I’ve relied heavily on this capability to create decoupled and testable components.

The core mechanism revolves around the `WithConstructorArgument` method, available within the fluent binding syntax. This method allows you to specify a value, a binding to another service, or a function that resolves the argument during object creation. The kernel uses these specified arguments when fulfilling a request for a bound type. It is important to understand that these are not default arguments; they supersede any auto-resolution attempts by Ninject. Furthermore, the injection mechanism evaluates all constructor arguments specified in the binding before default value resolution, therefore when constructor arguments are explicitly specified, they are always favored over default parameter values in the class constructor.

Specifically, when a constructor has multiple overloads, and some overload has less parameters than another, the constructor with the most matching arguments from the explicit binding arguments, or auto-bound ones, will be called. If multiple constructors have equal matching arguments, the constructor with less arguments will be called first.

One crucial consideration is the timing of argument evaluation; the configured argument is evaluated when Ninject attempts to create an instance of the bound type, not at the time the binding is declared. This is important in situations where the value of an argument is subject to change.

Here's how you would leverage `WithConstructorArgument` in various scenarios:

**Scenario 1: Providing a Literal Value**

Consider a `Logger` class which takes a string representing the logger's name in its constructor. Without providing a value, Ninject would not know how to resolve the dependency. Instead of hardcoding the loggers name inside the `Logger` class constructor, we can inject the parameter from the binding itself.

```csharp
public class Logger
{
    public string LoggerName { get; }

    public Logger(string loggerName)
    {
        LoggerName = loggerName;
    }
}

public interface ILoggerConsumer {
    ILogger Logger { get; }
}

public class LoggerConsumer : ILoggerConsumer {
	public ILogger Logger {get;}

	public LoggerConsumer(ILogger logger) {
		Logger = logger;
	}
}


public class ExampleModule : NinjectModule
{
    public override void Load()
    {
        Bind<ILogger>().To<Logger>().WithConstructorArgument("loggerName", "ApplicationLogger");
		Bind<ILoggerConsumer>().To<LoggerConsumer>();
    }
}
```

In this example, `WithConstructorArgument("loggerName", "ApplicationLogger")`  specifies that the `loggerName` parameter of the `Logger` constructor should be injected with the literal string "ApplicationLogger". When `Kernel.Get<ILogger>()` is called a new instance of `Logger` is created where its `LoggerName` property has a value equal to `"ApplicationLogger"`. Furthermore, the `LoggerConsumer` when requested from the `Kernel` will be instantiated with a logger whose `LoggerName` will be set to `"ApplicationLogger"`. This is effective for settings that are relatively static.

**Scenario 2: Injecting a Service**

Suppose you have a `DatabaseConnection` class that requires an `IConnectionStringProvider` in its constructor and you need to instantiate this connection at runtime.

```csharp
public interface IConnectionStringProvider
{
    string GetConnectionString();
}

public class DefaultConnectionStringProvider : IConnectionStringProvider
{
    public string GetConnectionString()
    {
		return "DefaultConnectionString";
    }
}


public class DatabaseConnection
{
    public string ConnectionString {get;}

    public DatabaseConnection(IConnectionStringProvider connectionStringProvider)
    {
		ConnectionString = connectionStringProvider.GetConnectionString();
    }
}

public interface IConnectionConsumer
{
    DatabaseConnection Connection {get;}
}

public class ConnectionConsumer: IConnectionConsumer {

	public DatabaseConnection Connection {get;}
	public ConnectionConsumer(DatabaseConnection connection) {
		Connection = connection;
	}
}

public class ExampleModule : NinjectModule
{
    public override void Load()
    {
        Bind<IConnectionStringProvider>().To<DefaultConnectionStringProvider>();
        Bind<DatabaseConnection>().ToSelf().WithConstructorArgument("connectionStringProvider", c => c.Kernel.Get<IConnectionStringProvider>());
		Bind<IConnectionConsumer>().To<ConnectionConsumer>();
    }
}
```

Here, `WithConstructorArgument("connectionStringProvider", c => c.Kernel.Get<IConnectionStringProvider>())` utilizes a lambda to instruct Ninject to resolve the argument by querying the kernel for `IConnectionStringProvider` when a `DatabaseConnection` instance is needed. This allows for dynamic connections through the use of an external data source. The constructor parameter `connectionStringProvider` is injected with an instance of `DefaultConnectionStringProvider`. When `Kernel.Get<IConnectionConsumer>()` is called a `DatabaseConnection` instance is created and passed as the constructor argument where its `ConnectionString` property will be equal to "DefaultConnectionString".

**Scenario 3: Using a Factory Function**

In some cases, creating the argument might require a more complex computation or the value depends on a custom logic. For this we need to use a factory pattern. Let's say we have a `ReportGenerator` class that takes a Func<string> that returns the reports source path. This path depends on various configurations in the application.

```csharp
public class ReportGenerator
{
	public string ReportSourcePath {get;}
    public ReportGenerator(Func<string> reportSourcePathFunc)
    {
        ReportSourcePath = reportSourcePathFunc();
    }
}

public interface IReportGeneratorConsumer {
	ReportGenerator Generator { get;}
}

public class ReportGeneratorConsumer : IReportGeneratorConsumer {

	public ReportGenerator Generator {get;}
	public ReportGeneratorConsumer(ReportGenerator generator) {
		Generator = generator;
	}
}

public class ExampleModule : NinjectModule
{
    public override void Load()
    {
        Bind<ReportGenerator>().ToSelf().WithConstructorArgument("reportSourcePathFunc", c => (Func<string>)(() => GetReportSourcePath(c.Kernel)));
		Bind<IReportGeneratorConsumer>().To<ReportGeneratorConsumer>();
    }

	private string GetReportSourcePath(IKernel kernel) {
		// Logic that can use kernel and other configurations to resolve the desired path
		return "ExampleReport.pdf";
	}
}
```

In this scenario, the lambda within `WithConstructorArgument` provides a custom function to be executed when a `ReportGenerator` instance is requested. This approach makes it possible to inject arguments that require some form of runtime calculation. In this case a call to the `GetReportSourcePath` is made. When a `ReportGeneratorConsumer` instance is requested through `Kernel.Get<IReportGeneratorConsumer>()`, a new `ReportGenerator` instance will be created and its `ReportSourcePath` property will be equal to `"ExampleReport.pdf"`. This strategy is very powerful when you need to inject configurations that may be loaded from other services or configurations at runtime.

Through the examples provided, it should be clear that  `WithConstructorArgument` is a robust tool, enabling both simple value injection and complex service resolution. The choice between a literal value, service binding, or a factory function depends on the specific needs of the component and the overall design of your application. The flexibility of lambda expression allows injection of parameters dynamically with the use of other services available in the kernel.

For deeper understanding of Ninject and its binding system, I recommend referring to the official documentation available on the Ninject website. Moreover, studying examples of dependency injection practices, especially in modular application design, will provide practical context. Furthermore, various online forums dedicated to C#/.NET development often contain discussions and advanced techniques related to Ninject. Lastly, exploring examples on Github for libraries that have implemented Ninject for different scenarios will give deeper context on how Ninject can be applied. These resources have been invaluable to my own understanding and implementation of Ninject’s features.
