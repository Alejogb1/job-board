---
title: "How can .NET JIT compilation be optimized for fastest startup?"
date: "2025-01-30"
id: "how-can-net-jit-compilation-be-optimized-for"
---
The single largest contributor to .NET application startup latency is the Just-In-Time (JIT) compilation process. Reducing this overhead, particularly for cold starts, is paramount for achieving responsive user experiences. I’ve spent several years optimizing high-throughput .NET microservices where milliseconds of improvement at startup translate to significant resource savings. Based on that experience, several techniques exist, ranging from ahead-of-time compilation options to judicious code layout strategies.

The JIT compiler translates Common Intermediate Language (CIL) into native machine code on demand as methods are invoked for the first time. This “on-demand” approach enables platform independence, allowing the same CIL to run on various architectures; however, this translation incurs a startup cost. To minimize this, one must understand how the JIT compiler operates and employ techniques to reduce its workload.

The first significant optimization involves employing ReadyToRun (R2R) compilation. Instead of generating native code during runtime, R2R precompiles assemblies during the build process. This eliminates the majority of JIT work during startup. R2R is not a complete replacement for JIT; rather, it minimizes the initial JIT overhead. Specifically, it creates a 'ready-to-run' image containing pre-compiled code for the most commonly used methods. The JIT is still used to compile other rarely used methods at runtime. However, as the code is loaded directly from disk as native image, a lot of time is saved. The trade-off for R2R is larger disk footprint; the native code is embedded inside the assemblies. R2R can be enabled using the .NET CLI. In the application's project file (.csproj), add `<PublishReadyToRun>true</PublishReadyToRun>`. In conjunction with the `PublishSingleFile` option, the entire application, along with the runtime, can be packaged into one executable. This further improves startup performance by eliminating the need to load a multitude of DLLs.

Another significant area of optimization lies in understanding method inlining, a JIT optimization where the code of a called method is inserted directly into the calling method. This reduces overhead associated with function calls and improves execution speed. However, the JIT is not always able to inline methods, especially large ones. Thus, strategically designing smaller methods that are likely to be inlined is another key aspect of optimization. The JIT is also influenced by the code layout in the DLLs. The more code is accessed, the more potential for increased JIT times if those methods are not placed together. By placing frequently used code together in a locality, the system can load only the necessary pages into memory during start-up, speeding things up. Consider a common initialization pattern that might benefit from this strategy:

```csharp
public class StartupService
{
    private readonly IConfiguration _configuration;

    public StartupService(IConfiguration configuration)
    {
       _configuration = configuration;
       Initialize();
    }

    private void Initialize()
    {
       LoadConfigurationValues();
       RegisterDependencies();
    }

    private void LoadConfigurationValues()
    {
       // configuration loading logic...
       var host = _configuration.GetValue<string>("Host");
       var port = _configuration.GetValue<int>("Port");
    }

    private void RegisterDependencies()
    {
       // Dependency registration logic...
       var services = new ServiceCollection();
       services.AddScoped<ISomeService, SomeService>();
       // ...
       var provider = services.BuildServiceProvider();
       // ...
    }
}

```
In the above example, `StartupService` uses several methods at the application start. This method and the called methods would be candidates to optimize. By structuring the code as such, R2R might be able to pre-compile the code, reducing overhead. In a non-trivial application, several methods are executed during startup. Ensuring that these are laid out well can reduce the startup time significantly. In some frameworks like ASP.NET Core, these can be identified by looking at the request pipeline, or startup services.

Another critical aspect often overlooked is avoiding unnecessary type initialization during startup. .NET type initialization is a one-time operation that runs when a type is first accessed. This initialization can sometimes include complex operations such as loading resource files or initializing static variables. Deferred type initialization is often the best solution to mitigate performance problems caused by static constructors during startup. This means only accessing static members or invoking static methods when they are actually needed, rather than at app startup, unless strictly required. In complex systems with many services, the usage of reflection or dependency injection may sometimes trigger unwanted static initializers during start up. One must be careful to only load the absolute required initializers. Consider this scenario:

```csharp
public static class ServiceHelper
{
    private static readonly Lazy<ILogger> _logger = new(()=>
    {
        var factory = LoggerFactory.Create(builder => {
                builder.AddConsole();
         });
        return factory.CreateLogger("ServiceHelper");
    });

    public static void LogInfo(string message)
    {
        _logger.Value.LogInformation(message);
    }

    static ServiceHelper()
    {
        //Initialization of static properties that might not be used
        string  staticValue = "default_value"; //some static initializations.
        Console.WriteLine(staticValue);
    }
}
```

In this case the static constructor is initialized during application start, even if the `ServiceHelper.LogInfo` method is never called during startup process. While in this example, the initialization cost is not much, such a pattern on multiple types could easily add up to a significant load on application start. In this case, using lazy initialization and not initializing values in the static constructor (unless necessary) is a better practice. The static constructor was changed to the following:

```csharp
public static class ServiceHelper
{
   private static readonly Lazy<ILogger> _logger = new(()=>
   {
       var factory = LoggerFactory.Create(builder => {
           builder.AddConsole();
       });
       return factory.CreateLogger("ServiceHelper");
   });

   public static void LogInfo(string message)
   {
       _logger.Value.LogInformation(message);
   }
}
```

As can be seen, the static initializer has been removed entirely. In scenarios where statics have to be initialized, it should be done with lazy loading.

The third technique involves optimizing the layout of DLLs on disk. The order of types in assembly files can affect the JIT's work and memory loading. By using a tool such as `ngen.exe`, available on Windows, the .NET assemblies can be pre-compiled in such a way that frequently used code is laid out in proximity on disk, thereby reducing disk seek time and memory page loads during startup. While the concept of reordering types is similar to R2R, these techniques can be combined. One can use R2R to prepare an assembly and then use ngen.exe to reorder it on the disk. Here is a very simple example:

```csharp
using System;

namespace SimpleApp
{
    public class ServiceA
    {
        public void DoSomething()
        {
            Console.WriteLine("Service A did something");
        }
    }

    public class ServiceB
    {
        public void DoAnotherThing()
        {
            Console.WriteLine("Service B did another thing");
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            ServiceA serviceA = new ServiceA();
            serviceA.DoSomething();
            // ServiceB is instantiated but not called
        }
    }
}
```

In the given example, `ServiceA` is invoked, and `ServiceB` is instantiated but not called during startup. By strategically placing `ServiceA` close to the program's entry point in the assembly, we can reduce startup costs by loading only the relevant code segments during the initial run. While we cannot directly control this through the .NET compiler, we can use `ngen.exe` to reorder the code in the assembly on disk. After compiling the application, the `ngen install` command can be used to precompile the assembly and possibly reorder it. The command would be `ngen install <assembly_path>`. Note that you need to run the ngen tool as an administrator. The tool compiles and reorders code based on previous usage patterns and optimizes it for faster loading. The ngen tool must be ran after the application has been deployed.
It is important to note that using the ngen tool can make the assembly dependent on the computer where it was compiled.

To further refine this, profiling tools that show JIT activity can be essential. .NET performance tracing tools such as PerfView can identify methods that cause high JIT overhead during startup. With this information, one can prioritize optimization efforts on the most costly areas. It is important to note that a combination of the above techniques would usually yield the best results; using only one in isolation might not be very impactful. In addition to the approaches outlined above, it is important to mention other methods, such as using tiered compilation. Tiered compilation is enabled by default in recent .NET versions. It allows the compiler to first compile code with an optimizing JIT and then later re-optimize it.

In conclusion, optimizing .NET JIT compilation for faster startup involves employing various techniques, from precompilation using R2R, carefully structuring the layout of methods, to deferring static initializations, and finally, possibly reorganizing disk layout. Profiling is key to identifying the most time-consuming code sections. This understanding, coupled with disciplined coding, yields tangible performance improvements in .NET applications.
For further in-depth information, the .NET documentation offers extensive coverage of R2R compilation, JIT internals and performance profiling techniques. The official .NET performance blog also provides detailed analysis and real-world case studies.
