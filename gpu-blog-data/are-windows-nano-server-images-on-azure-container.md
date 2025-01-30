---
title: "Are Windows Nano Server images on Azure Container Instances logging-capable?"
date: "2025-01-30"
id: "are-windows-nano-server-images-on-azure-container"
---
Windows Nano Server, while designed for minimal resource consumption, presents specific challenges regarding logging within Azure Container Instances (ACI). I've personally encountered these limitations while deploying several microservices using this architecture. Primarily, the standard system event logs and application logs typically found in full Windows Server deployments are absent, necessitating alternative logging strategies when using Nano Server in ACI.

The key fact is that Nano Server's reduced footprint inherently omits many conventional logging mechanisms. Standard tools like Event Viewer and the associated APIs are not available. Consequently, traditional methods of writing log data to the Windows Event Log or relying on system-level logging aren't directly applicable within Nano Server containers in ACI. Instead, logging from your applications within Nano Server images running on ACI requires either forwarding logs to an external service or writing to standard output (stdout) and standard error (stderr), which ACI can then capture.

To elaborate, unlike full Windows Server images, Nano Server is designed for core functionality. This minimalist philosophy eliminates elements deemed unnecessary, including a robust suite of logging utilities. While this approach contributes to faster startup times and reduced resource consumption, it forces a change in how logging is approached. The absence of traditional logging structures means developers need to explicitly configure application logging to function within this environment. Therefore, relying on established practices becomes problematic and necessitates more manual configuration.

My experience has involved several scenarios that highlighted the need for this modified approach. For instance, consider a simple .NET application. In a standard Windows environment, logging might typically leverage libraries that write directly to the Event Log. This approach is simply not viable when running a compiled .NET application on a Nano Server image inside an ACI. The absence of the necessary APIs means the application will encounter errors or the log writes will be ignored.

The primary strategy, then, becomes redirecting log information to stdout and stderr. ACI automatically captures these streams, and you can subsequently use Azure Monitor to analyze the logs. Alternatively, you may elect to use an external logging platform, requiring more complex configuration of the container. Here are three illustrative examples of how one would handle logging, moving from an unsuitable traditional method to viable solutions.

**Code Example 1: Inappropriate Traditional Logging (Illustrative of what *not* to do)**

```csharp
// Incorrect example. Will not work on Nano Server inside ACI

using System;
using System.Diagnostics;

public class BadLogger
{
    public static void LogError(string message)
    {
        string sourceName = "MyApp";

        if (!EventLog.SourceExists(sourceName))
        {
            EventLog.CreateEventSource(sourceName, "Application");
        }

        EventLog myLog = new EventLog();
        myLog.Source = sourceName;
        myLog.WriteEntry(message, EventLogEntryType.Error);
    }

    public static void Main(string[] args)
    {
        LogError("This error will not be logged to Windows Event Log inside Nano Server/ACI");
    }
}
```

*Commentary:* This first code snippet attempts to use the standard .NET `System.Diagnostics.EventLog` API, a common approach for logging in Windows applications. This approach is demonstrably incorrect in the Nano Server context. The necessary services for interacting with the event log are simply missing from Nano Server. This is here to illustrate the challenge and a common pitfall when porting applications to this type of environment. When run within a Nano Server image on ACI, the code will likely throw an exception or silently fail to log due to the absence of the required Event Log service. This reinforces the critical understanding that conventional Windows logging tools do not readily translate to a Nano Server environment.

**Code Example 2: Basic stdout logging**

```csharp
// Correct example: Basic logging to stdout.

using System;
using System.Threading;

public class GoodLogger
{
    public static void LogInfo(string message)
    {
        Console.WriteLine($"[INFO] {DateTime.Now}: {message}");
    }

      public static void LogError(string message)
    {
        Console.Error.WriteLine($"[ERROR] {DateTime.Now}: {message}");
    }
    public static void Main(string[] args)
    {
       while (true) {
          LogInfo("Application is running, sending info message");
          LogError("Something went wrong, sending error message");
          Thread.Sleep(5000); // Add a delay to avoid excessive logs.
       }
    }
}
```

*Commentary:* This second example demonstrates the more appropriate approach for logging within a Nano Server container hosted on ACI. The application writes informational messages to `Console.WriteLine` and error messages to `Console.Error.WriteLine`. These two methods redirect the output to standard output and standard error streams, respectively. ACI captures and exposes these streams through Azure Monitor logs. In essence, instead of relying on system services that are missing on Nano Server, we are utilizing the core standard I/O streams. This approach is fundamental to working with Nano Server in ACI and ensures that log information is effectively captured. The timestamp added provides a basic level of log clarity. While basic, this approach is generally applicable and highly portable. The infinite loop with a delay is just for illustrative purposes here, as the sample is intended to provide examples of logs being written.

**Code Example 3: Using Serilog with stdout**

```csharp
//Correct Example: Using Serilog to stdout

using System;
using Serilog;
using Serilog.Formatting.Compact;


public class SerilogLogger
{
    public static void Main(string[] args)
    {
         Log.Logger = new LoggerConfiguration()
        .WriteTo.Console(new CompactJsonFormatter())
        .CreateLogger();

        try {
            Log.Information("Application started");
            throw new Exception("Something went wrong");

        }
        catch (Exception ex)
        {
            Log.Error(ex, "An exception occurred");
        }
        finally {
            Log.Information("Application finished");
            Log.CloseAndFlush();
        }
    }
}
```
*Commentary:* This final example shows a more structured approach using Serilog, a popular .NET logging library.  Here, Serilog is configured to write to the console using the `CompactJsonFormatter`, resulting in JSON-formatted logs written to standard output, which can then be efficiently parsed and processed. This is a more sophisticated approach to stdout logging by utilizing a library that allows for structured logging while still maintaining compatibility with the minimal environment of Nano Server. The use of JSON can make consuming the output into different logging aggregation services simpler. Error logging includes exception details enhancing analysis. Using a well-established library also provides a robust and flexible framework to expand logging functionality when requirements evolve, for instance, writing to different sinks as needed or including more log attributes. This example illustrates that external libraries can be leveraged while still working within the confines of Nano Server in ACI by relying on stdout and stderr.

Regarding resource recommendations, I would suggest focusing on resources that delve into the specifics of container logging on Azure. Microsoft's official documentation for Azure Container Instances and Azure Monitor logs are essential. Furthermore, any articles or tutorials focusing on logging from .NET Core (or whichever language you are using) to standard output are extremely valuable. When using third-party libraries, such as Serilog, review their documentation closely, paying special attention to the available sinks, specifically those capable of writing to stdout or stderr. For any specific questions when using Azure Monitor, their official documentation can provide relevant guidance. Finally, exploring how different tools can aggregate and ingest the captured stdout/stderr logs is vital for monitoring your application effectively.
