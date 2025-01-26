---
title: "How can a single method be profiled in VSTS?"
date: "2025-01-26"
id: "how-can-a-single-method-be-profiled-in-vsts"
---

Single method profiling within Visual Studio Team Services (VSTS), now Azure DevOps, requires a nuanced understanding of both the platform’s capabilities and the underlying profiling tools. Unlike full application profiling, which captures a wide range of performance metrics, isolating a single method demands targeted instrumentation and data collection. My experience has shown that this process is best achieved by combining Azure DevOps's build pipelines with command-line profiling tools, specifically, the Visual Studio Performance Profiler, invoked through its command-line interface, `VSPerfCmd.exe`. The challenge lies in automating the invocation of the profiler, filtering results, and then integrating relevant information back into the Azure DevOps ecosystem.

The key to this method profiling hinges on understanding how the Visual Studio Performance Profiler operates. It leverages instrumentation techniques, either by inserting probes into the compiled code (instrumentation profiling) or by sampling the call stack at regular intervals (sampling profiling). For pinpointing performance issues within a specific method, instrumentation profiling generally provides more granular insights as it tracks entry and exit points, leading to more precise timing data. Sampling profiling can be useful for understanding the overall application behavior around the targeted method, but less accurate for isolating a single, brief execution. Therefore, my approach focuses on instrumentation.

Here’s how I’ve successfully achieved single method profiling within an Azure DevOps pipeline:

First, I established a dedicated build configuration within my Visual Studio project, specifically tailored for profiling. This configuration contains debug symbols, is not optimized, and sets the compiler flag `/PROFILE`, enabling instrumentation. It’s essential to realize that optimized builds distort the execution flow, rendering accurate instrumentation profiling impossible. I then designed a simple test application or test method specifically designed to call the method being profiled under controlled conditions. This is crucial for generating meaningful data. The test driver’s role is to exercise the targeted method multiple times, to get a reasonable average of timings. The entire process is then formalized as an Azure DevOps build pipeline.

The build pipeline, instead of a direct compilation, first compiles the project using this custom "Profiling" configuration. Following the build, the pipeline then executes the profiling steps using a PowerShell task. The core of this script lies in the usage of `VSPerfCmd.exe`. I've found the command-line interface provides more control for targeted profiling than the visual interface.

Here’s an annotated PowerShell snippet showcasing how I invoke the profiler and process the resulting data:

```powershell
# Define paths and variables.
$VSPerfCmdPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools\VSPerfCmd.exe"
$ProfileOutput = "profile_output.vsp"
$TestExecutable = ".\TestApp.exe"
$MethodName = "MyMethodToProfile"
$DurationSeconds = 5 # Profiling duration in seconds

# Start the profiler with instrumentation for a specific executable.
Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/start:trace /output:$ProfileOutput" -Wait

# Execute the test application which will execute the method.
Start-Process -FilePath $TestExecutable -Wait

# Stop the profiler.
Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/stop /output:$ProfileOutput" -Wait

# Collect the performance data from vsp file (not shown in this snippet).
# Additional analysis is required to extract the specific method timings.
# ...
# The vsp file can be analyzed with VS or vsperfreport.exe.

Write-Host "Profiling completed."
```

This snippet defines the essential paths and sets variables for the profiling process. It starts the profiler in trace mode, executes the test application (which will in turn execute the `MyMethodToProfile`), and then stops the profiler. Critically, it’s the `Start-Process` calls to the Visual Studio Performance Profiler tool (`VSPerfCmd.exe`) that enable the data acquisition. The `vsp` file, the profiler's output, contains all the collected instrumentation data and is essential for performance analysis. Note that analysis of the `vsp` file to extract specific timing from the method isn't displayed in this sample. The `vsp` is a proprietary format, and requires further analysis either within Visual Studio or using the command-line `vsperfreport.exe`.

Here's an example illustrating how a more complex scenario, specifically filtering specific module’s symbols before analysis, can be implemented. It also assumes the use of `vsperfreport.exe` to parse the `vsp` file:

```powershell
# Define paths and variables.
$VSPerfCmdPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools\VSPerfCmd.exe"
$VSPerfReportPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools\vsperfreport.exe"
$ProfileOutput = "profile_output.vsp"
$TestExecutable = ".\TestApp.exe"
$MethodName = "MyMethodToProfile"
$ModuleName = "MyModule.dll"
$ReportOutput = "report.txt"
$DurationSeconds = 5 # Profiling duration in seconds

# Start the profiler with instrumentation for a specific executable.
Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/start:trace /output:$ProfileOutput" -Wait

# Execute the test application which will execute the method.
Start-Process -FilePath $TestExecutable -Wait

# Stop the profiler.
Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/stop /output:$ProfileOutput" -Wait

# Generate a performance report, filtered by specific module and aggregated at method level.
Start-Process -FilePath $VSPerfReportPath -ArgumentList "/summary:Methods /module:$ModuleName /input:$ProfileOutput /output:$ReportOutput" -Wait

# Output the resulting report to the console to view the method timings.
Get-Content $ReportOutput

Write-Host "Profiling and analysis completed."
```

In this version, I've added the path to `vsperfreport.exe`. Critically, the `vsperfreport.exe` call uses `/module:$ModuleName` to filter the report, ensuring that we only see data related to `MyModule.dll`. This makes the analysis more manageable when profiling complex application. Moreover, the `/summary:Methods` option generates a method-level aggregated summary, making it easier to pinpoint the execution time of `$MethodName` within that module.

A further refined example incorporates a basic check to verify whether the profiler was successfully started:

```powershell
# Define paths and variables.
$VSPerfCmdPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\Performance Tools\VSPerfCmd.exe"
$VSPerfReportPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Team Tools\vsperfreport.exe"
$ProfileOutput = "profile_output.vsp"
$TestExecutable = ".\TestApp.exe"
$MethodName = "MyMethodToProfile"
$ModuleName = "MyModule.dll"
$ReportOutput = "report.txt"
$DurationSeconds = 5

# Start the profiler with instrumentation for a specific executable, capturing exit code
$ProfilerStartProcess = Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/start:trace /output:$ProfileOutput" -Wait -PassThru
$ProfilerStartExitCode = $ProfilerStartProcess.ExitCode

if ($ProfilerStartExitCode -ne 0) {
    Write-Error "Failed to start the profiler. Exit code: $ProfilerStartExitCode"
    return # Exit script if profiler failed to start.
}


# Execute the test application which will execute the method.
Start-Process -FilePath $TestExecutable -Wait


# Stop the profiler.
$ProfilerStopProcess = Start-Process -FilePath $VSPerfCmdPath -ArgumentList "/stop /output:$ProfileOutput" -Wait -PassThru
$ProfilerStopExitCode = $ProfilerStopProcess.ExitCode

if ($ProfilerStopExitCode -ne 0) {
     Write-Error "Failed to stop the profiler. Exit code: $ProfilerStopExitCode"
     return # Exit script if profiler failed to stop
}

# Generate a performance report, filtered by specific module and aggregated at method level.
Start-Process -FilePath $VSPerfReportPath -ArgumentList "/summary:Methods /module:$ModuleName /input:$ProfileOutput /output:$ReportOutput" -Wait

# Output the resulting report to the console to view the method timings.
Get-Content $ReportOutput

Write-Host "Profiling and analysis completed."
```

This final example adds error checking, ensuring the script exits if the profiler fails to start or stop. This makes the script more robust in a pipeline setting.  Each `Start-Process` call is modified to use the `-PassThru` parameter which enables capturing the exit code of the called process.  This exit code is then checked, and if it’s non-zero, an error is output, and the script terminates to prevent subsequent errors.

Following data acquisition, I typically integrated the textual report generated by `vsperfreport.exe` into the Azure DevOps pipeline build summary, either by writing the report's content directly to the build output or publishing it as a build artifact. This approach enables me to track the performance of a method over time and identify potential regressions.

For learning and deeper understanding, I would recommend exploring the documentation related to the Visual Studio Performance Profiler, both the UI and the command-line versions. The official Microsoft documentation offers extensive information on the various profiling modes and their respective settings.  Additionally, I recommend exploring general resources on Windows performance analysis and instrumentation to deepen the theoretical basis for profiling. Finally, experimenting with simple applications using various compiler settings, and observing how these settings influence the generated profile, is invaluable.
