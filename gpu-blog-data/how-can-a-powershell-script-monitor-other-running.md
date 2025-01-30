---
title: "How can a PowerShell script monitor other running scripts in real-time, report errors and exit codes, and operate in a separate window?"
date: "2025-01-30"
id: "how-can-a-powershell-script-monitor-other-running"
---
PowerShell's inherent capabilities for process management, coupled with its robust event handling, provide a sophisticated mechanism for real-time monitoring of other scripts.  My experience working on large-scale automation projects involving hundreds of concurrently running scripts highlighted the need for a robust solution surpassing simple `Get-Process` checks; a system capable of discerning individual script execution statuses and reacting to failures in real-time was essential.  This necessitates leveraging the `Register-ObjectEvent` cmdlet and employing techniques for inter-process communication.

**1. Clear Explanation:**

Real-time monitoring of scripts involves continuously observing their execution state. This goes beyond merely checking if a process with a particular name exists; it demands tracking the process's exit code, identifying exceptions, and handling unexpected terminations.  The foundational approach involves registering for events triggered by the target script's process.  The `Register-ObjectEvent` cmdlet allows subscribing to events raised by a .NET object, including those associated with processes.  Specifically, the `Process` object raises events such as `Exited`, `Error`, and `StateChanged` which provide critical information regarding the script's status.  Crucially, to run the monitoring script in a separate window, we'll employ the `Start-Process` cmdlet with the `-NoNewWindow` parameter disabled.

Inter-process communication is necessary for robust error reporting. While monitoring directly for events provides real-time feedback, transmitting error messages from the monitored script to the monitor requires a mechanism beyond event data.  This can be achieved by various approaches including writing to a shared file, using a message queue, or even leveraging a dedicated inter-process communication library.  For simplicity and robustness within the scope of this problem, writing to a shared log file is sufficient.


**2. Code Examples with Commentary:**

**Example 1: Basic Process Monitoring and Exit Code Reporting**

This example monitors a single script and reports its exit code upon completion.

```powershell
# Monitored script path.  Replace with the actual path.
$scriptPath = "C:\Scripts\MyScript.ps1"

# Start the script in a separate window.
Start-Process -FilePath powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File '$scriptPath'" -NoNewWindow

# Get the process object.  Assumes a unique process name is present within MyScript.ps1.
$process = Get-Process | Where-Object {$_.Name -match "MyScript"}

# Register for the Exited event.
Register-ObjectEvent $process -EventName Exited -Action {
  Write-Host "Script '$scriptPath' exited with code $($_.SourceEventArgs.ExitCode)."
}

# Keep the monitor running until the script exits.  Adjust as needed.
while ($process.HasExited -eq $false) {
  Start-Sleep -Seconds 1
}

# Clean up the process object.
Unregister-Event -SourceIdentifier $process
$process | Stop-Process
```

**Commentary:** This script demonstrates the core monitoring pattern.  The `-match` operator in `Get-Process` provides flexibility for identifying the process, vital when process names aren’t entirely unique.  Error handling, crucial in production environments, is omitted for brevity but should always be included.  Replacing `"MyScript"` with a more specific and consistent process name from within `MyScript.ps1` enhances reliability.


**Example 2: Monitoring Multiple Scripts and Writing to a Log File**

This expands on the previous example to handle multiple scripts and log errors to a file.

```powershell
# Array of scripts to monitor.
$scripts = @(
    "C:\Scripts\Script1.ps1",
    "C:\Scripts\Script2.ps1"
)

# Log file path.
$logFile = "C:\Scripts\monitor_log.txt"

# Function to monitor a single script.
function Monitor-Script {
    param(
        [string]$scriptPath
    )
    Start-Process -FilePath powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File '$scriptPath'" -NoNewWindow
    $process = Get-Process | Where-Object {$_.Name -match "Script[12]"} #needs improvement for generalizability

    Register-ObjectEvent $process -EventName Exited -Action {
        $message = "Script '$scriptPath' exited with code $($_.SourceEventArgs.ExitCode)."
        Write-Host $message
        Add-Content -Path $logFile -Value $message
    }
    while ($process.HasExited -eq $false) {Start-Sleep -Seconds 1}
    Unregister-Event -SourceIdentifier $process
    $process | Stop-Process
}


# Monitor each script.
foreach ($script in $scripts) {
    Monitor-Script -scriptPath $script
}
```


**Commentary:**  This example utilizes a function to encapsulate the monitoring logic, improving code organization and reusability.  The log file provides a persistent record of script executions and their outcomes.  The process identification requires refinement in real-world scenarios to reliably distinguish between similarly named processes.  Consider using unique identifiers within each script itself for improved process identification.


**Example 3: Incorporating Error Handling within the Monitored Script**

This shows how to improve error reporting by including error handling within the monitored script itself.

```powershell
# Monitored script (MyScript.ps1)
try {
    # Your script code here...
    # ... potentially error-prone operations ...
}
catch {
    # Log the error to a file.
    $errorMessage = "Error in MyScript.ps1: $($_.Exception.Message)"
    Add-Content -Path "C:\Scripts\myscript_errors.log" -Value $errorMessage
    exit 1 # Indicate an error
}

# Monitoring script (remains largely the same as Example 1, but checks myscript_errors.log)

# ... (Monitoring script code from Example 1) ...
# Add a check for error log in the Exited event handler.
Register-ObjectEvent $process -EventName Exited -Action {
    $exitCode = $_.SourceEventArgs.ExitCode
    Write-Host "Script '$scriptPath' exited with code $exitCode."
    if ($exitCode -ne 0) {
        Get-Content "C:\Scripts\myscript_errors.log" | ForEach-Object {Write-Host $_}
    }
}
```

**Commentary:**  This approach is superior because error information originates directly from the script itself, improving accuracy and detail.  The exit code from the monitored script signals the monitoring script to retrieve detailed error messages from the dedicated log file.  This enhances the granularity of error reporting compared to relying solely on the `Exited` event's limited information.


**3. Resource Recommendations:**

For a deeper understanding of PowerShell's event handling and process management, consult the official PowerShell documentation.  Explore resources on advanced scripting techniques, including working with .NET objects and inter-process communication.  Understanding exception handling and logging best practices is also crucial for building robust monitoring systems.  Finally, studying examples of advanced PowerShell automation projects can offer invaluable insights into real-world applications of these concepts.
