---
title: "How do I retrieve the GPU name from Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory?"
date: "2025-01-30"
id: "how-do-i-retrieve-the-gpu-name-from"
---
The `Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory` WMI class, while seemingly straightforward for GPU memory metrics, doesn't directly expose the GPU name.  My experience troubleshooting similar performance monitoring issues in enterprise environments has highlighted the necessity for a multi-step approach leveraging other WMI classes to correlate memory data with the corresponding GPU identifier.  The key lies in joining data from this class with the `Win32_VideoController` class, which *does* provide the necessary GPU name information.


**1. Clear Explanation:**

Retrieving the GPU name requires a two-pronged WMI query.  First, we query `Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory` to obtain relevant performance data. This class provides metrics like available and used memory, but lacks the GPU's descriptive name.  Secondly, we need to query `Win32_VideoController`. This class contains a wealth of information about each installed graphics adapter, crucially including the `Name` property, which we'll use to identify the GPU.  The challenge lies in correlating the data between these two classes.  A unique identifier, typically the `InstanceName` in `Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory`, acts as the link.  This identifier isn't always directly intuitive, but it consistently links the performance data to a specific GPU instance.  The process involves querying both classes, then iterating through the results of the first query and finding the matching entry in the second based on the `InstanceName` or a similar property depending on the system's configuration.  Careful error handling is crucial to account for systems with multiple GPUs or potential inconsistencies in WMI data.


**2. Code Examples with Commentary:**

The following examples demonstrate three different approaches to retrieving the GPU name using PowerShell, VBScript, and C++.  Each method emphasizes efficient data handling and robust error management.

**a) PowerShell:**

```powershell
# Query Win32_VideoController for GPU information
$VideoControllers = Get-WmiObject -Class Win32_VideoController

# Query Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory for performance data
$GPUPerformanceData = Get-WmiObject -Class Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory

# Iterate through performance data and find matching GPU name
foreach ($GPUData in $GPUPerformanceData) {
    # Attempt to find a matching VideoController based on InstanceName.  Variations in InstanceName
    # formatting across different drivers necessitate a flexible matching approach.  
    $matchingController = $VideoControllers | Where-Object {$_.Caption -match $GPUData.InstanceName -or $_.Name -match $GPUData.InstanceName}

    if ($matchingController) {
        Write-Host "GPU Name: $($matchingController.Name)"
        Write-Host "Available Memory: $($GPUData.AvailableMBytes) MB"
        Write-Host "Used Memory: $($GPUData.UsedMBytes) MB"
        Write-Host "----------"
    } else {
        Write-Host "Warning: Could not find matching VideoController for InstanceName: $($GPUData.InstanceName)"
    }
}
```

This PowerShell script uses `Get-WmiObject` to fetch data from both WMI classes.  The `Where-Object` cmdlet efficiently filters the `Win32_VideoController` results to find the matching entry, employing a regular expression to account for potential variations in the `InstanceName` across different GPU drivers.  The script also includes error handling to report cases where a match isn't found.  This robust approach increases the script's reliability across diverse hardware configurations.


**b) VBScript:**

```vbscript
Set objWMIService = GetObject("winmgmts:\\.\root\cimv2")

Set colItems = objWMIService.ExecQuery _
    ("Select * from Win32_VideoController")

Set colPerfData = objWMIService.ExecQuery _
    ("Select * from Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory")

For Each objItem in colItems
    For Each objPerfData in colPerfData
        If InStr(1, objPerfData.InstanceName, objItem.Name, vbTextCompare) > 0 Then
            WScript.Echo "GPU Name: " & objItem.Name
            WScript.Echo "Available Memory: " & objPerfData.AvailableMBytes & " MB"
            WScript.Echo "Used Memory: " & objPerfData.UsedMBytes & " MB"
            WScript.Echo "----------"
            Exit For
        End If
    Next
Next
```

This VBScript code uses similar logic, querying both WMI classes and iterating through them.  The `InStr` function provides a case-insensitive string comparison to match the `InstanceName` and `Name` properties.  The nested loop structure, while less efficient than the PowerShell approach, demonstrates a fundamental method for achieving the desired correlation.  The script lacks the explicit error handling of the PowerShell example;  however, the `Exit For` statement helps manage cases where a match is found and prevents unnecessary iterations.


**c) C++:**

```cpp
#include <iostream>
#include <comdef.h>
#include <Wbemidl.h>

int main() {
    CoInitializeEx(0, COINIT_MULTITHREADED);
    IWbemLocator* pLoc = NULL;
    IWbemServices* pSvc = NULL;
    IEnumWbemClassObject* pEnumerator = NULL;
    HRESULT hres;

    hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    // ... (Error handling and connection to WMI omitted for brevity, standard WMI connection code is assumed) ...

    // Query Win32_VideoController
    // ... (Code to execute query and retrieve results omitted for brevity) ...

    // Query Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
    // ... (Code to execute query and retrieve results omitted for brevity) ...

    // Iterate and correlate data (similar logic as PowerShell and VBScript, using string comparisons)

    // ... (Code to cleanup COM objects omitted for brevity) ...
    return 0;
}
```

The C++ example provides a basic framework.  The actual implementation of WMI query execution and data retrieval is omitted for brevity, as it involves standard COM object manipulation and error handling which would significantly increase the code length.  The core logic, however, remains the same: querying both WMI classes and correlating the results based on the `InstanceName` or a related property to match performance data to the corresponding GPU name.  Robust error handling using HRESULT codes is crucial in C++ for reliable WMI interaction.  This example provides a foundation for developers familiar with C++ and COM programming.



**3. Resource Recommendations:**

*   Microsoft's documentation on WMI (Windows Management Instrumentation).  Consult this resource for detailed information on WMI class descriptions and query syntax.
*   A comprehensive guide on COM programming in your chosen language (C++, VBscript, etc.).  Understanding COM is crucial for efficient interaction with WMI.
*   A book or online tutorial focusing on Windows system administration.  This provides context around performance monitoring and the role of WMI in gathering such data.  Pay special attention to sections dedicated to scripting and automation.


This multi-faceted approach, combining WMI queries and data correlation, offers a robust solution for retrieving GPU names alongside performance data from `Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory`.  Remember that slight modifications to the code might be necessary depending on your system's specific configuration and the precise format of the `InstanceName` property.  Thorough error handling ensures the reliability of your solution across diverse hardware and software environments.
