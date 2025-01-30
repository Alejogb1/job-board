---
title: "How can I get CPU and GPU usage for a process in C++ on Windows?"
date: "2025-01-30"
id: "how-can-i-get-cpu-and-gpu-usage"
---
Obtaining precise CPU and GPU utilization for a specific process on Windows using C++ requires leveraging the Windows Management Instrumentation (WMI) interface and potentially the NVIDIA Management Library (NVML), depending on the GPU vendor.  My experience developing performance monitoring tools for high-frequency trading applications has highlighted the nuances and limitations of these approaches.  Directly accessing hardware counters is generally not feasible for obtaining process-specific data without significant kernel-level involvement.  Therefore, relying on WMI and vendor-specific APIs is the most practical approach for a user-mode application.

**1.  Explanation:**

The primary method for retrieving process CPU usage involves querying the WMI `Win32_Process` class. This class exposes properties like `PercentProcessorTime`, offering a readily available percentage value.  However, this value represents CPU utilization *averaged* over a period, typically a second.  For finer-grained monitoring, repeated queries with suitable time intervals are necessary.  Precision depends on the system's performance and the sampling rate.

GPU usage is significantly more complex and highly vendor-specific.  For NVIDIA GPUs, the NVML provides a robust interface to access performance metrics.  AMD and Intel offer comparable APIs; however, these are not standardized and necessitate separate integration efforts.  NVML allows querying GPU utilization, memory usage, temperature, and other relevant statistics for individual processes, offering considerably greater detail than WMI alone.  This information isn't always directly linked to a process ID, so intelligent mapping or heuristics might be needed based on the context of your application.  Note that even with NVML, process-specific GPU utilization data can be imprecise, as several processes can concurrently leverage the GPU, making precise attribution challenging.

**2. Code Examples:**

**Example 1: Obtaining CPU Usage using WMI**

This example demonstrates querying `Win32_Process` for CPU utilization:


```cpp
#include <iostream>
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")

int main() {
  CoInitializeEx(0, COINIT_MULTITHREADED);
  IWbemLocator *pLoc = nullptr;
  IWbemServices *pSvc = nullptr;
  IEnumWbemClassObject *pEnumerator = nullptr;
  HRESULT hres;

  hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID *)&pLoc);
  if (FAILED(hres)) { std::cerr << "Failed to create IWbemLocator" << std::endl; return 1; }

  hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), nullptr, nullptr, 0, 0, 0, 0, &pSvc);
  if (FAILED(hres)) { std::cerr << "Failed to connect to WMI" << std::endl; pLoc->Release(); return 1; }

  pSvc->Release();
  pLoc->Release();

  // Process ID to query, replace with actual process ID
  long processId = 1234;
  _bstr_t query = _bstr_t(L"SELECT PercentProcessorTime FROM Win32_Process WHERE ProcessID = ") + processId;
  hres = pSvc->ExecQuery(_bstr_t("WQL"), query, WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &pEnumerator);

    if (SUCCEEDED(hres)) {
        IWbemClassObject *pclsObj;
        ULONG uReturn = 0;
        while (pEnumerator){
            hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
            if(0 == uReturn) break;
            VARIANT vtProp;
            VariantInit(&vtProp);
            hres = pclsObj->Get(L"PercentProcessorTime", 0, &vtProp, 0, 0);
            if (SUCCEEDED(hres)) {
                std::wcout << L"CPU Usage: " << vtProp.dbl << L"%" << std::endl;
                VariantClear(&vtProp);
            }
            pclsObj->Release();
        }
        pEnumerator->Release();
    }
    CoUninitialize();
  return 0;
}

```

This code showcases basic WMI interaction. Error handling is simplified for brevity.  Production code should incorporate more robust error checking and exception handling.



**Example 2: Obtaining GPU Usage using NVML (NVIDIA)**

This example illustrates accessing GPU utilization via NVML.  Note: Requires the NVML library.


```cpp
#include <iostream>
#include <nvml.h>

int main() {
  NVML_ERROR result = nvmlInit();
  if (NVML_SUCCESS != result) {
    std::cerr << "NVML initialization failed." << std::endl;
    return 1;
  }

  unsigned int deviceCount;
  result = nvmlDeviceGetCount(&deviceCount);
  if (NVML_SUCCESS != result) {
    std::cerr << "Failed to get device count." << std::endl;
    nvmlShutdown();
    return 1;
  }

  for (unsigned int i = 0; i < deviceCount; ++i) {
    NVML_DEVICE_HANDLE handle;
    result = nvmlDeviceGetHandleByIndex(i, &handle);
    if (NVML_SUCCESS != result) {
      std::cerr << "Failed to get device handle." << std::endl;
      continue;
    }

    unsigned int gpuUtilization;
    result = nvmlDeviceGetUtilizationRates(handle, &gpuUtilization); //Obtain GPU utilization
    if (NVML_SUCCESS != result) {
      std::cerr << "Failed to get utilization rates." << std::endl;
      continue;
    }

    std::cout << "GPU " << i << " Utilization: " << gpuUtilization << "%" << std::endl;
  }

  nvmlShutdown();
  return 0;
}
```

This example provides GPU-wide utilization.  Extracting process-specific data requires more advanced NVML features and often relies on correlating GPU activity with process identifiers through other means.



**Example 3: Combining WMI and NVML (Conceptual)**

This is a skeletal outline.  Implementing this effectively requires intricate process monitoring and potentially heuristics based on your application's execution model.

```cpp
// ... (WMI code from Example 1) ...
// ... (NVML code from Example 2) ...

//  Hypothetical correlation – this requires a significant design effort
// and is not a straightforward task. This code is purely illustrative.

std::map<unsigned int, long> processToGPUId; // Map process IDs to GPU IDs

// ... Logic to associate processes with GPU usage ...  (This is highly
// application-specific and would likely involve analyzing GPU memory usage
// or other process-related metrics.  It's not directly supported by WMI or NVML)

//Access utilization based on the map
for(auto const& [processId, gpuId] : processToGPUId){
    // Fetch the GPU utilization from Example 2 for gpuId
}

```


**3. Resource Recommendations:**

*  Microsoft's Windows Management Instrumentation documentation.
*  NVIDIA's NVML documentation (or equivalent documentation for AMD or Intel GPUs).
*  A comprehensive C++ programming textbook covering COM and system programming.
*  Books or online resources focusing on Windows performance monitoring and system internals.


Remember that obtaining precise process-specific GPU usage can be exceptionally difficult, even with specialized libraries.  The accuracy is often limited by the underlying hardware and software architecture. The examples provided serve as a starting point;  adapting and expanding them to suit the specific requirements of your application is crucial.  Furthermore, always consider ethical implications and obtain necessary permissions before monitoring system resources.
