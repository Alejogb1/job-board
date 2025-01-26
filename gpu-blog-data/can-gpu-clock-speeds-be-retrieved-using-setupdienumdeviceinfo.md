---
title: "Can GPU clock speeds be retrieved using SetupDiEnumDeviceInfo?"
date: "2025-01-26"
id: "can-gpu-clock-speeds-be-retrieved-using-setupdienumdeviceinfo"
---

Within the Windows operating system, the Device Setup API (SetupDi), particularly the `SetupDiEnumDeviceInfo` function, provides a mechanism to enumerate devices installed on a system. However, it does not expose a direct method to obtain GPU clock speeds. My experience in low-level driver interaction and hardware monitoring confirms this; while `SetupDiEnumDeviceInfo` is powerful for discovering device information like vendor ID, product ID, and driver details, it operates at an abstraction layer too high to access real-time clock frequencies of specific components like GPUs. The data provided through `SetupDiEnumDeviceInfo` is largely static, reflecting device properties as stored in the registry or driver INF files, rather than dynamic operational characteristics.

The core issue lies in the API’s purpose. `SetupDiEnumDeviceInfo` is designed for device enumeration, identification, and installation management. It is not geared toward performance monitoring or real-time hardware status reporting. Clock speeds are runtime parameters dependent on several factors including power management profiles, load, and thermal conditions. These values are maintained and exposed within the GPU’s driver itself, or through specific performance monitoring interfaces.

Instead of relying on `SetupDiEnumDeviceInfo`, one must leverage alternative APIs that are explicitly built for performance monitoring and hardware telemetry. These APIs are typically device-specific or driver-specific, meaning that their use often requires knowledge of the specific GPU vendor and potentially its associated SDK. The common methods involve querying the GPU’s driver directly via proprietary interfaces or utilizing Windows Performance Counters, which often indirectly access performance data managed by the underlying driver.

To clarify, `SetupDiEnumDeviceInfo` produces device information structures, typically containing GUIDs representing the device’s class, vendor IDs, and instance paths. The information is textual and static, not numerical and dynamic. We can obtain basic hardware description, but not real-time values. Let’s demonstrate this with code.

**Code Example 1: Enumerating Display Adapters with `SetupDiEnumDeviceInfo`**

```c++
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <iostream>
#include <vector>

int main() {
  HDEVINFO hDevInfo = SetupDiGetClassDevs(&GUID_DEVCLASS_DISPLAY, NULL, NULL, DIGCF_PRESENT);
  if (hDevInfo == INVALID_HANDLE_VALUE) {
      std::cerr << "Error getting device information set." << std::endl;
      return 1;
  }

  SP_DEVINFO_DATA deviceInfoData;
  deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
  DWORD deviceIndex = 0;

  while (SetupDiEnumDeviceInfo(hDevInfo, deviceIndex, &deviceInfoData)) {
    deviceIndex++;

    char deviceInstanceId[MAX_DEVICE_ID_LEN];
    if (SetupDiGetDeviceInstanceId(hDevInfo, &deviceInfoData, deviceInstanceId, MAX_DEVICE_ID_LEN, NULL)) {
        std::cout << "Found Display Adapter: " << deviceInstanceId << std::endl;

        // Attempt to get the friendly name
        char friendlyName[256];
        if (SetupDiGetDeviceRegistryPropertyA(hDevInfo, &deviceInfoData, SPDRP_FRIENDLYNAME, NULL, (BYTE*)friendlyName, sizeof(friendlyName), NULL)) {
            std::cout << "  Friendly Name: " << friendlyName << std::endl;
        }
        else
        {
             std::cerr << "  Failed to get friendly name" << std::endl;
        }
    }
    else {
        std::cerr << "  Error getting device instance id." << std::endl;
    }
  }
  
  if(GetLastError() != ERROR_NO_MORE_ITEMS) {
        std::cerr << "Error enumerating devices: " << GetLastError() << std::endl;
    }


  SetupDiDestroyDeviceInfoList(hDevInfo);
  return 0;
}
```
This code snippet iterates through display adapters, retrieving their instance IDs and friendly names. This is the typical information readily available through `SetupDiEnumDeviceInfo`. Note that the code does not return the clock speed, nor is there a `SPDRP_CLOCKSPEED` property available. The `SPDRP` constants available in the `setupapi.h` header do not include real time parameters of the hardware. The relevant properties deal with device characteristics not operational parameters.

**Code Example 2: Illustrating the limitation, showing properties retrieved**

```c++
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <iostream>
#include <vector>

void printProperty(HDEVINFO hDevInfo, PSP_DEVINFO_DATA pDeviceInfoData, DWORD property) {
    DWORD propertyDataType = 0;
    BYTE propertyBuffer[2048] = {0};
    DWORD propertyBufferSize = sizeof(propertyBuffer);
    
    if (SetupDiGetDeviceRegistryPropertyA(hDevInfo, pDeviceInfoData, property, &propertyDataType, propertyBuffer, propertyBufferSize, NULL)) {
      
        if (propertyDataType == REG_SZ) {
              std::cout << "  " << property << " : " << (char*)propertyBuffer << std::endl;
        } else {
               std::cout << "  " << property << " (Binary Data): "  << std::endl;
               for (DWORD i = 0; i < propertyBufferSize; ++i) {
                    if(i%16==0) std::cout << "      ";
                    std::cout << std::hex << (int)propertyBuffer[i] << " ";
                    if((i+1)%16==0) std::cout << std::endl;
               }
               std::cout << std::endl;

        }
    } else {
       std::cerr << "  Error getting property " << property << std::endl;
    }
}

int main() {
  HDEVINFO hDevInfo = SetupDiGetClassDevs(&GUID_DEVCLASS_DISPLAY, NULL, NULL, DIGCF_PRESENT);
  if (hDevInfo == INVALID_HANDLE_VALUE) {
      std::cerr << "Error getting device information set." << std::endl;
      return 1;
  }

  SP_DEVINFO_DATA deviceInfoData;
  deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
  DWORD deviceIndex = 0;

  while (SetupDiEnumDeviceInfo(hDevInfo, deviceIndex, &deviceInfoData)) {
    deviceIndex++;

    char deviceInstanceId[MAX_DEVICE_ID_LEN];
    if (SetupDiGetDeviceInstanceId(hDevInfo, &deviceInfoData, deviceInstanceId, MAX_DEVICE_ID_LEN, NULL)) {
        std::cout << "Found Display Adapter: " << deviceInstanceId << std::endl;

        printProperty(hDevInfo, &deviceInfoData, SPDRP_DEVICEDESC);
        printProperty(hDevInfo, &deviceInfoData, SPDRP_HARDWAREID);
        printProperty(hDevInfo, &deviceInfoData, SPDRP_FRIENDLYNAME);
        printProperty(hDevInfo, &deviceInfoData, SPDRP_MFG);
        printProperty(hDevInfo, &deviceInfoData, SPDRP_CLASS);
        printProperty(hDevInfo, &deviceInfoData, SPDRP_DRIVER);

    }
    else {
        std::cerr << "  Error getting device instance id." << std::endl;
    }
  }

  if(GetLastError() != ERROR_NO_MORE_ITEMS) {
        std::cerr << "Error enumerating devices: " << GetLastError() << std::endl;
    }

  SetupDiDestroyDeviceInfoList(hDevInfo);
  return 0;
}
```
This example showcases the type of information obtainable via the property mechanism `SetupDiGetDeviceRegistryPropertyA`, demonstrating that none of the exposed values directly correlate to GPU clock frequencies. It displays descriptions, hardware IDs, manufacturer information, etc., but nothing dynamic. The `SPDRP_DRIVER` property returns a registry entry representing the location of the driver files, not its operational state.

**Code Example 3: Conceptual illustration, hypothetical clock property**

```c++
// This is for demonstration only. The following does not work in practice
// SetupDiGetDeviceRegistryPropertyA does not provide such functionality

// #define SPDRP_GPU_CLOCK_SPEED  0x20000000  //Hypothetical constant
// ...

// printProperty(hDevInfo, &deviceInfoData, SPDRP_GPU_CLOCK_SPEED); //This would fail
// ...
```

The hypothetical code block comments out a conceptual attempt to retrieve a non-existent `SPDRP_GPU_CLOCK_SPEED` property. This demonstrates why the question is fundamentally flawed: no such property exists in the `SetupDi` API, reinforcing that the interface is not designed for this type of query. Attempting to use it to retrieve real-time clock speed will fail.

Instead of using SetupDi, one should investigate the following avenues to retrieve GPU clock speeds:

1. **Vendor-Specific SDKs:** Both NVIDIA and AMD offer proprietary SDKs, like NVIDIA’s NVAPI and AMD’s ADL SDK (legacy) or the newer AMD GPU Services API. These SDKs provide access to low-level GPU performance data, including clock speeds, memory usage, and temperatures. The complexity of using these SDKs varies depending on the specific vendor and the desired level of detail.

2. **Windows Performance Counters (PDH):** The Performance Data Helper (PDH) API in Windows can access performance counters exposed by devices. While the default set might not include raw GPU clock speeds, the drivers often publish specific counters related to GPU performance, which might include core clock, memory clock, and more. These counters are typically less granular than those provided directly by SDKs. The counters are usually formatted strings.

3.  **DirectX APIs:** Certain DirectX APIs offer access to performance data, although they are typically oriented towards rendering performance metrics. However, some of them might expose information that indirectly hints towards GPU clock speed. However these are not explicitly clock speed values.
4. **Third-Party Libraries:** Several third-party libraries offer a cross-vendor approach to GPU performance monitoring. These libraries often abstract away vendor-specific details, providing a more consistent interface, but might not give access to all possible data points.

In summary, while `SetupDiEnumDeviceInfo` is a valuable tool for identifying and managing devices within a Windows system, it is unsuitable for retrieving real-time clock speeds of GPU devices. The API operates on a level of abstraction that does not provide direct access to these dynamic operational characteristics. The solution involves employing vendor-specific SDKs, leveraging Windows Performance Counters, or using other performance-oriented libraries or APIs designed for hardware monitoring.
