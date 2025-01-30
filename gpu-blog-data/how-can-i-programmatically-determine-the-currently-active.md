---
title: "How can I programmatically determine the currently active GPU in C/C++?"
date: "2025-01-30"
id: "how-can-i-programmatically-determine-the-currently-active"
---
The heterogeneous compute landscape often requires explicit knowledge of which Graphics Processing Unit (GPU) is currently active within an application, especially when aiming for targeted performance optimization or resource management. Determining this programmatically in C/C++ necessitates leveraging platform-specific APIs, as no single cross-platform standard directly exposes this information. My experience with game engine development and high-performance computing has demonstrated that a deep dive into operating system-specific capabilities is the most effective approach.

The primary challenge arises from the fact that “active” can have varying interpretations. In many scenarios, this refers to the GPU responsible for rendering the primary display. However, a system may possess multiple discrete GPUs, and specific workloads might be assigned to non-primary devices. Hence, the code must gracefully handle these cases and return appropriate information based on the application’s needs. We’ll explore solutions for both Windows and Linux, the two most common operating systems for development, while acknowledging that embedded systems or less common OSes require tailored implementations.

**Windows (DirectX)**

On Windows, the DirectX Graphics Infrastructure (DXGI) provides a mechanism to enumerate available adapters and identify the primary adapter, which typically corresponds to the GPU driving the display output. DXGI’s interfaces offer direct access to adapter-specific details like vendor IDs, device IDs, and descriptions, which may be helpful when selecting specific GPUs beyond the primary one. The key is to use the `IDXGIFactory` interface to create an enumerator, then iterate over the available adapters. The adapter with the `LUID` (Locally Unique Identifier) that matches the primary output is usually what most consider the “active” GPU.

Here is an example outlining how to identify this primary GPU:

```c++
#include <dxgi.h>
#include <d3d11.h>
#include <iostream>
#include <vector>
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

struct GPUInfo {
    std::wstring description;
    int vendorID;
    int deviceID;
    LUID adapterLUID;
};

std::optional<GPUInfo> getActiveGPUInfo() {
  IDXGIFactory* factory = nullptr;
  HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
  if (FAILED(hr)) {
      return std::nullopt;
  }

  IDXGIAdapter* adapter = nullptr;
  std::vector<GPUInfo> availableAdapters;
  for(UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i){
    DXGI_ADAPTER_DESC desc;
    adapter->GetDesc(&desc);
    GPUInfo info;
    info.description = desc.Description;
    info.vendorID = desc.VendorId;
    info.deviceID = desc.DeviceId;
    info.adapterLUID = desc.AdapterLuid;
    availableAdapters.push_back(info);
    adapter->Release();
    adapter = nullptr;
  }

  IDXGIOutput* output = nullptr;
  IDXGIAdapter* primaryAdapter = nullptr;
  for (UINT i = 0; factory->EnumOutputs(i, &output) != DXGI_ERROR_NOT_FOUND; i++) {
      if (output != nullptr) {
          DXGI_OUTPUT_DESC outdesc;
          output->GetDesc(&outdesc);
          if (outdesc.AttachedToDesktop) {
             hr = output->GetAdapter(&primaryAdapter);
             break;
           }
           output->Release();
           output = nullptr;
        }
  }


    if (primaryAdapter != nullptr) {
        DXGI_ADAPTER_DESC desc;
        primaryAdapter->GetDesc(&desc);
        for (const auto& info : availableAdapters) {
            if(info.adapterLUID.HighPart == desc.AdapterLuid.HighPart &&
                info.adapterLUID.LowPart == desc.AdapterLuid.LowPart) {
                primaryAdapter->Release();
                factory->Release();
                if(output) output->Release();
                return info;
           }
        }
    }

  if(output) output->Release();
  if(primaryAdapter) primaryAdapter->Release();
  factory->Release();
  return std::nullopt;

}

int main() {
    auto gpuInfo = getActiveGPUInfo();
    if (gpuInfo.has_value()) {
        std::wcout << L"Active GPU: " << gpuInfo->description << std::endl;
        std::cout << "Vendor ID: " << std::hex << gpuInfo->vendorID << std::endl;
        std::cout << "Device ID: " << std::hex << gpuInfo->deviceID << std::endl;
    } else {
        std::cerr << "Failed to retrieve active GPU info." << std::endl;
    }
    return 0;
}
```

*   The code begins by creating a DXGI factory.
*   It then iterates through all available adapters, storing their descriptions, vendor IDs, device IDs and LUIDs.
*   Next, the code iterates through all outputs and check which is attached to desktop and gets the adapter associated with that output.
*   After obtaining the adapter associated with the primary output it compares the LUID of that adapter with the LUIDs of the previously enumerated adapters to find the one matching the primary output and return its information.
*   Finally, the main function calls this function and print the information on the console.
*   Error handling is present by checking the result of API calls, returning `std::nullopt` if any operation fails.
*   This solution requires that the system support DXGI and that the graphics driver is correctly installed.
*   Memory is managed explicitly and all allocated COM objects are released.

**Linux (libdrm & X11/Wayland)**

On Linux, the situation is more fragmented due to the diverse display server options (X11, Wayland). The Direct Rendering Manager (DRM), accessed via `libdrm`, provides a low-level interface to interact directly with graphics hardware. However, directly determining the active GPU based solely on `libdrm` is challenging, as it primarily deals with device file access and not display system topology. Additional information from the windowing system is often necessary.

The following code shows how to retrieve all available GPU from the drm subsystem:

```c++
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <string>
#include <cstring>
#include <optional>

struct GPUInfo {
    std::string name;
    uint32_t vendorID;
    uint32_t deviceID;
    std::string path;
};

std::vector<GPUInfo> getAvailableGPUs() {
    std::vector<GPUInfo> gpus;
    drmDevicePtr *devices;
    int count = drmGetDevices2(0, &devices);
    if (count < 0) {
        return gpus;
    }

    for (int i = 0; i < count; ++i) {
        if (devices[i]->available == 1 && (devices[i]->bustype == DRM_BUS_PCI || devices[i]->bustype == DRM_BUS_HOST1X) ) {
             GPUInfo gpuInfo;
             gpuInfo.path = devices[i]->nodes[DRM_NODE_RENDER];

            int fd = open(gpuInfo.path.c_str(), O_RDWR);
            if (fd < 0) {
                continue;
            }
            drmVersionPtr version = drmGetVersion(fd);
            if(version) {
                 gpuInfo.name = version->name;
                 drmFreeVersion(version);
            }

            drm_pci_info pci_info;
            if(drmGetPCIBusInfo(devices[i], &pci_info) == 0) {
                gpuInfo.vendorID = pci_info.vendor_id;
                gpuInfo.deviceID = pci_info.device_id;
            }
            close(fd);
           gpus.push_back(gpuInfo);
         }
    }

    drmFreeDevices(devices, count);
    return gpus;
}

int main() {
    auto gpus = getAvailableGPUs();
    if (gpus.empty()) {
        std::cerr << "No GPUs found." << std::endl;
        return 1;
    }
    for (const auto& gpu : gpus) {
        std::cout << "GPU Name: " << gpu.name << std::endl;
        std::cout << "Vendor ID: " << std::hex << gpu.vendorID << std::endl;
        std::cout << "Device ID: " << std::hex << gpu.deviceID << std::endl;
        std::cout << "DRM Path: " << gpu.path << std::endl;
    }

    return 0;
}
```

* This code utilizes `libdrm` to access available GPUs.
*   `drmGetDevices2` obtains an array of available devices.
*   It filters out non-available devices and devices with incorrect bus types and stores the path, name, vendorID and deviceID into the GPUInfo structure
*   The devices list is then freed after use.
*   The program prints the information for each device found.
*   This code is fairly straightforward, and does not need any extra libraries. It only needs `libdrm` installed on the system.

The above example does not provide which of these GPUs is the active one. This is where a windowing system integration can provide more context. For instance, using X11 with `libXrandr` allows querying the output configurations and the graphics device associated with each output. For Wayland, equivalent operations can be performed using Wayland’s API, which tends to vary between compositor implementations. Integrating this with the above logic would yield a more complete solution.

Here is an example for X11. It is very verbose and for sake of space, it only outputs the first available output information.

```c++
#include <iostream>
#include <vector>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include <string>
#include <optional>
#include <algorithm>


std::optional<std::string> getActiveGPUPath(const std::vector<GPUInfo>& availableGPUs) {
  Display *display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Could not open X display." << std::endl;
        return std::nullopt;
    }

    Window root = DefaultRootWindow(display);
    XRRScreenResources *screenResources = XRRGetScreenResources(display, root);

    if (!screenResources) {
        XCloseDisplay(display);
        std::cerr << "Could not get screen resources." << std::endl;
        return std::nullopt;
    }

  if(screenResources->noutput <= 0) {
    XRRFreeScreenResources(screenResources);
    XCloseDisplay(display);
    std::cerr << "No outputs found" << std::endl;
    return std::nullopt;
  }

  std::optional<std::string> activeGpuPath;
  for(int i=0; i < screenResources->noutput; ++i) {
    XRROutputInfo *outputInfo = XRRGetOutputInfo(display, screenResources, screenResources->outputs[i]);
    if(!outputInfo){
      continue;
    }
    if(outputInfo->connection == RR_Connected) {
      if(outputInfo->crtc) {
        XRRCrtcInfo* crtcInfo = XRRGetCrtcInfo(display, screenResources, outputInfo->crtc);
        if(crtcInfo){
          for(const auto& gpu : availableGPUs){
            if(crtcInfo->framebuffer == 0 && strstr(crtcInfo->outputs[0]->name, gpu.path.c_str())){
              activeGpuPath = gpu.path;
              XRRFreeCrtcInfo(crtcInfo);
              break;
            } else if(std::find(crtcInfo->outputs, crtcInfo->outputs + crtcInfo->noutput , screenResources->outputs[i]) != crtcInfo->outputs + crtcInfo->noutput ) {
              activeGpuPath = gpu.path;
              XRRFreeCrtcInfo(crtcInfo);
              break;
            }
          }
          if(activeGpuPath) break;
        }
      }
    }
      XRRFreeOutputInfo(outputInfo);
  }

    XRRFreeScreenResources(screenResources);
    XCloseDisplay(display);
   return activeGpuPath;
}
int main() {
  auto gpus = getAvailableGPUs();

  auto activePath = getActiveGPUPath(gpus);

  if(activePath) {
    for(const auto& gpu : gpus) {
      if(gpu.path == *activePath) {
        std::cout << "Active GPU Name: " << gpu.name << std::endl;
         std::cout << "Vendor ID: " << std::hex << gpu.vendorID << std::endl;
        std::cout << "Device ID: " << std::hex << gpu.deviceID << std::endl;
        std::cout << "DRM Path: " << gpu.path << std::endl;
        return 0;
      }
    }
  }

  std::cerr << "Failed to get active GPU Path." << std::endl;
    return 1;

}
```

*   This code opens an X display connection using `XOpenDisplay`.
*   It retrieves the screen resources.
*   The code loops through each output and its associated information. If the output is connected, and its crtc is defined it retrieves more info on the crtc.
*  The code then uses the `libdrm` implementation to cross reference the output with the DRM path.
*   After finding the output that corresponds with a drm device it outputs the corresponding device information.
*   Finally, it releases all allocated memory and close the display connection.
*   This code assumes the X server is available and running. A similar implementation could be made using Wayland APIs.
*   It only returns one of the GPUs since, generally, only one GPU is active on the primary display.

**Resource Recommendations**

For comprehensive understanding, the following resources are highly recommended:

*   **DirectX Documentation:** The Microsoft DirectX documentation provides in-depth information on DXGI and related APIs, crucial for Windows development.
*   **libdrm Documentation:** The kernel's `libdrm` documentation details the low-level access to GPU devices on Linux.
*   **Xlib and libXrandr Documentation:** Relevant for X11 environments, these libraries handle display management.
*   **Wayland Protocol Documentation:** Explore specific compositor documentation for Wayland-based solutions.
*   **Operating System Specific Kernel Documentation:** For advanced cases, the kernel documentation offers information on hardware interaction, especially for device enumeration.

Successfully identifying the active GPU in C/C++ involves navigating platform-specific nuances. The provided examples demonstrate fundamental techniques, but comprehensive solutions might demand additional considerations based on the specific needs and context of the application.
