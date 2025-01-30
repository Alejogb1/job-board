---
title: "How can I retrieve GPU name information in C?"
date: "2025-01-30"
id: "how-can-i-retrieve-gpu-name-information-in"
---
Retrieving GPU name information programmatically in C necessitates leveraging platform-specific APIs due to the lack of a unified, cross-platform approach within the standard C library. My experience across various rendering and compute projects has consistently shown that the underlying operating system and its graphics driver ecosystem are the primary conduits for this information. The method I've found most reliable involves direct interaction with these lower-level interfaces.

Fundamentally, a successful strategy must diverge based on the target operating system. On Windows, the DirectX API offers a structured and mature approach, while Linux systems frequently rely on querying the hardware through the Vulkan API or the system's file structure. Therefore, the problem decomposes into implementing a conditional compilation strategy, targeting each system's specific interface to obtain the desired GPU name string.

For a Windows environment, the process involves incorporating the DirectX headers, specifically `d3d11.h`, `dxgi.h`, and `dxgiformat.h`. Using the Direct3D 11 API allows us to enumerate adapters and retrieve their descriptions. First, we create a D3D11 device, then iterate through the available adapters using `IDXGIFactory::EnumAdapters`. For each adapter found, `IDXGIAdapter::GetDesc` populates a `DXGI_ADAPTER_DESC` structure, one member of which is `Description`. This member stores the adapter's human-readable name. The code should gracefully handle API calls, and free resources correctly, especially COM objects like devices and adapters.

```c
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <stdio.h> // For printf

char* GetWindowsGpuName() {
    IDXGIFactory* pFactory = NULL;
    IDXGIAdapter* pAdapter = NULL;
    char* gpuName = NULL;
    HRESULT hr;

    // Create DXGI factory.
    hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    if (FAILED(hr)) {
        return NULL;
    }

    // Enumerate adapters, choosing the first non-software adapter
    for (UINT i = 0; ; i++) {
       hr = pFactory->EnumAdapters(i, &pAdapter);
        if(FAILED(hr)) {
            break;
        }

        DXGI_ADAPTER_DESC desc;
        pAdapter->GetDesc(&desc);

        // Check if software renderer
        if (desc.VendorId != 0x1414 && desc.VendorId != 0x8086) {
             size_t stringLength = wcslen(desc.Description);
            gpuName = (char*)malloc(stringLength+1);
            size_t convertedCharacters;
            wcstombs_s(&convertedCharacters, gpuName, stringLength+1, desc.Description, stringLength);

            pAdapter->Release();
            break; // Exit after retrieving one
        }
        pAdapter->Release();

    }

     if(pFactory) pFactory->Release();
    return gpuName;
}

int main() {
     char* gpu_name = GetWindowsGpuName();
     if (gpu_name != NULL)
    {
         printf("GPU Name: %s\n", gpu_name);
        free(gpu_name);
    } else {
        printf("Failed to retrieve GPU name on Windows.\n");
    }

    return 0;
}
```

This code example initially creates a DXGI factory, then iterates through available adapters. The initial adapter that is not from vendor ID 0x1414 or 0x8086(Microsoft Basic Render Driver and Intel integrated graphics) is chosen. The adapter's description is extracted. Error checking is vital throughout, particularly in handling failed API calls.  Note that the retrieved GPU name needs to be converted from wchar_t* to char* for printing and a memory allocation is needed for storing the char* before conversion. This string is then returned and printed. Crucially, I've added explicit memory release calls for the COM objects to prevent resource leaks.

On Linux, the Vulkan API offers a direct way to discover GPU information. While Vulkan is primarily a graphics API, it also provides facilities for querying physical devices, including their names. The procedure involves creating a Vulkan instance, then querying the available physical devices. We can retrieve the `VkPhysicalDeviceProperties` structure, which contains the `deviceName` string.  This avoids the inconsistencies sometimes observed with file-based approaches.

```c
#define VK_USE_PLATFORM_XCB_KHR
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char* GetLinuxGpuName(){

        VkInstance instance;
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPUInfo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;


    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

     VkResult result = vkCreateInstance(&createInfo, NULL, &instance);
        if (result != VK_SUCCESS) {
        return NULL;
    }
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

    if(deviceCount ==0){
       vkDestroyInstance(instance, NULL);
        return NULL;
    }
      VkPhysicalDevice* physicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);
    char* gpuName= NULL;
    for (uint32_t i = 0; i < deviceCount; i++) {
         VkPhysicalDeviceProperties deviceProperties;
         vkGetPhysicalDeviceProperties(physicalDevices[i],&deviceProperties );
         if(strcmp(deviceProperties.deviceName,"llvmpipe") != 0){
             gpuName = strdup(deviceProperties.deviceName);
             break;
         }
    }
     free(physicalDevices);
    vkDestroyInstance(instance, NULL);

     return gpuName;
}
 int main() {
     char* gpu_name = GetLinuxGpuName();
     if (gpu_name != NULL)
    {
         printf("GPU Name: %s\n", gpu_name);
        free(gpu_name);
    } else {
         printf("Failed to retrieve GPU name on Linux.\n");
    }

    return 0;
}

```
This second code example demonstrates the Linux approach using Vulkan. It creates a Vulkan instance, queries available physical devices, and then retrieves the properties of each. Similar to the Windows approach, error checking is crucial. I've added a check to ensure the "llvmpipe" driver is skipped to target actual hardware GPUs. The device name is copied into a new memory location for return.  Importantly, all Vulkan objects are destroyed and allocated memory freed after use.

For macOS, the `Metal` framework is the primary API to obtain the information. While a full implementation is complex, the basic principle remains the same: query the system for GPU devices and retrieve their names. This approach is substantially more involved than the previous examples due to the Objective-C based interface of Metal, requiring an Objective-C++ bridge in a C project, which is beyond the direct scope of a C-only solution. The necessary framework is `Metal.framework`. Briefly, one would use `MTLCreateSystemDefaultDevice()` to get the default GPU device, and its `name` property is a string containing the GPU name.  A C compatible method would involve writing an objective-c++ helper class which could be linked and called from C.

```objectivec++
 //GPUHelper.h
 #import <Foundation/Foundation.h>
 #import <Metal/Metal.h>
 @interface GPUHelper : NSObject
 +(const char *) getGpuName;
 @end

 //GPUHelper.mm
 #import "GPUHelper.h"
 @implementation GPUHelper
 +(const char *) getGpuName {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *deviceName = [device name];
    return [deviceName UTF8String];
}
@end

//main.c
 #include <stdio.h>
 #include <stdlib.h>

 extern const char* getGpuName(void);

 int main(){

    const char* gpuName = getGpuName();

    if(gpuName != NULL){
        printf("GPU Name: %s\n", gpuName);
    } else {
         printf("Failed to retrieve GPU name on macOS\n");
    }
     return 0;
 }
```

This final example highlights how to achieve this on MacOS using an Objective-C++ bridge. The `GPUHelper` class in `GPUHelper.mm` abstracts the underlying Metal API, obtaining the `MTLDevice` and extracting its name which is returned as a C style `char*` through a C compatible function declaration in the header file. The C main function then utilizes this function. Note that this compilation requires the g++ and linking against the `Metal` framework, for example when using gcc with clang as the compiler : `clang main.c GPUHelper.mm -fobjc-arc -framework Foundation -framework Metal -o gpu_info`.  Again, if successful, the device name is printed, else an error message is displayed.

Several resources provide further information on these platform-specific APIs. Microsoft’s documentation on DirectX is comprehensive, detailing device enumeration and adapter descriptions. Khronos Group offers detailed specifications and tutorials on the Vulkan API. Apple's developer documentation provides extensive coverage on the Metal framework. Studying these resources can enable a deeper understanding and more robust error handling than can be easily illustrated here. Finally, examining open-source graphics libraries, like those utilized by emulators or game engines, can provide insight into real-world practices for device enumeration. Always test implementations across varied hardware configurations to ensure consistent behavior.
