---
title: "How can GPU information be extracted using C++ (potentially with WMI)?"
date: "2025-01-30"
id: "how-can-gpu-information-be-extracted-using-c"
---
GPUs, unlike CPUs, typically do not expose their characteristics directly through conventional system calls in a cross-platform manner.  Consequently, obtaining detailed GPU information in C++ often necessitates platform-specific APIs or, in the case of Windows, using Windows Management Instrumentation (WMI). My own experience building a high-performance rendering engine underscored this reality; I had to integrate both DirectX and WMI queries to handle varying hardware configurations gracefully across client machines.

Fundamentally, achieving this falls into two broad categories: leveraging low-level graphics APIs or querying system management interfaces. Low-level graphics APIs like DirectX (on Windows), Vulkan, or OpenGL primarily focus on device interaction for rendering purposes, but they also provide structures containing crucial information such as the vendor, device name, and memory capacity of the GPU. WMI, conversely, offers a more general view of system components, including graphics adapters, but it requires interacting with COM objects and often returns textual descriptions that demand further parsing. The choice of method depends entirely on the level of granularity and the type of information needed. For instance, if you specifically need information on DirectX feature levels, the API directly provides that. If you are creating a system monitoring utility needing details beyond render context compatibility like driver version or chipset series, WMI might be more suitable. The preferred approach is typically direct API interaction whenever the information can be retrieved this way, as it’s generally more performant and less prone to breaking changes with system updates.

Let's first examine how to get GPU information using the DirectX API on Windows. The approach relies on creating a DXGI factory, enumerating the adapters, and extracting information from associated structures. Note that I assume a basic understanding of DirectX initialization is already in place; my focus here is the GPU enumeration.

```cpp
#include <iostream>
#include <vector>
#include <dxgi.h>
#include <d3d11.h> // For D3D11_CREATE_DEVICE_FLAG

struct GPUInfo {
    std::wstring name;
    std::wstring vendor;
    UINT dedicatedMemory;
};


std::vector<GPUInfo> getDirectXGPUs() {
    std::vector<GPUInfo> gpus;

    IDXGIFactory * pFactory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    if (FAILED(hr)) {
        return gpus; // return empty vector
    }

    IDXGIAdapter * pAdapter;
    for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC adapterDesc;
        pAdapter->GetDesc(&adapterDesc);

        GPUInfo info;
        info.name = adapterDesc.Description;
        switch (adapterDesc.VendorId)
        {
            case 0x10DE: info.vendor = L"NVIDIA"; break;
            case 0x1002: info.vendor = L"AMD"; break;
            case 0x8086: info.vendor = L"Intel"; break;
            default: info.vendor = L"Unknown"; break;
        }
        info.dedicatedMemory = static_cast<UINT>(adapterDesc.DedicatedVideoMemory / (1024 * 1024)); // in MB

        gpus.push_back(info);
        pAdapter->Release();
    }

    pFactory->Release();

    return gpus;
}


int main() {
  std::vector<GPUInfo> gpuList = getDirectXGPUs();
    for (const auto& gpu : gpuList) {
      std::wcout << L"Name: " << gpu.name << std::endl;
      std::wcout << L"Vendor: " << gpu.vendor << std::endl;
      std::wcout << L"Dedicated Memory: " << gpu.dedicatedMemory << " MB" << std::endl;
      std::wcout << std::endl;
    }
    return 0;
}

```

In this DirectX example, I first create an `IDXGIFactory`. Then I iterate through each adapter using `EnumAdapters`, obtaining a pointer to an `IDXGIAdapter`. Through this adapter pointer, we get a description with `GetDesc`, which provides a structure containing vital information such as the adapter name, vendor ID, and dedicated video memory.  The vendor ID is translated into a human-readable string via a switch statement.  The memory is converted from bytes into megabytes. Note that more elaborate vendor identification can be implemented with a lookup table. After enumerating each adapter, the factory and adapters are released to prevent memory leaks. If DirectX initialization fails, the function returns an empty vector. The `main` function demonstrates how to invoke this function and output the extracted information.

Next, let's explore using WMI to gather GPU information. This method uses COM objects and requires a different set of headers. This approach offers broader system information but might be slower and involve more parsing compared to the DirectX API approach.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <comdef.h>
#include <Wbemidl.h>

#pragma comment(lib, "wbemuuid.lib")

struct GPUInfoWMI
{
  std::wstring name;
  std::wstring adapterType;
  std::wstring videoProcessor;
  std::wstring driverVersion;
};


std::vector<GPUInfoWMI> getWmiGPUs()
{
    std::vector<GPUInfoWMI> gpuList;
    HRESULT hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres))
    {
        return gpuList;
    }

    hres = CoInitializeSecurity(nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr);
    if (FAILED(hres))
    {
        CoUninitialize();
        return gpuList;
    }

    IWbemLocator* pLoc = nullptr;

    hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres))
    {
        CoUninitialize();
        return gpuList;
    }


    IWbemServices* pSvc = nullptr;

    hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), nullptr, nullptr, nullptr, 0, 0, 0, &pSvc);

    if (FAILED(hres))
    {
        pLoc->Release();
        CoUninitialize();
        return gpuList;
    }


    hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE);
    if (FAILED(hres))
    {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return gpuList;
    }

    IEnumWbemClassObject* pEnumerator = nullptr;

    hres = pSvc->ExecQuery(bstr_t("WQL"), bstr_t("SELECT * FROM Win32_VideoController"), WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &pEnumerator);

    if (FAILED(hres))
    {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return gpuList;
    }


    IWbemClassObject* pclsObj = nullptr;
    ULONG uReturn = 0;

    while (pEnumerator)
    {
        hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn)
        {
            break;
        }

        GPUInfoWMI gpuInfo;

        VARIANT vtProp;

        hres = pclsObj->Get(L"Name", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hres) && vtProp.vt == VT_BSTR) {
            gpuInfo.name = vtProp.bstrVal;
            VariantClear(&vtProp);
        }
        hres = pclsObj->Get(L"AdapterType", 0, &vtProp, 0, 0);
         if (SUCCEEDED(hres) && vtProp.vt == VT_BSTR) {
          gpuInfo.adapterType = vtProp.bstrVal;
          VariantClear(&vtProp);
        }

        hres = pclsObj->Get(L"VideoProcessor", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hres) && vtProp.vt == VT_BSTR) {
          gpuInfo.videoProcessor = vtProp.bstrVal;
          VariantClear(&vtProp);
        }
        hres = pclsObj->Get(L"DriverVersion", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hres) && vtProp.vt == VT_BSTR) {
          gpuInfo.driverVersion = vtProp.bstrVal;
          VariantClear(&vtProp);
        }
        gpuList.push_back(gpuInfo);
        pclsObj->Release();
    }
        pEnumerator->Release();
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();

    return gpuList;
}
int main() {
  std::vector<GPUInfoWMI> gpuList = getWmiGPUs();
    for (const auto& gpu : gpuList) {
      std::wcout << L"Name: " << gpu.name << std::endl;
       std::wcout << L"Adapter Type: " << gpu.adapterType << std::endl;
      std::wcout << L"Video Processor: " << gpu.videoProcessor << std::endl;
      std::wcout << L"Driver Version: " << gpu.driverVersion << std::endl;
      std::wcout << std::endl;
    }
    return 0;
}


```

Here, the WMI query retrieves data from the `Win32_VideoController` class. The initialization sequence involves calls to `CoInitializeEx`, `CoInitializeSecurity`, and creating an `IWbemLocator`. A connection to the WMI service is established via `ConnectServer` then we execute a WQL query to fetch all `Win32_VideoController` instances. A while loop processes the results via `IEnumWbemClassObject::Next`, retrieving each GPU instance as an `IWbemClassObject`. Then, we fetch properties like "Name," "AdapterType", "VideoProcessor", and "DriverVersion" using `IWbemClassObject::Get` and store them in a structure. The critical point is that data from COM objects are retrieved in a `VARIANT` data type, which must be queried to verify that it is of a BSTR type, and then we must extract the data and then release it via `VariantClear`. Again, appropriate resource cleanup is done at the end. The `main` function performs output similar to the DirectX example.

Finally, a combined example will show how to attempt to use the graphics API first, and if that fails, fall back to using WMI. This strategy is advisable when attempting to guarantee a minimum information set across a variety of machines.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <comdef.h>
#include <Wbemidl.h>
#include <dxgi.h>
#include <d3d11.h>

struct GPUInfoHybrid {
    std::wstring name;
    std::wstring vendor;
    UINT dedicatedMemory;
     std::wstring adapterType;
  std::wstring videoProcessor;
    std::wstring driverVersion;
    bool wmiUsed = false;
};

std::vector<GPUInfoHybrid> getHybridGPUInfo()
{
  std::vector<GPUInfoHybrid> hybridGpus;
     std::vector<GPUInfo> dxGpus = getDirectXGPUs();
     if (!dxGpus.empty())
     {
         for (const auto& dxGpu : dxGpus)
         {
             GPUInfoHybrid hybridGpu;
             hybridGpu.name = dxGpu.name;
             hybridGpu.vendor = dxGpu.vendor;
             hybridGpu.dedicatedMemory = dxGpu.dedicatedMemory;
             hybridGpus.push_back(hybridGpu);
         }
          return hybridGpus;
     }


     std::vector<GPUInfoWMI> wmiGpus = getWmiGPUs();
     for (const auto& wmiGpu : wmiGpus)
     {
         GPUInfoHybrid hybridGpu;
         hybridGpu.name = wmiGpu.name;
         hybridGpu.adapterType = wmiGpu.adapterType;
           hybridGpu.videoProcessor = wmiGpu.videoProcessor;
           hybridGpu.driverVersion = wmiGpu.driverVersion;
           hybridGpu.wmiUsed = true;
         hybridGpus.push_back(hybridGpu);
     }
     return hybridGpus;
}


int main() {
   std::vector<GPUInfoHybrid> gpuList = getHybridGPUInfo();
     for (const auto& gpu : gpuList) {
         std::wcout << L"Name: " << gpu.name << std::endl;
        if(gpu.wmiUsed){
            std::wcout << L"Retrieved using WMI" << std::endl;
             std::wcout << L"Adapter Type: " << gpu.adapterType << std::endl;
           std::wcout << L"Video Processor: " << gpu.videoProcessor << std::endl;
           std::wcout << L"Driver Version: " << gpu.driverVersion << std::endl;

        } else {
            std::wcout << L"Vendor: " << gpu.vendor << std::endl;
            std::wcout << L"Dedicated Memory: " << gpu.dedicatedMemory << " MB" << std::endl;
        }

        std::wcout << std::endl;
    }
    return 0;
}
```

The combined approach first calls the DirectX retrieval, and if it succeeds, then it uses those results. If the DirectX retrieval fails (an empty vector is returned) then the WMI retrieval is performed and the `wmiUsed` flag is set to true in each returned element. This flag is used in the output loop to differentiate which data is available.

For supplementary information, consider exploring textbooks focusing on Windows systems programming, DirectX development, and COM fundamentals. Specific resources may include Microsoft's DirectX documentation, articles on the Windows Management Instrumentation architecture, and general guides on C++ system-level programming. These are best consulted directly from their publishers and dedicated sources as they are updated periodically.
