---
title: "How does DirectX 11 handle multiple GPUs?"
date: "2025-01-30"
id: "how-does-directx-11-handle-multiple-gpus"
---
DirectX 11’s approach to multi-GPU support, while not fully transparent in all aspects, primarily relies on the concept of “linked adapters” managed through the DXGI (DirectX Graphics Infrastructure) layer. I’ve worked extensively on rendering engines and this is a particular area where I’ve encountered both performance gains and development complexities. The driver, not DirectX itself, decides if multiple physical GPUs can operate as a single logical unit to the application. This involves a blend of explicit multi-adapter programming and implicit behaviors driven by the driver, where some tasks are automatically delegated across GPUs with minimal application intervention.

The foundation is the `IDXGIAdapter` interface. Each physical GPU is represented by an `IDXGIAdapter` object. When multiple suitable adapters exist, DXGI will expose all of them. The application has the option to query the properties of each adapter, such as the amount of video memory, vendor ID, and device name, allowing it to make informed decisions regarding how it might use available resources. Crucially, one of the adapters will always be designated as the primary or default adapter. This is typically the one connected to the display. Applications operating without explicit multi-adapter considerations will primarily interact with this primary adapter; other adapters will effectively remain unused for graphics rendering, unless explicitly instructed by the application to do otherwise.

A common use case for multi-GPU scenarios is Alternate Frame Rendering (AFR), where each GPU renders a different frame in sequence. This is typically not something an application directly implements with DirectX 11. The application generates the command list, submits it for execution, and the driver and hardware abstract away the underlying process of distributing that workload across the linked adapters. This works when the drivers handle splitting the workload. DirectX 11 also provides explicit tools for leveraging multiple GPUs; however, the onus is on the application to carefully manage command lists, render targets, and other resources across different adapters. The simplest approach, and the most often used, relies on drivers splitting command lists or using Alternate Frame Rendering, AFR, which, again, is the most common approach.

DirectX 11 exposes specific functions to handle linked adapters. To determine the number of available GPUs, I would iterate through `IDXGIAdapter1` objects. I've often encountered situations where laptops report two GPUs, one from the iGPU and another from a dedicated NVIDIA or AMD GPU, but they do not work as "linked adapters." The primary adapter will always be there, but it does not mean it works in multi-GPU. You can query their properties using the `DXGI_ADAPTER_DESC1` structure to figure out which are capable of working together.

Here's code demonstrating how one might enumerate available adapters:

```c++
#include <dxgi.h>
#include <iostream>
#include <vector>

void EnumerateAdapters()
{
    IDXGIFactory1* pFactory;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    if(FAILED(hr)) {
        std::cerr << "Failed to create DXGIFactory1" << std::endl;
        return;
    }

    std::vector<IDXGIAdapter1*> adapters;
    IDXGIAdapter1* pAdapter;
    UINT i = 0;
    while(pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
       adapters.push_back(pAdapter);
        i++;
    }
    pFactory->Release();

    for(size_t j = 0; j < adapters.size(); ++j) {
        DXGI_ADAPTER_DESC1 desc;
        adapters[j]->GetDesc1(&desc);

        wprintf(L"Adapter %zu:\n", j);
        wprintf(L"  Description: %s\n", desc.Description);
        wprintf(L"  Vendor ID: %x\n", desc.VendorId);
        wprintf(L"  Dedicated Video Memory: %llu MB\n", desc.DedicatedVideoMemory / (1024 * 1024));
       adapters[j]->Release();
    }

}

int main()
{
    EnumerateAdapters();
    return 0;
}
```

This code snippet creates a factory object, iterates through the available adapters, retrieves their descriptions, and prints details like the description, vendor ID, and dedicated video memory. This provides critical insight into what adapters are present on the system, which is the first step in deciding how to handle a multiple adapter setup. The descriptions usually include information as the NVIDIA GeForce RTX or AMD Radeon, making it easier to identify GPUs from integrated ones. It’s important to release each adapter object after use to prevent resource leaks.

For explicit multi-GPU management in DirectX 11, the application must create multiple device contexts, one for each adapter selected. This requires careful command list management and data sharing between contexts. Here is an example of how one would do that:

```c++
#include <d3d11.h>
#include <iostream>
#include <vector>

struct GPUContext {
    ID3D11Device* device;
    ID3D11DeviceContext* context;
};


void CreateGPUContexts(std::vector<IDXGIAdapter1*>& adapters, std::vector<GPUContext>& gpuContexts){
    for(auto adapter: adapters){
         GPUContext ctx;
        HRESULT hr = D3D11CreateDevice(
           adapter,
           D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            &ctx.device,
            nullptr,
           &ctx.context
        );
         if (FAILED(hr)){
             std::cerr << "Failed to create device for adapter" << std::endl;
           continue;
        }
        gpuContexts.push_back(ctx);
    }

}

int main() {

  IDXGIFactory1* pFactory;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    if(FAILED(hr)) {
        std::cerr << "Failed to create DXGIFactory1" << std::endl;
        return -1;
    }

    std::vector<IDXGIAdapter1*> adapters;
    IDXGIAdapter1* pAdapter;
    UINT i = 0;
    while(pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
       adapters.push_back(pAdapter);
        i++;
    }
   pFactory->Release();

    std::vector<GPUContext> gpuContexts;
    CreateGPUContexts(adapters, gpuContexts);

   for(const auto& ctx : gpuContexts){
       if (ctx.device) ctx.device->Release();
        if (ctx.context) ctx.context->Release();
    }

   for(auto adapter : adapters){
      adapter->Release();
   }

   return 0;
}
```

Here, I iterate through the available adapters and create a Direct3D 11 device and device context for each, storing them in a `GPUContext` struct. This allows the application to have an API entry point per GPU. The critical step here is to ensure that each context operates on its own resources, or that they access shared resources correctly. This code doesn't include rendering logic, but the creation of multiple contexts is the key element. The `D3D11_CREATE_DEVICE_BGRA_SUPPORT` flag is recommended for better compatibility with older GPUs. It is critical to remember that these contexts will operate independently unless you explicitly set up a way for them to share information or work with shared resources. This is critical for handling data in multi-GPU configurations. For example, when multiple GPUs work together on the same frame you'd need to share the result of a render or copy information between them. This example also shows that all created objects must be released properly.

Finally, explicit multi-adapter management is complicated, involving the creation of render targets, buffers, and constant buffers on specific adapters, careful synchronization of resources and command lists across multiple devices, and sometimes, explicit data transfer between GPU memory spaces. A simple approach might involve rendering different sub-regions of the viewport across the GPUs, then combining the results into a single frame. This requires carefully created command lists for each device. An example would be:

```c++
#include <d3d11.h>
#include <iostream>
#include <vector>

// Assuming GPUContext is defined as in the previous example

void RenderOnEachGPU(std::vector<GPUContext>& gpuContexts){
    for(size_t i = 0; i < gpuContexts.size(); ++i){
        ID3D11DeviceContext* context = gpuContexts[i].context;
         //Example: rendering to a color target
        D3D11_VIEWPORT viewport = {0.0f, 0.0f, 1280.f/gpuContexts.size(), 720.f, 0.0f, 1.0f};
        context->RSSetViewports(1, &viewport);
        float clearColor[4] = { (float)i/gpuContexts.size(), 0.0f, 0.0f, 1.0f };
        // Assuming a previously set render target
        ID3D11RenderTargetView* pRTView = nullptr;
        context->OMSetRenderTargets(1, &pRTView, nullptr);
        context->ClearRenderTargetView(pRTView, clearColor);

        //Add your actual rendering commands here, for example a simple triangle
        // and draw a simple triangle.
        //This would require setting up the input layout, vertex shader, and pixel shader
        //but it's skipped for demonstration purposes
         context->Flush(); //Submit commands for this GPU
    }
}


int main() {

  IDXGIFactory1* pFactory;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    if(FAILED(hr)) {
        std::cerr << "Failed to create DXGIFactory1" << std::endl;
        return -1;
    }

    std::vector<IDXGIAdapter1*> adapters;
    IDXGIAdapter1* pAdapter;
    UINT i = 0;
    while(pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
       adapters.push_back(pAdapter);
        i++;
    }
   pFactory->Release();


   std::vector<GPUContext> gpuContexts;
   CreateGPUContexts(adapters, gpuContexts);
   if(gpuContexts.size() > 0){
        RenderOnEachGPU(gpuContexts);
    }

    for(const auto& ctx : gpuContexts){
       if (ctx.device) ctx.device->Release();
       if (ctx.context) ctx.context->Release();
    }

   for(auto adapter : adapters){
      adapter->Release();
   }


   return 0;

}
```

This is a highly simplified approach to show how to render different things across GPUs. In this example, each GPU renders to its viewport, and a different clear color for visibility. In practice, you need to handle the creation of resources, setting up viewports and render targets, and synchronizing rendering between them. Without proper care, you'll encounter synchronization issues or resource contention, resulting in poor performance or rendering artifacts.

For further exploration of DirectX multi-GPU capabilities, I recommend the following resources. The DirectX Graphics documentation included in the Windows SDK provides detailed reference material regarding DXGI and device creation. Additionally, various online tutorials and books provide general advice and insight. Look at materials detailing advanced rendering techniques, which often cover multi-GPU concepts, and specifically the DirectX documentation section regarding linked adapters. Finally, sample applications included in the Windows SDK are always useful because they provide a hands-on approach that is often overlooked when reading the documentation.
