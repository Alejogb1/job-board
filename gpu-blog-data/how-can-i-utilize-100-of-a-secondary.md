---
title: "How can I utilize 100% of a secondary GPU's VRAM from a single process in Windows 10?"
date: "2025-01-30"
id: "how-can-i-utilize-100-of-a-secondary"
---
The core challenge in fully utilizing a secondary GPU's Video RAM (VRAM) from a single process on Windows 10 stems from the system’s default graphics device selection and the limitations imposed by standard graphics APIs. Direct control over which GPU executes a particular rendering operation is not always straightforward, particularly when targeting a secondary device. While Windows attempts to distribute workload across available GPUs, it often defaults to the primary display adapter, leaving the secondary GPU underutilized if not explicitly directed. Achieving 100% VRAM usage requires careful management of device selection, memory allocation, and kernel execution.

My experience developing a high-performance ray tracing application highlighted these issues. Initially, my application, though designed to leverage all available GPUs, primarily burdened the primary card, leaving the second one largely idle. This spurred an investigation into Direct3D 12's multi-adapter capabilities, ultimately leading to a solution. The key is not just identifying the desired GPU but actively allocating resources and executing commands on that specific device context.

**1. Explicit GPU Selection & Device Creation**

The first step is programmatically identifying and selecting the secondary GPU. Windows exposes multiple GPU adapters, and standard enumeration methods often prioritize the primary display device. To circumvent this, we utilize the DirectX Graphics Infrastructure (DXGI) API to query all available adapters and filter based on characteristics like vendor or device description. The critical element is accessing the `IDXGIAdapter` object associated with the desired secondary GPU. Subsequently, the Direct3D 12 API needs to use that adapter to create a device instance, ensuring all future memory allocations and execution commands target that specific device. Standard device creation without specifying the adapter will often default to the primary one.

**2. Resource Allocation on the Secondary GPU**

Once the secondary device is created, all subsequent resource allocations must be bound to it. This includes creating textures, buffers, and any other data residing in VRAM. The crucial aspect is using the device instance obtained from the secondary GPU to create resource heap descriptions (`D3D12_HEAP_PROPERTIES`) and memory heaps (`ID3D12Heap`). A failure to specify the correct device during heap creation will result in the memory being allocated on the default device, circumventing our intention. Furthermore, it's worth noting that allocating large amounts of contiguous memory can sometimes fail. We can mitigate that with careful segmentation if required for large textures. Furthermore, the `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` flag can also help in certain circumstances.

**3. Command Execution on the Secondary GPU**

Finally, all rendering commands or computational kernels need to execute on the secondary device's command queue. Direct3D 12 utilizes command lists and command queues for asynchronous GPU operations. When creating command lists or allocating command allocators, they must also be associated with the secondary device’s queue. Failing to do so will result in commands being submitted to the default device's queue, leading to unexpected behavior and ineffective utilization of the secondary GPU's VRAM. Once the command list is executed, the associated memory is resident on that particular GPU for use and is also accounted for in its VRAM usage.

**Code Examples**

**Example 1: Adapter Enumeration and Device Creation**

```cpp
#include <dxgi.h>
#include <d3d12.h>
#include <iostream>

// Assume error handling is performed and is omitted for brevity

ComPtr<IDXGIAdapter4> GetSecondaryAdapter()
{
    ComPtr<IDXGIFactory4> factory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));

    UINT adapterIndex = 0;
    ComPtr<IDXGIAdapter1> adapter;
    std::vector<ComPtr<IDXGIAdapter4>> availableAdapters;

    while (factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            adapterIndex++;
            continue;
        }

        ComPtr<IDXGIAdapter4> adapter4;
        adapter->QueryInterface(IID_PPV_ARGS(&adapter4));

        availableAdapters.push_back(adapter4);
        adapterIndex++;
    }

     // Assuming the second adapter is the desired secondary GPU
    if (availableAdapters.size() > 1)
    {
       return availableAdapters[1];
    }

     return nullptr; //Handle case with no secondary GPU
}


ComPtr<ID3D12Device> CreateSecondaryDevice() {
    ComPtr<IDXGIAdapter4> adapter = GetSecondaryAdapter();
    if (adapter == nullptr)
    {
        // Handle lack of second adapter
        return nullptr;
    }

    ComPtr<ID3D12Device> device;
    D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    return device;
}

int main() {
    ComPtr<ID3D12Device> secondaryDevice = CreateSecondaryDevice();
    if (secondaryDevice) {
       std::cout << "Secondary Device Created Successfully" << std::endl;
    } else {
       std::cout << "Failed to create Secondary Device" << std::endl;
    }
    return 0;
}

```

*   This code snippet demonstrates the explicit adapter enumeration and subsequent device creation. It obtains all adapters and explicitly selects the second one available for use.

**Example 2: VRAM Allocation on the Secondary Device**

```cpp
#include <d3d12.h>

// Assuming secondaryDevice from example 1 is valid
// Assume error handling is performed and is omitted for brevity

void AllocateVRAM(ID3D12Device* secondaryDevice, UINT64 allocationSize, ID3D12Heap** outHeap)
{
        D3D12_HEAP_PROPERTIES heapProperties;
        heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProperties.CreationNodeMask = 1;
        heapProperties.VisibleNodeMask = 1;

        D3D12_HEAP_DESC heapDesc;
        heapDesc.SizeInBytes = allocationSize;
        heapDesc.Properties = heapProperties;
        heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        heapDesc.Flags = D3D12_HEAP_FLAG_NONE;


        secondaryDevice->CreateHeap(&heapDesc, IID_PPV_ARGS(outHeap));
}


int main()
{
  ComPtr<ID3D12Device> secondaryDevice = CreateSecondaryDevice();
  if (secondaryDevice) {
       ID3D12Heap* heap = nullptr;
       AllocateVRAM(secondaryDevice.Get(), 1024 * 1024 * 1024 ,&heap); //Allocate 1 GB
       if (heap) {
          std::cout << "VRAM allocated successfully" << std::endl;
          heap->Release();
       }
      else {
        std::cout << "Failed to allocate VRAM on the secondary device" << std::endl;
      }
   }
   else
  {
     std::cout << "Secondary device not initialized. Failed allocation." << std::endl;
  }

  return 0;
}
```

*   This example illustrates how to explicitly allocate memory on the secondary device. It specifies `D3D12_HEAP_TYPE_DEFAULT` indicating GPU local memory and uses `secondaryDevice` to create the heap. Any buffer or texture created using this heap will reside in the secondary GPU's VRAM.

**Example 3: Command List Execution on the Secondary Device**

```cpp
#include <d3d12.h>

// Assuming secondaryDevice from example 1 and heap from example 2 is valid
// Assume error handling is performed and is omitted for brevity

void ExecuteCommandList(ID3D12Device* secondaryDevice, ID3D12Heap* heap)
{
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    secondaryDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator));

    ComPtr<ID3D12GraphicsCommandList> commandList;
    secondaryDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList));

     commandList->Close();

    ComPtr<ID3D12CommandQueue> commandQueue;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    secondaryDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue));


     ID3D12CommandList* ppCommandLists[] = {commandList.Get()};
      commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

     commandQueue->Signal(nullptr,1);
}


int main()
{
   ComPtr<ID3D12Device> secondaryDevice = CreateSecondaryDevice();
   if (secondaryDevice) {
      ID3D12Heap* heap = nullptr;
      AllocateVRAM(secondaryDevice.Get(), 1024*1024*1024, &heap);
      if (heap)
      {
          ExecuteCommandList(secondaryDevice.Get(), heap);
          heap->Release();
      } else {
          std::cout << "Failed to allocate heap" << std::endl;
      }

   } else
   {
      std::cout << "Failed to create secondary device" << std::endl;
   }

  return 0;
}
```

*   This snippet focuses on executing commands using the secondary device.  It creates a command queue and executes the command list which is bound to the allocated secondary device memory, ensuring all operations are performed on that specific GPU. The VRAM allocation and command execution are separate functions to illustrate the proper allocation flow.

**Resource Recommendations**

For a deeper understanding of Direct3D 12 and multi-adapter programming, I recommend exploring resources such as:

*   The Microsoft DirectX documentation on MSDN (now Microsoft Learn). This resource contains in-depth articles, tutorials, and API reference materials for DirectX. Focus on the sections related to multi-adapter programming, resource management, and command list creation.
*   "Introduction to 3D Game Programming with DirectX 12" (Frank Luna). This book provides a comprehensive overview of the Direct3D 12 API and offers practical guidance on complex topics such as resource management and rendering pipelines. Pay particular attention to chapters on command queues and resource binding.
*   The GitHub repository for DirectX-Graphics-Samples provides numerous working examples of DirectX 12 features, including those related to multiple GPU configurations. Examine the source code of relevant examples to understand best practices and implementation details.

Achieving full VRAM utilization on a secondary GPU is not a trivial task. It requires meticulous attention to device selection, memory allocation, and command execution. By leveraging the capabilities of Direct3D 12’s multi-adapter features and following the principles outlined above, developers can maximize the performance of multi-GPU setups, leading to more performant and scalable applications.
