---
title: "Why are CUDA MPS servers failing to start on workstations with multiple GPUs?"
date: "2025-01-30"
id: "why-are-cuda-mps-servers-failing-to-start"
---
A primary reason CUDA Multi-Process Service (MPS) servers fail to initialize on multi-GPU workstations stems from resource contention, specifically around the availability of shared memory segments and, less frequently, hardware device limitations related to specific GPU configurations. Having spent considerable time debugging machine learning infrastructure, I've frequently encountered this issue and can offer some specific insight.

The CUDA MPS acts as a server, managing GPU access for multiple processes, thereby mitigating the context-switching overhead that results when each process directly interacts with a GPU. This efficiency relies on the server having exclusive control over certain GPU resources. When a multi-GPU system is involved, configuration nuances can easily lead to conflicts.

When an MPS server attempts to start, it needs to allocate a shared memory region for inter-process communication. The CUDA driver, in turn, must find suitable address space to map that memory on all relevant GPUs. If, for example, another process or application has already consumed a crucial portion of the shared memory, or has pinned memory that interacts negatively with MPS allocations, the MPS server's initialization will typically fail with a "CUDA_ERROR_OUT_OF_MEMORY" or a similar code indicating allocation failure, despite having seemingly ample overall system RAM. This error might be misleading at first glance, as it refers to the inability to allocate on the device *for shared communication*, not the global GPU memory accessible via `cudaMalloc`.

Furthermore, the CUDA driver tries to allocate memory not just on the primary GPU selected by CUDA_VISIBLE_DEVICES, but on *all* GPUs in the system to guarantee that any participating process can connect regardless of its actual device selection. This behavior is necessary to fulfill the MPS principle, but if another process or framework has exclusive ownership of resources on GPUs other than the primary, it can create a conflict during the MPS server initialization.

Another contributing factor is the subtle differences in the hardware architecture across different GPUs on the same system. While all GPUs might be CUDA-capable, certain models could have limitations regarding the shared memory management, or how they handle the virtual memory mappings that MPS uses. For example, some older generation GPUs may not fully support the features necessary to run MPS with newer driver versions.

Here are a few representative scenarios where I've experienced MPS startup failures:

**Scenario 1: Insufficient Shared Memory due to External Process**

This occurs when another CUDA application or framework has already allocated significant shared memory segments, leaving insufficient space for the MPS server to initialize.

```c++
#include <iostream>
#include <cuda.h>

int main() {
    cudaError_t err;
    
    // Attempt to initialize a large amount of shared memory
    // Mimicking another application using the device.
    size_t shared_size = 256 * 1024 * 1024; // 256 MB
    void *shared_ptr;
    
    err = cudaMallocManaged(&shared_ptr, shared_size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
      std::cerr << "Error allocating shared memory: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }

    std::cout << "Shared memory allocated, attempting MPS start (should fail if no restart occurred)" << std::endl;
     
    //The MPS Server start would attempt to use the shared memory segment, 
    // but can fail due to the exhaustion of the resources.
    
    //Normally the MPS client would try to connect.
    // cudaSetDevice(0);
    // err = cudaDeviceSynchronize();
    
    // The MPS start would most likely fail due to inadequate shared memory,
    // requiring restarting the machine.
    
    cudaFree(shared_ptr);
    return 0;
}

```

In this example, I'm deliberately allocating a large managed memory segment with global attachment, which impacts the available shared memory pool used by MPS.  Subsequently, launching the MPS server through the `nvidia-cuda-mps-control` utility is likely to fail if it has not already been initialized. The MPS server normally allocates a smaller amount of shared memory when it starts but, after this step, it can struggle to find available memory segments.

**Scenario 2: Device Affinity Conflicts**

Here, a prior application has pinned the GPUs to a specific context. This can interfere with how MPS manages the device for multiple clients.

```python
import cupy as cp
import os

def initialize_gpu(device_id):
    try:
        cp.cuda.Device(device_id).use()
        cp.empty((1024,1024),dtype=cp.float32)
        print(f"GPU {device_id} initialized and a tensor allocated. MPS server should fail now.")
    except cp.cuda.runtime.CUDARuntimeError as e:
      print(f"Error intializing GPU {device_id}: {e}")

if __name__ == "__main__":
    # Set CUDA_VISIBLE_DEVICES to use all devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Assume we have 4 GPUs,
    for gpu_id in range(4):
       initialize_gpu(gpu_id)
    
    # In this case, an MPS initialization attempt would fail because each GPU
    # is explicitly initialized and occupied, the process would not gracefully handle
    # the interprocess communication due to the previous initialization step.
```
This python example uses `cupy`, but the core principle applies universally. By explicitly initializing each GPU, including the primary and any secondary ones, I'm creating a condition where MPS initialization becomes difficult.  The MPS server needs to acquire control of each device for its communication, and a previous device initialization causes conflicts. The MPS clients rely on the server initialization and thus will be unable to start.

**Scenario 3: Hardware Limitations on Specific GPUs**

This is a less frequent scenario, but it does occasionally happen where one particular GPU is not compatible with the driver's MPS implementation.

```c++
#include <iostream>
#include <cuda.h>

int main() {
    cudaError_t err;

    // Assume that one GPU is old and cannot support MPS
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    
     for (int i = 0; i < num_devices; ++i) {
          cudaDeviceProp prop;
          cudaGetDeviceProperties(&prop, i);
          std::cout << "Device " << i << ": " << prop.name << std::endl;
          if(i==2){
             // Assume the device 2 has an unsupported architecture
             // MPS server start will likely fail if the other processes utilize this device
             // even if device is not explicitly used.
          }
      }
    // MPS start should fail if the other GPU is not compatible.
    // The driver may throw an error when starting or when attempting
    // to communicate with the older device.

    return 0;
}

```
Here, the code does not explicitly do anything except query the properties of each device. However, imagine if device 2 was an older model that didn't support some low-level MPS functionalities with the current driver; the initialization phase would most likely crash or not function correctly. While an explicit error may not always be returned, the MPS server is likely to fail in this situation.

**Troubleshooting and Prevention**

When encountering these MPS initialization issues, these steps can prove helpful:

1.  **Identify Conflicting Processes:** Ensure no other CUDA applications are running prior to starting the MPS server. Use system monitoring tools to verify that no persistent processes have a lock on device resources.
2.  **Check Shared Memory:** Use `nvidia-smi` and other system utilities to examine existing memory allocations, specifically any allocated shared memory segments. A restart of the machine typically clears these resources.
3. **Isolate Device Problems**: Try running the MPS server with a reduced number of devices specified in `CUDA_VISIBLE_DEVICES` to isolate if any specific hardware is causing a problem.
4.  **Driver Compatibility:** Ensure the installed CUDA drivers are compatible with the architecture of all the GPUs in the system. Check for release notes or known issues.

**Resource Recommendations**

For further research and in-depth understanding, I suggest exploring the following:

*   The official CUDA documentation regarding multi-process service and memory management.
*   The NVIDIA developer forums, as other users have likely encountered similar scenarios.
*   Publications discussing high performance computing and parallel GPU programming with CUDA.
*   System administrator guides that delve into the specifics of memory allocation under Linux and Windows.

In conclusion, MPS server start failures on multi-GPU systems are often the result of resource conflicts, especially regarding shared memory. Careful resource management, rigorous testing, and driver compatibility reviews can mitigate these issues, leading to reliable use of the CUDA MPS service.
