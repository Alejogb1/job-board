---
title: "How can I expose multiple GPUs using CUDA_VISIBLE_DEVICES environment variable?"
date: "2025-01-30"
id: "how-can-i-expose-multiple-gpus-using-cudavisibledevices"
---
The efficacy of CUDA_VISIBLE_DEVICES hinges on the underlying CUDA driver's ability to manage multiple GPUs and the application's architecture to utilize them concurrently.  My experience optimizing high-performance computing applications across diverse hardware configurations has shown that simply setting the environment variable is often insufficient; careful consideration of process management and application design is crucial.  Ignoring these aspects can lead to unpredictable behavior, including unexpected single-GPU utilization despite specifying multiple devices.

**1. Clear Explanation**

The `CUDA_VISIBLE_DEVICES` environment variable dictates which GPUs a CUDA application can access.  It's a comma-separated string of GPU indices, where the index corresponds to the order presented by the `nvidia-smi` command.  For example, `CUDA_VISIBLE_DEVICES=0,1,2` makes GPUs 0, 1, and 2 visible to the CUDA application.  However,  the operating system and CUDA driver handle the actual allocation and scheduling of resources onto these devices.  The application itself must be designed to leverage multiple GPUs.  This usually involves utilizing either multiprocessing techniques (spawning multiple processes, each accessing a subset of the visible GPUs) or multithreading within a single process (using CUDA streams and threads to distribute work across devices).  The critical distinction lies in whether your application inherently supports parallel execution across multiple GPUs.  A single-threaded application will only utilize a single GPU, regardless of the `CUDA_VISIBLE_DEVICES` setting.

Furthermore, resource contention can be a significant issue.  While you can expose multiple GPUs, memory bandwidth, PCIe bus saturation, and inter-GPU communication latency can limit performance gains from adding more GPUs.  A poorly designed application might experience a performance decrease by adding more GPUs due to these overhead factors exceeding the benefits of parallel processing.  Careful profiling and optimization are essential, particularly for applications with significant data transfer between devices.  My experience indicates that simply adding more GPUs does not automatically translate to linear performance scaling.  Instead, expect diminishing returns as these resource limitations become more prominent.


**2. Code Examples with Commentary**

**Example 1: Multiprocessing (Python)**

```python
import os
import subprocess
import multiprocessing

def run_cuda_process(gpu_id, command):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, shell=True)
    process.wait()

if __name__ == "__main__":
    num_gpus = 3 # Modify according to your system
    command = "my_cuda_application" # Replace with your application command
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run_cuda_process, args=(i, command))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
```

*Commentary:* This example demonstrates the use of multiprocessing to run separate instances of the CUDA application (`my_cuda_application`), each assigned a single GPU via the environment variable.  Each process has exclusive access to its designated GPU, minimizing resource contention.  This approach is suitable for applications that can easily be parallelized across multiple independent tasks.  It's crucial that `my_cuda_application` is designed to operate independently on its assigned GPU and not attempt to communicate extensively with other processes.


**Example 2: Multithreading with CUDA Streams (C++)**

```cpp
#include <cuda_runtime.h>
// ... other includes

int main() {
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);

  if (num_gpus < 2) {
      printf("Less than 2 GPUs detected. Exiting.\n");
      return 1;
  }

  cudaSetDevice(0); //Example: Setting the device 0

  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);


  // Launch kernels on different streams for different GPUs.
  // Kernel launches here on stream0 and stream1 for GPU 0 and GPU 1 respectively

  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);

  return 0;
}
```


*Commentary:* This C++ example leverages CUDA streams to perform concurrent operations across multiple GPUs within a single process.  This approach is more suitable for applications where significant data sharing is required between the GPUs or when fine-grained control over GPU execution is necessary. The code snippet only demonstrates stream creation; actual kernel launches and data transfers within the streams are application-specific and would need additional code.  Efficient stream management is crucial to avoid performance bottlenecks.  This necessitates a deep understanding of CUDA programming and hardware architecture.

**Example 3:  Using NVIDIA NCCL (C++)**

```cpp
#include <nccl.h>
// ... other includes

int main() {
    int rank, size;
    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclUniqueId id;
    // ... handle id exchange (using MPI or other suitable method)

    ncclCommInitRank(&comm, size, id, rank);

    // Allocate data on each GPU and perform operations using NCCL collective communication routines (e.g., ncclAllReduce)

    ncclCommDestroy(comm);
    return 0;
}
```

*Commentary:*  This example illustrates the use of NVIDIA's NCCL (NVIDIA Collective Communications Library) for efficient inter-GPU communication. NCCL provides optimized primitives for collective operations (like reduction, broadcast, all-gather), vital for applications requiring extensive data exchange between multiple GPUs.  NCCL integration usually necessitates a distributed computing framework such as MPI (Message Passing Interface).  This approach requires significant expertise in distributed computing and is most beneficial for large-scale applications demanding high inter-GPU communication bandwidth.


**3. Resource Recommendations**

For a comprehensive understanding of CUDA programming and multi-GPU applications, I strongly recommend consulting the official NVIDIA CUDA documentation.  Detailed guides on multiprocessing in Python, using CUDA streams, and implementing NCCL are readily available within this documentation. In addition, textbooks focusing on parallel and distributed computing provide valuable theoretical background and algorithm design principles.  Finally,  performance analysis tools such as NVIDIA Nsight Systems and Nsight Compute are crucial for identifying bottlenecks and optimizing your application for multi-GPU utilization.  Systematic performance profiling is paramount to achieving efficient multi-GPU performance.
