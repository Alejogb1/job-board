---
title: "How does inter-GPU communication impact MPI+OpenACC performance?"
date: "2025-01-30"
id: "how-does-inter-gpu-communication-impact-mpiopenacc-performance"
---
Inter-GPU communication, particularly when utilizing a combination of Message Passing Interface (MPI) and OpenACC, introduces significant performance challenges that can overshadow the benefits of parallel execution across multiple GPUs. The primary issue stems from the inherent overhead of moving data between GPUs connected via various interconnects like PCIe, NVLink, or others. This data movement, often implicit within MPI communication primitives, can create bottlenecks if not managed carefully, negating the performance gains offered by parallel computation. My experiences implementing distributed numerical simulations on multi-GPU systems have consistently reinforced this understanding.

When we consider the combined MPI+OpenACC paradigm, we are typically leveraging MPI to orchestrate data distribution and communication across distinct compute nodes, each potentially equipped with one or more GPUs, while OpenACC offloads compute-intensive kernels onto the local GPUs. This architecture results in a complex interplay between host (CPU) memory, device (GPU) memory, and network communication. For optimal performance, it’s crucial to minimize data transfers across both the network and the PCIe bus.

Let’s break down the key areas where inter-GPU communication affects performance. First, the latency and bandwidth of the interconnect directly limit the speed at which data can be exchanged between GPUs. Consider a scenario where large arrays are distributed using MPI across multiple nodes, each having a GPU. Before any computation can occur on a remote GPU, the required portion of the array must be transferred over the interconnect, serialized, and placed into the remote GPU's memory. This adds delay, sometimes dramatically, if the interconnect isn't sufficiently fast for the workload. Furthermore, any non-contiguous data requires additional overhead to pack, transfer, and unpack.

Secondly, the usage of CPU buffers as intermediate staging areas can significantly degrade performance. Typically, when MPI communication is initiated, data from the host memory is copied into a send buffer, serialized, transmitted over the network, received, deserialized into a receive buffer on the target node, and then, in this case, copied from the host to the GPU memory. This double-copy process, first from GPU to CPU memory and then CPU to GPU memory, is inefficient, especially for large datasets. Ideally, we want to transfer data directly between GPU memories. Newer technologies and MPI libraries are increasingly supporting GPU-direct communication.

Thirdly, the overlap between communication and computation is often limited. The time a GPU spends waiting for data from another GPU or processing unit is wasted computing time. While it’s possible to perform asynchronous data transfers and overlapping computation on separate streams, a poorly structured MPI-OpenACC application may unintentionally introduce synchronisation points, causing all GPUs to remain idle. Efficient utilization of asynchronous communication and non-blocking MPI calls is essential to fully utilize the performance available on a multi-GPU system.

Let me demonstrate with three example scenarios, each presenting a different aspect of the issue:

**Example 1: Naive Data Transfer with CPU Buffers**

This example illustrates the most basic, yet often inefficient, approach to data transfer: the use of CPU buffers for inter-GPU data exchange. I will assume a basic MPI setup with multiple processes running on different nodes, each having a GPU. This C++ code snippet highlights the problematic transfer method:

```cpp
#include <mpi.h>
#include <openacc.h>
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = 1024*1024;
  std::vector<float> host_data(n);
  std::vector<float> device_data;
  if(rank==0){
      for (int i=0; i<n; ++i){
        host_data[i] = static_cast<float>(i);
    }
  }

  #pragma acc enter data copyin(host_data[0:n])
  #pragma acc update device(host_data[0:n])
  device_data = host_data;
  if (rank==0){
     for (int i = 0; i < size; ++i){
      if(i != rank){
         MPI_Send(host_data.data(), n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      }
     }
  }
  else {
      MPI_Recv(host_data.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
   #pragma acc update device(host_data[0:n])
   device_data = host_data;
  #pragma acc exit data delete(host_data[0:n])

  MPI_Finalize();
  return 0;
}
```

Here, `host_data` resides in host memory. Even after transferring data to the GPU via `#pragma acc update device`, `MPI_Send` and `MPI_Recv` involve the CPU buffer. In a real scenario, the received data, even when updated to the GPU, will require a copy into the actual device arrays for processing. This code does not show any processing to keep the focus on the transfer. This clearly exemplifies the suboptimal CPU-centric data movement we want to avoid.

**Example 2: GPU-Aware MPI Communication with Staging Area**

This next example explores using GPU memory for data staging on the sending side. I use the `MPI_Send` and `MPI_Recv` with data from device memory. This does not use direct memory access, instead the data is copied from device to host, transmitted, received, and copied to device memory on the receiving side.

```cpp
#include <mpi.h>
#include <openacc.h>
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = 1024*1024;
  std::vector<float> device_data(n);
  std::vector<float> host_data(n);


    if(rank == 0){
       for (int i = 0; i < n; ++i){
          device_data[i] = static_cast<float>(i);
        }
    }
  #pragma acc enter data copyin(device_data[0:n])
  
    if(rank == 0){
        #pragma acc update self(device_data[0:n])
          for(int i=0; i< size; ++i){
             if(i!=rank){
                #pragma acc update host(device_data[0:n])
                 MPI_Send(device_data.data(), n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
           }
        }
    }
    else{
       #pragma acc update host(device_data[0:n])
       MPI_Recv(device_data.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  #pragma acc update device(device_data[0:n])
  #pragma acc exit data delete(device_data[0:n])
  MPI_Finalize();
  return 0;
}
```
While this approach eliminates the unnecessary intermediate host buffer from the first code example, the `update` directives still results in two memory copies per transmission, device -> host on the sending side, host -> device on the receiving side. The `MPI_Send` and `MPI_Recv` calls themselves still operate from the host memory.

**Example 3: Potential with GPU-Direct Communication**

 This final example is conceptual, demonstrating how direct GPU communication should ideally look. I will highlight the needed components to take advantage of the GPU direct features, but I will leave out the detailed implementation, since specific library setups vary a lot.

```cpp
#include <mpi.h>
#include <openacc.h>
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1024*1024;
    std::vector<float> device_data(n);
    
     if(rank == 0){
        for(int i = 0; i < n; ++i){
        device_data[i] = static_cast<float>(i);
      }
     }
   
    #pragma acc enter data copyin(device_data[0:n])
  
    if(rank == 0){
        for(int i=0; i< size; ++i){
         if(i!=rank){
             // Attempt to obtain an MPI datatype describing the device memory.
           MPI_Datatype device_datatype;
           // Create the MPI datatype from the device memory.
           // MPI_Type_create_hindexed_block(...)
            
           // Send device data directly
              MPI_Send(device_data.data(), n, device_datatype, i, 0, MPI_COMM_WORLD);
             MPI_Type_free(&device_datatype);
         }
      }
    } else {
      // Attempt to obtain an MPI datatype describing the device memory
      MPI_Datatype device_datatype;
      // Create the MPI datatype from the device memory
      // MPI_Type_create_hindexed_block(...)
      
      // Receive device data directly
      MPI_Recv(device_data.data(), n, device_datatype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Type_free(&device_datatype);
    }
    #pragma acc exit data delete(device_data[0:n])
    MPI_Finalize();
    return 0;
}
```

The key here is the creation of an MPI datatype that specifically describes the location and structure of data in the device memory. Libraries such as MVAPICH2-GDR and OpenMPI with CUDA-aware support enable these direct data transfers. The `MPI_Type_create_hindexed_block` is a placeholder, as implementation details vary.  This way, MPI can directly access and transfer the data from GPU to GPU, minimizing the involvement of the CPU memory and the associated overhead.

In summary, the performance impact of inter-GPU communication in MPI+OpenACC applications is significant. Minimizing CPU involvement in data transfers, leveraging GPU-direct communication wherever feasible, and careful planning of communication patterns are crucial steps to achieving scalable parallel execution across multiple GPUs.

For a deeper understanding, I suggest exploring resources on the following topics:
* CUDA-Aware MPI implementations
* OpenACC best practices for multi-GPU systems
* Interconnect technologies for multi-GPU systems (e.g., NVLink, InfiniBand)
* Asynchronous programming in OpenACC
* Optimization techniques for distributed memory systems
* Profiling tools to pinpoint communication bottlenecks.
By focusing on these areas, you will be better equipped to mitigate the performance challenges arising from inter-GPU communication.
