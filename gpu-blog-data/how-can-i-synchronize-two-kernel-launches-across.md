---
title: "How can I synchronize two kernel launches across two streams on the GPU?"
date: "2025-01-30"
id: "how-can-i-synchronize-two-kernel-launches-across"
---
Kernel synchronization across multiple streams on a GPU, particularly when striving for efficient parallel execution, presents a common challenge in heterogeneous computing. Achieving correct and optimized concurrency requires understanding not only stream semantics but also the appropriate synchronization primitives. Without explicit synchronization, kernels within different streams might execute out-of-order relative to one another, leading to race conditions and unpredictable program behavior. My experience working on large-scale numerical simulations has made understanding these nuances crucial for achieving both correctness and performance.

Let's first clarify that CUDA streams, or their equivalents in other compute APIs like OpenCL or SYCL, provide a mechanism for asynchronous execution of work on the GPU. Operations submitted to different streams can theoretically execute concurrently, provided their underlying data dependencies are met and the hardware resources are available. However, the order of operations within a single stream is guaranteed to be maintained. Therefore, to synchronize operations across distinct streams, we must introduce synchronization mechanisms that explicitly govern the relative ordering of kernels.

The standard method involves employing CUDA events, which act as markers in a stream's execution timeline. These events can be captured using `cudaEventRecord()`, which records the state of the stream when the function is called. Crucially, recorded events can then be associated with another stream by using `cudaStreamWaitEvent()`. This function blocks execution in the second stream until the specified event has occurred on the first stream. This mechanism forces the intended synchronization. Note that these functions have counterparts in other APIs, although the naming might differ. For example, OpenCL would use command queues with events for a similar purpose.

Consider a scenario where two kernels, `kernelA` and `kernelB`, must operate on different data. `kernelA` produces intermediate results that are consumed by `kernelB`. We aim to execute `kernelA` on stream 0 and `kernelB` on stream 1, but `kernelB` needs the results of `kernelA`. The correct workflow dictates that we execute `kernelA` on stream 0, record an event after its completion, and then have stream 1 wait for that event before launching `kernelB`.

The simplest implementation would follow this pattern:

```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelA(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f; // Simulate some work
    }
}

__global__ void kernelB(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
        data[idx] = data[idx] + 1.0f; // Simulate some work
    }
}


int main() {
    int size = 1024;
    float* dataA, * dataB;
    float* deviceDataA, * deviceDataB;

    cudaMallocManaged(&dataA, size * sizeof(float));
    cudaMallocManaged(&dataB, size * sizeof(float));

    cudaMallocManaged(&deviceDataA, size * sizeof(float));
    cudaMallocManaged(&deviceDataB, size * sizeof(float));
    
    for (int i = 0; i < size; i++){
      dataA[i] = 1.0f;
      dataB[i] = 2.0f;
    }
    cudaMemcpy(deviceDataA, dataA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataB, dataB, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream0, stream1;
    cudaEvent_t event;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaEventCreate(&event);

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    kernelA<<<gridSize, blockSize, 0, stream0>>>(deviceDataA, size);
    cudaEventRecord(event, stream0); // Record the event on stream 0
    
    cudaStreamWaitEvent(stream1, event, 0); // Make stream 1 wait for the event
    kernelB<<<gridSize, blockSize, 0, stream1>>>(deviceDataA, size);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaMemcpy(dataA, deviceDataA, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dataB, deviceDataB, size * sizeof(float), cudaMemcpyDeviceToHost);


    for(int i=0; i<5; ++i)
      std::cout << "dataA[" << i << "] = " << dataA[i] << " dataB[" << i << "] = " << dataB[i] <<  "\n";
    
    cudaFree(deviceDataA);
    cudaFree(deviceDataB);
    cudaFree(dataA);
    cudaFree(dataB);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaEventDestroy(event);
    return 0;
}
```

In this first code example, `kernelA` modifies `deviceDataA`. Before launching `kernelB`, a CUDA event is recorded on `stream0` immediately after `kernelA`'s launch. Then, we force `stream1` to wait for the recorded event using `cudaStreamWaitEvent` prior to launching `kernelB` onto the same data. This ensures that the operations on `deviceDataA` are properly serialized and `kernelB` sees the modifications made by `kernelA`. We finally sync both streams before copying results back to the host for verification.

Another approach, useful when data transfers are also involved, uses events for both kernel completion and data availability. For example, one might need to copy data to the GPU on one stream, then perform a computation on another, then copy the results back.

```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelC(float* data, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 3.0f;
    }
}


int main() {
    int size = 1024;
    float* dataC, * dataD;
    float* deviceDataC, * deviceDataD;

    cudaMallocManaged(&dataC, size * sizeof(float));
    cudaMallocManaged(&dataD, size * sizeof(float));

    cudaMallocManaged(&deviceDataC, size * sizeof(float));
    cudaMallocManaged(&deviceDataD, size * sizeof(float));
    
    for (int i = 0; i < size; i++){
      dataC[i] = 1.0f;
      dataD[i] = 2.0f;
    }

    cudaStream_t stream0, stream1;
    cudaEvent_t copyEvent, kernelEvent;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaEventCreate(&copyEvent);
    cudaEventCreate(&kernelEvent);


    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    cudaMemcpyAsync(deviceDataC, dataC, size * sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaEventRecord(copyEvent, stream0);
    
    cudaStreamWaitEvent(stream1, copyEvent, 0);
    kernelC<<<gridSize, blockSize, 0, stream1>>>(deviceDataC, size);
    cudaEventRecord(kernelEvent, stream1);
    
    cudaStreamWaitEvent(stream0, kernelEvent, 0);
    cudaMemcpyAsync(dataC, deviceDataC, size * sizeof(float), cudaMemcpyDeviceToHost, stream0);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    for(int i=0; i<5; ++i)
        std::cout << "dataC[" << i << "] = " << dataC[i] << " dataD[" << i << "] = " << dataD[i] <<  "\n";
        
    cudaFree(deviceDataC);
    cudaFree(deviceDataD);
    cudaFree(dataC);
    cudaFree(dataD);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaEventDestroy(copyEvent);
    cudaEventDestroy(kernelEvent);

    return 0;
}

```

Here, we copy `dataC` to `deviceDataC` on stream 0 asynchronously and record `copyEvent` when done. Stream 1 waits for this event, then launches `kernelC`. We subsequently record `kernelEvent` after this kernel's completion, before initiating a transfer of the results to `dataC` on the host stream, but only after making the host stream wait on `kernelEvent`, avoiding a race. Both streams then synchronize, ensuring data transfers are complete and results are valid.

For complex applications, using events for pairwise synchronization can become cumbersome. In such cases, consider using CUDA graph primitives, which allow you to explicitly define the data dependencies within the kernels at a higher level. This, however, comes at the expense of flexibility and increased complexity and should be considered only when simpler options are insufficient.

Finally, advanced memory models can offer a subtle form of implicit synchronization. The Unified Virtual Memory (UVM) model in CUDA, for instance, can improve performance by managing data migration implicitly. When two kernels access the same data and they do not both run on the same device, the driver often handles the transfer and may create a dependency for you, although it's difficult to specify when this happens. Relying on this without using events is unreliable, particularly when explicit stream synchronization is required. Here's a quick modification of the first example to illustrate how a UVM copy *can* introduce the necessary dependency in this limited case, although I recommend using explicit events as detailed above for reliable inter-stream synchronization:

```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelE(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernelF(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
}


int main() {
    int size = 1024;
    float* dataE, * dataF;
    float* deviceDataE, * deviceDataF;

    cudaMallocManaged(&dataE, size * sizeof(float));
    cudaMallocManaged(&dataF, size * sizeof(float));

    cudaMallocManaged(&deviceDataE, size * sizeof(float));
    cudaMallocManaged(&deviceDataF, size * sizeof(float));
    
    for (int i = 0; i < size; i++){
      dataE[i] = 1.0f;
      dataF[i] = 2.0f;
    }
    cudaMemcpy(deviceDataE, dataE, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataF, dataF, size * sizeof(float), cudaMemcpyHostToDevice);


    cudaStream_t stream0, stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);


    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);


    kernelE<<<gridSize, blockSize, 0, stream0>>>(deviceDataE, size);
    // No explicit event, relying on UVM potential implicit behavior
    kernelF<<<gridSize, blockSize, 0, stream1>>>(deviceDataE, size); // using deviceDataE again

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaMemcpy(dataE, deviceDataE, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dataF, deviceDataF, size * sizeof(float), cudaMemcpyDeviceToHost);

   for(int i=0; i<5; ++i)
      std::cout << "dataE[" << i << "] = " << dataE[i] << " dataF[" << i << "] = " << dataF[i] <<  "\n";

    cudaFree(deviceDataE);
    cudaFree(deviceDataF);
    cudaFree(dataE);
    cudaFree(dataF);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    return 0;
}
```

In this third example, both kernels operate on the same data array, `deviceDataE`. We *are* relying on the implicit behavior of the UVM system to introduce a data dependency between the kernel calls in different streams in this specific instance because `kernelF` is accessing a page that is still being used in `kernelE` but such behavior is *not* a reliable basis for synchronization. It is provided purely to illustrate the potential for implicit behavior when such a setup occurs. *Do not rely on this behavior when implementing real-world solutions.* The prior two examples are preferred methods of synchronization between streams and are the methods that should be used in practical cases.

For further information on stream management, I recommend consulting the CUDA programming guide, particularly the sections on stream and event management. Texts on parallel programming and GPU computing also often dedicate specific chapters to these topics. Additionally, performance analysis tools such as NVIDIA Nsight Systems can be instrumental in visualizing stream behavior and verifying synchronization correctness. Always prioritize correctness of results over perceived performance gains as subtle race conditions can lead to unexpected outcomes. Explicit synchronization via events, as described, remains the most reliable method for inter-stream kernel ordering, and in most scenarios this approach will provide the best trade-off between performance and correctness.
