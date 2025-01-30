---
title: "How can Nsight Systems be used to trace custom CUDA kernels?"
date: "2025-01-30"
id: "how-can-nsight-systems-be-used-to-trace"
---
Nsight Systems, a performance analysis tool, leverages its unique ability to sample GPU activity and correlate it with CPU execution to trace the performance of custom CUDA kernels. I have personally used Nsight Systems extensively on projects involving high-performance scientific computing with custom CUDA implementations, and based on that experience, it’s important to understand that this tracing isn’t done through traditional debugging breakpoints. Instead, it utilizes sampling and profiling to construct a temporal representation of the kernel's execution and resource usage.

Fundamentally, Nsight Systems operates by gathering data through various mechanisms. When profiling CUDA applications, it captures GPU activity including kernel launches, memory transfers, and resource utilization. This data is then correlated with the CPU timeline to provide a holistic view of the application's execution flow. Custom kernels are treated as any other CUDA kernel within this profiling framework. Therefore, the primary challenge isn't *how* to make Nsight Systems recognize a custom kernel, but rather *how* to interpret the generated trace to pinpoint performance bottlenecks within that kernel. Nsight Systems does not require any special annotation within the custom CUDA code itself to trace the kernel, a common misconception.

A key feature of Nsight Systems relevant to custom kernels is its ability to identify specific instances of kernel launches. Each launch of a CUDA kernel will appear as a distinct region in the timeline view. This allows you to differentiate between different calls to the same kernel if that kernel is invoked with varying parameters or execution conditions. Furthermore, Nsight Systems provides detailed information about each kernel execution, including the grid and block dimensions, the occupancy of the multiprocessors, and the memory bandwidth used during the kernel execution. This data is indispensable for identifying potential performance issues.

The process of tracing custom kernels involves these key steps: First, the application needs to be executed under the control of the Nsight Systems profiling tool. This is typically done through the command line or the Nsight Systems user interface. Once the application executes, Nsight Systems gathers trace data. This data is then opened in the Nsight Systems graphical interface where it can be analyzed. This includes examining the GPU utilization timeline, focusing specifically on the regions where custom kernels are executing. From there, we analyze the collected data, looking for anomalies in execution time, memory usage, or occupancy. We typically focus on regions of high resource utilization, potential bottlenecks, and stalls.

Now, let's consider a few code examples to illustrate these points.

**Example 1: Basic Custom Kernel Analysis**

Consider a simple custom kernel that performs a vector addition:

```C++
__global__ void vectorAdd(float* a, float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
```

In this scenario, we would compile the CUDA code and run it using the Nsight Systems CLI:

```bash
nsys profile --gpu-metrics-all ./my_application
```

After capturing the trace and opening it within Nsight Systems, we can locate the `vectorAdd` kernel execution on the timeline. By selecting this kernel instance, we can inspect the launch parameters such as the grid and block size as well as the kernel duration. For example, if the kernel was launched with a suboptimal block size, we would observe low occupancy on the GPU multiprocessors, which is immediately apparent within the Nsight Systems interface, highlighting it as a potential optimization area. The resource usage timeline accompanying the kernel launch also reveals how bandwidth is consumed which could be used to identify memory bottlenecks.

**Example 2: Memory Bandwidth Bottleneck Identification**

Suppose we have a more complex kernel that uses a large amount of shared memory:

```C++
__global__ void sharedMemoryKernel(float* in, float* out, int size) {
    extern __shared__ float sharedData[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        sharedData[threadIdx.x] = in[i];
        __syncthreads();
        out[i] = sharedData[threadIdx.x];
    }
}

```

We would profile this using the same command:

```bash
nsys profile --gpu-metrics-all ./my_application
```

By selecting the `sharedMemoryKernel` region within Nsight Systems, we observe that the memory transfer timeline indicates a high bandwidth usage during the transfers from global to shared memory. The duration of this transfer compared to the compute portion of the kernel might highlight that we’re memory bound, potentially due to excessive shared memory use. Analyzing metrics like L2 cache hits and the rate of global memory accesses will be crucial to identify potential ways to alleviate this bottleneck. Additionally, we could evaluate the impact of different shared memory sizes. Nsight Systems would allow us to visualize the resulting change in performance by rerunning and comparing timelines.

**Example 3: Occupancy Analysis and Kernel Execution Time**

Consider a slightly more complex kernel where conditional execution is prevalent:

```C++
__global__ void conditionalKernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (data[i] > 0.5) {
            data[i] *= 2.0;
        } else {
            data[i] /= 2.0;
        }
    }
}
```
Again, profiling is done in the same manner:

```bash
nsys profile --gpu-metrics-all ./my_application
```

By focusing on the `conditionalKernel` region, Nsight Systems provides detailed information about the multiprocessor utilization. Observing regions of low occupancy and long kernel duration, in combination with instruction-level data provided, we would suspect that branch divergence within the kernel is a potential culprit. Threads within a warp that diverge on a conditional branch execute serially. Observing the instruction mix within the execution metrics of the selected kernel instance within Nsight Systems can further confirm this. We can therefore adjust our implementation to reduce branch divergence. Nsight Systems makes this type of analysis much easier, as we can see both the global timeline and the low-level instruction stream.

When performing this type of analysis, the primary goal is to understand the execution flow of the custom kernel within the overall application, how the different parts of the kernel interact with the memory hierarchy, and the resource utilization of the GPU hardware. Nsight Systems helps correlate these different aspects, thus allowing the user to make informed decisions about code optimization.

For further study, consult resources such as the NVIDIA CUDA documentation, focusing on memory access patterns and kernel optimization. Additionally, resources covering GPU microarchitecture, particularly the specific architecture of the targeted GPUs, can aid in interpreting the detailed performance metrics captured by Nsight Systems. Understanding occupancy limitations and the impact of memory access patterns will be key to effective CUDA code development and analysis. Finally, tutorials and examples related to performance analysis with Nsight Systems are very valuable in gaining a comprehensive grasp on the functionality of the tool. This combination of theoretical understanding of CUDA programming and practical use of tools like Nsight Systems will be essential in developing highly optimized custom CUDA kernels.
