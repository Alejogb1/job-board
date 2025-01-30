---
title: "How can CUDA kernels be executed sequentially without CPU intervention?"
date: "2025-01-30"
id: "how-can-cuda-kernels-be-executed-sequentially-without"
---
The inherent parallelism of CUDA often overshadows the need for sequential kernel execution, but specific scenarios, like phased computations or dependency management within a single GPU operation, necessitate such a pattern. While the common approach involves CPU intervention to launch each kernel, a more efficient, purely GPU-driven method exists using CUDA graphs and cooperative groups. I've personally employed this technique in a particle simulation where intermediate force calculations needed to be completed before integrating velocities, all within the GPU's scope. The central challenge is structuring the workload so that data dependencies are respected without the CPU's scheduling and synchronization overhead.

The problem stems from CUDA’s default model: a host thread (on the CPU) initiates kernels via the `cudaLaunchKernel` function. This command is inherently blocking on the CPU's side, at least to some degree. To achieve sequential kernel execution purely on the GPU, we leverage CUDA graphs. A CUDA graph defines a directed acyclic graph (DAG) of operations, including kernel launches, memory copies, and event recordings, that can be executed en bloc. The graph, once created, resides entirely on the GPU, removing the CPU bottleneck during execution. The mechanism allows for the specification of dependencies: a kernel's execution can be gated by the completion of a prior kernel (or other operation).

Cooperative groups provide the necessary constructs to programmatically establish these dependencies *within* the kernels themselves, further solidifying the GPU-driven execution pattern. This reduces overhead compared to manually setting the dependencies in the graph. When a kernel participating in a cooperative group completes, it can signal another kernel, scheduled downstream, to begin its execution. This enables a form of intra-GPU task management controlled entirely by the device, all without returning to the host to launch kernels individually. The critical element here is that the scheduler (or dependency checker) resides on the GPU rather than the CPU. The GPU manages its own task completion signals.

Let's illustrate with a concrete example. Consider a simplified image processing pipeline where we have two kernels: a blurring kernel and a sharpening kernel. Conventionally, the CPU would launch the blurring kernel, wait for completion, then launch the sharpening kernel. Using a CUDA graph with cooperative groups, the dependency is handled within the GPU.

First, let's look at the code for the two kernels.

```cpp
// Kernel 1: Blurring
__global__ void blur_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Simplified blur using a 3x3 kernel
        float sum = 0.0f;
        int count = 0;
        for(int i = -1; i <= 1; ++i) {
            for (int j = -1; j <=1; ++j){
                int nx = x + i;
                int ny = y + j;
                if(nx >= 0 && nx < width && ny >= 0 && ny < height){
                   sum += input[ny * width + nx];
                   count++;
                }
            }
         }
        output[y*width+x] = sum / count;
    }
    // Signaling done
    if (x == 0 && y == 0){
       cuda::thread_group group = cuda::this_thread_block_group();
       cuda::sync(group);
       group.arrive();
    }
}
// Kernel 2: Sharpening
__global__ void sharpen_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Simple sharpen
        output[y*width+x] = 2.0f * input[y*width+x] -  input[y*width+x-1] - input[y*width+x+1];
    }
    // Signaling done
   if(x==0 && y==0)
    {
       cuda::thread_group group = cuda::this_thread_block_group();
       cuda::sync(group);
       group.arrive();
   }
}
```

In these kernels, each thread computes a pixel, and the first thread of the block group is responsible for signaling the group with `cuda::sync(group)` and the `group.arrive()` function to mark completion. The first thread is also used because it is guaranteed to be available within a block and it is deterministic, as opposed to race conditions if any random thread was used to signal.

The next code section shows how to set up a CUDA graph to execute them sequentially.

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Define dimensions
const int width = 512;
const int height = 512;

int main() {
    // Allocate memory
    float* d_input, *d_temp, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_temp, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));
    // Initialize input data (omitted for brevity)
    float* h_input = new float[width * height];
    for(int i=0; i < width * height; ++i) h_input[i] = i;
    cudaMemcpy(d_input, h_input, width*height * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_input;

    // Define the grid and block sizes
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create CUDA graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Create a graph node for the blur kernel
    cudaGraphNode_t blur_node;
    cudaKernelNodeParams blur_params = {0};
    blur_params.func = (void*)blur_kernel;
    blur_params.gridDim = grid_dim;
    blur_params.blockDim = block_dim;
    blur_params.sharedMemBytes = 0;
    void* blur_args[] = {&d_input, &d_temp, &width, &height};
    blur_params.kernelParams = blur_args;
    cudaGraphAddKernelNode(&blur_node, graph, nullptr, 0, &blur_params);

    // Create a graph node for the sharpen kernel
    cudaGraphNode_t sharpen_node;
    cudaKernelNodeParams sharpen_params = {0};
    sharpen_params.func = (void*)sharpen_kernel;
    sharpen_params.gridDim = grid_dim;
    sharpen_params.blockDim = block_dim;
    sharpen_params.sharedMemBytes = 0;
    void* sharpen_args[] = {&d_temp, &d_output, &width, &height};
    sharpen_params.kernelParams = sharpen_args;
    cudaGraphAddKernelNode(&sharpen_node, graph, &blur_node, 1, &sharpen_params); // Dependency

    // Instantiate the graph
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, 0);
    
    // Launch the graph
    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);

    // Copy the output back to the host (omitted for brevity)
    float* h_output = new float[width*height];
    cudaMemcpy(h_output, d_output, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}
```

Here, `cudaGraphAddKernelNode` constructs our DAG. The sharpen node is given a dependency on the blur node (`&blur_node, 1`). The graph can then be launched entirely on the GPU with `cudaGraphLaunch`. This is the key part, the CPU is not actively managing the kernel execution, the CUDA graph does that for us. Inside the kernels, cooperative groups (using `cuda::thread_group`) are used to ensure all threads within the block complete before signaling the group's completion.

To add a further level of control, it is possible to incorporate CUDA events into the graph. These are also GPU-resident signaling mechanisms that can be included as nodes in the graph to demarcate stages of execution. For example, an event can be inserted between the blur and sharpen kernels. The next kernel can be scheduled to start when this event is signaled, adding another layer of synchronization, but all done via GPU instructions.

```cpp
 // Create a CUDA event
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Create an event record node
    cudaGraphNode_t event_record_node;
    cudaGraphAddEventRecordNode(&event_record_node, graph, &blur_node, 1, event);

    // Create a graph node for the sharpen kernel which depends on the event
    cudaGraphNode_t sharpen_node;
    cudaKernelNodeParams sharpen_params = {0};
    sharpen_params.func = (void*)sharpen_kernel;
    sharpen_params.gridDim = grid_dim;
    sharpen_params.blockDim = block_dim;
    sharpen_params.sharedMemBytes = 0;
    void* sharpen_args[] = {&d_temp, &d_output, &width, &height};
    sharpen_params.kernelParams = sharpen_args;
    cudaGraphAddKernelNode(&sharpen_node, graph, &event_record_node, 1, &sharpen_params);
  
    // Create an event wait node (to ensure the sharpen doesn't start before the event)
    cudaGraphNode_t event_wait_node;
    cudaGraphAddEventWaitNode(&event_wait_node, graph, &sharpen_node, 1, event);
```

In this case, the `sharpen_kernel` is scheduled after the blur kernel and the `event` has completed. There are several benefits to using this approach. First, the CPU is freed to perform other tasks, like application logic or data preparation, as the GPU proceeds with the kernel executions. Second, there is reduced latency, as the CPU is not required for each kernel launch, so this avoids the overhead of each invocation via the host interface. This GPU-centric approach enables a deeper level of efficiency and can contribute to better overall performance.

For anyone looking to delve further into these techniques, the official CUDA documentation is essential, particularly the sections on CUDA graphs and cooperative groups. The “CUDA C++ Programming Guide” provides a comprehensive overview and the "CUDA Toolkit API Reference Manual" is crucial for a lower-level understanding. Several online tutorials and example repositories by NVIDIA and the community are also available. I recommend exploring these resources to grasp the more nuanced aspects of implementing these methods effectively, which include topics like different CUDA graph configurations, cooperative group types, and the specific limitations of different CUDA architectures.

This level of control over kernel execution without the CPU acting as an intermediary highlights the power of the CUDA programming model and provides a way to optimize computations with dependencies entirely on the GPU. This is a critical technique for many performance-demanding applications.
