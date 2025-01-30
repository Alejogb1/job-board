---
title: "How can CUDA streams be paused and resumed?"
date: "2025-01-30"
id: "how-can-cuda-streams-be-paused-and-resumed"
---
CUDA streams, fundamental for asynchronous task execution, do not inherently support pausing and resuming in the way a typical thread might. The very nature of a stream is to represent an ordered sequence of operations that a GPU should perform; halting this sequence mid-execution without risking data corruption or unpredictable behavior requires careful manipulation of synchronization primitives and, potentially, creating the illusion of pausing through judicious use of events and separate, dependent streams. My direct experience optimizing dense linear algebra kernels on NVIDIA Tesla V100 GPUs has underscored the importance of understanding these nuances, particularly when dealing with real-time data processing pipelines.

The core issue stems from the fact that a CUDA stream, once a kernel or memory operation is submitted, is effectively a fire-and-forget command. The host CPU relinquishes control of that operation’s execution to the GPU's scheduler. There is no API call to simply “pause” the stream midway. What *is* feasible, and what constitutes a pause/resume pattern in practice, involves introducing *events* into the stream and subsequently creating *dependent streams*. These dependent streams then wait on those events to signal that the preceding tasks in the initial stream have reached a designated point. This approach allows for control over the timing and execution order of operations, effectively creating staged execution.

Let's clarify. We are not directly pausing a stream. We are instead demarcating points within it using events, which are lightweight synchronization objects, and then arranging for subsequent work in other streams to be contingent on those events. This achieves the effect of controlling the rate of progress in the overall execution path and provides a surrogate pause-and-resume function.

Here’s a breakdown of how this is accomplished, accompanied by code examples:

**Example 1: Basic Event-Based Synchronization**

This example illustrates a fundamental synchronization pattern. We’ll submit two kernels onto two separate streams. The second kernel will only execute after the first kernel signals the event.

```c++
#include <cuda_runtime.h>
#include <iostream>

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}


int main() {
  cudaStream_t stream1, stream2;
  cudaEvent_t event;

  checkCudaError(cudaStreamCreate(&stream1));
  checkCudaError(cudaStreamCreate(&stream2));
  checkCudaError(cudaEventCreate(&event));

  // Dummy kernels (replaced by actual work)
  auto kernel1 = [](float* d_out, size_t size){
        for (size_t i = 0; i < size; ++i) d_out[i] = i*1.0f;
  };

  auto kernel2 = [](float* d_out, size_t size){
        for (size_t i = 0; i < size; ++i) d_out[i] = i*2.0f;
  };
  
    size_t size = 1024;
    float* d_data1, *d_data2;
    cudaMalloc(&d_data1, size * sizeof(float));
    cudaMalloc(&d_data2, size * sizeof(float));


  // Submit work to the first stream
  
  kernel1<<<128, 128,0,stream1>>>(d_data1, size);
  checkCudaError(cudaGetLastError());

  // Record an event
  checkCudaError(cudaEventRecord(event, stream1));

  // Submit work to the second stream, which waits on the event
  checkCudaError(cudaStreamWaitEvent(stream2, event, 0));
    
  kernel2<<<128, 128,0,stream2>>>(d_data2, size);
  checkCudaError(cudaGetLastError());

  // Synchronize to see if operations have completed
  checkCudaError(cudaStreamSynchronize(stream2));

    cudaFree(d_data1);
    cudaFree(d_data2);
  checkCudaError(cudaEventDestroy(event));
  checkCudaError(cudaStreamDestroy(stream1));
  checkCudaError(cudaStreamDestroy(stream2));
  return 0;
}
```

In this example, `cudaEventRecord` marks a specific point in `stream1`. `cudaStreamWaitEvent` on `stream2` ensures that any operations submitted to `stream2` will not begin until the event has been recorded. This establishes the dependency. Effectively, the operations in `stream2` are ‘paused’ until `stream1` completes its tasks up to the event.

**Example 2: Pausing and Resuming within a Data Pipeline**

This example simulates a data pipeline where data must be preprocessed on the GPU before another kernel can consume it. We use events to allow for other operations to occur on the CPU while data is being preprocessed.

```c++
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  cudaStream_t preprocess_stream, compute_stream;
  cudaEvent_t preprocess_done;
  
  checkCudaError(cudaStreamCreate(&preprocess_stream));
  checkCudaError(cudaStreamCreate(&compute_stream));
  checkCudaError(cudaEventCreate(&preprocess_done));

  // Dummy kernels for preprocessing and computation
  auto preprocess_kernel = [](float* d_in, float* d_out, size_t size){
    for (size_t i=0; i < size; i++){
      d_out[i] = d_in[i] * 0.5f;
    }
  };

    auto compute_kernel = [](float* d_in, float* d_out, size_t size){
        for (size_t i=0; i < size; i++){
            d_out[i] = d_in[i] + 1.0f;
        }
    };
  
  size_t size = 1024;
  float* d_input, *d_processed, *d_result;
  float* h_input = new float[size];
    for (size_t i = 0; i < size; i++) h_input[i] = static_cast<float>(i);
    

  checkCudaError(cudaMalloc(&d_input, size * sizeof(float)));
  checkCudaError(cudaMalloc(&d_processed, size * sizeof(float)));
  checkCudaError(cudaMalloc(&d_result, size * sizeof(float)));

  // Copy input data to GPU
  checkCudaError(cudaMemcpyAsync(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice, preprocess_stream));


  // Submit preprocessing kernel and record event
  preprocess_kernel<<<128, 128,0, preprocess_stream>>>(d_input, d_processed, size);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaEventRecord(preprocess_done, preprocess_stream));


  // Simulate CPU work while preprocessing occurs asynchronously
  std::cout << "CPU doing some work while preprocess on GPU is underway..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Wait for preprocessing to complete and submit computation kernel on a separate stream
  checkCudaError(cudaStreamWaitEvent(compute_stream, preprocess_done, 0));
    compute_kernel<<<128, 128,0, compute_stream>>>(d_processed, d_result, size);
    checkCudaError(cudaGetLastError());
  

  // Synchronize and cleanup
  checkCudaError(cudaStreamSynchronize(compute_stream));
    
  cudaFree(d_input);
  cudaFree(d_processed);
    cudaFree(d_result);
    delete[] h_input;
  checkCudaError(cudaEventDestroy(preprocess_done));
  checkCudaError(cudaStreamDestroy(preprocess_stream));
  checkCudaError(cudaStreamDestroy(compute_stream));
  return 0;
}
```

Here, we simulate a pipeline. We asynchronously transfer data to the GPU for preprocessing, recording an event when the preprocessing stage is completed.  While the preprocessing is ongoing, the CPU simulates doing other tasks. Only after `preprocess_done` is signaled does the dependent `compute_stream` execute its tasks. This shows how an event can create a well-defined 'pause point', allowing us to control the pipeline's flow and schedule work to other streams that depend on prior task completion.

**Example 3: Conditional Execution Based on Data Availability**

Consider a scenario where computations depend on external data fetched by a separate module. We can use CUDA events to trigger the computation stream only when the data is available. This example demonstrates that control over when to begin subsequent operations can be highly conditional.

```c++
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}


int main() {
  cudaStream_t data_fetch_stream, compute_stream;
  cudaEvent_t data_ready;
  std::atomic<bool> data_is_ready(false);


  checkCudaError(cudaStreamCreate(&data_fetch_stream));
  checkCudaError(cudaStreamCreate(&compute_stream));
  checkCudaError(cudaEventCreate(&data_ready));

  // Dummy data processing kernel that depends on data
    auto compute_kernel = [](float* d_in, float* d_out, size_t size){
        for (size_t i = 0; i < size; i++) d_out[i] = d_in[i] * 3.0f;
    };
    
    size_t size = 1024;
    float* d_input, *d_result;
  checkCudaError(cudaMalloc(&d_input, size * sizeof(float)));
  checkCudaError(cudaMalloc(&d_result, size * sizeof(float)));


  // Separate thread simulates external data acquisition
  std::thread data_fetch_thread([&](){
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Simulate some work
    float* h_data = new float[size];
        for(size_t i = 0; i < size; i++) h_data[i] = static_cast<float>(i);

    checkCudaError(cudaMemcpyAsync(d_input, h_data, size * sizeof(float), cudaMemcpyHostToDevice, data_fetch_stream));
        delete[] h_data;
    data_is_ready = true;

    checkCudaError(cudaEventRecord(data_ready, data_fetch_stream));
  });
  
  // Computation only happens once data is available

  if (data_is_ready){
      checkCudaError(cudaStreamWaitEvent(compute_stream, data_ready, 0));
        compute_kernel<<<128, 128, 0, compute_stream>>>(d_input, d_result, size);
        checkCudaError(cudaGetLastError());

  } else {
    std::cout << "Data not yet ready... skipping computation" << std::endl;
  }

    

  // Join thread and cleanup
  data_fetch_thread.join();
  checkCudaError(cudaStreamSynchronize(compute_stream));
    cudaFree(d_input);
    cudaFree(d_result);
  checkCudaError(cudaEventDestroy(data_ready));
  checkCudaError(cudaStreamDestroy(data_fetch_stream));
  checkCudaError(cudaStreamDestroy(compute_stream));
  return 0;
}
```

In this third example, we introduce a thread simulating external data fetching. The `compute_stream` will only begin executing its kernel *if* the external data thread sets `data_is_ready` and records the event. This illustrates a conditional 'pause-resume' mechanism based on data availability, a typical scenario in many real-world applications.

To reiterate, CUDA streams cannot be paused directly. The practice involves using events to mark specific points within a stream's execution and then making other streams dependent on those events. This creates the illusion of pausing and resuming, allows for controlled execution and enables task dependencies to be handled effectively within the asynchronous model of CUDA programming.

For further exploration, I recommend consulting the NVIDIA CUDA documentation focusing on stream management, event handling, and asynchronous execution. Specifically, examine the descriptions of `cudaStreamCreate`, `cudaEventCreate`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaStreamSynchronize` and `cudaMemcpyAsync`. Also, the "CUDA Programming Guide" and the "Best Practices Guide" from NVIDIA will provide invaluable insights into advanced stream techniques. Learning about dependency graphs and asynchronous pipeline design is crucial for proficiently exploiting CUDA.
