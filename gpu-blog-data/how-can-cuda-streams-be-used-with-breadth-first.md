---
title: "How can CUDA streams be used with breadth-first processing for kernel execution and device-to-host transfers?"
date: "2025-01-30"
id: "how-can-cuda-streams-be-used-with-breadth-first"
---
Employing CUDA streams to manage breadth-first processing and data transfers demands careful attention to concurrency and synchronization. My experience developing a high-performance pathfinding algorithm for a large-scale simulation platform highlighted the critical importance of this technique, specifically when dealing with wavefront propagation. Improper stream usage leads to severe performance bottlenecks, defeating the purpose of using a GPU in the first place.

The core idea behind breadth-first processing is to examine all nodes at a given level before moving onto the next. When mapped to a GPU, each level constitutes a batch of work that can be parallelized, potentially employing multiple kernels. Naively launching these kernels and initiating data transfers sequentially will serialize the process, negating the benefits of the GPU's parallel architecture. CUDA streams offer a solution by encapsulating these operations, enabling concurrent execution. Each stream can operate independently, scheduling kernel launches and data transfers. The crucial aspect is to ensure that operations within a stream adhere to data dependencies while operations across streams can occur concurrently.

Implementing a breadth-first traversal involves launching a kernel for processing a specific level, potentially transferring data to the host, and preparing data for the next level. The goal is to overlap these three phases. This is achieved through these steps: first, creating multiple streams. Then, for each level, allocate a separate stream to execute the processing kernel, which will modify the graph data in device memory. This kernel output data determines the nodes of the next level, and data must be transferred if the processing requires examination of the results on the CPU. Finally, data is prepared for the next level's processing, a potentially costly step, and the process can be repeated.

Consider the following example illustrating this process with a conceptual breadth-first search (BFS):

```cpp
// Example 1: Simple BFS implementation with streams

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Assume device functions (e.g., bfs_kernel) are defined elsewhere.
extern void bfs_kernel(int* current_level, int* next_level, int num_nodes, int* adjacency_list, int num_edges);

int main() {
    int num_levels = 5; // Number of breadth-first levels to process
    int num_nodes = 1024; // Example number of nodes
    int num_edges = 2048; // Example number of edges (placeholder)
    std::vector<int> adjacency_list(num_edges); // In a real app, this is created on the host
    std::vector<int> initial_level(num_nodes, 0); // Node 0 starts the search

    // Allocate memory on the device
    int *d_current_level, *d_next_level, *d_adjacency_list;
    cudaMalloc(&d_current_level, num_nodes * sizeof(int));
    cudaMalloc(&d_next_level, num_nodes * sizeof(int));
    cudaMalloc(&d_adjacency_list, num_edges * sizeof(int));

    // Create streams
    cudaStream_t streams[num_levels];
    for (int i = 0; i < num_levels; ++i) {
        cudaStreamCreate(&streams[i]);
    }
   
    // Copy the adjacency list to the device (once)
    cudaMemcpy(d_adjacency_list, adjacency_list.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int* h_current_level = initial_level.data();
    //Breadth First Traversal
    for (int level = 0; level < num_levels; ++level) {
        // Copy the current level to the device
        cudaMemcpyAsync(d_current_level, h_current_level, num_nodes * sizeof(int), cudaMemcpyHostToDevice, streams[level]);
        // Execute the BFS kernel on the current level
        bfs_kernel(d_current_level, d_next_level, num_nodes, d_adjacency_list, num_edges);
        // Asynchronously copy the next level back to the host
        cudaMemcpyAsync(h_current_level, d_next_level, num_nodes * sizeof(int), cudaMemcpyDeviceToHost, streams[level]);

        //The level here should now contain the results of the previous level.
        cudaStreamSynchronize(streams[level]); // Wait for the current stream to finish before proceeding
        // In real implementation, you would need to implement additional data processing for the next level
     }
    // Destroy streams
    for (int i = 0; i < num_levels; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Free device memory
    cudaFree(d_current_level);
    cudaFree(d_next_level);
    cudaFree(d_adjacency_list);
    return 0;
}
```

In Example 1, `streams` array holds CUDA stream objects. For each BFS level, I perform a host-to-device copy of data representing the current level, followed by the kernel launch, and finally, a device-to-host copy of the next level's data. These operations are encapsulated within a single stream for each level. Critically, I synchronize each stream after copying back to the host before moving to the next level; this ensures that we've received the data for level `n` before processing data for level `n+1`. This allows overlapping of computation and data transfers for each level, but the levels themselves are processed sequentially. To enhance this, consider processing multiple levels in parallel. This adds another layer of complexity, but can drastically reduce execution time. The key is managing dependencies between levels, and potentially requires a staging buffer for copying intermediate results.

This introduces dependencies between levels if intermediate processing happens on the host. To resolve this, one might consider performing multiple levels in parallel.

```cpp
// Example 2: Parallel BFS on multiple streams with limited overlap of data processing

// Assume device functions (e.g., bfs_kernel) are defined elsewhere.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern void bfs_kernel(int* current_level, int* next_level, int num_nodes, int* adjacency_list, int num_edges);

int main() {
    int num_levels = 5;
    int num_nodes = 1024;
    int num_edges = 2048;
    std::vector<int> adjacency_list(num_edges);
    std::vector<int> initial_level(num_nodes, 0);

     // Allocate memory on the device
    int *d_current_level, *d_next_level, *d_adjacency_list;
    cudaMalloc(&d_current_level, num_nodes * sizeof(int));
    cudaMalloc(&d_next_level, num_nodes * sizeof(int));
    cudaMalloc(&d_adjacency_list, num_edges * sizeof(int));

    // Create streams, ideally more streams for parallelism than levels.
    const int num_streams = 3;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Copy the adjacency list to the device (once)
    cudaMemcpy(d_adjacency_list, adjacency_list.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> h_current_level = initial_level;

    for (int level = 0; level < num_levels; ++level) {
    
      int stream_idx = level % num_streams; // Cycle through streams

      // Copy current level data to the device asynchronously
      cudaMemcpyAsync(d_current_level, h_current_level.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice, streams[stream_idx]);

      // Execute the BFS kernel on the device asynchronously
      bfs_kernel(d_current_level, d_next_level, num_nodes, d_adjacency_list, num_edges);

       // Asynchronously copy the next level back to the host
      std::vector<int> h_next_level(num_nodes);
      cudaMemcpyAsync(h_next_level.data(), d_next_level, num_nodes * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_idx]);

      // Synchronize stream for correctness
      cudaStreamSynchronize(streams[stream_idx]);

      // Set the new level and process on the host
      h_current_level = h_next_level;
    }

    // Destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Free device memory
    cudaFree(d_current_level);
    cudaFree(d_next_level);
    cudaFree(d_adjacency_list);
    return 0;
}

```

In Example 2, I've introduced multiple streams to manage concurrency of levels. Each level of the BFS is assigned to a stream using a modulo operation. As a result, kernels for levels 0, 3, 6,... will execute in parallel, as will levels 1, 4, 7,... and 2, 5, 8,... The level data is also processed asynchronously. It is critical to synchronize the stream before proceeding to the next stage of processing on the host, especially if the subsequent processing depends on the result.

In both of the previous examples, there is a synchronization with the host, a potentially expensive operation. In many cases, the preparation for the next level can be accomplished on the device, avoiding the need to copy the intermediate level data back to the host. Example 3 demonstrates this approach, keeping all data transfer and computation solely on the device.

```cpp
// Example 3: Device-side BFS with stream pipelining

// Assume device functions (e.g., bfs_kernel, prepare_next_level) are defined elsewhere.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern void bfs_kernel(int* current_level, int* next_level, int num_nodes, int* adjacency_list, int num_edges);
extern void prepare_next_level(int* next_level, int* next_next_level, int num_nodes);

int main() {
    int num_levels = 5;
    int num_nodes = 1024;
    int num_edges = 2048;
    std::vector<int> adjacency_list(num_edges);
    std::vector<int> initial_level(num_nodes, 0);

    // Allocate memory on the device
    int *d_current_level, *d_next_level, *d_next_next_level, *d_adjacency_list;
    cudaMalloc(&d_current_level, num_nodes * sizeof(int));
    cudaMalloc(&d_next_level, num_nodes * sizeof(int));
    cudaMalloc(&d_next_next_level, num_nodes * sizeof(int));
    cudaMalloc(&d_adjacency_list, num_edges * sizeof(int));
    
    // Create streams
    const int num_streams = 3;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
     // Copy the adjacency list to the device (once)
    cudaMemcpy(d_adjacency_list, adjacency_list.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);


    cudaMemcpy(d_current_level, initial_level.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice); //Initialize first level

    for (int level = 0; level < num_levels; ++level) {
        int stream_idx = level % num_streams;
        
        // Kernel to compute next BFS level
        bfs_kernel(d_current_level, d_next_level, num_nodes, d_adjacency_list, num_edges);
        
        // Kernel to prepare the data for next level, writing into next_next_level buffer.
        prepare_next_level(d_next_level, d_next_next_level, num_nodes);

        // Swap the buffers
        std::swap(d_current_level, d_next_next_level);
        
     }

    // Copy result back to host
    std::vector<int> h_final_level(num_nodes);
    cudaMemcpy(h_final_level.data(), d_current_level, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    // Destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Free device memory
    cudaFree(d_current_level);
    cudaFree(d_next_level);
    cudaFree(d_next_next_level);
    cudaFree(d_adjacency_list);

    return 0;
}

```

In Example 3, intermediate results are kept on the device, eliminating redundant host-device transfers for each level. The results of the current level are written into one device memory buffer, which then becomes the input for the subsequent level. `prepare_next_level` performs the necessary operations, such as determining the next level's active nodes. Stream pipelining allows the kernel for level `n` to execute concurrently with data preparation for level `n+1`, increasing overall throughput. The final result is only copied back to the host at the very end of the traversal.

For further study, I recommend consulting the NVIDIA CUDA documentation, which provides a detailed guide to stream programming. Additionally, the CUDA samples included with the CUDA Toolkit are a good resource. The book "CUDA by Example" offers practical guidance on CUDA programming. These resources should help in understanding the nuances of CUDA stream usage and optimizing applications for parallel processing. Finally, exploring examples of similar algorithms like graph traversal in research papers focused on high-performance GPU computing provides valuable insights.
