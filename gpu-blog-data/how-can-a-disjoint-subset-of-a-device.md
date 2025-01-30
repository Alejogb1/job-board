---
title: "How can a disjoint subset of a device array be asynchronously copied to the host using CUDA and Thrust?"
date: "2025-01-30"
id: "how-can-a-disjoint-subset-of-a-device"
---
The efficient transfer of data between device and host memory in CUDA applications is often a critical performance bottleneck, and this becomes more challenging when only specific subsets of device data are needed. Rather than copying the entire array, employing asynchronous memory transfers with Thrust and careful indexing can significantly reduce transfer overhead and enable concurrent computation.

A direct copy operation using `cudaMemcpy` would be the naive approach; however, that operation is synchronous and blocks the host thread until the transfer completes. Furthermore, copying the entire array when only a subset is needed wastes time and bandwidth. Asynchronous transfers, specifically with streams, offer overlap between computation and data transfer, enhancing resource utilization. To achieve this, we must also utilize specific functions that work with streams. Thrust, while primarily designed for device-side computations, can be used to generate index sets and manage device memory, which complements the efficient CUDA asynchronous copy calls.

The core concept involves generating indices that correspond to the desired subset within the device array. These indices are then used to gather the corresponding elements into a contiguous device memory region. This contiguous region is then copied to the host asynchronously using a dedicated stream. The final step on the host side involves processing this contiguous region.

Here are three examples, each demonstrating different approaches and complexities:

**Example 1: Static Index List**

This example uses a predefined array of indices on the host. We transfer this index array to the device, use Thrust to access these indices, copy them to a contiguous temporary array, and transfer the temporary array to the host.

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

int main() {
    // 1. Setup
    const int arraySize = 100;
    const int subsetSize = 10;

    // Host indices
    thrust::host_vector<int> hostIndices(subsetSize);
    for (int i=0; i < subsetSize; ++i){
      hostIndices[i] = i * 10;
    }

    // Device data
    thrust::device_vector<int> deviceData(arraySize);
    for(int i=0; i < arraySize; ++i){
        deviceData[i] = i;
    }

    // Device indices
    thrust::device_vector<int> deviceIndices = hostIndices;
    
    // Device result
    thrust::device_vector<int> deviceSubset(subsetSize);

    // Host result
    thrust::host_vector<int> hostSubset(subsetSize);

    // CUDA Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 2. Device Copy
    thrust::gather(thrust::device, deviceIndices.begin(), deviceIndices.end(), deviceData.begin(), deviceSubset.begin());
    
    // 3. Asynchronous transfer to host
    CUDA_CHECK(cudaMemcpyAsync(hostSubset.data(), deviceSubset.data(), subsetSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // 4. Synchronize the stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5. Print result
    std::cout << "Copied subset:\n";
    for (int val : hostSubset) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
```

In this example, the host-defined index list is used by `thrust::gather`. This gathers the elements at the corresponding indices into a separate device memory buffer (`deviceSubset`). The contents of `deviceSubset` are then copied to `hostSubset` asynchronously using the established CUDA stream. A `cudaStreamSynchronize` call is crucial; otherwise, the host would proceed to print results before the data transfer completes, resulting in unpredictable behavior. This example shows a simple, yet effective, way to accomplish the desired data transfer.

**Example 2: Dynamic Index Generation on Device**

This example illustrates how to generate the desired subset indices directly on the device using Thrust, rather than transferring them from the host. This approach avoids extra transfers. Here, we generate an arithmetic sequence as the subset indices.

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <iostream>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

int main() {
    // 1. Setup
    const int arraySize = 100;
    const int subsetSize = 10;

    // Device data
    thrust::device_vector<int> deviceData(arraySize);
    for(int i=0; i < arraySize; ++i){
        deviceData[i] = i;
    }

    // Device indices
    thrust::device_vector<int> deviceIndices(subsetSize);
    thrust::sequence(thrust::device, deviceIndices.begin(), deviceIndices.end(), 0);

    // Calculate indices
    thrust::transform(thrust::device, deviceIndices.begin(), deviceIndices.end(), deviceIndices.begin(), 
                      thrust::placeholders::_1 * 10);
    
    // Device result
    thrust::device_vector<int> deviceSubset(subsetSize);

    // Host result
    thrust::host_vector<int> hostSubset(subsetSize);

    // CUDA Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 2. Device Copy
    thrust::gather(thrust::device, deviceIndices.begin(), deviceIndices.end(), deviceData.begin(), deviceSubset.begin());
    
    // 3. Asynchronous transfer to host
    CUDA_CHECK(cudaMemcpyAsync(hostSubset.data(), deviceSubset.data(), subsetSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // 4. Synchronize the stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5. Print result
    std::cout << "Copied subset:\n";
    for (int val : hostSubset) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
```

Here, we use `thrust::sequence` to create a sequence of integers (0, 1, 2, ...). Then we use `thrust::transform` to multiply each element by 10, creating the desired subset indices on the device. This is very efficient since it avoids host-to-device data transfers, as the operations are entirely device-side.

**Example 3: Conditional Selection and Asynchronous Transfer**

This example demonstrates how to select elements based on a condition, using a boolean mask generated with Thrust, copying these selected elements, and asynchronously transferring them. This is a more complex use case.

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <iostream>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

int main() {
    // 1. Setup
    const int arraySize = 100;

    // Device data
    thrust::device_vector<int> deviceData(arraySize);
    for(int i=0; i < arraySize; ++i){
        deviceData[i] = i;
    }

    // Device mask (select even numbers)
    thrust::device_vector<bool> deviceMask(arraySize);
    thrust::transform(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(arraySize), deviceMask.begin(), [](int i) { return (i % 2) == 0; });

    // Use `reduce_by_key` to compact
    thrust::device_vector<int> indices(arraySize);
    thrust::sequence(thrust::device, indices.begin(), indices.end(), 0);
    
    thrust::device_vector<int> compacted_indices(arraySize);
    auto new_end = thrust::copy_if(thrust::device, indices.begin(), indices.end(), deviceMask.begin(), compacted_indices.begin()).first;
    
    size_t subsetSize = new_end - compacted_indices.begin();
    compacted_indices.resize(subsetSize);

     // Device result
    thrust::device_vector<int> deviceSubset(subsetSize);
    
    // Host result
    thrust::host_vector<int> hostSubset(subsetSize);

    // CUDA Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 2. Device Copy
    thrust::gather(thrust::device, compacted_indices.begin(), compacted_indices.end(), deviceData.begin(), deviceSubset.begin());
    
    // 3. Asynchronous transfer to host
    CUDA_CHECK(cudaMemcpyAsync(hostSubset.data(), deviceSubset.data(), subsetSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // 4. Synchronize the stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5. Print result
    std::cout << "Copied subset (even numbers):\n";
    for (int val : hostSubset) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
```

In this more sophisticated approach, a boolean mask (deviceMask) is created, indicating which elements to select (in this case, even numbers).  `thrust::copy_if` then compacts the indices based on the mask. This creates a compacted list of indices that corresponds to all elements that satisfy the condition. This dynamically generated compact index list is then used to gather from the original device data into a compacted device subset, which is then asynchronously transferred to the host. This showcases a more complex subset selection logic.

These examples highlight fundamental strategies for performing asynchronous transfers of disjoint subsets. Thrust provides powerful device-side primitives that can be utilized effectively with CUDA's stream-based asynchronous transfers.

For continued learning, I suggest studying the CUDA documentation, particularly sections on memory management and streams. Thrust documentation is also invaluable for understanding the execution policies and algorithms. Texts on parallel programming with CUDA can provide broader context and deeper insights. Furthermore, exploring publicly available code samples and engaging with the CUDA community on forums can enhance practical understanding and troubleshooting skills. By employing asynchronous techniques thoughtfully, one can achieve significant performance gains in CUDA applications that deal with large datasets.
