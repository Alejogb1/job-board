---
title: "How can CUDA arrays be checked for and repaired if duplicates exist?"
date: "2025-01-30"
id: "how-can-cuda-arrays-be-checked-for-and"
---
CUDA, fundamentally designed for parallel computation, often relies on data structures that assume uniqueness or, at the very least, require consistent indices. The presence of duplicate entries in a CUDA array can introduce subtle errors, leading to incorrect results or program crashes, especially when used in conjunction with atomic operations or where data integrity is paramount. Detecting and correcting these duplicates within the parallel processing context of CUDA necessitates careful consideration of performance overhead and memory management.

My experience working on a large-scale Monte Carlo simulation for particle physics highlighted the practical importance of this issue. We utilized CUDA arrays to store particle collision data. Initial data generation sometimes, due to unforeseen numerical instabilities, produced duplicate particle IDs, resulting in erroneous calculations. This led to the development of specific strategies for duplicate detection and repair, which are the foundation of my current approach.

The primary challenge in addressing duplicate entries on the GPU stems from the nature of its parallel architecture. A sequential scan, commonly used in CPU-based duplicate detection, becomes incredibly inefficient. We instead need to leverage CUDA's inherent parallelism to achieve acceptable processing times. One practical strategy revolves around using a combination of sorting and adjacent comparison. The first step, sorting, brings all identical elements into contiguous locations, making it easier to identify duplicates. Post-sorting, a comparison step identifies and flags or eliminates these duplicates.

Consider a simple scenario: you have an array of unsigned integers representing IDs where some IDs might be repeated. The following CUDA code provides an example of how this can be achieved using Thrust's sorting and adjacent difference functionality:

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <iostream>

// Example 1: Simple duplicate detection and removal

void removeDuplicates(thrust::device_vector<unsigned int>& d_data) {
    // 1. Sort the array
    thrust::sort(thrust::execution::par, d_data.begin(), d_data.end());

    // 2. Remove consecutive duplicates
    auto new_end = thrust::unique(thrust::execution::par, d_data.begin(), d_data.end());
	
	// 3. Resize the array
	d_data.resize(new_end - d_data.begin());
}


int main() {
    // Example usage:
    thrust::device_vector<unsigned int> d_data = {1, 5, 2, 2, 8, 5, 9, 1, 4};
    
    std::cout << "Original Data: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;
    

    removeDuplicates(d_data);
    
    std::cout << "Data without duplicates: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

In this example, the `removeDuplicates` function utilizes Thrust’s `sort` algorithm to arrange the array elements in ascending order. Then, `unique`, an algorithm that preserves order while collapsing consecutive duplicates, effectively compacts the vector by returning a pointer to the end of the unique range. Finally the vector is resized to contain only the unique values. Note that no memory is allocated or deallocated in this process. It is strictly manipulating the internal memory location and the size parameter of the vector.

For larger datasets, particularly those containing a high degree of duplicate entries, an alternate, less computationally intense approach could prove beneficial. This involves allocating a flag array alongside the data array. The flag array is then used to mark duplicate entries. Following this marking step, the original array is compacted using a technique that leverages the flag array. Here’s how that looks in CUDA:

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <iostream>

// Example 2: Duplicate detection and compaction with a flag array

void compactDuplicates(thrust::device_vector<unsigned int>& d_data, thrust::device_vector<bool>& d_flags) {
    int size = d_data.size();

    // Allocate a temporary vector and init
    thrust::device_vector<unsigned int> d_indices(size);
    thrust::sequence(thrust::execution::par, d_indices.begin(), d_indices.end());
	
	// Create a pair of <data, index> and sort
    thrust::device_vector<thrust::pair<unsigned int, unsigned int>> d_pairs(size);
    thrust::copy(thrust::execution::par, thrust::make_zip_iterator(thrust::make_tuple(d_data.begin(), d_indices.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_data.end(), d_indices.end())), d_pairs.begin());
    thrust::sort(thrust::execution::par, d_pairs.begin(), d_pairs.end());

	// Extract the sorted index
    thrust::device_vector<unsigned int> d_sortedIndices(size);
    thrust::transform(thrust::execution::par, d_pairs.begin(), d_pairs.end(), d_sortedIndices.begin(), [](auto pair){ return pair.second; });
    
    // Reorder the data
    thrust::device_vector<unsigned int> d_reordered(size);
    thrust::gather(thrust::execution::par, d_sortedIndices.begin(), d_sortedIndices.end(), d_data.begin(), d_reordered.begin());
    d_data = d_reordered;
    
    // Mark duplicates
    thrust::adjacent_difference(thrust::execution::par, d_data.begin(), d_data.end(), d_flags.begin());
    thrust::transform(thrust::execution::par, d_flags.begin(), d_flags.end(), d_flags.begin(), [](unsigned int val){ return val != 0;});
    d_flags[0] = true; // Always keep first
   
    // Compact by copying only flagged indices
    thrust::device_vector<unsigned int> d_temp(size);
    thrust::copy_if(thrust::execution::par, d_data.begin(), d_data.end(), d_flags.begin(), d_temp.begin());

    // Resize
    int newSize = thrust::reduce(thrust::execution::par, d_flags.begin(), d_flags.end(), 0, thrust::plus<int>());
    d_data.resize(newSize);
    thrust::copy(thrust::execution::par, d_temp.begin(), d_temp.begin() + newSize, d_data.begin());
}

int main() {
    // Example usage:
    thrust::device_vector<unsigned int> d_data = {1, 5, 2, 2, 8, 5, 9, 1, 4};
    thrust::device_vector<bool> d_flags(d_data.size(), false);
    
    std::cout << "Original Data: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    compactDuplicates(d_data, d_flags);
    
     std::cout << "Data without duplicates: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;


    return 0;
}

```

In this implementation, an initial sort is performed by creating a paired vector with indices, sorting the pairs by their data values, and then extracting the indices to reorder the original data. This preserves the original index mapping for each value, which can be useful in some contexts. The flag array is then generated using `adjacent_difference` and marking values not equal to zero as `true`, indicating the start of a unique segment of the data. Lastly, `copy_if` copies only the elements that are marked as unique and the vector is resized.

However, depending on the use case and nature of duplicates (e.g. their proximity to one another) this can be further optimized. Let’s illustrate a method using atomic operations, which can provide performance benefits in specific cases. This example assumes that you have a pre-allocated scratch array of the same size as your data and that the range of values in the data is reasonable such that it can fit within the allocated scratch array's address space.

```cpp
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <iostream>

// Example 3: Atomic based duplicate marking

__global__ void markDuplicatesKernel(unsigned int* data, bool* flags, unsigned int* scratch, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size) return;
   
    unsigned int value = data[i];
    unsigned int scratchAddress = value;
    if (atomicCAS(&scratch[scratchAddress], 0, 1) == 0) { //Attempt to write a one to the address. 0 if no value was written before
        flags[i] = true; // If the value is unique, mark it as such
    }
}

void atomicDuplicates(thrust::device_vector<unsigned int>& d_data, thrust::device_vector<bool>& d_flags, thrust::device_vector<unsigned int>& d_scratch) {
   int size = d_data.size();
   int threadsPerBlock = 256;
   int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize scratch to all zeroes
   thrust::fill(thrust::execution::par, d_scratch.begin(), d_scratch.end(), 0);

   markDuplicatesKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_flags.data()), thrust::raw_pointer_cast(d_scratch.data()), size);

   //Compact the results
    thrust::device_vector<unsigned int> d_temp(size);
    thrust::copy_if(thrust::execution::par, d_data.begin(), d_data.end(), d_flags.begin(), d_temp.begin());

    // Resize
    int newSize = thrust::reduce(thrust::execution::par, d_flags.begin(), d_flags.end(), 0, thrust::plus<int>());
    d_data.resize(newSize);
    thrust::copy(thrust::execution::par, d_temp.begin(), d_temp.begin() + newSize, d_data.begin());
}

int main() {
    // Example usage:
    thrust::device_vector<unsigned int> d_data = {1, 5, 2, 2, 8, 5, 9, 1, 4};
    thrust::device_vector<bool> d_flags(d_data.size(), false);
    thrust::device_vector<unsigned int> d_scratch(100, 0);
    
    std::cout << "Original Data: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    atomicDuplicates(d_data, d_flags, d_scratch);
	
    std::cout << "Data without duplicates: ";
    for(unsigned int x : d_data){
        std::cout << x << " ";
    }
    std::cout << std::endl;


    return 0;
}
```
In this example, the `markDuplicatesKernel` function iterates through the data, attempting to perform an atomic compare-and-swap on the scratch array for each value of the input data.  If the value is encountered for the first time, a one will be written to it and a zero will be returned, in which case it marks the element in the flag array as being unique. This is followed by the compaction process using the flag array as shown in the previous example. The scratch array size must be at least as large as the largest value within the data to avoid memory access conflicts.

Choosing the right approach depends on the specific needs of your application, including the size of the data, the expected number of duplicates, and the available resources. It’s also important to be mindful of synchronization overhead, particularly when utilizing atomic operations. When dealing with a massive array and few duplicates, sorting may become a bottleneck. Conversely, when the range of values is relatively small and there are lots of duplicates, the scratch-based atomic approach could prove to be the most effective. Experimentation and benchmarking with representative data sets are the optimal way to determine what will work best for a given use case.

For further learning, I would strongly recommend delving into the NVIDIA documentation on Thrust, paying close attention to algorithms related to sorting, searching and transformations. Advanced CUDA programming books are also a good reference, particularly those that contain in-depth discussions of parallel patterns. Understanding the underlying memory hierarchy of the GPU and experimenting with kernel configurations based on your specific GPU architecture can also lead to performance enhancements. Finally, practical experience with different data sets will solidify the most suitable approach.
