---
title: "How can duplicate keys be handled when merging lists in CUDA?"
date: "2025-01-30"
id: "how-can-duplicate-keys-be-handled-when-merging"
---
When performing parallel data aggregation with CUDA, specifically merging lists where duplicate keys are a possibility, a naive approach of direct insertion can lead to race conditions and incorrect results. This arises because multiple threads might attempt to modify the same location in memory associated with a given key simultaneously. To effectively manage this, atomic operations, specifically atomicAdd or atomicMax/Min depending on the aggregation logic, combined with a carefully designed data structure, are essential.

The core issue stems from the shared memory environment of CUDA threads within a block. Consider, for example, several threads needing to contribute data associated with the same key to an output array. Without proper synchronization, one thread might overwrite the change made by another, leading to data loss and incorrect summations or aggregates. A simplistic ‘if not exists add, else increment’ approach implemented without atomic operations will not be thread-safe.

My experience working on particle simulations, where merging force contributions from multiple nearby particles onto a single particle required precisely this type of key-value aggregation, has demonstrated the need for a robust solution. Using atomic operations and a hash table-like structure, I was able to implement performant parallel reduction.

Let's break down the practical aspects:

**1. Atomic Operations:**

Atomic operations are CUDA intrinsics that provide guarantees about memory access. When a thread executes an atomic operation, it completes the entire operation without interruption. This prevents the interleaved access and overwrite problem associated with non-atomic operations in parallel contexts. The most relevant operations for key-value aggregation are:

*   **`atomicAdd(float* address, float val)`:** Adds `val` to the floating point value at address, returning the value held *before* the addition. Critically, this is done atomically.
*   **`atomicAdd(int* address, int val)`:** Same as above, but for integers.
*   **`atomicMax(int* address, int val)`:** Compares `val` against the integer at address. Sets the value at address to the larger of the two. Returns the value held *before* the comparison.
*   **`atomicMin(int* address, int val)`:** Same as `atomicMax` but sets the address to the smaller of the two.
*   **`atomicCAS(int* address, int compare, int val)`:** Compare and swap. If the value at address equals `compare`, the value at address is updated to `val`. Returns the value held *before* the comparison.

The choice of which atomic operation to use depends entirely on the nature of the aggregation. If you need to sum values, `atomicAdd` is appropriate; if you need to find a maximum or minimum, `atomicMax` or `atomicMin` respectively are the operations of choice. If a more complicated operation is necessary, such as bitwise operations, `atomicCAS` may be necessary to ensure thread-safety within a loop.

**2. Data Structure Design:**

While atomic operations provide thread safety, efficient access to the memory locations associated with a key is also critical. A naive approach of directly indexing into an array will require knowing all possible keys beforehand and ensuring memory allocation is sufficient. This becomes impractical or impossible with sparse key-value pairs. A more versatile and efficient approach is to leverage the structure of a hash table, storing key-value pairs in an array.

In general, I prefer to use a single device array in my implementations, the device array itself acts as a hash table. I typically use a hashing function that transforms the keys into indices within this array. Crucially, multiple keys may hash to the same index – this is known as a collision. Handling collisions effectively is essential. Here, I typically employ open addressing (linear probing) techniques. I then use atomic operations to either update the value at this location or, if the location is already occupied (for a different key), I probe through subsequent indices until an open location is found. When a collision has occurred, the current key and value need to be temporarily stored. This method is far more efficient than using shared memory due to limitations of size and a lack of built-in atomic operations on dynamically allocated shared memory in many versions of CUDA.

**3. Code Examples:**

Let's explore a few simple scenarios with example code to illustrate:

**Example 1: Simple Summation with Integer Keys**

```cuda
__global__ void merge_lists_sum(int* keys, int* values, int* output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int key = keys[i];
    int value = values[i];
    int index = key % OUTPUT_SIZE;  // Simple hash
    while (true){
       int* address = &output[index];
       int previous_value = atomicAdd(address, value);
       if (previous_value == 0) {
          // The initial value is always 0, signifying it's a new key location or previous value was zero
          break;
       } else if (address == &output[index])
       {
          index++; // Linear probing, need to probe again
          index = index % OUTPUT_SIZE;
       }
       else {
          break; // This should be impossible, but good to check
       }
    }
}
```
*Commentary:* This kernel performs a simple summation. It calculates an initial `index` by modulo operation. Within a `while` loop, it checks if the initial value at the computed index is zero. If it is, then no other thread has written to that index, and we are adding to a new value or zero.  Otherwise we linear probe until an empty slot is found (this could also be a slot where the value is 0). The atomic add increments the value located at the current output index, avoiding race conditions.

**Example 2: Finding the Maximum Value Per Key (Integer Keys and Values)**

```cuda
__global__ void merge_lists_max(int* keys, int* values, int* output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int key = keys[i];
    int value = values[i];
    int index = key % OUTPUT_SIZE; //Simple hash

    while(true) {
       int* address = &output[index];
       int previous_value = atomicMax(address, value);
       if (previous_value == 0)
       {
         break; // New value found
       }
       else if (address == &output[index]) {
          index++; // Probing
          index = index % OUTPUT_SIZE;
       }
        else {
          break; //Should be impossible
       }
    }
}
```

*Commentary:* This kernel finds the maximum value for each key. The structure is similar to the summation example, using atomicMax instead of atomicAdd. The initial value at each position of output array is set to 0, which correctly functions when combined with `atomicMax` to find a maximum. This pattern is analogous for finding the minimum value with `atomicMin`.

**Example 3: Using a Value Object to Store More Information**

```cpp
struct Value {
    int count;
    float total;
};

__global__ void merge_lists_avg(int* keys, float* values, Value* output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int key = keys[i];
    float value = values[i];
    int index = key % OUTPUT_SIZE; //Simple hash

    while(true){
        Value* address = &output[index];
        int previous_count = atomicAdd(&address->count, 1);
        if (previous_count == 0) {
           address->total = value;
           break;
        }
        else {
           atomicAdd(&address->total, value);
            if (address == &output[index])
            {
              index++; // Probing
              index = index % OUTPUT_SIZE;
            } else
            {
              break;
            }
         }
    }

}
```

*Commentary:* This illustrates a slightly more complex scenario where, instead of simply summing or finding a maximum, we need to maintain a count and sum of each value to compute an average. The Value struct contains both the `count` and `total`. The kernel uses `atomicAdd` to update both `count` and `total`, illustrating how atomic operations can be used with structures.

**Resource Recommendations:**

For further exploration and understanding of parallel programming with CUDA, the following resources are useful. The CUDA Programming Guide by NVIDIA provides an exhaustive reference for all of the CUDA API functionality, including atomic operations. For broader knowledge on parallel programming patterns, textbooks such as "Introduction to Parallel Computing" and "Parallel Computer Architecture: A Hardware/Software Approach" are beneficial. Furthermore, online forums and discussion boards specifically dedicated to CUDA programming offer a platform for troubleshooting, discussion, and learning from the broader CUDA development community.
