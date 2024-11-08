---
title: "CUDA Threads Hierarchy:  A Simple Example to Get You Started"
date: '2024-11-08'
id: 'cuda-threads-hierarchy-a-simple-example-to-get-you-started'
---

```cpp
__device__ int GetMaxValue(const int value) {  //every thread has different value
    //the input must be a bitonic sequence
    constexpr auto All = -1u; //all threads in the warp take part
    const auto Neighbor = __shfl_down_sync(All, value, 1); //thread 31 gets its own value back
    const auto mask = __ballot_sync(All, Neighbor > value); //e.g. 0b0000000011111111
    const auto MaxBit = __popc(mask); //The thread holding the max value
    const auto result = __shfl_sync(All, value, MaxBit); //share max value with all
    return result; //all threads return the same result
}

__device__ int GetMaxValue_NoWarp(int value, int* tempstorage) {
    tempstorage[32] = 0;
    tempstorage[threadIdx.x] = value;
    __syncthreads();
    const auto Neighbor = tempstorage[threadIdx.x + 1];
    if (threadIdx.x == 31) { Neighbor = value; }
    const auto diff = int(Neighbor > value); 
    atomicOr(&tempstorage[32], diff << threadIdx.x);
    __syncthreads();
    const auto mask = tempstorage[32];
    const auto MaxBit = __popc(mask); 
    result = tempstorage[MaxBit];
    return result;
} 
```
