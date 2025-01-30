---
title: "How to determine CUDA device shared memory bank size?"
date: "2025-01-30"
id: "how-to-determine-cuda-device-shared-memory-bank"
---
Determining the shared memory bank size on a CUDA device is not a property exposed directly via a runtime API call, but rather, is an architectural detail dictated by the device's compute capability. I've spent years optimizing CUDA kernels and have come to rely on understanding these underlying hardware characteristics to avoid performance bottlenecks arising from shared memory bank conflicts.

Fundamentally, shared memory is organized into banks to allow simultaneous memory access by threads within a warp. When multiple threads within a warp try to access the same bank, a bank conflict occurs, leading to serialization of memory requests and a performance hit. To optimize memory access, it is imperative to know the bank size and strive to design data access patterns that avoid conflicts as much as possible.

The bank size, in bytes, is a hardware constant that depends on the specific GPU architecture. For devices with compute capability less than 6.0 (e.g., Kepler, Maxwell), the bank size is 4 bytes. For devices with compute capability 6.0 and greater (e.g., Pascal, Volta, Turing, Ampere), the bank size is 8 bytes.

This distinction is critically important when constructing algorithms utilizing shared memory. When incorrectly assuming bank sizes during algorithm development, you might experience unexpected performance degradations.

Here is a breakdown of how to indirectly determine the bank size and how I have implemented strategies in my code:

**1. Compute Capability Inspection:**

The most straightforward method, although not a runtime check, is to use the CUDA SDK device query utility `deviceQuery` or a similar device discovery application. When executed, it presents information related to a device's compute capability. While not showing bank size explicitly, this value is critical to determine it. Based on the aforementioned rule, compute capability below 6.0 means 4-byte bank size, and compute capability 6.0 and higher means 8-byte bank size. This method is generally sufficient for pre-compilation adjustments.

**2. Runtime Logical Deduction (Not Recommended as primary approach):**

You might try to deduce the bank size by analyzing timing differences, where you would attempt to create memory access patterns known to cause conflicts (or not) based on each size. This is not reliable and could lead to inaccurate conclusions or inconsistencies across varied architectures, particularly as future hardware may introduce more layers of complexity. Although I've occasionally used this in experimental scenarios, it is not suitable for production code, primarily because of its heavy reliance on indirect observation, as timing differences may have other causes.

**3. Code Example 1: Avoiding Bank Conflicts - 4-byte case:**

Suppose we have a matrix of size `TILE_WIDTH x TILE_WIDTH` stored in shared memory. If `TILE_WIDTH` is a power of 2, then threads within a warp often have a pattern of colliding banks while accessing the columns due to successive accesses being spaced by `TILE_WIDTH` bytes. This is especially true when the bank size is 4-bytes. To avoid these bank conflicts on older architectures, we might introduce padding. Consider this Kernel which avoids bank conflicts by ensuring a non power of 2 width.

```c++
__global__ void matrixTranspose_bankConflictAvoidance_4bytes(float* output, float* input, int width) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row >= width || col >= width) return;

    extern __shared__ float tile[];

    const int TILE_WIDTH = 16;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int paddedWidth = TILE_WIDTH + 1;

    tile[local_row * paddedWidth + local_col] = input[row * width + col];

    __syncthreads();

    output[col * width + row] = tile[local_col * paddedWidth + local_row];
}
```
*Commentary:*
In this example, using `paddedWidth` with `16 + 1 = 17`, the memory offsets for successive threads in a warp when accessing columns avoid landing in the same bank due to bank size being 4 bytes. The `TILE_WIDTH` is typically determined by experimentation. `__syncthreads()` is called to ensure that all reads are completed and data is in shared memory before the transpose writes begin to prevent race conditions. The original dimensions of input are maintained; `paddedWidth` is only used for shared memory indexing to prevent bank conflicts. This demonstrates the necessity of adjusting memory access strategy to architecture. I have personally used this technique many times on Kepler based GPUs

**4. Code Example 2: Avoiding Bank Conflicts - 8-byte case:**

When devices have a larger 8-byte bank size, similar strategies can still be beneficial, although the stride can be less aggressive. Consider the same matrix transpose example where the bank size is now 8 bytes:

```c++
__global__ void matrixTranspose_bankConflictAvoidance_8bytes(float* output, float* input, int width) {
   const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= width || col >= width) return;

    extern __shared__ float tile[];

   const int TILE_WIDTH = 32;


    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int paddedWidth = TILE_WIDTH + 2;

   tile[local_row * paddedWidth + local_col] = input[row * width + col];


   __syncthreads();

    output[col * width + row] = tile[local_col * paddedWidth + local_row];
}

```

*Commentary:*
Here, `paddedWidth` is set to `32 + 2 = 34`. While the 8-byte bank size mitigates many potential conflicts compared to older architectures, padding can still provide further performance gains. Although the relative benefit may be less than when mitigating conflicts on 4 byte GPUs. The key is that both code samples apply similar logic, which is crucial when porting or optimizing code for different devices. Again, `__syncthreads()` is used to prevent race conditions. I have observed that the optimal padding values need to be chosen experimentally based on the use case.

**5. Code Example 3: Using Data Structures to exploit bank interleaving**

Sometimes, instead of padding, one can re-arrange shared memory access. Here's a simple example using a 2D `float` array to calculate the sum of the elements.

```c++
__global__ void sharedMemorySum_interleaved(float* output, float* input, int size) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId >= size) return;

    extern __shared__ float localData[][32]; //Array of floats

    int local_x = threadIdx.x;
    
    localData[threadIdx.y][local_x] = input[globalId];
    __syncthreads();
    
    // Reduction within warp (simplified)
    for (int offset = blockDim.x/2; offset>0; offset/=2){
        if (local_x < offset){
            localData[threadIdx.y][local_x] += localData[threadIdx.y][local_x+offset];
        }
        __syncthreads();
    }
   
    if (local_x == 0)
      output[blockIdx.x] = localData[threadIdx.y][0];
}
```

*Commentary:*
Here, we use a 2D array within shared memory `localData`. Because successive threads will access adjacent `localData` entries, when `blockDim.x` is 32, accessing `localData[threadIdx.y][local_x]` provides a natural memory interleaving pattern as bank accesses are distributed across the banks of the shared memory module. The array acts like a transposed matrix which achieves bank interleaved access during the read and reduction.  As you can see, careful use of multi-dimensional shared memory arrays can be valuable even without using padding.

**Resource Recommendations**

To further develop knowledge on this topic, I recommend consulting:

1.  **CUDA Programming Guides:** These documents provided by NVIDIA offer a complete overview of the CUDA architecture, including descriptions of shared memory organization, bank conflicts, and related programming considerations.

2.  **GPU Architecture Documentation:** NVIDIA publishes technical documents that describe the microarchitecture of their GPUs. These documents offer highly technical information on how shared memory is implemented.

3.  **Online Community Forums:** Forums like those associated with CUDA and NVIDIA developer sites, or more general developer forums, contain threads by researchers and experts discussing these nuances. Pay special attention to threads discussing performance and optimization.

In conclusion, determining the shared memory bank size on a CUDA device involves understanding the compute capability of the device. While no direct API provides this information, knowing that architectural feature is key to effectively using shared memory and avoiding performance penalties due to bank conflicts. Applying strategies like those demonstrated above can aid in writing optimized code for various CUDA-capable devices.
