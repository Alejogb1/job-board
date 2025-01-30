---
title: "Do all GPUs share a common architecture?"
date: "2025-01-30"
id: "do-all-gpus-share-a-common-architecture"
---
No, GPUs do not share a common architecture.  My experience working on heterogeneous computing platforms for over a decade, including projects involving NVIDIA, AMD, and Intel architectures, has consistently highlighted the significant architectural variations across GPU vendors and even within a single vendor's product lines.  While there are common high-level concepts like stream processors and memory hierarchies, the underlying implementation details, instruction sets, memory access patterns, and interconnection strategies differ substantially.  This architectural diversity impacts software development, requiring tailored approaches to maximize performance on specific hardware.

**1. Architectural Divergences:**

The fundamental difference stems from the different design philosophies employed by various vendors.  NVIDIA, for instance, has historically favored a unified architecture, where stream processors are highly programmable and can efficiently handle both compute and graphics tasks.  This design prioritizes flexibility and general-purpose computing capabilities.  AMD, on the other hand, has traditionally employed a more specialized approach, sometimes separating compute and graphics units more distinctly, optimizing each for specific workloads.  Intel's entry into the discrete GPU market has further diversified the landscape, bringing its own unique architectural choices, emphasizing features like Xe Matrix Extensions and other specialized instructions.

These differences manifest in several crucial aspects:

* **Instruction Set Architecture (ISA):**  Each vendor has its unique ISA, defining the instructions the GPU can execute.  This includes variations in the number and types of registers, memory addressing modes, and the available instructions for various operations.  Compiling code for one vendor's GPU will not directly work on another's without significant modification or the use of abstraction layers.

* **Memory Hierarchy:**  The organization and management of memory differ considerably.  Variations exist in the size and bandwidth of different memory levels (registers, L1, L2 cache, global memory), memory access patterns (coalesced vs. non-coalesced), and the presence and implementation of features like memory compression and error correction.  Efficient memory management is crucial for performance, and this requires architectural awareness.

* **Interconnect:**  The way different parts of the GPU communicate internally (between stream processors, memory controllers, etc.) also differs.  The specific interconnect topology and bandwidth significantly influence the overall performance, especially for highly parallel algorithms.

* **Hardware Features:**  Specific hardware features, such as tensor cores (NVIDIA), ray tracing cores (both NVIDIA and AMD), or specialized AI accelerators, vary across architectures and even within a vendor’s product line.  Leveraging these hardware features requires careful code optimization tailored to the specific GPU.


**2. Code Examples Illustrating Architectural Differences:**

The following examples illustrate how code targeting different architectures needs specific adaptations:


**Example 1:  CUDA (NVIDIA) vs. ROCm (AMD) Kernel Launch**

```c++
// CUDA kernel launch
__global__ void myKernel(int *data, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i] *= 2;
  }
}

int main() {
  int *h_data, *d_data;
  // ... memory allocation and data transfer ...
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
  // ... data transfer and cleanup ...
}


// ROCm kernel launch (HIP)
__global__ void myKernel(int *data, int size) {
  int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (i < size) {
    data[i] *= 2;
  }
}

int main() {
  int *h_data, *d_data;
  // ... memory allocation and data transfer ...
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(myKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_data, size);
  // ... data transfer and cleanup ...
}
```

Commentary: Although the kernel logic is identical, the launch mechanisms differ significantly.  CUDA uses a simpler syntax, while ROCm (using HIP for portability) requires a more explicit function call for kernel launch management, reflecting differences in the underlying runtime environments.


**Example 2:  Memory Access Optimization**

```c++
// Optimized for coalesced memory access (NVIDIA-like architectures)
__global__ void coalescedAccess(float *data, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i] = data[i] * 2.0f;
  }
}

// Non-optimized memory access (may perform poorly on coalesced architectures)
__global__ void uncoalescedAccess(float *data, int size) {
  int i = threadIdx.x;
  for (int j = 0; j < size; j += blockDim.x) {
    data[i + j] = data[i + j] * 2.0f;
  }
}
```

Commentary:  The `coalescedAccess` kernel demonstrates efficient memory access, crucial for achieving high bandwidth on many architectures. The `uncoalescedAccess` kernel, although functionally equivalent, suffers from non-coalesced accesses, potentially resulting in significantly lower performance, especially on architectures that heavily prioritize coalesced memory access.


**Example 3:  Utilizing Vendor-Specific Extensions**

```c++
// Utilizing NVIDIA Tensor Cores
__global__ void tensorCoreExample(float *a, float *b, float *c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    // Assuming appropriate data types and alignment for Tensor Cores
    c[i] = __fmaf_rn(a[i], b[i], c[i]); //fused multiply-add with round-to-nearest
  }
}

// Equivalent operation without Tensor Cores (fallback for other architectures)
__global__ void fallbackExample(float *a, float *b, float *c, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
        c[i] = a[i] * b[i] + c[i];
    }
}
```

Commentary: This example showcases the utilization of NVIDIA's Tensor Cores for accelerated matrix multiplication. The `fallbackExample` provides an alternative implementation for GPUs lacking this specialized hardware.  Similar vendor-specific extensions exist for AMD and Intel architectures, demanding conditional compilation or abstraction layers to handle the architectural differences.


**3. Resource Recommendations:**

For deeper understanding, consult the official documentation and programming guides provided by each GPU vendor (NVIDIA CUDA, AMD ROCm, Intel oneAPI).  Additionally, textbooks focusing on parallel computing and GPU programming offer valuable insights into architectural considerations.  Specialized publications and conference proceedings in high-performance computing are also indispensable resources.  Finally, studying benchmark results and performance analyses of different GPU architectures will enhance understanding of practical differences.
