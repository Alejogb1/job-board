---
title: "How should Julia AMDGPU.jl be launched?"
date: "2025-01-30"
id: "how-should-julia-amdgpujl-be-launched"
---
AMDGPU.jl, the Julia package for programming AMD GPUs, necessitates a specific launch process that differs significantly from CPU-bound Julia code. Directly executing a Julia script containing AMDGPU.jl functions without proper initialization will result in errors, often related to the lack of a designated GPU device context or insufficient resource allocation. My experience working on a fluid dynamics simulation using an integrated AMD GPU underlines this point acutely. The initial naive approach, simply running a script with `@kernel` and `@gpu` macros, consistently failed until I understood the intricate setup required.

The core of the launch process centers on three key elements: selecting a target device, establishing a GPU context, and appropriately launching kernels. The absence of any of these results in a failure to utilize the GPU for computation. Unlike CPU code, which implicitly relies on the host architecture's processing capabilities, GPU programming in Julia via AMDGPU.jl demands explicit direction to the desired hardware. This involves utilizing functions within the AMDGPU.jl library itself to control these steps.

The first necessary action is device selection. This step is critical, particularly if multiple GPUs are available, or if you need to target a specific integrated GPU. The function `AMDGPU.devices()` returns a vector of available device objects. These objects contain metadata, including the device's name, vendor, and compute capabilities. Inspecting these objects via `println` allows a user to identify and select the proper device for their task. Once a specific device is chosen, the `AMDGPU.device!(device_object)` function sets it as the active device for subsequent GPU operations. Failing to set an active device results in an error, as operations cannot be dispatched to a processing unit without an established target.

The next phase involves setting up a GPU context. A context can be viewed as the operating environment within which the GPU computation will be executed. This includes managing resources, such as allocated memory on the device. The primary function here is `AMDGPU.context()`. It automatically creates a context object when called for the first time, associated with the active device. The context object can be passed as an argument to some kernel launch functions, which is useful for managing more complex or multi-device programs, although simple scripts often function with the automatic assignment within AMDGPU.jl. If an operation does not require an explicit context object, it will utilize the current context that has been set. Not explicitly utilizing contexts can lead to issues where the desired context might not be active for multi-device setups. Although the default auto context selection will suffice for single GPU usage, it does not hurt to create one yourself to understand the procedure. It is important to understand that all memory allocations on the GPU are associated with a particular context and device.

Finally, the kernel launch itself requires specific instructions. Instead of calling a kernel function directly, it must be wrapped using the `@gpu` macro. `@gpu` transforms a regular Julia function call into a GPU-executed operation. Its usage is of the form `@gpu threads=threads groups=groups kernel_function(args...)`. The `threads` and `groups` parameters specify the work distribution on the GPU, crucial for performance tuning. The `threads` parameter defines the number of threads within each work group and `groups` specifies how many work groups should be used for the launch of the kernel, essentially structuring the parallel workload. The kernel itself must be compiled with `@kernel` annotation and take `threadIdx()` and `blockIdx()` functions as arguments which allow the individual threads and blocks to determine which section of data they must operate on. Incorrectly specified work groups or threads will result in a kernel either failing to launch properly, or executing in unexpected ways leading to wrong results. These factors are often application specific. Failing to properly understand how your data can be mapped onto these threads will not yield desired results.

Here are three illustrative code examples with commentary:

**Example 1: Simple Vector Addition**

```julia
using AMDGPU

function vector_addition(A, B, C, N)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if index <= N
        @inbounds C[index] = A[index] + B[index]
    end
    return nothing
end

function main()
    AMDGPU.device!(AMDGPU.devices()[1])
    N = 1024
    A = ROCArray(rand(Float32, N))
    B = ROCArray(rand(Float32, N))
    C = ROCArray(zeros(Float32, N))
    threads = 256
    groups = ceil(Int, N / threads)
    @roc groups=groups threads=threads vector_addition(A,B,C,N)
    println(Array(C))
end
main()
```
This example demonstrates a basic vector addition operation. First, the first available GPU device is selected, and then `ROCArray` types are used to allocate data on the GPU itself. The `threads` and `groups` are specified, as well as passing the data to the kernel. The kernel itself is annotated with `@roc` which executes the kernel on the currently selected GPU device. Finally the data from `C` is copied back to the CPU using `Array`. This code is illustrative, and the selection of threads and groups is not optimized.

**Example 2: Matrix Multiplication (Simplified)**

```julia
using AMDGPU

function matrix_mul(A, B, C, M, N, K)
    row = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if row <= M && col <= N
         temp = zero(eltype(C))
         for k=1:K
            temp += A[row,k] * B[k,col]
         end
         C[row,col] = temp
    end
    return nothing
end

function main()
    AMDGPU.device!(AMDGPU.devices()[1])
    M = 1024; N = 1024; K = 1024
    A = ROCArray(rand(Float32, M, K))
    B = ROCArray(rand(Float32, K, N))
    C = ROCArray(zeros(Float32, M, N))
    threads = (16, 16)
    groups = (ceil(Int, N/threads[1]),ceil(Int,M/threads[2]))
    @roc groups=groups threads=threads matrix_mul(A,B,C,M,N,K)
    println(Array(C)[1:10,1:10])
end

main()
```

This example shows a basic matrix multiplication. `threads` and `groups` parameters must be tuples here as the access is two-dimensional for the matrices being multiplied. This example highlights how the block and thread indices can be used to access the matrix data in a 2D manner. There is still no consideration to memory locality for the performance of this kernel.

**Example 3: Using a custom context**

```julia
using AMDGPU

function test(a, b, c)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    c[idx] = a[idx] + b[idx]
    return nothing
end

function main()
    AMDGPU.device!(AMDGPU.devices()[1])
    ctx = AMDGPU.context()
    N = 1024
    a = ROCArray(rand(Float32, N), ctx = ctx)
    b = ROCArray(rand(Float32, N), ctx = ctx)
    c = ROCArray(zeros(Float32, N), ctx = ctx)
    threads = 256
    groups = ceil(Int, N/threads)
    @roc groups=groups threads=threads test(a, b, c)
    println(Array(c))
end

main()
```
This example demonstrates the creation of a context, as well as the passing of the context to the `ROCArray` constructor. This ensures that the memory allocated on the GPU is associated with the created context. It also shows that the launch can work by using the current selected GPU if there is only one GPU detected on the system.

Resources for deepening one's understanding of AMDGPU.jl include:
* The official AMDGPU.jl documentation provides a comprehensive overview of the API, core concepts, and examples.
* The JuliaGPU organization’s repository on GitHub contains the most recent versions of the code as well as issues and pull requests from the community.
* AMD’s ROCm documentation elucidates the underlying ROCm platform, its system architecture, and its role in supporting GPU programming.
* The book "Julia Programming for Operations Research" by Changhyun Kwon contains detailed discussions of parallel programming patterns and strategies applicable to Julia and GPU computing.

In summary, launching Julia code using AMDGPU.jl demands a meticulous approach. Properly selecting a device, establishing a context, and launching kernels with the correct workgroup parameters are not optional steps, but fundamental requirements. Neglecting these will result in errors and failing to utilize GPU acceleration effectively. The provided examples, coupled with a methodical study of recommended resources, will facilitate the creation of robust, performant GPU code in Julia using AMDGPU.jl.
