---
title: "Do int and float atomicAdd operations in shared memory exhibit different SASS behavior?"
date: "2025-01-30"
id: "do-int-and-float-atomicadd-operations-in-shared"
---
The atomic addition operation, specifically when applied to integer (`int`) and floating-point (`float`) types within shared memory in a CUDA context, exhibits notable differences at the SASS (Streaming Assembly) level due to their fundamentally disparate handling by the hardware. These variations aren't simply syntactic sugar; they reflect the distinct instruction sets and underlying mechanisms employed for each type. My experience optimizing CUDA kernels for high-performance computing reveals that understanding these nuances is crucial for efficient parallel code, especially when dealing with shared memory where contention can easily become a bottleneck.

At the core, integer atomic operations, including `atomicAdd`, are typically implemented using a read-modify-write sequence directly supported by the GPU's memory architecture. The SASS code corresponding to an integer `atomicAdd` often translates to instructions involving atomic memory exchange or compare-and-swap operations, leveraging the hardware's capabilities for direct bit manipulation. In contrast, floating-point atomic operations face a more complex reality. The inherent imprecision of floating-point arithmetic, coupled with the need to maintain IEEE 754 compliance, necessitates a different approach. Instead of direct hardware support for atomic floating-point addition, these operations are frequently implemented through more involved routines. These routines typically involve a loop where a thread reads the current value, performs the floating-point addition, and then attempts to atomically write back the new value, potentially retrying if another thread has modified the memory location in the interim. This difference results in distinct SASS patterns, often making the floating-point variant significantly less performant than the integer counterpart.

To illustrate, consider a hypothetical scenario in which threads within a block increment a shared memory variable, first an integer and then a float. We can use a simplified example to compare the generated SASS instructions.

**Example 1: Integer Atomic Addition**

```cpp
__shared__ int shared_int;
__global__ void int_atomicAdd_kernel(int* output) {
    int tid = threadIdx.x;
    if (tid == 0) {
        shared_int = 0;
    }
    __syncthreads();
    atomicAdd(&shared_int, 1);
    __syncthreads();

    if (tid == 0)
        *output = shared_int;
}
```
The SASS generated for the `atomicAdd(&shared_int, 1)` statement will likely resemble something like the following (simplified):
```sass
//Simplified hypothetical SASS instructions

	/*  0x00000000000133d0 */         LD.S.64         R11, [0x00000000]        ;0x00000000000133d0
	/*  0x00000000000133d8 */         IADD3           R10, R11, 0x1    ; 
	/*  0x00000000000133e0 */    MEMBAR.CTA 
	/*  0x00000000000133e8 */    ATOMCAS         R11, R11, R10, [R6.S]    ;0x00000000000133e8
  /*  0x00000000000133f0 */    ISETP.NE.AND P0, PT, R11, R10, PT;
  /*  0x00000000000133f8 */    @P0 BRA 0x00000000000133d8;
```

This snippet utilizes an atomic compare-and-swap (ATOMCAS) instruction. The shared memory address, referred to symbolically as `R6.S`, is read into `R11`. `R11` is incremented and stored into `R10`. The `ATOMCAS` instruction will attempt to write `R10` back to the shared memory address, provided the current value at the address is still `R11`. If the compare fails, the loop will attempt the read, add, and write process again. The key is that this is achieved through specialized hardware support, making it relatively fast.

**Example 2: Float Atomic Addition**

Now, let's examine the equivalent with floating-point numbers:
```cpp
__shared__ float shared_float;
__global__ void float_atomicAdd_kernel(float* output) {
    int tid = threadIdx.x;
    if (tid == 0) {
        shared_float = 0.0f;
    }
    __syncthreads();
    atomicAdd(&shared_float, 1.0f);
    __syncthreads();
    if(tid==0)
        *output = shared_float;

}
```
The SASS representation of the `atomicAdd(&shared_float, 1.0f)` operation differs significantly:
```sass
//Simplified Hypothetical SASS instructions

      /* 0x00000000000133d0 */         LD.S.32      R11, [0x00000000]        ;0x00000000000133d0
      /* 0x00000000000133d8 */      FADD R10, R11, 0x3f800000;  // 1.0f represented as hex
      /* 0x00000000000133e0 */    MEMBAR.CTA 
	  /*  0x00000000000133e8 */    ATOMCAS R11, R11, R10, [R6.S]
      /*  0x00000000000133f0 */    ISETP.NE.AND P0, PT, R11, R10, PT;
      /*  0x00000000000133f8 */    @P0 BRA 0x00000000000133d8;

```
Here, the SASS now explicitly performs an FADD, rather than a simple increment operation. Additionally the compare-and-swap loop remains to ensure atomicity since FADD is not atomic itself. The floating point addition itself also takes significantly longer than a simple integer increment, hence impacting overall performance of the kernel.
**Example 3: Alternative Float Atomic Implementation Using a Loop**

In some cases, particularly on older architectures, a more explicit loop might be observed. The SASS may resemble the following pattern:

```sass
//Simplified Hypothetical SASS instructions

       LoopStart:
    /* 0x00000000000133d0 */         LD.S.32      R11, [R6.S]       ; Read current value
    /* 0x00000000000133d8 */      FADD R10, R11, 0x3f800000;    // Perform float add
    /* 0x00000000000133e0 */    MEMBAR.CTA
    /*  0x00000000000133e8 */    ATOMCAS R12,R11, R10,[R6.S] // Attempt atomic write
    /*  0x00000000000133f0 */    ISETP.NE.AND P0, PT, R11, R12, PT; //Check if the swap was successful
      /*  0x00000000000133f8 */    @P0 BRA LoopStart; //If failed, go back to read step.
```
The key distinction in this case is the explicit looping. The float addition is performed outside of an atomic instruction and then a compare and swap is used to handle race conditions. This adds considerable overhead. While modern architectures often have hardware that will make this a fast implementation behind the scenes, it is still very important to note the difference when optimizing.
The primary divergence stems from the architectural support and complexity of the underlying operations. Integer addition is a fundamental operation that maps efficiently to the hardware's arithmetic logic unit (ALU). Conversely, floating-point addition requires more complex circuitry and adheres to stringent IEEE 754 standards, leading to the aforementioned software-based atomic handling.

Consequently, when dealing with atomic operations in shared memory, developers should consider the type used with the greatest care. If an application permits, using integer arithmetic and mapping results to floats only after the atomic operations are completed could drastically improve performance. Careful profiling is often necessary to fully grasp the scope of the performance bottleneck.

For those looking to delve deeper, I would recommend consulting the following resources:
*   The NVIDIA CUDA Programming Guide, especially sections on shared memory, atomic operations, and SASS instruction sets.
*   Publications on GPU architecture, specifically how different arithmetic and atomic operations are handled at the hardware level.
*   CUDA optimization guides focused on maximizing performance in shared memory contexts.

Understanding these low-level nuances is key for developers striving to write efficient, highly concurrent GPU applications. The seemingly small differences in data type can lead to substantial performance impacts, a lesson learned through extensive kernel development and optimization exercises.
