---
title: "What happened to GLSL and Cg, given the rise of CUDA, OpenCL, and PGI?"
date: "2025-01-26"
id: "what-happened-to-glsl-and-cg-given-the-rise-of-cuda-opencl-and-pgi"
---

The landscape of GPU programming has undergone a significant shift since the early 2000s, heavily impacting the roles of GLSL (OpenGL Shading Language) and Cg (NVIDIA’s C for Graphics). While neither language vanished, their primary use cases and perceived importance within the broader GPU compute ecosystem have fundamentally changed, largely due to the rise of CUDA, OpenCL, and technologies from PGI (now NVIDIA HPC SDK).

Before the widespread adoption of general-purpose GPU (GPGPU) computation, GLSL and Cg were pivotal for graphics rendering. GLSL, designed for OpenGL pipelines, and Cg, while also usable with other APIs, were the primary means to write shaders – short programs executed on the GPU to determine how pixels are rendered. These languages excelled within the graphics domain, providing constructs tailored for vertex processing, fragment processing, and specific texture operations. I spent considerable time back then fine-tuning vertex shaders in GLSL for complex particle effects, and witnessed how Cg could produce elegant material rendering. The focus was predominantly on visual effects and the direct manipulation of the rendering pipeline.

The advent of CUDA, NVIDIA’s parallel computing platform, fundamentally altered this paradigm. CUDA offered a C/C++ based programming model, allowing developers to directly address the massive parallelism of GPUs for tasks beyond graphics. This capability opened up entirely new fields like scientific simulations, machine learning, and large-scale data processing. CUDA provided a richer set of tools for memory management, thread synchronization, and specialized hardware features absent in the original graphics-focused shading languages. OpenCL emerged as a cross-vendor alternative to CUDA, providing comparable GPGPU capabilities, further cementing the move towards a more generalized GPU programming model. With OpenCL gaining traction, developers no longer needed to limit GPU use to rendering pipelines.

The significance of PGI, specifically its compilers for various platforms and languages (especially Fortran and C), also played a role. Though not a direct alternative to GLSL or Cg, PGI enabled developers working in scientific computing to leverage GPUs with their existing codebases. This was achieved through compiler directives and extensions, avoiding a full rewrite in CUDA or OpenCL. These tools provided a different path towards GPU acceleration, targeting high performance numerical applications rather than pixel processing. PGI’s focus on scientific code helped broaden the GPU application domain further away from graphics.

In essence, GLSL and Cg were designed for a specific purpose within a particular architectural model; the graphics pipeline. CUDA, OpenCL, and PGI, on the other hand, were architected to serve a much broader scope. This specialization meant that while GLSL and Cg remained critical for rendering, they were not well-suited for the diverse requirements of general-purpose computation. Therefore, they did not disappear but rather were relegated to their specific niche.

Here are a few code examples demonstrating this paradigm shift:

**Example 1: GLSL - Fragment Shader**

```glsl
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D ourTexture;

void main()
{
    FragColor = texture(ourTexture, TexCoord);
}
```

*Commentary:* This GLSL code fragment is a very basic example of a fragment shader. It samples a texture based on texture coordinates passed from the vertex shader, and writes this color to the output. This type of shader is absolutely vital for rendering operations and directly maps to specific steps in the rendering pipeline. It demonstrates the core use case of GLSL: direct pixel manipulation as part of the graphics rendering process. While this code is essential for rendering a texture, it lacks the mechanisms for arbitrary compute tasks that CUDA or OpenCL would provide.

**Example 2: CUDA - Vector Addition Kernel**

```cuda
#include <cuda.h>

__global__ void vectorAdd(float* a, float* b, float* c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
      c[i] = a[i] + b[i];
  }
}

// Host-side invocation (simplified for brevity)
// float* a_h, b_h, c_h; ... (host data)
// float* a_d, b_d, c_d; ... (device data)
// cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
// ...
// int numBlocks = (size + blockSize - 1) / blockSize;
// vectorAdd<<<numBlocks, blockSize>>>(a_d, b_d, c_d, size);
// ...
```

*Commentary:* This CUDA code demonstrates a kernel performing vector addition. The kernel code, annotated with `__global__`, is executed on the GPU’s parallel cores, each assigned a portion of the vector. The host-side code, which would include memory allocation and data transfer to the device, is not shown here for brevity. This illustrates the parallel nature of GPU computation with CUDA and the distinct programming model. This compute example would be difficult and inefficient to reproduce using GLSL or Cg. Here, we are performing computations unrelated to any graphic operations, something GLSL was not designed to support.

**Example 3: OpenCL - Matrix Multiplication Kernel**

```opencl
__kernel void matrixMul(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int widthA, int widthB)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < widthA && col < widthB) {
      float sum = 0.0f;
      for (int k = 0; k < widthA; k++) {
         sum += A[row * widthA + k] * B[k * widthB + col];
      }
      C[row * widthB + col] = sum;
    }
}

// Host side example (simplified)
// cl_mem a_buff, b_buff, c_buff; ... (create and load buffers)
// cl_kernel kernel = clCreateKernel(...);
// size_t global_work_size[2] = {widthA, widthB};
// clEnqueueNDRangeKernel(..., kernel, 2, NULL, global_work_size, NULL, ...);
//
```

*Commentary:* This OpenCL example presents a simplified matrix multiplication kernel. Similar to the CUDA example, each kernel invocation represents a thread performing a part of the calculation. This kernel operates on input and output buffers located in GPU memory, demonstrating another approach to general purpose parallel computation. OpenCL demonstrates a platform-independent approach to GPGPU programming, offering a viable alternative to CUDA. Like the CUDA example, this code's focus is not on visual effects or rendering, but on general numerical operations. This is something that goes beyond the initial scope of GLSL and Cg.

In conclusion, GLSL and Cg did not disappear but instead became specialized tools within the domain of graphics rendering, a consequence of the broader availability of powerful GPGPU tools like CUDA and OpenCL, and approaches such as PGI’s compiler tools, which better address the requirements of large-scale compute applications. These new tools enabled developers to harness the power of GPUs for a wide range of problems beyond the narrow scope of graphics. GLSL and Cg are now utilized for graphics, while CUDA, OpenCL, and PGI’s offerings have become standard for general-purpose GPU computation.

For further study, I would recommend delving into books on:

1.  OpenGL Programming, to understand the contemporary role of GLSL within graphics.
2.  CUDA Programming, to gain a deeper understanding of NVIDIA's parallel computing architecture.
3.  OpenCL Programming, for a deeper grasp of cross-platform GPU computing.
4.  Scientific Computing with High Performance Compilers, to better understand the role of tools like PGI in HPC.
