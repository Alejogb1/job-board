---
title: "How can MATLAB invoke CUDA GPU code without using kernel functions?"
date: "2025-01-30"
id: "how-can-matlab-invoke-cuda-gpu-code-without"
---
Utilizing CUDA capabilities in MATLAB without direct kernel implementation hinges on leveraging pre-compiled CUDA libraries or compiled C/C++ code that interacts with CUDA, then making that interaction available to MATLAB through its MEX-file interface. Direct kernel writing within the MATLAB environment, while possible with Parallel Computing Toolbox functionalities, is not the only path. My experiences with large-scale image processing pipelines have led me to often favor this indirect approach for flexibility and performance reasons.

Fundamentally, the challenge lies in bridging MATLAB’s interpreted environment with CUDA's compiled execution model. MATLAB is designed for ease of prototyping and numerical computation, whereas CUDA operates on the GPU through explicitly defined kernels, which are low-level functions running on the massively parallel architecture. To sidestep direct kernel writing within MATLAB, we exploit existing CUDA libraries, often written in C/C++, where the computationally intensive GPU operations are encapsulated. We then compile these C/C++ libraries into shared object files (or DLLs on Windows) and create a corresponding MEX-file interface that acts as a translator between MATLAB and the underlying library.

The general process involves four main steps: (1) Writing the C/C++ code that incorporates CUDA calls, (2) compiling this code into a shared library, (3) Creating a MATLAB MEX-file to interface with the library, and (4) utilizing the MEX-file from MATLAB. The crucial point is that the computationally intensive CUDA computations are handled outside of the MATLAB environment, which calls the precompiled function via the MEX-file interface.

For illustration, let's first consider a simple example using a hypothetical CUDA-accelerated library for array addition, named "gpu_math_lib". This library would have a C function like "add_arrays_gpu(float *a, float *b, float *c, int size)".  The kernel responsible for the actual addition would be within the library. MATLAB’s role is limited to data transfer and function calls using the MEX interface.

Here’s the C/C++ code for a simplified add_arrays_gpu function, which would be part of our `gpu_math_lib` and compiled separately:

```c++
// gpu_math.cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Device (GPU) kernel for array addition
__global__ void add_arrays_kernel(float *a, float *b, float *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

extern "C" {

  // CPU-side wrapper to call the GPU kernel
  void add_arrays_gpu(float *a, float *b, float *c, int size) {
      int threadsPerBlock = 256;
      int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

      float *d_a, *d_b, *d_c;

      // Allocate device memory
      cudaMalloc((void**)&d_a, size * sizeof(float));
      cudaMalloc((void**)&d_b, size * sizeof(float));
      cudaMalloc((void**)&d_c, size * sizeof(float));

      // Copy data from host to device
      cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

      // Launch the kernel
      add_arrays_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

      // Copy the result from device to host
      cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

      // Free device memory
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
  }
}
```

This code illustrates the CUDA kernel and its host-side wrapper function. It’s compiled into a shared library using a CUDA compiler (like `nvcc`) with appropriate flags. The key point is that this C++ code, including the CUDA kernel, runs completely independently of MATLAB.

Second, let's examine the corresponding MEX-file interface (named, for example, `mex_gpu_add.cpp`) which allows MATLAB to call `add_arrays_gpu`.

```c++
// mex_gpu_add.cpp
#include "mex.h" // MATLAB MEX API
#include "gpu_math.h" // Include the header for our CUDA library

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check for correct number of inputs and outputs
    if (nrhs != 2 || nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:add_arrays_gpu:nargin",
                         "Two inputs required, one output.");
    }

    // Check if inputs are single-precision floating-point arrays
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1])) {
        mexErrMsgIdAndTxt("MyToolbox:add_arrays_gpu:invalidInputType",
                           "Inputs must be single-precision arrays.");
    }

    // Get dimensions
    size_t sizeA = mxGetNumberOfElements(prhs[0]);
    size_t sizeB = mxGetNumberOfElements(prhs[1]);

    if (sizeA != sizeB){
        mexErrMsgIdAndTxt("MyToolbox:add_arrays_gpu:invalidSize",
                           "Input arrays must have the same size.");
    }

    int size = static_cast<int>(sizeA);

    // Get input array pointers
    float *a = (float *)mxGetPr(prhs[0]);
    float *b = (float *)mxGetPr(prhs[1]);


    // Create output array in MATLAB
    plhs[0] = mxCreateNumericMatrix(1, size, mxSINGLE_CLASS, mxREAL);

    // Get output array pointer
    float *c = (float *)mxGetPr(plhs[0]);

    // Call the CUDA function
    add_arrays_gpu(a, b, c, size);
}
```

This MEX-file code receives MATLAB arrays, extracts their pointers, and passes those pointers along with the array size to the CUDA function `add_arrays_gpu`. The result is then copied back to the allocated output MATLAB array. The MATLAB user interacts with this mex function.

Finally, to illustrate a more practical scenario, consider the case of a pre-compiled library for a custom image filter. We’d follow the same four step process. This time, let's envision a C++ library that contains a CUDA implementation for a Gaussian filter, with function signature `gaussian_filter_gpu(float* image, float* filtered_image, int rows, int cols, float sigma)`.  The MEX-file interface (named something like `mex_gpu_gaussian.cpp`) would pass the image data, dimensions, and sigma value to this library. Here is a conceptual snippet illustrating how the MEX-file would facilitate the function call:

```c++
// mex_gpu_gaussian.cpp (Conceptual)
#include "mex.h" // MATLAB MEX API
#include "image_processing_lib.h" // Assume the filter functions is here

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input argument checks similar to the previous example ...
    // ...

    // Get the image and dimensions from the input arguments
    float* input_image = (float*)mxGetPr(prhs[0]);
    int rows = mxGetM(prhs[0]);
    int cols = mxGetN(prhs[0]);
    float sigma = (float)mxGetScalar(prhs[1]);


     // Create an output array for the filtered image
    plhs[0] = mxCreateNumericMatrix(rows, cols, mxSINGLE_CLASS, mxREAL);
    float* output_image = (float*)mxGetPr(plhs[0]);


    // Call the pre-compiled CUDA gaussian function.
    gaussian_filter_gpu(input_image, output_image, rows, cols, sigma);


   // MATLAB automatically handles memory management for plhs[0]
}
```

Crucially, the CUDA kernel for the Gaussian filter is encapsulated within the `image_processing_lib.h` and its compiled library. The MEX-file simply acts as a thin wrapper that facilitates data transfer between MATLAB and the CUDA library.

To summarize, I’ve found that bypassing direct MATLAB kernel implementation using this approach is often beneficial. It allows for the use of highly optimized, pre-existing CUDA libraries or the integration of custom, finely tuned CUDA code, compiled separately for maximum performance. MATLAB, in this case, acts as the orchestrator, controlling data transfer and invoking the necessary functions but relying on an external code base to accomplish the heavy computational lifting.

For further information, I recommend examining books on CUDA programming. Resources focusing on C++ libraries designed for high-performance computation can also prove useful. Additionally, reading detailed documentation for MATLAB's MEX interface and the CUDA toolkit is essential for understanding the technicalities of interaction between MATLAB and CUDA code. Finally, practical projects applying this technique will solidify understanding.
