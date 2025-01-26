---
title: "How to use the {N, H, W, C} format in custom PyTorch operations?"
date: "2025-01-26"
id: "how-to-use-the-n-h-w-c-format-in-custom-pytorch-operations"
---

Understanding the {N, H, W, C} data format is fundamental when designing custom operations in PyTorch, especially when working with image data or other multi-dimensional tensors exhibiting a spatial structure. This format, representing *batch size*, *height*, *width*, and *channels*, dictates how the data is arranged in memory and consequently impacts how custom PyTorch functions, implemented through C++ or CUDA extensions, should process the data. I've encountered firsthand how mismatched assumptions regarding data layout can lead to subtle errors in data processing that might be challenging to identify, especially when optimizing for GPU acceleration.

The {N, H, W, C} format, commonly used in deep learning for image processing, contrasts with the standard PyTorch tensor layout of {N, C, H, W}. While PyTorch itself generally utilizes the latter, the need for the former often arises when integrating with libraries or custom code that assumes this layout. A prime example is when working with image processing pipelines where individual pixel access, or operations that directly benefit from contiguous memory layout of the spatial dimensions, are necessary. This requires that custom C++ or CUDA kernels be explicitly designed to interpret the tensor data as structured according to the {N, H, W, C} convention.

When I implemented a custom deformable convolution operation, initially I incorrectly assumed the data to be formatted in {N, C, H, W} within the CUDA kernel which led to completely incorrect spatial offsets. I ended up using a conversion step before kernel launch, but this resulted in substantial performance degradation. The correct approach involves accessing the tensor elements within the kernel using correct strides. The memory location of a pixel at batch index `n`, height `h`, width `w`, and channel `c` can be calculated as:

`memory_location = n * H * W * C + h * W * C + w * C + c`

This calculation translates the 4-dimensional index to a 1-dimensional memory address, allowing direct and efficient access in low-level code. It is critical to perform these index calculations correctly to guarantee correct and performant functionality.

To illustrate, consider three specific situations encountered in implementing custom PyTorch operations: a simple per-pixel addition, a spatial pooling operation, and a more complex channel manipulation.

**Example 1: Per-Pixel Addition**

The following C++ code snippet demonstrates how to implement a simple per-pixel addition of a scalar value to a tensor with the {N, H, W, C} layout using a custom CUDA kernel:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

void pixel_add_cuda_kernel(float *data, int N, int H, int W, int C, float scalar) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * H * W * C) {
      int n = index / (H * W * C);
      int remaining = index % (H * W * C);
      int h = remaining / (W * C);
      remaining = remaining % (W * C);
      int w = remaining / C;
      int c = remaining % C;
    data[index] += scalar;
  }
}

void pixel_add_cuda(torch::Tensor &input, float scalar) {
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);

    float *data = input.data_ptr<float>();
    cudaError_t error;
    pixel_add_cuda_kernel<<< (N * H * W * C + 255) / 256 , 256 >>>(data, N, H, W, C, scalar);
    error = cudaGetLastError();
     if (error != cudaSuccess)
     {
          std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
     }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pixel_add_cuda", &pixel_add_cuda, "Per-pixel addition (CUDA)");
}
```

*Commentary:* This code defines a CUDA kernel `pixel_add_cuda_kernel` that adds a scalar value to every element of the tensor. The critical aspect is the calculation of `n`, `h`, `w`, and `c` from the flat `index` using division and modulo operations. This ensures correct spatial and channel alignment. In the Python binding, the necessary size parameters are extracted from the PyTorch tensor and the kernel is launched, ensuring correct grid and block sizes. It is critical to perform error checking after each CUDA kernel launch.

**Example 2: Spatial Pooling**

This example illustrates a max pooling operation performed directly on a {N, H, W, C} tensor:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

void max_pool_cuda_kernel(float *input, float *output, int N, int H, int W, int C, int kernel_size, int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int outputH = (H - kernel_size) / stride + 1;
    int outputW = (W - kernel_size) / stride + 1;

  if (index < N * outputH * outputW * C)
   {
    int n = index / (outputH * outputW * C);
    int remaining = index % (outputH * outputW * C);
    int oh = remaining / (outputW * C);
    remaining = remaining % (outputW * C);
    int ow = remaining / C;
    int c = remaining % C;
    float max_val = -std::numeric_limits<float>::infinity();
    for(int kh = 0; kh < kernel_size; kh++){
        for(int kw = 0; kw < kernel_size; kw++){
            int h = oh * stride + kh;
            int w = ow * stride + kw;
            int input_idx = n * H * W * C + h * W * C + w * C + c;

            max_val = std::max(max_val, input[input_idx]);
        }
    }
     output[index] = max_val;
    }
}

void max_pool_cuda(torch::Tensor &input, torch::Tensor &output, int kernel_size, int stride) {
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
    int outputH = (H - kernel_size) / stride + 1;
    int outputW = (W - kernel_size) / stride + 1;
    float *input_data = input.data_ptr<float>();
    float *output_data = output.data_ptr<float>();
    cudaError_t error;
    max_pool_cuda_kernel<<< (N * outputH * outputW * C + 255) / 256 , 256 >>>(input_data,output_data, N, H, W, C, kernel_size, stride);
     error = cudaGetLastError();
      if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_pool_cuda", &max_pool_cuda, "Max Pooling (CUDA)");
}
```

*Commentary:* This example shows how to perform pooling. Again, note the calculation of `n`, `oh`, `ow`, and `c` to correspond to the output index, while `h` and `w`  are calculated to retrieve input values inside the kernel window, reflecting the correct use of strides. We extract these from the appropriate output dimensions to ensure we iterate through the output tensor. We also correctly calculate the output height and width. The kernel's access to memory using these indices is crucial for the correct outcome of the pooling operation.

**Example 3: Channel Manipulation**

Finally, consider a scenario where channels need to be rearranged, here using a simple channel shuffle operation.

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

void channel_shuffle_cuda_kernel(float *input, float* output, int N, int H, int W, int C, int groups) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N * H * W * C) {
         int n = index / (H * W * C);
        int remaining = index % (H * W * C);
        int h = remaining / (W * C);
        remaining = remaining % (W * C);
        int w = remaining / C;
        int c = remaining % C;
        int group_size = C / groups;

        int c_prime = (c / group_size) + (c % group_size) * groups;

        int input_idx = n * H * W * C + h * W * C + w * C + c;
        int output_idx = n * H * W * C + h * W * C + w * C + c_prime;

        output[output_idx] = input[input_idx];
    }
}

void channel_shuffle_cuda(torch::Tensor &input, torch::Tensor &output, int groups) {
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
     float *input_data = input.data_ptr<float>();
     float *output_data = output.data_ptr<float>();
    cudaError_t error;
     channel_shuffle_cuda_kernel<<< (N * H * W * C + 255) / 256 , 256 >>>(input_data, output_data,N, H, W, C, groups);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("channel_shuffle_cuda", &channel_shuffle_cuda, "Channel Shuffle (CUDA)");
}
```

*Commentary:* This kernel implements a channel shuffle operation, which permutes channels based on a given number of groups. The critical component is the calculation of `c_prime`, the shuffled channel index, which determines where each input channel will be written in the output tensor.  It must be calculated based on the number of groups and correctly index the channel in the output tensor. Like previous examples, correct error handling is paramount when using CUDA kernels.

To delve deeper into related concepts and implementation best practices, resources such as the official CUDA programming guide, and materials on parallel algorithms for GPUs are excellent choices. Investigating books on high-performance computing and optimization techniques also offer valuable insight. Furthermore, examining the source code of popular deep learning libraries can offer further real-world examples and best practices. Always verify your custom operations against known correct implementations for sanity checking and use rigorous testing to identify any potential bugs.
