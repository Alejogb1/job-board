---
title: "Does CUDA Thrust support FP16 data types?"
date: "2025-01-30"
id: "does-cuda-thrust-support-fp16-data-types"
---
Yes, CUDA Thrust supports FP16 data types, but with specific considerations regarding hardware support and library versions. My experience working on GPU-accelerated machine learning inference pipelines has required me to carefully evaluate the performance implications of different precision levels. The use of FP16 (half-precision floating-point) provides significant performance gains compared to FP32 (single-precision), primarily due to reduced memory bandwidth requirements and increased compute throughput on newer NVIDIA architectures. However, this gain is contingent on the underlying hardware and Thrust library capabilities.

**Explanation of FP16 Support in CUDA Thrust**

Thrust, a C++ template library designed for CUDA, provides a high-level interface for parallel algorithms, simplifying GPU programming without requiring extensive CUDA kernel writing. The library's support for FP16 is not implicit in all functions across all versions. Crucially, the use of FP16 often relies on the existence of dedicated Tensor Cores within the GPU architecture. Tensor Cores are specialized units optimized for matrix multiplication, and half-precision computations are their core functionality. Without these cores, FP16 operations fall back to slower emulated routines, negating the potential performance boost. Therefore, the hardware's compute capability is paramount when deciding to use FP16. NVIDIA GPUs with compute capability 7.0 and above (Volta, Turing, Ampere, and later architectures) provide robust Tensor Core support for FP16 computations, enabling high performance.

The Thrust library exposes the `__half` data type (defined in `cuda_fp16.h`) as the primary representation for FP16 values. Functionality such as reductions, scans, and transformations can be adapted to use `__half` as the input and output data type. The specific functions supporting FP16 operations are primarily those built upon highly optimized CUDA primitives, such as `thrust::reduce`, `thrust::transform`, and `thrust::scan`.

However, a crucial consideration is the interoperability of FP16 with standard C++ types. Because `__half` is a CUDA-specific type, explicit type conversions may be required when moving data between the host (CPU) and the device (GPU). One often converts standard floating-point representations to `__half` on the host before transmitting them to the GPU for processing. I have found that careful consideration of such data movement minimizes potential bottlenecks.

**Code Examples with Commentary**

Below are three code examples illustrating Thrust's FP16 support with accompanying explanations:

**Example 1: Reducing an array of `__half` values**

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

int main() {
  // Create a host vector of float values
  std::vector<float> host_vec_float = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // Convert float values to half precision
  std::vector<__half> host_vec_half(host_vec_float.size());
  for (size_t i = 0; i < host_vec_float.size(); ++i) {
      host_vec_half[i] = __float2half(host_vec_float[i]);
  }

  // Create a device vector and transfer the half precision data to the GPU
  thrust::device_vector<__half> device_vec(host_vec_half);

  // Reduce the device vector
  __half sum = thrust::reduce(device_vec.begin(), device_vec.end(), __float2half(0.0f));

    // Convert the result back to float for printing
    float sum_float = __half2float(sum);

    std::cout << "Sum of half-precision vector: " << sum_float << std::endl;
  return 0;
}
```

**Commentary:** This example demonstrates a basic reduction operation using `thrust::reduce` with `__half` data type. It showcases the explicit conversion from standard float values on the host (`float`) to `__half` before moving the data to the device. I also included the reverse conversion to display the final result in a user-friendly format. The initialization value for the reduction operation is also provided as a half-precision value. Such explicit data conversions between host and device are essential for correct usage of FP16 data with Thrust.

**Example 2: Element-wise transformation with `__half` data**

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda_fp16.h>
#include <iostream>

// Unary function for element-wise square operation
struct SquareHalf {
  __host__ __device__ __half operator()(__half x) {
    return x * x;
  }
};

int main() {
    // Create a host vector of float values
    std::vector<float> host_vec_float = {1.0f, 2.0f, 3.0f, 4.0f};

    // Convert float values to half precision
    std::vector<__half> host_vec_half(host_vec_float.size());
    for (size_t i = 0; i < host_vec_float.size(); ++i) {
        host_vec_half[i] = __float2half(host_vec_float[i]);
    }

    // Create a device vector and transfer the half precision data to the GPU
  thrust::device_vector<__half> device_vec(host_vec_half);

  // Create a output vector for transformed result
  thrust::device_vector<__half> transformed_vec(device_vec.size());

  // Transform the device vector
  thrust::transform(device_vec.begin(), device_vec.end(), transformed_vec.begin(), SquareHalf());

    // Copy the transformed vector back to the host
    std::vector<__half> host_transformed_vec(transformed_vec.size());
    thrust::copy(transformed_vec.begin(), transformed_vec.end(), host_transformed_vec.begin());


  // Print the transformed values after conversion to float
    std::cout << "Transformed half-precision vector: ";
    for(const auto& half_val : host_transformed_vec){
        std::cout << __half2float(half_val) << " ";
    }
    std::cout << std::endl;

  return 0;
}
```

**Commentary:** This example demonstrates the use of `thrust::transform` with FP16 data. I defined a custom functor `SquareHalf` to perform an element-wise square operation on `__half` values. The core Thrust function, along with the device memory allocation and transfers, handle the parallelization on the GPU. The key takeaway is the ability to apply custom, highly optimized operations on the half-precision data directly on the GPU via Thrust.

**Example 3: Combining `thrust::zip_iterator` and `__half` data**

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cuda_fp16.h>
#include <iostream>

// Binary function to perform element-wise sum of two half precision numbers
struct HalfAdd {
    __host__ __device__ __half operator()(thrust::tuple<__half, __half> t) {
      return thrust::get<0>(t) + thrust::get<1>(t);
    }
};


int main() {

    // Host vectors
    std::vector<float> host_vec_float_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> host_vec_float_b = {5.0f, 6.0f, 7.0f, 8.0f};

    // Convert to half precision
    std::vector<__half> host_vec_half_a(host_vec_float_a.size());
    std::vector<__half> host_vec_half_b(host_vec_float_b.size());

    for (size_t i = 0; i < host_vec_float_a.size(); ++i) {
        host_vec_half_a[i] = __float2half(host_vec_float_a[i]);
        host_vec_half_b[i] = __float2half(host_vec_float_b[i]);
    }

  // Device vectors
  thrust::device_vector<__half> device_vec_a(host_vec_half_a);
  thrust::device_vector<__half> device_vec_b(host_vec_half_b);

  // Result vector
  thrust::device_vector<__half> device_result_vec(device_vec_a.size());

  // Use thrust::zip_iterator for element-wise add
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(device_vec_a.begin(), device_vec_b.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(device_vec_a.end(), device_vec_b.end())),
      device_result_vec.begin(),
      HalfAdd()
      );

    // Copy the transformed vector back to the host
    std::vector<__half> host_result_vec(device_result_vec.size());
    thrust::copy(device_result_vec.begin(), device_result_vec.end(), host_result_vec.begin());

  // Print results after conversion to float
    std::cout << "Element-wise sum using zip iterator: ";
    for(const auto& half_val : host_result_vec){
        std::cout << __half2float(half_val) << " ";
    }
    std::cout << std::endl;

  return 0;
}
```

**Commentary:**  This example uses `thrust::zip_iterator` to perform an element-wise addition of two half-precision vectors, further illustrating Thrust’s flexibility with `__half` data types. The `HalfAdd` functor performs the addition of two half-precision values encapsulated in a `thrust::tuple`.  The use of zip iterators is common in parallel algorithms for handling element-wise operations on multiple input arrays, and this example confirms that this technique is readily available with FP16 data in Thrust.

**Resource Recommendations**

For more in-depth information on CUDA Thrust and its support for FP16, I recommend consulting the following resources:

1.  **CUDA Toolkit Documentation:** The official CUDA documentation, particularly the sections on the Thrust library and `cuda_fp16.h`, provide the most up-to-date and accurate information. Reviewing the API documentation and performance recommendations is crucial.

2.  **NVIDIA Developer Blog:** The NVIDIA Developer Blog often publishes articles covering performance optimization strategies for CUDA, including guidance on how to use FP16 effectively, especially in scenarios like deep learning.

3.  **CUDA Programming Guides:** Standard books and guides on CUDA programming will cover the foundational aspects of the architecture and library, especially how to manage memory transfers and understand hardware limitations. Pay close attention to the section on half-precision floating-point.
