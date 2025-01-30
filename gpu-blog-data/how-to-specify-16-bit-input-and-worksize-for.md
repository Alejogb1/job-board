---
title: "How to specify 16-bit input and workSize for cuFFT's `cufftXtMakePlanMany` on a single GPU?"
date: "2025-01-30"
id: "how-to-specify-16-bit-input-and-worksize-for"
---
The primary challenge when using `cufftXtMakePlanMany` with 16-bit input on a single GPU stems from the fact that cuFFT's native transform functions do not directly support 16-bit floating-point types (half-precision or `fp16`). Therefore, proper data conversion and intermediary buffer management become critical aspects of the implementation. I've encountered this issue firsthand while working on a real-time signal processing application that demanded both high throughput and minimal memory footprint, which led me to optimize for `fp16`.

Fundamentally, `cufftXtMakePlanMany` provides an extended interface, enabling batch Fourier transforms of multi-dimensional data. However, directly supplying a 16-bit array pointer as input to the cuFFT plan will result in errors, as it only recognizes float (`float`), double (`double`), complex float (`cuComplex`), and complex double (`cuDoubleComplex`). To work around this, the general approach involves the following steps: first, convert the 16-bit input to 32-bit floats (or complex floats); then, perform the FFT using cuFFT’s standard types; and lastly, convert the result back to 16-bit, if required. This conversion must happen within host memory, and then the host memory is passed to the GPU. The `workSize` parameter, in this scenario, is primarily concerned with the cuFFT workspace requirements when processing the 32-bit float data, and not the 16-bit input itself. The core of my approach has always been to leverage intermediate, explicitly allocated 32-bit buffers for the calculation.

To make this concrete, consider a one-dimensional FFT of `N` 16-bit floating-point values. The input data, designated `input_fp16`, is stored as a linear array in memory with data type of `half` (`fp16`). We need to convert this array to 32-bit floats, let's call the output array `input_fp32`, and then execute the FFT on `input_fp32` and output to `output_fp32` of same type, and finally, convert it back to `output_fp16` of data type `half`.

Here is an example of how the cuFFT planning and execution would look in a C++ context.

```cpp
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <half.hpp> // For using half precision, https://github.com/boostorg/half

void runCufftFft(const std::vector<half>& input_fp16, std::vector<half>& output_fp16) {
    const int N = input_fp16.size();
    std::vector<float> input_fp32(N);
    std::vector<cuComplex> output_fp32(N); // Complex output

    // 1. Convert half-precision (fp16) to single-precision (fp32) float
    std::transform(input_fp16.begin(), input_fp16.end(), input_fp32.begin(), [](const half& val){
        return static_cast<float>(val);
    });

    // 2. Allocate GPU memory for both input and output in 32-bit float
    float* d_input_fp32;
    cuComplex* d_output_fp32;
    cudaMalloc((void**)&d_input_fp32, N * sizeof(float));
    cudaMalloc((void**)&d_output_fp32, N * sizeof(cuComplex));

    // 3. Copy input data from host to device
    cudaMemcpy(d_input_fp32, input_fp32.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Create cuFFT plan
    cufftHandle plan;
    int rank = 1;
    int n[] = { N };
    int istride = 1;
    int ostride = 1;
    int idist = N;
    int odist = N;
    int inembed[] = {N};
    int onembed[] = {N};
    cufftXtMakePlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, CUFFT_BATCHED, 1,
                      d_input_fp32, d_output_fp32, CUFFT_XT_FORMAT_32F, CUFFT_XT_FORMAT_32F_COMPLEX);
    size_t workSize;
    cufftGetSize(plan, &workSize);


    // 5. Execute the FFT
    cufftExecR2C(plan, d_input_fp32, d_output_fp32);

    // 6. Copy output data back to host
     cudaMemcpy(output_fp32.data(), d_output_fp32, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // 7. Convert single-precision (fp32) complex output to half-precision (fp16) complex output
    output_fp16.resize(N);
    std::transform(output_fp32.begin(), output_fp32.end(), output_fp16.begin(), [](const cuComplex& val) {
    // For this simple example, discarding the imaginary component.
      return half(val.x);
    });

    // 8. Clean up memory
    cufftDestroy(plan);
    cudaFree(d_input_fp32);
    cudaFree(d_output_fp32);
}

int main() {
    const int N = 1024;
    std::vector<half> input_fp16(N);
    std::vector<half> output_fp16(N);
    // populate input_fp16 with some data
    for (int i = 0; i < N; ++i){
        input_fp16[i] = half(sin(2.0 * M_PI * i / N));
    }

    runCufftFft(input_fp16, output_fp16);

  // Print the first few output elements
    std::cout << "First few output elements (fp16):" << std::endl;
      for (int i = 0; i < std::min(10, (int)output_fp16.size()); ++i) {
        std::cout << static_cast<float>(output_fp16[i]) << " ";
      }
      std::cout << std::endl;

    return 0;
}

```

In this code, we first convert the `half` precision input data to `float`, and copy it over to the GPU, while keeping the output in the complex float format. The `cufftXtMakePlanMany` is given type information through the last two arguments, using the 32-bit format. We also query the required workspace by using `cufftGetSize`. After FFT execution, we copy the result back to the host in a vector of `cuComplex` data type and convert to `half` for output. The important aspect to notice is that `workSize` is not directly relevant to our 16-bit input size, but the size needed to support the 32-bit FFT calculation.

Now, consider another case where we need to perform a 2D FFT, still using 16-bit input. Here, `input_fp16` is a two-dimensional array.

```cpp
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <half.hpp>


void runCufftFft2D(const std::vector<std::vector<half>>& input_fp16, std::vector<std::vector<half>>& output_fp16) {
    const int rows = input_fp16.size();
    const int cols = (rows > 0) ? input_fp16[0].size() : 0;
    if (rows == 0 || cols == 0) {
        std::cerr << "Input matrix must have dimensions greater than 0." << std::endl;
        return;
    }

    std::vector<std::vector<float>> input_fp32(rows, std::vector<float>(cols));
    std::vector<std::vector<cuComplex>> output_fp32(rows, std::vector<cuComplex>(cols));


    // 1. Convert each element of the half-precision input to single-precision float
    for(int i = 0; i < rows; i++){
      std::transform(input_fp16[i].begin(), input_fp16[i].end(), input_fp32[i].begin(), [](const half& val) {
            return static_cast<float>(val);
        });
    }

    // 2. Allocate GPU memory for input and output in 32-bit float
    float* d_input_fp32;
    cuComplex* d_output_fp32;
    cudaMalloc((void**)&d_input_fp32, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output_fp32, rows * cols * sizeof(cuComplex));

    // 3. Flatten input and copy to GPU
    std::vector<float> flat_input;
    for(const auto& row: input_fp32) {
      flat_input.insert(flat_input.end(), row.begin(), row.end());
    }
    cudaMemcpy(d_input_fp32, flat_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Create cuFFT plan
    cufftHandle plan;
    int rank = 2;
    int n[] = { rows, cols };
    int istride = 1;
    int ostride = 1;
    int idist = cols;
    int odist = cols;
    int inembed[] = {rows, cols};
    int onembed[] = {rows, cols};


    cufftXtMakePlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, CUFFT_BATCHED, 1,
                      d_input_fp32, d_output_fp32, CUFFT_XT_FORMAT_32F, CUFFT_XT_FORMAT_32F_COMPLEX);

    size_t workSize;
    cufftGetSize(plan, &workSize);

    // 5. Execute the FFT
    cufftExecR2C(plan, d_input_fp32, d_output_fp32);

    // 6. Copy output back to host
     std::vector<cuComplex> flat_output(rows * cols);
     cudaMemcpy(flat_output.data(), d_output_fp32, rows * cols * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // 7. Convert output to a 2D structure of half
    output_fp16.resize(rows, std::vector<half>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j){
            output_fp16[i][j] = half(flat_output[i * cols + j].x);
        }
    }

    // 8. Clean up
    cufftDestroy(plan);
    cudaFree(d_input_fp32);
    cudaFree(d_output_fp32);
}


int main() {
    const int rows = 32;
    const int cols = 64;
    std::vector<std::vector<half>> input_fp16(rows, std::vector<half>(cols));
    std::vector<std::vector<half>> output_fp16;

    // Initialize input data (for example, using a simple pattern)
      for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
              input_fp16[i][j] = half(sin(2.0 * M_PI * i / rows + 2.0 * M_PI * j / cols));
          }
      }

    runCufftFft2D(input_fp16, output_fp16);

    // Print the first few output elements
      std::cout << "First few output elements (fp16):" << std::endl;
      for (int i = 0; i < std::min(5, (int)output_fp16.size()); ++i) {
        for(int j = 0; j < std::min(5, (int)output_fp16[i].size()); ++j){
            std::cout << static_cast<float>(output_fp16[i][j]) << " ";
        }
         std::cout << std::endl;
      }


    return 0;
}
```

In this example, the logic is extended to 2D FFTs. The input matrix is flattened before copying to the device, and then reshaped back in the host memory after execution and back-conversion to `half` type. Again, the `workSize` is calculated based on the 32-bit data used by `cufftXtMakePlanMany`. The key idea remains that the provided 16-bit input type is not directly recognized by cuFFT.

Finally, to demonstrate how batching can be implemented with `cufftXtMakePlanMany`, consider performing multiple 1D FFTs simultaneously.

```cpp
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <half.hpp>

void runBatchedCufftFft(const std::vector<std::vector<half>>& input_fp16, std::vector<std::vector<half>>& output_fp16) {
    const int batchSize = input_fp16.size();
    const int N = (batchSize > 0 && !input_fp16[0].empty()) ? input_fp16[0].size() : 0;

    if (batchSize == 0 || N == 0) {
        std::cerr << "Input must have batchSize and N greater than 0." << std::endl;
        return;
    }

    std::vector<std::vector<float>> input_fp32(batchSize, std::vector<float>(N));
    std::vector<std::vector<cuComplex>> output_fp32(batchSize, std::vector<cuComplex>(N));


    for(int b = 0; b < batchSize; b++){
         std::transform(input_fp16[b].begin(), input_fp16[b].end(), input_fp32[b].begin(), [](const half& val){
            return static_cast<float>(val);
        });
    }


    float* d_input_fp32;
    cuComplex* d_output_fp32;

    cudaMalloc((void**)&d_input_fp32, batchSize * N * sizeof(float));
    cudaMalloc((void**)&d_output_fp32, batchSize * N * sizeof(cuComplex));

    std::vector<float> flat_input;
    for(const auto& vec: input_fp32){
        flat_input.insert(flat_input.end(), vec.begin(), vec.end());
    }

    cudaMemcpy(d_input_fp32, flat_input.data(), batchSize * N * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle plan;
    int rank = 1;
    int n[] = { N };
    int istride = 1;
    int ostride = 1;
    int idist = N;
    int odist = N;
    int inembed[] = { N };
    int onembed[] = { N };
    int batch = batchSize;


    cufftXtMakePlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, CUFFT_BATCHED, batch,
                      d_input_fp32, d_output_fp32, CUFFT_XT_FORMAT_32F, CUFFT_XT_FORMAT_32F_COMPLEX);

    size_t workSize;
    cufftGetSize(plan, &workSize);

    cufftExecR2C(plan, d_input_fp32, d_output_fp32);

    std::vector<cuComplex> flat_output(batchSize * N);
    cudaMemcpy(flat_output.data(), d_output_fp32, batchSize * N * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    output_fp16.resize(batchSize, std::vector<half>(N));
    for(int b = 0; b < batchSize; ++b){
         for (int i = 0; i < N; ++i) {
            output_fp16[b][i] = half(flat_output[b * N + i].x);
         }
    }

    cufftDestroy(plan);
    cudaFree(d_input_fp32);
    cudaFree(d_output_fp32);
}


int main() {
    const int batchSize = 4;
    const int N = 128;
    std::vector<std::vector<half>> input_fp16(batchSize, std::vector<half>(N));
    std::vector<std::vector<half>> output_fp16;

    // Initialize input data (for example, using a simple pattern)
        for (int b = 0; b < batchSize; ++b) {
          for (int i = 0; i < N; ++i) {
            input_fp16[b][i] = half(sin(2.0 * M_PI * i / N + (double)b/batchSize ));
          }
        }

    runBatchedCufftFft(input_fp16, output_fp16);

      std::cout << "First few output elements (fp16):" << std::endl;
      for (int b = 0; b < std::min(2, (int)output_fp16.size()); ++b) {
        for(int i = 0; i < std::min(5, (int)output_fp16[b].size()); ++i){
             std::cout << static_cast<float>(output_fp16[b][i]) << " ";
        }
         std::cout << std::endl;
      }

    return 0;
}
```

Here, the code is expanded to handle batched 1D FFTs, where we effectively perform multiple independent 1D FFTs concurrently within the cuFFT call. The `batch` parameter specifies the number of transforms to execute simultaneously. This demonstrates that `workSize` continues to be primarily driven by the temporary storage needed for 32-bit data even when batching is implemented, instead of the original 16-bit data size.

For further study, I recommend looking into the NVIDIA cuFFT documentation, focusing on data type conversions, memory management patterns for device to host data transfers, and the specific usage of `cufftXtMakePlanMany`. The CUDA Toolkit Programming Guide offers detailed insights on memory allocation strategies using `cudaMalloc` and `cudaMemcpy`. For more complex scenarios, exploration of different data layouts and advanced cuFFT features will be beneficial. Additionally, examining high performance computing literature related to memory-optimized FFT implementations could reveal valuable patterns.
