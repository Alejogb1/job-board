---
title: "How can I begin CUDA development on Ubuntu 9.04?"
date: "2025-01-30"
id: "how-can-i-begin-cuda-development-on-ubuntu"
---
Ubuntu 9.04, Jaunty Jackalope, presents unique challenges for modern CUDA development given its age, requiring a specific approach to establish a functional environment. The primary hurdle involves compatibility issues with current CUDA toolkits and drivers. I recall wrestling with similar constraints while building a proof-of-concept fluid dynamics simulator back in 2010, which involved getting CUDA 3.0 to function on aging hardware. This experience informs my recommendations.

The key to achieving any measure of CUDA development on Ubuntu 9.04 revolves around procuring and correctly installing older versions of the CUDA toolkit and the corresponding NVIDIA drivers. The latest iterations are inherently incompatible. Trying to force newer versions would result in kernel module conflicts, missing dependencies, and ultimately, a non-functional CUDA system. I have personally seen this result in kernel panics and system instability.

Here’s a breakdown of the process:

1.  **Driver Acquisition:** Begin by identifying the latest NVIDIA driver version officially compatible with your graphics card *and* supported by CUDA versions released around the Ubuntu 9.04 timeframe. NVIDIA driver archives are the ideal source for these legacy drivers. Do not attempt to download drivers directly from the modern website as they will be mismatched. Research is key here, checking both your GPU model and the historical compatibility charts provided within the NVIDIA support forums. For context, around 2009/2010, the GeForce 8 series through 200 series cards were common, and drivers like version 195 or 256 were relevant for those GPUs. You may have to go back further. Once identified, download the driver `.run` installer file.

2.  **Driver Installation:** Before executing the `.run` file, you need to halt the X server. This can usually be achieved with a command like `sudo service gdm stop` or `sudo /etc/init.d/gdm stop`. The exact command might depend on your desktop environment setup. Once X is stopped, execute the driver installer with `sudo sh NVIDIA-Linux-x86-xxx.xx.run`, replacing "xxx.xx" with the appropriate version number. I suggest carefully following the on-screen prompts and, if presented, choosing the option to build the NVIDIA kernel modules. In my experience, selecting the recommended options will save headaches later. Pay special attention to error messages, noting any missing kernel headers or build tools. A reboot is necessary after installation.

3.  **CUDA Toolkit Installation:** After installing the correct drivers, download the appropriate legacy version of the CUDA toolkit. I remember working directly with the Nvidia website's archive pages for this purpose. Target a CUDA toolkit version that matches both the driver version and your desired CUDA capability. For example, if you have drivers around the 256 series, CUDA toolkit 3.0 or 3.1 may be your target. Older toolkits tend to be available as `.run` installers as well. As with drivers, install the toolkit as root via `sudo sh cudatoolkit-x.x.run`. The installation will likely include header files, libraries, and the `nvcc` compiler. It is important to follow the installation prompts and pay attention to any instructions regarding environment variable setup.

4.  **Environment Setup:** Once both driver and toolkit are installed, you might need to manually configure your system to locate the CUDA tools. This involves modifying your `.bashrc` file or a system-wide profile to include the relevant CUDA paths. The most crucial variables are `PATH` and `LD_LIBRARY_PATH`. `PATH` will need to include the directory where `nvcc` lives. `LD_LIBRARY_PATH` should point to the directory where the CUDA libraries such as `libcuda.so` reside. Failure to properly set these will render the compiler and linker unable to find the necessary components.

**Code Examples and Commentary:**

Here are three examples illustrating CUDA code compilation using the environment configured above.

**Example 1: Simple Vector Addition**

```c++
// vector_add.cu
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 1024;
  size_t size = n * sizeof(float);
  float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

  // Allocate memory on host
  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c = (float *)malloc(size);

  // Initialize host arrays
  for(int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // Allocate memory on device
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Launch the kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  // Copy results back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify results
    for(int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]){
            printf("Error at %d \n", i);
            return 1;
        }
    }

  printf("Vector addition completed successfully!\n");

  // Free allocated memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
```

*   **Commentary:** This code demonstrates the basic structure of a CUDA program. It allocates memory on both host and device, transfers data, executes a kernel, copies results back, and performs a simple verification. Compilation is done via `nvcc vector_add.cu -o vector_add`.

**Example 2: Matrix Multiplication**

```c++
// matrix_mul.cu
#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 16

__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}


int main() {
    int width = 256;
    size_t size = width * width * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Allocate memory on host
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host matrices
    for(int i = 0; i < width; i++){
        for(int j=0; j < width; j++){
            h_A[i*width + j] = (float)i+j;
            h_B[i*width + j] = (float)i-j;
        }
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verification (simplified)
    float test_value = h_A[1 * width + 0] * h_B[0 * width + 0] + h_A[1*width + 1] * h_B[1*width+0];
    if (h_C[1 * width + 0] != test_value)
    {
        printf("Verification failed \n");
        return 1;
    }

    printf("Matrix multiplication completed successfully!\n");

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

*   **Commentary:** This example showcases a more complex operation: matrix multiplication. It utilizes a two-dimensional grid and block configuration for parallel execution. Note that there are many more efficient ways to perform matrix multiplication in CUDA. This serves as a demonstration of multi-dimensional launching. Compilation: `nvcc matrix_mul.cu -o matrix_mul`.

**Example 3: Reduction**

```c++
// reduction.cu
#include <stdio.h>
#include <cuda.h>

__global__ void reduce(float *g_idata, float *g_odata, int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float mySum = 0;
  if (i < n) {
    mySum = g_idata[i];
  }

  sdata[tid] = mySum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

int main() {
  int n = 1024;
  int blocksize = 256;
  int numblocks = (n + blocksize - 1) / blocksize;
  size_t size = n * sizeof(float);
  size_t outsize = numblocks * sizeof(float);

  float *h_idata, *h_odata, *d_idata, *d_odata;

  h_idata = (float *)malloc(size);
  h_odata = (float *)malloc(outsize);

  for (int i = 0; i < n; i++) {
    h_idata[i] = (float)i;
  }


  cudaMalloc((void **)&d_idata, size);
  cudaMalloc((void **)&d_odata, outsize);

  cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
  reduce<<<numblocks, blocksize, blocksize * sizeof(float)>>>(d_idata, d_odata, n);

  cudaMemcpy(h_odata, d_odata, outsize, cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i=0; i < numblocks; i++){
        sum += h_odata[i];
    }
  printf("Sum of the array: %f\n", sum);

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
  return 0;
}
```
*   **Commentary:** This code demonstrates a reduction using shared memory. Each block computes a partial sum, which are then aggregated on the host. This showcases the use of shared memory and the complexity of reducing data on the device. Compilation is done with `nvcc reduction.cu -o reduction`.

**Resource Recommendations**

For learning CUDA programming basics I recommend, 'Programming Massively Parallel Processors' by David Kirk. For an understanding of CUDA architecture, NVIDIA’s CUDA programming guide is invaluable. Finally, I suggest examining any available sample code included with the specific CUDA toolkit you install; it will provide a solid foundation for learning. Be prepared to spend time on the NVIDIA website's archives, scouring forums, as documentation for those older versions may be sparse.

Working with Ubuntu 9.04 and older CUDA versions can be challenging, but with persistence and a clear understanding of the limitations involved, it’s possible to establish a functional development environment. Prioritize driver compatibility and exact toolkit matching. My experience suggests that these two points are the most critical in a legacy environment.
