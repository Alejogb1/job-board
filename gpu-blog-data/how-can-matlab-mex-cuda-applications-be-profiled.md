---
title: "How can MATLAB mex CUDA applications be profiled using the NVIDIA visual profiler?"
date: "2025-01-30"
id: "how-can-matlab-mex-cuda-applications-be-profiled"
---
The effective profiling of MATLAB MEX CUDA applications with the NVIDIA Visual Profiler requires a careful understanding of the interplay between the MATLAB execution environment, the MEX interface, and the CUDA runtime. Unlike standalone CUDA applications, MEX files execute within MATLAB's process space, necessitating specific strategies for capturing performance data. A key fact is that the Visual Profiler, now superseded by NVIDIA Nsight Compute and Nsight Systems, relies on intercepting CUDA API calls. This interception requires configuring MATLAB to permit the profiler to function. My experience in developing high-performance signal processing algorithms using MATLAB MEX with CUDA has shown that failing to appropriately set up the profiling environment often results in incomplete or inaccurate data.

Fundamentally, to profile MEX CUDA code with NVIDIA’s profiling tools, you must ensure that the target MEX function is launched in a manner that allows the profiler to hook into CUDA calls made by the device code. This involves two primary considerations: (1) ensuring that the NVIDIA driver and associated tools are correctly installed and that the CUDA runtime can be detected; and (2) configuring MATLAB’s environment so that the profiler can intercept CUDA API calls. Since the Visual Profiler is deprecated, I will focus on the successor tools, NVIDIA Nsight Compute and NVIDIA Nsight Systems, as they represent the current best practices. Nsight Compute focuses on kernel-level performance analysis, while Nsight Systems provides a system-wide view including CPU and GPU interactions.

The process typically follows these steps: First, compile your MEX CUDA code ensuring the `-g` flag is used to embed debug information. This is crucial for Nsight Compute to provide source-level correlation. Second, you must set up your execution environment within the MATLAB context.  Since MATLAB effectively manages CUDA contexts, it's not generally necessary to initialize the CUDA context directly, but it is crucial the Nsight profilers can access it. You can indirectly control the environment through system environment variables that MATLAB passes to the underlying operating system.  Third, launch the MATLAB script or function which invokes your MEX file either directly or via Nsight’s command line tool. Finally, analyze the generated reports to identify performance bottlenecks.

The challenge comes from the fact that MATLAB itself is not directly under the user's direct control with respect to system environment variables and context manipulation. As such, one must rely on setting environment variables that affect its child processes, or utilize Nsight directly to execute the MATLAB engine with the appropriate configuration. This will allow NVIDIA’s tools to correctly attach to the MATLAB process and monitor the CUDA API calls made from within the MEX function.

Here are three illustrative examples demonstrating this process:

**Example 1: Kernel Performance Analysis with Nsight Compute**

Let's say we have a simple MEX file named `add_arrays_cuda.cu` that adds two arrays using CUDA:

```c++
// add_arrays_cuda.cu
#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2 || nlhs != 1) {
        mexErrMsgTxt("Usage: C = add_arrays_cuda(A, B)");
    }

    float *a = (float *)mxGetData(prhs[0]);
    float *b = (float *)mxGetData(prhs[1]);
    int n = mxGetNumberOfElements(prhs[0]);

    mxArray *c_mex = mxCreateNumericMatrix(1, n, mxSINGLE_CLASS, mxREAL);
    float *c = (float *)mxGetData(c_mex);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    plhs[0] = c_mex;
}
```

To profile this with Nsight Compute, compile the MEX file first using `mex -v -g add_arrays_cuda.cu -lcuda`. Then, in MATLAB, you would set a system environment variable by calling a function to alter MATLAB's execution environment such as `setenv('CUDA_VISIBLE_DEVICES','0')` to specify which GPU to target (note that this call is usually best executed before the MEX file is called for maximum effect).  Then, execute the MEX function and then, through the Nsight Compute GUI, select "New Project" and choose the “MATLAB” executable as the application to profile. On macOS, one would select the executable `MATLAB_R20XXx.app/Contents/MacOS/MATLAB` whereas, on Windows, it would likely be located at  `C:/Program Files/MATLAB/R20XXx/bin/matlab.exe`. Select your MATLAB script as the input program and profile the call to your MEX function. The Nsight compute will intercept the CUDA calls made inside the MEX function `add_arrays_cuda` and provide detailed metrics on kernel performance and memory access.

**Example 2: System-wide analysis with Nsight Systems**

Nsight Systems provides a more holistic view. In addition to the code snippet from example 1, let’s assume the MEX file is used in the following MATLAB script:

```matlab
% test_add_arrays.m
A = rand(1, 2^20, 'single');
B = rand(1, 2^20, 'single');
C = add_arrays_cuda(A, B);

tic;
C = add_arrays_cuda(A, B);
toc;
```
After compiling the MEX file with debug information as described in Example 1, to profile this with Nsight Systems, you must use the Nsight Systems command line interface. You will have to explicitly specify the command to launch MATLAB, the script to run and the environment variables to correctly attach the NVIDIA profiler.  The command might look like:
`nsys profile --capture-range cuda --stats=true -o profile_output "C:/Program Files/MATLAB/R2023b/bin/matlab.exe" -nosplash -r "run('test_add_arrays.m')" ` (This example is a Windows path to the MATLAB executable.  This syntax can differ between platforms such as macOS or Linux).

The `--capture-range cuda` flag focuses profiling on CUDA API calls, `--stats=true` generates statistics, and `-o profile_output` specifies the output file.  The critical part here is that you must launch the MATLAB executable with the desired commands to call the function. The resulting report from Nsight Systems will allow you to visualize the execution timeline, including CPU activity and CUDA kernel launches and memory transfers done from within `add_arrays_cuda.cu`.  This demonstrates a system-wide performance picture.

**Example 3: Multi-MEX File Profiling**

Often, real applications involve several MEX files which are sequentially invoked from within MATLAB. Let’s assume another MEX file, `multiply_array.cu`, is executed after `add_arrays_cuda.cu`.

```c++
// multiply_array.cu
#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void multiply_kernel(float *a, float scalar, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * scalar;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2 || nlhs != 1) {
        mexErrMsgTxt("Usage: C = multiply_array(A, scalar)");
    }

    float *a = (float *)mxGetData(prhs[0]);
    float scalar = (float)mxGetScalar(prhs[1]);
    int n = mxGetNumberOfElements(prhs[0]);

    mxArray *c_mex = mxCreateNumericMatrix(1, n, mxSINGLE_CLASS, mxREAL);
    float *c = (float *)mxGetData(c_mex);

    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);

    plhs[0] = c_mex;
}
```
And suppose it is used in conjunction with the earlier add MEX function as shown here:
```matlab
% test_multi_mex.m
A = rand(1, 2^20, 'single');
B = rand(1, 2^20, 'single');
C = add_arrays_cuda(A, B);
D = multiply_array(C, 2.5);

tic;
C = add_arrays_cuda(A, B);
D = multiply_array(C, 2.5);
toc;
```

In this scenario, both MEX files, when compiled with the `-g` debug flag, will be profiled seamlessly by Nsight Compute or Nsight Systems as if they were part of a single CUDA application. Executing either of the approaches described in the earlier examples will capture all CUDA calls made in both `add_arrays_cuda` and `multiply_array` from within a single profiling session.  You do not need to configure each mex call independently. This allows you to understand data transfer costs between the MEX functions and evaluate relative execution times.

For further learning and mastery of CUDA profiling, I recommend exploring NVIDIA’s documentation on Nsight Compute and Nsight Systems. The official CUDA Programming Guide provides crucial understanding of CUDA runtime. Additionally, tutorials and examples available from NVIDIA's developer portal are highly beneficial. These resources offer specific information on advanced profiling techniques and best practices for maximizing performance of CUDA applications within complex execution environments like MATLAB MEX. Careful study of these sources will equip you with the practical skills to efficiently debug and improve the performance of your MATLAB MEX CUDA applications.
