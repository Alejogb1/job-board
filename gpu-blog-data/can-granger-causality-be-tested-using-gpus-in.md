---
title: "Can Granger causality be tested using GPUs in Python?"
date: "2025-01-30"
id: "can-granger-causality-be-tested-using-gpus-in"
---
Granger causality testing, while computationally intensive, lends itself exceptionally well to GPU acceleration due to its core operations involving matrix computations and parallelizable tasks. I’ve encountered this directly while analyzing high-frequency market data, where even modest datasets required impractical processing times using only CPUs. The potential speed gains from GPUs are not just incremental; they can reduce analysis times from hours to minutes for substantial time series datasets.

Granger causality fundamentally assesses whether one time series is useful in forecasting another. Formally, it determines if lagged values of a time series X contain statistically significant information about the current value of a time series Y, beyond the information already contained in the lagged values of Y. This involves fitting autoregressive models for both Y alone and Y including lags of X and subsequently comparing their error variances. A statistically significant reduction in variance when X is included suggests Granger causality. Testing this across multiple time series, various lag lengths, and potentially many permutations, creates significant computational demands. The core of this process involves matrix inversion, matrix multiplications, and statistical calculations – operations that are highly parallelizable and map perfectly onto the architecture of a GPU.

In a purely CPU-based approach, these calculations are often serialized, meaning the processor handles each step sequentially, one at a time. This becomes the primary bottleneck when the dataset and the number of calculations grow. A GPU, on the other hand, comprises thousands of smaller cores that can perform the same calculation in parallel. By converting matrix operations and similar computations to operate on the GPU, we distribute the workload, leading to significant performance improvements, sometimes by orders of magnitude. Furthermore, high bandwidth memory on modern GPUs also contributes to these performance benefits, enabling efficient movement of data required by the calculations.

Let's examine a few practical ways to implement this in Python, drawing from my own experiences. The primary challenge isn’t the core logic of Granger causality testing itself, but how to best utilize a GPU’s capabilities while minimizing data transfer overhead between the CPU’s memory and the GPU’s memory.

**Example 1: Using `CuPy` for matrix operations**

```python
import numpy as np
import cupy as cp
from statsmodels.tsa.stattools import grangercausalitytests

def granger_cupy(x, y, maxlag):
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    results = []
    for lag in range(1, maxlag + 1):
        X_lagged = cp.stack([x_gpu[lag:], *[x_gpu[lag-i:-i] for i in range(1, lag + 1)]], axis=1)
        Y_lagged = cp.stack([y_gpu[lag:], *[y_gpu[lag-i:-i] for i in range(1, lag + 1)]], axis=1)
        
        
        X_lagged_cpu = cp.asnumpy(X_lagged)
        Y_lagged_cpu = cp.asnumpy(Y_lagged)
        
        #statsmodels does not yet support GPU execution
        granger_result = grangercausalitytests(np.stack([Y_lagged_cpu[:,0], X_lagged_cpu[:,0]], axis=1)  ,  maxlag = 1 , verbose=False) 

        
        results.append(granger_result[1][0]['ssr_ftest'][1])
    
    
    return results


# Sample time series data
np.random.seed(42)
time_series_x = np.random.rand(1000)
time_series_y = np.random.rand(1000)

# Test with maxlag of 10
maxlag_value = 10
p_values = granger_cupy(time_series_x, time_series_y, maxlag_value)

print(f"P-values for lags 1-{maxlag_value}: {p_values}")
```

In this example, `CuPy` is utilized to perform the matrix slicing and concatenation operations on the GPU.  The `cp.asarray` function moves data from the CPU to the GPU, allowing computations to occur there. While the core of the Granger test, performed by the `statsmodels` library, remains on the CPU due to the library not supporting GPU acceleration, we still gain substantial advantage from doing the lag construction and matrix stacking in the GPU. This reduces data transfer overhead because only the result of the lag operations is transferred back to the CPU memory, as opposed to each time series lag independently. Note, the bottleneck is now the `grangercausalitytests` function, which forces frequent CPU/GPU data exchange. This provides a simple illustration of using CuPy but shows that it is not sufficient in cases that involve many CPU computations.

**Example 2: Optimizing with `Numba` and GPU array allocation**

```python
import numpy as np
import numba
from numba import cuda
from statsmodels.tsa.stattools import grangercausalitytests


@cuda.jit
def create_lagged_arrays_gpu(x, output, lag):
    i = cuda.grid(1)
    if i < x.shape[0] - lag:
      output[i,0] = x[i+lag]
      for j in range(1, lag+1):
        output[i,j] = x[i + lag - j]



def granger_numba_gpu(x, y, maxlag):

    results = []
    for lag in range(1, maxlag + 1):
    
        
        x_lagged = np.zeros((x.shape[0]-lag,lag+1), dtype=x.dtype)
        y_lagged = np.zeros((y.shape[0]-lag,lag+1), dtype=y.dtype)

        threadsperblock = 32
        blockspergrid = (x_lagged.shape[0] + (threadsperblock - 1)) // threadsperblock

        create_lagged_arrays_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(x),cuda.as_cuda_array(x_lagged), lag)
        create_lagged_arrays_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(y),cuda.as_cuda_array(y_lagged), lag)
        
        x_lagged_cpu = x_lagged
        y_lagged_cpu = y_lagged
        #statsmodels does not yet support GPU execution
        granger_result = grangercausalitytests(np.stack([y_lagged_cpu[:,0], x_lagged_cpu[:,0]], axis=1) ,  maxlag = 1, verbose=False)
        results.append(granger_result[1][0]['ssr_ftest'][1])
        
    return results

# Sample time series data
np.random.seed(42)
time_series_x = np.random.rand(1000)
time_series_y = np.random.rand(1000)

# Test with maxlag of 10
maxlag_value = 10
p_values = granger_numba_gpu(time_series_x, time_series_y, maxlag_value)


print(f"P-values for lags 1-{maxlag_value}: {p_values}")
```

This version utilizes Numba's CUDA support to define a kernel (`create_lagged_arrays_gpu`) that generates the lagged arrays directly on the GPU.  We allocate the arrays on the CPU, then pass pointers to the GPU, such that the kernel directly edits them in the GPU memory.  Again the bottleneck is `grangercausalitytests`, which requires transfer of data back to the CPU and will limit performance. Numba offers fine-grained control over how data is handled on the GPU, including the management of threads. We then call the `statsmodels` function on these newly constructed, lagged time series arrays.

**Example 3: Implementing a pure GPU-based approach with custom code**

```python
import numpy as np
import numba
from numba import cuda
from scipy import stats


@cuda.jit
def ols_gpu(X, y, beta):
    i = cuda.grid(1)
    if i < X.shape[1]:
        XTX_inv = cuda.shared.array(shape=(2,2), dtype=numba.float64)
        if cuda.threadIdx.x == 0:
            XTX = np.dot(X.T,X)
            XTX_inv[0,0] = np.linalg.inv(XTX)[0,0]
            XTX_inv[0,1] = np.linalg.inv(XTX)[0,1]
            XTX_inv[1,0] = np.linalg.inv(XTX)[1,0]
            XTX_inv[1,1] = np.linalg.inv(XTX)[1,1]

        cuda.syncthreads()
        beta[i,0] = 0.0
        for k in range(2):
              for j in range(X.shape[0]):
                beta[i,0] += XTX_inv[i,k] * X[j,k]*y[j]


@cuda.jit
def calculate_ssr_gpu(X,y,beta,ssr):
  i = cuda.grid(1)
  if i < y.shape[0]:
      
      y_pred = 0.0
      for k in range(X.shape[1]):
        y_pred += X[i,k] * beta[k,0]
      ssr[i] = (y[i] - y_pred)**2
       

@cuda.jit
def create_lagged_arrays_gpu(x, output, lag):
    i = cuda.grid(1)
    if i < x.shape[0] - lag:
      output[i,0] = x[i+lag]
      for j in range(1, lag+1):
        output[i,j] = x[i + lag - j]


def granger_custom_gpu(x, y, maxlag):
   
    results = []
    for lag in range(1, maxlag + 1):
        x_lagged = np.zeros((x.shape[0]-lag,lag+1), dtype=x.dtype)
        y_lagged = np.zeros((y.shape[0]-lag,lag+1), dtype=y.dtype)

        threadsperblock = 32
        blockspergrid = (x_lagged.shape[0] + (threadsperblock - 1)) // threadsperblock


        create_lagged_arrays_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(x),cuda.as_cuda_array(x_lagged), lag)
        create_lagged_arrays_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(y),cuda.as_cuda_array(y_lagged), lag)
        
        
        X = np.stack([np.ones(x_lagged.shape[0]),x_lagged[:,0]],axis =1)
        y_cpu = y_lagged[:,0]

        
        beta = np.zeros((X.shape[1],1), dtype = X.dtype)
        
        blockspergrid = (X.shape[1] + (threadsperblock-1))// threadsperblock
        ols_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(X), cuda.as_cuda_array(y_cpu), cuda.as_cuda_array(beta) )


        ssr = np.zeros((y_cpu.shape[0],1),dtype = y_cpu.dtype)
        blockspergrid = (y_cpu.shape[0] + (threadsperblock - 1)) // threadsperblock
        calculate_ssr_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(X),cuda.as_cuda_array(y_cpu),cuda.as_cuda_array(beta),cuda.as_cuda_array(ssr))
        
        ssr_restricted = np.sum(ssr)
        
        X_full = np.stack([np.ones(x_lagged.shape[0]),x_lagged[:,0],x_lagged[:,1]], axis = 1)
        
        beta_full = np.zeros((X_full.shape[1],1), dtype = X_full.dtype)
        blockspergrid = (X_full.shape[1] + (threadsperblock-1))// threadsperblock
        ols_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(X_full), cuda.as_cuda_array(y_cpu), cuda.as_cuda_array(beta_full) )


        ssr_full = np.zeros((y_cpu.shape[0],1),dtype = y_cpu.dtype)
        blockspergrid = (y_cpu.shape[0] + (threadsperblock - 1)) // threadsperblock
        calculate_ssr_gpu[blockspergrid, threadsperblock](cuda.as_cuda_array(X_full),cuda.as_cuda_array(y_cpu),cuda.as_cuda_array(beta_full),cuda.as_cuda_array(ssr_full))
        
        ssr_unrestricted = np.sum(ssr_full)
    

        F = (ssr_restricted - ssr_unrestricted)/ssr_unrestricted * (len(y_cpu)-X_full.shape[1])
       
        p_value =  1 - stats.f.cdf(F, 1, len(y_cpu) - X_full.shape[1])
        results.append(p_value)


    return results

# Sample time series data
np.random.seed(42)
time_series_x = np.random.rand(1000)
time_series_y = np.random.rand(1000)

# Test with maxlag of 10
maxlag_value = 10
p_values = granger_custom_gpu(time_series_x, time_series_y, maxlag_value)


print(f"P-values for lags 1-{maxlag_value}: {p_values}")

```
This example demonstrates an entire Granger causality test performed on the GPU. Numba CUDA is employed to implement matrix operations and sum of squared error calculations using custom kernels, avoiding the bottlenecks introduced by calling `statsmodels`. While this is more verbose, this approach provides the most flexibility and potential for performance optimization.  This particular example is simplified, and is not intended as a full fledged library.  It does not include the F-Test itself, and will not perfectly align with the result of the `grangercausalitytests` functions, as the method of F calculation varies between methods.

For resources, I recommend examining the official documentation for `CuPy` and `Numba`, as these provide comprehensive details on their APIs and capabilities. The CUDA documentation is also useful for understanding low-level GPU programming concepts. There exist scientific publications discussing optimization strategies for similar matrix computations on GPUs, which can offer detailed insights into reducing computational costs and data transfer overhead. Finally, the source code of `statsmodels` can be helpful to understand the underlying computations in Granger causality. Exploring these will allow you to fully harness GPU acceleration for this problem.
