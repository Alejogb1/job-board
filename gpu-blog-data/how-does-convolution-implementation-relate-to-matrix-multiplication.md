---
title: "How does convolution implementation relate to matrix multiplication (GEMM) and 1x1 kernels?"
date: "2025-01-30"
id: "how-does-convolution-implementation-relate-to-matrix-multiplication"
---
The core relationship between convolutional implementation and general matrix multiplication (GEMM) lies in the fundamental mathematical equivalence achievable through clever reshaping of input tensors.  This equivalence, particularly pronounced with 1x1 kernels, allows leveraging highly optimized GEMM libraries for significant performance gains in convolutional neural networks (CNNs).  My experience optimizing CNN inference engines for embedded systems heavily relies on this understanding.

**1. Clear Explanation:**

A convolution operation, at its heart, involves sliding a kernel (a small matrix) across an input feature map, performing element-wise multiplication at each position, and summing the results to produce an output feature map.  This seemingly simple process can be computationally expensive, especially for large input images and deep networks.  However, by carefully restructuring the input and kernel tensors, we can reformulate the convolution as a series of matrix multiplications.

Consider a single input channel and a single output channel for simplicity.  Let's denote the input feature map as a matrix `I` of size `M x N`, and the kernel as a matrix `K` of size `k x k`.  A naive implementation would involve nested loops iterating through the input, performing the kernel convolution at each position.  This is computationally inefficient.

The key insight is that we can *im2col* (image to column) transform the input matrix `I`. This transformation extracts all `k x k` sub-matrices from `I` and arranges them as columns in a new matrix `I_col` of size `(M-k+1)(N-k+1) x k^2`.  Similarly, the kernel `K` can be reshaped into a column vector `K_vec` of size `k^2 x 1`.  The convolution operation now becomes a simple matrix multiplication:

`O_col = I_col * K_vec`

where `O_col` is a column vector representing the output feature map.  Finally, `O_col` is reshaped back into a matrix `O` of size `(M-k+1) x (N-k+1)`, representing the convolved output.  This transformation allows us to leverage optimized GEMM routines, which are highly efficient due to years of algorithmic refinement and hardware-specific optimizations.

For multiple input and output channels, this process is extended by considering each input channel independently and summing the results.  The resulting matrix multiplication becomes significantly larger, but the underlying principle remains the same: converting convolution into a GEMM operation.

The special case of 1x1 kernels simplifies this process dramatically.  Since the kernel is a 1x1 matrix, the `im2col` transformation becomes trivial.  The input feature map is directly treated as `I_col`, and the 1x1 kernel `K` is just a scalar (for single input/output channels) or a vector (for multiple output channels). The matrix multiplication then becomes a simple element-wise multiplication followed by summation, making the process extremely efficient. This is why 1x1 convolutional layers, despite their seemingly limited spatial effect, are computationally inexpensive yet capable of learning complex non-linear combinations of features.



**2. Code Examples with Commentary:**

**Example 1: Naive Convolution (Python with NumPy):**

```python
import numpy as np

def naive_convolution(image, kernel):
    M, N = image.shape
    k = kernel.shape[0]
    output = np.zeros((M - k + 1, N - k + 1))
    for i in range(M - k + 1):
        for j in range(N - k + 1):
            output[i, j] = np.sum(image[i:i+k, j:j+k] * kernel)
    return output

image = np.random.rand(5, 5)
kernel = np.random.rand(3, 3)
output = naive_convolution(image, kernel)
```

This demonstrates a basic, inefficient implementation.  Its complexity is O(M*N*k^2), significantly less efficient than GEMM-based approaches.


**Example 2: Im2col and GEMM (Python with NumPy):**

```python
import numpy as np

def im2col(image, kernel_size):
    k = kernel_size
    M, N = image.shape
    col = np.zeros(((M - k + 1) * (N - k + 1), k*k))
    idx = 0
    for i in range(M - k + 1):
        for j in range(N - k + 1):
            col[idx, :] = image[i:i+k, j:j+k].flatten()
            idx += 1
    return col

image = np.random.rand(5, 5)
kernel = np.random.rand(3, 3).flatten()
I_col = im2col(image, 3)
output_col = np.dot(I_col, kernel)
output = output_col.reshape(image.shape[0]-2, image.shape[1]-2)
```

This shows the `im2col` transformation and subsequent GEMM using `np.dot`.  This is considerably faster than the naive approach for larger inputs, though still not as optimized as dedicated GEMM libraries.


**Example 3: 1x1 Convolution (Python with NumPy):**

```python
import numpy as np

def one_by_one_convolution(image, kernel):
    return image * kernel  # Element-wise multiplication

image = np.random.rand(5, 5)
kernel = np.random.rand(1,1) #1x1 kernel
output = one_by_one_convolution(image,kernel)
```

This trivial example showcases the extreme efficiency of 1x1 convolutions.  It avoids explicit looping and directly uses element-wise multiplication, leveraging NumPy's vectorized operations.  This is the most efficient approach for 1x1 kernels.


**3. Resource Recommendations:**

For a deeper understanding of GEMM optimizations, I recommend exploring publications on the BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage) libraries.  Furthermore, delve into the literature on efficient convolution implementations, focusing on techniques like Winograd convolution and FFT-based methods.  Finally, a strong foundation in linear algebra is crucial for fully grasping the mathematical underpinnings of these operations.  Studying the inner workings of popular deep learning frameworks will reveal how these concepts are implemented in practice.
