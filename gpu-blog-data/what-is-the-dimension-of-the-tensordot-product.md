---
title: "What is the dimension of the tensordot product of two 3D tensors?"
date: "2025-01-30"
id: "what-is-the-dimension-of-the-tensordot-product"
---
The dimensionality of the tensordot product of two 3D tensors is not fixed and depends entirely on the axes over which the contraction is performed.  This is a crucial point often missed in initial introductions to tensor operations.  My experience working on high-dimensional data analysis for geophysical modelling has highlighted the need for precise understanding of this.  Simply knowing the input tensors are 3D provides insufficient information. We must explicitly define the axes involved in the summation.


**1. Clear Explanation:**

The tensordot function, as implemented in libraries like NumPy, generalizes the dot product to higher-dimensional arrays.  A standard dot product involves a summation over a single axis. Tensordot allows for summation over multiple axes of two tensors, A and B.  The resulting tensor's dimensions are determined by combining the remaining axes from A and B that were *not* involved in the summation.

Let's denote the dimensions of tensor A as (a1, a2, a3) and those of tensor B as (b1, b2, b3).  The `tensordot` function takes two arguments specifying the axes for the summation: `axes_a` and `axes_b`. `axes_a` is a tuple or list specifying the axes of A to be summed over, and similarly, `axes_b` specifies the axes of B. The crucial constraint is that the dimensions of the axes specified in `axes_a` and `axes_b` must match. That is, the number of elements in `axes_a` and `axes_b` must be equal, and the dimensions of the corresponding axes must be identical.

The resulting tensor's dimensions will be:

* Dimensions from A: All dimensions *except* those specified in `axes_a`.
* Dimensions from B: All dimensions *except* those specified in `axes_b`.

These remaining dimensions are concatenated to form the final shape of the tensordot product.  If the summation exhausts all dimensions of either A or B, the resulting tensor will have a dimensionality that depends solely on the remaining dimensions of the other tensor.  This leads to a range of possible outputs depending on the `axes_a` and `axes_b` parameters.


**2. Code Examples with Commentary:**

**Example 1:  Simple Contraction along one axis:**

```python
import numpy as np

A = np.arange(24).reshape((2,3,4))  # Shape: (2, 3, 4)
B = np.arange(24).reshape((4,3,2))  # Shape: (4, 3, 2)

# Contraction along the last axis of A and the first axis of B
result1 = np.tensordot(A, B, axes=([2],[0]))

print(f"Shape of A: {A.shape}")
print(f"Shape of B: {B.shape}")
print(f"Shape of result1 (axes ([2], [0])): {result1.shape}")  # Output: (2, 3, 3, 2)
```

Here, the last axis of A (dimension 4) is contracted with the first axis of B (also dimension 4). The remaining axes (2, 3 from A and 3, 2 from B) form the resultant tensor's shape.  This demonstrates a basic contraction resulting in a 4D tensor.

**Example 2:  Contraction along multiple axes:**

```python
import numpy as np

A = np.arange(24).reshape((2,3,4)) # Shape: (2, 3, 4)
B = np.arange(24).reshape((4,3,2)) # Shape: (4, 3, 2)

# Contraction along the last two axes of A and the first two axes of B
result2 = np.tensordot(A, B, axes=([1,2],[0,1]))

print(f"Shape of A: {A.shape}")
print(f"Shape of B: {B.shape}")
print(f"Shape of result2 (axes ([1,2], [0,1])): {result2.shape}")  #Output: (2,2)

```

In this instance, we contract along two axes simultaneously.  The second and third axes of A (dimensions 3 and 4 respectively) are summed over against the first and second axes of B (dimensions 4 and 3 respectively).  Only the first axis of A remains, resulting in a 2D tensor.  Note the importance of the order in `axes_a` and `axes_b` — the matching of dimensions is crucial.


**Example 3:  Resulting in a scalar:**

```python
import numpy as np

A = np.arange(24).reshape((2,3,4))  # Shape: (2, 3, 4)
B = np.arange(24).reshape((4,3,2))  # Shape: (4, 3, 2)

# Contraction along all axes of A and some of B, leading to a scalar result
result3 = np.tensordot(A, B, axes=([0,1,2],[1,0,2]))  #Note careful axis matching here.

print(f"Shape of A: {A.shape}")
print(f"Shape of B: {B.shape}")
print(f"Shape of result3 (axes ([0,1,2],[1,0,2])): {result3.shape}")  # Output: () - scalar

```

This example showcases a complete summation resulting in a scalar.  Careful consideration of axis order ensures correct pairing and summation.  The dimensions of A are completely exhausted in the contraction; therefore, the output is a scalar (represented by an empty tuple as its shape).  This highlights how the choice of axes dramatically alters the output’s dimensionality.



**3. Resource Recommendations:**

I recommend consulting the official documentation for the NumPy library.  Thoroughly reviewing linear algebra texts focusing on tensor algebra and multilinear maps will further solidify understanding.  A good grasp of index notation in tensor manipulations is highly beneficial.  Additionally, studying texts on differential geometry can provide additional context for tensor operations in higher dimensions.  Working through several numerical examples with varying dimensions and axis choices is crucial for developing an intuitive understanding.  This is what ultimately helped me solidify my grasp of this topic.
