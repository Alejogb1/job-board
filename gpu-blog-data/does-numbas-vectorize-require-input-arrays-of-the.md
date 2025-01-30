---
title: "Does Numba's @vectorize require input arrays of the same shape?"
date: "2025-01-30"
id: "does-numbas-vectorize-require-input-arrays-of-the"
---
The `@vectorize` decorator in Numba does not, strictly speaking, require input arrays of precisely the same shape for *all* scenarios; rather, it operates based on the broadcasting rules defined by NumPy, extending these rules to allow for scalar and compatible array operations. This subtle distinction is crucial for understanding when `@vectorize` will perform as intended and when it might raise an error or return unexpected results. I've encountered this in numerous data processing pipelines while optimizing numerical computations, particularly those involving multi-dimensional datasets where aligning the shape of each input was unnecessarily costly.

Let me clarify: the core functionality of `@vectorize` is to automatically generate a Universal Function (ufunc) that operates element-wise. When given a function and type signature, Numba transforms it into optimized machine code designed to execute efficiently across arrays of compatible shapes. The critical point of compatibility lies within the concept of NumPy broadcasting. This implies that if you supply two arrays, A and B, to a `@vectorize` ufunc, and those arrays are not exactly the same shape, NumPy will try to *broadcast* them to a common shape before performing the element-wise operations dictated by your function. Broadcasting expands the dimensions of lower-dimensional arrays along their leading axes to match the dimensions of higher-dimensional arrays. The general rule states: two array dimensions are compatible when either (a) they are equal, or (b) one of them is 1. If these conditions are not met, then broadcasting cannot occur, and Numba, through NumPy’s machinery, will raise an error, even with the `@vectorize` decorator.

To illustrate, consider a simple example. Let's say I want to add a scalar value to all elements of a NumPy array. I define the following vectorized function:

```python
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def add_scalar(a, b):
    return a + b
```

Here, I've created `add_scalar` which adds two floating point numbers. Now, if I provide a NumPy array, `my_array`, and a scalar, say `my_scalar = 2.0`, using the `@vectorize` decorator, broadcasting is invoked. The scalar is treated as a zero-dimensional array, effectively being "stretched" to the shape of `my_array` during the element-wise addition.

```python
my_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
my_scalar = 2.0
result = add_scalar(my_array, my_scalar)
print(result) # Output: [3. 4. 5. 6. 7.]
```

The vectorized function, even though defined to operate on two scalars, correctly handles the scalar and array due to broadcasting. No explicit reshaping of the scalar is needed. The input shapes are considered compatible.

However, if broadcasting is not possible, `@vectorize` will fail. Let’s consider this example:

```python
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def element_multiply(a, b):
    return a * b

a = np.array([[1.0, 2.0], [3.0, 4.0]]) #Shape (2, 2)
b = np.array([1.0, 2.0, 3.0])       #Shape (3)

try:
  result_bad = element_multiply(a, b)
except ValueError as e:
  print(f"ValueError caught correctly: {e}") #ValueError: operands could not be broadcast together with shapes (2,2) (3,)
```

Here, we try to multiply a 2x2 array with a 1D array of size 3. These shapes are not broadcast compatible. The error shows that this operation fails because of the inability to broadcast dimensions of shapes `(2, 2)` and `(3,)`. The vectorized function, while technically applicable to element-wise multiplication, cannot proceed when broadcasting is not feasible. This is, in essence, a limitation imposed by NumPy's broadcasting rules, not an inherent limitation in `@vectorize` per se.

Let's explore another case to further clarify broadcasting. When applying a `@vectorize` ufunc to two arrays with different numbers of dimensions, the lower dimension array will often have its shape prepended with ones to allow broadcasting. This assumes that the dimensions are compatible under broadcasting rules. For example:

```python
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def element_subtract(a, b):
    return a - b


array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
array_1d = np.array([1.0, 2.0, 3.0])  # Shape (3,)

result_compatible = element_subtract(array_2d, array_1d)
print(result_compatible)

# Output:
# [[0. 0. 0.]
#  [3. 3. 3.]]
```

In this scenario, `array_1d` with shape `(3,)` is effectively interpreted as having shape `(1, 3)` for broadcasting purposes. It aligns correctly with the second axis of `array_2d` with shape `(2, 3)`, and the subtraction is performed element-wise according to the broadcasted shapes. Again, the `@vectorize` functionality leverages NumPy’s broadcasting, and the input arrays are not required to be of the *exact* same shape.

In practice, when I use `@vectorize`, I usually check my array shapes beforehand, and try to understand how broadcasting will behave, especially when working with high dimensional arrays where the rules can become less intuitive. Understanding this underlying mechanism is crucial for effectively using Numba to speed up my numerical routines. In general, I find it useful to use NumPy's own broadcasting methods, like `np.broadcast_arrays` to gain insight into what the final shape would be, before employing the `@vectorize` decorator.

For further understanding of both Numba and NumPy’s capabilities, I would recommend reading the official documentation, exploring tutorials on universal functions and broadcasting and referencing the source code of the relevant libraries to examine how the core broadcasting algorithm is implemented. Numerous open-source resources offer deep insights into the inner workings and best practices for vectorized operations in Python. These resources, collectively, form the best way for gaining practical familiarity with the subject matter and should be reviewed in that order.
