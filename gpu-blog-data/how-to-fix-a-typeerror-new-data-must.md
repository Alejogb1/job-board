---
title: "How to fix a 'TypeError: new(): data must be a sequence' when using numpy.float64?"
date: "2025-01-30"
id: "how-to-fix-a-typeerror-new-data-must"
---
The `TypeError: new(): data must be a sequence` encountered when using `numpy.float64` typically arises from an attempt to initialize a `numpy.float64` object with a single, non-sequence value, instead of a sequence (like a list or tuple) or another suitable NumPy array. This is a crucial point of misunderstanding about how NumPy's scalar types are designed for both single value representations and as containers. The error stems from the way `numpy.float64`'s constructor is intended to operate, particularly when explicitly instantiated. While we often implicitly utilize `numpy.float64` when a numerical operation on a NumPy array returns a float of double precision, directly attempting `numpy.float64(5)` where 5 is an integer or float, leads to this error.

Specifically, the constructor for NumPy scalar types like `numpy.float64` primarily intends to convert a variety of inputs (including lists, tuples, or other NumPy arrays) into a scalar object of that type. It examines the first item of the input sequence and attempts to create a scalar of the required type with it. If it encounters a single, non-sequence data point, instead of a list or array, it cannot process and raises the `TypeError`. This design allows `numpy.float64` to work seamlessly when converting arrays of numeric types to float64; however, this behavior is not straightforward when directly creating an object from a single value.

I've encountered this situation multiple times during development of numerical simulations where I wanted to manually set a specific floating-point value and naively tried `numpy.float64(my_value)`. The unexpected error, as well as the required correction, highlighted a key concept about NumPy’s scalar representation. Here is an explanation of how to resolve this and some examples:

**Explanation:**

The correct approach is to recognize that when you intend to represent a single float64 number, a direct instantiation like `numpy.float64(5)` is not how the constructor was designed to handle single numbers. Instead, you should typically allow NumPy's type coercion to implicitly handle it, or leverage other creation methods. Here’s a breakdown of the solutions:

1.  **Direct Assignment (Implicit Type Conversion):** This is the most common and efficient way to work with `numpy.float64`. In this approach, you assign a regular Python float to a variable that you want to be a NumPy float64 type, where NumPy will implicitly and efficiently coerce the type if needed. This approach works fine as long as the goal is to use single values, rather than explicitly create object. This avoids the direct call to the constructor.

2.  **Using NumPy Array Creation Functions:** Utilizing functions like `numpy.array` or `numpy.full` allows you to create arrays where the elements are of type `numpy.float64`. If you need a single float64 value, you create an array with one element, then access the first element.

3.  **Utilizing the `dtype` Argument:** While less common for setting a single value, when creating arrays of specified types, you can explicitly specify `dtype=numpy.float64` when creating the NumPy arrays.

**Code Examples with Commentary:**

**Example 1: Incorrect Instantiation and Corrected Assignment**

```python
import numpy as np

# Incorrect approach leading to the error
try:
    incorrect_float = np.float64(5)
except TypeError as e:
    print(f"Error Encountered: {e}")

# Correct approach using assignment.
correct_float_implicit = 5.0 # Regular Python float, numpy may coerce
print(f"Implicitly coerced float: {type(correct_float_implicit)}")
correct_float_np = np.float64(np.array([5.0])[0])
print(f"Explicitly made via numpy: {type(correct_float_np)}")
```

*Commentary:* The first attempt directly uses `np.float64(5)`, which causes the described `TypeError`. The subsequent attempt uses implicit type coercion simply by assigning a float literal. The final attempt is to create a one element array and then retrieve the item at index 0, which is implicitly converted to type `numpy.float64`.

**Example 2: Utilizing Array Creation Functions**

```python
import numpy as np

# Creating a NumPy array with a single float64 element
single_float_array = np.array([5], dtype=np.float64)
single_float = single_float_array[0] #extracting first element
print(f"Float64 created with numpy.array: {type(single_float)}, value: {single_float}")

# Creating a full numpy array with the same value, all float64
full_float_array = np.full((3,3), 5, dtype=np.float64)
print(f"Full array type: {full_float_array.dtype}")
print(f"Single element: {full_float_array[0][0]}, type: {type(full_float_array[0][0])}")
```

*Commentary:* Here, we create arrays using `numpy.array` and `numpy.full`, explicitly setting the data type to `numpy.float64`. This method is particularly effective when dealing with entire arrays or matrices of that type. The first example shows explicitly creating a single element array and extracting the value of interest. The second example shows that when creating a matrix or array, one can specify the `dtype` argument to force that array to be comprised of elements of that type. Note that the element accessed from the `full_float_array` is of type `numpy.float64` due to the specification of `dtype` during the array creation.

**Example 3: Passing a Sequence**

```python
import numpy as np

# Passing a list to the numpy float64 constructor
my_list = [5.0]
float_from_list = np.float64(my_list)
print(f"Float64 created from a list: {type(float_from_list)}, value: {float_from_list}")

# Passing a tuple
my_tuple = (5.0,)
float_from_tuple = np.float64(my_tuple)
print(f"Float64 created from a tuple: {type(float_from_tuple)}, value: {float_from_tuple}")

# passing a np array
my_np_array = np.array([5.0])
float_from_np_array = np.float64(my_np_array)
print(f"Float64 created from np.array: {type(float_from_np_array)}, value: {float_from_np_array}")
```

*Commentary:* This demonstrates how `numpy.float64`'s constructor can be used by passing it a sequence. In each case, the constructor is passed a valid sequence, either a Python list, tuple, or a NumPy array, each containing a single numerical element. The constructor extracts the element and creates the `numpy.float64` object. Note that while this is technically "correct" and avoids the `TypeError`, this is generally an unusual way to create a single float value. The simpler methods (direct assignment or numpy array creation) are more idiomatic when seeking a single `numpy.float64` value.

**Resource Recommendations:**

*   **Official NumPy Documentation:** The comprehensive guide on NumPy's functionalities is indispensable. Look specifically for sections covering scalar types and array creation.

*   **NumPy User Guide:** The user guide provides a more accessible and tutorial-oriented explanation of NumPy, including clear examples of data type management.

*   **Textbooks on Numerical Computing with Python:** Many texts cover NumPy in detail, providing a broader context for its usage in scientific and numerical applications. Such references often cover the specifics of numerical types.

In conclusion, the `TypeError: new(): data must be a sequence` when directly initializing a `numpy.float64` with a single numerical value highlights an important detail about how NumPy handles scalar types and their instantiation. The correct approach involves either implicit type conversion through direct assignment of a floating-point literal, creating single-element arrays via `numpy.array` and then accessing them, or utilizing other array creation methods with explicit `dtype` specification.
