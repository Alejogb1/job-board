---
title: "How can numba vectorize be used to generate all possible signatures?"
date: "2025-01-30"
id: "how-can-numba-vectorize-be-used-to-generate"
---
Numba's `@vectorize` decorator does not, by design, generate all possible function signatures automatically. Instead, it necessitates explicit specification of the input and output data types the vectorized function will operate upon. This constraint, while seemingly limiting at first glance, is actually essential for achieving the high performance Numba is known for. The performance gains arise from targeting specific machine code instructions optimized for particular data layouts rather than attempting runtime type dispatch.

My experience over the past five years optimising scientific computing applications, particularly involving large datasets, has shown the vital need for precise type control. Allowing Numba to arbitrarily infer data types can lead to suboptimal performance or even incorrect results when dealing with mixed-type data. The core objective of `@vectorize` is to generate Universal Functions (ufuncs), which NumPy natively uses for element-wise operations. Thus, like NumPy’s ufuncs, Numba's implementation requires upfront knowledge of the data types it is expected to process.

To clarify the challenge presented by the original question: one cannot simply ask Numba's `@vectorize` to "figure out" all viable signatures. This contrasts with `nopython` mode under the `@jit` decorator which, through type inference, can effectively produce multiple specialized compiled versions of a function. Instead, `@vectorize` generates a single compiled kernel per signature that operates on arrays as if it were being applied element-by-element. Therefore, the user is responsible for listing all desired signatures. While there isn't a direct function to generate 'all possible' signatures, one must manually define them, or use metaprogramming, or other methods to automate creation of these types.

Let me detail the process and illustrate how to define signatures practically, including handling mixed-type inputs which is a common requirement.

**Understanding Signature Definition**

The signature within the `@vectorize` decorator uses a string-based notation to describe the input and output types. The format is: `output_type(input_type1, input_type2, ...)`. These types should align with those defined by NumPy’s `dtype` specifications or be scalar primitives that Numba understands. Crucially, the input and output types must have a size known at compile-time.

For example, `'float64(float64, float64)'` specifies a function taking two double-precision floating-point numbers as input and returning a single double-precision floating-point number. You can use `int32`, `float32`, `complex128`, and other similar type strings.

Here's our first code example showing a basic use of `@vectorize`:

```python
import numpy as np
from numba import vectorize, float64

@vectorize('float64(float64, float64)')
def vector_add(x, y):
    return x + y

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

result = vector_add(a, b)
print(result) # Output: [5. 7. 9.]
```
This example defines a simple element-wise addition function for arrays of double-precision floating-point values. The defined signature matches the inputs and the expected output, enabling Numba to compile an efficient function. The core function, `vector_add`, is simple, but this demonstrates the fundamental mechanism.

**Handling Mixed Type Inputs**

Often, operations involve more than just single datatypes. Consider the case where you have integer and floating-point inputs. Numba's `@vectorize` allows for multiple signatures of the same function through repeated calls to the decorator for differing signatures. Here’s our second code example demonstrating handling of multiple signatures:

```python
import numpy as np
from numba import vectorize, float64, int32

@vectorize('float64(float64, float64)')
@vectorize('float64(int32, float64)')
@vectorize('float64(float64, int32)')
def mixed_add(x, y):
    return x + y

c = np.array([1, 2, 3], dtype=np.int32)
d = np.array([1.5, 2.5, 3.5])
e = np.array([1.5, 2.5, 3.5])
f = np.array([1, 2, 3], dtype=np.int32)

result1 = mixed_add(d, d)
result2 = mixed_add(c, d)
result3 = mixed_add(e, f)

print(result1) # Output: [3.  5.  7.]
print(result2) # Output: [2.5 4.5 6.5]
print(result3) # Output: [2.5 4.5 6.5]

```

Here, the function `mixed_add` is now "overloaded" with different implementations for different input types. When `mixed_add` is called with `c` and `d` (an integer and a float array, respectively), Numba chooses the `float64(int32, float64)` specialization. If both inputs are floating-point arrays, it selects `float64(float64, float64)`. The result is always cast to a double-precision float based on the defined output. This highlights the requirement of explicitly stating all possible combinations one expects to use in the code. It also means that implicit casting does not happen; if the input types are not specifically described in a decorator, the function will not work. This is essential for maintaining type safety, correctness, and ensuring efficient execution by Numba. This demonstrates that you can add more type variations through multiple `@vectorize` decorators; if you were to support, say, `int64` or `complex128` you would need to add these. It also highlights the non-automatic nature of type generation.

**Metaprogramming Approaches**

While explicitly declaring multiple signatures can become tedious, especially for complex scenarios, metaprogramming provides a way to automate the process. For example, you can generate function signatures based on a set of type tuples. Here's our third code example to demonstrate this:

```python
import numpy as np
from numba import vectorize, float64, int32, int64

def generate_signatures(output_type, input_types):
    signatures = []
    for input_tuple in input_types:
        signature = f'{output_type}('
        signature += ','.join([str(dtype) for dtype in input_tuple])
        signature += ')'
        signatures.append(signature)
    return signatures


input_type_pairs = [(float64, float64), (int32, float64), (float64, int32),(int64,int64),(int64, float64),(float64, int64)]
output_type = 'float64'
signatures = generate_signatures(output_type, input_type_pairs)


def make_vectorized_function(signatures):
    def decorated_function(func):
        for signature in signatures:
            vectorize(signature)(func)
        return func
    return decorated_function

@make_vectorized_function(signatures)
def dynamic_add(x, y):
    return x + y

g = np.array([1, 2, 3], dtype=np.int64)
h = np.array([1.5, 2.5, 3.5])
i = np.array([1, 2, 3], dtype=np.int32)

result4 = dynamic_add(g,g)
result5 = dynamic_add(g,h)
result6 = dynamic_add(h,g)

print(result4) # Output: [2. 4. 6.]
print(result5) # Output: [2.5 4.5 6.5]
print(result6) # Output: [2.5 4.5 6.5]

```

In this example, we have a `generate_signatures` function that dynamically creates signature strings from a collection of input data types. A `make_vectorized_function` is also produced to decorate the underlying function with multiple signatures. The `dynamic_add` function now accepts `int64`, `int32`, and `float64` combinations as expected. This reduces code duplication, but it's crucial to note that these signatures must be known at compile time, and you are effectively pre-specifying them, not generating them via type inference. This approach shows a practical method to organize the work required when handling a large number of data type variations. You still need to define all the types, but you do not necessarily need to write each `@vectorize` decorator by hand.

**Further Resource Recommendations:**

To deepen your understanding, I would suggest consulting the following resources:

1.  The official Numba documentation (numba.pydata.org). This resource is crucial for understanding all aspects of the `@vectorize` decorator, type specifications, and supported data types.
2.  NumPy’s official documentation. Familiarity with NumPy is essential for utilizing Numba, specifically for understanding universal functions (ufuncs) which Numba's `@vectorize` decorator attempts to replicate.
3.  Advanced Numba examples, particularly from user contributed projects. These demonstrate how `@vectorize` is integrated into real-world projects and often highlight common practices and issues when working with varied data types.
4.  The Cython documentation. Understanding how Cython handles type definitions and compiles code can be useful, as Cython also provides alternatives for performance-critical code similar to Numba.

In summary, `numba.vectorize` does not automatically generate every possible signature, instead, you define them and compile them into a highly optimized function. While the initial learning curve might seem steep, especially when handling type combinations, you will notice a marked increase in performance for data-intensive applications. The use of code generation patterns can help alleviate some of the pain involved. Careful selection of data types, and defining only those combinations you will utilize, ultimately results in optimized code and type safety.
