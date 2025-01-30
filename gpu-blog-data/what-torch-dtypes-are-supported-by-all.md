---
title: "What torch dtypes are supported by `all`?"
date: "2025-01-30"
id: "what-torch-dtypes-are-supported-by-all"
---
The fundamental consideration when using `torch.all` with different `torch.dtype`s stems from its logical operation: determining if all elements in a tensor evaluate to `True`. This necessitates that the tensor's underlying data type supports a meaningful interpretation of truthiness. Inherently, numeric and boolean types do, while others, like complex numbers, do not readily lend themselves to such direct assessment without specific comparisons. My experience building custom training loops for generative models exposed this nuance; failing to account for dtype compatibility led to unexpected behavior and debugging challenges during loss computation across different tensor representations.

`torch.all` primarily supports tensor elements with `bool` and numeric dtypes directly. This support arises from the core implementation of `all`, where each element’s value is implicitly converted to boolean in one form or another. For `bool` tensors, the operation is straightforward: a `True` element remains `True`, and a `False` element becomes `False`. The aggregation of these boolean values through the `and` operation determines the final output. In numeric tensors, a non-zero element is implicitly cast to `True`, and a zero element is cast to `False`. The aggregation proceeds as in boolean tensors. This implicit conversion is crucial. Specifically, floating-point types (like `torch.float32`, `torch.float64`, `torch.float16`, `torch.bfloat16`) and integer types (like `torch.int64`, `torch.int32`, `torch.int16`, `torch.int8`, `torch.uint8`) are readily usable with `torch.all`. Complex number types such as `torch.complex64` and `torch.complex128`, however, are not directly supported by `torch.all` because they don’t inherently map to a single truth value. A separate comparison to either a fixed value, or to zero, must be carried out first.

The behavior of `torch.all` can be further clarified with code examples. Consider a straightforward scenario with a boolean tensor.

```python
import torch

bool_tensor_true = torch.tensor([True, True, True], dtype=torch.bool)
bool_tensor_mixed = torch.tensor([True, False, True], dtype=torch.bool)
bool_tensor_false = torch.tensor([False, False, False], dtype=torch.bool)


all_true_bool = torch.all(bool_tensor_true)
all_mixed_bool = torch.all(bool_tensor_mixed)
all_false_bool = torch.all(bool_tensor_false)

print(f"All True Bool: {all_true_bool}")
print(f"All Mixed Bool: {all_mixed_bool}")
print(f"All False Bool: {all_false_bool}")
```

Here, `bool_tensor_true` contains all `True` values, and `torch.all` correctly returns `True`. `bool_tensor_mixed` contains a mix of `True` and `False` values, which results in `False`, and `bool_tensor_false` all `False` so returns `False` as well. This example demonstrates the expected logical aggregation of boolean values. A `torch.bool` tensor is a direct input to the `all` operation, where elements act as logical boolean values. This example is common when checking all flags in a boolean mask generated during data processing.

Now consider a numeric tensor:

```python
import torch

float_tensor_all_non_zero = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
float_tensor_mixed = torch.tensor([1.0, 0.0, 3.0], dtype=torch.float32)
int_tensor_all_non_zero = torch.tensor([1, 2, 3], dtype=torch.int32)
int_tensor_mixed = torch.tensor([1, 0, 3], dtype=torch.int32)
float_tensor_all_zero = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


all_non_zero_float = torch.all(float_tensor_all_non_zero)
all_mixed_float = torch.all(float_tensor_mixed)
all_non_zero_int = torch.all(int_tensor_all_non_zero)
all_mixed_int = torch.all(int_tensor_mixed)
all_zero_float = torch.all(float_tensor_all_zero)


print(f"All Non-zero Float: {all_non_zero_float}")
print(f"Mixed Float: {all_mixed_float}")
print(f"All Non-zero Int: {all_non_zero_int}")
print(f"Mixed Int: {all_mixed_int}")
print(f"All Zero Float: {all_zero_float}")
```

As we see, in numeric tensors, only a zero value is treated as `False`, and all non-zero values are considered `True` by `torch.all`, resulting in the correct logical aggregation. The implicit conversion from numeric type to boolean values is clear in this example. This use case is common when checking gradient values or loss terms.

Finally, consider a more complex scenario, where we try to directly utilize a complex type, which is not supported by the `all` function in the way a numeric or boolean type is. In this situation, a pre-processing step to convert the values to booleans is required.

```python
import torch

complex_tensor_true = torch.tensor([1+1j, 2+2j, 3+3j], dtype=torch.complex64)
complex_tensor_mixed = torch.tensor([1+1j, 0+0j, 3+3j], dtype=torch.complex64)
complex_tensor_all_zero = torch.tensor([0+0j, 0+0j, 0+0j], dtype=torch.complex64)

all_true_complex = torch.all(complex_tensor_true != 0+0j)
all_mixed_complex = torch.all(complex_tensor_mixed != 0+0j)
all_zero_complex = torch.all(complex_tensor_all_zero != 0+0j)


print(f"All Non-zero Complex: {all_true_complex}")
print(f"Mixed Complex: {all_mixed_complex}")
print(f"All Zero Complex: {all_zero_complex}")
```

Here, attempting to directly use `torch.all(complex_tensor_true)` will generate a type error. Instead, I use the not-equals-to operation, `!=`, and then evaluate `all` on a `bool` tensor. We evaluate a comparison of the complex numbers to `0+0j`. This comparison gives us a bool mask suitable to be used by `torch.all`. This pattern is important to generalize `all` for types that are not directly compatible, since `torch.all` does not support the use of a custom evaluation function. In the absence of more intricate custom comparison functions, comparison to zero is the most common operation.

In summary, while `torch.all` primarily operates on boolean and numeric types, its behavior stems from the implicit conversion of numeric values to boolean values (`0` being `False`, non-zero being `True`). Complex numbers require a pre-processing step to convert them into suitable inputs for `torch.all`. This knowledge was pivotal in my work on multi-modal models, where data might be represented as various dtypes; correctly interpreting these types and ensuring that operations behave as intended is important for reliable training. When encountering type errors with `torch.all`, ensuring you are working with a suitable `dtype` or have performed required pre-processing is the first troubleshooting step.

For further exploration of data types in PyTorch, one may refer to official PyTorch documentation sections on tensor creation and manipulation. Detailed descriptions of tensor operations provide additional context, as does reading documentation on specific data types in computer science literature more broadly. Books on machine learning and deep learning which cover the PyTorch library in detail can also prove to be a helpful learning tool.
