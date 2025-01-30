---
title: "How do I fix a 'TypeError: expected Long but found double' error in PyTorch?"
date: "2025-01-30"
id: "how-do-i-fix-a-typeerror-expected-long"
---
Type mismatches are a persistent source of errors in numerical computation, particularly when dealing with deep learning frameworks like PyTorch. The specific `TypeError: expected Long but found double` arises when an operation or a tensor is expecting integer values (specifically of the `torch.long` dtype), but receives floating-point values (the `torch.double` dtype). This usually stems from implicit or explicit type conversions done incorrectly. I’ve debugged this particular issue several times, mostly in situations involving indexing tensors or providing data for specific layer types that demand long integers.

Fundamentally, PyTorch, like most numeric libraries, is strongly typed for performance and memory efficiency. Operations are often optimized for specific data types, leading to errors when a tensor is not of the expected type. The `torch.long` dtype is a 64-bit signed integer, often used for indexing operations because it supports a large range of indices, whereas `torch.double` represents double-precision (64-bit) floating-point numbers. When, for example, we try to use a floating-point tensor to select elements within another tensor or define the shape of a tensor, this discrepancy triggers the aforementioned error. The error message is precise: PyTorch is informing you that it expects an integer for a particular operation but has received a floating-point value.

The root cause can often be traced back to several typical programming practices. First, an operation might have implicitly converted an integer tensor to floating-point without us noticing. This happens easily when a tensor with floating points is combined with an integer tensor through an arithmetic operation, resulting in a floating-point result. For example, adding a tensor of ones cast as floating points, `torch.ones(10, dtype=torch.double)`, to a tensor of indices such as `torch.arange(10)`, will inadvertently cast the indices into the float dtype. Second, you might have loaded data from a file that was implicitly represented as floating-point, where your model required it as an integer. Third, sometimes data transformations used before loading data into your PyTorch model might unintentionally be converting your integers into doubles. Debugging requires a careful examination of how tensors are generated, loaded, and utilized in the surrounding code.

To address this error, we must enforce type compatibility using explicit casting. The key function for type conversion is `tensor.to(dtype)`. We should carefully identify where `double` type tensors are being employed incorrectly and convert them into `torch.long`. It’s best practice to cast the tensor right before it is used by a particular operation requiring the `torch.long` type. We could also enforce the `torch.long` data type from the beginning of our code or when loading data to reduce the chance of accidental type coercion. The choice will depend on the complexity and structure of the program but in either case, the goal is to explicitly manage tensor data types to avoid these type mismatch issues.

Let’s look at some specific examples with code demonstrating solutions.

**Example 1: Indexing tensors with incorrect type**

```python
import torch

# Incorrect approach
data = torch.randn(5, 5)
indices = torch.arange(0, 5).double() # Indices are doubles
try:
    indexed_data = data[indices] # Raises TypeError
except TypeError as e:
    print(f"Error caught: {e}")

# Corrected approach
indices_corrected = torch.arange(0, 5).long() # Indices are longs
indexed_data_corrected = data[indices_corrected]
print("Indexed data:", indexed_data_corrected)
```

In this example, the original code attempts to index the `data` tensor with a `indices` tensor of type `torch.double`. This immediately throws a `TypeError`, because PyTorch expects a tensor of integers (specifically `torch.long` or `torch.int64`) for indexing. The corrected code explicitly casts the `indices` tensor to type `torch.long` before indexing, resolving the issue. The corrected code ensures all values are handled as integers for indexing purposes.

**Example 2: Using a `torch.double` tensor for an operation requiring `torch.long`**

```python
import torch
import torch.nn as nn

class DummyEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        return embedded

embedding_dim = 128
num_embeddings = 10
embedding = DummyEmbedding(embedding_dim, num_embeddings)
# Incorrect Approach
indices = torch.randint(0, num_embeddings, (1,)).double()
try:
    embedded_output = embedding(indices)  # raises TypeError
except TypeError as e:
     print(f"Error caught: {e}")

# Corrected Approach
indices_corrected = torch.randint(0, num_embeddings, (1,)).long()
embedded_output_corrected = embedding(indices_corrected)
print("Embedded Output", embedded_output_corrected)
```

This example focuses on a common scenario with `nn.Embedding` layers. The input to an embedding layer must always be a tensor of type `torch.long`. The incorrect approach attempts to pass a floating-point `indices` tensor, which will cause a `TypeError`. The corrected approach explicitly casts the tensor to the required type before passing it to the embedding layer. This demonstrates a situation where implicit casting is not sufficient, emphasizing the need for explicit type management.

**Example 3: Incorrect data loading types**

```python
import torch
import numpy as np

# Simulating data loading
# Incorrect Approach
loaded_data = np.random.rand(10).astype(np.float64)
data_tensor = torch.tensor(loaded_data) # Implicitly becomes a double
try:
    indices_tensor = data_tensor[0:5]
except TypeError as e:
    print(f"Error caught {e}")

# Corrected approach
loaded_data_corrected = np.random.randint(0, 10, 10).astype(np.int64)
data_tensor_corrected = torch.tensor(loaded_data_corrected, dtype=torch.long)
indices_tensor_corrected = data_tensor_corrected[0:5]
print("Corrected indices tensor:", indices_tensor_corrected)

```

This example highlights a common situation when loading data from file formats that could be implicitly represented as `double` by default, especially with numpy arrays. In this example the `np.random.rand` function generates floating point values which result in a `double` type tensor in PyTorch. The `torch.tensor` by default tries to match the underlying data type. The corrected example demonstrates enforcing the correct type of data generation using `np.random.randint` and ensuring the resulting tensor is of type `torch.long` upon its construction. This avoids issues arising from implicitly generated floating-point data. In a real-world application, similar practices should be applied to how data is parsed from files into tensors.

To improve debugging strategies and prevent similar errors in future, there are a few practices I recommend. Firstly, always check the data types of your tensors, using `tensor.dtype`, before operations that may be sensitive to input types. Consider using `assert` statements to enforce types early in your code if you need specific type constraints. Secondly, develop the habit of explicit type casting as near as possible to the operations where data is required. Finally, review and check the documentation of layers and functions to explicitly understand which types they require. Resources like the PyTorch official documentation, and numerous online tutorials can help provide a comprehensive understanding of the library's data types and their usage. I’ve found having those close at hand helps reduce debugging time significantly.
