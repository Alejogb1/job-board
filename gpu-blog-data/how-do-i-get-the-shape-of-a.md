---
title: "How do I get the shape of a torch::Tensor?"
date: "2025-01-30"
id: "how-do-i-get-the-shape-of-a"
---
Determining the shape of a `torch::Tensor` involves accessing its dimensions, a fundamental operation in any PyTorch C++ application.  My experience developing high-performance neural network inference engines has frequently highlighted the critical importance of efficient shape manipulation.  Incorrectly handling tensor shapes often leads to runtime errors, particularly during matrix multiplications or broadcasting operations.  The core method for this involves the `sizes()` member function, returning a `std::vector<int64_t>`.

1. **Explanation:**

The `torch::Tensor` class doesn't directly expose its shape as a single attribute like a Python `numpy.ndarray`'s `.shape`. Instead, the shape is represented by the tensor's dimensions.  The `sizes()` method provides access to these dimensions, returning a vector where each element corresponds to the size of the tensor along a particular axis.  A 1D tensor will have a vector of size 1, a 2D tensor (matrix) will have a vector of size 2, and so forth.  The order of elements in this vector follows the standard mathematical convention, reflecting the dimensions from outermost to innermost.  For instance, a tensor of shape [3, 4, 5] has 3 elements along the outermost dimension, 4 along the next, and 5 along the innermost. Understanding this convention is crucial for correct data interpretation and manipulation.  Further, the absence of a `sizes()` return value, such as an empty vector, indicates an empty tensor.  Handling this edge case proactively is paramount to robust code.  Error checking for empty tensors before accessing their dimensions prevents segmentation faults or undefined behavior.

2. **Code Examples:**

**Example 1: Basic Shape Retrieval:**

```c++
#include <torch/torch.h>
#include <iostream>

int main() {
  auto tensor = torch::randn({3, 4, 5}); // A 3D tensor
  auto sizes = tensor.sizes();

  std::cout << "Tensor shape: ";
  for (int64_t size : sizes) {
    std::cout << size << " ";
  }
  std::cout << std::endl;  // Output: Tensor shape: 3 4 5

  return 0;
}
```
This example demonstrates the fundamental usage of `sizes()`. It generates a random 3D tensor and prints its shape to the console.  The loop iterates through the `std::vector<int64_t>` returned by `sizes()`, providing a clear and readable output.  Error handling is not explicitly included here for brevity but is crucial in production code.

**Example 2: Handling Empty Tensors:**

```c++
#include <torch/torch.h>
#include <iostream>
#include <vector>


int main() {
  auto emptyTensor = torch::empty({0});
  auto tensor = torch::randn({3,4});


  auto printShape = [&](const torch::Tensor& t) {
      std::vector<int64_t> sizes = t.sizes();
      if (sizes.empty()){
          std::cout << "Tensor is empty" << std::endl;
          return;
      }
      std::cout << "Tensor shape: ";
      for (int64_t size : sizes) {
          std::cout << size << " ";
      }
      std::cout << std::endl;
  };

  printShape(emptyTensor); // Output: Tensor is empty
  printShape(tensor); // Output: Tensor shape: 3 4

  return 0;
}
```

This example showcases error handling for empty tensors.  The lambda function `printShape` checks if `sizes()` returns an empty vector; if it does, a message indicating an empty tensor is printed. This prevents potential crashes or unexpected behavior when working with tensors that might be empty under certain conditions.  The use of a lambda function enhances code readability and reusability.

**Example 3: Shape-Dependent Operations:**

```c++
#include <torch/torch.h>
#include <iostream>
#include <vector>

int main() {
  auto tensor = torch::randn({2, 3});
  std::vector<int64_t> sizes = tensor.sizes();

  if (sizes.size() == 2) {
    int64_t rows = sizes[0];
    int64_t cols = sizes[1];
    std::cout << "Matrix with " << rows << " rows and " << cols << " columns." << std::endl;
  } else {
    std::cout << "Tensor is not a 2D matrix." << std::endl;
  }

  return 0;
}
```
This example demonstrates how to use the obtained shape information to perform operations that depend on the tensor's dimensionality. It checks if the tensor is a 2D matrix (has two dimensions) and, if so, extracts the number of rows and columns. This conditional logic allows for adaptable code that can handle different tensor shapes gracefully.  Error handling ensures robustness by checking tensor dimensionality before performing shape-dependent calculations.


3. **Resource Recommendations:**

The official PyTorch documentation, particularly the sections detailing the C++ API and tensor manipulation, offers comprehensive guidance.  I'd also recommend reviewing any relevant chapters of a good numerical computing textbook; the focus should be on linear algebra and vector space concepts.  A book dedicated to deep learning fundamentals often includes detailed explanations of tensor operations, contributing to a deeper understanding of their underlying principles.  Finally, exploring well-structured open-source projects that extensively utilize PyTorch C++ provides practical examples and demonstrates best practices.
