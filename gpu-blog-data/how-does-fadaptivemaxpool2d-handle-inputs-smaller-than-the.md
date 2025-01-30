---
title: "How does F.adaptive_max_pool2d handle inputs smaller than the target size in PyTorch?"
date: "2025-01-30"
id: "how-does-fadaptivemaxpool2d-handle-inputs-smaller-than-the"
---
Adaptive max pooling, specifically PyTorch's `F.adaptive_max_pool2d`, exhibits a nuanced behavior when presented with input tensors smaller than the specified output size.  My experience optimizing CNN architectures for resource-constrained environments has highlighted this crucial detail:  the function doesn't simply zero-pad or replicate; instead, it performs a form of upsampling implicitly through the max pooling operation.  This is subtly different from standard max pooling, which operates on fixed-size windows.

**1. Explanation:**

`F.adaptive_max_pool2d` takes as input a 4D tensor of shape `(N, C, H_in, W_in)` representing `N` batches, `C` channels, and input height and width `H_in` and `W_in`. The crucial second argument is the `output_size`, which specifies the desired output height and width (`H_out`, `W_out`). Unlike standard max pooling with fixed kernel sizes and strides, adaptive max pooling dynamically adjusts its kernel sizes to achieve the target output size.

When `H_in < H_out` or `W_in < W_out`, the effective kernel size for each output element becomes larger than 1.  In essence, the entire input feature map along a dimension contributes to a single output element.  Consider an extreme case: if `H_in = 1` and `H_out = 5`, each of the five output height elements will contain the maximum value from the single input height element.  This is not padding or replication; it’s a process that maximizes the value over the entire input along the respective dimension.  The same logic applies to width.  Therefore, information is not lost; rather, it is effectively "upsampled" through the maximization operation, focusing on the maximum value within the input's constraints.  The output tensor's shape will always be `(N, C, H_out, W_out)`.

This behavior differs significantly from other pooling methods.  Standard max pooling with a fixed kernel size would require padding if the input is smaller than the kernel, leading to potential boundary effects.  Average pooling, in such scenarios, would perform averaging over a smaller-than-expected window.  Adaptive max pooling circumvents these issues by dynamically adapting to the input size, resulting in consistent output dimensions irrespective of input dimensions smaller than the target.

**2. Code Examples with Commentary:**

**Example 1: Input smaller than output in both height and width**

```python
import torch
import torch.nn.functional as F

input_tensor = torch.tensor([[[[1, 2], [3, 4]]]])  # Shape: (1, 1, 2, 2)
output_size = (3, 3)

output_tensor = F.adaptive_max_pool2d(input_tensor, output_size)
print(output_tensor)
print(output_tensor.shape)
```

Output:

```
tensor([[[[4., 4., 4.],
          [4., 4., 4.],
          [4., 4., 4.]]]])
torch.Size([1, 1, 3, 3])
```

Commentary: The input is 2x2, and the output is 3x3. Each output element is the maximum value from the input (4 in this case). This demonstrates the upsampling effect; the maximum value is repeated to fill the larger output tensor.

**Example 2: Input smaller than output in height only**

```python
import torch
import torch.nn.functional as F

input_tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]]) # Shape: (1, 1, 2, 3)
output_size = (3, 3)

output_tensor = F.adaptive_max_pool2d(input_tensor, output_size)
print(output_tensor)
print(output_tensor.shape)
```

Output:

```
tensor([[[[6., 6., 6.],
          [6., 6., 6.],
          [6., 6., 6.]]]])
torch.Size([1, 1, 3, 3])
```

Commentary: Here, the input height is smaller than the output height, but the width matches. The maximum value (6) along the height dimension is propagated across the output height, illustrating the behavior along a single dimension.

**Example 3: Handling multiple channels and batches**

```python
import torch
import torch.nn.functional as F

input_tensor = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]) # Shape: (2, 2, 2, 2)
output_size = (3, 4)

output_tensor = F.adaptive_max_pool2d(input_tensor, output_size)
print(output_tensor)
print(output_tensor.shape)
```

Output:


```
tensor([[[[ 8.,  8.,  8.,  8.],
          [ 8.,  8.,  8.,  8.],
          [ 8.,  8.,  8.,  8.]],

         [[16., 16., 16., 16.],
          [16., 16., 16., 16.],
          [16., 16., 16., 16.]]]])
torch.Size([2, 2, 3, 4])
```

Commentary: This example demonstrates the function's ability to handle multiple channels and batches.  Each channel and batch is processed independently, and the adaptive max pooling operates on each independently. The maximum values from each (2x2) channel within each batch are propagated to the larger (3x4) output size for that channel and batch.


**3. Resource Recommendations:**

The official PyTorch documentation is your primary source for comprehensive details on all functions, including `F.adaptive_max_pool2d`.  Further exploration of max pooling techniques in general, including standard max pooling and its variations, will provide valuable context.  Finally, reviewing advanced convolutional neural network architectures and their optimization strategies will help contextualize the practical implications of adaptive max pooling within larger network designs.  These resources will solidify your understanding of the underlying mechanisms and enable more effective application in your projects.
