---
title: "Why is Pytorch DirectML having computational inconsistency?"
date: "2024-12-16"
id: "why-is-pytorch-directml-having-computational-inconsistency"
---

Alright, let's talk about the computational inconsistencies sometimes observed when using pytorch with directml. This is a topic I've spent a fair bit of time troubleshooting over the years, especially back when I was heavily involved in optimizing deep learning models for diverse hardware configurations. It's a nuanced issue, and there isn't a single root cause, but rather a confluence of factors. Let’s unpack it systematically.

Fundamentally, the problem stems from the fact that directml is a different hardware acceleration backend compared to nvidia’s cuda or even intel’s oneapi. These backends, while all striving to achieve the same computational results, often have different numerical implementations under the hood, especially in lower-level operations. It’s crucial to remember that floating-point arithmetic isn't always deterministic. The order of operations, the precise algorithms used in specific kernel implementations, and even small differences in rounding modes can lead to minute, but consequential variations, particularly when accumulated across many layers of a deep neural network.

One of the primary challenges is the precision level at which computations are performed. While we might typically specify fp32 (single-precision) or fp16 (half-precision) at the pytorch level, directml itself might handle some sub-operations in a slightly different manner. This could involve different levels of intermediate precision or subtly altered algorithmic approaches to achieve performance gains, leading to discrepancies from the outputs obtained with cuda, for example. When dealing with gradients, the issue can be exacerbated because small numerical differences can compound over backpropagation.

Another source of discrepancies emerges from how directml interprets and executes certain operations that are handled differently by other backends. For example, reductions, pooling operations, or even complex tensor manipulations like matrix multiplication, may be implemented using diverse algorithms that have their own inherent precision characteristics and sensitivities. The choice of algorithm is often a trade-off between speed and numerical accuracy. Directml's approach might prioritize performance, sometimes at the cost of bit-for-bit accuracy compared to cuda or other backends.

Furthermore, there's the issue of operator support and optimization levels. Not all pytorch operations might have fully optimized kernels in directml, and some might be implemented using fallback methods that introduce variations. When directml encounters an operation that doesn't have a specific optimized version, it will resort to a generic implementation which can again affect numerical consistency. These variations in how operators are handled can be a significant factor behind the inconsistencies we're observing.

Now, let’s illustrate these points with a few simplified, but practical examples.

**Example 1: The impact of accumulated error in a simple matrix multiplication**

Consider this snippet where we are doing basic multiplication using numpy, and a directml pytorch tensor to see if we get exactly the same result.

```python
import torch
import numpy as np

# Define a simple matrix
np_matrix = np.random.rand(3, 3).astype(np.float32)
torch_matrix = torch.tensor(np_matrix, dtype=torch.float32)
torch_matrix_dml = torch_matrix.to(device="directml")

# Perform a simple multiplication operation multiple times
for _ in range(100):
   np_matrix = np_matrix @ np.random.rand(3,3).astype(np.float32)
   torch_matrix_dml = torch_matrix_dml @ torch.rand(3,3,dtype=torch.float32,device="directml")

torch_matrix_dml_cpu=torch_matrix_dml.cpu().numpy()

# Compare the results, we must assume slight difference
difference = np.max(np.abs(np_matrix-torch_matrix_dml_cpu))
print(f"Max difference between numpy and directml tensor multiplication after 100 matrix multplications : {difference}")

```

While a single matrix multiplication might appear consistent, performing this iteratively amplifies the subtle differences in the computational behavior and causes a drift, especially if using directml, this is due to its different sub-level implementation. Running this, one will observe a small difference in the final matrices, something not seen with cuda. This is because the backend implementations for the same matrix operation will have a different way of accumulating the floating point errors inherent in any floating point arithmatic system.

**Example 2: A basic convolutional layer discrepancy**

Now consider this example, focusing on a basic convolutional operation, highlighting the differences that might arise from varying backend implementations:

```python
import torch
import torch.nn as nn

# Define a simple convolutional layer
conv_layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)

# Generate a sample tensor
input_tensor = torch.randn(1, 3, 28, 28)

# Run on CPU for comparison
output_cpu = conv_layer(input_tensor)

# Move to directml and run
conv_layer_dml = conv_layer.to(device="directml")
input_tensor_dml = input_tensor.to(device="directml")
output_dml = conv_layer_dml(input_tensor_dml)
output_dml_cpu = output_dml.cpu()

# Compare the results
difference = torch.max(torch.abs(output_cpu-output_dml_cpu))

print(f"Max difference between cpu and directml convolution layer output : {difference}")
```

This highlights that differences can exist even in a seemingly basic operation such as a conv layer. Again, subtle algorithmic differences, perhaps optimizations done by the directml backend and not by the cpu, or even sub-precision differences in intermediate calculations.

**Example 3: Activation Function comparison**

Let's look at a common neural network activation function, relu, for any differences:

```python
import torch
import torch.nn as nn

# Define a sample input
input_tensor = torch.randn(1, 100)

# Run on CPU
relu_cpu = nn.ReLU()
output_cpu = relu_cpu(input_tensor)

# Move to DirectML and run
input_tensor_dml = input_tensor.to(device="directml")
relu_dml = nn.ReLU().to(device="directml")
output_dml = relu_dml(input_tensor_dml)
output_dml_cpu = output_dml.cpu()

#Compare the results
difference = torch.max(torch.abs(output_cpu-output_dml_cpu))
print(f"Max difference between cpu and directml relu activation output : {difference}")
```

Even such a simple function can return slightly different outputs. The directml implementation may use a different numerical strategy to efficiently compute the non-linearity.

To navigate these discrepancies in practice, several strategies can be employed. The most fundamental one is to always test and validate your model’s output across different backends. This way, you’ll not be surprised by such discrepancies when you ship a product. Furthermore, be aware of the different performance characteristics of these backends. For instance, if you observe the discrepancies getting worse after adding more layers, perhaps you can try to limit or change the operations that have the greatest precision impact.

For resources, I would recommend checking out the official documentation for both pytorch and directml, ensuring you are aware of the officially supported operations. Specifically, the pytorch documentation on numerical precision is a must-read for anyone doing work in the field. You should also explore the technical papers from the deep learning community that focus on numerical precision in training and inference (I would recommend anything from the DeepMind group on this). Lastly, “Numerical Recipes: The Art of Scientific Computing,” by William H. Press et al., remains a gold standard reference for understanding numerical computation issues in general.

In my experience, complete bit-for-bit consistency between different backends, especially different hardware architectures, isn't always achievable, nor is it always necessary. However, understanding the underlying reasons for the differences and mitigating their impact can make a real difference to your development process. Being aware of these nuances and implementing proper testing and validation becomes indispensable. It’s about understanding the precision limits and making sure they are within acceptable margins for the specific application.
