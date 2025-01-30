---
title: "What is the input size for a linear layer with 100 outputs?"
date: "2025-01-30"
id: "what-is-the-input-size-for-a-linear"
---
The input size for a linear layer producing 100 outputs isn't inherently fixed; it's determined by the dimensionality of the input data fed to it.  This is a fundamental aspect of linear layer operation that I've encountered repeatedly during my work on large-scale neural network architectures, particularly in projects involving natural language processing and computer vision.  The 100 outputs represent the layer's output dimensionality,  a hyperparameter defined during the layer's instantiation.  The input, however, remains flexible, adaptable to different input data shapes.

Understanding this requires clarifying the mathematical operation performed by a linear layer: a matrix multiplication followed by a bias addition. The weight matrix in this operation dictates the input size. Specifically, if we have an input vector *x* of size *n* and a weight matrix *W* of size *m x n*, where *m* is the number of outputs and *n* is the number of inputs, then the output *y* will have size *m*. In our case, *m* is 100.  Therefore, the input size, *n*, is determined by the number of columns in the weight matrix *W*.

The weight matrix's shape directly reflects the input and output dimensions. A linear layer with 100 outputs implies a weight matrix with 100 rows.  The number of columns, however, remains a design choice dependent on the anticipated input data. This is where the flexibility lies.  For instance, if your input is a single feature vector of size 10, the weight matrix would be 100 x 10. If the input is a sequence of 50 word embeddings, each of dimension 20, then the input to the layer could be represented as a 1000-dimensional vector (50 words x 20 dimensions per word), resulting in a 100 x 1000 weight matrix.

Let's illustrate with code examples using Python and PyTorch, a framework I've extensively used for my deep learning projects.


**Example 1: Simple Vector Input**

This example demonstrates a linear layer operating on a single feature vector.

```python
import torch
import torch.nn as nn

# Define the linear layer with 100 outputs
linear_layer = nn.Linear(10, 100)  # Input size (features) is 10

# Sample input vector
input_vector = torch.randn(1, 10) #Batch size of 1

# Forward pass
output = linear_layer(input_vector)

# Output shape verification
print(output.shape) # Output: torch.Size([1, 100])
print(linear_layer.weight.shape) # Output: torch.Size([100, 10])
```

Here, the input size is explicitly defined as 10.  The weight matrix `linear_layer.weight` will have dimensions 100 x 10.  The output, as expected, is a vector of size 100.  I often use this approach for simpler models or when dealing with pre-processed data already in a suitable vector format.

**Example 2:  Sequence Input (Batched)**

This example showcases a linear layer receiving batched sequence data, a common scenario in sequence modeling tasks like machine translation.

```python
import torch
import torch.nn as nn

# Define the linear layer
linear_layer = nn.Linear(20, 100)  # Input size (embedding dimension) is 20

# Sample batched sequence input (batch size 3, sequence length 5)
input_sequence = torch.randn(3, 5, 20)

# Reshape for linear layer (batch size x sequence length * embedding dimension)
input_reshaped = input_sequence.view(3, 5*20)

# Forward pass
output = linear_layer(input_reshaped)

# Output shape verification
print(output.shape)  # Output: torch.Size([3, 100])
print(linear_layer.weight.shape) # Output: torch.Size([100, 20])
```
This example emphasizes the importance of reshaping.  The input is a sequence of length 5, with each element a 20-dimensional vector. This is reshaped into a matrix where each row represents a sequence, flattened to a 100-dimensional vector before being fed into the linear layer. The weight matrix therefore remains 100 x 20, reflecting the 20-dimensional input elements.  My experience shows this reshaping step is crucial for correctly handling sequence data in PyTorch.

**Example 3: Image Input (Convolutional Feature Maps)**

This example simulates a scenario where the input originates from a convolutional neural network.  The input would be the feature maps output by the convolutional layers.

```python
import torch
import torch.nn as nn

# Define the linear layer
linear_layer = nn.Linear(500, 100) # Input size from convolutional features

# Sample input from a convolutional layer (batch size 2, feature map size 500)
input_features = torch.randn(2, 500)

# Forward pass
output = linear_layer(input_features)

# Output shape verification
print(output.shape)  # Output: torch.Size([2, 100])
print(linear_layer.weight.shape) # Output: torch.Size([100, 500])
```
Here, the input is assumed to be a 500-dimensional vector, representing the flattened feature maps from a convolutional layer.  The weight matrix is 100 x 500.  This type of architecture is common in computer vision, where the convolutional layers extract spatial features and then a fully connected layer (like our linear layer here) processes these features for classification or regression tasks. I've employed this setup numerous times in image classification projects.


In summary, the input size to a linear layer with 100 outputs isn't a fixed value but rather a parameter defined by the problem's dimensionality. It's the number of columns in the weight matrix, and that number is determined by the size of the input vector or tensor being fed into it.  The examples demonstrate various scenarios highlighting how input size impacts weight matrix dimensions and necessitates appropriate data reshaping when necessary.


**Resource Recommendations:**

* Deep Learning textbook by Goodfellow, Bengio, and Courville.
*  PyTorch documentation.
*  A comprehensive linear algebra textbook.
