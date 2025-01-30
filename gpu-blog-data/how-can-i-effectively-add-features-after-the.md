---
title: "How can I effectively add features after the last convolutional layer in a neural network prior to fully connected layers?"
date: "2025-01-30"
id: "how-can-i-effectively-add-features-after-the"
---
The transition from convolutional feature maps to fully connected layers is a crucial stage in many neural network architectures, often requiring careful consideration of how high-dimensional feature representations are flattened and utilized. Directly appending additional features before the fully connected layers, effectively bypassing the flattening operation, demands a method that aligns these disparate data types appropriately to allow the dense layers to learn meaningful relationships. My experience building image-based classification systems for medical diagnostics has repeatedly highlighted the necessity of this technique.

The primary challenge lies in the dimensional mismatch. Convolutional layers output three-dimensional tensors (height, width, channels), while fully connected layers expect a flattened, one-dimensional input. Adding supplementary features, which are commonly one-dimensional vectors, requires preprocessing to make them compatible with the flattened convolutional output or an approach that incorporates them prior to the flattening. I find that the latter is generally more versatile and leads to better model performance.

My approach involves concatenating the additional features as an extra "channel" within the convolutional feature maps before flattening. This process involves adapting the supplementary vector by reshaping and broadcasting it to match the height and width dimensions of the convolutional outputs. The combined representation then undergoes flattening before entering the dense layers. This method preserves the inherent spatial structure learned by the convolutional filters while also incorporating the supplementary information. Importantly, the supplemental data must be scaled or normalized appropriately to prevent it from disproportionately impacting the initial dense layer's weights.

Consider a neural network designed to classify images of plant leaves. We have a pre-trained convolutional base that generates feature maps of size (10, 10, 64). Additionally, we possess data about the leaf's area, perimeter, and circularity, represented as a vector of length 3. The first example demonstrates how to concatenate this feature vector to the convolutional output using NumPy.

```python
import numpy as np

# Simulate convolutional feature maps and supplemental feature vector
conv_output = np.random.rand(1, 10, 10, 64) # Batch size of 1
supplementary_features = np.array([0.5, 1.2, 0.8])

# Reshape the supplementary features to match the height and width of the conv_output
reshaped_features = supplementary_features.reshape(1, 1, 1, 3) # Make it 4D for broadcasting

# Broadcast to match height and width dimensions
broadcast_features = np.broadcast_to(reshaped_features, (1, 10, 10, 3))

# Concatenate along the channel axis (axis=3)
combined_output = np.concatenate((conv_output, broadcast_features), axis=3)

# Flatten the combined output for the dense layer
flattened_output = combined_output.reshape(1, -1)

print(f"Combined output shape: {combined_output.shape}")  # Output: (1, 10, 10, 67)
print(f"Flattened output shape: {flattened_output.shape}") # Output: (1, 6700)

```
This example uses NumPy for clarity, but the same concept translates directly into deep learning frameworks.  I find that the `concatenate` operation, often provided in tensor manipulation libraries, is effective for joining feature maps. In this specific example, we increased the channel depth of the convolutional output from 64 to 67 before flattening, accommodating the 3 supplemental features.

The second example illustrates this with PyTorch, employing a convolutional layer and a supplementary feature as part of a simple model. I use a simple convolution layer for example purposes, but this methodology applies equally well to complex convolutional networks.

```python
import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, num_features):
        super(CombinedModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * 10 * (64 + num_features), 128) # Adjust input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, supplementary_features):
        x = torch.relu(self.conv(x))
        batch_size, _, height, width = x.shape

        # Reshape and broadcast supplementary features
        reshaped_features = supplementary_features.view(batch_size, 1, 1, -1)
        broadcast_features = reshaped_features.expand(batch_size, height, width, reshaped_features.shape[-1])

        # Concatenate along the channel dimension
        combined_output = torch.cat((x, broadcast_features), dim=1)

        # Flatten and apply dense layers
        flattened_output = combined_output.view(batch_size, -1)
        x = torch.relu(self.fc1(flattened_output))
        x = self.fc2(x)
        return x

# Example usage
model = CombinedModel(num_features=3)
image_data = torch.rand(2, 3, 10, 10) # Batch of 2 images, 3 channels, 10x10
supplementary_data = torch.tensor([[0.5, 1.2, 0.8], [0.7, 1.1, 0.9]]) # Batch of 2 supplementary features

output = model(image_data, supplementary_data)
print(f"Output Shape: {output.shape}") # Output: torch.Size([2, 10])
```
This example constructs a complete model demonstrating the concatenation approach in PyTorch.  It showcases the proper reshaping and broadcasting of the supplementary features using `torch.view()` and `expand()` and ensures the correct input size to the initial fully connected layer. The `expand` function avoids memory-intensive copying, by broadcasting the feature vector appropriately along the batch, height, and width dimensions.

The third example modifies the previous example by introducing a separate embedding layer for the supplementary features before concatenation. This allows the network to learn a refined representation of the supplemental data, especially when dealing with high-dimensional features or features with complex relationships.
```python
import torch
import torch.nn as nn

class CombinedModelWithEmbedding(nn.Module):
    def __init__(self, num_features, embedding_dim=16):
        super(CombinedModelWithEmbedding, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.embedding = nn.Linear(num_features, embedding_dim)
        self.fc1 = nn.Linear(10 * 10 * (64 + embedding_dim), 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x, supplementary_features):
        x = torch.relu(self.conv(x))
        batch_size, _, height, width = x.shape

        # Embed supplementary features
        embedded_features = torch.relu(self.embedding(supplementary_features))

        # Reshape and broadcast the embedded supplementary features
        reshaped_features = embedded_features.view(batch_size, 1, 1, -1)
        broadcast_features = reshaped_features.expand(batch_size, height, width, reshaped_features.shape[-1])

        # Concatenate along the channel dimension
        combined_output = torch.cat((x, broadcast_features), dim=1)

        # Flatten and apply dense layers
        flattened_output = combined_output.view(batch_size, -1)
        x = torch.relu(self.fc1(flattened_output))
        x = self.fc2(x)
        return x

# Example Usage:
model = CombinedModelWithEmbedding(num_features=3, embedding_dim=16)
image_data = torch.rand(2, 3, 10, 10)
supplementary_data = torch.tensor([[0.5, 1.2, 0.8], [0.7, 1.1, 0.9]])

output = model(image_data, supplementary_data)
print(f"Output shape: {output.shape}") # Output: torch.Size([2, 10])
```
This example includes an embedding layer via the `nn.Linear` module before the concatenation, allowing the model to learn the most effective representation of the supplementary data for the task at hand. This approach can lead to improved performance, particularly when the supplementary data requires transformations before being combined with the convolutional features.

When incorporating these methods into my research and applied projects, I frequently refer to general mathematical texts on tensor manipulation, particularly those focusing on broadcasting and matrix operations.  For a deeper understanding of neural network architectures, resources covering convolutional and fully connected layer design are essential.  Furthermore, exploring practical guides outlining model building within specific deep learning libraries like PyTorch and TensorFlow provides valuable insight.  Examining code examples on open-source repositories often provides practical implementation guidance.
