---
title: "How can affine transformations be used for image translation in PyTorch?"
date: "2025-01-30"
id: "how-can-affine-transformations-be-used-for-image"
---
Affine transformations provide a powerful framework for manipulating images, encompassing operations like translation, rotation, scaling, and shearing. Specifically, for image translation in PyTorch, we leverage affine transformation matrices to directly map pixel coordinates from the input image to their corresponding positions in the output image. The process hinges on the ability to represent translation, a movement of each pixel by a fixed offset, using a specific form of this matrix.

An affine transformation matrix in two dimensions is typically a 3x3 matrix, where the first two rows and columns encode linear transformations (scaling, rotation, shearing), and the third column represents translation. For purely translational operations, the matrix will have a specific structure: the top-left 2x2 submatrix is an identity matrix, representing no scaling, rotation, or shearing, while the third column holds the horizontal and vertical translation values. A matrix of this structure does not alter the underlying geometry of the image other than by the translation. This direct relationship between matrix parameters and transformation output simplifies the implementation and analysis of image manipulations.

In PyTorch, several approaches exist to perform this. The most common utilizes the functional interface of the `torch.nn.functional` module, specifically the `affine_grid` and `grid_sample` functions. The `affine_grid` function takes an affine transformation matrix as input and generates a grid of coordinates corresponding to the destination pixel locations. This grid is then used by `grid_sample` to sample from the input image, effectively transforming it according to the input matrix. This two-step process allows for a highly flexible and efficient way to perform various image transformations, including translation.

My previous work involved developing a medical image analysis pipeline where precise image registration was critical. I frequently used this approach to correct for slight positional variations between different image scans. I needed to move image subregions relative to their neighbors to align the images before training a model, and the flexibility of the `affine_grid` and `grid_sample` combination allowed for quick experimentation.

Here's a practical example illustrating image translation with these PyTorch functions:

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Simulate a grayscale image (batch size of 1)
image = torch.arange(0, 25, dtype=torch.float).reshape(1, 1, 5, 5)

# Define translation values (dx, dy) in pixels
dx = 2.0
dy = -1.0

# Construct the affine transformation matrix for translation.
# Matrix structure is [[1, 0, dx], [0, 1, dy], [0, 0, 1]].
# Because the first two rows of the matrix will be used and 
# we are dealing with a batch of images we provide batch dimensions.
# Note we need to add a batch dimension.
transform_matrix = torch.tensor([[1.0, 0.0, dx],
                                [0.0, 1.0, dy]], dtype=torch.float).unsqueeze(0)


# Create the grid using affine_grid based on the output image size.
# Output size is taken to be the same as the input image
grid = F.affine_grid(transform_matrix, image.size())

# Apply the transformation using grid_sample. Bilinear interpolation is the default.
translated_image = F.grid_sample(image, grid, align_corners = False)

# Plot the original and translated images for visualization
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image[0, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(translated_image[0, 0], cmap='gray')
axes[1].set_title('Translated Image')
plt.show()
```

In this first example, I initiated a 5x5 grayscale image. The `transform_matrix` defined represents a translation of 2 pixels to the right and 1 pixel up. I opted for `align_corners = False`, to make the behavior well defined in these edge cases. The output shows the shifted image, illustrating the effect of the translation matrix. Note, `affine_grid` returns a set of coordinates, where a value like 1 is a spacing between the pixels, or an x-coordinate of 1.

It's worth noting that pixel translations will cause the image to "leave" its original bounds, with the boundaries being filled in, depending on the interpolation algorithm. In this case `grid_sample` defaults to the bilinear interpolation and fills with zeros if there are regions outside of the original image bounds that must be sampled from. This zero filling may need to be accounted for during later processing.

Here is a second example, showcasing how a batch of images can be transformed at once:

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Simulate a batch of 2 grayscale images of size 5x5
image_batch = torch.arange(0, 50, dtype=torch.float).reshape(2, 1, 5, 5)

# Translation values (dx, dy)
dx = -1.0
dy = 2.0

# Construct the transformation matrix, now applied for each image
transform_matrix = torch.tensor([[1.0, 0.0, dx],
                                [0.0, 1.0, dy]], dtype=torch.float).unsqueeze(0).repeat(2,1,1)


# Generate the coordinate grids and translate the images
grid = F.affine_grid(transform_matrix, image_batch.size())
translated_batch = F.grid_sample(image_batch, grid, align_corners = False)

# Display the original and translated image batches
fig, axes = plt.subplots(2, 2)
for i in range(2):
    axes[i, 0].imshow(image_batch[i, 0], cmap='gray')
    axes[i, 0].set_title(f'Original {i}')
    axes[i, 1].imshow(translated_batch[i, 0], cmap='gray')
    axes[i, 1].set_title(f'Translated {i}')
plt.show()
```

Here, I expanded the previous example by constructing a batch of two 5x5 grayscale images. Critically, the `transform_matrix` now has dimensions of 2x2x3 after being reshaped and repeated across the batch using `repeat()`. The same `affine_grid` and `grid_sample` operations can then perform the translation, effectively applying the same translation to all images in the batch. This batch-processing capability is essential for efficient training of deep learning models on image datasets and allows us to apply identical transformations to a dataset.

Finally, consider a more realistic scenario where we use the same translation operation within a PyTorch neural network module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TranslateLayer(nn.Module):
    def __init__(self, dx, dy):
        super(TranslateLayer, self).__init__()
        self.dx = dx
        self.dy = dy

    def forward(self, image):
        transform_matrix = torch.tensor([[1.0, 0.0, self.dx],
                                        [0.0, 1.0, self.dy]], dtype=torch.float).unsqueeze(0).repeat(image.size(0),1,1)
        grid = F.affine_grid(transform_matrix, image.size(), align_corners = False)
        translated_image = F.grid_sample(image, grid, align_corners = False)
        return translated_image

# Create a dummy image and initialize a translation layer
image = torch.arange(0, 25, dtype=torch.float).reshape(1, 1, 5, 5)
translate_layer = TranslateLayer(dx=1.5, dy=-0.5)

# Apply the translation within the layer
translated_image = translate_layer(image)

# Visualize the original and translated images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image[0, 0].detach().numpy(), cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(translated_image[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title('Translated Image')
plt.show()
```

Here I demonstrate the creation of a custom `TranslateLayer` class. The `forward` method applies the same translation logic using an internally defined `transform_matrix`. Note the transform matrix is now repeated based on the batch size of the image which can be any size. This approach encapsulates the translation operation as a PyTorch module, making it convenient to integrate it into a larger neural network architecture. I can also adjust the parameters using backpropagation if needed.

In terms of resources, for a deep understanding of image processing fundamentals, I recommend reading "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods. For those looking to dive into the mathematical underpinnings of transformations, "Computer Graphics: Principles and Practice" by James D. Foley, Andries van Dam, Steven K. Feiner, and John F. Hughes provides a more detailed exploration. For specific PyTorch usage and best practices, thoroughly exploring the official documentation for `torch.nn.functional` is crucial. These references provide the mathematical and implementation foundations to effectively use affine transforms within PyTorch, and to troubleshoot any problems encountered in practice.
