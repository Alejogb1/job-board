---
title: "How can I generate images per batch in PyTorch?"
date: "2025-01-30"
id: "how-can-i-generate-images-per-batch-in"
---
Generating images per batch in PyTorch during training or inference offers crucial benefits, especially for visually inspecting model progress or for downstream tasks requiring intermediate image outputs. Directly accessing and saving images after each batch requires careful management of data flow and PyTorch’s tensor structure.  I've dealt with this scenario in several projects, including a recent GAN implementation for medical image synthesis, and the key challenge is efficiently extracting pixel data from the tensors representing the generated images and then converting that data into a usable image format.

The core problem arises from the nature of PyTorch tensors. They are multidimensional arrays holding numerical values representing pixel intensities, not image files directly.  These tensors need to be transformed into a format that can be saved as an image file, typically a NumPy array compatible with libraries like Pillow (PIL) or OpenCV.  Furthermore, PyTorch works with batched inputs and outputs; each tensor output from a model represents a batch of multiple images. Extracting and handling each image within the batch separately requires iterating through the tensor’s batch dimension.

Here's a step-by-step approach detailing how I routinely handle this using common PyTorch and supporting libraries:

**1. Understanding Tensor Dimensions:**

PyTorch, by convention, often represents images with tensors of shape `(B, C, H, W)`, where `B` is the batch size, `C` is the number of channels (e.g., 3 for RGB, 1 for grayscale), `H` is the height, and `W` is the width. Before attempting to extract individual images, one must ensure the tensor conforms to this expected shape. Additionally, if the pixel values are normalized (e.g., to the range [-1, 1]), they need to be unnormalized to the typical pixel range of [0, 255] before conversion.

**2. Extracting Images Per Batch:**

After a model forward pass produces the image tensor output, the process involves:
   a.  Iterating through the batch dimension of the tensor.
   b.  Slicing the tensor to isolate a single image `(C, H, W)`.
   c.  If necessary, undoing any normalization.
   d.  Permuting the dimensions to `(H, W, C)` (this is commonly required for image processing libraries).
   e.  Converting the tensor to a NumPy array.
   f.  Using an image library (PIL, OpenCV) to save the NumPy array as an image file.

**3. Code Examples:**

Here are three code snippets illustrating different aspects of this process:

**Example 1: Basic Image Saving after Un-normalization**

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def save_batch_images(images, batch_idx, output_dir, unnormalize=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Saves images from a batch to disk.

    Args:
        images (torch.Tensor):  Tensor of shape (B, C, H, W).
        batch_idx (int): The current batch number.
        output_dir (str):  Directory to save the images.
        unnormalize (bool, optional):  If images are normalized, should be set to true. Defaults to False.
        mean (tuple, optional): Mean of normalization values. Defaults to (0.5, 0.5, 0.5).
        std (tuple, optional): Std of normalization values. Defaults to (0.5, 0.5, 0.5).
    """
    B, C, H, W = images.shape
    os.makedirs(output_dir, exist_ok=True)

    if unnormalize:
        mean_tensor = torch.tensor(mean).reshape(1, C, 1, 1).to(images.device)
        std_tensor = torch.tensor(std).reshape(1, C, 1, 1).to(images.device)
        images = images * std_tensor + mean_tensor  # Unnormalize the images

    for image_idx in range(B):
        image = images[image_idx] # shape (C, H, W)
        image = image.permute(1, 2, 0).cpu().detach().numpy() # Shape (H,W,C)
        image = np.clip(image * 255, 0, 255).astype(np.uint8) # Clip to 0-255 and convert to uint8
        image_pil = Image.fromarray(image)
        image_pil.save(os.path.join(output_dir, f"batch_{batch_idx}_image_{image_idx}.png"))

# Example Usage (assuming `output` is a tensor from a model with shape (B, C, H, W))
# Generate a dummy output
output = torch.rand(10, 3, 64, 64)
save_batch_images(output, 0, "output_images")
```

*Explanation:*  This example demonstrates the basic structure of saving images from a batch. It iterates through the batch dimension, undoes normalization if indicated, permutes dimensions to the `(H, W, C)` convention, then uses PIL to create and save each image file. The clipping and conversion to `uint8` are critical to ensure correct representation when writing to an image file.

**Example 2: Saving Gray-Scale Images**

```python
import torch
import numpy as np
from PIL import Image
import os

def save_grayscale_batch_images(images, batch_idx, output_dir):
     """Saves grayscale images from a batch to disk.

    Args:
        images (torch.Tensor):  Tensor of shape (B, 1, H, W).
        batch_idx (int): The current batch number.
        output_dir (str):  Directory to save the images.
    """
    B, C, H, W = images.shape

    if C != 1:
      raise ValueError("Input tensor must have a single channel for grayscale.")

    os.makedirs(output_dir, exist_ok=True)

    for image_idx in range(B):
        image = images[image_idx]
        image = image.squeeze(0).cpu().detach().numpy()  # Shape (H, W) and remove channel dimension
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image, mode='L') # 'L' mode for grayscale
        image_pil.save(os.path.join(output_dir, f"batch_{batch_idx}_image_{image_idx}.png"))

# Example usage (assuming  `output_gray` is a tensor of size (B, 1, H, W)
output_gray = torch.rand(5, 1, 32, 32)
save_grayscale_batch_images(output_gray, 2, "grayscale_output")
```

*Explanation:* This example showcases the handling of single-channel (grayscale) images. The key difference is that after slicing the tensor to obtain a single image, the channel dimension is removed (using `squeeze(0)`) before conversion. Also the PIL `Image.fromarray` function is used with the 'L' mode specifier to treat the image data as grayscale, which is critical for proper file storage.

**Example 3:  Saving Images with Custom Format and Quality**

```python
import torch
import numpy as np
from PIL import Image
import os

def save_batch_images_custom_options(images, batch_idx, output_dir, unnormalize=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), save_format="jpeg", quality=85):
    """Saves images with custom format and quality options.

    Args:
        images (torch.Tensor):  Tensor of shape (B, C, H, W).
        batch_idx (int): The current batch number.
        output_dir (str):  Directory to save the images.
        unnormalize (bool, optional):  If images are normalized, should be set to true. Defaults to False.
        mean (tuple, optional): Mean of normalization values. Defaults to (0.5, 0.5, 0.5).
        std (tuple, optional): Std of normalization values. Defaults to (0.5, 0.5, 0.5).
        save_format (str, optional): Format to save images. Defaults to 'jpeg'.
        quality (int, optional): Quality of JPEG images. Defaults to 85.
    """
    B, C, H, W = images.shape
    os.makedirs(output_dir, exist_ok=True)

    if unnormalize:
        mean_tensor = torch.tensor(mean).reshape(1, C, 1, 1).to(images.device)
        std_tensor = torch.tensor(std).reshape(1, C, 1, 1).to(images.device)
        images = images * std_tensor + mean_tensor

    for image_idx in range(B):
        image = images[image_idx]
        image = image.permute(1, 2, 0).cpu().detach().numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        image_pil.save(os.path.join(output_dir, f"batch_{batch_idx}_image_{image_idx}.{save_format.lower()}"), format=save_format, quality=quality)


# Example usage (assuming  `output_custom` is a tensor of size (B, 3, H, W)
output_custom = torch.rand(5, 3, 64, 64)
save_batch_images_custom_options(output_custom, 1, "custom_output", save_format="jpeg", quality=90)

```
*Explanation:* This example demonstrates more fine-grained control over the image saving process. It adds parameters to specify the output image format (e.g., "jpeg", "png") and quality settings, if the saving format is lossy. This offers a way to further optimize storage during training, or for producing images suitable for further analysis.

**Resource Recommendations:**

For continued learning and exploration, consider the following resources. The official PyTorch documentation offers deep dives into tensor manipulation and working with computational graphs. Study the documentation on `torch.Tensor` and understand the various manipulation techniques, such as `permute`, `unsqueeze`, and `squeeze`.  Pillow (PIL) documentation is useful for image saving and manipulation with different formats.  Exploring tutorials focused on image data augmentation with PyTorch and torchvision can also enhance understanding of typical tensor workflows.  Finally, NumPy’s documentation is key to understanding array manipulations, as it is crucial to understand how to convert between different data types. These resources combined provide a solid base for image generation and processing in a PyTorch context.
