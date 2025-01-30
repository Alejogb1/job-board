---
title: "How to split an image into four equal patches using PyTorch/TensorFlow?"
date: "2025-01-30"
id: "how-to-split-an-image-into-four-equal"
---
Splitting an image into four equal patches using either PyTorch or TensorFlow involves manipulating tensor dimensions to effectively extract quadrants from the input image. This operation, fundamental to tasks like patch-based image processing or parallel model training, leverages each framework’s tensor slicing capabilities. Having implemented custom image processing pipelines extensively for various machine learning projects, I've found a consistent, efficient methodology.

Essentially, the process involves first determining the height and width of the input image. Then, we calculate the midpoint for both dimensions, which serves as the splitting point for creating the four quadrants. Finally, we extract the patches using tensor slicing operations specific to each framework. Both PyTorch and TensorFlow offer robust mechanisms for this task, though their syntax differs slightly. The core concept, however, remains consistent: addressability of pixels as tensors and their slicing.

Let me illustrate with code examples.

**Example 1: PyTorch Implementation**

```python
import torch

def split_image_pytorch(image_tensor):
    """
    Splits a PyTorch image tensor into four equal patches.

    Args:
        image_tensor (torch.Tensor): A tensor of shape (C, H, W) representing an image.
                                     C is number of channels, H is height, and W is width.

    Returns:
        list: A list of four PyTorch tensors, representing the four patches.
              The order of the patches is top-left, top-right, bottom-left, bottom-right.
    """
    _, height, width = image_tensor.shape  # Image tensor should have shape (C, H, W)
    mid_height = height // 2
    mid_width = width // 2

    top_left = image_tensor[:, :mid_height, :mid_width]
    top_right = image_tensor[:, :mid_height, mid_width:]
    bottom_left = image_tensor[:, mid_height:, :mid_width]
    bottom_right = image_tensor[:, mid_height:, mid_width:]

    return [top_left, top_right, bottom_left, bottom_right]


if __name__ == '__main__':
    # Example usage
    # Create a sample 3-channel image (e.g., RGB) of 256x256 pixels.
    dummy_image = torch.rand(3, 256, 256)
    patches = split_image_pytorch(dummy_image)

    # Verify shapes of patches
    for i, patch in enumerate(patches):
        print(f"Patch {i+1} shape: {patch.shape}")
    #Expected output: Patch 1 shape: torch.Size([3, 128, 128])
    #                Patch 2 shape: torch.Size([3, 128, 128])
    #                Patch 3 shape: torch.Size([3, 128, 128])
    #                Patch 4 shape: torch.Size([3, 128, 128])

```

In this PyTorch implementation, the `split_image_pytorch` function takes a tensor representing the image, extracts its height and width using the `shape` attribute, computes the midpoints for both dimensions using integer division (`//`), and then uses standard tensor slicing notation to extract the four patches. The `:` operator signifies "all," when used before a dimension index. For example, `[:, :mid_height, :mid_width]` means "take all the channels, up to the mid-height in the vertical dimension, and up to mid-width in the horizontal dimension," effectively selecting the top-left quadrant. The remaining code comments on the expected shapes of the resultant patch tensors, each being half the original image's height and width while retaining the same channel dimension.

**Example 2: TensorFlow Implementation**

```python
import tensorflow as tf

def split_image_tensorflow(image_tensor):
    """
    Splits a TensorFlow image tensor into four equal patches.

    Args:
        image_tensor (tf.Tensor): A tensor of shape (H, W, C) representing an image.
                                  H is height, W is width, and C is number of channels.

    Returns:
        list: A list of four TensorFlow tensors, representing the four patches.
              The order of the patches is top-left, top-right, bottom-left, bottom-right.
    """

    height = tf.shape(image_tensor)[0]
    width = tf.shape(image_tensor)[1]
    mid_height = height // 2
    mid_width = width // 2

    top_left = image_tensor[:mid_height, :mid_width, :]
    top_right = image_tensor[:mid_height, mid_width:, :]
    bottom_left = image_tensor[mid_height:, :mid_width, :]
    bottom_right = image_tensor[mid_height:, mid_width:, :]

    return [top_left, top_right, bottom_left, bottom_right]


if __name__ == '__main__':
    # Example usage:
    # Create a sample 3-channel image (e.g., RGB) of 256x256 pixels.
    dummy_image = tf.random.normal(shape=(256, 256, 3))
    patches = split_image_tensorflow(dummy_image)

    # Verify the shape of the patches
    for i, patch in enumerate(patches):
        print(f"Patch {i+1} shape: {patch.shape}")
    #Expected output: Patch 1 shape: (128, 128, 3)
    #                Patch 2 shape: (128, 128, 3)
    #                Patch 3 shape: (128, 128, 3)
    #                Patch 4 shape: (128, 128, 3)
```

The TensorFlow variant `split_image_tensorflow` accomplishes the same task as its PyTorch counterpart. However, it operates on tensors with a shape of (H, W, C), where the channel dimension is last.  Importantly, it employs `tf.shape` to obtain the height and width as TensorFlow tensors which are then utilized for slicing. The slicing syntax remains similar to PyTorch, demonstrating the underlying consistency in the manipulation of tensor dimensions across the two frameworks. The output shapes are, as expected, consistent with each patch being of size 128x128x3 (assuming a 256x256 RGB input).

**Example 3: Handling Non-Divisible Dimensions**

```python
import torch
import tensorflow as tf

def split_image_pytorch_flexible(image_tensor):
    """
    Splits a PyTorch image tensor into four patches, handling non-divisible dimensions.

    Args:
        image_tensor (torch.Tensor): A tensor of shape (C, H, W) representing an image.
    Returns:
        list: A list of four PyTorch tensors, representing the four patches.
    """
    _, height, width = image_tensor.shape
    mid_height = height // 2
    mid_width = width // 2

    top_left = image_tensor[:, :mid_height, :mid_width]
    top_right = image_tensor[:, :mid_height, mid_width:]
    bottom_left = image_tensor[:, mid_height:, :mid_width]
    bottom_right = image_tensor[:, mid_height:, mid_width:]

    return [top_left, top_right, bottom_left, bottom_right]

def split_image_tensorflow_flexible(image_tensor):
    """
     Splits a TensorFlow image tensor into four patches, handling non-divisible dimensions.
     Args:
        image_tensor (tf.Tensor): A tensor of shape (H, W, C) representing an image.
    Returns:
        list: A list of four TensorFlow tensors, representing the four patches.
    """
    height = tf.shape(image_tensor)[0]
    width = tf.shape(image_tensor)[1]
    mid_height = height // 2
    mid_width = width // 2

    top_left = image_tensor[:mid_height, :mid_width, :]
    top_right = image_tensor[:mid_height, mid_width:, :]
    bottom_left = image_tensor[mid_height:, :mid_width, :]
    bottom_right = image_tensor[mid_height:, mid_width:, :]

    return [top_left, top_right, bottom_left, bottom_right]


if __name__ == '__main__':

    # Create a sample PyTorch image tensor with odd dimensions (e.g., 255x255)
    dummy_image_torch = torch.rand(3, 255, 255)
    patches_torch = split_image_pytorch_flexible(dummy_image_torch)
    for i, patch in enumerate(patches_torch):
        print(f"PyTorch Patch {i+1} shape: {patch.shape}")
    # Expected Output : PyTorch Patch 1 shape: torch.Size([3, 127, 127])
    #                   PyTorch Patch 2 shape: torch.Size([3, 127, 128])
    #                   PyTorch Patch 3 shape: torch.Size([3, 128, 127])
    #                   PyTorch Patch 4 shape: torch.Size([3, 128, 128])


    # Create a sample TensorFlow image tensor with odd dimensions (e.g., 255x255)
    dummy_image_tf = tf.random.normal(shape=(255, 255, 3))
    patches_tf = split_image_tensorflow_flexible(dummy_image_tf)
    for i, patch in enumerate(patches_tf):
        print(f"TensorFlow Patch {i+1} shape: {patch.shape}")

    # Expected Output: TensorFlow Patch 1 shape: (127, 127, 3)
    #                   TensorFlow Patch 2 shape: (127, 128, 3)
    #                   TensorFlow Patch 3 shape: (128, 127, 3)
    #                   TensorFlow Patch 4 shape: (128, 128, 3)

```

This third example showcases how to handle situations where the height and/or width of the input image are not perfectly divisible by two (odd dimensions). Integer division (`//`) will produce an integer result, which will cause the patches to slightly differ in size, when compared to a case where even dimensions are used. For instance, a 255x255 image will yield patches of sizes 127x127, 127x128, 128x127 and 128x128 when split. This is often acceptable as the patch shapes are as close to even as possible. Both the PyTorch and TensorFlow implementations are adapted, but remain consistent in logic, handling this scenario implicitly through integer division. This allows flexibility and ensures that the framework is functional even on non-uniform image sizes. Note the comment on the expected output that explains the variance in the produced shape.

For further understanding of tensor manipulation, I would recommend consulting the official documentation for both PyTorch and TensorFlow. Specifically, the sections on tensor indexing and slicing. Furthermore, exploring introductory materials on convolutional neural networks can provide context on why patch-based image processing is a common task. Several online tutorials covering these topics are available. Specifically I would recommend the PyTorch and Tensorflow’s official tutorials, and certain online courses like those found at Coursera or Udacity. These will go into further detail and can help give context into more advanced use-cases.
