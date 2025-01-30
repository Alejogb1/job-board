---
title: "How can I divide an image into patches using Keras/TensorFlow?"
date: "2025-01-30"
id: "how-can-i-divide-an-image-into-patches"
---
The core task of dividing an image into patches, often referred to as image tiling or cropping, is fundamental in many computer vision applications, including training convolutional neural networks (CNNs) with limited memory, performing fine-grained analysis, and implementing patch-based matching algorithms. My experience in developing object detection models using satellite imagery frequently required efficient image patching, which pushed me to explore the optimal implementations within TensorFlow and Keras.

The process involves extracting smaller, non-overlapping or overlapping rectangular regions from a larger input image. This operation can be approached using a combination of TensorFlow's tensor manipulation functions, primarily `tf.image.extract_patches` and some manual reshaping if necessary. While it’s achievable using nested loops, these are demonstrably slower than leveraging TensorFlow’s optimized tensor operations. The method I’ve found to be most performant hinges on configuring `tf.image.extract_patches` correctly. This function, at its core, slides a window of specified size across the input image with defined strides, extracting the content within the window at each location. The result is a tensor containing all the extracted patches.

The input image is first represented as a tensor with dimensions `(batch, height, width, channels)`. If you have a single image, the batch dimension should be 1. Parameters crucial for `tf.image.extract_patches` are: `sizes`, `strides`, and `rates`. `sizes` dictates the dimensions of each patch, formatted as `[1, patch_height, patch_width, 1]`. The initial and final `1` are required for the batch and channel dimensions, respectively; the patch height and width are the dimensions of the area to extract. `strides` controls the movement of the window during extraction, with a format of `[1, stride_height, stride_width, 1]`. Overlapping patches occur if strides are less than the patch size. Finally, `rates` defines dilation in the patch sampling; `[1, 1, 1, 1]` denotes no dilation or no skips within the receptive field of each patch.

Here’s a breakdown through practical examples:

**Example 1: Non-Overlapping Patches**

This first example demonstrates the extraction of non-overlapping patches, a common scenario for training with smaller chunks of a very large image. Here, I will generate a dummy 100x100 image, treat it as a single image in a batch, and then divide into 25x25 patches.

```python
import tensorflow as tf
import numpy as np

# Create a dummy image (batch size 1)
image = np.random.rand(1, 100, 100, 3).astype(np.float32)
image_tensor = tf.convert_to_tensor(image)

# Parameters for patching
patch_size = [1, 25, 25, 1]
strides = [1, 25, 25, 1]
rates = [1, 1, 1, 1]
padding = 'VALID' # No padding

# Extract patches
patches = tf.image.extract_patches(
    images=image_tensor,
    sizes=patch_size,
    strides=strides,
    rates=rates,
    padding=padding
)

print(f"Shape of patches tensor: {patches.shape}")

# Reshape for visualization/further processing (optional)
patch_count_height = 100 // 25
patch_count_width = 100 // 25

patches_reshaped = tf.reshape(patches, (patch_count_height * patch_count_width, 25, 25, 3))
print(f"Shape of reshaped patches: {patches_reshaped.shape}")

```
In this code, the `padding='VALID'` indicates that no additional pixels are added around the image border. If a patch extends beyond image edges, it will be truncated. The resulting `patches` tensor will have the shape `(1, 4, 4, 1875)`, indicating that there are 4x4 patches each containing 25x25x3 = 1875 values. The reshaping step combines patches into one batch of (16, 25, 25, 3).

**Example 2: Overlapping Patches**

Here, I extract overlapping patches, which are used in approaches requiring information from the same area within the image; such as segmentation, where overlapping patches can help recover objects at patch boundaries.

```python
import tensorflow as tf
import numpy as np

# Create a dummy image (batch size 1)
image = np.random.rand(1, 100, 100, 3).astype(np.float32)
image_tensor = tf.convert_to_tensor(image)

# Parameters for patching
patch_size = [1, 25, 25, 1]
strides = [1, 10, 10, 1] # Smaller strides for overlapping patches
rates = [1, 1, 1, 1]
padding = 'VALID' # No padding

# Extract patches
patches = tf.image.extract_patches(
    images=image_tensor,
    sizes=patch_size,
    strides=strides,
    rates=rates,
    padding=padding
)

print(f"Shape of patches tensor: {patches.shape}")

# Reshape for visualization/further processing (optional)
height_patches = (100 - 25) // 10 + 1
width_patches = (100 - 25) // 10 + 1

patches_reshaped = tf.reshape(patches, (height_patches * width_patches, 25, 25, 3))
print(f"Shape of reshaped patches: {patches_reshaped.shape}")
```
The critical difference in this example is that the `strides` are set to `[1, 10, 10, 1]`. Because the stride (10) is less than the patch size (25), extracted patches overlap. The shape of the resulting patches tensor is `(1, 8, 8, 1875)`, which is  8x8 overlapping patches each containing 1875 values after reshape into a batch `(64, 25, 25, 3)`. Note how the formula `(input_size - patch_size) // stride + 1` gives number of patches in each dimension.

**Example 3: Applying Padding and Extracting Patches**

Sometimes it's needed to process complete images where the dimensions aren't multiples of the patch size. Padding is key here, ensuring patches are extracted even for pixel areas near the boundary of the image. Here, I demonstrate the use of ‘SAME’ padding.

```python
import tensorflow as tf
import numpy as np

# Create a dummy image (batch size 1) with non-uniform size.
image = np.random.rand(1, 103, 107, 3).astype(np.float32)
image_tensor = tf.convert_to_tensor(image)

# Parameters for patching
patch_size = [1, 25, 25, 1]
strides = [1, 25, 25, 1]
rates = [1, 1, 1, 1]
padding = 'SAME' # Apply Padding

# Extract patches
patches = tf.image.extract_patches(
    images=image_tensor,
    sizes=patch_size,
    strides=strides,
    rates=rates,
    padding=padding
)

print(f"Shape of patches tensor: {patches.shape}")

# Reshape for visualization/further processing (optional)
height_patches = (103 + 24) // 25
width_patches = (107 + 24) // 25
patches_reshaped = tf.reshape(patches, (height_patches * width_patches, 25, 25, 3))
print(f"Shape of reshaped patches: {patches_reshaped.shape}")

```

By using `'SAME'` padding, TensorFlow adds the necessary pixels around the input image border to ensure that every pixel of the original image is fully covered by patches. Note that the actual padded image is never stored directly, but `tf.image.extract_patches` acts as if that padding has occurred. The shapes of the output patches are `(1, 5, 5, 1875)`, and after reshape it becomes `(25, 25, 25, 3)`.

In summary, `tf.image.extract_patches` provides a high-performance way to decompose images into patches. Careful attention to the `sizes`, `strides`, and `padding` parameters allows for flexible patch extraction behavior, whether non-overlapping, overlapping, or with padding applied. When the resulting patches need to be reshaped into a conventional `(num_patches, patch_height, patch_width, channels)` format, simple reshaping with `tf.reshape` is sufficient.

For further exploration of tensor manipulation and image preprocessing within TensorFlow, I recommend exploring the official TensorFlow documentation, particularly the sections dedicated to image operations and tensor transformations. There are also excellent online courses specifically covering deep learning with TensorFlow and Keras that provide many exercises and case studies relevant to this task. In addition, consulting examples in the Keras application codebase provides valuable insight into best practices. Finally, numerous academic publications on image analysis and CNNs can greatly deepen the theoretical understanding of the principles behind patch-based processing. Understanding the foundational math behind convolution will help understand how this tensor operation is fundamentally different than an explicit loop implementation. These various resources should help any developer become proficient in this critical image processing step.
