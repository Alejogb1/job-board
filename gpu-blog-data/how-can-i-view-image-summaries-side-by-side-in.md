---
title: "How can I view image summaries side-by-side in TensorBoard?"
date: "2025-01-30"
id: "how-can-i-view-image-summaries-side-by-side-in"
---
The visualization of image summaries in TensorBoard, particularly when seeking side-by-side comparison across different training epochs or model configurations, requires specific logging techniques and the appropriate structuring of image data. It's not an automatic feature; one must deliberately shape the data fed to TensorBoard's image logging API to achieve a comparative layout. My past work involving convolutional neural network training for medical image analysis repeatedly demanded this capability for diagnostic assessment, where small variations across training phases could become paramount.

The core principle rests on packaging multiple images within a single loggable tensor. TensorBoard interprets a single image tensor as either a grayscale (H x W), a grayscale with an alpha channel (H x W x 2), or a color image (H x W x 3 or H x W x 4). Critically, it also recognizes batches of images when a fourth dimension is added, yielding (N x H x W x C), where N is the batch size. TensorBoard then lays out these images in a grid, which is precisely what we manipulate to create side-by-side visual comparisons.

To illustrate this, consider that instead of logging separate image summaries for images `image_A` and `image_B`, we will combine them into a single tensor. We can achieve this by concatenating them along their vertical or horizontal axes. When combined horizontally, an image of size H x W is placed to the left of an image of the same size. Vertically, one will be positioned above the other. TensorBoard then displays this composite image as a single entry in the image summaries tab. To create side-by-side summaries, each composite image in a batch of composite images will be interpreted separately.

Let’s move to concrete code examples using Python with TensorFlow to demonstrate how this works.

**Code Example 1: Horizontal Side-by-Side Comparison**

This first example demonstrates how to combine two images horizontally. I'm assuming that `image_A` and `image_B` are already loaded or created as NumPy arrays or TensorFlow tensors of the same shape, specifically H x W x C, where C is the number of color channels (typically 3 for RGB).

```python
import tensorflow as tf
import numpy as np

def create_side_by_side_horizontal(image_A, image_B):
    """
    Combines two images horizontally for side-by-side comparison.

    Args:
        image_A: A TensorFlow tensor or NumPy array of shape (H, W, C).
        image_B: A TensorFlow tensor or NumPy array of shape (H, W, C).

    Returns:
        A TensorFlow tensor of shape (H, 2*W, C) representing the composite image.
    """
    combined_image = tf.concat([image_A, image_B], axis=1)
    return combined_image

# Example usage with dummy images
image_height, image_width, channels = 64, 64, 3
image_A = np.random.rand(image_height, image_width, channels).astype(np.float32)
image_B = np.random.rand(image_height, image_width, channels).astype(np.float32)

combined_image_tensor = create_side_by_side_horizontal(image_A, image_B)

# Add a batch dimension so it can be logged as a summary
combined_image_tensor = tf.expand_dims(combined_image_tensor, axis=0)

# Log to tensorboard:
log_dir = "logs/example_horizontal"
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.image("Side-by-Side Images (Horizontal)", combined_image_tensor, step=0)
```
This code first defines a function `create_side_by_side_horizontal` to take two images and combine them horizontally using `tf.concat` along the width dimension (`axis=1`).  I've included a docstring to clarify input and output. The example usage creates two random images of size 64x64x3. After these are combined, `tf.expand_dims` adds a batch dimension. Finally, the generated composite image is logged using `tf.summary.image`. When you load this TensorBoard instance, a single image entry named “Side-by-Side Images (Horizontal)” will appear, consisting of `image_A` on the left and `image_B` on the right.

**Code Example 2: Vertical Side-by-Side Comparison**

The next example illustrates a vertical arrangement, useful when comparing two sets of images that are naturally read top-to-bottom.

```python
import tensorflow as tf
import numpy as np

def create_side_by_side_vertical(image_A, image_B):
    """
    Combines two images vertically for side-by-side comparison.

    Args:
        image_A: A TensorFlow tensor or NumPy array of shape (H, W, C).
        image_B: A TensorFlow tensor or NumPy array of shape (H, W, C).

    Returns:
        A TensorFlow tensor of shape (2*H, W, C) representing the composite image.
    """
    combined_image = tf.concat([image_A, image_B], axis=0)
    return combined_image

# Example usage with dummy images
image_height, image_width, channels = 64, 64, 3
image_A = np.random.rand(image_height, image_width, channels).astype(np.float32)
image_B = np.random.rand(image_height, image_width, channels).astype(np.float32)

combined_image_tensor = create_side_by_side_vertical(image_A, image_B)

# Add a batch dimension so it can be logged as a summary
combined_image_tensor = tf.expand_dims(combined_image_tensor, axis=0)

# Log to tensorboard:
log_dir = "logs/example_vertical"
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.image("Side-by-Side Images (Vertical)", combined_image_tensor, step=0)
```
This is structurally similar to the first example.  The key difference is using `tf.concat` with `axis=0` to stack the two images vertically, resulting in a composite image of height `2 * image_height` and the same width. The rest of the logging process is analogous. In TensorBoard, you will now see a single image summary where `image_A` is stacked atop `image_B`. This is critical when comparing, for example, the input and output of an image segmentation model.

**Code Example 3: Batching Side-by-Side Comparisons**

The previous examples only demonstrated combining two images into one composite image, logging it as one single image in TensorBoard. To observe multiple such combinations at once, we must create a batch of composite images.

```python
import tensorflow as tf
import numpy as np

def create_batched_side_by_side_horizontal(image_pairs):
   """
    Combines multiple image pairs horizontally for side-by-side comparison
    in a batch.

    Args:
        image_pairs: A list of tuples, where each tuple contains two TensorFlow tensors or NumPy arrays,
                     each of shape (H, W, C).

    Returns:
        A TensorFlow tensor of shape (N, H, 2*W, C), where N is the number of image pairs.
    """
   combined_images = []
   for image_A, image_B in image_pairs:
        combined_image = tf.concat([image_A, image_B], axis=1)
        combined_images.append(combined_image)

   batched_images = tf.stack(combined_images, axis=0)
   return batched_images

# Example usage with dummy images, creating a batch of 3 pairs
image_height, image_width, channels = 64, 64, 3
image_pairs = []
for i in range(3):
  image_A = np.random.rand(image_height, image_width, channels).astype(np.float32)
  image_B = np.random.rand(image_height, image_width, channels).astype(np.float32)
  image_pairs.append((image_A, image_B))


batched_combined_image_tensor = create_batched_side_by_side_horizontal(image_pairs)

# Log to tensorboard:
log_dir = "logs/example_batched"
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
  tf.summary.image("Batched Side-by-Side Images (Horizontal)", batched_combined_image_tensor, step=0, max_outputs=3)
```
This final example demonstrates how to create a batched series of combined images, by generalizing the functionality of `create_side_by_side_horizontal` to `create_batched_side_by_side_horizontal`. The example now creates three pairs of random images, iterates through them to horizontally concatenate them, and stacks these into a single batch using `tf.stack`. I've included a `max_outputs=3` argument for clarity, but TensorBoard will likely handle more composite images. The TensorBoard will then display the three combined images below each other. This is invaluable when comparing multiple image pairs at different epochs.

To summarize, achieving side-by-side image summaries in TensorBoard is a matter of structuring the image data before logging it.  You must combine separate image tensors into a single composite tensor either horizontally or vertically, before adding a batch dimension and log them to TensorBoard. To view multiple such comparisons at once, log a batch of such composite images.

For more detailed information on TensorFlow image handling, explore the TensorFlow documentation for `tf.concat` and `tf.summary.image`. Further investigation into NumPy’s array manipulation functions can help with creating custom visual layouts and processing steps before tensors are passed into TensorBoard. Also examine tutorials and code examples related to image preprocessing within the context of convolutional neural networks to understand common manipulation techniques.
