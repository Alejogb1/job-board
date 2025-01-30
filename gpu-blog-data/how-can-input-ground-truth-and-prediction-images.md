---
title: "How can input, ground truth, and prediction images be displayed side-by-side in TensorBoard?"
date: "2025-01-30"
id: "how-can-input-ground-truth-and-prediction-images"
---
The core challenge in visualizing input, ground truth, and prediction images alongside each other within TensorBoard lies in structuring the data appropriately for TensorBoard's image summarization capabilities.  My experience working on a medical image segmentation project highlighted the necessity of a well-defined data pipeline to achieve this effective visualization.  TensorBoard doesn't inherently support a three-way comparison;  the solution involves carefully crafting a single summary image containing all three components.

**1. Data Preparation and Structuring:**

The fundamental step is creating a composite image.  This single image will contain the input, ground truth, and prediction images arranged side-by-side.  The dimensions of each component must be consistent to avoid distortions in the final composite.  I typically utilize NumPy for this task due to its efficiency in array manipulations.  The precise method depends on the image format (e.g., grayscale, RGB).  For RGB images, the approach differs slightly from grayscale, requiring careful handling of color channels.  Here's the general principle:

* **Input Image:** This is the raw input data fed into the model.
* **Ground Truth Image:**  The corresponding correct labels or segmentation map.
* **Prediction Image:** The output of your model, representing the predicted segmentation or classification.

All three images must be resized to a common height and width to maintain a consistent aspect ratio within the composite.  If the images have different numbers of channels (e.g., grayscale vs. RGB), one must be appropriately converted or padded before concatenation.

**2. Code Examples:**

The following examples demonstrate the creation of the composite image using Python and NumPy.  These examples assume the input, ground truth, and prediction are already loaded as NumPy arrays.  Error handling (e.g., checking for shape mismatches) is omitted for brevity, but is crucial in a production environment.  My experience taught me the importance of robust error handling to prevent unexpected behavior.

**Example 1: Grayscale Images**

```python
import numpy as np
import tensorflow as tf

def create_composite_grayscale(input_img, ground_truth, prediction):
    """Creates a composite image for grayscale images."""
    # Ensure all images have the same dimensions. Resize if necessary.
    target_size = (max(input_img.shape[0], ground_truth.shape[0], prediction.shape[0]),
                   max(input_img.shape[1], ground_truth.shape[1], prediction.shape[1]))
    input_img = np.resize(input_img, target_size)
    ground_truth = np.resize(ground_truth, target_size)
    prediction = np.resize(prediction, target_size)

    # Stack the images horizontally.  Expand dims for grayscale images to ensure correct stacking.
    composite = np.concatenate((np.expand_dims(input_img, axis=-1),
                                np.expand_dims(ground_truth, axis=-1),
                                np.expand_dims(prediction, axis=-1)), axis=1)
    return composite

# Example usage (replace with your actual image data)
input_img = np.random.rand(64, 64)
ground_truth = np.random.rand(64, 64)
prediction = np.random.rand(64, 64)

composite_image = create_composite_grayscale(input_img, ground_truth, prediction)

tf.summary.image("composite_image", np.expand_dims(composite_image, axis=0))
```

**Example 2: RGB Images**

```python
import numpy as np
import tensorflow as tf

def create_composite_rgb(input_img, ground_truth, prediction):
    """Creates a composite image for RGB images."""
    # Ensure all images have the same dimensions. Resize if necessary.  Handles RGB channels.
    target_size = (max(input_img.shape[0], ground_truth.shape[0], prediction.shape[0]),
                   max(input_img.shape[1], ground_truth.shape[1], prediction.shape[1]))
    input_img = np.resize(input_img, target_size + (3,))
    ground_truth = np.resize(ground_truth, target_size + (3,))
    prediction = np.resize(prediction, target_size + (3,))


    composite = np.concatenate((input_img, ground_truth, prediction), axis=1)
    return composite

# Example Usage (replace with your actual image data)
input_img = np.random.rand(64, 64, 3)
ground_truth = np.random.rand(64, 64, 3)
prediction = np.random.rand(64, 64, 3)

composite_image = create_composite_rgb(input_img, ground_truth, prediction)

tf.summary.image("composite_image", np.expand_dims(composite_image, axis=0))
```

**Example 3: Handling different channel numbers**

```python
import numpy as np
import tensorflow as tf

def create_composite_mixed(input_img, ground_truth, prediction):
    """Handles images with different channel numbers (e.g., grayscale and RGB)."""
    # Resize to a common size. The following assumes ground truth and prediction are RGB, and input is grayscale
    target_size = (max(input_img.shape[0], ground_truth.shape[0], prediction.shape[0]),
                   max(input_img.shape[1], ground_truth.shape[1], prediction.shape[1]))
    input_img = np.resize(input_img, (target_size[0], target_size[1], 1))
    ground_truth = np.resize(ground_truth, target_size + (3,))
    prediction = np.resize(prediction, target_size + (3,))

    #Replicate grayscale to RGB
    input_img = np.repeat(input_img, 3, axis=-1)

    composite = np.concatenate((input_img, ground_truth, prediction), axis=1)
    return composite

# Example usage:
input_img = np.random.rand(64, 64)
ground_truth = np.random.rand(64, 64, 3)
prediction = np.random.rand(64, 64, 3)

composite_image = create_composite_mixed(input_img, ground_truth, prediction)
tf.summary.image("composite_image", np.expand_dims(composite_image, axis=0))
```


**3. Integration with TensorBoard:**

Once the composite image is generated, utilize `tf.summary.image` within your TensorFlow training loop to write the summary to TensorBoard.  Remember that `tf.summary.image` expects a tensor of shape `[N, height, width, channels]`, where N is the batch size (often 1 for individual image comparisons). The examples above demonstrate this usage.

**4. Resource Recommendations:**

The official TensorFlow documentation on summaries and TensorBoard is indispensable.  Familiarize yourself with the various summary types and their applications.  Furthermore, a strong grasp of NumPy array manipulation and image processing techniques is crucial for effective data preparation.  Mastering these core concepts will allow for adaptable and robust visualization strategies for a wide array of image-related tasks.  Understanding image resizing algorithms and their impact on image quality is equally valuable.
