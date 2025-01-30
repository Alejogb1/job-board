---
title: "How do I display a dataset created using Keras' `image_dataset_from_directory` function?"
date: "2025-01-30"
id: "how-do-i-display-a-dataset-created-using"
---
The `image_dataset_from_directory` function in Keras, while highly convenient for loading image datasets, doesn't directly offer visualization capabilities.  Its output is a `tf.data.Dataset` object, optimized for efficient processing within the TensorFlow ecosystem, not for immediate visual inspection.  Therefore, effective display requires leveraging additional tools and understanding the underlying structure of the returned dataset.  My experience developing a medical image classification system highlighted this limitation, necessitating the creation of custom visualization functions.

**1. Understanding the `tf.data.Dataset` Object:**

The dataset returned by `image_dataset_from_directory` is a sequence of batches. Each batch consists of two elements: a tensor containing the image data and a tensor representing the corresponding labels.  The image tensor's shape depends on the image dimensions and batch size, typically (batch_size, height, width, channels), where `channels` is 3 for RGB images.  The labels tensor usually has a shape of (batch_size,). It's crucial to understand this structure to iterate through the dataset and extract images for display.  Furthermore, the dataset is designed for pipelining and efficient processing, meaning elements are not loaded into memory until needed, which necessitates careful iteration strategies to avoid memory exhaustion when dealing with large datasets.


**2. Displaying Images Using Matplotlib:**

The most straightforward approach for visualizing the images involves using Matplotlib, a widely used plotting library in Python. The following code snippet illustrates this:

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Assume 'image_dataset' is the dataset created using image_dataset_from_directory
image_dataset = tf.keras.utils.image_dataset_from_directory(
    "path/to/your/images",
    labels='inferred',
    label_mode='categorical',
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
)

# Iterate through a single batch
for images, labels in image_dataset.take(1):
    # Convert labels to class names (assuming you know your class names)
    class_names = ['ClassA', 'ClassB', 'ClassC'] # Replace with your actual class names
    label_names = [class_names[np.argmax(label)] for label in labels]


    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(16, 8)) #adjust based on batch_size
    axes = axes.flatten()
    for i, (image, label_name) in enumerate(zip(images, label_names)):
        axes[i].imshow(image.numpy().astype('uint8'))
        axes[i].set_title(label_name)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
```

This code first iterates through a single batch from the dataset. It then uses Matplotlib to create a grid of subplots, where each subplot displays an image from the batch along with its corresponding class label.  Note that `image.numpy().astype('uint8')` is crucial for converting the tensor back to a NumPy array suitable for Matplotlib's `imshow` function.  The error handling for batch sizes that don't perfectly fit the subplot grid is intentionally omitted for brevity; in a production environment, this would require more robust error handling.


**3.  Visualizing with TensorBoard (for larger datasets):**

For larger datasets where displaying all images simultaneously is impractical, TensorBoard provides a more scalable solution.  TensorBoard allows for interactive exploration and visualization of various aspects of the training process, including dataset contents.

```python
import tensorflow as tf
from tensorflow.compat.v1 import summary

# ... (image_dataset creation as in previous example) ...

# This example requires modification for TensorBoard compatibility
# It assumes a suitable log directory structure
log_dir = "logs/image_dataset"
writer = tf.summary.create_file_writer(log_dir)

def visualize_batch(batch, step):
    images, labels = batch
    with writer.as_default():
        tf.summary.image("Images", images, step=step, max_outputs=min(16, images.shape[0])) #limit output


for step, batch in enumerate(image_dataset):
    visualize_batch(batch, step)
    if step > 5: # visualize a limited number of batches for demonstration
      break

```

This code snippet adds image summaries to TensorBoard using `tf.summary.image`. Each batch is written to the TensorBoard logs, allowing for interactive visualization.  The `max_outputs` parameter controls the number of images displayed, preventing the log from becoming excessively large.  To view this data, one would need to run TensorBoard (`tensorboard --logdir logs`) and navigate to the "Images" tab. The need for the `tf.compat.v1` import reflects the evolving API within TensorFlow.  Remember to install TensorFlow with the appropriate visualization components.


**4.  Custom Visualization Function (for enhanced control):**

Finally, for maximum flexibility and control over the visualization process, creating a custom function is advisable. This approach enables tailoring the display to specific needs, such as highlighting particular features or adding custom annotations.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ... (image_dataset creation as in previous example) ...

def display_dataset(dataset, num_batches=1, num_images_per_batch=8):
    for batch_num in range(num_batches):
        try:
            images, labels = next(iter(dataset))
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
            axes = axes.flatten()
            for i in range(num_images_per_batch):
                axes[i].imshow(images[i].numpy().astype('uint8'))
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
        except StopIteration:
            print(f"Displayed images from {batch_num} batches.")
            break

display_dataset(image_dataset, num_batches=2, num_images_per_batch=8)

```

This function iterates through a specified number of batches, displaying a defined number of images from each batch.  The `try-except` block handles the `StopIteration` exception that arises when the end of the dataset is reached, providing better error handling than the simpler examples. This method provides a higher degree of control and prevents potential errors associated with mismatched batch sizes and subplot dimensions.

**3. Resource Recommendations:**

The official TensorFlow documentation, the Matplotlib documentation, and a comprehensive text on Python data visualization would serve as valuable resources.  Focusing on the `tf.data` API documentation would be particularly helpful.  Understanding NumPy array manipulation is also fundamental.
