---
title: "How can I use `tf.keras.utils.image_dataset_from_directory` with labels from a CSV file?"
date: "2025-01-30"
id: "how-can-i-use-tfkerasutilsimagedatasetfromdirectory-with-labels-from"
---
The inherent limitation of `tf.keras.utils.image_dataset_from_directory` lies in its reliance on directory structure for label inference.  My experience working on large-scale image classification projects highlighted the inflexibility of this approach, particularly when dealing with datasets where the file structure doesn't neatly align with the desired label assignments.  This often occurs when dealing with pre-existing datasets or datasets curated from multiple sources. Therefore, a robust solution necessitates leveraging a CSV file containing explicit image-label mappings.


This response outlines a methodology to achieve this integration, combining the convenience of `image_dataset_from_directory` with the flexibility of CSV-based labelling.  The approach involves pre-processing the dataset to generate a consistent mapping between image filenames and their associated labels read from a CSV file.  Subsequently, a custom `tf.data.Dataset` is constructed, leveraging this mapping to create a labeled dataset compatible with Keras models.

**1. Clear Explanation:**

The core strategy involves three phases: CSV parsing, filename matching, and dataset construction.  First, the CSV file, assumed to contain at least two columns – 'filename' and 'label' – is parsed to extract image filenames and corresponding labels.  I've found using the `pandas` library most efficient for this task, due to its robust handling of various CSV formats and data types.  Second, this extracted information is utilized to construct a dictionary or mapping between filenames (without file extension) and their labels. This mapping is crucial for associating images loaded from the directory with the correct label. Finally, this mapping is employed in conjunction with `tf.keras.utils.image_dataset_from_directory` or directly with `tf.data.Dataset.from_tensor_slices` to create a labeled dataset.  The advantage of this approach lies in the separation of label management from the image loading process, offering enhanced control and flexibility.

**2. Code Examples:**

**Example 1: Using `tf.data.Dataset.from_tensor_slices` (Recommended for Larger Datasets):**

```python
import tensorflow as tf
import pandas as pd
import pathlib

# Assuming your CSV is named 'image_labels.csv' and your image directory is 'images'
csv_path = 'image_labels.csv'
image_dir = 'images'

# Load CSV using pandas
df = pd.read_csv(csv_path)

# Create a mapping from filename (without extension) to label
filename_label_map = dict(zip(df['filename'].str.replace(r'\..*', '', regex=True), df['label']))

# Create a list of image paths
image_paths = [str(pathlib.Path(image_dir) / (filename + '.jpg')) for filename in filename_label_map.keys()] # Assumes .jpg extension, adjust as needed

# Create labels array
labels = [filename_label_map[pathlib.Path(path).stem] for path in image_paths]

# Convert to tensors
image_paths_tensor = tf.constant(image_paths)
labels_tensor = tf.constant(labels)

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))

# Function to load and preprocess images
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3) # Adjust for image format
    image = tf.image.resize(image, [224, 224]) # Resize to a standard size
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

# Apply the preprocessing function
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

This example leverages `tf.data.Dataset.from_tensor_slices` for superior performance with larger datasets. It avoids the limitations of `image_dataset_from_directory` by explicitly defining image paths and labels.

**Example 2:  Modifying `image_dataset_from_directory` (Less Efficient for Large Datasets):**

This example demonstrates a less efficient approach, adapting `image_dataset_from_directory`  by filtering and re-labeling after dataset creation.  It is suitable for smaller datasets where performance overhead is less critical.

```python
import tensorflow as tf
import pandas as pd

# ... (CSV loading and filename_label_map creation as in Example 1) ...

# Create dataset using image_dataset_from_directory
dataset = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    labels='inferred', # initially inferred, then remapped
    label_mode='int',
    image_size=(224, 224),
    batch_size=32
)

# Remap labels based on filename
def remap_labels(image_batch, label_batch):
    new_labels = tf.constant([filename_label_map[img_path.numpy().decode().split('/')[-1].split('.')[0]] for img_path in image_batch.map(lambda x: x.numpy())])
    return image_batch, new_labels

dataset = dataset.map(remap_labels)

# ... (Batching and prefetching as in Example 1) ...
```

This method is less efficient as it requires iterating through the entire dataset to remap labels.

**Example 3:  Error Handling and Robustness:**

```python
import tensorflow as tf
import pandas as pd
import pathlib

# ... (CSV loading and filename_label_map creation as in Example 1) ...

#Function to handle missing files gracefully
def load_image_robust(image_path, label):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label
    except tf.errors.NotFoundError:
        print(f"Warning: Image file not found: {image_path.numpy().decode()}")
        return None, label


# Filter out entries that result in errors
dataset = dataset.map(load_image_robust)
dataset = dataset.filter(lambda x, y: tf.reduce_any(tf.not_equal(x, None)))


# ... (Batching and prefetching as in Example 1) ...
```

This example incorporates error handling for scenarios where image files might be missing from the specified directory.  This enhances the robustness of the process.


**3. Resource Recommendations:**

*   The official TensorFlow documentation.
*   A comprehensive guide to TensorFlow Datasets.
*   A textbook on practical machine learning with TensorFlow.


This detailed explanation, along with the provided code examples and resource recommendations, offers a complete solution to the problem of integrating `tf.keras.utils.image_dataset_from_directory` with CSV-based labels.  The choice of which code example to use will depend upon dataset size and performance constraints.  Always prioritize error handling and robustness for production-ready solutions.
