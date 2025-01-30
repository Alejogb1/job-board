---
title: "How can I create a custom image dataset similar to tf.keras.datasets.cifar10?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-image-dataset"
---
The core challenge when creating a custom image dataset resembling `tf.keras.datasets.cifar10` lies in emulating its structured data loading capabilities, particularly its return of pre-split training and testing sets, alongside accompanying labels, directly usable within TensorFlow's Keras. I've found that building such datasets requires careful management of file paths, image loading, and label association, ultimately yielding a `tf.data.Dataset` object for efficient training.

A crucial element is defining the expected file structure on disk. Unlike the built-in datasets which often download and manage files internally, a custom dataset demands a specific organizational format. In my experience, a common and easily managed structure is a directory-based approach where each subdirectory represents a distinct class. Within each subdirectory are the image files belonging to that class. For example, a dataset of images of 'cats', 'dogs' and 'birds' might have directories named 'cats', 'dogs' and 'birds' with relevant images inside each one respectively.

The primary workflow then involves several key steps: first, generating file paths and extracting associated labels; second, implementing a function to load and decode image data; and third, utilizing TensorFlow’s `tf.data.Dataset` API to generate efficient, batched data ready for model training.

**Step 1: Generating File Paths and Labels**

The initial step involves parsing your directory structure to generate pairs of image file paths and corresponding class labels. In my projects, I typically use Python’s standard library `os` to achieve this.

```python
import os
import tensorflow as tf

def create_file_label_pairs(base_dir):
    """Generates file path and label pairs from a directory structure.

    Args:
      base_dir: The root directory containing class subdirectories.

    Returns:
      A tuple of lists: (image_file_paths, labels).
    """
    image_file_paths = []
    labels = []
    class_names = sorted(os.listdir(base_dir)) # Ensure deterministic order
    for class_index, class_name in enumerate(class_names):
      class_dir = os.path.join(base_dir, class_name)
      if os.path.isdir(class_dir): # Handle non-directory entries if any
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_file_paths.append(image_path)
            labels.append(class_index)
    return image_file_paths, labels
```

This function `create_file_label_pairs` traverses the `base_dir`, identifying class subdirectories. It then iterates through each image file, constructing the full path and appending both path and corresponding class index (label) to respective lists. Sorting the directory listing guarantees a consistent order for classes.

**Step 2: Image Loading and Preprocessing**

Once we have file paths and labels, we need to write a function to load and potentially preprocess each image. Common preprocessing steps include decoding (converting from a file format like JPEG or PNG to a tensor) and resizing.

```python
def load_and_preprocess_image(image_path, label):
    """Loads, decodes, and preprocesses an image.

    Args:
      image_path: Path to the image file (string).
      label: The class label (integer).

    Returns:
      A tuple: (preprocessed_image_tensor, label).
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assumes JPEG images
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Normalize
    image = tf.image.resize(image, [32, 32])  # Resize to CIFAR-10 shape
    return image, label
```

This function `load_and_preprocess_image` takes an `image_path` string and a corresponding `label`. It utilizes `tf.io.read_file` to load the raw file data, decodes it as a JPEG using `tf.image.decode_jpeg`, ensuring it has three color channels. It normalizes to range [0,1] by using `tf.image.convert_image_dtype` which also converts it into a float32 tensor and then resizes the image to 32x32 pixels, matching the CIFAR-10 dataset using `tf.image.resize`.  The function returns a tensor representing the image and the associated label. Error handling for mismatched file types should be added to a production implementation.

**Step 3: Generating a `tf.data.Dataset`**

Finally, the key step: generating a `tf.data.Dataset` object. This involves using `tf.data.Dataset.from_tensor_slices` to create a dataset from our file paths and labels, and then applying our `load_and_preprocess_image` function to each element in parallel using `map` with `num_parallel_calls` for efficiency. We also configure the dataset for batching and prefetching.

```python
def create_image_dataset(base_dir, batch_size=32, seed=None):
    """Creates a batched tf.data.Dataset.

    Args:
      base_dir: The root directory containing class subdirectories.
      batch_size: The batch size for the dataset.
      seed: Optional random seed for shuffling.

    Returns:
      A tf.data.Dataset object.
    """
    image_file_paths, labels = create_file_label_pairs(base_dir)
    dataset = tf.data.Dataset.from_tensor_slices((image_file_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_file_paths), seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```
This function, `create_image_dataset`, takes a `base_dir` (where our class directories reside), a desired `batch_size`, and an optional `seed` for shuffling. It first retrieves file path and label pairs. Then, a `tf.data.Dataset` is initialized using `tf.data.Dataset.from_tensor_slices`. The `load_and_preprocess_image` function is applied, employing `num_parallel_calls` equal to `tf.data.AUTOTUNE` to maximize processing throughput. The `shuffle` method randomizes data order; if a seed is provided this becomes deterministic. The `batch` method organizes data into batches and prefetching with `tf.data.AUTOTUNE` keeps the data pipeline performing optimally.

**Splitting into Training and Test Sets**

Splitting this into training and testing sets requires some additional processing. After creating an initial dataset from all the data, you could split the generated list of file paths and labels into training and testing subsets based on a predefined ratio (e.g., 80/20 split). You can then feed each separate list into `tf.data.Dataset.from_tensor_slices` and apply the same transformations.

**Resource Recommendations**

For those looking to deepen their understanding, I recommend exploring the official TensorFlow documentation, specifically the `tf.data` module. The documentation on building custom input pipelines is invaluable. In addition, textbooks on machine learning and deep learning often feature dedicated sections on data preparation, which is a key aspect of any successful model implementation. Finally, carefully examine example repositories that leverage `tf.data` to get a sense of how others manage and structure complex data workflows. Examining more complex examples, such as handling image augmentations and more diverse file formats can also be helpful.
