---
title: "How can I use `image_dataset_from_directory` with labels from a CSV file?"
date: "2024-12-23"
id: "how-can-i-use-imagedatasetfromdirectory-with-labels-from-a-csv-file"
---

Let's tackle this head-on. The challenge of aligning image data with labels stored separately in a CSV is a common hurdle, and it's something I've encountered countless times in my projects, particularly when dealing with custom datasets. The `image_dataset_from_directory` function in TensorFlow, while convenient, primarily works with a directory structure where labels are inferred from folder names. That won’t work for us. So, we need a different approach. Instead of relying on directory-based inference, we'll construct our dataset explicitly, using the data provided by our CSV file.

First, let's talk about the high-level strategy. The crux of the issue is that `image_dataset_from_directory` is built to map directory paths to labels. Our labels are in a CSV, and our images are, presumably, in some kind of folder hierarchy. Therefore, we’re essentially going to manually build a dataset from file paths and their corresponding labels using tensorflow's `tf.data.Dataset` API directly. We’ll achieve this in three primary steps:

1.  **CSV parsing:** Load and process the CSV to obtain a list of file paths and their associated labels.
2.  **Dataset creation:** Generate a `tf.data.Dataset` from the lists of file paths and labels.
3.  **Image loading and preprocessing:** Create a function to load and preprocess the images on the fly as data is requested from the dataset.

Here's the breakdown in more detail with practical code examples:

**1. CSV parsing and Data Extraction:**

I've often found that pandas is invaluable for processing tabular data like CSVs. For this task, I'd use the library to parse the CSV file into a dataframe for easier access to the paths and labels. It's essential to carefully examine the CSV format. Ensure that the path column corresponds to the location of your images, and the label column represents the class or numerical target.

```python
import pandas as pd
import tensorflow as tf

def load_data_from_csv(csv_path, image_dir):
  """Loads filepaths and labels from a csv, ensuring correct file path construction.

    Args:
        csv_path (str): Path to the CSV file.
        image_dir (str): Directory where images are located.

    Returns:
        tuple: Lists of image file paths and corresponding labels.
  """
  df = pd.read_csv(csv_path)

  # Assuming your CSV has columns named 'image_path' and 'label'
  # Here we make some robust handling for relative paths as well.
  image_paths = [
       os.path.join(image_dir, path) if not os.path.isabs(path) else path
       for path in df['image_path'].tolist()
   ]
  labels = df['label'].tolist()


  return image_paths, labels

# Example Usage
import os
csv_file = 'labels.csv' #replace with actual
image_directory = 'images/' #replace with actual.
image_paths, labels = load_data_from_csv(csv_file, image_directory)
print(f"Found {len(image_paths)} images and {len(labels)} labels")


```

In the above code, the crucial step is handling file paths correctly. In my experience, relative paths in CSVs are very common, so we include `os.path.join` with the `image_dir` to form the full paths. Additionally, an abs path check is used to handle paths that might have been supplied as absolute paths. This method also gives us the foundation for later steps.

**2. Dataset creation:**

Now that we've extracted our paths and labels, we can construct a `tf.data.Dataset`.  We'll be using `tf.data.Dataset.from_tensor_slices` which builds a dataset from tuples of path and label lists.

```python
def create_dataset(image_paths, labels):
    """Creates a tf.data.Dataset from file paths and labels."""

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    return dataset


# Example Usage:
dataset = create_dataset(image_paths, labels)
print(f"Created dataset with {len(list(dataset.as_numpy_iterator()))} elements.")

```

This step is foundational. We are building the dataset at this stage but we haven’t loaded any images yet.

**3. Image loading and preprocessing:**

We’ll now create a loading and preprocessing function that will be passed to `dataset.map()`.  Here we will read our images and perform any required preprocessing, such as resizing or rescaling. This is a critical performance step because, rather than loading everything into memory at the start, it will be executed only when data is requested, significantly reducing memory footprint.

```python
IMG_WIDTH = 224
IMG_HEIGHT = 224

def load_and_preprocess_image(image_path, label):
  """Loads an image, decodes it, and resizes and rescales it"""

  image = tf.io.read_file(image_path)
  image = tf.io.decode_jpeg(image, channels=3) #adjust to image type
  image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32) #rescale values to [0,1]
  return image, label


#Apply the function to the dataset:
preprocessed_dataset = dataset.map(load_and_preprocess_image)


# Example:
for image, label in preprocessed_dataset.take(2):
    print(f"Image shape: {image.shape}, Label: {label}")

```

Here, `tf.io.read_file` is used to load image files, `tf.io.decode_jpeg` (or other applicable decoding function) is used to convert it to a tensor, and `tf.image.resize` is used to standardize the size. Finally, `tf.image.convert_image_dtype` performs rescaling, normalizing pixel values to the [0,1] range. This rescaling is a good practice for most deep learning models.

**Putting It All Together:**

We’ve effectively pieced together a complete workflow, where data from a CSV file is used to create a `tf.data.Dataset`. We can now iterate over this dataset using standard batching and shuffling techniques and pass it to model training processes.

**Recommended Resources:**

*   **TensorFlow Documentation:** The official TensorFlow documentation is essential for detailed information on the `tf.data` API and image processing functionalities. Specifically, review the documentation for `tf.data.Dataset`, `tf.io`, and `tf.image`.
*   **"Deep Learning with Python" by François Chollet:** This is an excellent resource that covers the basics and advanced concepts of deep learning with Keras and TensorFlow.
*   **“Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron:** This book provides a more practical hands-on approach to building machine learning and deep learning models with Keras and TensorFlow.
*   **"Programming TensorFlow" by Ian Goodfellow, et al.:** A more technical reference, this is useful if you want a deeper understanding of the internals of TensorFlow.

**Important considerations:**

*   **Error Handling:** Consider adding error handling to your file path validation, and for any potential issues with image loading or decoding.
*   **Preprocessing choices:** Image preprocessing is highly problem dependent, so you may need to explore other preprocessing steps (e.g. data augmentation).

In summary, bypassing the directory structure and using a CSV file with `tf.data` gives you fine-grained control, flexibility, and robust handling of datasets. This approach allows me, and you, to effectively train models with more complex or atypical data setups, as I've seen in many real-world projects. The three code snippets above, combined with proper study of the suggested resources, should give you a working framework for your project.
