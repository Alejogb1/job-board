---
title: "How do I extract filenames from a TensorFlow Dataset?"
date: "2025-01-30"
id: "how-do-i-extract-filenames-from-a-tensorflow"
---
Extracting filenames from a TensorFlow `Dataset` is not a direct operation because the `Dataset` API is designed for data transformation and pipeline creation, not for exposing underlying filepaths. The dataset, once constructed, typically operates on batches of tensors. Consequently, the file paths used during the initial construction phase are abstracted away. This means a dataset might be built from an already processed in-memory data structure, a text file, or even a remote server. Therefore, attempting to extract the filenames directly from an existing dataset object is not generally feasible. However, during the initial dataset construction, several methods allow you to store filenames alongside the data, which you can later extract. My experience building large-scale image processing pipelines in TensorFlow has shown me that careful planning of dataset construction is key to accessing filenames, and this usually involves explicit steps.

The core challenge lies in how you initially construct the dataset. If you created the dataset using `tf.data.Dataset.list_files`, the filenames are indeed known but are no longer accessible once the file paths are read, processed, and cached. Therefore, you must preserve this information during loading if you want to later extract the filenames. The most common approach involves mapping the filepaths to data and corresponding filenames as tuple values, ensuring that the filename component is explicitly included in the output of the mapping function. Furthermore, if you're using functions like `tf.io.decode_jpeg` or `tf.io.read_file`, it's not uncommon to use a lambda function to encapsulate them, which provides a perfect opportunity to include the filename output.

I will now detail three code examples demonstrating how to retain filepaths during dataset construction:

**Example 1: Simple Filename Preservation with `tf.data.Dataset.list_files` and `map`**

This example demonstrates the fundamental technique of passing the filename into the dataset elements. We are using list\_files to find image filepaths. Then, we are passing these paths to the `load_image` function, which performs `tf.io.read_file` and decoding and outputs a tuple with `image` and `filename`.

```python
import tensorflow as tf
import os

def load_image(file_path):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_jpeg(image, channels=3)
  return image, file_path

# Create dummy image files for the purpose of this example
os.makedirs("dummy_images", exist_ok=True)
for i in range(3):
  with open(f"dummy_images/image_{i}.jpg", 'w') as f:
    f.write("Fake Image Data") # Dummy content

file_paths = tf.data.Dataset.list_files("dummy_images/*.jpg")
image_dataset = file_paths.map(load_image)


#Example of iterating and extracting filename
for image, file_path in image_dataset:
    print(f"Loaded image from: {file_path.numpy().decode('utf-8')}")

#Cleanup dummy images
import shutil
shutil.rmtree("dummy_images")

```

*   **Explanation:** This code utilizes `tf.data.Dataset.list_files` to create a dataset of filepaths. The `map` transformation applies the `load_image` function, which reads and decodes the image and includes the original file path as the second component of the output tuple.
*   **Commentary:** The `load_image` function encapsulates the necessary file processing operations and returns both the processed image and its path, allowing for later access. The filename is outputted as a tensor, and therefore must be converted to a string representation using `.numpy()` and `.decode()` when printed. Notice that because of eager execution the first time `file_path` is referenced in the `for` loop, it is already a tensor object.

**Example 2: Handling Class Labels and Filenames**

This scenario includes retrieving both the file path and class label from folder names, a common scenario when working with image datasets. This makes the code a little more complex but highlights the flexibility of the approach.

```python
import tensorflow as tf
import os
import shutil

#create dummy image folders and files
os.makedirs("data/class_a", exist_ok=True)
os.makedirs("data/class_b", exist_ok=True)

for i in range(2):
    with open(f"data/class_a/image_{i}.jpg", "w") as f:
        f.write("Fake Data A")

for i in range(2):
    with open(f"data/class_b/image_{i}.jpg", "w") as f:
        f.write("Fake Data B")

def load_image_with_label(file_path):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_jpeg(image, channels=3)
  label = tf.strings.split(file_path, os.path.sep)[-2]
  return image, label, file_path


image_dataset_with_labels = tf.data.Dataset.list_files("data/*/*.jpg").map(load_image_with_label)


for image, label, file_path in image_dataset_with_labels:
    print(f"Image from: {file_path.numpy().decode('utf-8')}, class: {label.numpy().decode('utf-8')}")

shutil.rmtree("data")

```

*   **Explanation:** This code first generates folder structures simulating a class-labeled image set. The function `load_image_with_label` extracts class labels based on the parent directory name, demonstrating how to enrich data extraction from file paths during dataset construction.
*   **Commentary:** String manipulation in TensorFlow allows extraction of data from file paths, in this case the class label. This ensures that we can easily process both the actual image and additional metadata. Notice that the `split` command uses `os.path.sep` instead of simply `/` which improves portability across operating systems.

**Example 3: Using a Generator and `from_generator` with Filenames**

This final example showcases a scenario where you might be generating data programmatically or retrieving it from a database, while still wanting to preserve file paths (or equivalent identifiers). While this example will not work with images because the data generation is a generator and would need to be modified to read images, the underlying approach is valuable.

```python
import tensorflow as tf

def data_generator():
    for i in range(3):
        yield (f"Data record_{i}", f"record_{i}.txt")

dataset_from_generator = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.string))
)

for data, file_path in dataset_from_generator:
    print(f"Data: {data.numpy().decode('utf-8')}, File Path: {file_path.numpy().decode('utf-8')}")
```

*   **Explanation:** A generator function, `data_generator`, yields both data and its corresponding "filepath." The `from_generator` method creates a dataset from this generator. This is an abstraction away from using list\_files and allows for more customizable data sources.
*   **Commentary:** The explicit `output_signature` argument is essential. It defines the structure and data type of outputted dataset elements, specifically ensuring that the filepaths are passed as string tensors. If you omitted the output\_signature, TensorFlow would not know the format of the data generated.

In summary, accessing filenames directly from a `tf.data.Dataset` is not supported. You must strategically incorporate file paths during dataset creation, typically by mapping a load function that returns both the processed data and the file path (or equivalent identifier). It is important to understand the source of data. For datasets from a directory structure, the use of `tf.data.Dataset.list_files` is fundamental. When the data is generated, it's essential to ensure that filepaths are explicitly included within the data structures produced by your generator. Careful attention during dataset construction enables later extraction of this essential metadata, as demonstrated by the examples above.

For a deeper understanding of these topics, I suggest consulting the TensorFlow official documentation for the `tf.data` module, specifically around: `tf.data.Dataset.list_files`, `tf.data.Dataset.map`, `tf.io.read_file`, and `tf.data.Dataset.from_generator`. Also, a deep review of "TensorFlow Guide on Datasets" will help in conceptual understanding and problem solving. Further exploration into image processing techniques in TensorFlow with a focus on data loading practices can be helpful as well. Finally, examining community implementations on platforms like GitHub may provide valuable insight into diverse applications of these techniques.
