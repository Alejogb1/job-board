---
title: "Why is there a NotImplementedError in Keras data utility functions?"
date: "2025-01-30"
id: "why-is-there-a-notimplementederror-in-keras-data"
---
The occurrence of `NotImplementedError` within Keras data utility functions, particularly those interfacing with data loading and processing, typically indicates that an abstract method, defined in a base class, has not been overridden by a specific implementation. This situation arises from the design pattern that leverages abstract base classes (ABCs) to define contracts for data handling. These ABCs outline the required methods that concrete data utility classes must implement to properly function within the Keras framework. When a user attempts to utilize a concrete class that has not fulfilled all of the obligations set forth by the ABC, this error is raised.

I've encountered this firsthand while developing custom data generators for handling time-series data within a deep learning project. The `tf.keras.utils.Sequence` class, which serves as the foundation for custom data loaders in Keras, mandates the implementation of `__len__` and `__getitem__` methods. Failing to provide these results in a `NotImplementedError` upon instantiation or during use. This design allows Keras to abstract away the specifics of data fetching and batching, providing a consistent interface for training and evaluation irrespective of the data source.

The error essentially conveys a violation of the Liskov Substitution Principle. This principle dictates that subtypes must be substitutable for their base types without altering the correctness of the program. In the context of Keras data utility classes, concrete subclasses of data loaders must provide concrete implementations of the methods defined as abstract in the base classes. The principle ensures consistent behavior across data loading strategies, facilitating interoperability and simplifying the Keras framework's architecture. Without the enforcement of this pattern, Keras would be required to anticipate all possible data loading scenarios explicitly within its core code, making the framework brittle and difficult to maintain.

The `NotImplementedError` also acts as a signal to the developer that specific, concrete implementations need to be put in place. It serves as a fail-fast mechanism, prompting the user to provide the required methods immediately, rather than allowing the program to fail later in a less predictable manner. This proactive approach makes debugging easier and improves code robustness. It also implicitly documents the minimum requirements for any custom data loader to be successfully integrated into the Keras framework.

Let me illustrate with a few examples.

**Example 1: A Minimalist Sequence Implementation**

Consider a simplistic attempt to create a custom data generator using `tf.keras.utils.Sequence`, initially neglecting to implement the required methods:

```python
import tensorflow as tf
import numpy as np

class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

try:
  generator = MyDataGenerator(np.random.rand(100), 32)
  generator[0]
except NotImplementedError as e:
    print(f"Error caught: {e}")
```

This code instantiates `MyDataGenerator` with some dummy data. It attempts to access the first batch via index access (`generator[0]`). Because the `__getitem__` method is missing, a `NotImplementedError` is raised, informing the user of the missing implementation. The user then needs to provide the data loading logic for a single batch within this method. Similarly, if the method `__len__` was missing, that would raise a `NotImplementedError` at the point of use.

**Example 2: Providing the Missing Methods**

To rectify the error from the previous example, the `__len__` and `__getitem__` methods are necessary:

```python
import tensorflow as tf
import numpy as np

class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.data))
        batch_data = self.data[batch_start:batch_end]
        return batch_data, np.zeros(batch_data.shape) # Dummy labels

generator = MyDataGenerator(np.random.rand(100, 10), 32)
first_batch, _ = generator[0]
print(f"First batch shape: {first_batch.shape}")
print(f"Total number of batches: {len(generator)}")
```

In this corrected code snippet, `__len__` returns the total number of batches and `__getitem__` constructs and returns a single batch from the data. Accessing the data and number of batches now executes successfully without raising the `NotImplementedError`. Notice that this implementation loads a subset of the entire dataset into memory. In real applications, I usually load data lazily from files to handle larger datasets.

**Example 3: Custom Data Loader from Files**

Let's say you have files containing images for training:

```python
import tensorflow as tf
import os
import numpy as np

class ImageSequence(tf.keras.utils.Sequence):
  def __init__(self, image_paths, batch_size, target_size=(64, 64)):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.target_size = target_size

  def __len__(self):
      return int(np.ceil(len(self.image_paths) / self.batch_size))

  def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[batch_start:batch_end]
        batch_images = [self._load_and_preprocess(path) for path in batch_paths]
        return np.stack(batch_images), np.zeros((len(batch_images)))  # Dummy Labels

  def _load_and_preprocess(self, path):
      img = tf.io.read_file(path)
      img = tf.image.decode_image(img, channels=3)
      img = tf.image.resize(img, self.target_size)
      img = tf.cast(img, tf.float32) / 255.0 # Normalize
      return img.numpy()


# Create dummy files
if not os.path.exists("dummy_images"):
    os.makedirs("dummy_images")
for i in range(50):
  img = np.random.rand(100, 100, 3) * 255
  tf.keras.preprocessing.image.save_img(os.path.join("dummy_images", f"img_{i}.png"), img.astype(np.uint8))

image_paths = [os.path.join("dummy_images", f"img_{i}.png") for i in range(50)]

image_generator = ImageSequence(image_paths, 32)
first_batch_images, _ = image_generator[0]
print(f"First batch images shape {first_batch_images.shape}")
```

This example demonstrates a custom data loader reading image data from files and performs resizing and normalization. This is a more realistic use-case demonstrating that data is loaded per batch from the file system. The `_load_and_preprocess` method shows that there can be an internal method to implement data processing. If one of the `__len__` or `__getitem__` were removed, or unimplemented, the familiar `NotImplementedError` would be raised.

The `NotImplementedError` within Keras data utility functions, therefore, acts as a signal to the programmer, guiding them to provide the specific, concrete implementation demanded by the abstract base classes. This design enforces a consistent and modular approach to data loading, benefiting the robustness, maintainability, and extensibility of the framework. It also enforces the Liskov substitution principle.

For further exploration, consult the Keras documentation on data loading and the use of the `tf.keras.utils.Sequence` class. Additionally, research articles on abstract base classes and the Liskov Substitution Principle will provide a broader context for understanding this design pattern. Reading the TensorFlow source code can help understand implementation details. Finally, practical experience with custom data loaders, such as those developed in the examples above, will give a deep understanding on how `NotImplementedError` acts as a form of communication in the framework.
