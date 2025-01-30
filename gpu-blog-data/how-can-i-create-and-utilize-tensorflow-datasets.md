---
title: "How can I create and utilize TensorFlow Datasets from a custom data generator?"
date: "2025-01-30"
id: "how-can-i-create-and-utilize-tensorflow-datasets"
---
The core challenge in leveraging custom data generators with TensorFlow Datasets (TFDS) lies in bridging the gap between the generator's iterative output and TFDS's expectation of a structured, potentially parallelizable data source.  My experience building large-scale image classification models underscored the importance of this efficiency. Simply feeding data directly from a generator to the `tf.data.Dataset` API is often insufficient for optimal performance, especially with complex data transformations or large datasets.  TFDS provides a framework for creating reusable, documented datasets that transcends simple generator usage.  The solution hinges on employing the `tfds.core.GeneratorBasedBuilder` class.


**1. Clear Explanation**

The `tfds.core.GeneratorBasedBuilder` acts as an intermediary, allowing you to define a dataset based on a custom data generation function.  This function, typically named `_generate_examples`, iterates through your data source and yields example dictionaries. These dictionaries must conform to a predetermined feature specification, detailing the data types and shapes of your input features and labels. The builder then manages the intricacies of sharding, caching, and data transformations within the TFDS ecosystem.  Crucially, this approach facilitates parallel processing during data preparation, a significant advantage over direct generator feeding. Moreover, it promotes reproducibility and facilitates easy dataset sharing, which was invaluable during collaborative model development projects I've undertaken.  This structured approach contrasts with the more ad-hoc method of directly feeding a generator into the `tf.data.Dataset` pipeline, which may lack the scalability and maintainability required for more significant projects.

The process involves several key steps:

* **Defining Features:**  Precisely specifying the data types and shapes of your features (e.g., images, text, numerical labels) using `tfds.features`.  This informs TFDS about the data structure, enabling efficient data handling.

* **Implementing `_generate_examples`:** Writing a generator function that iterates through your data, yielding example dictionaries conforming to the feature specifications.  This is where the logic for your data acquisition and preprocessing resides.

* **Building the Dataset:** Utilizing the `tfds.core.GeneratorBasedBuilder` to construct the dataset. This manages the process of converting your generator's output into a TFDS-compatible dataset, including sharding and caching for performance optimization.

* **Registering the Dataset:** Registering your custom dataset with TFDS, making it accessible via `tfds.load`.  This simplifies future usage and integration with other TFDS tools.


**2. Code Examples with Commentary**


**Example 1: Simple Numerical Dataset**

This example generates a dataset of 1000 pairs of random numbers.

```python
import tensorflow_datasets as tfds
import numpy as np

class RandomNumberDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='A dataset of random number pairs.',
            features=tfds.features.FeaturesDict({
                'x': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                'y': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
            }),
        )

    def _generate_examples(self):
        for i in range(1000):
            yield i, {'x': np.random.rand(1), 'y': np.random.rand(1)}

builder = RandomNumberDataset()
ds = builder.as_dataset(split='train')
for example in ds.take(5):
    print(example)
```

This showcases the minimal structure needed.  Note how `_generate_examples` yields a tuple, where the first element is a unique example ID, and the second is a dictionary matching the `_info`'s feature specification.


**Example 2:  Image Classification Dataset**

This expands on the prior example, incorporating image data (replace with your actual image loading logic).

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from PIL import Image

class ImageClassificationDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='A dataset of images and their labels.',
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'label': tfds.features.ClassLabel(names=['cat', 'dog']),
            }),
        )

    def _generate_examples(self):
        image_dir = 'path/to/your/images' #Replace with your directory
        for filename in os.listdir(image_dir):
            filepath = os.path.join(image_dir, filename)
            try:
                image = Image.open(filepath)
                label = 'cat' if 'cat' in filename else 'dog' #Rudimentary label extraction.  Adapt as needed.
                yield filename, {'image': image, 'label': label}
            except IOError as e:
                print(f"Error processing {filename}: {e}")


builder = ImageClassificationDataset()
ds = builder.as_dataset(split='train')
for example in ds.take(5):
    print(example)
```

This example demonstrates handling images.  Error handling is included for robustness. Replace placeholder comments with your actual image loading and labeling logic. The `ClassLabel` feature provides an important structure for classification tasks.


**Example 3:  Text Classification with Preprocessing**

This example handles text data, including preprocessing steps within the generator.

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt') #Ensure punkt is downloaded

class TextClassificationDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='A dataset of text and its labels.',
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text(),
                'label': tfds.features.ClassLabel(names=['positive', 'negative']),
            }),
        )

    def _generate_examples(self):
        text_data = [
            ("This is a positive sentence.", "positive"),
            ("This is a negative sentence.", "negative"),
            # ... more data
        ]
        for i, (text, label) in enumerate(text_data):
            tokens = word_tokenize(text.lower()) #Simple tokenization
            yield i, {'text': ' '.join(tokens), 'label': label}

builder = TextClassificationDataset()
ds = builder.as_dataset(split='train')
for example in ds.take(5):
    print(example)
```

This demonstrates text preprocessing (tokenization in this case) directly within the generator, a common practice to streamline data pipeline operations.


**3. Resource Recommendations**

The official TensorFlow Datasets documentation.  A thorough understanding of the `tfds.core.GeneratorBasedBuilder` class and its methods is crucial.  Explore the examples provided in the documentation for various data types and structures.  Consult advanced guides on the `tf.data` API for dataset optimization techniques, such as prefetching, caching, and parallel processing.  Familiarize yourself with common feature specifications provided by TFDS to ensure consistency and compatibility.  Finally, resources focusing on best practices in data preprocessing and feature engineering will significantly improve your data handling strategy.
