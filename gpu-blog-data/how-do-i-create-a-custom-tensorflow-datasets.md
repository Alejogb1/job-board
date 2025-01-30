---
title: "How do I create a custom TensorFlow Datasets (TFDS) dataset?"
date: "2025-01-30"
id: "how-do-i-create-a-custom-tensorflow-datasets"
---
Creating custom TensorFlow Datasets (TFDS) offers significant advantages for reproducibility and streamlined data management within your machine learning workflows.  My experience building and deploying large-scale image recognition models highlighted the crucial role of a well-structured TFDS; the process, while initially seeming complex, results in significantly improved data pipeline efficiency.  A key aspect often overlooked is the careful consideration of the dataset's metadata, which directly influences the ease of use and scalability down the line.

**1.  Explanation: The TFDS Builder Class**

The cornerstone of custom TFDS creation is the `tfds.core.DatasetBuilder` class.  This class acts as a blueprint, defining the methods for generating, downloading, and processing your dataset.  It requires careful planning and implementation to ensure data integrity and compatibility with the TFDS ecosystem.  In essence, you are instructing TensorFlow how to access, format, and serve your data. This is achieved by inheriting from the `DatasetBuilder` class and overriding several key methods.  These include:

* **`__init__`:** This method initializes the builder, accepting parameters such as data directories and dataset configuration. I've found it crucial to include robust error handling here to prevent issues during dataset instantiation. This includes checks for the existence of necessary files and validation of user-provided parameters.

* **`info`:**  This method returns a `tfds.core.DatasetInfo` object containing metadata about your dataset. This metadata is critical; it informs TensorFlow about features (like image dimensions, labels, and data types), splits (like train, validation, and test sets), and other crucial information for effective data loading and utilization.  The accuracy of this information is paramount for downstream model compatibility. Neglecting it can lead to unexpected errors during data preprocessing.

* **`_info`:** This is a private helper method providing the essential information for the `info` method. I’ve often found this method useful for organizing the creation of dataset metadata, separating it from the init method for better readability and maintainability.

* **`_generate_examples`:** This is the core method.  It iterates over your data source and yields examples in a dictionary format.  Each key in the dictionary corresponds to a feature in your dataset, and the values are the corresponding data.  This method is where your custom data loading and pre-processing logic resides. Ensuring consistent data formatting at this stage prevents downstream compatibility issues.   I've learned through experience the importance of efficient implementation here, using generators to avoid memory issues when dealing with large datasets.

* **`download_and_prepare`:** This orchestrates the download and preparation of the data.  It calls `_generate_examples`, which does the actual work of building the dataset from your source.  While generally handled internally, having a clear understanding of this process allows for optimization during dataset generation.


**2. Code Examples**

**Example 1: A Simple Text Dataset**

This example demonstrates creating a dataset from a text file, where each line represents a single example with a single feature: "text".

```python
import tensorflow_datasets as tfds
import tensorflow as tf

class TextDataset(tfds.core.DatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  def __init__(self, **kwargs):
    super().__init__(
        data_dir=kwargs.pop('data_dir', None),
        **kwargs,
    )

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'text': tfds.features.Text(),
        }),
        supervised_keys=('text', 'text'), #Example supervised key for simplicity
    )

  def _generate_examples(self, data_path):
    with tf.io.gfile.GFile(data_path, 'r') as f:
      for line_num, line in enumerate(f):
        yield line_num, {'text': line.strip()}

  def download_and_prepare(self):
    #In a real scenario, this would download data
    #Here we assume the data is in the data_dir
    data_path = self.data_dir / 'my_text_data.txt'
    self._generate_examples(data_path)


# Create a builder instance with a data directory
builder = TextDataset(data_dir='./data')

# Download and prepare the dataset
builder.download_and_prepare()

# Load the dataset
ds = builder.as_dataset(split='train')

# Access the data
for example in ds:
  print(example['text'].numpy().decode('utf-8'))
```

**Example 2: An Image Classification Dataset**

This showcases a more complex scenario, including image loading and label assignment.

```python
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image

class ImageClassificationDataset(tfds.core.DatasetBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'label': tfds.features.ClassLabel(names=['cat', 'dog']),
            }),
            supervised_keys=('image', 'label'),
        )

    def _generate_examples(self, data_dir):
        for label_id, label_dir in enumerate(data_dir.glob('*')):
          for image_path in label_dir.glob('*.jpg'):
              image = Image.open(image_path)
              yield str(image_path), {'image': image, 'label': label_id}

#Assuming data is structured as ./data/cat/*.jpg and ./data/dog/*.jpg
builder = ImageClassificationDataset(data_dir='./data')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')

for example in ds:
    print(example['label'].numpy(), example['image'].shape)

```


**Example 3:  A Dataset with Multiple Features**

This example illustrates a dataset with multiple features, demonstrating flexibility in handling diverse data types.

```python
import tensorflow_datasets as tfds
import tensorflow as tf

class MultiFeatureDataset(tfds.core.DatasetBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text(),
                'image': tfds.features.Image(),
                'numerical_feature': tfds.features.Tensor(shape=(10,), dtype=tf.float32),
            }),
            supervised_keys=('text', 'numerical_feature'), #Example, choose appropriately
        )

    def _generate_examples(self, data_path):
      #Simulate data loading with multiple features
      for i in range(10):
        yield i, {
            'text': f'Example text {i}',
            'image': tf.zeros((28, 28, 3), dtype=tf.uint8),
            'numerical_feature': tf.random.normal((10,))
        }

builder = MultiFeatureDataset(data_dir='./data') #data_dir not strictly needed here
builder.download_and_prepare()
ds = builder.as_dataset(split='train')

for example in ds:
    print(example)
```


**3. Resource Recommendations**

The official TensorFlow Datasets documentation.  The TensorFlow core documentation covering data input pipelines.  A comprehensive guide to Python's `itertools` module for efficient data iteration.  Understanding generators and their memory-efficient properties is essential for handling large datasets.  Finally, familiarizing yourself with different data serialization formats, such as TFRecord, will enhance dataset efficiency and scalability.  These resources, studied carefully, provide the foundational knowledge necessary for building and managing complex custom TensorFlow Datasets.
