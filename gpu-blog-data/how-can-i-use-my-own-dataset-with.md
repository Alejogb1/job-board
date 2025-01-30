---
title: "How can I use my own dataset with tfds.load in Google Colab?"
date: "2025-01-30"
id: "how-can-i-use-my-own-dataset-with"
---
TensorFlow Datasets (TFDS) offers a streamlined interface for accessing a wide variety of curated datasets.  However, the utility extends beyond its pre-built offerings; integrating custom datasets is a crucial aspect of leveraging TFDS effectively.  My experience working on large-scale image classification projects, specifically the development of a proprietary dataset for identifying rare plant species, highlighted the importance of this capability.  Directly loading a custom dataset into TFDS requires understanding its underlying mechanisms and implementing a suitable `tfds.core.DatasetBuilder`.


**1. Clear Explanation:**

TFDS relies on a structured approach to dataset representation. Each dataset is defined by a `DatasetBuilder` class, which handles the intricacies of data loading, processing, and splitting.  This builder is responsible for downloading, extracting, and parsing the data according to the specifics of your dataset’s format.  To use your own dataset, you must create a custom `DatasetBuilder` inheriting from `tfds.core.DatasetBuilder`. This custom class will override methods defining the dataset's characteristics, such as data location, data format, and feature specifications.

The core methods you'll need to implement are `_info()`, which describes the dataset’s structure and features, and `_generate_examples()`, which yields the actual data instances.  `_info()` returns a `tfds.core.DatasetInfo` object, specifying aspects like features (e.g., image, label), splits (e.g., train, validation, test), and metadata. `_generate_examples()` is a generator function that iterates through your data files and yields dictionaries, where keys represent the feature names and values represent the corresponding feature data.  This generator is crucial for loading data in a TensorFlow-compatible format.

Error handling is paramount. Robust error handling within `_generate_examples()` prevents data loading failures from halting the entire process.  This often involves gracefully skipping corrupted files or handling inconsistencies in data formats.  Thorough documentation within the custom builder class is equally important, ensuring clarity and maintainability, especially in collaborative projects.


**2. Code Examples with Commentary:**

**Example 1:  Simple CSV Dataset**

This example demonstrates loading a dataset from a CSV file containing image filenames and corresponding labels.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

class MyCsvDataset(tfds.core.DatasetBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='My custom CSV dataset',
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'label': tfds.features.ClassLabel(names=['class_a', 'class_b']),
            }),
            supervised_keys=('image', 'label'),
        )

    def _generate_examples(self, data_path):
        df = pd.read_csv(data_path)
        for index, row in df.iterrows():
            try:
                image = tf.io.read_file(row['image_path'])
                image = tf.image.decode_jpeg(image, channels=3) # Assuming JPEG images
                yield index, {'image': image, 'label': row['label']}
            except Exception as e:
                print(f"Error processing row {index}: {e}")

builder = MyCsvDataset('my_csv_dataset', data_dir='./data')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')
```

This code defines a `MyCsvDataset` builder.  The `_info()` method specifies the dataset features: an image and a label.  `_generate_examples()` reads the CSV, decodes JPEG images, and yields example dictionaries. Error handling is implemented to prevent crashes on corrupted data.


**Example 2:  Directory of Images with Subfolders as Labels**

This example showcases loading images organized into subfolders, where each subfolder represents a different class.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import os

class MyImageDataset(tfds.core.DatasetBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def _info(self):
      return tfds.core.DatasetInfo(
          builder=self,
          description='Images organized by subfolders',
          features=tfds.features.FeaturesDict({
              'image': tfds.features.Image(),
              'label': tfds.features.ClassLabel(num_classes=len(self.label_names)),
          }),
          supervised_keys=('image', 'label'),
      )

    def _generate_examples(self, data_path):
        self.label_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        for label_index, label in enumerate(self.label_names):
            label_path = os.path.join(data_path, label)
            for image_filename in os.listdir(label_path):
                image_path = os.path.join(label_path, image_filename)
                try:
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_jpeg(image, channels=3)
                    yield image_path, {'image': image, 'label': label_index}
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

builder = MyImageDataset('my_image_dataset', data_dir='./data')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')
```

This example leverages the directory structure for label encoding. It dynamically determines the number of classes and iterates through images, assigning labels based on their parent directory.  Error handling is again integrated to manage potential issues during image loading.


**Example 3:  Handling Multiple File Types and Data Augmentation**

This illustrates a more complex scenario involving multiple image file types and basic data augmentation.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import os

class MyMultiFormatDataset(tfds.core.DatasetBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def _info(self):
        return tfds.core.DatasetInfo(...) # Define features similar to previous examples, add metadata if needed.

    def _generate_examples(self, data_path):
        for root, _, files in tf.io.gfile.walk(data_path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    img = tf.io.read_file(filepath)
                    if filepath.lower().endswith('.jpg'):
                        img = tf.image.decode_jpeg(img, channels=3)
                    elif filepath.lower().endswith('.png'):
                        img = tf.image.decode_png(img, channels=3)
                    else:
                        continue # Skip unsupported formats

                    # Example data augmentation: random cropping and flipping
                    img = tf.image.random_crop(img, size=[224, 224, 3])
                    img = tf.image.random_flip_left_right(img)
                    yield filepath, {'image': img, 'label': self._get_label_from_path(filepath)} #requires _get_label_from_path
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    def _get_label_from_path(self, filepath):
        # Implement logic to extract label from filepath (e.g., using parent directory name)
        pass

builder = MyMultiFormatDataset('my_multi_format_dataset', data_dir='./data')
builder.download_and_prepare()
ds = builder.as_dataset(split='train')
```


This example handles JPEG and PNG images, implementing rudimentary data augmentation for improved model robustness. The `_get_label_from_path` function (not fully implemented here for brevity) would require custom logic depending on your dataset's labeling scheme within the file paths.


**3. Resource Recommendations:**

The official TensorFlow Datasets documentation provides comprehensive guidance on building custom datasets.  The TensorFlow documentation itself offers valuable information on data input pipelines and image preprocessing.  Finally, review materials on Python generators and exception handling for robust code development.  Understanding file I/O operations in Python, particularly within the context of large datasets, is also crucial.
