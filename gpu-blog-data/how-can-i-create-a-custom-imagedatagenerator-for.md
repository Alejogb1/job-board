---
title: "How can I create a custom ImageDataGenerator for large, multi-input datasets in Keras?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-imagedatagenerator-for"
---
Efficiently handling large, multi-input datasets within Keras's `ImageDataGenerator` requires a nuanced understanding of its underlying mechanisms and limitations.  My experience developing image classification models for satellite imagery, often exceeding terabyte-scale datasets, highlighted the critical need for custom generators optimized for memory management and data throughput.  Standard `ImageDataGenerator` instances, while convenient for smaller datasets, frequently become bottlenecks with larger volumes, triggering excessive memory consumption and slowing training significantly.  The solution lies in creating a custom generator that leverages efficient data loading strategies and minimizes in-memory data storage.


**1.  Clear Explanation**

The key to building a performant custom `ImageDataGenerator` for multi-input datasets lies in iterative data loading. Instead of loading the entire dataset into memory at once—a recipe for disaster with large datasets—we employ a generator that yields batches of data on demand.  This allows processing of significantly larger datasets than would otherwise be possible. For multi-input scenarios, where each data point comprises multiple images or features, we need to manage the loading and augmentation of each input stream concurrently.

The process involves three core steps:

* **Data Source Definition:**  Clearly define the location and structure of your multi-input dataset. This often involves specifying paths to directories containing images or other data files for each input branch.  A structured file system is crucial for efficient access.

* **Batch Generation:**  Create a generator function that yields batches of data.  Each batch should contain a tuple of NumPy arrays, one for each input branch. The generator must handle data augmentation and preprocessing steps individually for each input before yielding the batch.

* **Integration with Keras:**  Integrate the custom generator into the Keras `fit` or `fit_generator` (if using TensorFlow 2.x, `model.fit` with a generator is preferred) method by specifying the generator as the `x` argument (or as the generator argument) and specifying the `steps_per_epoch`.  `steps_per_epoch` is calculated by dividing the total number of samples by the batch size.

**2. Code Examples with Commentary**

**Example 1: Simple Two-Input Image Generator**

This example demonstrates a basic generator for two image inputs.  It assumes images are stored in subdirectories `input_a` and `input_b` within a base directory.  Augmentation is applied independently to each input.

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

class MultiInputGenerator(Sequence):
    def __init__(self, base_dir, batch_size, img_size=(224, 224)):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.filenames_a = sorted([f for f in os.listdir(os.path.join(base_dir,'input_a')) if f.endswith('.jpg')])
        self.filenames_b = sorted([f for f in os.listdir(os.path.join(base_dir,'input_b')) if f.endswith('.jpg')])
        self.n_samples = len(self.filenames_a) # Assume equal number of files in both directories

    def __len__(self):
        return int(np.ceil(self.n_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_a = []
        batch_x_b = []
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.n_samples)):
            img_a = load_img(os.path.join(self.base_dir, 'input_a', self.filenames_a[i]), target_size=self.img_size)
            img_a = img_to_array(img_a) / 255.0 # Normalize
            # Apply augmentations to img_a here (e.g., using Keras's ImageDataGenerator)

            img_b = load_img(os.path.join(self.base_dir, 'input_b', self.filenames_b[i]), target_size=self.img_size)
            img_b = img_to_array(img_b) / 255.0 # Normalize
            # Apply augmentations to img_b here

            batch_x_a.append(img_a)
            batch_x_b.append(img_b)

        return [np.array(batch_x_a), np.array(batch_x_b)], np.zeros((self.batch_size,1)) #Placeholder for Y

    def on_epoch_end(self):
        pass # Shuffle data here if needed

import os
# Example usage:
train_generator = MultiInputGenerator('./my_dataset', 32)
# model.fit(train_generator, steps_per_epoch=len(train_generator), ...)
```

**Example 2: Incorporating Preprocessing Functions**

This refines Example 1 by introducing custom preprocessing functions for each input type, enhancing flexibility and modularity.

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# ... (rest of imports)

def preprocess_input_a(img):
    # Custom preprocessing for input A
    img = ... # Apply your specific preprocessing steps
    return img

def preprocess_input_b(img):
    # Custom preprocessing for input B
    img = ... # Apply your specific preprocessing steps
    return img

# ... (MultiInputGenerator class from Example 1, modified as follows)

    def __getitem__(self, idx):
        # ... (rest of the function)
            img_a = preprocess_input_a(img_a)
            img_b = preprocess_input_b(img_b)
        # ... (rest of the function)
```


**Example 3: Handling Diverse Data Types**

This expands the capability to handle more than just images; for example, combining images with tabular data.

```python
import pandas as pd
# ... (other imports)

class MultiModalGenerator(Sequence):
    def __init__(self, image_dir, csv_path, batch_size, img_size=(224, 224)):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_samples = len(self.data)

    def __len__(self):
        return int(np.ceil(self.n_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_images = []
        batch_x_tabular = []
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.n_samples)):
            img_path = os.path.join(self.image_dir, self.data.iloc[i]['image_filename'])
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0
            batch_x_images.append(img)
            batch_x_tabular.append(self.data.iloc[i][['feature1', 'feature2', 'feature3']].values) # Extract relevant features

        return [np.array(batch_x_images), np.array(batch_x_tabular)], self.data.iloc[idx * self.batch_size:min((idx + 1) * self.batch_size, self.n_samples)]['target'].values


# Example Usage (assuming 'target' column exists in CSV)
train_generator = MultiModalGenerator('./image_data', './data.csv', 32)
# model.fit(train_generator, steps_per_epoch=len(train_generator), ...)

```

**3. Resource Recommendations**

For deeper understanding of data generators, consult the official Keras documentation.  Study the source code of Keras's `ImageDataGenerator` to understand its internal workings.  A strong foundation in Python's generator functions and NumPy array manipulation is also essential.  Finally, becoming proficient in efficient file I/O operations in Python will significantly benefit the development of custom data generators.
