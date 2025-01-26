---
title: "How can Keras image augmentation be implemented using a custom generator?"
date: "2025-01-26"
id: "how-can-keras-image-augmentation-be-implemented-using-a-custom-generator"
---

Deep learning models, especially those processing image data, often suffer from overfitting due to limited training data. Data augmentation serves as a crucial technique to artificially expand the training set, improving generalization. I’ve found that while Keras’ built-in `ImageDataGenerator` is convenient for standard augmentations, custom generators offer flexibility for complex pipelines or when data needs specific preprocessing beyond basic transformations. This response will detail how to implement Keras image augmentation using a custom Python generator.

A custom generator in Keras is essentially a Python function designed to yield batches of image data and corresponding labels (or other target variables). It overcomes the limitations of pre-packaged solutions by enabling control over each step of the data loading and augmentation pipeline. This approach becomes invaluable when working with diverse image formats, non-standard augmentations, or pre-processing steps not readily available in `ImageDataGenerator`. The key lies in crafting a generator function that efficiently loads data, applies chosen augmentations, and yields batches suitable for model training.

The core idea is to create a function that, within a loop, selects a subset of training data, performs augmentations, and produces a batch. Keras leverages Python's generator behavior. Instead of loading all data into memory, the `yield` keyword returns batches one at a time. This avoids memory issues when dealing with large image datasets.

Here are three different code examples showing how to construct custom generators for image augmentation using various approaches:

**Example 1: Basic Random Augmentations with NumPy**

This example focuses on using NumPy and libraries like `scipy.ndimage` for basic random transformations. I've utilized this technique in projects where lightweight augmentations were sufficient.

```python
import numpy as np
import os
from PIL import Image
from scipy.ndimage import rotate, zoom, shift
import tensorflow as tf

def basic_image_generator(image_dir, batch_size, target_size=(256, 256), augment=True):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)
    while True:
        batch_images = []
        batch_labels = []
        indices = np.random.choice(num_images, size=batch_size, replace=False)
        for i in indices:
            img_path = os.path.join(image_dir, image_files[i])
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img = np.asarray(img, dtype=np.float32) / 255.0  # Normalize

            if augment:
                if np.random.rand() > 0.5:
                  angle = np.random.uniform(-20, 20)
                  img = rotate(img, angle, reshape=False, mode='nearest')
                if np.random.rand() > 0.5:
                   scale = np.random.uniform(0.8, 1.2)
                   img = zoom(img, scale, mode='nearest')
                if np.random.rand() > 0.5:
                   shift_x = np.random.uniform(-0.1, 0.1) * target_size[0]
                   shift_y = np.random.uniform(-0.1, 0.1) * target_size[1]
                   img = shift(img, [shift_y, shift_x, 0], mode='nearest')

            batch_images.append(img)
            # Assume image filename is the label e.g., "cat_1.jpg" label = "cat"
            batch_labels.append(image_files[i].split('_')[0])
        batch_labels = np.asarray(batch_labels)
        
        #Label encoding, assuming you have predefined labels
        label_encoder = tf.keras.layers.StringLookup(mask_token=None)
        label_encoder.adapt(batch_labels)
        encoded_labels = label_encoder(batch_labels)

        batch_images = np.stack(batch_images)
        yield batch_images, encoded_labels
```
This generator first loads and normalizes images. Then, with a probabilistic approach, it applies random rotations, zooming, and shifts if the augment flag is enabled. I’ve found that using a random number generation for each augmentation step introduces needed randomness, further preventing overfitting. One crucial step included in this implementation is encoding of the image labels, using Tensorflow’s `StringLookup` layer.

**Example 2: Augmentations with TensorFlow’s `tf.image` Module**

This example showcases using the TensorFlow `tf.image` module. I often utilize this in combination with other Tensorflow functionalities to ensure GPU acceleration of augmentations. This also eliminates third party library dependencies and maintains compatibility within the Tensorflow ecosystem.

```python
import tensorflow as tf
import os
import numpy as np

def tf_image_generator(image_dir, batch_size, target_size=(256, 256), augment=True):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)
    while True:
        batch_images = []
        batch_labels = []
        indices = np.random.choice(num_images, size=batch_size, replace=False)
        for i in indices:
            img_path = os.path.join(image_dir, image_files[i])
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)  # Or decode_png
            img = tf.image.resize(img, target_size)
            img = tf.cast(img, tf.float32) / 255.0

            if augment:
              if tf.random.uniform(shape=()) > 0.5:
                img = tf.image.random_brightness(img, max_delta=0.3)
              if tf.random.uniform(shape=()) > 0.5:
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
              if tf.random.uniform(shape=()) > 0.5:
                  img = tf.image.random_flip_left_right(img)
            
            batch_images.append(img)
            # Assume image filename is the label e.g., "cat_1.jpg" label = "cat"
            batch_labels.append(image_files[i].split('_')[0])
        
        batch_labels = np.asarray(batch_labels)
        
        #Label encoding, assuming you have predefined labels
        label_encoder = tf.keras.layers.StringLookup(mask_token=None)
        label_encoder.adapt(batch_labels)
        encoded_labels = label_encoder(batch_labels)
        
        batch_images = tf.stack(batch_images)
        yield batch_images, encoded_labels
```

This example leverages functions within `tf.image` for augmentations like random brightness adjustments, contrast changes, and horizontal flipping. I prefer this approach when working within a pure Tensorflow workflow due to the optimized operations which can run on compatible GPU devices. Furthermore, it facilitates more complex augmentations that can be directly composed using Tensorflow's powerful API. Again, label encoding with `StringLookup` is included.

**Example 3: Custom Augmentations with a Class for State Management**

This example shows how to create a class-based custom generator for state management. This often proves advantageous when the augmentations require a specific state or need to be computed with parameters, as the generator class can maintain the state during iterations.

```python
import tensorflow as tf
import os
import numpy as np
import random

class CustomImageGenerator(tf.keras.utils.Sequence):
  def __init__(self, image_dir, batch_size, target_size=(256, 256), augment=True):
      self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
      self.num_images = len(self.image_files)
      self.batch_size = batch_size
      self.target_size = target_size
      self.augment = augment
      self.label_encoder = tf.keras.layers.StringLookup(mask_token=None)
      self.labels = [f.split('_')[0] for f in self.image_files]
      self.label_encoder.adapt(self.labels)

  def __len__(self):
        return int(np.ceil(self.num_images / self.batch_size))

  def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_images)
        batch_files = self.image_files[start:end]
        batch_images = []
        batch_labels = []
        for file_name in batch_files:
            img_path = os.path.join(image_dir, file_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)  # Or decode_png
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.float32) / 255.0

            if self.augment:
                if tf.random.uniform(shape=()) > 0.5:
                  angle = tf.random.uniform(shape=(), minval=-30, maxval=30)
                  img = tf.keras.layers.RandomRotation(factor=angle)(img[tf.newaxis,...])[0]
                if tf.random.uniform(shape=()) > 0.5:
                   scale = tf.random.uniform(shape=(), minval=0.8, maxval=1.2)
                   img = tf.image.central_crop(img, central_fraction=scale)
                   img = tf.image.resize(img, self.target_size)
                if tf.random.uniform(shape=()) > 0.5:
                  img = tf.image.random_flip_left_right(img)

            batch_images.append(img)
            batch_labels.append(file_name.split('_')[0])
            
        encoded_labels = self.label_encoder(batch_labels)

        return tf.stack(batch_images), encoded_labels
```

Here, the generator is defined as a class that inherits `tf.keras.utils.Sequence`, allowing for more control over the generation process.  State, such as the `label_encoder`, is stored within the class.  The `__len__` method returns the number of batches and `__getitem__` is responsible for generating each batch, using the start and end indices for slicing.  This approach can simplify more complex custom augmentation pipelines that require keeping track of parameters or state across iterations and allows for use of `model.fit` as a training workflow.

These examples demonstrate various approaches to creating custom image data generators.  The approach selected often depends on the specific requirements of the project.  Key aspects common to all three include loading images, normalizing them, applying augmentations using random selections and using the `yield` keyword or sequence to return batches.

For further reading, I would recommend the official Tensorflow documentation for `tf.image` module and their section on using Keras with custom generators.  Additionally, exploring NumPy’s array manipulation capabilities will enable complex transformation using a Numpy array.  Also, the Keras documentation offers insights into using data sequences. Lastly, understanding Python's generator behavior is crucial for optimal implementation.  These resources will provide a strong foundation for developing robust and effective custom image augmentation pipelines.
