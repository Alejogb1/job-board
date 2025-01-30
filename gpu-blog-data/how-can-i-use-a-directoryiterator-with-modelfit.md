---
title: "How can I use a DirectoryIterator with `model.fit()`?"
date: "2025-01-30"
id: "how-can-i-use-a-directoryiterator-with-modelfit"
---
It's commonly understood that `model.fit()` in TensorFlow (or Keras) expects either a NumPy array or a TensorFlow Dataset object as input for training data. Directly passing a `DirectoryIterator` obtained from `ImageDataGenerator.flow_from_directory()` will raise an error. This is because the iterator itself is not the data but an interface to access the data in batches. My experience working on image recognition projects has frequently required transforming data loading methods to comply with the requirements of `model.fit()`, and here I will outline methods that I've successfully used.

The core issue stems from the fact that `DirectoryIterator`, while generating batches of data, is not a data container compatible with TensorFlow's model training loop. Specifically, it produces a Python generator that yields tuples of (images, labels) on demand. `model.fit()` expects an object from which it can directly load data in a form acceptable to TensorFlow, either a dataset, or data that can be consumed directly as array format. The `DirectoryIterator` fulfills the latter requirement only when iterated through, or consumed by another library. Thus, the immediate approach is to integrate the iterator with a process that allows `model.fit()` to consume the output of the iterator indirectly.

One such method is to utilize the `tf.data.Dataset.from_generator` function. This function accepts a Python generator (like one provided by a `DirectoryIterator`) and generates a `tf.data.Dataset` object. This created dataset is then fully compatible with `model.fit()`. We specify the data types of the output within the `from_generator` call, ensuring that the tensor shapes are properly understood by TensorFlow. This is particularly important because `tf.data` benefits from the ability to cache, shuffle, and prefetch the data, improving training performance. This methodology essentially moves data production to TensorFlow's efficient framework.

The primary concern when directly iterating through the iterator outside of the framework using python-based processing is the lack of GPU acceleration for batch processing. TensorFlow's `Dataset` api allows the loading and pre-processing of data on the GPU when possible, hence a large performance boost.

Let's illustrate this with code. Assume we have a directory structure like this:
```
data/
   train/
      class_a/
         image1.jpg
         image2.jpg
         ...
      class_b/
         image1.jpg
         image2.jpg
         ...
   validation/
      class_a/
         image1.jpg
         image2.jpg
         ...
      class_b/
         image1.jpg
         image2.jpg
         ...
```

**Code Example 1: Converting DirectoryIterator to Dataset**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data parameters
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/validation'
NUM_CLASSES = 2 # Assuming binary classification

# Create an ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create DirectoryIterators
train_iterator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Or 'binary', 'sparse' etc
)
validation_iterator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# Convert the iterator to a tf.data.Dataset
def generator_wrapper(iterator):
    while True:
        try:
           yield next(iterator)
        except StopIteration:
           break

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(train_iterator),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        tf.TensorShape((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.TensorShape((BATCH_SIZE, NUM_CLASSES)) #Change this if not binary/categorical
    )
)

validation_dataset = tf.data.Dataset.from_generator(
   lambda: generator_wrapper(validation_iterator),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        tf.TensorShape((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.TensorShape((BATCH_SIZE, NUM_CLASSES))
    )
)


# Create a basic model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Or 'sigmoid' for binary
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Modify as needed

# Train the model
steps_per_epoch = train_iterator.samples // BATCH_SIZE
validation_steps = validation_iterator.samples // BATCH_SIZE
model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_dataset,
    validation_steps=validation_steps
)
```

In this code, `generator_wrapper` manages the iteration process and provides the next batch from the iterator.  `tf.data.Dataset.from_generator` then converts this into a dataset object with the correct output shapes and types. The shapes are defined as `tf.TensorShape` which takes care of the dimension specification of each tensor.

**Code Example 2: Using the `steps_per_epoch` argument of `model.fit()` directly with the iterator.**

Alternatively, while not as performant as converting to `tf.data.Dataset`, one can directly utilize the `DirectoryIterator` by explicitly handling the number of steps.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Parameters from before
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/validation'
NUM_CLASSES = 2

# Generators from the previous example
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_iterator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
validation_iterator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# A sample model from previous
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate steps
steps_per_epoch = train_iterator.samples // BATCH_SIZE
validation_steps = validation_iterator.samples // BATCH_SIZE

# Pass the iterator and steps to the fit method
model.fit(
    train_iterator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_iterator,
    validation_steps=validation_steps,
)

```

Here, the `DirectoryIterator` is directly used by `model.fit()`, but `steps_per_epoch` and `validation_steps` arguments are specified so the model knows how many iterations compose a full training epoch. This approach works but does not benefit from TensorFlow's dataset pipeline and thus does not have the performance optimizations possible with the `tf.data.Dataset` approach.

**Code Example 3: Implementing a custom generator function for compatibility with Keras.**

In some more complex scenarios, custom data processing may be required. In these cases, a custom generator can be created and consumed by `model.fit()` using `tf.data.Dataset.from_generator`.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

# Parameters from before
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/validation'
NUM_CLASSES = 2

# Custom function to get all file names
def list_all_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
      for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to implement custom pre-processing
def custom_image_processor(image_path, label):
    image = Image.open(image_path).resize((IMG_WIDTH,IMG_HEIGHT))
    image = np.array(image)
    image = image/255.0  # Normalize from 0-255 to 0-1 range
    return image.astype('float32'), label.astype('float32')

def custom_data_generator(image_paths, class_map, batch_size):
    i = 0
    while True:
        batch_paths = image_paths[i*batch_size:(i+1)*batch_size]
        batch_images = []
        batch_labels = []
        for path in batch_paths:
            label_str = path.split(os.sep)[-2] # Split on OS specific path separator to extract folder name
            label_vec = class_map[label_str]
            image, label = custom_image_processor(path,label_vec)
            batch_images.append(image)
            batch_labels.append(label)

        yield (np.array(batch_images), np.array(batch_labels))

        i = (i + 1) % (len(image_paths) // batch_size) # loop through the data and not exceed

train_paths = list_all_image_paths(TRAIN_DIR)
valid_paths = list_all_image_paths(VALID_DIR)

# Create a class map for indexing folders
unique_folders_train = set(path.split(os.sep)[-2] for path in train_paths)
unique_folders_valid = set(path.split(os.sep)[-2] for path in valid_paths)
classes = list(sorted(unique_folders_train.union(unique_folders_valid)))
class_map = {classes[i]:tf.one_hot(i, len(classes)) for i in range(len(classes))}


# Get the training dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: custom_data_generator(train_paths, class_map, BATCH_SIZE),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        tf.TensorShape((None, IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.TensorShape((None, NUM_CLASSES))
    )
)
# Get the validation dataset
validation_dataset = tf.data.Dataset.from_generator(
  lambda: custom_data_generator(valid_paths, class_map, BATCH_SIZE),
    output_types=(tf.float32, tf.float32),
     output_shapes=(
         tf.TensorShape((None, IMG_HEIGHT, IMG_WIDTH, 3)),
         tf.TensorShape((None, NUM_CLASSES))
     )
 )


# Model from before
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(valid_paths) // BATCH_SIZE


model.fit(
    train_dataset,
    steps_per_epoch = steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    epochs=10
)
```

In this approach we construct a custom generator that will load and transform images. We also created our own class mapping and loading functionality. This showcases that you are free to implement an arbitrary set of processing and loading functionality which will then be converted to a proper dataset.

For further reference, I recommend exploring the following resources: the official TensorFlow documentation on `tf.data`, particularly the sections on dataset creation and input pipelines. Also, the Keras documentation on using generators with `model.fit()` can provide further insights. Additionally, examples and tutorials on implementing custom data loading can be found in books on deep learning with TensorFlow/Keras, as well as on websites which specialize on ML tutorials. Specifically, focusing on sections on performance optimizations in data processing will greatly improve training speeds. Careful study of these will greatly improve the understanding of how to utilize data processing and loading in TensorFlow effectively.
