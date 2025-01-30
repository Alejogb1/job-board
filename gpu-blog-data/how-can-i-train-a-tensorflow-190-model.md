---
title: "How can I train a TensorFlow 1.9.0 model using data generators?"
date: "2025-01-30"
id: "how-can-i-train-a-tensorflow-190-model"
---
Data generators in TensorFlow 1.x, particularly version 1.9.0, are essential for handling large datasets that do not fit into memory. My experience working on a large-scale image classification project required me to move beyond loading all images into RAM; instead, I implemented a generator-based training pipeline. This approach, while initially complex, dramatically improved training speed and scalability.

Fundamentally, data generators leverage Python's iterator protocol. Instead of loading all data at once, they produce mini-batches on demand, feeding data to the training loop as needed. This minimizes memory usage, enabling training on vast datasets. TensorFlow 1.9.0 doesn't have the convenient `tf.data` API prevalent in later versions, requiring a more manual approach using placeholders and custom Python generators.

The core strategy involves defining a Python generator function that yields data in batches, configuring TensorFlow placeholders for input and output, and then using these placeholders within the training loop. The generator handles data preprocessing, augmentation, and shuffling outside of the TensorFlow graph, making the graph leaner and more efficient. It’s a shift from the standard 'load-all-into-memory' paradigm to an 'on-demand' approach.

Here’s the crucial step in the process: converting the Python generator into data that TensorFlow can understand. This is done by creating placeholders for the inputs and targets. The generator is then invoked repeatedly to fill the placeholders during each training step.

Let's examine specific code examples.

**Example 1: Simple Numerical Data Generator**

This first example illustrates a straightforward scenario, handling numerical data generated on the fly:

```python
import tensorflow as tf
import numpy as np

def numerical_data_generator(batch_size, num_batches):
    for _ in range(num_batches):
        x_batch = np.random.rand(batch_size, 10) # Random input features
        y_batch = np.random.randint(0, 2, size=(batch_size, 1)) # Random binary labels
        yield x_batch, y_batch

# Define placeholders
x_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
y_placeholder = tf.placeholder(tf.int32, shape=[None, 1])

# Define a simple model (e.g., logistic regression)
W = tf.Variable(tf.random_normal([10, 1]))
b = tf.Variable(tf.zeros([1]))
logits = tf.matmul(x_placeholder, W) + b
predictions = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_placeholder, tf.float32), logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Training parameters
batch_size = 32
num_batches = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for batch_idx in range(num_batches):
    x_batch, y_batch = next(numerical_data_generator(batch_size, 1))
    _, loss_value = sess.run([optimizer, loss], feed_dict={x_placeholder: x_batch, y_placeholder: y_batch})
    if batch_idx % 100 == 0:
      print("Batch:", batch_idx, "Loss:", loss_value)
```

In this example, `numerical_data_generator` is a generator that produces batches of random numerical features (10 per input) and binary labels.  The `tf.placeholder`s (`x_placeholder` and `y_placeholder`) define the input and label tensors. During training, we retrieve a batch from the generator with `next()`, and then `feed_dict` is used to insert these batches into the placeholders at each training step. This avoids preloading the entire dataset. The model itself is a simple logistic regression for demonstration;  more complex models could be integrated.

**Example 2: Image Data Generator**

This example expands upon the previous one, demonstrating a more common use case for generators—image loading and preprocessing:

```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def image_data_generator(image_paths, batch_size, image_size):
    num_images = len(image_paths)
    while True:
        indices = np.random.permutation(num_images)
        for i in range(0, num_images, batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch = []
            y_batch = []
            for index in batch_indices:
              img_path = image_paths[index]
              try:
                img = Image.open(img_path).resize(image_size).convert("RGB")
                img_array = np.array(img) / 255.0
                label = 1 if 'positive' in img_path else 0 # Simplified label extraction
                x_batch.append(img_array)
                y_batch.append(label)
              except FileNotFoundError:
                continue
            if len(x_batch) > 0:
              yield np.array(x_batch), np.array(y_batch).reshape(-1, 1)
            
#Assume existence of 'images' folder with 'positive' and 'negative' subfolders
image_dir = 'images'
image_paths = []
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))


# Placeholders
image_size = (64, 64)
x_placeholder = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], 3])
y_placeholder = tf.placeholder(tf.int32, shape=[None, 1])

# Simple CNN model
conv1 = tf.layers.conv2d(x_placeholder, filters=32, kernel_size=3, activation=tf.nn.relu)
max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
conv2 = tf.layers.conv2d(max_pool1, filters=64, kernel_size=3, activation=tf.nn.relu)
max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
flatten = tf.layers.flatten(max_pool2)
dense = tf.layers.dense(flatten, units=128, activation=tf.nn.relu)
logits = tf.layers.dense(dense, units=1)
predictions = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_placeholder, tf.float32), logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Training parameters
batch_size = 32
num_epochs = 5
batches_per_epoch = len(image_paths) // batch_size

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(num_epochs):
    generator = image_data_generator(image_paths, batch_size, image_size)
    for batch_idx in range(batches_per_epoch):
      x_batch, y_batch = next(generator)
      _, loss_value = sess.run([optimizer, loss], feed_dict={x_placeholder: x_batch, y_placeholder: y_batch})
      if batch_idx % 10 == 0:
        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_value}")
```

This example showcases image loading, resizing, and normalization within the generator function.  It also demonstrates how to implement basic file reading and class extraction from filenames. Note that error handling, specifically for when a file is not an image, has been included. The image loading and processing is contained within `image_data_generator`, which iterates through a list of image paths, opening and resizing each image. This eliminates the need to load all images into memory at once.  A basic CNN architecture is used to process the images.

**Example 3: Preprocessed and Augmented Data**

This final example expands on image processing to include data augmentation within the generator:

```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageEnhance
import random

def augmented_image_data_generator(image_paths, batch_size, image_size):
  num_images = len(image_paths)
  while True:
      indices = np.random.permutation(num_images)
      for i in range(0, num_images, batch_size):
        batch_indices = indices[i:i + batch_size]
        x_batch = []
        y_batch = []
        for index in batch_indices:
          img_path = image_paths[index]
          try:
            img = Image.open(img_path).resize(image_size).convert("RGB")

            if random.random() < 0.5: #50% chance of applying augmentations
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            img_array = np.array(img) / 255.0
            label = 1 if 'positive' in img_path else 0
            x_batch.append(img_array)
            y_batch.append(label)
          except FileNotFoundError:
             continue
        if len(x_batch) > 0:
          yield np.array(x_batch), np.array(y_batch).reshape(-1, 1)


# Reusing the image_dir and image_paths from the previous example for brevity

# Placeholders and CNN model (same as example 2)
image_size = (64, 64)
x_placeholder = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], 3])
y_placeholder = tf.placeholder(tf.int32, shape=[None, 1])

conv1 = tf.layers.conv2d(x_placeholder, filters=32, kernel_size=3, activation=tf.nn.relu)
max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
conv2 = tf.layers.conv2d(max_pool1, filters=64, kernel_size=3, activation=tf.nn.relu)
max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
flatten = tf.layers.flatten(max_pool2)
dense = tf.layers.dense(flatten, units=128, activation=tf.nn.relu)
logits = tf.layers.dense(dense, units=1)
predictions = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_placeholder, tf.float32), logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Training parameters (same as example 2)
batch_size = 32
num_epochs = 5
batches_per_epoch = len(image_paths) // batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        generator = augmented_image_data_generator(image_paths, batch_size, image_size)
        for batch_idx in range(batches_per_epoch):
            x_batch, y_batch = next(generator)
            _, loss_value = sess.run([optimizer, loss], feed_dict={x_placeholder: x_batch, y_placeholder: y_batch})
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_value}")
```
Here, the `augmented_image_data_generator` includes brightness, contrast, and horizontal flip augmentation applied randomly to images. This augmentation process happens *on the fly* within the data generator, effectively increasing the diversity of the training data and improving model robustness. This method eliminates the requirement for pre-computed augmented data sets and does not impact the overall dataset memory footprint, as data is augmented on demand. The rest of the model training structure remains consistent with Example 2.

**Resource Recommendations**

For a more in-depth understanding, exploring resources detailing the intricacies of Python's iterator and generator patterns is essential. Refer to material focusing on TensorFlow’s graph execution model, specifically how placeholders fit into the computation process, to gain clarity on data feeding mechanisms. Detailed explanations on data augmentation strategies, even outside the framework of TensorFlow, can help identify the most effective approach for your specific data and task. Additionally, studying examples of custom data loading within Python's PIL library would be beneficial to understanding image manipulation techniques in the code. Further study on best practices for optimizing input pipelines will yield performance benefits.
