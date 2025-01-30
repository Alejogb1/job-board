---
title: "How to access MNIST data in TensorFlow Keras without the `read_data_sets` attribute?"
date: "2025-01-30"
id: "how-to-access-mnist-data-in-tensorflow-keras"
---
TensorFlow's `tensorflow.examples.tutorials.mnist` module, specifically its `read_data_sets` function, is often cited as the canonical method for accessing the MNIST dataset in older tutorials and examples. However, this module has been deprecated and is no longer the recommended approach within the current TensorFlow ecosystem, particularly when working with Keras. The most appropriate method involves utilizing the `tf.keras.datasets.mnist` module, which provides pre-processed data directly without the need for custom data loading or the deprecated `read_data_sets` function.

The transition from `read_data_sets` to `tf.keras.datasets.mnist` significantly simplifies the process. Instead of downloading, unpacking, and processing raw data files, `tf.keras.datasets.mnist` provides direct access to NumPy arrays representing training and testing images and their corresponding labels. This pre-processing makes the dataset immediately compatible with TensorFlow and Keras models without additional intermediate steps.

To elaborate, `read_data_sets` was part of a larger, more general data loading framework within TensorFlow designed to support various types of datasets, not solely MNIST. It required explicit specification of data paths and custom logic for parsing image files. The `tf.keras.datasets` API, in contrast, focuses on providing pre-processed, common datasets directly as NumPy arrays, streamlining the developer experience for model development, particularly when Keras is the chosen high-level API. The deprecated approach involved the creation of custom `Dataset` objects and handling the loading and shuffling of data, a task now handled internally by Keras. This eliminates a substantial amount of boiler plate code that would have been required with older versions.

The advantage of the current approach is clear. The focus shifts directly to model building and experimentation, without the additional burden of manual data processing. Below are code examples demonstrating this shift.

**Example 1: Basic MNIST Loading and Exploration**

```python
import tensorflow as tf
import numpy as np

# Load the MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print shapes of training and test data
print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Explore the data type and range
print(f"Data type of images: {x_train.dtype}")
print(f"Min pixel value: {np.min(x_train)}")
print(f"Max pixel value: {np.max(x_train)}")

# Display the first image
import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
```

This first example illustrates the core functionality of `tf.keras.datasets.mnist.load_data()`. This function directly returns a tuple of two tuples: the training data and labels as NumPy arrays, and then the testing data and labels. The dimensions of these arrays show that we have 60,000 training images of 28x28 pixels and 10,000 testing images of the same size, along with their respective labels. The images are stored as unsigned 8-bit integers, representing pixel intensities ranging from 0 to 255. The last section displays the first image from the training set using `matplotlib` along with the corresponding numerical label.

**Example 2: Preparing Data for a Simple Neural Network**

```python
import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images for fully connected layers
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))


# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# Display first five reshaped images
print(f"First five flattened image shapes (training): {x_train[:5].shape}")
print(f"First five reshaped label shapes (training): {y_train[:5].shape}")

# Build a simple sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=2, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy*100:.2f}%")
```

This example builds upon the previous one by preparing the data for a simple neural network. The pixel values are normalized to the [0, 1] range, which is a common preprocessing step. The 28x28 images are flattened into 784-dimensional vectors as required by fully connected layers. Labels are converted to a one-hot encoded categorical format using `tf.keras.utils.to_categorical`. The code demonstrates the first five flattened image shapes and the first five one-hot encoded label shapes of the training data. It then defines and compiles a basic sequential model and evaluates performance of the trained model on the test data.

**Example 3: Using Convolutional Neural Networks (CNNs)**

```python
import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add a channel dimension for CNNs
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# Display shape of reshaped image data
print(f"Training images shape (CNN): {x_train.shape}")
print(f"Testing images shape (CNN): {x_test.shape}")


# Build a CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=2, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy*100:.2f}%")
```
This final example demonstrates how to use the MNIST dataset with a Convolutional Neural Network (CNN). The same data loading and normalization steps are applied. However, an additional step is required to add a channel dimension using `tf.newaxis`, which transforms the shape from (num_images, 28, 28) to (num_images, 28, 28, 1), which is the shape needed as an input to the convolutional layers.  The code demonstrates the shape after this transformation. This is required because convolutional layers expects the input to be multi-dimensional with one of the dimensions representing the number of channels (in this case, the images are black and white, hence 1 channel). A simple CNN model is built and trained. It consists of two convolutional layers each followed by max pooling layers, then flattened and passed into a final dense layer for classification.  The model's performance on the test dataset is then measured and the result printed to the console.

To further enhance one's understanding of data loading and preprocessing in TensorFlow Keras, I recommend consulting the official TensorFlow documentation, which provides comprehensive guides and tutorials on `tf.keras.datasets`, as well as the Keras API reference. For a more general deep learning background, exploring texts specifically focused on practical applications of deep learning can be beneficial. These resources, while not explicitly referencing the deprecated `read_data_sets`, offer relevant best practices that are essential for modern TensorFlow-based development. These practices include data normalization, one-hot encoding, and correct use of tensors for both model building and model evaluation.
