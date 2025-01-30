---
title: "How can I change the number of classes in a MNIST TensorFlow model?"
date: "2025-01-30"
id: "how-can-i-change-the-number-of-classes"
---
The output layer's dimensionality in a TensorFlow model processing MNIST data is directly tied to the number of classes the model is trained to distinguish. Modifying this output layer, and consequently the model's architecture, to accommodate a different number of classes requires a careful reconstruction of the final layers. My experience building image classification models has demonstrated that a simple tweak isn't sufficient; the entire data pipeline and loss function often need corresponding adjustments.

The key area to focus on is the final fully connected (dense) layer of the model, often preceding a softmax activation for classification tasks. For standard MNIST, designed to differentiate between 10 digits (0-9), this layer typically has 10 output nodes. Altering this number means changing both the number of nodes and potentially the shape of the weight matrix connecting this layer to the preceding one. Furthermore, the loss function used during training must align with the adjusted output structure. For example, categorical cross-entropy is common for multi-class scenarios. If converting to a binary problem, it would require the output layer to have one node, representing the probability of belonging to the positive class and using binary cross-entropy loss.

Let's examine a few practical modifications, starting from a common model structure used for MNIST, and progressively modifying it for different classification scenarios.

**Example 1: Reducing MNIST to a Binary Classification (Even vs. Odd)**

Here we reduce the dataset to a binary problem: classifying images as either representing an even or an odd digit. This involves preprocessing the data and then altering the output layer.

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load and Preprocess MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create binary labels (0 for even, 1 for odd)
y_train_binary = np.array([1 if i % 2 != 0 else 0 for i in y_train])
y_test_binary = np.array([1 if i % 2 != 0 else 0 for i in y_test])

# Reshape for convolution (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the Model
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Modified Output Layer (1 node, sigmoid)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Binary Cross-Entropy Loss
              metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train_binary, epochs=5, batch_size=32)

# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test_binary, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')
```

In this example, the most significant modification is changing the final `Dense` layer to have only one output node with a `sigmoid` activation, since we are working with a binary classification problem.  Furthermore, the loss function was changed from `categorical_crossentropy` to `binary_crossentropy`, which is appropriate for binary classification. The labels were also converted into binary labels.

**Example 2: Reducing MNIST to a 5-Class Classification (Digits 0-4)**

This example modifies the model to categorize only the first five digits of the MNIST dataset (0 to 4).

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Load and Preprocess MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filter Data to Digits 0-4
mask_train = (y_train >= 0) & (y_train <= 4)
mask_test = (y_test >= 0) & (y_test <= 4)

x_train = x_train[mask_train]
y_train = y_train[mask_train]
x_test = x_test[mask_test]
y_test = y_test[mask_test]

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One-hot encode the labels (for categorical cross-entropy)
y_train_5class = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test_5class = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Reshape for convolution (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Define the Model
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # Modified Output Layer (5 nodes, softmax)
])

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', #Categorical Cross-Entropy Loss
              metrics=['accuracy'])


# Train the Model
model.fit(x_train, y_train_5class, epochs=5, batch_size=32)


# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test_5class, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')
```

Here, we filtered the dataset to include only digits 0-4. The output layer was changed to 5 nodes with `softmax` activation for 5-class classification. It's important to use `softmax` when dealing with multi-class problems where the output values are interpreted as probabilities. The labels are one-hot encoded to be compatible with the `categorical_crossentropy` loss function.

**Example 3: Increasing MNIST to 11 Class Classification**

To demonstrate increasing the number of classes, I will artificially augment the data, creating a dummy 11th class. In a real-world scenario, this would involve a meaningful augmentation of the dataset.

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load and Preprocess MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Create a dummy 11th class by duplicating a portion of the data and giving it a new class label 10.
dummy_size = 1000
dummy_images_train = x_train[:dummy_size].copy()
dummy_labels_train = np.full(dummy_size, 10)
x_train = np.concatenate((x_train, dummy_images_train), axis=0)
y_train = np.concatenate((y_train, dummy_labels_train), axis=0)

dummy_images_test = x_test[:dummy_size // 10].copy()
dummy_labels_test = np.full(dummy_size//10, 10)
x_test = np.concatenate((x_test, dummy_images_test), axis=0)
y_test = np.concatenate((y_test, dummy_labels_test), axis=0)

# One-hot encode the labels (for categorical cross-entropy)
y_train_11class = tf.keras.utils.to_categorical(y_train, num_classes=11)
y_test_11class = tf.keras.utils.to_categorical(y_test, num_classes=11)


# Reshape for convolution (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Define the Model
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(11, activation='softmax') # Modified Output Layer (11 nodes, softmax)
])


# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the Model
model.fit(x_train, y_train_11class, epochs=5, batch_size=32)

# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test_11class, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')
```

Here, I've created an additional (artificial) class and associated it with a duplicated subset of images, demonstrating that the network can be altered to accommodate more than 10 classes. The final dense layer now has 11 units and still uses a softmax activation function since this is a multi-class problem.

**Resource Recommendations**

For enhancing your understanding of these concepts, I recommend several resources. Consult textbooks and online materials focused on convolutional neural networks, particularly chapters or sections explaining the architecture and parameters of the fully connected and output layers. Review the documentation of the TensorFlow Keras API, focusing on dense layers, activation functions (sigmoid, softmax), and commonly used loss functions like `binary_crossentropy` and `categorical_crossentropy`. Additionally, explore research publications on the application of CNNs to image classification and investigate various techniques for data augmentation. Also, study different optimization algorithms to improve model performance and convergence times. These areas provide a robust foundation for further experimentation and mastery.
