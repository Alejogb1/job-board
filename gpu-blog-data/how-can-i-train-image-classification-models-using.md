---
title: "How can I train image classification models using Colab?"
date: "2025-01-30"
id: "how-can-i-train-image-classification-models-using"
---
Image classification model training in Google Colab leverages cloud-based computational resources, specifically GPUs and TPUs, significantly reducing the time required compared to local CPU-based training. This advantage, coupled with Colab's free tier and pre-installed machine learning libraries, makes it a practical choice for many practitioners. My own experience, spanning multiple projects from proof-of-concept experiments to refining larger datasets, consistently highlights the platform's efficacy for this task.

The core process involves several key stages: data preparation, model definition, training, and evaluation. In Colab, we typically start by connecting to Google Drive, where our datasets are commonly stored. This eliminates the need for downloading the data locally for each session. Utilizing libraries like TensorFlow or PyTorch, we can then load image datasets, preprocess them through scaling, resizing, and augmenting, and split them into training, validation, and potentially testing sets.

Next, a convolutional neural network (CNN) is defined, either from scratch or by utilizing a pre-trained model from a repository like TensorFlow Hub or PyTorch's model zoo. Pre-trained models, often trained on large datasets like ImageNet, offer a significant advantage by capturing general features present across many images. We typically “fine-tune” the pre-trained model by adding our own classification layer and adjusting a select few of the model's layers based on our specific data. If starting from scratch, the CNN architecture will need to be explicitly specified, incorporating convolutional, pooling, and fully connected layers tailored to the complexity of the classification problem.

Training then proceeds by feeding the preprocessed training data, in mini-batches, through the chosen CNN. An optimization algorithm, like Adam or SGD, iteratively adjusts the model's parameters to minimize a defined loss function (e.g., categorical cross-entropy), the discrepancy between predicted and true class labels. During training, the model's performance is monitored on a validation set, preventing overfitting on training data alone. This monitoring involves calculating metrics like accuracy or F1-score. The validation set is crucial for evaluating generalization capabilities on unseen data.

Finally, after model training, we evaluate the performance on a held-out test set, to obtain a final objective metric of how the model will perform on completely new, never-seen, data. The model can then be saved, allowing it to be reused later, either in Colab again, or in a separate deployment environment. This saved model can be uploaded back to Google Drive or downloaded to a local machine.

Here are three illustrative code examples using TensorFlow/Keras, reflecting typical training processes:

**Example 1: Training with Data Generator**

This example showcases how to leverage the `ImageDataGenerator` to load images from a directory and generate augmented data during training. This approach is particularly useful when handling large datasets.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define image dimensions and batch size
img_height, img_width = 150, 150
batch_size = 32

# Data augmentation and loading for training set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'path/to/training/data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Data loading for validation set
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'path/to/validation/data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model Definition - Simple CNN
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(len(train_generator.class_indices), activation='softmax') # Output layer
])

# Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

```

This code snippet first sets up the `ImageDataGenerator`, configured with augmentation parameters for the training set. The `flow_from_directory` function reads images from specified directories. The model architecture uses two convolutional layers followed by flattening and fully connected layers. Finally, it compiles the model with an appropriate optimizer and loss function, then begins the training process for 10 epochs. The number of epochs is intentionally low here to be illustrative.

**Example 2: Transfer Learning with Pre-Trained Model**

This example demonstrates how to utilize a pre-trained model (VGG16) from Keras' application library, freezing its convolutional base and adding a custom classification layer.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 224, 224  # Required input size for VGG16
batch_size = 32

# Load pre-trained VGG16 model, excluding the classifier
base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add a custom classification layer
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Build the new model
model = Model(inputs=base_model.input, outputs=output)

# Data loading with rescaling only
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path/to/training/data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'path/to/validation/data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

```

Here, the VGG16 model is loaded, with the top layers excluded (classification part), as that will be replaced by a custom classification layer suitable to the problem data. The convolutional base is frozen during training so we don't disrupt the model's learned weights from ImageNet. We add a fully connected layer with dropout for regularization and finally the output classification layer. Finally, we compile and train it as previously described.

**Example 3: Using a Custom Training Loop**

This example demonstrates how to define a custom training loop for more granular control over the training process, useful for tasks like gradient accumulation or custom logging.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Example data (replace with your actual dataset loading)
train_images = np.random.rand(1000, 150, 150, 3).astype(np.float32) #Example data of 1000 images
train_labels = np.random.randint(0, 5, 1000) # Example labels of 5 classes
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(100).batch(32)

img_height, img_width = 150, 150

# Model definition (same simple CNN as example 1)
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(5, activation='softmax') # Output layer for 5 classes
])

# Optimizer
optimizer = Adam(learning_rate=0.001)

# Loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Metrics
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# Training Loop
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_fn(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_accuracy.update_state(labels, predictions)
  return loss

epochs = 10
for epoch in range(epochs):
  for images, labels in train_dataset:
    loss = train_step(images, labels)

  print(f"Epoch: {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {train_accuracy.result():.4f}")
  train_accuracy.reset_state()

```

In this example, a custom training loop is defined to get the loss and accuracy and then to display the metrics at the end of each epoch. Notice we have to define our own training loop, optimizer, loss function, and metric. The `tf.function` decorator compiles the train_step function into a graph for efficient execution. This example also illustrates, very simply, how to make a custom data loading scheme using TensorFlow's dataset APIs, although you would more likely use `ImageDataGenerator` for real-world data loading.

To improve skills in this area, several resources are available. The official TensorFlow and PyTorch websites offer extensive documentation, tutorials, and example notebooks covering the topics discussed. A good understanding of convolutional neural networks can be gained through books like "Deep Learning" by Ian Goodfellow et al. and online courses such as those offered by Coursera or edX. Specifically, focusing on materials on data augmentation, transfer learning, and training loop implementations can greatly improve the results achieved in practical projects.
