---
title: "How can TensorFlow Inception be retrained?"
date: "2025-01-30"
id: "how-can-tensorflow-inception-be-retrained"
---
Transfer learning with TensorFlow Inception models significantly reduces training time and computational resources when adapting pre-trained models to new image classification tasks. Instead of building a convolutional neural network from scratch, you leverage the feature extraction capabilities of a network trained on a large dataset, like ImageNet, and fine-tune only specific layers for your target domain. This approach addresses the challenge of limited data in specific domains and mitigates the risk of overfitting. I've personally deployed several image recognition systems in industrial inspection using this technique, consistently achieving high accuracy with relatively small datasets.

The core concept involves freezing the weights of the pre-trained convolutional base and adding new, trainable layers on top. These trainable layers are then optimized during training to classify the new target categories. Inception's architecture, characterized by its multiple parallel convolutional layers (inception modules), provides a diverse set of learned features useful for a wide range of visual classification problems. The specific layers you choose to freeze and retrain impact performance and resource consumption. Typically, the earlier convolutional layers, capturing basic image features like edges and textures, remain frozen, while layers closer to the output, specialized for ImageNet's categories, require retraining.

A crucial step is preparing your dataset. Images should be organized into a directory structure that corresponds to the categories you intend to classify. Each category must be in its own subdirectory containing the corresponding images. TensorFlow's `tf.data.Dataset` API is ideally suited for efficiently loading and preprocessing image data. You will likely want to resize your images to the input dimensions expected by the Inception model. Typically this is 299 x 299 pixels for Inception v3, the model we will use in the examples. Additionally, data augmentation techniques, such as random rotations, flips, and zooms, are critical for enhancing the generalization capability of the model when trained with limited data.

Below are three code examples demonstrating how to retrain TensorFlow Inception:

**Example 1: Basic Retraining with tf.keras**

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# 1. Data Preparation
data_dir = 'path/to/your/image/dataset' #Replace with actual path
image_size = (299, 299)
batch_size = 32
train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=42
)
val_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)

num_classes = len(train_dataset.class_names)

# 2. Load InceptionV3 Base Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 3. Freeze the Base Layers
for layer in base_model.layers:
    layer.trainable = False

# 4. Add Custom Classifier Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #Optional intermediate layer
predictions = Dense(num_classes, activation='softmax')(x)

# 5. Create the Model
model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Model
epochs = 10
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# 8. Optional: Save Model
model.save('retrained_inception_model.h5')
```

This example showcases a standard retraining approach using the `tf.keras` API. First, it loads images from a directory using `image_dataset_from_directory`, then it loads a pre-trained InceptionV3 model with its top classification layer removed (`include_top=False`). It freezes all the layers of the base model and adds a GlobalAveragePooling2D layer, a dense layer, and a final dense layer with the number of output classes. The model is then compiled with an Adam optimizer and trained using the training and validation datasets. The model is saved as an H5 file for later use. This example is foundational and provides a starting point that may suit many image classification tasks.

**Example 2: Unfreezing Some Top Layers**

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# 1. Data Preparation (same as Example 1)
data_dir = 'path/to/your/image/dataset'
image_size = (299, 299)
batch_size = 32
train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=42
)
val_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)

num_classes = len(train_dataset.class_names)


# 2. Load InceptionV3 Base Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 3. Unfreeze Some of Top Base Layers
for layer in base_model.layers[:249]: # Unfreeze layers after layer 249
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True


# 4. Add Custom Classifier Layers (same as Example 1)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 5. Create the Model (same as Example 1)
model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile the Model (same as Example 1)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Model (same as Example 1)
epochs = 10
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# 8. Optional: Save Model (same as Example 1)
model.save('retrained_inception_model.h5')
```

Building on Example 1, this code demonstrates fine-tuning. After loading the InceptionV3 model and adding classification layers, I selectively unfreeze the top layers. This can improve model accuracy because the features captured by the later layers of the base model are more domain-specific and may need adjustment for the new classification problem. By setting  `layer.trainable = True` for layers after index 249,  the weights of the higher layers are also optimized during training. The threshold for unfreezing layers is task-dependent. You must experiment to find the optimal configuration for your dataset.

**Example 3: Incorporating Data Augmentation**

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import os

# 1. Data Preparation with Augmentation
data_dir = 'path/to/your/image/dataset'
image_size = (299, 299)
batch_size = 32

def augment_data(image, label):
    image = RandomFlip(mode="horizontal")(image)
    image = RandomRotation(factor=0.2)(image)
    image = RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))(image)
    return image, label

train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=42
).map(augment_data)

val_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=42
)

num_classes = len(train_dataset.class_names)


# 2. Load InceptionV3 Base Model (same as Example 1)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 3. Freeze the Base Layers (same as Example 1)
for layer in base_model.layers:
    layer.trainable = False

# 4. Add Custom Classifier Layers (same as Example 1)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 5. Create the Model (same as Example 1)
model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile the Model (same as Example 1)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Model (same as Example 1)
epochs = 10
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# 8. Optional: Save Model (same as Example 1)
model.save('retrained_inception_model.h5')
```

This example enhances the previous code by integrating data augmentation directly into the data loading pipeline. The `augment_data` function, containing RandomFlip, RandomRotation, and RandomZoom layers, is applied to each batch of the training dataset using the `.map` operation. Data augmentation artificially expands the dataset's diversity, which leads to increased model robustness and reduces overfitting, especially when working with limited training samples. The augmentation layers are included before the model training and act directly upon the image tensor, providing dynamically transformed images for the training process.

For further study and understanding, I recommend consulting resources like the TensorFlow documentation for `tf.keras.applications.InceptionV3`, `tf.data.Dataset`, and `tf.keras.layers`. Publications from the original Inception paper authors are also valuable for understanding the architecture. Specifically, I would advise studying the concepts of transfer learning, fine-tuning, and data augmentation.
