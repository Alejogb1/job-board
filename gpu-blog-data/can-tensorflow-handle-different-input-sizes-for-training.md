---
title: "Can TensorFlow handle different input sizes for training and testing?"
date: "2025-01-30"
id: "can-tensorflow-handle-different-input-sizes-for-training"
---
TensorFlow’s inherent graph-based execution model permits the use of variable input sizes for both training and testing phases, although careful consideration must be given to data pipeline construction and model architecture. Unlike frameworks that might impose strict batch size limitations or input dimensionality, TensorFlow achieves flexibility through symbolic placeholders and dynamic computations.

My experience deploying image classification models in a high-throughput manufacturing environment required exactly this capability. Training images, sourced from a controlled lab setting, were consistently sized at 224x224 pixels. However, real-world images captured on the factory floor could exhibit slight variations in dimensions, often resulting from minor shifts in camera positioning or image cropping. We addressed this discrepancy without needing to preprocess and resize all real-world inputs, thereby streamlining the inference pipeline. The key resides in the separation of the data input process from the model definition within TensorFlow.

Here's how it's generally done: The fundamental approach involves defining input tensors with `None` as the size for dimensions that can vary. In TensorFlow 2.x, this is managed through the `tf.keras.Input` layer, which serves as the entry point to your model’s architecture. Within that layer’s definition, if you specify a shape tuple, use `None` where the size is not fixed. For instance, an image input might look like `tf.keras.Input(shape=(None, None, 3))`, indicating that the width and height can vary, but the three color channels are fixed.

The crucial understanding is that even if you define a flexible input shape, operations within the model graph must be compatible with these varying dimensions. Convolutional layers, for example, will work with inputs of varying spatial extent. However, dense layers or flattening layers, which often directly follow convolutional layers, will require careful management. Commonly, you'll insert operations that can dynamically adapt to varying feature map sizes. One strategy I frequently employed involves global average pooling before entering dense layers, which condenses the spatially-varying feature maps to a fixed size.

The data ingestion pipeline plays an equally significant role. When training, it's common to batch data, but if you've defined your input shape with `None` for variable dimensions, you'll need to ensure that each batch consists of samples of the same size. You can achieve this through techniques like padding or, more practically, by grouping similarly sized samples together during the preprocessing stage before forming the batches. When testing or inferencing, the input image might be individual and need not be part of a batch, making the flexible input size a natural fit.

Let's explore some concrete code examples.

**Example 1: Basic Image Classifier with Variable Input Size**

```python
import tensorflow as tf

def build_model():
    input_tensor = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_tensor = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Example of training with images of varying sizes
# (simplified example, normally use tf.data.Dataset)
import numpy as np
batch_size = 32

# Generate dummy training data. 
# For brevity, let's assume 32 images of one size, then another 32 of another
train_images_1 = np.random.rand(batch_size, 224, 224, 3).astype('float32')
train_labels_1 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_1_one_hot = tf.keras.utils.to_categorical(train_labels_1, num_classes=10)

train_images_2 = np.random.rand(batch_size, 192, 192, 3).astype('float32')
train_labels_2 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_2_one_hot = tf.keras.utils.to_categorical(train_labels_2, num_classes=10)

model.train_on_batch(train_images_1, train_labels_1_one_hot)
model.train_on_batch(train_images_2, train_labels_2_one_hot)

# Example of inference with a single image
test_image = np.random.rand(256, 256, 3).astype('float32')
test_image = np.expand_dims(test_image, axis=0) # Add batch dimension
prediction = model.predict(test_image)
print(f"Prediction shape: {prediction.shape}")
```

In this example, `tf.keras.Input(shape=(None, None, 3))` defines the input shape, enabling the model to accept images of different heights and widths during training and inference. I opted for `GlobalAveragePooling2D` rather than flattening and dense layers directly following the convolutions because it allows for varying spatial dimensions on input.

**Example 2: Using a pre-trained model with dynamic input sizes**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

def build_transfer_learning_model():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output_tensor = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes
    model = tf.keras.Model(inputs=base_model.input, outputs=output_tensor)
    return model

transfer_model = build_transfer_learning_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
transfer_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Example of training with images of varying sizes (same as Example 1)
import numpy as np
batch_size = 32

train_images_1 = np.random.rand(batch_size, 224, 224, 3).astype('float32')
train_labels_1 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_1_one_hot = tf.keras.utils.to_categorical(train_labels_1, num_classes=10)

train_images_2 = np.random.rand(batch_size, 192, 192, 3).astype('float32')
train_labels_2 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_2_one_hot = tf.keras.utils.to_categorical(train_labels_2, num_classes=10)

transfer_model.train_on_batch(train_images_1, train_labels_1_one_hot)
transfer_model.train_on_batch(train_images_2, train_labels_2_one_hot)

# Example of inference with a single image
test_image = np.random.rand(256, 256, 3).astype('float32')
test_image = np.expand_dims(test_image, axis=0)
prediction = transfer_model.predict(test_image)
print(f"Prediction shape: {prediction.shape}")
```

This shows that pre-trained models (like `ResNet50` here, instantiated with `include_top=False`) can also leverage dynamic input shapes. We can modify the model by adding `GlobalAveragePooling2D` and new dense layers on top. I chose a common image net model in this case.

**Example 3: Using tf.data.Dataset with Variable Image Sizes**

```python
import tensorflow as tf
import numpy as np

def preprocess_image(image):
    # Add any needed preprocessing operations here
    return image

def create_dataset(images, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.map(lambda image, label: (preprocess_image(image), label))
  
  # group similar image sizes into a single batch
  dataset = dataset.batch(batch_size) 
  dataset = dataset.prefetch(tf.data.AUTOTUNE) 
  return dataset

# Generate dummy data as above
batch_size = 32
train_images_1 = np.random.rand(batch_size, 224, 224, 3).astype('float32')
train_labels_1 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_1_one_hot = tf.keras.utils.to_categorical(train_labels_1, num_classes=10)

train_images_2 = np.random.rand(batch_size, 192, 192, 3).astype('float32')
train_labels_2 = np.random.randint(0, 10, size=(batch_size,)).astype('float32')
train_labels_2_one_hot = tf.keras.utils.to_categorical(train_labels_2, num_classes=10)

# Merge the two types of input data
all_train_images = tf.concat([train_images_1,train_images_2], axis=0)
all_train_labels = tf.concat([train_labels_1_one_hot, train_labels_2_one_hot], axis=0)


train_dataset = create_dataset(all_train_images, all_train_labels, batch_size)

# Example model (simplified to save space)
input_tensor = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_tensor = tf.keras.layers.Dense(10, activation='softmax')(x) 
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Training Loop
for epoch in range(2):
  for images, labels in train_dataset:
    model.train_on_batch(images, labels)


# Example of inference with a single image
test_image = np.random.rand(256, 256, 3).astype('float32')
test_image = np.expand_dims(test_image, axis=0) # Add batch dimension
prediction = model.predict(test_image)
print(f"Prediction shape: {prediction.shape}")
```

Here, I've demonstrated usage of the `tf.data.Dataset` to handle data batches with dynamic shapes. I show a dummy `preprocess_image` function which can perform any needed transformation. The key here is the `tf.data.Dataset.from_tensor_slices` which creates a dataset from numpy data, the `batch` step, and also prefetching of the data. I included a dummy training loop that would fit this data. This approach is more common when training with different input sizes, because it can be readily modified to group image inputs by shape when creating batches. Note that in real usage, one will want to define a `tf.data.Dataset` from file paths, not directly from data, but the concept remains the same.

To further your understanding, I recommend exploring the official TensorFlow documentation on `tf.keras.Input`, `tf.data.Dataset`, and the `tf.keras.layers` that operate on tensors of dynamic sizes such as `Conv2D`, `MaxPooling2D`, and `GlobalAveragePooling2D`. Additionally, consulting the guides on building training pipelines can be invaluable when scaling up the use of varying input sizes. These resources provide deeper insight into the mechanisms that make dynamic input shapes possible.

In conclusion, TensorFlow effectively handles different input sizes due to its graph-based architecture and flexible layer implementations. The key to a successful implementation involves carefully designing model architectures and data pipelines to accommodate the varying shapes. This flexibility allows for more robust and efficient handling of diverse datasets, which is paramount in many practical applications.
