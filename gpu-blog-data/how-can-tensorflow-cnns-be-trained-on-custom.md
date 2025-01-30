---
title: "How can TensorFlow CNNs be trained on custom RGB images?"
date: "2025-01-30"
id: "how-can-tensorflow-cnns-be-trained-on-custom"
---
Training convolutional neural networks (CNNs) with TensorFlow on custom RGB images requires careful management of data input pipelines and model architecture. The default expectation of many TensorFlow training examples often revolves around pre-packaged datasets. However, real-world applications frequently involve bespoke image datasets with specific structures. My experience, gained while developing an automated medical image analysis tool, highlighted the nuances involved in preparing these datasets for efficient CNN training. Let's delve into the specifics.

First, consider the data. Raw image data, in its native format, is typically not suitable for direct feeding into a TensorFlow model. We need a mechanism to load, pre-process, and batch the images before they reach the neural network. This involves creating a robust input pipeline. TensorFlow's `tf.data` API is designed for precisely this purpose. It allows us to build complex data handling logic, optimize for performance, and integrates seamlessly with the training loop.

The process typically includes these steps:

1.  **File Listing:** We begin by generating a list of file paths that point to our RGB images. This can be achieved through Python's `glob` module or similar utilities. It is critical that file paths accurately reflect the location of the image files on disk.

2.  **Label Association:** We must associate each image with its corresponding label. Labels represent the ground truth or target output for the network. The association can be derived from the filename, folder structure, or an external mapping file. This process is crucial, as the neural network learns to predict these labels. For multi-class classification, labels need to be converted to one-hot vectors or integer encodings.

3.  **Image Loading and Decoding:** Raw image files are typically in compressed formats, such as JPEG or PNG. Using TensorFlow's `tf.io.read_file` function, we can load the raw bytes. Then, depending on the image type, `tf.io.decode_jpeg` or `tf.io.decode_png` decodes these bytes into a usable tensor representation. For RGB images, this is a rank-3 tensor with dimensions `(height, width, channels)`, where channels equals 3.

4.  **Data Pre-processing:** Image pre-processing often includes resizing, normalization, and augmentation. `tf.image.resize` allows you to reshape images to a consistent size before feeding them into the network. Normalization is important because neural networks perform best with data that is roughly within the range of 0 to 1. This can be accomplished by dividing pixel values by 255. Augmentation helps introduce diversity in the training data and reduces overfitting. Techniques such as random rotations, flips, and zooms can be used via `tf.image` functions.

5.  **Batching and Shuffling:** Batches group several samples together before feeding them into the network. Batching reduces training time and computational overhead. Shuffling the data prevents the network from learning spurious patterns based on the order of data. This can be achieved using `dataset.batch` and `dataset.shuffle` methods of the `tf.data.Dataset` object.

Let's look at some illustrative code examples.

**Example 1: Basic Image Loading and Preprocessing**

```python
import tensorflow as tf
import glob

def load_and_preprocess_image(image_path, label):
    """Loads, decodes, and preprocesses a single image."""

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)  # Or decode_png
    image = tf.image.resize(image, [224, 224])  # Standard size
    image = tf.cast(image, tf.float32) / 255.0    # Normalize
    return image, label

# Assuming image paths and labels are available in lists
image_paths = glob.glob("images/*.jpg")  # Placeholder for your image path glob
labels = [0 if 'cat' in path else 1 for path in image_paths] #Example binary classification
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image) # Apply the preprocessing function
dataset = dataset.batch(32) # Create batches of 32
dataset = dataset.shuffle(buffer_size=len(image_paths)) # Shuffle with buffer size of data size

# iterate through data
#for image_batch, label_batch in dataset:
#    print (image_batch.shape, label_batch.shape)
```

This code snippet demonstrates the core of building an input pipeline. The `load_and_preprocess_image` function encapsulates the loading, decoding, resizing, and normalization steps. We use `tf.data.Dataset.from_tensor_slices` to create a dataset from paths and labels, map the preprocessing function and then batch. The shuffle operation is crucial to preventing the network learning based on dataset sequence. Commented out is a simple print statement to display the shape of the resulting batch tensors.

**Example 2: Image Augmentation**

```python
import tensorflow as tf
import glob

def augment_image(image, label):
    """Augments image with random rotations and flips."""

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_rotation(image, tf.random.uniform([], -0.1, 0.1))
    return image, label


def load_and_preprocess_image(image_path, label):
    """Loads, decodes, preprocesses, and augments a single image."""

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # Or decode_png
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return augment_image(image, label) # Apply image augmentation


image_paths = glob.glob("images/*.jpg")  # Placeholder for your image path glob
labels = [0 if 'cat' in path else 1 for path in image_paths] #Example binary classification
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image) # Apply the preprocessing and augmentation
dataset = dataset.batch(32)
dataset = dataset.shuffle(buffer_size=len(image_paths))

#for image_batch, label_batch in dataset:
#    print (image_batch.shape, label_batch.shape)
```

Here we've added an `augment_image` function using TensorFlow's image manipulation functions to introduce variations. It has been integrated with the existing pipeline. We apply random horizontal flips, vertical flips and small rotations. This type of augmentation helps the model become more robust to image variations. Again, I have commented out the simple batch printing statement.

**Example 3: Training Loop Integration**

```python
import tensorflow as tf
import glob

def load_and_preprocess_image(image_path, label):
    """Loads, decodes, preprocesses a single image."""

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # Or decode_png
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

image_paths = glob.glob("images/*.jpg")
labels = [0 if 'cat' in path else 1 for path in image_paths] #Example binary classification
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image)
dataset = dataset.batch(32)
dataset = dataset.shuffle(buffer_size=len(image_paths))

model = tf.keras.applications.MobileNetV2(include_top=True, weights=None, classes=2)
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric.update_state(labels, predictions)
    return loss

epochs = 10

for epoch in range(epochs):
    for images, labels in dataset:
        loss = train_step(images, labels)
    accuracy = metric.result()
    metric.reset_states()
    print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}')

```

This final example shows integration into a training loop using a pre-trained MobileNetV2 as the base model. I've chosen sparse categorical crossentropy for the loss function and Adam as optimizer. We define the `train_step` within a `tf.function` for computational efficiency, enabling faster execution of training calculations. We are evaluating our metrics for each epoch and finally printing it out to the screen.

For resources outside of the basic TensorFlow documentation, I would suggest exploring online tutorials focusing on specific aspects such as `tf.data` pipelines for image loading, practical guides on image augmentation techniques with TensorFlow, and examples of custom training loops beyond the `model.fit` method. Additionally, resources detailing the impact of normalization and data types on training performance can prove valuable. Online courses dedicated to CNNs often provide a comprehensive understanding of the concepts applied in these code snippets, typically containing modules on data pre-processing specific to image data.
