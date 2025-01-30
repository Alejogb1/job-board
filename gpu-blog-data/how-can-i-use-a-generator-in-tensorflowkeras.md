---
title: "How can I use a generator in TensorFlow/Keras to train a model with two inputs?"
date: "2025-01-30"
id: "how-can-i-use-a-generator-in-tensorflowkeras"
---
Training Keras models with multiple inputs often necessitates a nuanced understanding of data pipelines, particularly when dealing with large datasets.  My experience working on a medical image analysis project involving both MRI and CT scans highlighted the critical need for efficient data handling, and generators provided the optimal solution.  The key lies in structuring your generator to yield appropriately shaped batches of data for both inputs simultaneously.  Failing to do so results in shape mismatches and training errors.


**1. Clear Explanation:**

TensorFlow/Keras models with multiple inputs require a generator that yields a tuple of NumPy arrays, where each array corresponds to a single input. The length of the tuple equals the number of inputs, and each array within the tuple should have a shape compatible with the respective input layer of the model. The generator needs to handle the pre-processing for each input independently before combining them into a single batch.  Furthermore, it's crucial to ensure the generator yields data in batches, matching the `batch_size` specified during model training to avoid memory overload, especially with large images or sequences.  The `fit_generator` (now deprecated in favor of `fit`) method expects this structured output.  I have encountered numerous scenarios where incorrect data shaping led to cryptic errors, necessitating careful consideration of this aspect.

Consider a scenario with two inputs:  input A (images) and input B (corresponding numerical features). Input A might be a set of 128x128 grayscale images, while input B might be a vector of 10 features.  Your generator must yield batches where each batch contains a NumPy array of shape (batch_size, 128, 128, 1) for input A and a NumPy array of shape (batch_size, 10) for input B.  The `fit` method will then appropriately distribute these batches to the corresponding input layers.


**2. Code Examples with Commentary:**

**Example 1: Simple Image and Numerical Feature Generator:**

```python
import numpy as np
import tensorflow as tf

def multi_input_generator(image_data, numerical_data, batch_size):
    """Generates batches of image and numerical data."""
    num_samples = len(image_data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # Shuffle for better generalization

    while True:
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            image_batch = image_data[batch_indices]
            numerical_batch = numerical_data[batch_indices]
            yield (image_batch, numerical_batch)


# Example Usage
image_data = np.random.rand(1000, 128, 128, 1) # 1000 images, 128x128, grayscale
numerical_data = np.random.rand(1000, 10)       # 1000 samples, 10 features
batch_size = 32

train_generator = multi_input_generator(image_data, numerical_data, batch_size)

# Model Definition (Illustrative)
input_a = tf.keras.Input(shape=(128, 128, 1))
input_b = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_a)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.concatenate([x, input_b])
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_generator, steps_per_epoch=len(image_data) // batch_size, epochs=10)

```

This example showcases a fundamental generator structure. It shuffles data for better generalization and yields batches of images and numerical features. The model definition is illustrative and can be adapted to any multi-input architecture.  The `steps_per_epoch` parameter is crucial; it prevents the generator from running indefinitely.


**Example 2: Handling Different Input Data Types:**

```python
import numpy as np
import tensorflow as tf

def multi_input_generator_diff_types(text_data, image_data, batch_size):
    # Assumes text_data is pre-processed into numerical representations (e.g., word embeddings)
    num_samples = len(text_data)
    indices = np.arange(num_samples)

    while True:
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            text_batch = text_data[batch_indices]
            image_batch = image_data[batch_indices]
            yield ([text_batch, image_batch], np.random.rand(batch_size,1)) # Example Output

# Example usage (adapt input shapes and data accordingly)

text_data = np.random.rand(1000,100) #Example word embeddings
image_data = np.random.rand(1000,64,64,3)

train_generator = multi_input_generator_diff_types(text_data, image_data, batch_size=32)

#Model Definition (Illustrative)
input_a = tf.keras.Input(shape=(100,))
input_b = tf.keras.Input(shape=(64,64,3))
x_a = tf.keras.layers.Dense(64, activation='relu')(input_a)
x_b = tf.keras.layers.Conv2D(32,(3,3), activation='relu')(input_b)
x_b = tf.keras.layers.Flatten()(x_b)
x = tf.keras.layers.concatenate([x_a, x_b])
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_generator, steps_per_epoch=len(text_data)//32, epochs=10)

```

This example demonstrates handling different data types (numerical vectors for text and images). The key is to pre-process the text data appropriately (e.g., converting text to word embeddings) before feeding it to the generator.  The output structure of the generator remains consistent; it still yields a tuple containing all inputs. The output is a single array in this example for illustrative purposes; adjust as needed for your model's output requirements.


**Example 3:  Generator with Pre-processing:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def multi_input_generator_preprocess(image_dir, numerical_data, batch_size, img_size=(128,128)):
    image_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
    image_generator = image_gen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None #Since class labels are in numerical data
    )
    num_samples = len(numerical_data)

    while True:
        image_batch = next(image_generator)
        batch_indices = np.random.choice(num_samples, batch_size, replace=False)
        numerical_batch = numerical_data[batch_indices]
        yield ([image_batch, numerical_batch], np.random.rand(batch_size,1)) #Example Output


# Example usage (adapt paths and data accordingly)
image_dir = "path/to/your/image/directory"  #Replace with your directory
numerical_data = np.random.rand(1000, 10)

train_generator = multi_input_generator_preprocess(image_dir, numerical_data, 32)

#Model Definition (Illustrative)
input_a = tf.keras.Input(shape=(128, 128, 3))
input_b = tf.keras.Input(shape=(10,))
x_a = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_a)
x_a = tf.keras.layers.Flatten()(x_a)
x = tf.keras.layers.concatenate([x_a, input_b])
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_generator, steps_per_epoch=len(numerical_data)//32, epochs=10)

```

This example integrates pre-processing using `ImageDataGenerator` for image augmentation. This streamlines the data pipeline, handling image loading and augmentation within the generator. Note the use of `flow_from_directory` which assumes a directory structure for images; this differs from the previous examples where image data was loaded directly.



**3. Resource Recommendations:**

The TensorFlow documentation provides comprehensive details on data input pipelines and generators.  Consult the official Keras documentation for best practices regarding model building and training.  A thorough understanding of NumPy array manipulation is essential for efficient data handling.  Finally, books on deep learning and TensorFlow/Keras will provide broader context and advanced techniques.
