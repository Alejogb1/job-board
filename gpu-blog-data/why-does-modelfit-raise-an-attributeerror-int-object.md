---
title: "Why does `model.fit()` raise an AttributeError: 'int' object has no attribute 'ndim'?"
date: "2025-01-30"
id: "why-does-modelfit-raise-an-attributeerror-int-object"
---
The `AttributeError: 'int' object has no attribute 'ndim'` during a Keras `model.fit()` call typically signals a mismatch between the expected input data structure and the actual data provided, specifically within the context of shape and dimensionality.  Having encountered this in various image processing projects, including a recent attempt to train a variational autoencoder with misconfigured input pipelines, I've found this error arises primarily when the model expects a multi-dimensional tensor (e.g., a NumPy array representing images or time-series data), but receives an integer. The `ndim` attribute is essential for Keras to understand the dimensionality of the input data, and the absence of this attribute on an integer object prevents the model from properly processing the input batch.

Fundamentally, Keras models are designed to operate on tensors, which are multi-dimensional arrays. The `fit()` method expects to receive inputs in the form of tensors – typically NumPy arrays or TensorFlow tensors – with a shape that’s compatible with the model's input layer. The error occurs when the input data, which should be a multi-dimensional array, is mistakenly converted to a scalar (an integer in this case) before being passed to `fit()`. This conversion often stems from incorrect data preprocessing steps, flaws in batching logic, or an oversight in how the training dataset is structured. The error occurs because the internal logic of Keras relies on checking the `ndim` attribute of incoming data to ensure it's a tensor before operating on it; an integer does not have such attribute.

The error tends to surface during data loading, batching, or custom generator implementation when: a) the input `x` passed to `model.fit(x, y)` is inadvertently an integer, b) the data yielded from a custom generator is incorrectly formatted to return single numerical values instead of tensors for a batch of examples, c) when accessing dataset elements you've retrieved a label rather than a feature set, or d) when you've specified a batch size of one using a custom generator, thereby yielding single examples that fail to have the tensor dimensionality that keras expects.

Here are three code examples, each depicting a scenario that can cause this error and its associated solution:

**Example 1: Incorrectly Formatted Batch Data**

```python
import numpy as np
import tensorflow as tf

# Problem:  Creating a batch where each sample is a scalar rather than vector

data_size = 100
features = np.random.rand(data_size, 10)
labels = np.random.randint(0, 2, data_size) # Example for classification

def generate_bad_batches(features, labels, batch_size):
    num_batches = len(features) // batch_size
    for i in range(num_batches):
       batch_indices = i*batch_size : (i+1)*batch_size
       yield (labels[batch_indices][0], features[batch_indices]) #Incorrect, passing an integer
    
# Dummy model for testing purposes
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Incorrect usage - triggers AttributeError
try:
    model.fit(generate_bad_batches(features, labels, 5), epochs=2, steps_per_epoch = 10 )
except Exception as e:
    print(f"Error: {e}")


# Solution: Correctly structure batch as a tuple of features, labels in list form
def generate_good_batches(features, labels, batch_size):
    num_batches = len(features) // batch_size
    for i in range(num_batches):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        yield (features[batch_indices], labels[batch_indices])


#Correct usage: Passes a batch of tensor with appropriate dimenstion

model.fit(generate_good_batches(features, labels, 5), epochs=2, steps_per_epoch = 10)
print("Model trained successfully!")
```

*Commentary:* In this example, `generate_bad_batches` incorrectly yields `labels[batch_indices][0]`, which is a single integer from the label array, rather than a tensor representing multiple labels for the batch. The solution involves yielding the complete batch of labels `labels[batch_indices]` alongside the batch of features. The corrected `generate_good_batches` function ensures that `fit` receives batch-level data with the appropriate tensor shapes.

**Example 2: Mishandled Data within Custom Generators**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class BadDataGenerator(Sequence):
    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.data_len = len(features)

    def __len__(self):
        return self.data_len // self.batch_size

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        return self.labels[batch_start], self.features[batch_start:batch_end] # Incorrect: Returning single label, not a batch

class GoodDataGenerator(Sequence):
      def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.data_len = len(features)

      def __len__(self):
         return self.data_len // self.batch_size

      def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        return self.features[batch_start:batch_end], self.labels[batch_start:batch_end]

# Dummy model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data_size = 100
features = np.random.rand(data_size, 10)
labels = np.random.randint(0, 2, data_size)

bad_generator = BadDataGenerator(features, labels, 5)
good_generator = GoodDataGenerator(features, labels, 5)
# Incorrect usage - triggers AttributeError
try:
    model.fit(bad_generator, epochs=2, steps_per_epoch = 10 )
except Exception as e:
    print(f"Error: {e}")

# Correct Usage
model.fit(good_generator, epochs = 2, steps_per_epoch = 10)
print("Model trained successfully!")
```

*Commentary:*  The `BadDataGenerator`'s `__getitem__` returns `self.labels[batch_start]` which is the first label of the current batch—an integer— instead of the batch of labels (`self.labels[batch_start:batch_end]`).  This again causes `fit()` to receive an integer as the input or output.  The `GoodDataGenerator` corrects this by properly passing batches of both features and labels to the `fit()` function.

**Example 3: Incorrect Data Indexing from NumPy Arrays**

```python
import numpy as np
import tensorflow as tf

# Problem: Accessing labels before features
data_size = 100
features = np.random.rand(data_size, 10)
labels = np.random.randint(0, 2, data_size)

# Dummy model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Incorrect usage - triggers AttributeError:
try:
    model.fit(labels, labels, epochs=2, batch_size = 5) # Wrong inputs
except Exception as e:
    print(f"Error: {e}")
# Solution: Correctly using features as input
model.fit(features, labels, epochs=2, batch_size=5)
print("Model trained successfully!")
```

*Commentary:* In this simplified example, the error results from directly passing the `labels` array as both input and output to `model.fit()` . The input to `fit` must match the input expected by the first layer. The fix is to use features, which matches the model input layer as the first input to fit.

To prevent this error, it is crucial to verify the following before calling `model.fit()`:

1. **Data Loading:** Ensure that the loaded data is in the appropriate tensor format (NumPy array or TensorFlow tensor), and not a scalar. Print out the shape of the inputs with `print(data.shape)` before feeding to `fit` and ensure this agrees with the input layer specified for your model.

2. **Batching:** If you’re using a custom generator, double-check that it yields a tuple or list where the first element is the training features batch and the second the training labels batch; and ensure both have the appropriate dimensionality. The batch should *not* yield a single scalar, nor a tuple or list where any elements are single scalars.

3. **Indexing/Slicing:** Carefully review how data is accessed from NumPy arrays or TensorFlow tensors. Verify that you're selecting entire batches rather than individual elements before yielding your batch.

4. **Input Layer Shape:** The input data shape must be compatible with the model's input layer. Ensure you have verified that your model input layer's `input_shape` parameter is correctly defined.

5. **Debugging:** Use print statements to examine the shapes and data types of your data before feeding to `fit`.

Resource recommendations for understanding and resolving this type of issue include:  the official Keras documentation, particularly sections on custom data generators and handling input data. Various tutorials exist that teach how to use the `tf.data` API. Additionally, referring to more general resources on numerical computation with NumPy and TensorFlow’s tensor operations provides further clarity on working with multi-dimensional arrays.
