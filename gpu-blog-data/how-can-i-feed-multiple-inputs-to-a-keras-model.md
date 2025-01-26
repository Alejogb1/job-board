---
title: "How can I feed multiple inputs to a Keras model?"
date: "2025-01-26"
id: "how-can-i-feed-multiple-inputs-to-a-keras-model"
---

The common approach to managing multiple inputs in a Keras model involves leveraging the functional API, rather than the sequential API, which restricts the model to a single input and output. This is foundational for handling diverse data streams, such as structured data combined with image data, or time-series data coupled with categorical features. I've encountered this frequently, particularly when constructing predictive models that require incorporating various contextual factors beyond a single data source.

The core principle revolves around defining separate `Input` layers for each distinct input source. Each of these `Input` layers acts as a starting point for its corresponding data path. Subsequently, the outputs of these separate pathways are then merged, typically using concatenation or other merging layers like `add` or `multiply`, depending on the semantic relationship between the inputs. The merged output is then fed into subsequent layers for further processing and prediction. This allows the network to learn from each input source independently before combining the learned representations. The functional API offers this explicit control over data flow which is not available with the sequential API.

**Code Example 1: Concatenating Multiple Input Vectors**

This example demonstrates how to combine two numerical feature vectors using concatenation. Imagine building a model to predict customer churn, where one input is customer transaction history (represented as a vector of aggregated values), and the other input is a demographic vector.

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define input layers
input_transaction = keras.Input(shape=(10,), name='transaction_input')
input_demographic = keras.Input(shape=(5,), name='demographic_input')

# Hidden layers for each input
dense_transaction = layers.Dense(32, activation='relu')(input_transaction)
dense_demographic = layers.Dense(16, activation='relu')(input_demographic)

# Concatenate the two hidden layer outputs
merged = layers.concatenate([dense_transaction, dense_demographic])

# Final dense layer
output = layers.Dense(1, activation='sigmoid')(merged)

# Create the model
model = keras.Model(inputs=[input_transaction, input_demographic], outputs=output)

# Generate dummy data for demonstration
transactions = np.random.rand(100, 10)
demographics = np.random.rand(100, 5)
labels = np.random.randint(0, 2, 100)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([transactions, demographics], labels, epochs=10, verbose=0)

print("Model trained successfully on concatenated inputs.")
```

In this example, `input_transaction` and `input_demographic` are `Input` layers defined with their respective shapes. Each input is processed through its own dense layer before being concatenated using `layers.concatenate`. The `keras.Model` is constructed with *lists* of inputs ( `[input_transaction, input_demographic]` ) and a single output, which is crucial when providing multiple inputs. Failure to use a list for the inputs argument will result in an error.

**Code Example 2: Processing Image and Text Data**

This example showcases a more complex scenario: feeding an image and a text sequence as inputs to a model for, perhaps, image captioning. This reflects a challenge I often face when integrating multimedia data.

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define input layers
input_image = keras.Input(shape=(64, 64, 3), name='image_input')
input_text = keras.Input(shape=(50,), name='text_input')

# Image processing pathway
conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
flattened_image = layers.Flatten()(pool2)
dense_image = layers.Dense(128, activation='relu')(flattened_image)

# Text processing pathway
embedding = layers.Embedding(input_dim=1000, output_dim=64)(input_text)
lstm = layers.LSTM(64)(embedding)

# Merge the image and text features
merged = layers.concatenate([dense_image, lstm])

# Final output layer
output = layers.Dense(10, activation='softmax')(merged)

# Create the model
model = keras.Model(inputs=[input_image, input_text], outputs=output)

# Generate dummy data for demonstration
images = np.random.rand(100, 64, 64, 3)
texts = np.random.randint(0, 1000, size=(100, 50))
labels = np.random.randint(0, 10, 100)


# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([images, texts], labels, epochs=10, verbose=0)

print("Model trained successfully on image and text inputs.")
```
Here, `input_image` takes a 3D input representing color image data, while `input_text` is a sequence of integer-encoded text tokens.  Convolutional and pooling layers process the image, and an embedding and LSTM layer process the text input. Both processed inputs are then concatenated.

**Code Example 3: Using the `add` Layer for a Residual Connection**

This example demonstrates using the `add` layer to create a residual connection, which I have found extremely useful when working with deeper models. The input vector is fed through the network using a standard processing path and then added to a version of the input vector that has undergone simple feature transformation.

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the input layer
input_layer = keras.Input(shape=(20,), name='main_input')

# Define the processing path
dense1 = layers.Dense(64, activation='relu')(input_layer)
dense2 = layers.Dense(64, activation='relu')(dense1)

# Create a linear transformation of the input
linear_transform = layers.Dense(64)(input_layer)

# Add the transformed input to the output of the processing path
added = layers.add([dense2, linear_transform])

# Final Output Layer
output = layers.Dense(1, activation='sigmoid')(added)

# Create the model
model = keras.Model(inputs=input_layer, outputs=output)

# Generate dummy data for demonstration
data = np.random.rand(100, 20)
labels = np.random.randint(0, 2, 100)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, verbose=0)

print("Model trained successfully using an add layer for a residual connection")

```
In this instance, the `add` layer performs element-wise addition of its inputs. Note that both the output of the main processing path, and the linear transformation have the same shape. For element-wise operations, such as addition or multiplication, all inputs to these merging layers must have identical dimensions. If these dimensions do not match, a dimension mismatch error will result.

When working with multiple inputs, providing the correct data during training and evaluation is crucial. The data must be provided as a list of numpy arrays or TensorFlow tensors, with each element in the list corresponding to one of the input layers. The order of arrays in the list must match the order in which `Input` layers were passed to the `keras.Model` constructor. For instance, if the `Model` is created with `inputs=[input_a, input_b]`, the data during the `.fit` call would be structured as `[data_for_input_a, data_for_input_b]`. This order is not automatically enforced; it’s a developer responsibility.

In addition to the merge types demonstrated (concatenate and add), you will also find that `layers.multiply`, `layers.maximum`, and `layers.average` layers are useful depending on the relationships between your inputs.

For a comprehensive understanding, refer to Keras documentation, specifically the sections on the functional API, input layers, merging layers, and custom models. A thorough understanding of fundamental neural network architectures will also be beneficial for effectively designing and implementing multiple input models. Additionally, examining pre-existing architectures on sites like TensorFlow Hub can provide insights into how multiple inputs are often implemented in practice. Reading research papers that tackle similar problems will prove useful as well.
