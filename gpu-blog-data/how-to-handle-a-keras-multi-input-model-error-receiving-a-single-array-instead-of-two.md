---
title: "How to handle a Keras multi-input model error receiving a single array instead of two?"
date: "2025-01-26"
id: "how-to-handle-a-keras-multi-input-model-error-receiving-a-single-array-instead-of-two"
---

A common pitfall when working with Keras multi-input models surfaces when the model expects multiple input arrays, but receives only one. This discrepancy usually manifests as a shape mismatch error during the training or prediction phase. I've encountered this particular issue multiple times, especially when refactoring data pipelines or when simplifying data loading processes initially developed for single-input architectures. The core problem stems from incorrectly passing the input data to the model, treating the multiple input streams as a singular, concatenated or improperly structured entity.

To correctly address this, it’s crucial to understand how Keras expects multi-input models to receive their data. A Keras model defined with multiple input layers, often using the `keras.layers.Input()` function, establishes a contract specifying the number and shape of input arrays the model expects. The input arrays must be provided as a *list* or *tuple* of NumPy arrays, where each array corresponds to a specific input layer in the model's architecture. When a single array is provided, the model incorrectly interprets it, leading to shape compatibility errors within its internal layers. The error typically appears when invoking the model's training methods like `fit()` or evaluation methods such as `predict()`. It's not solely a data formatting error. It is fundamentally about the alignment between the model's expected input structure and the provided input.

Let's illustrate with a conceptual example. Assume I have built a model designed to process both textual and numerical input: the textual data might be reviews, and the numerical data might represent user features. I have defined two input layers, one to receive text embeddings, and another to receive numerical features. I must then ensure that the data is appropriately formatted into separate arrays before feeding it to the model. Failing to structure the input data correctly will result in the Keras model treating the entire input as a single data stream for a single input layer which is not designed to handle both data types or shapes.

The error message varies slightly, but often highlights the dimension mismatch between the received input and the expected input dimensions of the corresponding input layer(s). It typically involves messages indicating an incompatibility between the shape of the input array being provided, and the expected shape defined by the input layer. Resolving this issue requires careful data preparation, ensuring each expected input array is provided independently, and that the order of inputs matches the order in which the input layers were defined.

Below are three code examples illustrating different scenarios and solutions.

**Example 1: Incorrect Single Array Input**

This demonstrates how passing a single, concatenated array leads to a shape error.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# Define input layers
input_text = Input(shape=(100,))
input_numeric = Input(shape=(10,))

# Define processing layers
text_dense = Dense(64, activation='relu')(input_text)
numeric_dense = Dense(32, activation='relu')(input_numeric)

# Concatenate
merged = concatenate([text_dense, numeric_dense])

# Output layer
output = Dense(1, activation='sigmoid')(merged)

# Define the model
model = Model(inputs=[input_text, input_numeric], outputs=output)

# Generate some fake data
text_data = np.random.rand(100, 100)
numeric_data = np.random.rand(100, 10)

# Incorrect: Passing a single, concatenated array
combined_data = np.concatenate((text_data, numeric_data), axis=1)

try:
    model.fit(combined_data, np.random.rand(100, 1), epochs=1)
except Exception as e:
    print(f"Error encountered:\n {e}")

```

Here, the intention was to pass `text_data` to `input_text` and `numeric_data` to `input_numeric`. However, I concatenated both `text_data` and `numeric_data` creating a single `combined_data` array. The model `fit()` method then attempts to reconcile the entire `combined_data` to both `input_text` and `input_numeric`. The resulting shape mismatch triggers the error, clearly stating that the input does not match the expected input shape. The error message usually is some variation of "ValueError: Input 0 is incompatible with the layer: expected min_ndim=2, got ndim=3". This indicates the model is expecting a list or tuple of two separate inputs, not one concatenated input.

**Example 2: Correctly Passing a List of Arrays**

This code shows the correct approach, passing two separate arrays within a list.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# Input layers (same as before)
input_text = Input(shape=(100,))
input_numeric = Input(shape=(10,))

# Processing layers (same as before)
text_dense = Dense(64, activation='relu')(input_text)
numeric_dense = Dense(32, activation='relu')(input_numeric)

# Concatenate
merged = concatenate([text_dense, numeric_dense])

# Output layer
output = Dense(1, activation='sigmoid')(merged)

# Define the model
model = Model(inputs=[input_text, input_numeric], outputs=output)

# Generate data (same as before)
text_data = np.random.rand(100, 100)
numeric_data = np.random.rand(100, 10)

# Correct: Passing a list of two arrays
model.fit([text_data, numeric_data], np.random.rand(100, 1), epochs=1)

print("Model trained successfully with separate input arrays")
```

In this corrected example, I have passed a list containing `text_data` and `numeric_data` to the `model.fit()` method. Keras understands that `text_data` corresponds to `input_text` and `numeric_data` corresponds to `input_numeric`. By structuring the input as a list or tuple of arrays, I have fulfilled the requirements of a multi-input model and have avoided the shape error.

**Example 3: Using a Dictionary for Input Mapping**

This demonstrates using a dictionary, which offers additional clarity, especially with many inputs.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# Input layers (same as before)
input_text = Input(shape=(100,), name='text_input')
input_numeric = Input(shape=(10,), name='numeric_input')

# Processing layers (same as before)
text_dense = Dense(64, activation='relu')(input_text)
numeric_dense = Dense(32, activation='relu')(input_numeric)

# Concatenate
merged = concatenate([text_dense, numeric_dense])

# Output layer
output = Dense(1, activation='sigmoid')(merged)

# Define the model
model = Model(inputs=[input_text, input_numeric], outputs=output)

# Generate data (same as before)
text_data = np.random.rand(100, 100)
numeric_data = np.random.rand(100, 10)

# Correct: Passing data using dictionary mapping
model.fit({'text_input': text_data, 'numeric_input': numeric_data}, np.random.rand(100, 1), epochs=1)

print("Model trained successfully with dictionary input mapping")

```

This approach uses a Python dictionary to explicitly map each input layer's name to the corresponding input array. When defining each input layer, I assigned names using `name='text_input'` and `name='numeric_input'`. This allows for a more readable and maintainable mapping especially when dealing with models that have a high number of input layers. The keys of the dictionary should correspond with the names of the input layers as assigned in the `keras.Input` function. The model can now successfully process the input because each input array is correctly associated with its corresponding input layer.

For further understanding, I recommend exploring resources focused on Keras functional API and its input layer definitions. The official Keras documentation provides thorough explanations, examples, and details on working with multiple inputs. Additionally, reviewing tutorials on custom data generators or pipelines for multi-input models can be helpful. It's also worthwhile examining code examples from community driven projects or articles demonstrating different techniques to structure data before passing it to the model. Examining codebases on sites like GitHub can provide real-world examples and insights on input management for Keras models. These resources can solidify your understanding of how Keras handles multiple inputs.
