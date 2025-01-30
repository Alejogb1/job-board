---
title: "How can I concatenate two Keras embedding layers and train the resulting model?"
date: "2025-01-30"
id: "how-can-i-concatenate-two-keras-embedding-layers"
---
The core challenge in concatenating Keras embedding layers stems from their inherent function: transforming discrete integer inputs into dense, continuous vector representations. Simply merging the raw embedding matrices is invalid; the output dimensions won't align with the model's subsequent layers. Instead, we must merge their *outputs* after the embedding lookup is performed, treating each embedding layer's output as a distinct input feature. This requires understanding the shape of these outputs and how Keras handles layer concatenation.

My experiences working with multi-modal text analysis models frequently involve combining diverse embedding spaces. For instance, a model might use one embedding for word tokens and another for character-level n-grams. To effectively combine them, I’ve consistently leveraged the `Concatenate` layer provided by Keras, making sure that both are flattened from a 3D tensor of batch size x sequence length x embedding size to a 2D tensor of batch size x (sequence length * embedding size).

**Explanation:**

Keras embedding layers generate a 3D tensor output with the dimensions (batch_size, sequence_length, embedding_dim). When concatenating two such layers, we cannot directly feed these 3D tensors into a `Concatenate` layer along the embedding dimension. If both have the same sequence length, and you wanted to combine embedding vectors at each time-step you can concatenate using the *last dimension* or axis=-1. But that would not be advisable and it would generally be more difficult to work with. For a more flexible approach, the approach to flatten the output of both embedding layers and then concatenate them will always work. The flattening operation essentially reshapes the (batch_size, sequence_length, embedding_dim) tensor to (batch_size, sequence_length * embedding_dim). This allows for subsequent fully connected layers or any other type of model architecture that requires a 2D tensor input.

The `Concatenate` layer expects inputs with matching dimensionality across all dimensions except for the concatenation axis. Since flattened embeddings will have two dimensions only, the axis argument is not required. The output shape, after concatenation, will be (batch_size, sum_of_flattened_embedding_sizes), meaning that you can proceed to create any architecture downstream. The critical aspect is calculating the correct combined size and ensuring the shape is appropriately considered when constructing the fully connected layers.

Once concatenated, the resulting tensor represents the combined features from both embedding spaces. It can then be used as input to other layers such as dense layers, recurrent layers, or convolutional layers, depending on the model’s task. Training such a model involves backpropagation through all connected layers including the embeddings, allowing the model to learn the optimal representation in both embedding spaces jointly and to also learn how to combine the representations.

**Code Examples:**

**Example 1: Basic Concatenation**

This example shows the simplest case of flattening two embedding layers and using concatenation to combine them before a dense layer.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input
from tensorflow.keras.models import Model

# Parameters
vocab_size_1 = 1000
embedding_dim_1 = 64
vocab_size_2 = 500
embedding_dim_2 = 32
sequence_length = 20

# Input layers
input_1 = Input(shape=(sequence_length,), name="input_1")
input_2 = Input(shape=(sequence_length,), name="input_2")


# Embedding layers
embedding_1 = Embedding(input_dim=vocab_size_1, output_dim=embedding_dim_1, name="embedding_1")(input_1)
embedding_2 = Embedding(input_dim=vocab_size_2, output_dim=embedding_dim_2, name="embedding_2")(input_2)

# Flatten layers
flatten_1 = Flatten(name="flatten_1")(embedding_1)
flatten_2 = Flatten(name="flatten_2")(embedding_2)

# Concatenate layer
merged = Concatenate(name="concatenate")([flatten_1, flatten_2])

# Dense layer
output = Dense(1, activation="sigmoid", name="output")(merged)

# Model definition
model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
```

In this code, `input_1` and `input_2` are our integer sequence inputs. The embedding layers, `embedding_1`, and `embedding_2`, transform these into dense vector representations. The `Flatten` layers then collapse the output of the embedding layers. The `Concatenate` layer merges these flattened vectors into a single vector which then passes to a dense output layer. The model expects two separate input sequences as it is a multi-input model and will train the embedding layers jointly by backpropagating from the output.

**Example 2: Using separate embedding dimensions and a different number of layers**

This example is similar to the previous one but also showcases a change in the embedding dimensionality of the two embedding layers and has some additional dense layers. This is meant to highlight the flexibility and potential utility of the approach.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input
from tensorflow.keras.models import Model

# Parameters
vocab_size_1 = 2000
embedding_dim_1 = 128
vocab_size_2 = 1000
embedding_dim_2 = 64
sequence_length = 30


# Input layers
input_1 = Input(shape=(sequence_length,), name="input_1")
input_2 = Input(shape=(sequence_length,), name="input_2")


# Embedding layers
embedding_1 = Embedding(input_dim=vocab_size_1, output_dim=embedding_dim_1, name="embedding_1")(input_1)
embedding_2 = Embedding(input_dim=vocab_size_2, output_dim=embedding_dim_2, name="embedding_2")(input_2)


# Flatten layers
flatten_1 = Flatten(name="flatten_1")(embedding_1)
flatten_2 = Flatten(name="flatten_2")(embedding_2)

# Concatenate layer
merged = Concatenate(name="concatenate")([flatten_1, flatten_2])

# Dense layers
dense_1 = Dense(128, activation='relu', name="dense_1")(merged)
dense_2 = Dense(64, activation='relu', name="dense_2")(dense_1)
output = Dense(1, activation="sigmoid", name="output")(dense_2)

# Model definition
model = Model(inputs=[input_1, input_2], outputs=output)


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
```

This example extends the first by including two intermediate dense layers between the concatenation and output layers. It also sets different vocab and embedding dimensions for the two different embedding layers. Note that the number of hidden units in the intermediate dense layers are arbitrary and can be optimized during training.

**Example 3: Handling Variable Sequence Length**

Often, sequences will have different lengths. This example shows how padding and masking can be used to allow this. This will not influence concatenation procedure which is based on shapes after embeddings. This is usually necessary in most realistic settings.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Concatenate, Flatten, Dense, Input, Masking
from tensorflow.keras.models import Model

# Parameters
vocab_size_1 = 1500
embedding_dim_1 = 80
vocab_size_2 = 750
embedding_dim_2 = 40
sequence_length = None # variable sequence length


# Input layers
input_1 = Input(shape=(sequence_length,), name="input_1")
input_2 = Input(shape=(sequence_length,), name="input_2")


# Embedding layers
embedding_1 = Embedding(input_dim=vocab_size_1, output_dim=embedding_dim_1, mask_zero = True, name="embedding_1")(input_1)
embedding_2 = Embedding(input_dim=vocab_size_2, output_dim=embedding_dim_2, mask_zero = True, name="embedding_2")(input_2)

# Flatten layers
flatten_1 = Flatten(name="flatten_1")(embedding_1)
flatten_2 = Flatten(name="flatten_2")(embedding_2)


# Concatenate layer
merged = Concatenate(name="concatenate")([flatten_1, flatten_2])

# Dense layer
output = Dense(1, activation="sigmoid", name="output")(merged)

# Model definition
model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Summary
model.summary()

```

This example defines variable sequence length by setting `sequence_length = None` in the input shape. It also adds the parameter `mask_zero=True` to each embedding layer. This is useful to ensure the embeddings for padded sequences do not contribute to the final learned representation which would skew the results. Note that `Masking` layers are optional but using `mask_zero = True` at the embedding layer is preferable to using `Masking` layers explicitly.

**Resource Recommendations:**

1.  **TensorFlow API documentation:** The official TensorFlow documentation provides detailed information on Keras layers including the `Embedding`, `Concatenate`, and `Flatten` layers. This is crucial for a deep understanding of how each layer functions and the expected shapes of input and output tensors.

2.  **Online Keras tutorials:** Many free online resources offer practical tutorials demonstrating various Keras functionalities, including concatenating different layers. These tutorials often provide additional context and application examples, aiding practical implementation.

3.  **Deep Learning textbooks:** Books on deep learning, such as "Deep Learning" by Goodfellow et al., cover fundamental concepts in neural networks and are essential for a theoretical understanding of embedding spaces, dimensionality, and concatenation techniques. These will be most helpful when making decisions about shapes and dimensionality.
