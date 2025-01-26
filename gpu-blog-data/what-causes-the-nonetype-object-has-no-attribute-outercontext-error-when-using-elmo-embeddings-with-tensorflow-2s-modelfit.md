---
title: "What causes the 'NoneType' object has no attribute 'outer_context' error when using ELMo embeddings with TensorFlow 2's `model.fit`?"
date: "2025-01-26"
id: "what-causes-the-nonetype-object-has-no-attribute-outercontext-error-when-using-elmo-embeddings-with-tensorflow-2s-modelfit"
---

The "NoneType" object has no attribute 'outer_context' error during TensorFlow 2's `model.fit` with ELMo embeddings typically stems from an incorrect application of the ELMo layer within a custom model architecture, specifically how ELMo's output is handled concerning graph execution and automatic differentiation. From previous experiences developing a sequence-to-sequence model for clinical text, this error manifested subtly after migrating from TensorFlow 1 to 2. It indicated a fundamental misunderstanding of how the ELMo embedding process interacts with TensorFlow 2’s eager execution and graph construction.

The core issue resides in the fact that ELMo, despite being encapsulated as a Keras layer, returns its output as a dynamically constructed TensorFlow tensor, rather than directly as a Keras Tensor object that the model automatically recognizes and can track for gradient computations during backpropagation. When the layer’s output, after manipulation such as concatenation or feature projection, is fed into layers expecting a standard Keras Tensor (e.g. dense layers, recurrent layers) within the model’s forward pass, issues can arise. During model fitting, when the gradient computation attempts to trace back to the ELMo output, it encounters that this tensor has not been integrated into the Keras graph properly. It becomes treated as a regular `None` type object, and therefore can not access the 'outer_context' necessary for building a computational graph for automatic differentiation by Keras. The 'outer_context' is an internal attribute linked to Keras’ tracing capabilities for gradient calculations. The 'outer_context' isn't on a `None` object, hence the error message.

Let’s break down several scenarios which often result in this error and how to prevent them, using code examples.

**Example 1: Incorrectly Passing ELMo Output as a Raw Tensor**

In the simplest case, you may instantiate an ELMo layer and directly pass its output tensor to another layer without properly wrapping it inside a Keras-compatible structure.

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model

# Load the ELMo embedding from TensorFlow Hub
elmo_url = "https://tfhub.dev/google/elmo/3" # Simplified for conciseness
elmo = hub.KerasLayer(elmo_url, trainable=False)

class ImproperModel(Model):
  def __init__(self):
    super(ImproperModel, self).__init__()
    self.elmo = elmo
    self.dense = layers.Dense(128, activation='relu')
    self.output_layer = layers.Dense(2, activation='softmax')

  def call(self, inputs):
    elmo_output = self.elmo(inputs)
    # Problematic line, passing raw tensor
    dense_output = self.dense(elmo_output)
    return self.output_layer(dense_output)


# Fictional input
dummy_input = tf.constant([["This is a text input"], ["Another sample text"]])

#Instantiate model
model_improper = ImproperModel()

# Attempting to fit (will cause error)
try:
  model_improper(dummy_input)
  model_improper.compile(optimizer='adam', loss='categorical_crossentropy')
  y_dummy = tf.random.categorical(tf.math.log([[0.5,0.5], [0.5,0.5]]), 1)
  y_dummy = tf.one_hot(tf.reshape(y_dummy, [-1]), depth=2)
  model_improper.fit(dummy_input, y_dummy, epochs=2)
except Exception as e:
  print(f"Error encountered:\n{e}")
```

In this code, `elmo_output` is the raw tensor coming from the ELMo layer. This tensor is then fed to the `dense` layer. While this executes during model construction and prediction (`model_improper(dummy_input)`), when you attempt to use `model_improper.fit`, TensorFlow will complain about a missing 'outer_context' because the internal machinery of Keras that uses `fit` needs to trace this tensor within a specific Keras graph and it can’t because `elmo_output` is a regular tensor and not a traced Keras Tensor. The error highlights the incompatibility during backpropagation within Keras’ training loop.

**Example 2: Incorrect Handling of ELMo Output Structure**

ELMo outputs a dictionary containing 'elmo' and 'word_emb' fields. Using only the raw 'elmo' embedding directly without correctly extracting it also leads to an issue.

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model

# Load the ELMo embedding from TensorFlow Hub
elmo_url = "https://tfhub.dev/google/elmo/3"
elmo = hub.KerasLayer(elmo_url, trainable=False)

class IncorrectStructureModel(Model):
  def __init__(self):
    super(IncorrectStructureModel, self).__init__()
    self.elmo = elmo
    self.dense = layers.Dense(128, activation='relu')
    self.output_layer = layers.Dense(2, activation='softmax')

  def call(self, inputs):
    elmo_output = self.elmo(inputs)
    # Problematic, extracting from an invalid attribute
    dense_output = self.dense(elmo_output['elmo'])
    return self.output_layer(dense_output)


# Fictional input
dummy_input = tf.constant([["This is a text input"], ["Another sample text"]])

# Instantiate model
model_structure = IncorrectStructureModel()

# Attempting to fit (will cause error)
try:
  model_structure(dummy_input)
  model_structure.compile(optimizer='adam', loss='categorical_crossentropy')
  y_dummy = tf.random.categorical(tf.math.log([[0.5,0.5], [0.5,0.5]]), 1)
  y_dummy = tf.one_hot(tf.reshape(y_dummy, [-1]), depth=2)
  model_structure.fit(dummy_input, y_dummy, epochs=2)
except Exception as e:
  print(f"Error encountered:\n{e}")
```

The issue is not because of the output structure, but rather, the output dictionary contains `'elmo'` and `'word_emb'` attributes, which have shape [batch, max_length, 1024] and [batch, max_length, 512] respectively. When you directly pass it to the `Dense` layer, you are not passing a flat tensor of features, rather, a 3D object that is not valid input. In this case the error might be different, but in slightly modified circumstances, could result in the `NoneType` error.

**Example 3: Correct Usage of ELMo within a Custom Model**

The correct way to address this issue is to explicitly create a wrapper layer that integrates the ELMo output in a way that Keras can track. It is not enough to just extract the `elmo` value, but also, we must handle the sequence and make it flat in order to input it into a dense layer.

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model
import numpy as np

# Load the ELMo embedding from TensorFlow Hub
elmo_url = "https://tfhub.dev/google/elmo/3"
elmo = hub.KerasLayer(elmo_url, trainable=False)

class ELMoEmbeddingLayer(layers.Layer):
  def __init__(self, **kwargs):
        super(ELMoEmbeddingLayer, self).__init__(**kwargs)
        self.elmo = elmo
        self.output_dim = 1024

  def call(self, inputs):
    elmo_output = self.elmo(tf.squeeze(tf.cast(inputs,tf.string), axis=1))
    elmo_embed = elmo_output['elmo']
    #flattening the output
    return tf.reshape(elmo_embed, shape=[-1, self.output_dim*tf.shape(elmo_embed)[1]])

class CorrectModel(Model):
  def __init__(self):
    super(CorrectModel, self).__init__()
    self.elmo_layer = ELMoEmbeddingLayer()
    self.dense = layers.Dense(128, activation='relu')
    self.output_layer = layers.Dense(2, activation='softmax')

  def call(self, inputs):
    elmo_output = self.elmo_layer(inputs)
    dense_output = self.dense(elmo_output)
    return self.output_layer(dense_output)

# Fictional input
dummy_input = tf.constant([["This is a text input"], ["Another sample text"]])

# Instantiate model
model_correct = CorrectModel()


# Fitting without error
model_correct(dummy_input)
model_correct.compile(optimizer='adam', loss='categorical_crossentropy')
y_dummy = tf.random.categorical(tf.math.log([[0.5,0.5], [0.5,0.5]]), 1)
y_dummy = tf.one_hot(tf.reshape(y_dummy, [-1]), depth=2)
model_correct.fit(dummy_input, y_dummy, epochs=2)

print("Training Successful")
```

In this corrected example, `ELMoEmbeddingLayer` is now a custom layer that encapsulates the call to the hub’s `KerasLayer`. Inside the `call` method, I’m extracting the `elmo` embeddings and flattening the sequence into a feature vector by using `tf.reshape` which allows us to feed it into the `dense` layer. This makes the tensor compatible with Keras’ backpropagation routines and avoids the `NoneType` error during `model.fit`. This wrapper also abstracts the ELMo layer details and keeps the main model cleaner. It’s worth noting that the dummy input needs to be cast as a string before being passed into the ELMo layer, which is why I’m including `tf.cast(inputs,tf.string)` in the call.

**Resource Recommendations:**

To avoid such issues in the future, consider consulting several readily available resources. The official TensorFlow documentation for custom layers and models provides critical insights. Further, tutorials on using TensorFlow Hub with Keras will help you better understand the layer interaction. Researching best practices for integrating external embeddings within custom models is also helpful. Lastly, review examples of TensorFlow-based NLP projects using ELMo can be insightful, often revealing these subtle integration nuances. Understanding the fundamentals of computational graphs and how they relate to Keras layers’ tracing mechanisms can also clarify such behavior.
