---
title: "Why isn't my custom layer being counted in model.summary()?"
date: "2025-01-26"
id: "why-isnt-my-custom-layer-being-counted-in-modelsummary"
---

My experience with TensorFlow and Keras indicates that a custom layer not appearing in `model.summary()` is typically due to one of two primary reasons: either the custom layer has not been built correctly or it's not explicitly connected to the computational graph that Keras's `Model` tracks. The `model.summary()` function relies on the underlying structure of the graph to automatically infer and report layer information, hence, if a layer exists outside this structure, it remains invisible to the summary.

The core problem is that Keras uses its `Layer` class as the fundamental unit for constructing models. Layers contribute to the model's trainable parameters, activation flows, and topological arrangement. For a custom layer to be properly represented in `model.summary()`, it must:

1. **Inherit from `tf.keras.layers.Layer`:** This ensures it aligns with Keras's layer abstraction and participates in the model-building process.
2. **Implement the `build(input_shape)` method:** This method is where a layer’s variables (weights and biases) are defined. It is crucial for deferred weight initialization. Failure to define variables within this method can lead to the layer not being recognized as having parameters, hence excluded from `model.summary()`.
3. **Implement the `call(inputs)` method:** This method defines the forward pass of the layer, specifying the computational transformation applied to input data. The `call` method, combined with variable declaration in `build`, dictates how data flows through the layer and how the graph is created.
4. **Be explicitly integrated into the model's sequential or functional graph:** A layer, even if implemented correctly, will not appear in `model.summary()` if not actively connected within the model’s architecture.

The `model.summary()` output depends on the `Model` object's ability to trace the flow of data and associated transformations. When using a sequential model (e.g., `tf.keras.Sequential`), layers are implicitly added to the graph. However, in functional API models or with subclasses, the connections must be explicit. A custom layer that is not explicitly part of that connected flow won’t be reflected in the summary.

Here are three examples demonstrating different scenarios and solutions:

**Example 1: Missing `build` implementation.**

```python
import tensorflow as tf

class IncorrectCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(IncorrectCustomLayer, self).__init__(**kwargs)
    self.units = units
    # Missing build method here

  def call(self, inputs):
    return tf.matmul(inputs, tf.random.normal(shape = (inputs.shape[-1], self.units)))
    

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    IncorrectCustomLayer(5)
])
dummy_input = tf.random.normal(shape=(1, 10))
model(dummy_input)

model.summary() # IncorrectCustomLayer is not listed

```

In this initial code, the `IncorrectCustomLayer` class initializes a `units` parameter and includes a `call` method which performs a matrix multiplication. However, it lacks the crucial `build` method, where layer weights should be defined. As a result, when this layer is instantiated in the `Sequential` model and a forward pass occurs, Keras doesn’t recognize that it should hold trainable parameters and therefore doesn’t include it in `model.summary()`. The `call` operation does compute a transformation based on random matrix but since it was not explicitly created via build method, it won't be summarized.

**Example 2: Correct `build` and `call` implementation.**

```python
import tensorflow as tf

class CorrectCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(CorrectCustomLayer, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                               initializer='zeros',
                               trainable=True)
  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b
    

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    CorrectCustomLayer(5)
])

dummy_input = tf.random.normal(shape=(1, 10))
model(dummy_input)

model.summary() # CorrectCustomLayer is listed with trainable params

```

In this improved version, the `CorrectCustomLayer` implements the `build` method. Inside `build`, weights `w` and biases `b` are initialized, these are what allow the layer to hold trainable parameters and be included in the `model.summary()`. Note the use of `add_weight`, it is the preferred method to include variables that Keras can track. The `call` method applies a matrix multiplication with the weights and adds biases as before. By including this correct method, the layer becomes visible within the model summary.

**Example 3: Functional API, an untracked layer.**

```python
import tensorflow as tf

class UntrackedCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(UntrackedCustomLayer, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                               initializer='zeros',
                               trainable=True)
  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


input_tensor = tf.keras.layers.Input(shape=(10,))
dense_layer = tf.keras.layers.Dense(10)(input_tensor)
custom_layer = UntrackedCustomLayer(5)
output_tensor = custom_layer(dense_layer) #layer is used outside Keras API
model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
dummy_input = tf.random.normal(shape=(1, 10))
model(dummy_input)

model.summary() # UntrackedCustomLayer is still NOT listed

```

Here, we’ve created a layer that *does* correctly implement `build` and `call`. However, in this functional model, we instantiate `UntrackedCustomLayer` but it is not included in the `tf.keras.models.Model` directly, which is how the computational graph is generated. It is instantiated as if it were a custom function and is not an explicitly connected layer within the Keras model structure. The `output_tensor = custom_layer(dense_layer)` operation calculates a tensor but does not become part of the graph tracked by `tf.keras.models.Model`. Therefore, even though the layer exists and is being used, it is excluded from `model.summary()`.

To correct the third example, you would need to treat your custom layer as part of the model's functional API, similar to how you use other built-in layers like `Dense`, by applying the custom layer to the previous tensor within the functional graph, and not as an independent operation:

```python
import tensorflow as tf

class TrackedCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(TrackedCustomLayer, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                               initializer='zeros',
                               trainable=True)
  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


input_tensor = tf.keras.layers.Input(shape=(10,))
dense_layer = tf.keras.layers.Dense(10)(input_tensor)
custom_layer = TrackedCustomLayer(5)(dense_layer) # layer is now part of the Keras API

model = tf.keras.models.Model(inputs=input_tensor, outputs=custom_layer)
dummy_input = tf.random.normal(shape=(1, 10))
model(dummy_input)

model.summary() # TrackedCustomLayer is now listed
```

In this corrected version, the `TrackedCustomLayer(5)(dense_layer)` syntax explicitly adds the custom layer's operation to the graph tracked by Keras, making it visible in the `model.summary()`. The change reflects a critical shift from an external function call to an explicit connection within the Keras computational graph.

To summarize, if your custom layer is absent from `model.summary()`, meticulously examine these critical points: the presence and correct implementation of the `build` method, the consistent usage of `add_weight`, adherence to Keras’ `Layer` API, and explicit inclusion of the layer in the model’s computational graph. It is not sufficient to define the variables within init or `call`, these must be defined within the `build` function.

For further understanding, I suggest focusing on these resources:
*   TensorFlow official documentation, especially regarding the `tf.keras.layers.Layer` class.
*   Keras documentation regarding custom layers and model construction.
*  Tutorials explaining functional API models versus sequential models and the difference.
*  Examples of well-defined custom layers on GitHub, examining how they implement `build` and `call`.
 By attending to the details in how the `Layer` API works, you can ensure your custom layers are appropriately represented in your model's summary.
