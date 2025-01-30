---
title: "How can I remove the last layer of a Keras subclass model while retaining its learned weights?"
date: "2025-01-30"
id: "how-can-i-remove-the-last-layer-of"
---
Detaching the final layer of a Keras subclass model while preserving its weights requires a careful manipulation of the model's architecture and subsequent weight transfer. Simply removing the layer using standard list operations on the model's layers attribute isn't sufficient; it erases both the layer and its learned parameters. Instead, one must extract the weights, reconstruct the model without the final layer, and then reassign those extracted weights to a newly constructed layer.

The challenge arises from the fact that Keras subclass models, unlike sequential or functional models, do not automatically maintain a linear representation of their layers. They are defined by a call method that may involve arbitrary code. Consequently, programmatically identifying the *last* layer is not trivial. The solution involves inspecting the model's `layers` attribute, which holds references to each instantiated layer. We will iteratively identify the last layer intended to be removed, extract its weights, build a new model architecture without this layer, and finally transfer the weights.

I've encountered this situation multiple times, particularly when fine-tuning pre-trained models or experimenting with different output layers for classification or regression tasks. The need to re-use existing convolutional backbones while changing downstream task heads makes this a very common requirement.

First, let's dissect the problem and define a suitable process. We begin by identifying the layer to be removed. While Keras doesn't inherently track the order of layer usage in the `call` method of a subclass model, the `.layers` attribute does preserve the instantiation order. This can be unreliable, but it’s generally sufficient for practical use. We'll assume the last layer in the list is what needs to be removed. Then, we extract the weights from this layer using `layer.get_weights()`, which provides a list of NumPy arrays. Next, we construct a new model instance using the same architecture, *excluding* the last layer. We typically achieve this by modifying the subclass `call` method to skip the instantiation of the unwanted final layer. Finally, we create a new layer with the same architecture as the removed layer and assign the extracted weights to it. This new layer is then attached to the new model. The weights will then be properly loaded into this layer.

Here's the process demonstrated through three code examples, increasing in complexity.

**Example 1: A Simple Fully-Connected Model**

This example demonstrates weight transfer from a simple model comprising two fully connected layers.

```python
import tensorflow as tf
import numpy as np

class SimpleModel(tf.keras.Model):
    def __init__(self, num_units_1=64, num_units_2=10, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_units_2)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Create a sample model and input
model = SimpleModel(num_units_1=32, num_units_2=5)
dummy_input = tf.random.normal((1, 100))
output = model(dummy_input) # Initialize the weights
print("Initial Model Output Shape:", output.shape)

# Remove the last layer and transfer weights
last_layer = model.layers[-1]
weights = last_layer.get_weights()

# Build a new model omitting the last layer
class SimpleModelModified(tf.keras.Model):
    def __init__(self, num_units_1=32, **kwargs):
      super().__init__(**kwargs)
      self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')

    def call(self, inputs):
      return self.dense1(inputs)

new_model = SimpleModelModified(num_units_1=32)
_ = new_model(dummy_input) # Initialize the weights
print("Modified Model Output Shape:", new_model(dummy_input).shape)


# Add a new compatible layer and load weights
new_layer = tf.keras.layers.Dense(5)
new_layer.build(input_shape=(None, 32))
new_layer.set_weights(weights)

# Attach this layer to the new model to extend it
class ExtendedModel(tf.keras.Model):
    def __init__(self, base_model, new_layer, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.new_layer = new_layer

    def call(self, inputs):
        x = self.base(inputs)
        return self.new_layer(x)

extended_model = ExtendedModel(new_model, new_layer)
output_extended = extended_model(dummy_input)

print("Extended Model Output Shape:", output_extended.shape)
```

Here, we extract the `dense2` layer's weights, create a new model `SimpleModelModified` that omits this layer, and then a new `Dense` layer with the same structure. We use `set_weights` to transfer the extracted weights and integrate it to the new model using `ExtendedModel`. This demonstrates the essential transfer and reconstruction process. The crucial operation is the `layer.set_weights` method, which performs the transfer itself. Notice that we initialize the layer shapes with `build` before assigning weights.

**Example 2: A Model with a Batch Normalization Layer**

This example adds a batch normalization layer between the fully connected layers, which requires more care because such layers hold additional parameters (moving means and variances) alongside the standard weights.

```python
import tensorflow as tf
import numpy as np

class BatchNormModel(tf.keras.Model):
    def __init__(self, num_units_1=64, num_units_2=10, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(num_units_2)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn(x, training=training)
        return self.dense2(x)

# Create a sample model and input
model = BatchNormModel(num_units_1=32, num_units_2=5)
dummy_input = tf.random.normal((1, 100))
output = model(dummy_input) # Initialize the weights
print("Initial Model Output Shape:", output.shape)

# Remove the last layer and transfer weights
last_layer = model.layers[-1]
weights = last_layer.get_weights()

# Build a new model omitting the last layer
class BatchNormModelModified(tf.keras.Model):
    def __init__(self, num_units_1=32, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.bn(x, training=training)

new_model = BatchNormModelModified(num_units_1=32)
_ = new_model(dummy_input) # Initialize the weights
print("Modified Model Output Shape:", new_model(dummy_input).shape)


# Add a new compatible layer and load weights
new_layer = tf.keras.layers.Dense(5)
new_layer.build(input_shape=(None, 32))
new_layer.set_weights(weights)


# Attach this layer to the new model to extend it
class ExtendedModelBN(tf.keras.Model):
    def __init__(self, base_model, new_layer, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.new_layer = new_layer

    def call(self, inputs, training=False):
        x = self.base(inputs, training=training)
        return self.new_layer(x)

extended_model = ExtendedModelBN(new_model, new_layer)
output_extended = extended_model(dummy_input)
print("Extended Model Output Shape:", output_extended.shape)
```

This example highlights the necessity of considering internal layer parameters when removing the final layer. The `BatchNormalization` layer introduces moving averages, which need to be accounted for when doing transfer learning with complex architectures. We correctly transfer just the weights of the final Dense layer, leaving batch normalization to be trainable by the new model structure.

**Example 3: A Model with a Custom Layer**

Finally, we extend this to the case of a custom Keras layer, demonstrating the versatility of this method.

```python
import tensorflow as tf
import numpy as np

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
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

class CustomModel(tf.keras.Model):
    def __init__(self, num_units_1=64, num_units_2=10, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')
        self.custom_layer = CustomLayer(num_units_2)


    def call(self, inputs):
        x = self.dense1(inputs)
        return self.custom_layer(x)


# Create a sample model and input
model = CustomModel(num_units_1=32, num_units_2=5)
dummy_input = tf.random.normal((1, 100))
output = model(dummy_input)  # Initialize weights
print("Initial Model Output Shape:", output.shape)


# Remove the last layer and transfer weights
last_layer = model.layers[-1]
weights = last_layer.get_weights()


# Build a new model omitting the last layer
class CustomModelModified(tf.keras.Model):
    def __init__(self, num_units_1=32, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_units_1, activation='relu')


    def call(self, inputs):
        return self.dense1(inputs)

new_model = CustomModelModified(num_units_1=32)
_ = new_model(dummy_input)  # Initialize weights
print("Modified Model Output Shape:", new_model(dummy_input).shape)

# Add a new compatible layer and load weights
new_layer = CustomLayer(5)
new_layer.build(input_shape=(None, 32))
new_layer.set_weights(weights)

# Attach this layer to the new model to extend it
class ExtendedModelCustom(tf.keras.Model):
    def __init__(self, base_model, new_layer, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.new_layer = new_layer

    def call(self, inputs):
        x = self.base(inputs)
        return self.new_layer(x)

extended_model = ExtendedModelCustom(new_model, new_layer)
output_extended = extended_model(dummy_input)
print("Extended Model Output Shape:", output_extended.shape)
```

Here, we've successfully transferred the weights of a custom layer. Note that we use the *same* custom layer definition both in the original model and the new, attached layer, because the `set_weights` function relies on the new layer having the same structure as the source layer, requiring a compatible build method.

In conclusion, detaching the final layer of a Keras subclass model while maintaining its weights necessitates an understanding of how Keras stores layers, along with careful extraction, reconstruction, and assignment procedures. The `layer.get_weights()` and `layer.set_weights()` methods are central to this process. For further understanding of custom layers and weight management, consult the official Keras documentation and advanced tutorials. Additionally, research transfer learning techniques and the use of pre-trained models within the context of Keras. Understanding the layer API, especially concerning the `build` and `call` functions, is crucial for this task.
