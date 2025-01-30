---
title: "How to fix a TypeError when saving a Keras model?"
date: "2025-01-30"
id: "how-to-fix-a-typeerror-when-saving-a"
---
The `TypeError` encountered during Keras model saving typically arises from inconsistencies between the model's architecture, its trainable weights, and the serialization method employed. Specifically, the issue often manifests when the model or its components are not inherently serializable using standard Python pickling or JSON formats. I've personally wrestled with this numerous times, most notably while fine-tuning a custom CNN with a complex layer built using TensorFlow operations.

The core problem resides in the discrepancy between what Keras' `save()` function expects (a serializable model representation) and what the model actually contains. Keras models are fundamentally composed of layers, activation functions, and optimization parameters, all neatly packed into TensorFlow graphs. These graphs, when fully defined, are readily convertible into a representation that can be stored on disk, then reconstructed later. However, when non-standard components are introduced, such as custom layers inheriting directly from `tf.keras.layers.Layer` but lacking the necessary implementation for the `get_config()` method, the default serialization process falters. This results in the `TypeError`, because the saving mechanism cannot understand how to convert these elements into storable bytes.

The standard `model.save()` function relies heavily on Python's built-in pickling mechanism and JSON serialization when it encounters components that do not support pickling. When a custom object lacking necessary serialization specifications is present in a component used by Keras, it raises the `TypeError`. This occurs because `pickle` or `JSON` can not interpret objects it doesn’t recognize as serializable. The `get_config()` method of custom layers provides instructions on how to reconstitute a given layer from a stored configuration. If this method is missing, the serialization process has no blueprint on which to rely.

There are several remedies. The most direct and generally effective approach involves ensuring that all custom layers and components are equipped with `get_config()` and `from_config()` methods. The `get_config()` method should return a dictionary containing all necessary attributes and arguments required to recreate the layer instance. The `from_config()` class method should be defined to take the dictionary returned by `get_config()` and instantiate a fresh instance of that class with these passed configurations. This allows Keras to save and restore custom components correctly by serializing their configurations instead of the entire python object. Additionally, it is generally recommended to set up model saving to the SavedModel format which uses a more robust mechanism for model export. It addresses potential issues when saving certain components such as custom layers and custom training loops that aren't as easy to pickle.

Here are three code examples illustrating common scenarios and the associated fixes:

**Example 1: Custom Layer Without Serialization Methods**

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer="random_normal", trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer="zeros", trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(10,)),
  MyCustomLayer(units=5)
])

try:
  model.save("my_model.keras") # This will raise a TypeError
except TypeError as e:
  print(f"Error saving model: {e}")
```

In this example, the `MyCustomLayer` lacks the necessary `get_config()` and `from_config()` methods. Executing `model.save()` results in a `TypeError`. Keras doesn't know how to serialize `MyCustomLayer`, since it is not recognized by its serialization process.

**Example 2: Custom Layer with Proper Serialization**

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer="random_normal", trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer="zeros", trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config

  @classmethod
  def from_config(cls, config):
        return cls(**config)


model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(10,)),
  MyCustomLayer(units=5)
])

model.save("my_model_corrected.keras") # This will save the model successfully
```

Here, I’ve added the `get_config()` and `from_config()` methods to the `MyCustomLayer`. `get_config` creates a dictionary of necessary parameters for a new instance, and `from_config` uses this dictionary to initialize a new instance from it. Now, the `model.save()` call executes without error, since the Keras framework can serialize and rebuild this layer.

**Example 3: Saving in SavedModel Format**

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer="random_normal", trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer="zeros", trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config

  @classmethod
  def from_config(cls, config):
        return cls(**config)


model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(10,)),
  MyCustomLayer(units=5)
])

model.save("my_saved_model", save_format="tf")  # Using SavedModel format
```

In this example, the same custom layer from Example 2 is used, but instead of saving to the h5 format, we save using the SavedModel format. This is generally preferred for production models as it uses a more flexible approach to model storage, and avoids some of the pickle based serialization problems. As this method does not rely on pickle for serialization, problems encountered with a model using custom layers which are not handled with the above code can be often bypassed simply by using this method.

For further reading and instruction, I would advise consulting the official Keras documentation, which provides comprehensive information regarding custom layers and serialization. Additionally, the TensorFlow documentation covering the `tf.saved_model` module can be highly beneficial for understanding the intricacies of the SavedModel format. Finally, there are numerous excellent machine learning books which discuss how to design custom layers for neural networks and which go into detail regarding the construction of more sophisticated models. These resources provide a more generalized understanding, and are useful in approaching more specific and complex model implementations.
