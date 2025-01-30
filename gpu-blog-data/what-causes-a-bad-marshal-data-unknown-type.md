---
title: "What causes a 'bad marshal data (unknown type code)' error in Keras?"
date: "2025-01-30"
id: "what-causes-a-bad-marshal-data-unknown-type"
---
The "bad marshal data (unknown type code)" error in Keras, especially when working with model saving and loading, typically arises from a mismatch in the serialization and deserialization process of custom objects, primarily those defined outside the standard Keras library, during a model's lifecycle. Specifically, the pickling mechanism used by Keras for saving and loading models cannot interpret the bytecode representations of these custom, unregistered elements during restoration. This often happens when a model containing custom layers, loss functions, metrics, or other custom objects saved in one environment attempts to be loaded in another environment where those custom objects are either missing or registered differently.

The error isn't directly a Keras bug; it stems from Python's standard library's `pickle` module, which is internally used by Keras for serializing model architectures and weights. `Pickle` converts Python objects into a byte stream representation. The core problem is that when custom Python classes (like custom layers) are encountered, `pickle` stores bytecode instructions for how to rebuild that object, along with its attributes. If, upon loading, the same precise environment, including the class definition, isn't present, or if `pickle` cannot find the specific class at the defined location, the “bad marshal data” error can occur. The error message "unknown type code" is particularly informative because it highlights that the `pickle` module has encountered a bytecode representation of a type it doesn't know how to handle during unpickling.

A frequently encountered case leading to this error is when a user defines a custom layer within the main script, trains the model, and saves it. Later, when loading the model in a separate script (or another environment) where the custom layer definition isn’t present, the pickle process fails because it attempts to locate and instantiate the custom layer, but cannot. This failure is a direct result of the mismatch between the environment in which the model was saved and the environment in which the model is loaded. Another scenario is when custom functions or classes are modified or reordered between saving and loading. Even a seemingly minor change to the custom component's definition can alter its bytecode representation, leading to unpickling issues. This highlights the brittle nature of relying on implicit or unregistered custom objects.

To mitigate this problem, Keras offers mechanisms for correctly serializing custom objects using `tf.keras.utils.register_keras_serializable`. This decorator, used in conjunction with custom class definitions, allows Keras to correctly register the custom component and its associated constructor and configuration details, allowing the Keras serialization system to handle it appropriately. We need to register both the class itself and any required configuration details to ensure seamless saving and loading. Essentially, the `register_keras_serializable` annotation informs Keras of the custom object and how to reconstruct it properly at load time. Without it, Keras's serialization will only preserve pointers to the objects, and if they do not exist at the loading location, the process fails.

Below are three code examples demonstrating the problem and its solution.

**Example 1: The Problem – Unregistered Custom Layer**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Defining a custom layer without registration
class CustomDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
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

# Creating and training a model
model = keras.Sequential([
    layers.Input(shape=(10,)),
    CustomDense(32),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
x = tf.random.normal((100, 10))
y = tf.random.normal((100, 1))
model.fit(x, y, epochs=2)

# Save model
model.save('my_model.h5')

#Attempt to load model in a new scope. This will throw an error
#with tf.keras.models.load_model('my_model.h5') as m: # This will fail

#The new scope would be a separate python file where CustomDense is not defined.
#The above attempted load would throw a bad marshal data error
```

This example shows the scenario described earlier. We define `CustomDense` but don't register it, which results in pickle trying to save a reference to its definition. Consequently, when we load the model in a different environment where `CustomDense` is undefined, it raises a “bad marshal data” error. This showcases the core problem: The lack of a mechanism for Keras to understand how to reconstruct the object. This will fail when the file `my_model.h5` is loaded into an environment where `CustomDense` is not defined.

**Example 2: The Solution – Registering Custom Layer**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# Defining a custom layer and registering it
@register_keras_serializable(package="my_custom_layers")
class CustomDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
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

    def get_config(self):
      config = super().get_config()
      config.update({
          'units': self.units
      })
      return config


# Creating and training a model
model = keras.Sequential([
    layers.Input(shape=(10,)),
    CustomDense(32),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
x = tf.random.normal((100, 10))
y = tf.random.normal((100, 1))
model.fit(x, y, epochs=2)

# Save model
model.save('my_model_registered.h5')

# Load model and this will work correctly
loaded_model = keras.models.load_model('my_model_registered.h5',
                        custom_objects={'CustomDense': CustomDense}) # This will work fine
print("Model loaded successfully")
```

In this improved version, we use the `@register_keras_serializable(package="my_custom_layers")` decorator to register `CustomDense`. Additionally, we define a `get_config` method to serialize any parameters defined in the init method. Now, `pickle` knows how to serialize and deserialize `CustomDense` correctly. This decorator tells Keras to store a registration token for the class, and allows Keras to correctly reconstruct the object using the class definition available from the loading environment. The `custom_objects` parameter in the load method is still required to ensure the layer is properly associated with the model graph. The `package` parameter helps to prevent naming conflicts when dealing with multiple custom elements.  Without these, the problem would return.

**Example 3: Custom Loss Function**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# Custom Loss Function
@register_keras_serializable(package="my_custom_losses")
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Custom Layer remains registered as in the previous example.
@register_keras_serializable(package="my_custom_layers")
class CustomDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
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

    def get_config(self):
      config = super().get_config()
      config.update({
          'units': self.units
      })
      return config

# Creating and training a model
model = keras.Sequential([
    layers.Input(shape=(10,)),
    CustomDense(32),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss=custom_loss)
x = tf.random.normal((100, 10))
y = tf.random.normal((100, 1))
model.fit(x, y, epochs=2)

# Save model
model.save('my_model_custom_loss.h5')

# Load model.
loaded_model = keras.models.load_model('my_model_custom_loss.h5', custom_objects={'custom_loss': custom_loss, 'CustomDense': CustomDense})
print("Model with custom loss loaded successfully.")

```

This final example extends the solution to include a custom loss function. The same principles apply: We must decorate the custom loss function with `@register_keras_serializable(package="my_custom_losses")`. As demonstrated, custom objects – whether layers or functions – require explicit registration and configuration serialization to ensure proper preservation during save and loading operations. Also shown, a specific parameter `custom_objects` must also be passed into the loading function and is critical to enable Keras to load those custom elements correctly.

Regarding resource recommendations for further exploration, I would suggest consulting the official TensorFlow and Keras documentation. Specifically, focus on the sections covering custom layers, serialization, the `register_keras_serializable` decorator, and saving and loading models. Additionally, reviewing examples in the Keras repository demonstrating best practices for custom component registration and serialization will provide additional context. Books or tutorials that cover advanced Keras concepts and custom implementations could provide insights as well. Remember that understanding the interplay between `pickle`, custom object registration, and Keras' internal model handling is the key to avoiding this type of error.
