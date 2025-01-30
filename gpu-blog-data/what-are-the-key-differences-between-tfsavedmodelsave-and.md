---
title: "What are the key differences between `tf.saved_model.save` and `tf.keras.models.save`?"
date: "2025-01-30"
id: "what-are-the-key-differences-between-tfsavedmodelsave-and"
---
When transitioning from simple Keras models to production-ready TensorFlow deployments, understanding the nuances between `tf.saved_model.save` and `tf.keras.models.save` is crucial because they handle model persistence differently, directly impacting deployment flexibility and potential for further processing. I’ve encountered scenarios where incorrectly chosen save methods led to deployment issues, requiring significant rework and impacting project deadlines. Specifically, I initially defaulted to `tf.keras.models.save` because of its ease, only to later struggle when integrating custom layers and operations outside the standard Keras ecosystem.

`tf.saved_model.save` is the recommended approach for exporting TensorFlow models intended for deployment, including those built with Keras. This function is part of TensorFlow's lower-level API and it saves the model as a SavedModel format, encompassing the model graph, weights, and associated signatures. Signatures are vital, as they define the input and output tensors and the computational graph associated with them. They are essentially a contract for how to interact with the model later, crucial when the model is loaded in environments different from the training environment, like TensorFlow Serving or TF Lite. `tf.saved_model.save` enables features such as custom operations and custom training loops. Moreover, this saving method is intended for model versioning and provides broader compatibility options for model consumption. It does not inherently rely on Keras infrastructure when loading, which allows flexibility and prevents issues associated with specific Keras versions. Essentially, `tf.saved_model.save` facilitates a more flexible, deployment focused approach.

In contrast, `tf.keras.models.save` offers a simpler, Keras-centric method. This function primarily targets easy preservation of model architectures and weights, and is often used for intermediate checkpoints during training or saving models for later training. When saving a Keras model, the function can leverage the Keras configuration to save both the model architecture as well as the model's weights, as serialized metadata. This approach is suitable for sharing models within a Keras-based environment and for quick prototyping, particularly when the model does not require non-Keras operations or is not deployed to environments outside Keras. However, it can be limited with custom layers or if it has external graph operations because `tf.keras.models.save` will rely on Keras internal representation. When loaded, the Keras framework itself is required to reconstruct the model using the metadata. This can become problematic if the precise version of Keras used for the save operation is not available. The format of the save can be either HDF5 or the TensorFlow SavedModel format. In recent versions, when using the `save()` method of Keras, you might notice that both formats can be saved. However, there is a key distinction: the Keras `save()` will still attempt to save the Keras model configuration, with the TensorFlow SavedModel format being only an option; not the intrinsic manner of saving.

Let's explore three scenarios, each illustrating the application and limitation of these save methods:

**Example 1: Basic Keras Model Saved via `tf.keras.models.save`**

```python
import tensorflow as tf
from tensorflow import keras

# Define a simple Keras model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Save the model using tf.keras.models.save in HDF5 format.
model.save('my_keras_model.h5')

# Load the model and verify its state
loaded_model = keras.models.load_model('my_keras_model.h5')

# Simple assertion to verify loaded model is of expected configuration.
assert model.layers[0].units == loaded_model.layers[0].units

print("Keras Model saved using tf.keras.models.save, and loaded back successfully.")
```

This example demonstrates a straightforward application of `tf.keras.models.save`. I've saved a basic Keras model in HDF5 format and then loaded it using `keras.models.load_model`. This works seamlessly within the Keras environment. However, this model is dependent on the Keras framework to reconstruct, and is therefore less flexible for cross-platform usage.

**Example 2: Basic Keras Model Saved via `tf.saved_model.save`**

```python
import tensorflow as tf
from tensorflow import keras

# Define a simple Keras model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Save the model using tf.saved_model.save
tf.saved_model.save(model, 'my_saved_model')

# Load the model using tf.saved_model.load
loaded_model = tf.saved_model.load('my_saved_model')

# Retrieve the inferencing function (signature)
infer = loaded_model.signatures["serving_default"]

# Create a random input as a test
test_input = tf.random.normal(shape=(1, 784))

# Verify the loaded model runs and outputs a result.
test_output = infer(test_input)
assert test_output is not None

print("Keras Model saved using tf.saved_model.save, and loaded back successfully.")
```
Here, I save the same Keras model using `tf.saved_model.save` in the SavedModel format. The main difference during loading is that I use `tf.saved_model.load` and then I access the model via the "serving\_default" signature. This approach provides a more robust representation of the model that can be utilized in various deployment scenarios, independent of the specific Keras version. Note I am loading the model using the TensorFlow framework and interacting with it via signatures, which allows flexibility of usage in environments not dependent on the Keras framework.

**Example 3: Model with Custom Layer saved via `tf.saved_model.save`**
```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

class CustomLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Define a model with the custom layer
model_with_custom = keras.Sequential([
    keras.Input(shape=(10,)),
    CustomLayer(100),
    layers.Dense(10, activation='softmax')
])

# Save the model using tf.saved_model.save
tf.saved_model.save(model_with_custom, 'my_saved_model_custom')

# Load the model using tf.saved_model.load
loaded_custom_model = tf.saved_model.load('my_saved_model_custom')

# Retrieve the inferencing function (signature)
infer = loaded_custom_model.signatures["serving_default"]

# Create a random input as a test
test_input = tf.random.normal(shape=(1, 10))

# Verify the loaded model runs and outputs a result.
test_output = infer(test_input)
assert test_output is not None

print("Model with custom layer saved using tf.saved_model.save, and loaded back successfully.")
```

This last example showcases the strength of `tf.saved_model.save`. By saving a model that includes a custom layer, `tf.saved_model.save` manages to encapsulate the custom layer's logic within the SavedModel format which makes it available without relying on any particular Keras configurations. Loading and interaction with the model is also consistent with the previous example. This demonstrates why, especially when one integrates custom operations, `tf.saved_model.save` is the preferred method for deployment. While the `tf.keras.models.save` function could be coerced to serialize the custom layer by ensuring the custom class is part of a custom Keras object, relying on this method can be brittle across Keras version upgrades, and introduces a dependency on loading Keras prior to use.

In conclusion, choosing between `tf.saved_model.save` and `tf.keras.models.save` should depend on the deployment needs and required model complexity. `tf.keras.models.save` works perfectly fine for quick prototyping and simple Keras models used within a consistent Keras environment, it can even be deployed as a SavedModel, but it will still include Keras config which can lead to potential version issues. However, if the aim is for broader deployment, and a higher degree of flexibility, especially for models with custom layers, operations, and deployment via TensorFlow serving, `tf.saved_model.save` is the ideal solution.

For further understanding, I recommend studying the official TensorFlow documentation on SavedModel format and Keras model saving. Additionally, exploring tutorials and examples that involve deploying models with TensorFlow Serving will clarify the practical implications of this distinction. Consulting specialized books on TensorFlow deployment engineering can also provide valuable insight into best practices when working with saved models.
