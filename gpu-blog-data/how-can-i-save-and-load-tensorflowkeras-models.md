---
title: "How can I save and load TensorFlow/Keras models containing custom classes/functions?"
date: "2025-01-30"
id: "how-can-i-save-and-load-tensorflowkeras-models"
---
TensorFlow’s model saving and loading mechanisms, especially when incorporating custom elements like layers or loss functions, necessitate a nuanced approach beyond the typical `model.save()` and `tf.keras.models.load_model()` workflow. These custom components, inherently non-standard within the library, require explicit handling during the serialization and deserialization processes. Overlooking this detail frequently leads to errors such as "UnknownObject" or issues where the loaded model fails to function correctly, thus emphasizing the importance of defining custom serialization logic. I’ve encountered this hurdle numerous times, particularly in research projects where highly specialized network architectures demanded tailored components, and developing reliable save/load routines became integral to reproducible workflows.

The primary challenge stems from the fact that TensorFlow's default saving routines primarily capture graph structures and weights. They do not inherently understand how to represent and reconstruct custom Python objects. Thus, to incorporate these elements, we must either implement custom serialization and deserialization mechanisms within the saved model format (using `get_config()` and `from_config()`) or manage the objects externally and re-inject them upon loading. The former approach is generally preferred, ensuring that all model components are encapsulated within the saved file, simplifying deployment and portability. This strategy relies on extending Keras objects (e.g., layers, metrics, losses) to provide the necessary metadata for reconstruction.

Let's explore specific approaches through code examples. First, consider a scenario involving a custom activation function. Standard activations are usually handled seamlessly; however, when a custom mathematical transformation is needed, we must provide TensorFlow with the necessary steps to rebuild it.

**Example 1: Custom Activation Function**

```python
import tensorflow as tf
import numpy as np

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return self.scale * tf.math.sigmoid(inputs)

    def get_config(self):
        config = super(CustomActivation, self).get_config()
        config.update({'scale': self.scale})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model_with_custom_activation():
    inputs = tf.keras.layers.Input(shape=(10,))
    x = tf.keras.layers.Dense(16)(inputs)
    x = CustomActivation(scale=2.0)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Model creation and saving:
model_1 = build_model_with_custom_activation()
model_1.save('custom_activation_model')

# Model loading:
loaded_model_1 = tf.keras.models.load_model('custom_activation_model')

# Verification (forward pass):
test_input = np.random.rand(1, 10)
output_original = model_1(test_input)
output_loaded = loaded_model_1(test_input)
assert np.allclose(output_original.numpy(), output_loaded.numpy())
print("Model with custom activation loaded successfully")
```

In this example, `CustomActivation` inherits from `tf.keras.layers.Layer` and implements `get_config()` and `from_config()`.  `get_config()` stores the instantiation parameter 'scale', and `from_config()` recreates the object during loading. When `tf.keras.models.load_model` encounters this layer, it uses the provided methods for reconstitution. Without these functions, the load operation will fail. This technique generalizes to other custom Keras objects.

Next, let’s address a custom loss function, an area where subtle errors are common due to misaligned handling of object attributes.

**Example 2: Custom Loss Function**

```python
import tensorflow as tf
import numpy as np

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, factor=0.5, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.factor = factor

    def call(self, y_true, y_pred):
        return self.factor * tf.reduce_mean(tf.square(y_true - y_pred))

    def get_config(self):
      config = super().get_config()
      config.update({"factor":self.factor})
      return config

    @classmethod
    def from_config(cls, config):
       return cls(**config)

def build_model_with_custom_loss():
    inputs = tf.keras.layers.Input(shape=(10,))
    x = tf.keras.layers.Dense(16)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=CustomLoss(factor=0.8))
    return model

# Model creation, training, and saving:
model_2 = build_model_with_custom_loss()
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
model_2.fit(x_train, y_train, epochs=2)
model_2.save('custom_loss_model')

# Model loading and validation:
loaded_model_2 = tf.keras.models.load_model('custom_loss_model')
loss_original = model_2.evaluate(x_train, y_train, verbose=0)
loss_loaded = loaded_model_2.evaluate(x_train, y_train, verbose=0)
assert np.allclose(loss_original, loss_loaded)
print("Model with custom loss loaded successfully")

```

Here, `CustomLoss` inherits from `tf.keras.losses.Loss`. Similar to the previous example, `get_config()` and `from_config()` are implemented to persist the loss function’s ‘factor’. Crucially, we demonstrate training using this model *before* saving, which helps expose any errors in the custom loss implementation. The loaded model is then evaluated to confirm functionality. Improper implementation of  `get_config()` or `from_config()` frequently manifests with the loss not being applied correctly during training or evaluation after the model reload.

Finally, let's look at a more complex case: a custom layer containing trainable parameters.

**Example 3: Custom Layer with Trainable Parameters**

```python
import tensorflow as tf
import numpy as np

class CustomTrainableLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomTrainableLayer, self).__init__(**kwargs)
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
        config = super(CustomTrainableLayer, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_model_with_custom_trainable_layer():
    inputs = tf.keras.layers.Input(shape=(10,))
    x = CustomTrainableLayer(units=64)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Model creation, training, and saving:
model_3 = build_model_with_custom_trainable_layer()
x_train_3 = np.random.rand(100, 10)
y_train_3 = np.random.rand(100, 1)
model_3.compile(optimizer='adam', loss='mse')
model_3.fit(x_train_3, y_train_3, epochs=2, verbose=0)
model_3.save('custom_trainable_layer_model')

# Model loading and verification:
loaded_model_3 = tf.keras.models.load_model('custom_trainable_layer_model')
output_original_3 = model_3(x_train_3)
output_loaded_3 = loaded_model_3(x_train_3)
assert np.allclose(output_original_3.numpy(), output_loaded_3.numpy())
print("Model with custom trainable layer loaded successfully")

```
This example demonstrates a custom layer, `CustomTrainableLayer`, that defines its weights. The crucial aspect lies in how the weights are *defined and initialized* within the `build` method and how the `units` parameter is preserved within `get_config()`. Saving and loading the model correctly restores these trainable parameters along with the overall layer structure.  The training step is essential to showcase the learnable weights’ functionality.

To further enhance your understanding, I recommend delving into the official TensorFlow documentation regarding custom layers and models.  Explore the use of `tf.keras.utils.register_keras_serializable()` as another way of making your custom components serializable and deserializable. In addition, review resources on serialization techniques and the detailed structure of Keras models to better understand the underlying mechanisms. Finally, consider exploring best practices related to model versioning and reproducibility for long-term project maintenance. These resources will provide a deeper comprehension of how TensorFlow manages model persistence and enables effective handling of custom components.
