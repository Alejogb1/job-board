---
title: "How can custom TensorFlow models save custom attributes?"
date: "2025-01-30"
id: "how-can-custom-tensorflow-models-save-custom-attributes"
---
The primary challenge when saving custom TensorFlow models alongside custom attributes lies in the serialization process. TensorFlow's SavedModel format, while robust for standard Keras models and layers, does not inherently persist arbitrary Python attributes defined on custom model classes. Direct assignment of such attributes to a model instance won't automatically translate into the saved model's structure. This requires a more deliberate approach, often leveraging TensorFlow’s configuration capabilities and class inheritance.

I've encountered this issue several times, particularly when developing models that incorporate specific preprocessing parameters or model-level configurations. Simply setting an attribute `model.scaling_factor = 2.0` wouldn't survive the save/load cycle unless explicitly managed. The key is to integrate these attributes with TensorFlow's machinery, typically through the model's `__init__` method, which is subsequently used during model reconstruction, and to manage serialization within save/load methods, including the use of model configs.

Here’s a breakdown of the method using three progressively complex code examples:

**Example 1: Basic Attribute Persistence using `get_config` and `from_config`**

This first example demonstrates the fundamental mechanism. The `get_config` method is crucial. When `model.save()` is called, TensorFlow looks for this method. If present, it is used to serialize the model. The `from_config` method handles model instantiation during loading.

```python
import tensorflow as tf

class CustomModelBasic(tf.keras.Model):
    def __init__(self, units, activation='relu', scaling_factor=1.0, **kwargs):
        super(CustomModelBasic, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.scaling_factor = tf.Variable(scaling_factor, dtype=tf.float32, trainable=False, name="scaling_factor")
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        scaled_inputs = inputs * self.scaling_factor
        output = self.dense(scaled_inputs)
        return self.activation(output)

    def get_config(self):
        config = super(CustomModelBasic, self).get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "scaling_factor": self.scaling_factor.numpy()
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["activation"] = tf.keras.activations.deserialize(config["activation"])
        return cls(**config)

# Example Usage
model_basic = CustomModelBasic(units=32, activation='sigmoid', scaling_factor=2.5)
inputs = tf.random.normal((1, 10))
_ = model_basic(inputs) # Build the model
model_basic.save("model_basic_saved")

loaded_model_basic = tf.keras.models.load_model("model_basic_saved")
print(f"Loaded Scaling Factor: {loaded_model_basic.scaling_factor.numpy()}")
print(f"Loaded Model Type: {type(loaded_model_basic)}")
```

Here, the `scaling_factor` which was passed in as a parameter to the `__init__` method and subsequently used in the `call` method is preserved. Note the use of `tf.Variable`, which allows explicit serialization, and how both the scaling factor value as a float, as well as the string corresponding to the activation are saved within the config as part of `get_config`, and later passed to the `from_config` constructor which builds the model correctly. The `numpy()` call ensures serialization of the variable itself instead of the reference. The `@classmethod` ensures we can create a model object directly using configuration data.

**Example 2:  Attribute Persistence for Complex Types and Non-Configurable Attributes**

This example extends the previous concept to handle more complex attribute types, such as lists or dictionaries, and incorporates an additional attribute which isn't meant to be saved as part of configuration but is relevant to model operation after loading.

```python
import tensorflow as tf

class CustomModelComplex(tf.keras.Model):
    def __init__(self, units, layer_sizes, preprocessing_params, **kwargs):
        super(CustomModelComplex, self).__init__(**kwargs)
        self.units = units
        self.layer_sizes = layer_sizes
        self.preprocessing_params = preprocessing_params
        self.layers_list = [tf.keras.layers.Dense(size) for size in layer_sizes]
        self.layer_count = len(layer_sizes)  # Example non-config attribute

    def call(self, inputs):
      x = inputs
      for layer in self.layers_list:
        x = layer(x)
      return x

    def get_config(self):
        config = super(CustomModelComplex, self).get_config()
        config.update({
            "units": self.units,
            "layer_sizes": self.layer_sizes,
            "preprocessing_params": self.preprocessing_params
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def load_layer_count(self):
        # Example of additional operation based on non-config attribute.
        print(f"This model has {self.layer_count} layers.")


# Example Usage
model_complex = CustomModelComplex(
    units=64,
    layer_sizes=[128, 64, 32],
    preprocessing_params={'mean': 0.5, 'std': 0.2}
)
inputs = tf.random.normal((1, 10))
_ = model_complex(inputs) # Build the model
model_complex.save("model_complex_saved")

loaded_model_complex = tf.keras.models.load_model("model_complex_saved")
print(f"Loaded Layer Sizes: {loaded_model_complex.layer_sizes}")
print(f"Loaded Preprocessing Params: {loaded_model_complex.preprocessing_params}")
loaded_model_complex.load_layer_count()

```

Here, `layer_sizes` (a list) and `preprocessing_params` (a dictionary) are directly included in the config and passed to the constructor. The `layer_count` attribute is not part of the config and will be determined during instantiation. Note the instantiation of a list of layers. After the model is loaded, the  `load_layer_count` demonstrates we have access to non-config based attributes that were set during model creation. This method provides a mechanism to encapsulate other behaviors which are based on the internal state of the model.

**Example 3: Attribute Persistence with Inherited Models and Custom Training Logic**

This final example illustrates persistence when inheriting from pre-existing model layers, where model attributes interact with training logic defined within the model itself.

```python
import tensorflow as tf
class CustomModelInherited(tf.keras.Model):
    def __init__(self, hidden_units, dropout_rate=0.2, **kwargs):
        super(CustomModelInherited, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1)


    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
           x = self.dropout(x)
        output = self.dense2(x)
        return output


    def get_config(self):
        config = super(CustomModelInherited, self).get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
           y_pred = self(x, training=True)
           loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# Example Usage
model_inherited = CustomModelInherited(hidden_units=128, dropout_rate=0.3)
model_inherited.compile(optimizer='adam', loss='mse', metrics=['mae'])
inputs = tf.random.normal((10, 10))
labels = tf.random.normal((10, 1))
_ = model_inherited(inputs)  # Build Model
model_inherited.fit(inputs, labels, epochs=1)

model_inherited.save("model_inherited_saved")

loaded_model_inherited = tf.keras.models.load_model("model_inherited_saved")
print(f"Loaded Hidden Units: {loaded_model_inherited.hidden_units}")
print(f"Loaded Dropout Rate: {loaded_model_inherited.dropout_rate}")

```

This example shows how the `dropout_rate` is used in the `call` method based on the value loaded from the config, and how it influences the training behavior implemented using `train_step`. In this case the `dropout_rate` is a model-level hyperparameter. The `training` parameter demonstrates another use case where these saved parameters can affect model behavior.

In summary, saving custom attributes in TensorFlow models requires explicit handling through configuration methods (`get_config` and `from_config`). This includes serializing attributes into dictionaries, and handling appropriate deserialization during model construction. When working with custom training loops, the stored parameters can influence model-level behavior, which needs to be taken into account when constructing the model. It is also important to distinguish between attributes that should be part of model configuration and those which should not, where the later will be present in an instantiated model but are not stored when the model is saved.

For further understanding, I would recommend consulting the official TensorFlow documentation focusing on Model subclassing, creating custom layers, and the SavedModel format. Reading through the source code of Keras core modules, especially related to `Model`, `Layer`, and `activations`, can also offer deeper insights into the design choices made during TensorFlow’s construction. Another useful resource is the "TensorFlow Advanced" section of many tutorials and books that explicitly focus on the more nuanced aspects of building custom models. Examining community implementations on GitHub can show different design patterns for these operations in practice.
