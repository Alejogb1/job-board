---
title: "How can I save and reload a subclassed TensorFlow 2.6 model without performance loss?"
date: "2025-01-30"
id: "how-can-i-save-and-reload-a-subclassed"
---
The persistent challenge with subclassed TensorFlow models, particularly when saving and reloading, lies in their inherently dynamic nature. Unlike sequential or functional API models, subclassed models define their architecture and computations within the `call()` method, which isn't directly serialized by TensorFlow's standard saving mechanisms. I've spent considerable time debugging this exact issue, experiencing first-hand the frustration of seemingly identical models exhibiting drastically different performance after reloading. The key is recognizing that we need to persist the model's architecture *and* its learned parameters, which requires careful handling of state and configuration.

A subclassed model derives from `tf.keras.Model`, granting it access to saving and loading capabilities through methods like `save()` and `tf.keras.models.load_model()`. However, the issue isn't *how* to save and load, but *what* gets saved and loaded. Standard saving procedures primarily capture the model's weights and, crucially, the computational graph for sequential or functional models. Subclassed models, lacking a statically defined graph, don't automatically translate to a serialized representation. The `call` method, which determines how data flows through the layers and custom operations, must be preserved by other means. Failing to do so can manifest in a variety of problems, from incorrect outputs due to layer misconfigurations to a degradation of training performance if optimization states are lost during the save/reload cycle. The essential point is to serialize not only weights but also the custom configuration embedded within the class itself.

The simplest path towards reliable save and reload behavior is to utilize the model's `get_config` and `from_config` methods. These methods, when properly implemented, serialize the essential components needed to reconstruct the model's architecture. This ensures that the model is not just a container of weights, but rather a faithful representation of its original design. Specifically, the `get_config` method should return a dictionary containing all the necessary information to rebuild the model structure, including layer types, hyperparameters, and any custom attributes controlling the model's behavior. Conversely, `from_config`, a class method, takes this configuration dictionary and reconstitutes the model. This two-way process allows for accurate reproduction of the model post-reload. It’s worth noting that this approach is distinct from saving individual weights into checkpoint files, which alone aren't sufficient to reconstruct a subclassed model.

Here is a first example illustrating the basic approach:

```python
import tensorflow as tf

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
             output = self.activation(output)
        return output

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config


    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.activations.deserialize(config["activation"])
        return cls(**config)

class MySubclassedModel(tf.keras.Model):
    def __init__(self, num_units, activation=None, **kwargs):
        super(MySubclassedModel, self).__init__(**kwargs)
        self.dense_1 = CustomDense(units = num_units, activation = activation)
        self.dense_2 = CustomDense(units = 1)
        self.activation = activation

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)


    def get_config(self):
       config= super(MySubclassedModel, self).get_config()
       config.update({
           "num_units":self.dense_1.units,
           "activation":tf.keras.activations.serialize(self.activation)

       })
       return config
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.activations.deserialize(config["activation"])
        return cls(**config)



# Example usage
model = MySubclassedModel(num_units = 32, activation = "relu")
input_data = tf.random.normal(shape=(1, 10))
_ = model(input_data) # force model to build

model.save('my_model')
loaded_model = tf.keras.models.load_model('my_model')
```

In this example, the `CustomDense` layer and the `MySubclassedModel` each implement their own `get_config` and `from_config`. Notice how I ensure the activation function is serialized, since it would be lost otherwise. The `get_config` method collects necessary attributes into a dictionary and passes it along, while `from_config` reconstructs the instance.  When saving a subclassed model in TensorFlow, you are only saving the configuration and the weight parameters. The actual function is not saved, which means we have to make sure that any activation functions, or any custom layers, are being serialized properly and are able to be reconstructed. This example demonstrates how to serialize activation functions.

The next critical aspect relates to handling layers which are not directly serialized using `get_config` and `from_config`. While built-in layers in `tf.keras.layers` generally handle this serialization automatically, we must carefully manage custom layers or functionalities introduced within the model. For instance, if our `call` method includes custom state variables or a non-standard workflow, we need to develop methods for saving and restoring this state. This often requires creating custom saving and loading mechanisms which might involve saving variables into checkpoint files. A common scenario where this arises is in models that retain internal state, such as Recurrent Neural Networks (RNNs) or models employing batch normalization where you can save and reload the batch norm variables through this procedure.

Let's look at an example involving a batch norm layer:

```python
import tensorflow as tf

class CustomBatchNormModel(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super(CustomBatchNormModel, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=units)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn(x, training=training) # Pass training=True during training.
        return x

    def get_config(self):
        config = super(CustomBatchNormModel, self).get_config()
        config.update({
            "units": self.dense.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
       return cls(**config)
    
# Example usage
model = CustomBatchNormModel(units=32)
input_data = tf.random.normal(shape=(1, 10))
_ = model(input_data, training = True) # Must run call() once to initialize BatchNorm variables

model.save('bn_model')
loaded_model = tf.keras.models.load_model('bn_model')

#Test that the model behaves correctly
test_input = tf.random.normal(shape=(1,10))
output_1 = model(test_input, training=False)
output_2 = loaded_model(test_input, training=False)
print(tf.reduce_sum(output_1 - output_2))

```

In this example, I am not storing the mean and variance of the batch normalization during the `get_config` and `from_config` methods, because these are saved automatically by `tf.keras.Model`. These internal states of the batch norm layer will not be serialized if `training=True` is not set to call once.  This ensures that the batch normalization layers' states are properly captured during the saving process. This approach guarantees that these layers can continue operating on the correct accumulated statistics. Additionally, I verify that after reloading and performing inference, that the difference between the outputs are close to 0.

Finally, even with careful implementation of `get_config` and `from_config`, there can be subtle performance differences, particularly if you are saving/loading within a complex pipeline with different execution contexts. These can be attributed to slight differences in graph optimization or minor variations in hardware acceleration. To mitigate this, you should rigorously benchmark the saved and reloaded models using representative data sets and monitor for any degradation in accuracy or inference speed. If discrepancies appear, carefully review the data preprocessing pipelines, model's construction, and execution environments to ensure the save and load operations have not inadvertently introduced any differences.

Here is one last example of how to handle saving custom variables inside a custom layer:

```python
import tensorflow as tf

class CustomVariableLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(CustomVariableLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.custom_var = None # initialize in the build method


    def build(self, input_shape):
        self.custom_var = self.add_weight(shape = (self.num_features,), initializer = 'ones', trainable = True)

    def call(self, inputs):
       return inputs * self.custom_var # custom operation


    def get_config(self):
        config = super(CustomVariableLayer, self).get_config()
        config.update({
            "num_features": self.num_features,
        })
        return config

    @classmethod
    def from_config(cls, config):
       return cls(**config)

class CustomModelWithVariableLayer(tf.keras.Model):
    def __init__(self, num_features, **kwargs):
        super(CustomModelWithVariableLayer, self).__init__(**kwargs)
        self.custom_layer = CustomVariableLayer(num_features = num_features)
        self.dense = tf.keras.layers.Dense(units = 1)


    def call(self, inputs):
        x = self.custom_layer(inputs)
        return self.dense(x)

    def get_config(self):
       config = super(CustomModelWithVariableLayer, self).get_config()
       config.update({
           "num_features":self.custom_layer.num_features,
       })
       return config

    @classmethod
    def from_config(cls, config):
       return cls(**config)


# Example usage
model = CustomModelWithVariableLayer(num_features = 5)
input_data = tf.random.normal(shape=(1, 5))
_ = model(input_data)

model.save('custom_var_model')
loaded_model = tf.keras.models.load_model('custom_var_model')


test_input = tf.random.normal(shape = (1,5))

output_1 = model(test_input)
output_2 = loaded_model(test_input)
print(tf.reduce_sum(output_1-output_2))
```

In this example, a custom variable is created in the build method and a custom operation is performed on the inputs using this variable in the call method. The key idea is to serialize this variable through the standard means using `add_weight`, since we do not want to try and save the variables ourselves using checkpoint files. The `get_config` and `from_config` methods do not require any special handling for the variables themselves. Like the batch norm example, I verify the model is performing correctly by testing with inference and ensuring the output difference is close to zero.

To expand your knowledge on this topic, I would recommend studying the following: the official TensorFlow documentation for custom layers and model subclassing, research papers on model serialization techniques, and the source code of tf.keras implementations (particularly the saving and loading logic). Examining these resources can reveal best practices for building robust save and load capabilities for custom models. Furthermore, you should explore advanced techniques such as the use of callbacks and the checkpoint API, although these are beyond the scope of the initial question. Properly handling your model saving strategy should yield a workflow that mirrors the performance of your original, trained model.
