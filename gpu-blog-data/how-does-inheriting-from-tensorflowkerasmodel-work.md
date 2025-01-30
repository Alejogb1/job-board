---
title: "How does inheriting from tensorflow.keras.Model work?"
date: "2025-01-30"
id: "how-does-inheriting-from-tensorflowkerasmodel-work"
---
Inheriting from `tensorflow.keras.Model` provides a robust and flexible approach for constructing custom neural network architectures within the TensorFlow ecosystem, moving beyond the sequential layering offered by `tf.keras.Sequential`. This practice allows for fine-grained control over layer connections, enabling the implementation of complex models like residual networks, graph neural networks, or any other design not easily expressible through a simple sequence.

The core concept involves creating a Python class that derives from `tf.keras.Model`, overriding the `__init__` and `call` methods. The `__init__` method serves primarily to instantiate the necessary layers, much like you would within a `Sequential` model, but here they are stored as class attributes. The `call` method, in contrast, defines the forward pass logic, explicitly specifying how data flows through these instantiated layers. This contrasts directly with `Sequential`, where the forward pass is implicitly defined by the ordered sequence of layers.

Consider, for example, building a straightforward multi-layer perceptron (MLP) using this approach. While `Sequential` could accomplish this, inheriting from `tf.keras.Model` offers a clear understanding of internal data flows. The `__init__` would initialize dense layers, and the `call` method would chain their application to the input tensor. This might seem verbose for such a simple example; however, consider a model with skip connections. Implementing those with `Sequential` becomes significantly more cumbersome or even impossible without custom layers or function abstractions. With `tf.keras.Model` inheritance, expressing those connections is done via tensor manipulation in the `call` method, thus affording greater flexibility.

Further, inheriting from `tf.keras.Model` provides access to crucial features, such as the model's internal weights through `model.trainable_variables`, facilitating manual or fine-grained training strategies not readily available with simpler model classes. It also implicitly integrates with the various `tf.keras.optimizers`, and loss functions within TensorFlow, allowing for a consistent development flow. By defining the model's structure explicitly, you gain greater clarity over its operation, particularly when debugging or attempting to replicate specific network architectures outlined in research papers.

I've encountered situations in production where we had to adapt a model initially built using `Sequential`. The flexibility provided by subclassing `tf.keras.Model` allowed us to introduce more complex operations, such as feature-wise attention, directly into the model's forward pass without needing to dramatically restructure the entire training pipeline. This experience solidified the necessity of understanding how to inherit from `tf.keras.Model`. This is why it's essential to understand this approach.

Here are some example implementations:

**Example 1: Basic MLP Implementation**

```python
import tensorflow as tf

class CustomMLP(tf.keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(CustomMLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


# Example instantiation and usage
mlp_model = CustomMLP(hidden_units=64, num_classes=10)
input_data = tf.random.normal((32, 100)) #batch of 32 samples with 100 features
output = mlp_model(input_data)
print(f"Output shape: {output.shape}")
```

*Commentary:* This illustrates a basic implementation of a two-layer MLP. The `__init__` method initializes the two `Dense` layers, and `call` passes the input first through the first layer then through the second.  The `super` call is essential for proper initialization of the base `tf.keras.Model` class, allowing for its inherent functionality like saving and loading weights. The provided instantiation and usage demonstrates the model being called directly on data.

**Example 2:  Model with a Skip Connection**

```python
class SkipConnectionModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(SkipConnectionModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense3 = tf.keras.layers.Dense(hidden_units)  # No activation for the final layer

    def call(self, inputs):
        x = self.dense1(inputs)
        y = self.dense2(x)
        z = self.dense3(y)
        return tf.keras.layers.Add()([inputs, z])  # Skip connection


# Example usage:
skip_model = SkipConnectionModel(hidden_units=128)
input_data = tf.random.normal((64, 256))
output = skip_model(input_data)
print(f"Output shape: {output.shape}")
```

*Commentary:* This example demonstrates how easily skip connections can be introduced. The input data `inputs` is added to the output of a series of dense layers (`z`). Critically, the ability to manipulate tensors directly in the `call` method enables this flexibility. This pattern mirrors residual blocks commonly found in deep convolutional architectures.

**Example 3: Dynamic Model with Conditional Layer Selection**

```python
class DynamicModel(tf.keras.Model):
    def __init__(self, hidden_units, use_dropout=False):
        super(DynamicModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units)
        self.dropout = tf.keras.layers.Dropout(0.5) if use_dropout else None
        self.use_dropout = use_dropout


    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if self.use_dropout and training: # only apply dropout in training and if set in init.
            x = self.dropout(x)
        return self.dense2(x)


# Example Usage
dynamic_model_with_dropout = DynamicModel(hidden_units = 32, use_dropout=True)
dynamic_model_without_dropout = DynamicModel(hidden_units = 32, use_dropout=False)
input_data = tf.random.normal((128, 64))
output_with_dropout = dynamic_model_with_dropout(input_data, training = True)
output_without_dropout = dynamic_model_without_dropout(input_data)


print(f"Output shape with dropout: {output_with_dropout.shape}")
print(f"Output shape without dropout: {output_without_dropout.shape}")
```

*Commentary:* This more advanced example showcases conditional logic within the `call` method.  Here a dropout layer is dynamically applied, depending on both initialization parameter and the `training` argument provided during the call. This is a common pattern in training deep learning models, emphasizing that `tf.keras.Model` allows for conditional or iterative processing within the forward pass, offering more complex model variations without requiring separate model definitions.

For resources, I highly recommend the official TensorFlow documentation which dedicates a section to custom models. Additionally, the *Deep Learning with Python* book by Francois Chollet and the *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* book by Aurelien Geron provides detailed examples. Exploring the source code of open-source machine learning libraries, like Hugging Face transformers, can reveal even more complex usage patterns. Practicing with various network architectures, and reimplementing examples from those books with this custom model pattern will solidify these concepts. By focusing on these resources, you'll obtain a robust understanding of creating and implementing your own custom `tf.keras.Model` architectures.
