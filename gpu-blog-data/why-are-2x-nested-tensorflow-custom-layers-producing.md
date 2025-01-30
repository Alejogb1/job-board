---
title: "Why are 2x nested TensorFlow custom layers producing zero trainable parameters?"
date: "2025-01-30"
id: "why-are-2x-nested-tensorflow-custom-layers-producing"
---
The core issue when two nested TensorFlow custom layers result in zero trainable parameters stems from an oversight in the layer's build method, specifically how the internal variables are created and assigned within nested layers. I’ve encountered this situation multiple times during model development, leading to frustrating debugging sessions. Typically, the problem isn't with TensorFlow's core functionality but rather a misunderstanding of how `tf.Variable` scope and tracking works when composing custom layers.

Let's examine the mechanics. In TensorFlow, a layer's trainable parameters are defined by `tf.Variable` instances that are created and subsequently added to the layer's trainable weights list. This addition typically happens within a custom layer's `build` method, which is called during the first time the layer is used on a tensor with a known shape. If the `build` method fails to create and add these variables correctly, the layer will have no trainable weights, and consequently, the model containing it will not learn anything. Nesting layers exacerbates this because the outer layer doesn't automatically know about or manage the variables declared inside the inner layer. The key is to ensure the variables are attached to the layer where they're actually being used and are properly initialized when the `build` method is invoked.

The common error often occurs when the inner layer's `build` method isn't executed due to an incorrect shape inference or the inner layer is instantiated improperly, meaning no variables are initialized. More fundamentally, if the inner layer's variables aren't explicitly included in the outer layer's trainable_weights, even if they *are* initialized in the inner layer, they won’t get updated during training. TensorFlow tracks trainable parameters at a layer level, so merely having `tf.Variable` instances exist isn’t sufficient. They need to be part of the layer's designated weight tracking system.

To illustrate, let's examine a scenario where we have two custom layers: an outer layer named `OuterLayer` and an inner layer named `InnerLayer`.

**Example 1: The Issue - Inner Layer's Variables Are Not Tracked by the Outer Layer**

```python
import tensorflow as tf

class InnerLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(InnerLayer, self).__init__(**kwargs)
        self.units = units
        self.w = None
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class OuterLayer(tf.keras.layers.Layer):
    def __init__(self, units_inner, **kwargs):
        super(OuterLayer, self).__init__(**kwargs)
        self.inner_layer = InnerLayer(units_inner)

    def call(self, inputs):
      return self.inner_layer(inputs)

# Model building and testing
input_tensor = tf.random.normal((10, 5))
outer_layer = OuterLayer(units_inner=3)
output_tensor = outer_layer(input_tensor)

print(f"Outer Layer Trainable Weights: {len(outer_layer.trainable_weights)}")

```
In this example, `InnerLayer`’s variable `w` is created with the correct shape in the `InnerLayer.build` method. However, the `OuterLayer`'s `build` method is never defined. This means that the `OuterLayer` does not track the trainable weights of the inner layer; instead it relies on the `InnerLayer`'s build process. Therefore when querying `outer_layer.trainable_weights`, it will show zero. When it comes to training, any gradients of the outer layer will not affect the inner layer's weights. We have successfully made a model that will not learn.

**Example 2: The Solution - Include Inner Layer's Weights in Outer Layer**
This solution implements the `OuterLayer`'s `build` function in such a way that the inner layers variables are tracked by the outer layer during training.
```python
import tensorflow as tf

class InnerLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(InnerLayer, self).__init__(**kwargs)
        self.units = units
        self.w = None
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class OuterLayer(tf.keras.layers.Layer):
    def __init__(self, units_inner, **kwargs):
        super(OuterLayer, self).__init__(**kwargs)
        self.inner_layer = InnerLayer(units_inner)

    def build(self, input_shape):
        self.inner_layer.build(input_shape)
        self._trainable_weights.extend(self.inner_layer.trainable_weights)

    def call(self, inputs):
        return self.inner_layer(inputs)


# Model building and testing
input_tensor = tf.random.normal((10, 5))
outer_layer = OuterLayer(units_inner=3)
output_tensor = outer_layer(input_tensor)

print(f"Outer Layer Trainable Weights: {len(outer_layer.trainable_weights)}")

```

In this updated implementation, the `OuterLayer`'s `build` method is called explicitly. Firstly, the `build` method of the `inner_layer` is called using the `input_shape`, ensuring the correct weights are initialized within `inner_layer`. Then, we extend the `_trainable_weights` list of the `OuterLayer` with the `inner_layer`'s trainable weights using the `extend` function. By explicitly copying the weights to the outer layer, they are now tracked and available for training.

**Example 3: Using tf.keras.layers.Layer Subclasses as Custom Layers**

```python
import tensorflow as tf

class InnerLayer(tf.keras.layers.Dense):
    def __init__(self, units, **kwargs):
        super(InnerLayer, self).__init__(units, **kwargs)

class OuterLayer(tf.keras.layers.Layer):
    def __init__(self, units_inner, **kwargs):
        super(OuterLayer, self).__init__(**kwargs)
        self.inner_layer = InnerLayer(units_inner)

    def call(self, inputs):
      return self.inner_layer(inputs)

# Model building and testing
input_tensor = tf.random.normal((10, 5))
outer_layer = OuterLayer(units_inner=3)
output_tensor = outer_layer(input_tensor)

print(f"Outer Layer Trainable Weights: {len(outer_layer.trainable_weights)}")
```

Here, I utilize `tf.keras.layers.Dense` as my inner layer. Because it is a pre-built TensorFlow layer, the layer already handles tracking its own weights. Since I am not creating variables with `add_weight`, I no longer need to implement a `build` function. When building the `OuterLayer`, I call the `inner_layer` with the input tensor, which initializes the layer. After this step, the inner layer's weights are automatically tracked by the layer during the backpropagation process and can be updated.

In summary, the primary reason for encountering zero trainable parameters in nested custom TensorFlow layers revolves around the proper initialization and tracking of `tf.Variable` instances. The most common culprit is the outer layer not properly acknowledging or including the inner layer's weights within its trainable weight list. Using the `.extend()` call in the outer layer’s `build` function or using TensorFlow's built in layers provides a good solution. Using the built-in layers eliminates the hassle of tracking weights by hand.

For further understanding, I would recommend exploring the official TensorFlow documentation, particularly the sections related to custom layers, model subclassing, and variable management. In addition, focusing on tutorials that cover intermediate-level model building techniques, particularly those that explicitly demonstrate how weights are tracked across complex model architectures, can be very beneficial. Examining open-source repositories with well-designed custom layers can also provide valuable insights into best practices.
