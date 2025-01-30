---
title: "Why are no gradients being calculated for TFCamemBERT?"
date: "2025-01-30"
id: "why-are-no-gradients-being-calculated-for-tfcamembert"
---
The absence of gradients during training for a custom model, TFCamemBERT in this case, is a common issue rooted in the way TensorFlow handles its computational graph. Typically, the issue boils down to one of two causes: either the operations within the custom model are not being tracked for gradient computation, or the optimizer is not configured to update the model's trainable variables. In my experience, I've encountered this repeatedly, both in my own projects and when assisting others, and the debugging process often involves systematically verifying these aspects. Specifically, with Transformer-based models like CamemBERT, the complexity of the architecture increases the probability of introducing subtle errors that break gradient flow.

The primary culprit is often a disconnect between operations performed during the forward pass and their corresponding gradient tracking. TensorFlow relies on a computational graph, where it automatically records operations that can be differentiated. However, this tracking only works if those operations are performed using TensorFlow's API functions, such as `tf.matmul`, `tf.nn.relu`, or the layers in `tf.keras.layers`. Custom operations, manual tensor manipulations, or functions imported from external libraries might not be integrated into this differentiable graph. When this happens, the backward pass encounters an operation it cannot compute a gradient for, effectively terminating the gradient flow. A similar situation arises if, for example, you are using an advanced activation function not supported by default, and manually implement this logic with numpy instead of using native tensorflow logic.

Consider also situations where you are assembling multiple smaller, previously trained models into a larger model. If you are not careful about specifying the parameter update behavior on these embedded models, you might be freezing them such that their weights remain constant during gradient descent. Thus, you might find gradients are updating in only specific parts of your model, leading to partial training only.

Secondly, even if the gradients are correctly calculated, they may not be applied if the optimizer is not targeting the correct variables. In TensorFlow, trainable variables are the weights and biases of layers that should be adjusted during backpropagation. The optimizer, typically an Adam or SGD variant, is responsible for updating these variables based on the calculated gradients. If these variables are not registered as trainable or are not correctly associated with the model's layers, the optimizer will have nothing to act upon. Further, ensure that your custom layers or models do inherit from the proper `tf.keras.layers.Layer` or `tf.keras.Model` classes. Without this, Tensorflow will not correctly register the trainable variables.

To illustrate these points, let's examine some concrete code examples.

**Example 1: Missing TensorFlow operations:**

```python
import tensorflow as tf
import numpy as np

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        # Incorrect: Using numpy matrix multiplication
        #outputs = np.dot(inputs, self.w.numpy()) + self.b.numpy()

        # Correct: Using TensorFlow matrix multiplication
        outputs = tf.matmul(inputs, self.w) + self.b

        return outputs


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    CustomLayer(5)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

x = tf.random.normal((32, 10))
y = tf.random.normal((32, 5))

with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = loss_fn(y, y_pred)

grads = tape.gradient(loss, model.trainable_variables)

optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Here, the initial, commented-out line attempts to use NumPy for matrix multiplication. This circumvents TensorFlow's computational graph, leading to `grads` being `None` for the weights and biases because no gradient information was captured during the forward pass. Replacing `np.dot` with `tf.matmul`, along with changing the added bias from numpy to tensorflow variable, ensures TensorFlow correctly tracks the operation. It is also imperative to use `self.add_weight` to instantiate the weight and bias to ensure they are treated as trainable variables by the Tensorflow system.

**Example 2: Incorrect variable registration:**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
  def __init__(self, units):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units)
    self.dense2 = tf.keras.layers.Dense(units)
    self._internal_variables = None


  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

  @property
  def trainable_variables(self):
        return self.dense1.trainable_variables + self.dense2.trainable_variables

model = MyModel(16)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

x = tf.random.normal((32, 10))
y = tf.random.normal((32, 16))

with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = loss_fn(y, y_pred)

grads = tape.gradient(loss, model.trainable_variables)

optimizer.apply_gradients(zip(grads, model.trainable_variables))
```
In this example we showcase a proper implementation of a custom `tf.keras.Model`. Note how the trainable variables are extracted from the constituent keras layer's trainable variable attribute and returned. This will result in the gradients being computed for the weights of both of our dense layers. If we were to simply return a list of all internal variables, Tensorflow may not correctly track the association between the weights and layers. The use of the getter function ensures that all internal parameters will be tracked, and that gradients will be computed and applied correctly.

**Example 3: Freezing model parameters:**

```python
import tensorflow as tf

class EmbeddedModel(tf.keras.Model):
    def __init__(self, units):
        super(EmbeddedModel, self).__init__()
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return self.dense(inputs)

class MainModel(tf.keras.Model):
    def __init__(self, units):
        super(MainModel, self).__init__()
        self.embedded = EmbeddedModel(units)
        self.final_dense = tf.keras.layers.Dense(units)


    def call(self, inputs):
        x = self.embedded(inputs)
        return self.final_dense(x)

model = MainModel(16)
model.embedded.trainable = False #freezing the embedded model!

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

x = tf.random.normal((32, 10))
y = tf.random.normal((32, 16))

with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = loss_fn(y, y_pred)

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Here we explicitly demonstrate the freezing of an inner model which might arise in more complex applications. The inner `embedded` model was instantiated as a keras layer which had its `trainable` property set to false before the training loop began. By doing so, no gradients will be calculated for the inner layer. In these situations, one has to manually check that each layer that should be training is trainable. If you are loading pretrained models into your system, be aware of these properties.

When faced with the issue of vanishing gradients in TFCamemBERT or any other custom model, a systematic approach is required. First, ensure that all operations within your model's `call` method are performed using TensorFlow's API. Secondly, ensure you are utilizing `self.add_weight` to create any variables to be tracked. Third, double check to ensure each sublayer or submodule is indeed trainable when required. Verify that you are not improperly overwriting the `trainable_variables` getter in your class inheritance, and ensure it returns all of your trainable parameters. Finally, make sure that the optimizer targets the correct `trainable_variables` associated with your model.

For further investigation, I recommend consulting the official TensorFlow documentation on custom layers and models, as well as the section on gradient computation and automatic differentiation. Furthermore, exploring tutorials on TensorFlow training loops and the intricacies of custom layer implementations will provide deeper insights. Finally, the `tf.GradientTape` documentation includes best practices for debugging gradient calculations and is quite helpful when encountering unexpected behavior.
