---
title: "Why does a custom layer cause a parameter count of zero in the model summary?"
date: "2025-01-30"
id: "why-does-a-custom-layer-cause-a-parameter"
---
A custom layer, when implemented incorrectly within a deep learning framework like TensorFlow or Keras, often results in a parameter count of zero when viewed through the model summary because its trainable variables are not properly registered with the framework's automatic differentiation and variable tracking mechanisms. This typically stems from neglecting the initialization or usage of `tf.Variable` objects or failing to define the `build()` method which is the correct method for creating layers with weights that are not immediately known at construction.

When creating a layer using inheritance, specifically from `tf.keras.layers.Layer`, two crucial stages define its behavior and integration: initialization and build. Initialization, achieved through `__init__`, is primarily for storing any user-supplied parameters related to the layer's structure—things like output dimensions, kernel sizes, or activation functions. It is **not** the place for creating the layer’s weights and biases.

The `build()` method, on the other hand, executes once the layer has been connected to a tensor of some known shape (i.e. during the first forward pass, where we have concrete input shapes). Here, we create variables which are then managed as trainable by the framework. A failure to define this method correctly, often in favor of trying to define weights in `__init__` without knowing the shape of the input, can lead to missing parameters.

Furthermore, parameters must be created as `tf.Variable` objects. Instantiating weights as raw tensors, such as using `tf.random.normal`, does not register them for automatic differentiation or updates during training. The framework views these tensors as constants, preventing their optimization by the training algorithm. The framework doesn't automatically 'convert' those tensors to variables that it tracks.

Consider a naive implementation attempting a custom linear layer. I encountered this very issue in my early work with custom architectures for time series prediction.

```python
import tensorflow as tf

class NaiveLinear(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(NaiveLinear, self).__init__(**kwargs)
        self.units = units

    def call(self, inputs):
      input_dim = inputs.shape[-1]
      self.w = tf.random.normal(shape=(input_dim, self.units))
      self.b = tf.zeros(shape=(self.units,))
      return tf.matmul(inputs, self.w) + self.b

# Creating and summarizing the model
inputs = tf.keras.layers.Input(shape=(10,))
x = NaiveLinear(units=5)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)
model.summary()
```

In this first example, the `NaiveLinear` layer creates weights (`self.w`) and biases (`self.b`) within the `call()` method using `tf.random.normal` and `tf.zeros`, respectively. When I ran this code, the `model.summary()` reported a total trainable parameter count of zero. This is because those tensors were not created as trainable variables and thus are not managed by the framework's gradient machinery. Each time the `call` is executed new tensors will be created and thus there is no persistent weight that the optimzer can modify.

To rectify this issue, the `build` method should be used to create the variables as `tf.Variable` objects. Moreover, the weights should be accessible as member variables of the class, not declared every time the layer is called.

```python
import tensorflow as tf

class CorrectLinear(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CorrectLinear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(shape=(input_dim, self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Creating and summarizing the model
inputs = tf.keras.layers.Input(shape=(10,))
x = CorrectLinear(units=5)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)
model.summary()
```

In the corrected `CorrectLinear` class, the `build()` method creates the weight (`self.w`) and bias (`self.b`) variables using `self.add_weight()`. The framework is now aware of these trainable variables. We can now access them from within the `call` method (we don't need to re-create them). When I ran this modified code, the model summary correctly displayed the trainable parameters (in this case 55, 50 for the weights and 5 for the biases). Additionally, by creating the variables using `add_weight`, the default trainability is set to true, which is the desired behavior most of the time. The use of the `add_weight` method also handles the creation of variables on the appropriate device.

Lastly, one of the most common errors I see stems from the fact that `build` is automatically called by the layer (or a class that uses that layer), it is not called manually by the user. Thus, we can't rely on the input to have a shape when we're doing manual instantiations of layers:

```python
import tensorflow as tf

class IncorrectEarlyLinear(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, **kwargs):
        super(IncorrectEarlyLinear, self).__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(shape=(input_dim, self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Creating and summarizing the model
inputs = tf.keras.layers.Input(shape=(10,))
# Notice, we must specify the input_dim here, which may not be known during complex model builds.
x = IncorrectEarlyLinear(units=5, input_dim=10)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)
model.summary()
```

In this final example, I attempt to circumvent the `build` call by creating the weight variables within the `__init__` constructor. While the model summary will *appear* to show the correct number of parameters when run (as it is also using `add_weight` to create the weights), there are several critical problems with this approach. First, it tightly couples the layer to a fixed input size, making the layer inflexible and prone to error when used with varying input shapes in different parts of the network. For a simpler example like this, one *could* pass that dimension in the constructor, but that is not always possible. Secondly, if that layer is instantiated (by keras, not by the user) during a complex graph traversal, the dimension may not even be known. For example, if we wanted to reuse that layer in a larger model with a feature extraction stage that has variable dimension, this code won't work. Keras does all this in `build`, and the user never really instantiates the weight variables directly. The key here is that the `build` method will have that information, the constructor never will.

To summarize, the primary cause of a zero parameter count with custom layers is typically a failure to properly register trainable variables through the `tf.Variable` object within the `build()` method. The `__init__` method of a custom layer is not the place to define those variables. Incorrect usage or initialization leads to the framework treating weights as constants.

For further study on custom layer implementation within TensorFlow and Keras I recommend reviewing the official TensorFlow documentation, paying close attention to sections on custom layers. Also the relevant chapters in online deep learning textbooks should help build a solid intuition about model construction. Furthermore, examining open-source implementations of popular layers can also prove extremely beneficial. Always start with the basic tutorials on keras layers, and build up your understanding. I have found that going too fast to custom implementations can lead to issues if the base knowledge is not sufficient.
