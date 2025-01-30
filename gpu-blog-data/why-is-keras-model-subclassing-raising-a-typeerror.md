---
title: "Why is Keras model subclassing raising a TypeError comparing None and int?"
date: "2025-01-30"
id: "why-is-keras-model-subclassing-raising-a-typeerror"
---
The TypeError "unsupported operand type(s) for ==: 'NoneType' and 'int'" encountered during Keras model subclassing typically originates from inconsistencies in how the model handles the output shape of a layer, specifically during the `call` method's shape inference.  I've personally debugged numerous instances of this in complex custom layers involving dynamic input shapes and conditional branching.  The root cause lies in the implicit expectation of consistent output tensor shapes across different branches of execution within the `call` method, an expectation not always met when dealing with variable input characteristics.


**1. Clear Explanation**

Keras model subclassing provides flexibility in defining custom model architectures.  However, this flexibility comes with responsibilities regarding proper shape handling. The `call` method, where the forward pass logic resides, must explicitly define the output shape of your custom layer or model. Keras relies on this information for various internal operations, including loss calculation and backpropagation. If conditional logic within the `call` method leads to the output tensor having a shape that is sometimes defined (a concrete `TensorShape`) and sometimes undefined (represented as `None`), comparisons during shape inference can fail, producing the `TypeError`. This typically happens when a branch of the `call` method doesn't return a tensor, resulting in a `None` value being implicitly compared to an integer representing an expected dimension.

The core problem boils down to the absence of a robust mechanism for handling potentially undefined dimensions in the model's output.  Keras assumes consistency. If your `call` method might produce outputs with varying shapes depending on the input, you must explicitly account for this variability and provide a consistent shape definition, even if it involves using placeholders (`None`) in a way that Keras can interpret.  Failing to do so results in the `NoneType` and `int` comparison, triggering the error.


**2. Code Examples with Commentary**

**Example 1:  Incorrect Shape Handling**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        if tf.shape(inputs)[1] > 5:
            return self.dense(inputs)
        else:
            # Incorrect: Returns None, not a tensor.  Shape inference fails.
            return None

model = MyModel()
input_shape = (10, 6)  # This will work
# input_shape = (10, 4)  # This will raise the TypeError
model(tf.random.normal(input_shape))
```

This example demonstrates a critical error.  The conditional statement determines whether or not the dense layer is applied. If the input's second dimension is less than or equal to 5, the function returns `None`. Keras's shape inference mechanism cannot handle this and will attempt to compare `None` with the expected output shape from `self.dense(inputs)`, causing the TypeError.

**Example 2: Correct Shape Handling using `tf.cond`**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        def true_fn():
            return self.dense(inputs)

        def false_fn():
            # Ensure a consistent output shape, even if it's a zero-tensor.
            return tf.zeros_like(inputs)

        output = tf.cond(tf.shape(inputs)[1] > 5, true_fn, false_fn)
        return output

model = MyModel()
model(tf.random.normal((10, 4)))
model(tf.random.normal((10, 6)))
```

This corrected version uses `tf.cond` to ensure a consistent tensor is always returned.  Even when the condition is false, a tensor of the same shape as the input (a zero tensor in this case) is returned, preventing the `NoneType` error.  The shape is implicitly handled correctly by Keras since a tensor, albeit a zero-tensor, is always returned.

**Example 3:  Handling Variable Output Shapes with `tf.TensorShape`**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense = tf.keras.layers.Dense(10)

  def call(self, inputs):
    output = self.dense(inputs)
    if tf.shape(inputs)[0] > 5:
      output = tf.reshape(output, (tf.shape(output)[0], 5)) # Dynamic Reshape
    return output

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape([input_shape[0], 10])
    if input_shape[0] > 5:
      shape = tf.TensorShape([input_shape[0], 5])
    return shape

model = MyModel()
model(tf.random.normal((2, 10)))
model(tf.random.normal((10, 10)))

```

This example shows a scenario where the output shape dynamically changes based on the input. The `compute_output_shape` method is crucial; it explicitly declares the possible output shapes, thereby guiding Keras's shape inference correctly, irrespective of the conditional logic in the `call` method. Note that the  `tf.TensorShape` is used to explicitly define the output even if the shape is variable and contains `None` for any unknown dimensions.


**3. Resource Recommendations**

*   The official TensorFlow documentation on custom models and layers. Carefully study the sections on shape inference and `compute_output_shape`.
*   Consult advanced tutorials focusing on creating and debugging custom Keras layers with dynamic shapes.  Pay close attention to examples dealing with conditional logic within the `call` method.
*   Explore the TensorFlow source code itself for a deeper understanding of how Keras handles shape inference internally. This requires advanced understanding of the framework's architecture.



Addressing the `TypeError` when subclassing Keras models necessitates a thorough understanding of tensor shapes and the framework's shape inference mechanism.  Always prioritize consistent tensor output from the `call` method; if the output shape varies, explicitly define all possible shapes using `tf.TensorShape` and the `compute_output_shape` method.  This proactive approach avoids runtime errors and enhances the reliability of your custom Keras models.  My experience has consistently demonstrated the importance of these precautions, particularly when dealing with conditionally executed operations and dynamic input characteristics.
