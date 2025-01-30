---
title: "How can I create a custom Keras/TensorFlow dense layer with 2D input, 2D weights, and 2D bias?"
date: "2025-01-30"
id: "how-can-i-create-a-custom-kerastensorflow-dense"
---
The core challenge in creating a custom Keras/TensorFlow dense layer handling 2D input, weights, and bias lies in correctly aligning the dimensions during matrix multiplication to produce the desired output.  Over the years, working on image processing and spatiotemporal data modeling projects, I've encountered this need repeatedly.  Directly using the standard `tf.keras.layers.Dense` layer won't suffice because it assumes a flattened input vector, unlike our requirement for preserving spatial information inherent in 2D data.  The solution involves a careful implementation of the matrix multiplication using TensorFlow operations, ensuring compatibility with Keras's layer structure.


**1. Clear Explanation**

We need a layer that takes a 2D input tensor of shape (batch_size, height, width),  a 2D weight tensor of shape (height, width), and a 2D bias tensor of shape (height, width).  The operation should essentially perform a spatially-aware weighted sum, with the bias added element-wise.  The resulting output will also be a 2D tensor of shape (batch_size, height, width).  Standard matrix multiplication isn't directly applicable because the weights are not connected to the input in a fully-connected fashion. Instead, we need to perform element-wise multiplication and summation across the spatial dimensions.

Consider the input tensor `X` with shape (batch_size, height, width).  Our weight tensor `W` has shape (height, width), and the bias tensor `B` has the same shape as `W`.  The operation can be broken down as follows:

1. **Broadcasting:** We leverage broadcasting capabilities of TensorFlow to apply the 2D weights and bias to each element in each sample of the input batch.  This means that `W` and `B` are implicitly replicated across the batch dimension.

2. **Element-wise multiplication:** We perform element-wise multiplication between the input tensor `X` and the broadcasted weight tensor `W`.

3. **Summation:** The result of the element-wise multiplication is then summed over a chosen dimension (usually no summation is necessary for a spatially-aware multiplication and addition, which would be a common use case).

4. **Bias addition:** Finally, we add the broadcasted bias tensor `B` to the result.

This process ensures that each spatial location in the output tensor is a weighted sum of its corresponding input value, considering spatial context given by the weights and bias.  The output preserves the spatial information, which is crucial for tasks like image processing or other 2D data analysis.


**2. Code Examples with Commentary**

**Example 1:  Basic 2D Convolution-like Layer**

```python
import tensorflow as tf

class Custom2DDense(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(Custom2DDense, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.w = self.add_weight(shape=(height, width), initializer='random_normal', name='kernel')
        self.b = self.add_weight(shape=(height, width), initializer='zeros', name='bias')

    def call(self, inputs):
        # Ensure input is 3D (batch_size, height, width)
        if len(inputs.shape) != 3:
            raise ValueError("Input tensor must be 3D (batch_size, height, width)")

        # Broadcasting and element-wise multiplication
        weighted_input = inputs * self.w

        # Bias addition
        output = weighted_input + self.b

        return output

# Example usage:
layer = Custom2DDense(3, 3)
input_tensor = tf.random.normal((2, 3, 3)) # batch size 2, 3x3 input
output_tensor = layer(input_tensor)
print(output_tensor.shape) # Output shape: (2, 3, 3)

```

This example demonstrates a fundamental 2D dense layer. Note the use of `self.add_weight` to manage the weights and biases within the Keras layer structure, ensuring proper weight updates during training.  Error handling is included to ensure the input is of the correct dimensionality.


**Example 2:  Adding Activation Function**

```python
import tensorflow as tf

class Custom2DDenseActivation(tf.keras.layers.Layer):
    def __init__(self, height, width, activation='relu', **kwargs):
        super(Custom2DDenseActivation, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.w = self.add_weight(shape=(height, width), initializer='random_normal', name='kernel')
        self.b = self.add_weight(shape=(height, width), initializer='zeros', name='bias')
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Input tensor must be 3D (batch_size, height, width)")
        weighted_input = inputs * self.w
        output = weighted_input + self.b
        return self.activation(output)

#Example Usage:
layer = Custom2DDenseActivation(3,3, activation='sigmoid')
input_tensor = tf.random.normal((2,3,3))
output_tensor = layer(input_tensor)
print(output_tensor.shape)
```

This builds on the first example by incorporating an activation function, a critical component for non-linearity in neural networks.  The `tf.keras.activations.get` function ensures flexibility in choosing activation functions.


**Example 3:  Adding Summation along a Dimension**

```python
import tensorflow as tf

class Custom2DDenseSummation(tf.keras.layers.Layer):
    def __init__(self, height, width, sum_axis=-1, **kwargs): #Adding sum_axis
        super(Custom2DDenseSummation, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.w = self.add_weight(shape=(height, width), initializer='random_normal', name='kernel')
        self.b = self.add_weight(shape=(height, width), initializer='zeros', name='bias')
        self.sum_axis = sum_axis

    def call(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Input tensor must be 3D (batch_size, height, width)")
        weighted_input = inputs * self.w
        output = weighted_input + self.b
        return tf.reduce_sum(output, axis=self.sum_axis)

#Example usage:
layer = Custom2DDenseSummation(3,3, sum_axis=1) #summing over the rows
input_tensor = tf.random.normal((2,3,3))
output_tensor = layer(input_tensor)
print(output_tensor.shape) #Output shape will depend on sum_axis

```

This example extends functionality by allowing summation along a specified axis.  This is useful when you want to reduce the dimensionality of the output, for example, aggregating information along rows or columns. The `sum_axis` parameter provides flexibility. Remember that adjusting `sum_axis` will affect the output shape.  Careful consideration of this parameter is required to achieve the desired dimensionality reduction.



**3. Resource Recommendations**

For deeper understanding of TensorFlow and Keras layer creation, I would recommend consulting the official TensorFlow documentation and the Keras documentation.  Exploring example code repositories related to custom Keras layers would be highly beneficial.  Furthermore, reviewing literature on convolutional neural networks, which frequently handle 2D data and similar operations, would provide valuable context and alternative approaches.  A solid grasp of linear algebra concepts, particularly matrix multiplication and broadcasting, is also essential.
