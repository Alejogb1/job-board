---
title: "Why does my TensorFlow custom gradient function report 'variables' but no ResourceVariables were used?"
date: "2024-12-23"
id: "why-does-my-tensorflow-custom-gradient-function-report-variables-but-no-resourcevariables-were-used"
---

Let’s tackle this one. I recall a rather perplexing debugging session from a project a few years back, involving a custom TensorFlow layer where, much like the situation you're facing, gradients reported variables seemingly out of thin air. It took some thorough examination to understand what was actually transpiring. The core of the issue isn't necessarily about the explicit use (or lack thereof) of `tf.Variable` instances directly in the gradient function, but rather how TensorFlow's automatic differentiation mechanics operate behind the scenes. Let's break down the subtleties.

When you implement a custom gradient function using `tf.custom_gradient`, TensorFlow's automatic differentiation system keeps track of the operations and the variables involved in the forward pass. Even if your forward pass does not explicitly create `tf.Variable` objects, if it involves TensorFlow operations that implicitly create and manage trainable parameters, those parameters will be caught by the gradient tracking mechanism. These operations aren't always obvious; they often involve internal, behind-the-scenes parameter management. For instance, a seemingly simple `tf.matmul(a, b)` might, if `a` or `b` are the outputs of layers with trainable parameters or are used with other operations that establish trainable dependencies, have an associated variable lineage captured by the autograd system. The fact they aren't directly *ResourceVariables* doesn't preclude their involvement.

The key here is the *autodiff tape*. In TensorFlow, during the forward pass, operations are recorded onto the "tape" if their gradients need to be tracked. This tape captures how operations are linked to each other and to the inputs they depend on. If, inside your custom gradient function, you compute a gradient with respect to a tensor that has a lineage traced back to a trainable variable through this tape, then that *variable*, not explicitly a `tf.Variable` in your scope, gets pulled into the gradient calculation. This becomes particularly apparent if you use layers such as `tf.keras.layers.Dense`, or use functions with trainable internal parameters without directly defining these parameters as your own `tf.Variable`. These internal parameters are represented internally as *ResourceVariables*, but the abstraction layers mask this fact until the gradients are computed. The reported "variables" in your custom gradient's error message aren't referring to variables defined explicitly within your `tf.custom_gradient` function itself, but to the dependency of the tracked tensor for which you are trying to calculate the gradient on trainable parameters from prior computations.

To illustrate, let's consider a simplified version of a scenario where this behavior might manifest. Assume we have a custom layer (though this effect is not layer-specific, and could be seen in a function or model context) that uses `tf.matmul` without any explicitly defined variable, and we want to define a custom gradient for it.

```python
import tensorflow as tf

@tf.custom_gradient
def custom_matmul(a, b):
    result = tf.matmul(a, b)
    def grad_fn(dy):
        # Trying to calculate gradient with respect to a and b directly.
        # Even without explicit variables in this function.
        grad_a = tf.matmul(dy, tf.transpose(b))
        grad_b = tf.matmul(tf.transpose(a), dy)
        return grad_a, grad_b
    return result, grad_fn

# Example usage:
a = tf.random.normal((3, 4))
b = tf.random.normal((4, 5))

with tf.GradientTape() as tape:
  tape.watch(a) # We're tracking our tensors
  tape.watch(b)
  result = custom_matmul(a,b)

grads = tape.gradient(result, [a,b])
print(grads)
```

In this example, `a` and `b` are not `tf.Variable` instances; they are tensors. Yet, if you were to introduce the usage of *a* and *b* as the output from other operations that do include *tf.Variables*, then the custom gradient would have a gradient relationship on these implicitly. Let's look at a second, more complex example that includes layers, which implicitly define `tf.Variable` instances as the layer parameters.

```python
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
      super().__init__(**kwargs)
      self.dense = tf.keras.layers.Dense(units)

    @tf.custom_gradient
    def call(self, inputs):
      x = self.dense(inputs)
      def grad_fn(dy):
          # Here, dy depends implicitly on parameters in self.dense
          grad_inputs = tf.matmul(dy, tf.transpose(self.dense.kernel)) #Accessing a variable directly for clarity
          return grad_inputs
      return x, grad_fn


# Example usage
input_tensor = tf.random.normal((1, 10))

layer = CustomLayer(5)
with tf.GradientTape() as tape:
  tape.watch(input_tensor) #watch the input as we usually would
  output = layer(input_tensor)
grads = tape.gradient(output, input_tensor)
print(grads)
```

In this second example, even if we attempt to return a gradient calculation purely from the output tensor of the dense layer, we have not included the derivative of the dense kernel. This will result in our *grad\_fn* having a lineage dependency on variables that we are not explicitly returning, or have marked as trainable during the forward pass. In other words, though we didn't explicitly define a `tf.Variable` object, we are still encountering dependency upon the parameters in the dense layer. Let's now demonstrate how we might correctly create our derivative, including the variable parameters from our dense layer.

```python
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units)

    @tf.custom_gradient
    def call(self, inputs):
        x = self.dense(inputs)
        def grad_fn(dy):
            # Corrected grad_fn to include gradients wrt all params
            dense_vars = self.dense.trainable_variables
            grad_dense = tape.gradient(x, dense_vars, output_gradients = dy)
            grad_inputs = tf.matmul(dy, tf.transpose(self.dense.kernel))
            return grad_inputs, *grad_dense #unpack the resulting tensor
        return x, grad_fn


# Example usage
input_tensor = tf.random.normal((1, 10))

layer = CustomLayer(5)

with tf.GradientTape() as tape:
  tape.watch(input_tensor)
  output = layer(input_tensor)

grads = tape.gradient(output, [input_tensor, *layer.dense.trainable_variables])
print(grads)
```

Here in our third example, you will see we have explicitly obtained the trainable parameters of the dense layer, and have included the gradient calculation for each of the parameters, which resolves our previous issue. We have also unpacked the gradients when returning the derivative to include our derivative with respect to the input and with respect to all of the trainable parameters within our dense layer.

The error you're experiencing underscores the often implicit nature of variable tracking in TensorFlow. While you might not directly create `tf.Variable` instances in your `tf.custom_gradient` function, the autograd system captures dependencies to ensure that gradient calculations consider all trainable parameters involved in the forward pass.

For deeper understanding, I recommend these resources:

1. **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** Specifically, the chapter on backpropagation provides a strong foundation for understanding how automatic differentiation works, which underlies these behaviors.
2. **TensorFlow documentation:** Pay particular attention to the sections about `tf.GradientTape`, `tf.custom_gradient`, and how automatic differentiation operates. The documentation provides detailed explanations and examples of the interaction between the `GradientTape` and custom gradients.
3. **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This book offers practical examples and discussions about custom layers and gradient implementation in TensorFlow, which is very useful for understanding how these theoretical aspects play out in practice. It also explains how the variable tracking works in terms of the underlying implementation.

The key takeaway is that even if your gradient function doesn't directly manage variables, the framework's automatic differentiation keeps track of the parameter dependencies. Therefore, your gradient calculations, when using custom gradients and functions involving trainable parameters, must account for *all* such dependencies, including parameters within other layers or operations that have been internally tracked by the tape. Often, explicitly passing trainable variables from layers, or the result of computations that use layers and therefore variables, will help solve this particular problem.
