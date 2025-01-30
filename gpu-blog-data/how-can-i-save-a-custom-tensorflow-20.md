---
title: "How can I save a custom TensorFlow 2.0 model (not Keras)?"
date: "2025-01-30"
id: "how-can-i-save-a-custom-tensorflow-20"
---
TensorFlow 2.0 allows saving custom models, beyond those constructed using the high-level Keras API, through the `tf.saved_model` module. This requires careful consideration of the model's architecture and the operations involved in its forward pass, specifically how TensorFlow can trace and recreate that computation during loading. Unlike Keras models, which have pre-defined saving mechanisms, custom models demand explicit definition of the saving and loading procedures.

My experience with building models for research often necessitates moving beyond Keras's predefined layers and structures. I’ve found that a strong understanding of TensorFlow's graph execution and autograph is crucial for successful custom model persistence. The core principle lies in the fact that TensorFlow needs to 'understand' how your model operates in order to serialize it. This 'understanding' is built by tracing the model's forward pass. This means your custom model, typically inheriting from `tf.Module`, must implement the call method. The call method acts as the point where Tensorflow’s auto graph functionality will trace your model. The parameters used during the call method will be tracked as input signatures for later use when loading the model. Any tensors created, modified, or accessed within the scope of this method, will be tracked and their values can be reloaded. In essence, the call method will be the core of our model when we want to load and deploy it.

Let’s dive into a practical demonstration. The simplest custom model is one that performs a matrix multiplication, with the matrix as a learnable parameter. Here’s how you’d define and save such a model:

```python
import tensorflow as tf

class CustomDense(tf.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomDense, self).__init__()
        self.w = tf.Variable(tf.random.normal([input_dim, output_dim]), name='weights')
        self.b = tf.Variable(tf.zeros([output_dim]), name='bias')


    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

# Instantiate the model
dense_model = CustomDense(input_dim=5, output_dim=3)

# Create a dummy input
dummy_input = tf.random.normal((1, 5))

# Run a forward pass so that the weights are initialized
dense_model(dummy_input)

# Save the model
tf.saved_model.save(dense_model, "custom_dense_model")

```

In this snippet, the key part lies in defining the `__call__` method, which now includes an `input_signature`.  I have found `input_signature` to be the most reliable method for saving models that are not Keras models.  The `input_signature` specifies the shape and type of the expected input tensor. This allows TensorFlow to statically define the model's input format during loading.  By using `tf.TensorSpec`, we declare that the input `x` will be a tensor of arbitrary batch size but have the second dimension as whatever is needed, and the datatype as `tf.float32`. The `tf.function` decorator ensures that the function is traced into a graph for efficient execution and saving. After calling the model to ensure all necessary tensors are initialised, the model is saved using `tf.saved_model.save`. Note that if you do not perform an initial forward pass, the weights won't be created and saved.

The loading procedure mirrors saving and it's crucial to be familiar with it to confirm whether your saving method is successful. Here’s how we’d load the `CustomDense` model we just saved and use it:

```python
# Load the saved model
loaded_dense_model = tf.saved_model.load("custom_dense_model")

# Create new input
new_input = tf.random.normal((2, 5))

# Run inference with loaded model
output = loaded_dense_model(new_input)

# Print the output to verify correct loading
print(output)

```

As you can see, loading the model is as simple as using `tf.saved_model.load()`, which automatically loads the saved graph. The loaded `loaded_dense_model` has the same call method with the same input signature we defined. This is crucial because this function was traced and is what is being run by the loaded model. We are passing the new data through the loaded model and we expect that the calculations we previously saved were loaded, and now performed.

Now, let’s explore a slightly more complex model that introduces a custom operation that is not a Keras layer. Imagine a model that takes a tensor and applies a custom activation function, that involves exponentiating values based on a learnable parameter.

```python
import tensorflow as tf

class CustomActivationModel(tf.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.exponent = tf.Variable(tf.random.normal([]), name='exponent') #scalar learnable parameter

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def __call__(self, x):
        return tf.pow(tf.math.abs(x), self.exponent)

# Instantiate the model
activation_model = CustomActivationModel()

# Create dummy input
dummy_input = tf.random.normal((1, 4))

# Run an initial forward pass for weight initialisation
activation_model(dummy_input)

# Save the model
tf.saved_model.save(activation_model, "custom_activation_model")
```

Here, the `exponent` is the learnable parameter, and `tf.pow` performs the operation. Again, the key is the `@tf.function` decorator and the `input_signature` specified in the `__call__` method. Tensorflow traces this operation, so that when it is loaded it is able to recreate this computation from the information stored in the saved model. The saving procedure is identical to the previous example, showing the adaptability of the `tf.saved_model` API.

Loading this model and using it works as expected:

```python
# Load the saved model
loaded_activation_model = tf.saved_model.load("custom_activation_model")

# Create new input
new_input = tf.random.normal((2, 4))

# Run inference
output = loaded_activation_model(new_input)

# Print result to confirm correct operation
print(output)
```

These examples demonstrate the core principles of saving custom TensorFlow 2.0 models. The focus should always be on implementing the `__call__` method correctly with an appropriate `input_signature` specified and that the necessary calculations that we want to be saved are performed in the forward pass that Tensorflow is tracing. While this approach requires a lower-level understanding of TensorFlow than using Keras, it allows for greater flexibility in model design and operations. This provides crucial control when working with custom models, especially when dealing with research or other specialized applications.

For further learning, the TensorFlow official documentation on `tf.saved_model` offers a comprehensive guide. Additionally, exploring the details of autograph and graph execution within TensorFlow is helpful.  Also, the TensorFlow tutorials often include examples of custom models, and studying these will reveal specific design decisions and good coding practices. Additionally, a deep dive into tensor specifications for defining input formats is also useful. These resources provide a deeper theoretical understanding and practical examples.
