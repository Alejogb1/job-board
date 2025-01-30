---
title: "How to obtain a tensor's shape during TensorFlow training?"
date: "2025-01-30"
id: "how-to-obtain-a-tensors-shape-during-tensorflow"
---
During my years optimizing TensorFlow models for high-throughput inference, reliably accessing tensor shapes mid-training became a frequent necessity for dynamic graph construction and debugging. The core challenge arises because TensorFlow operates on symbolic graphs, and tensor shapes often aren't fully known until execution. Directly printing the shape of a placeholder or intermediate tensor during graph construction typically yields `TensorShape(None, None, ...)` or similar representations with `None` dimensions, indicating runtime resolution. Accessing the concrete shape during the actual training process, however, requires a different approach.

The key lies in using the functions and methods provided by TensorFlow's eager execution mode or within a graph execution context (such as training within a `tf.function`). We can retrieve the shape as a `tf.Tensor` object, which can be evaluated using `numpy` or utilized for further TensorFlow operations. The concrete shape manifests at the point where the tensor's value is resolved during execution. This means we cannot access the shape before the tensor has been computed. Consequently, we must operate on these shape tensors during training in either of these execution contexts.

Let's explore how this looks in practice. For operations within an eager execution context (where computations happen immediately, rather than deferred to a graph), we can use `tensor.shape` directly, converting it to a NumPy array for ease of access.

```python
import tensorflow as tf
import numpy as np

# Example 1: Shape retrieval during eager execution
@tf.function # convert to a tf function
def train_step(images, labels, model):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    shape_tensor = tf.shape(images)  # Obtain the shape as a Tensor
    shape_array = shape_tensor.numpy()  # Convert it to numpy for display
    tf.print("Image Tensor Shape:", shape_array)

    return loss

#dummy inputs
batch_size = 32
image_height = 28
image_width = 28
num_channels = 3

num_classes = 10
learning_rate = 0.001
epochs = 2

images = tf.random.normal((batch_size, image_height, image_width, num_channels))
labels = tf.one_hot(tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32), depth = num_classes)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(image_height,image_width, num_channels)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate)

# Train and print shapes for several epochs
for i in range(epochs):
    loss = train_step(images, labels, model)
    print('loss: ',loss)
```

In this example, I’ve wrapped the operations in a `tf.function`, which will execute in graph mode to improve performance. Within the `train_step` function, the tensor `images` has a specific, batch size-dependent shape by the time we call `tf.shape`. This operation returns a `tf.Tensor` representing the shape, which I convert into a NumPy array for easy printing. The important aspect here is that `tf.shape(images)` does not return the dimensions themselves but a tensor that, when evaluated, resolves to the shape. It’s crucial to use `numpy()` or `tf.print` (for printing) to view the resolved shape.

Now consider a more complex scenario involving a custom training loop, where a model's output shapes during intermediate steps become vital for analysis.

```python
import tensorflow as tf
import numpy as np

# Example 2: Shape inspection within a custom training loop
def build_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    shape_tensor = tf.shape(x)
    shape_tensor_reshaped = tf.reshape(shape_tensor, [1, tf.shape(shape_tensor)[0]])
    tf.print('shape of conv output', shape_tensor_reshaped)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    shape_tensor_max = tf.shape(x)
    shape_tensor_max_reshaped = tf.reshape(shape_tensor_max, [1, tf.shape(shape_tensor_max)[0]])
    tf.print('shape of max pool output', shape_tensor_max_reshaped)

    x = tf.keras.layers.Flatten()(x)
    shape_tensor_flat = tf.shape(x)
    shape_tensor_flat_reshaped = tf.reshape(shape_tensor_flat, [1, tf.shape(shape_tensor_flat)[0]])
    tf.print('shape of flat output', shape_tensor_flat_reshaped)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

#dummy inputs
batch_size = 32
image_height = 28
image_width = 28
num_channels = 1

num_classes = 10
learning_rate = 0.001
epochs = 2

images = tf.random.normal((batch_size, image_height, image_width, num_channels))
labels = tf.one_hot(tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32), depth = num_classes)
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
      predictions = model(images)
      loss = tf.keras.losses.categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for i in range(epochs):
   loss = train_step(images, labels)
   print('loss: ', loss)

```

This second example showcases how to inspect intermediate layer output shapes. The `build_model` function now incorporates `tf.shape` calls directly to analyze the output of the convolution and max pool layers, providing insight into how their shape evolves during the model's forward pass. Importantly, these are all operations that occur during graph execution, so the shapes are resolved only during the execution of `train_step`. I utilize a reshape in the function in order to format the tensor shape and the `tf.print` function is used for viewing the shape.

Lastly, let us consider a third example where we can dynamically adjust the shape of a tensor within the training loop based on its size, demonstrating a practical application of runtime shape information.

```python
import tensorflow as tf
import numpy as np


# Example 3: Shape-based tensor manipulation
@tf.function
def process_tensor(tensor):
  shape = tf.shape(tensor)
  # Use conditional logic based on the tensor's shape

  if tf.size(shape) == 2: # if tensor is 2D
      # do some operation
      resized_tensor = tf.reshape(tensor, [shape[0] * shape[1], 1])
  elif tf.size(shape) == 3:
      resized_tensor = tf.reshape(tensor, [shape[0], shape[1] * shape[2]])
  else:
       resized_tensor = tensor
  tf.print('shape of input tensor', tf.shape(tensor))
  tf.print('shape of resized tensor', tf.shape(resized_tensor))
  return resized_tensor


batch_size = 64
dimensions = 10
# Create dummy input tensors of different sizes
tensor1 = tf.random.normal((batch_size, dimensions))
tensor2 = tf.random.normal((batch_size, dimensions, dimensions))
tensor3 = tf.random.normal((batch_size, dimensions, dimensions, dimensions))

# process and print each tensor shape
tensor1_output = process_tensor(tensor1)
tensor2_output = process_tensor(tensor2)
tensor3_output = process_tensor(tensor3)
```

This example demonstrates how to use the shape information to perform different operations based on the shape. I’m using the size of the shape tensor to make conditional operations.  Here, the `process_tensor` function reshapes the input tensor based on whether it's 2D or 3D using `tf.reshape`, offering an example of shape-dependent tensor manipulation. Crucially, this resizing occurs based on the dynamically retrieved shape, showcasing the ability to leverage shapes for adaptable data processing during training.

In summary, while TensorFlow operates on a symbolic graph, we can access the concrete shapes of tensors when they are evaluated during eager execution or within a graph execution context. The `tf.shape` function provides a `tf.Tensor` that resolves to the actual tensor's shape during training or prediction. To utilize this information, one can convert the `tf.Tensor` to a NumPy array using `numpy()` or use `tf.print` to directly print the tensor shape in the graph. This ability becomes essential for debugging and performing dynamic operations dependent on tensor shapes, enabling robust and flexible model development.

For further learning, I recommend consulting the official TensorFlow documentation, particularly sections on eager execution, graph mode, and the `tf.shape` operation, in addition to resources on tensor manipulation. Explore the API for `tf.print` and `tf.Tensor.numpy()` in the TensorFlow library reference. A deeper understanding of the differences between eager and graph modes is also crucial for effectively using this technique. Experimentation and application in diverse projects will further solidify your proficiency in handling tensor shapes within TensorFlow workflows.
