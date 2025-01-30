---
title: "How does TensorFlow handle untrainable layers?"
date: "2025-01-30"
id: "how-does-tensorflow-handle-untrainable-layers"
---
TensorFlow does not truly "handle" untrainable layers as a separate mechanism, but rather it leverages its core computational graph structure and gradient propagation algorithms to effectively bypass them during the backpropagation phase of training. Untrainable layers, typically implemented by setting the `trainable` attribute of a layer or a variable to `False`, are treated differently only when gradients are computed; their forward pass operates identically regardless of their training status. I've encountered this behavior extensively, particularly while fine-tuning pre-trained models where a significant portion of the network is deliberately frozen.

The fundamental concept here is that TensorFlow's automatic differentiation engine only calculates gradients for trainable variables. During the forward pass, the output of each layer, whether trainable or not, is computed and propagated through the network. However, during backpropagation, the gradients are calculated using the chain rule, flowing backward from the loss function. If a layer is marked as untrainable, its weights and biases are effectively treated as constants for the purposes of gradient calculation. The gradient simply bypasses these layers, preventing any updates to their internal parameters. This is not a special treatment *per se* but a consequence of how gradients are computed over the computational graph and conditional application of the automatic differentiation algorithm.

To illustrate, consider the common scenario of employing a pre-trained convolutional neural network (CNN) as a feature extractor. We might want to freeze the convolutional base and train only the newly added fully connected layers for our specific task. This is achieved by selectively setting the `trainable` attribute of the model's layers. When `trainable = False`, the weights of a layer remain unaltered regardless of optimization passes over training data, preserving the knowledge learned during the initial training of the model.

Let's examine some code examples to clarify this.

**Example 1: Freezing a Single Layer**

```python
import tensorflow as tf

# Define a simple dense layer
layer = tf.keras.layers.Dense(units=10, activation='relu')

# Set the layer to untrainable
layer.trainable = False

# Example input
input_data = tf.random.normal(shape=(1, 5))

# Forward pass
output = layer(input_data)

# Display layer weights and biases
print("Initial weights:", layer.weights[0].numpy())
print("Initial biases:", layer.weights[1].numpy())

# Assume a dummy loss and compute gradients
with tf.GradientTape() as tape:
  loss = tf.reduce_sum(output)

gradients = tape.gradient(loss, layer.weights)

# Confirm no gradients are produced for the untrainable layer
print("Gradients (should be None):", gradients)
```

In this snippet, a dense layer is created, and its `trainable` attribute is set to `False`. After a forward pass, we calculate the gradients using `tf.GradientTape`. The output confirms that gradients are `None` for the untrainable layer's weights and biases. This demonstrates how TensorFlow's gradient calculation mechanism effectively skips over non-trainable layers. Despite the forward pass still involving the layer's calculations, the optimization process will not attempt to modify them because of the missing gradient.

**Example 2: Freezing Multiple Layers in a Sequential Model**

```python
import tensorflow as tf

# Define a Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Freeze the first two layers
model.layers[0].trainable = False
model.layers[1].trainable = False


# Example input
input_data = tf.random.normal(shape=(1, 100))

# Forward pass
output = model(input_data)

# Display weights of the model before any training
print("Model weights before training:\n", [layer.weights[0].numpy() for layer in model.layers])

# Assume dummy loss and training steps
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for _ in range(2):
    with tf.GradientTape() as tape:
       predictions = model(input_data)
       loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        tf.one_hot(tf.random.uniform([1], minval=0, maxval=10, dtype=tf.int32), 10), predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Display weights after pseudo-training
print("Model weights after pseudo-training:\n", [layer.weights[0].numpy() for layer in model.layers])
```

This example creates a sequential model with three dense layers.  The first two layers are then marked as untrainable.  A forward pass is performed, followed by a dummy training loop. We then compare the weights before and after this training. The weights of the first two layers remain unchanged while the weights of the third layer are altered during gradient-based training, further illustrating that `trainable=False` prevents update during backpropagation. This underscores that `trainable` attribute dictates whether parameter updates occur during backpropagation and that the layers still operate during the forward pass.

**Example 3: Using a Functional API Model with Frozen Layers**

```python
import tensorflow as tf

# Define input layer
inputs = tf.keras.Input(shape=(10,))

# Define functional model
x = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
x = tf.keras.layers.Dense(units=32, activation='relu')(x)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Freeze specific layers, access via names if needed
model.get_layer(index = 1).trainable = False # First dense layer
model.get_layer(index = 2).trainable = False # Second dense layer

# Example input
input_data = tf.random.normal(shape=(1, 10))

# Forward pass
output = model(input_data)

# Display weights of the model before any training
print("Model weights before training:\n", [layer.weights[0].numpy() for layer in model.layers[1:]])

# Assume dummy loss and training steps
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
with tf.GradientTape() as tape:
   predictions = model(input_data)
   loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.constant([[0.]]), predictions))
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Display weights after pseudo-training
print("Model weights after pseudo-training:\n", [layer.weights[0].numpy() for layer in model.layers[1:]])
```

This example demonstrates using the Functional API and explicitly freezing layers based on index.  Again, weights of the frozen layers are unchanged after gradient updates demonstrating that this `trainable` attribute also applies to models created using Functional API and not just sequential models. This flexibility further allows for more targeted adjustment of training parameters.

In summary, TensorFlow doesn't manage untrainable layers as an exceptional case, but rather relies on its underlying automatic differentiation system. By setting the `trainable` attribute to `False`, the layer’s parameters are excluded from the gradient calculation and update process. This enables flexible and efficient handling of pre-trained models, feature extraction, and other scenarios where specific parts of a network should remain constant.

For further information, the TensorFlow website provides comprehensive documentation on `tf.keras.layers.Layer`, including the `trainable` property. I would strongly suggest reviewing the guides on automatic differentiation and gradient tape. Furthermore, the API documentation for `tf.GradientTape` should be reviewed to understand how to perform automatic differentiation. I also suggest reading about the implementation of optimizers, as these algorithms use the `trainable_variables` provided by a model to specifically update weights that are marked as trainable. I would also recommend exploring research on fine-tuning methodologies for a greater understanding of where freezing layers is most applicable. These resources are comprehensive and contain more detailed information about this topic.
