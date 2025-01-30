---
title: "Why can't a sequential model be used in TensorFlow?"
date: "2025-01-30"
id: "why-cant-a-sequential-model-be-used-in"
---
The fundamental limitation preventing the direct use of a sequential model for certain types of problems in TensorFlow stems from its architectural constraints: a sequential model assumes a single, linear flow of information from input to output. This fixed topology becomes inadequate when faced with situations requiring multiple input branches, skip connections, shared layers, or dynamic graph structures. My years working on diverse deep learning projects have repeatedly highlighted this.

The core of the issue lies in the sequential model's inherent structure. It's essentially a stack of layers, where the output of one layer directly feeds into the input of the next. This chain of operations is defined during model construction and remains static throughout the training and inference phases. This simplicity is an advantage for basic classification or regression tasks where the data processing follows a linear path. However, real-world applications often necessitate far more complex architectures, which cannot be easily, or sometimes at all, expressed using a purely sequential model.

Consider image segmentation, for instance. A common architecture is an encoder-decoder with skip connections. The encoder reduces the spatial resolution of the input image, extracting features at different scales. The decoder then upsamples these features, creating a segmentation mask. Crucially, skip connections pass feature maps from the encoder directly to corresponding layers in the decoder. These connections, which allow for the preservation of fine-grained detail, violate the sequential nature required by the sequential API. The decoder requires not only the output from the preceding decoder layer but also the output from a layer earlier in the encoder. This multi-source input pattern is not supported.

Another example is recurrent neural networks (RNNs) with attention mechanisms. Attention modules typically need access to the output of all previous steps of the RNN and, sometimes, specific hidden states, which isn't a purely sequential process. This requires flexibility in data flow and a graph structure where connections can be established dynamically.

To illustrate these limitations, consider these three code examples.

**Example 1: Linear Regression using Sequential API**

This example demonstrates the appropriate use of the sequential API. The linear regression model, with a single dense layer, fits perfectly within the sequential paradigm.

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [-1.0,  0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
```

In this instance, the data flows directly through the `Dense` layer, resulting in a prediction. The API is well-suited for this type of forward-propagation. However, this simplicity is a key weakness in more complex use cases. The model's inputs are explicitly defined when creating the layer and the subsequent output becomes the input to the next layer.

**Example 2: Attempting to create a skip connection with Sequential API (Incorrect)**

This code demonstrates why the sequential API fails for architectures with skip connections.

```python
import tensorflow as tf

input_tensor = tf.keras.layers.Input(shape=(10,))
layer1 = tf.keras.layers.Dense(units=5)(input_tensor)
layer2 = tf.keras.layers.Dense(units=5)(layer1)

# Attempt to create a skip connection. This doesn't work in sequential API
# layer3 = tf.keras.layers.concatenate([layer2, input_tensor])
# layer4 = tf.keras.layers.Dense(units=2)(layer3)

# Attempt to create a skip connection. This doesn't work in sequential API
# The sequential model inherently connects the output of a layer to the next
# layer. You cannot skip ahead.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=5, input_shape=(10,)),
    tf.keras.layers.Dense(units=5), # Here the input to this layer is assumed to be the output of the previous layer
    # tf.keras.layers.concatenate([tf.keras.layers.Dense(units=5), tf.keras.layers.Dense(units=5, input_shape=(10,))]) # This is invalid
])

#This leads to issues as we cannot concatenate the input tensor
#  This will throw an error: ValueError: A merge layer should be called on a list of inputs
# The Sequential API fundamentally cannot create the merge that skip connections require.


#We would use the functional API to build the model.
```

The commented-out lines and the attempt to create a concatenation demonstrates the core problem. The sequential API only allows for data to flow in a linear fashion between layers. While the code does not execute the problematic lines, the intended functionality is made clear. Adding a layer to concatenate the `input_tensor` with the `layer2` output is not a sequential operation and thus fails.

**Example 3: Creating a simple model using the Functional API (Correct)**

This example showcases how the functional API is used to overcome the limitation of the sequential API. It recreates the basic linear flow as before, but importantly shows how flexible the API is. This approach is needed to create complex models with more than simple sequential operations.

```python
import tensorflow as tf

input_tensor = tf.keras.layers.Input(shape=(1,))
layer1 = tf.keras.layers.Dense(units=1)(input_tensor)

model = tf.keras.models.Model(inputs=input_tensor, outputs=layer1)

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [-1.0,  0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
```

Here, instead of using a `Sequential` model, we define an input tensor using `Input()`, create a dense layer, and explicitly specify the input and output tensors when constructing the `Model`. This may seem like a more verbose way of writing the previous example, but is necessary to introduce complex branching and skip connections, which is not possible in the sequential paradigm. This flexibility allows us to construct models that the Sequential API cannot handle.

In summary, the sequential model is a specific instance of model creation within the TensorFlow ecosystem; it is designed for a highly specific type of computational graph. While simple to use, its simplicity is also its greatest limitation. The inability to handle non-sequential data flows, especially those arising from skip connections, branched inputs, and shared layers, renders it inadequate for a vast number of relevant deep-learning problems. The functional API offers the necessary flexibility to address these challenges, enabling users to craft intricate, custom model architectures.

For further understanding, I would recommend exploring the following: the TensorFlow official documentation, particularly the sections on model building with both sequential and functional APIs; various tutorials demonstrating advanced model architectures, such as U-Nets or ResNets, which use skip connections, and research papers introducing specific model designs that often require complex computational graphs. These resources provide a thorough grounding in TensorFlow and a practical understanding of the issues at hand. They also delve into the specifics of alternative approaches for model design. Finally, examining real-world example projects that use the functional API can help to concretize the application of these techniques, as the best way to learn is often by doing.
