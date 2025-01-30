---
title: "How to save TensorFlow graph and variables, excluding Adam optimizer variables?"
date: "2025-01-30"
id: "how-to-save-tensorflow-graph-and-variables-excluding"
---
TensorFlow’s default saving mechanism persists all graph components, including optimizer state. This can lead to unnecessarily large checkpoint files and potential incompatibilities when transferring models across different training environments or retraining stages, especially when the Adam optimizer is used and its momentum and velocity variables are not needed. I've frequently faced this issue while deploying models trained in distributed settings and have learned to carefully manage what's persisted. The challenge lies in discerning which variables are integral to the model's architecture and which represent the optimizer's dynamic state, then selectively saving only the former.

The core principle involves filtering the `tf.trainable_variables` collection, which encompasses the weights and biases that directly define your model’s parameters. We then create a `tf.train.Saver` object specifying only this curated set. The Adam optimizer, like other optimizers, maintains additional variables (e.g., `m` and `v` for moving average of gradients and squared gradients) alongside the trainable variables. These optimizer-specific variables are found in the global collection but are not relevant for model inference or transfer learning.

My usual workflow breaks down into the following process: First, after defining your model and optimizer, retrieve all trainable variables. Second, construct a `Saver` object using a dictionary where keys are the original variable names and values are the variables you want to save. Third, ensure to use `saver.save(sess, checkpoint_path)` to store this filtered collection of variables. Loading, likewise, involves building the same dictionary mapping, creating a new saver, and calling `saver.restore(sess, checkpoint_path)`. This approach guarantees that we load and save only our model's weights and biases, excluding the optimizer's state variables. The restoration process ensures that all layers are correctly initialized based on the saved values, effectively decoupling the model structure from its training history. This is beneficial for freezing layers or for fine-tuning purposes.

Here's a demonstration with a simplified model, employing a dense layer and a basic training loop using a placeholder for input:

```python
import tensorflow as tf
import numpy as np

# Define a simple model with a dense layer
def simple_model(inputs):
    W = tf.Variable(tf.random.normal([2, 3]), name='W')
    b = tf.Variable(tf.random.normal([3]), name='b')
    output = tf.matmul(inputs, W) + b
    return output

# Placeholders and model instantiation
inputs = tf.placeholder(tf.float32, shape=(None, 2), name='input_placeholder')
output = simple_model(inputs)
labels = tf.placeholder(tf.float32, shape=(None, 3), name='label_placeholder')

# Loss and Adam optimizer setup
loss = tf.reduce_mean(tf.square(output - labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# Extract trainable variables
trainable_vars = tf.trainable_variables()

# Create the saver using a dictionary comprehension
saver = tf.train.Saver({var.name: var for var in trainable_vars})

# Example training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    input_data = np.random.rand(10, 2)
    label_data = np.random.rand(10, 3)
    
    for i in range(100):
        _, current_loss = sess.run([train_op, loss], 
                                   feed_dict={inputs:input_data, labels:label_data})
        if (i+1) % 20 == 0:
          print(f"Iteration {i+1}, Loss: {current_loss}")

    # Save only model variables
    saver.save(sess, './model/simple_model.ckpt')
    print("Model saved.")

```

In this first example, the code constructs a simple linear model and uses the Adam optimizer. After training, it collects all trainable variables into `trainable_vars`. The `tf.train.Saver` is then created using a dictionary comprehension that maps the variable names to the actual variable tensors.  Critically, only these variables, and not the Adam optimizer's variables, will be included when the `saver.save` is called.  I've found using dictionary comprehensions to be very useful for quickly creating name to variable mappings.  The `print` statements are included to illustrate that the loss is decreasing and to indicate when the saving occurs.  Note the lack of explicit inclusion of optimizer variables in the `Saver`, which implicitly prevents them from being persisted.

Next, I’ll show how to extend this to a multi-layered network, using Keras layers within a TensorFlow context for a more realistic setup. I’ve found integrating Keras layers allows for concise model definitions while leveraging TensorFlow’s runtime engine. The saving process remains structurally identical, which is crucial for consistency across different model architectures.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense

# Define a multi-layered model using Keras layers
def multi_layer_model(inputs):
    x = Dense(16, activation='relu', name='dense1')(inputs)
    output = Dense(3, name='dense2')(x)
    return output

# Placeholders and model instantiation
inputs = tf.placeholder(tf.float32, shape=(None, 2), name='input_placeholder')
output = multi_layer_model(inputs)
labels = tf.placeholder(tf.float32, shape=(None, 3), name='label_placeholder')

# Loss and optimizer setup
loss = tf.reduce_mean(tf.square(output - labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# Extract trainable variables
trainable_vars = tf.trainable_variables()

# Create the saver
saver = tf.train.Saver({var.name: var for var in trainable_vars})


# Example training process and saving (similar to previous example)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    input_data = np.random.rand(10, 2)
    label_data = np.random.rand(10, 3)
    
    for i in range(100):
        _, current_loss = sess.run([train_op, loss], 
                                   feed_dict={inputs:input_data, labels:label_data})
        if (i+1) % 20 == 0:
          print(f"Iteration {i+1}, Loss: {current_loss}")


    saver.save(sess, './model/multi_layer_model.ckpt')
    print("Model saved.")

```

This second code segment defines a more complex model using two Keras `Dense` layers. The core saving mechanism remains the same. The `tf.trainable_variables` is called to retrieve all the model's trainable weights and biases, and the `tf.train.Saver` is constructed to only keep the weights and biases.  It's important to note that even if the keras layers define internal state, these states are automatically managed as `trainable_variables`.  Again, the critical aspect is the careful filtering of variables using the dictionary comprehension to create the Saver.  This example highlights the consistency of the variable filtering approach across model complexities.

Finally, demonstrating how to load back these models is crucial. Here's the example for loading and using the previously saved `simple_model`.  The loading process is designed to exactly mirror the saving process; that is, you must load back variables that the saver is aware of.

```python
import tensorflow as tf
import numpy as np

# Define the same simple model for loading
def simple_model(inputs):
    W = tf.Variable(tf.random.normal([2, 3]), name='W')
    b = tf.Variable(tf.random.normal([3]), name='b')
    output = tf.matmul(inputs, W) + b
    return output

# Placeholders and model instantiation
inputs = tf.placeholder(tf.float32, shape=(None, 2), name='input_placeholder')
output = simple_model(inputs)

# Extract trainable variables
trainable_vars = tf.trainable_variables()

# Create the saver
saver = tf.train.Saver({var.name: var for var in trainable_vars})


# Example loading and inference
with tf.Session() as sess:
    # Initialize global variables (important when loading)
    sess.run(tf.global_variables_initializer())

    # Restore only model variables
    saver.restore(sess, './model/simple_model.ckpt')
    print("Model loaded.")
    
    # Perform inference with new input data
    test_input = np.random.rand(5, 2)
    test_output = sess.run(output, feed_dict={inputs: test_input})
    print("Inference output:", test_output)

```

This final example illustrates how to load the previously saved `simple_model`. It's crucial to call `tf.global_variables_initializer()` before attempting to load variables.  The core loading mechanism matches the saving mechanism. The Saver is created with the same variable dictionary, and when the `restore` method is called, only the weights and biases of the model will be loaded. This guarantees the original model state is restored for inference. The code demonstrates a basic prediction process with some test inputs, showcasing the functionality of loading and inference on the restored model.

For further exploration, I would recommend studying TensorFlow’s official documentation regarding `tf.train.Saver`, `tf.trainable_variables`, and the details around various optimizer implementations. Books dedicated to TensorFlow and deep learning model deployment will also contain valuable information on checkpoint management and efficient model loading. Additionally, examining example model implementations from TensorFlow’s Model Garden or other open-source projects will provide valuable practical insights.
