---
title: "How do I convert TensorFlow 1.x code to TensorFlow 2.x?"
date: "2025-01-30"
id: "how-do-i-convert-tensorflow-1x-code-to"
---
TensorFlow 2.x fundamentally alters the execution model, moving from a graph-based, session-oriented approach in 1.x to an eager execution paradigm. This necessitates substantial code refactoring, impacting everything from variable management to control flow operations. Having personally migrated several large-scale TensorFlow 1.x models, I’ve found that approaching this conversion systematically, with an understanding of the core changes, significantly reduces potential errors and debugging time.

The primary challenge lies in the elimination of the concept of explicit `tf.Session` objects. In TensorFlow 1.x, you defined a computational graph and then launched a session to execute it. TensorFlow 2.x executes operations directly as they're called, simplifying debugging and development but demanding alterations in the code architecture. This change particularly impacts how variables, placeholders, and control flow are handled.

A common pitfall is the reliance on placeholders (`tf.placeholder`) to feed data into the graph. In 2.x, you generally utilize TensorFlow's Dataset API for data input. This change, however, is not simply replacing placeholders with dataset iterators; it implies a shift to batch-oriented data processing and often requires data preprocessing to be incorporated into the dataset pipeline. Additionally, manually managed variables (`tf.Variable`) need updating to use the object-oriented classes, like `tf.Variable` initialized within a `tf.Module` or a model class based on `tf.keras.Model`.

The concept of TensorFlow scopes (via `tf.name_scope` and `tf.variable_scope`) is greatly diminished. While `tf.name_scope` still exists for visualization and debugging, `tf.variable_scope` is largely irrelevant due to the eager execution paradigm. Variable sharing is now managed through class hierarchies, instance variables, or direct variable creation. Legacy code that depends on `variable_scope` for variable re-use will likely need a significant refactor.

Below, I'll present three common migration scenarios and code examples, focusing on the key transformations:

**Example 1: Replacing Placeholders with `tf.data.Dataset`**

TensorFlow 1.x code might look like this, using placeholders:

```python
# TensorFlow 1.x style

import tensorflow as tf

# Define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_hat = tf.nn.softmax(y)
y_true = tf.placeholder(tf.float32, shape=[None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Dummy data
data_x =  [[float(i) for i in range(784)] for _ in range(1000)]
data_y =  [[float(i % 10 == j) for j in range(10)] for i in range(1000)]

# Start session, iterate and feed data
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10):
     _, current_loss = sess.run([optimizer, loss], feed_dict={x:data_x, y_true:data_y})
     print(f'Iteration {i}: Loss = {current_loss}')
```

In the TensorFlow 2.x equivalent, we utilize `tf.data.Dataset`:

```python
# TensorFlow 2.x style
import tensorflow as tf

# Build dataset
data_x =  [[float(i) for i in range(784)] for _ in range(1000)]
data_y =  [[float(i % 10 == j) for j in range(10)] for i in range(1000)]

dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).batch(32)

# Define model
W = tf.Variable(tf.random.normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

@tf.function
def compute_loss(x, y_true):
    y = tf.matmul(x, W) + b
    y_hat = tf.nn.softmax(y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
    return loss

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# training loop
for i in range(10):
   for x_batch, y_true_batch in dataset:
     with tf.GradientTape() as tape:
        current_loss = compute_loss(x_batch, y_true_batch)

     grads = tape.gradient(current_loss, [W, b])
     optimizer.apply_gradients(zip(grads, [W,b]))
   print(f'Iteration {i}: Loss = {current_loss.numpy()}')
```

Here, the placeholder inputs `x` and `y_true` are replaced by a `tf.data.Dataset` instance. The data is batched, and each iteration now consumes a batch of data instead of feeding the entire dataset. Eager execution is employed, so there is no session. The loss and gradients are computed within a gradient tape, which is essential for training in TensorFlow 2. `tf.function` enhances performance by compiling the `compute_loss` into a TensorFlow graph. Finally, we explicitly invoke `numpy()` for printing as the result is a tensor object.

**Example 2: Refactoring `tf.variable_scope` Usage**

TensorFlow 1.x often used `tf.variable_scope` for variable sharing:

```python
# TensorFlow 1.x style

import tensorflow as tf

def dense_layer(x, units, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[x.shape[1], units], initializer=tf.random_normal_initializer())
        b = tf.get_variable("b", shape=[units], initializer=tf.zeros_initializer())
        return tf.matmul(x, W) + b

x = tf.placeholder(tf.float32, shape=[None, 10])
y1 = dense_layer(x, 20, "dense1")
y2 = dense_layer(y1, 30, "dense2")
y3 = dense_layer(y2, 10, "dense1") # Using same scope as y1, reusing variables

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y3, feed_dict={x: [[1.0]*10]}))
    print(sess.run(tf.report_uninitialized_variables()))
```

In TensorFlow 2.x, we should utilize class structures to achieve variable sharing. Here’s an example implementing layers using `tf.keras.layers.Layer`:

```python
# TensorFlow 2.x style
import tensorflow as tf

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DenseLayer, self).__init__()
        self.units = units
        self.W = self.add_weight(shape=(None, units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs):
      self.W = tf.Variable(tf.random.normal([inputs.shape[1], self.units]))
      self.b = tf.Variable(tf.zeros([self.units]))
      return tf.matmul(inputs, self.W) + self.b

x = tf.constant([[1.0]*10])
dense1 = DenseLayer(20)
y1 = dense1(x)
dense2 = DenseLayer(30)
y2 = dense2(y1)

dense1_reused = DenseLayer(10)
y3 = dense1_reused(y2)

print(y3)
print([v.name for v in dense1_reused.trainable_variables])
```

In this version, we replace the `tf.variable_scope` and `tf.get_variable` mechanism with `tf.keras.layers.Layer`. Each layer instance manages its own weights. In this example, the final `dense1_reused` instance is a new layer, so the variables will not be shared automatically, like in the 1.x example. If desired, we can instantiate and reuse the original `dense1` object. We also explicitly use `tf.Variable` and assign them in `call()`. This demonstrates that we are no longer depending on `variable_scope`.

**Example 3: Transforming `tf.control_dependencies`**

TensorFlow 1.x used `tf.control_dependencies` for imposing execution order, particularly for operations with side effects.

```python
# TensorFlow 1.x style
import tensorflow as tf

var = tf.Variable(0)
add_op = tf.assign_add(var, 1)

with tf.control_dependencies([add_op]):
    read_op = tf.identity(var)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(read_op))  # Will output 1
```

In TensorFlow 2.x, control flow is generally handled within the eager execution context, or more complex control flow is handled using `tf.cond`, `tf.while_loop` or simply Python. Often, operations with side effects can be executed in their natural order within Python itself. The example below directly executes the `add_op` before reading the variable.

```python
# TensorFlow 2.x style
import tensorflow as tf

var = tf.Variable(0)
add_op = var.assign_add(1)
add_op.numpy() # execute operation to modify variable

read_op = tf.identity(var)
print(read_op.numpy())
```

Here, the update is executed eagerly by applying `numpy()` to the result of the `assign_add` operation; there is no explicit control dependency needed.

**Resource Recommendations:**

For a thorough understanding of the migration process, focus on the TensorFlow official documentation. Specifically, review the guides on "Migrate from TF1 to TF2", "Eager execution," and the "Datasets API". The TensorFlow API documentation (tensorflow.org/api_docs) is essential for understanding the available operations and their new behaviors. Additionally, tutorials on using `tf.keras.Model` and custom layer implementations can be beneficial. Finally, exploring publicly available TensorFlow 2.x codebases offers practical insights into best practices.

The transition from TensorFlow 1.x to 2.x can present initial challenges, but understanding the fundamental shifts and the new APIs enables more efficient and streamlined development processes. Careful planning, incremental updates, and a thorough understanding of TensorFlow 2’s execution paradigm are vital for successful model migration.
