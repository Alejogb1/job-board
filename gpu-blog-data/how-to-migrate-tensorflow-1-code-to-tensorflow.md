---
title: "How to migrate TensorFlow 1 code to TensorFlow 2?"
date: "2025-01-30"
id: "how-to-migrate-tensorflow-1-code-to-tensorflow"
---
TensorFlow 2's shift to eager execution by default fundamentally alters how computations are structured and executed compared to TensorFlow 1’s graph-based paradigm. This transition necessitates a careful migration strategy to avoid unexpected behavior and ensure the continued functionality of existing models. Having spent the last two years migrating several large-scale machine learning projects from TensorFlow 1.x, I’ve observed several consistent pain points and best practices. The core challenge stems from the removal of global state management in TF2, requiring explicit dependency tracking and changes in API usage.

The primary task in migrating TF1 to TF2 involves transitioning from graph construction through `tf.Session()` and placeholders to the more intuitive eager execution, where operations are executed immediately. TensorFlow 2 encourages a functional, object-oriented approach using layers and models as classes inherited from `tf.keras.Model`, directly handling the construction of the computational graph within the forward pass (`call` method) and leveraging auto-differentiation for gradient computation. This eliminates the need to explicitly declare input placeholders and run a `Session`.

The initial phase of migration should focus on evaluating the existing TF1 code for usage of features that are deprecated or have changed in TF2. Specifically, these include:

*   **`tf.compat.v1`:** TensorFlow 2 retains access to many TF1 API elements through the `tf.compat.v1` module. While this can offer a quick path to getting code working initially, relying heavily on it will impede fully transitioning to the intended TF2 paradigm and can introduce future maintenance issues. It's best treated as a temporary bridge, with a plan to refactor.
*   **Placeholders and `tf.Session`:** These elements no longer directly exist in TF2. Placeholders are replaced by direct tensor inputs in eager mode or input layers in `keras` models, and `tf.Session` is rendered obsolete by immediate execution.
*   **Variable Scope and `tf.get_variable`:** Variable creation now primarily leverages `tf.Variable` objects and the instantiation mechanisms of `keras.layers` and `keras.models`. The legacy `variable_scope` pattern should be rewritten.
*   **Control Flow Operations:**  Operations like `tf.while_loop` and `tf.cond`, often used for dynamic batching or variable length sequences, require careful review. TF2 provides improved alternatives or can sometimes be replaced by standard Python control flow in eager execution scenarios.
*   **Global Variable Initialization:** Initializers must now be called explicitly and before using the variables, often using the model construction or layer initializations, as opposed to the `tf.global_variables_initializer()` in TF1.
*   **Data Input Pipeline:** TF1’s `tf.data.Dataset` and queue-based input pipelines require scrutiny. TF2 provides generally compatible implementations of the data pipeline but some configurations of legacy systems may require adjustment.
*   **Summaries and Logging:** TensorBoard usage requires adaptations given the eager and functional nature of TensorFlow 2.

With these points in mind, the following examples illustrate common migration paths:

**Example 1: Transition from `tf.Session` to Eager Execution**

*TensorFlow 1.x Code:*

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # ensure it is TF1 session style
a = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
b = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
c = tf.add(a, b)
with tf.compat.v1.Session() as sess:
    result = sess.run(c, feed_dict={a: [[1, 2], [3, 4]], b: [[5, 6], [7, 8]]})
    print(result)
```

*TensorFlow 2.x Code:*

```python
import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
c = tf.add(a, b)
print(c.numpy())
```

*Commentary:* The TF1 code establishes placeholders `a` and `b` in the computational graph. The actual addition only takes place during the `sess.run()` call, with data provided through `feed_dict`. The TF2 code directly uses `tf.constant` to create tensors and performs the addition immediately. The result is accessed with the `.numpy()` method. This highlights the shift from graph construction and deferred execution to direct, eager computation. No session or placeholders are required.

**Example 2: Converting a Simple Model with `tf.get_variable` to Keras Model**

*TensorFlow 1.x Code:*

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
def simple_model(x):
    w = tf.compat.v1.get_variable("weight", shape=[2, 2], initializer=tf.random_normal_initializer())
    b = tf.compat.v1.get_variable("bias", shape=[2], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

input_tensor = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
output_tensor = simple_model(input_tensor)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    result = sess.run(output_tensor, feed_dict={input_tensor: [[1, 2], [3, 4]]})
    print(result)
```

*TensorFlow 2.x Code:*

```python
import tensorflow as tf
class SimpleModel(tf.keras.Model):
    def __init__(self):
      super(SimpleModel, self).__init__()
      self.w = tf.Variable(tf.random.normal(shape=[2, 2]), name='weight')
      self.b = tf.Variable(tf.zeros(shape=[2]), name='bias')

    def call(self, x):
       return tf.matmul(x, self.w) + self.b

model = SimpleModel()
input_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
output_tensor = model(input_tensor)
print(output_tensor.numpy())
```

*Commentary:* In the TF1 code, we used `tf.get_variable` to create the weights and biases and had to initialize them separately with `global_variables_initializer`. In the TF2 version, the `SimpleModel` is defined as a class inheriting from `tf.keras.Model`. The weights and biases are initialized as `tf.Variable` within the constructor. The forward computation is encapsulated in the `call` method.  Note that the model is directly callable with the data input. The automatic variable management and initialization within the `tf.keras.Model` system substantially simplifies model definition.

**Example 3: Migrating a TF1 Training Loop**

*TensorFlow 1.x Code:*

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y_input = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
w = tf.compat.v1.get_variable("weight", shape=[2, 1], initializer=tf.random_normal_initializer())
b = tf.compat.v1.get_variable("bias", shape=[1], initializer=tf.zeros_initializer())
y_pred = tf.matmul(x_input, w) + b
loss = tf.reduce_mean(tf.square(y_pred - y_input))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss)

x_train = [[1, 2], [3, 4], [5, 6]]
y_train = [[3], [7], [11]]

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(100):
        _, loss_value = sess.run([optimizer, loss], feed_dict={x_input: x_train, y_input: y_train})
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss_value)

```

*TensorFlow 2.x Code:*

```python
import tensorflow as tf
class LinearModel(tf.keras.Model):
  def __init__(self):
    super(LinearModel, self).__init__()
    self.w = tf.Variable(tf.random.normal(shape=[2, 1]))
    self.b = tf.Variable(tf.zeros(shape=[1]))

  def call(self, x):
    return tf.matmul(x, self.w) + self.b

model = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
x_train = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
y_train = tf.constant([[3], [7], [11]], dtype=tf.float32)

def loss_fn(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_pred - y_true))

for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss = loss_fn(y_train, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 10 == 0:
      print("Epoch:", epoch, "Loss:", loss.numpy())

```

*Commentary:* The TF1 code defines placeholders and executes the gradient computation and updates within a `tf.Session`. The TF2 version utilizes a `LinearModel` class derived from `tf.keras.Model`. Training happens within a loop where the forward pass, loss calculation, gradient computation, and updates are executed step-by-step, with explicit tracking of the trainable parameters.  `tf.GradientTape` is used to track operations during the forward pass for automatic gradient calculations, and the optimizer's `apply_gradients` performs the parameter update. This illustrates the transition to explicit, eager training in TF2.

For further in-depth understanding, several resources are beneficial:

*   **TensorFlow Official Documentation:** The official TensorFlow documentation is the most detailed and regularly updated source. It provides examples for specific operations, APIs, and usage patterns.
*   **TensorFlow Tutorials:** The official tutorials offer guided, practical examples for various tasks from basic models to advanced techniques, often covering both TF1 and TF2 methodologies.
*   **Online Courses:** Platforms like Coursera, edX, and Udacity frequently offer comprehensive machine learning courses that emphasize the latest practices in TensorFlow 2.

A careful and methodical approach, focusing on refactoring key components like session handling, placeholder usage, and variable management, combined with the provided examples and resource, should enable a successful migration from TensorFlow 1.x to 2.x, laying a stable foundation for future development.
