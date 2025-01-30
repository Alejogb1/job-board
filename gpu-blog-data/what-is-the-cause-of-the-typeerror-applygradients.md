---
title: "What is the cause of the 'TypeError: apply_gradients() got an unexpected keyword argument 'global_step' '?"
date: "2025-01-30"
id: "what-is-the-cause-of-the-typeerror-applygradients"
---
The `TypeError: apply_gradients() got an unexpected keyword argument 'global_step'` arises primarily from an incompatibility between TensorFlow versions when using gradient application within custom training loops. Specifically, it signals that the `apply_gradients()` method of an optimizer instance, in its current form within your environment, does not expect or process a `global_step` argument, despite the user providing one. This usually indicates usage of an older API where explicit step tracking was not integrated into the optimizer's gradient application function. I encountered this directly during a transition of our model training pipeline, migrating from TF 1.x to TF 2.x, which involved custom training iterations.

The root of the problem lies in how TensorFlow's API evolved regarding the management of the global training step. In TensorFlow 1.x, optimizers often required manual updates to the global step using a separate operation. This global step variable served as a counter, tracking the progress of the training process, and was also essential for features like learning rate decay, which might depend on the current training iteration. However, TensorFlow 2.x streamlined this process, incorporating the step count update within the optimizer's `apply_gradients()` method itself, typically within the `minimize()` function. Therefore, directly passing a `global_step` argument is no longer needed and indeed, will cause the error we're discussing because the method signature in TF2 does not include it.

The confusion can sometimes occur because legacy code from TensorFlow 1.x often includes a variable tracking global step, and there might be calls where this variable is passed to `apply_gradients`. When transitioning to TF 2.x, such calls will produce the error, as the function is no longer structured to accept such an argument.

Let me provide some illustrative examples based on my experience debugging this particular issue, clarifying the difference between the older and the newer approaches.

**Code Example 1: Incorrect Usage (TF 2.x with legacy approach)**

```python
import tensorflow as tf

# Assume model and loss are defined elsewhere
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

global_step = tf.Variable(0, trainable=False) # Defining a global step variable like in TF1.x

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_fn(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=global_step) # Incorrect in TF 2.x
  return loss

# Example usage (simplified)
images = tf.random.normal((32, 10))
labels = tf.random.normal((32, 10))
loss_val = train_step(images, labels)

print(f"Loss value: {loss_val}")
```

This code, while structurally similar to older TensorFlow paradigms, produces the `TypeError`. I deliberately set the `global_step` argument in `apply_gradients()` to highlight the problem. The optimizer in TensorFlow 2.x does not expect this parameter; thus, the error occurs. In real scenarios, you might encounter this if the legacy TensorFlow 1.x code was not thoroughly updated to comply with TensorFlow 2.x standards.

**Code Example 2: Correct Usage (TF 2.x, manual training loop)**

```python
import tensorflow as tf

# Assume model and loss are defined elsewhere
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# No explicit global_step variable needed when using the apply_gradients() with TF 2.x.
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_fn(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Corrected call for TF 2.x
  return loss

# Example usage (simplified)
images = tf.random.normal((32, 10))
labels = tf.random.normal((32, 10))
loss_val = train_step(images, labels)
print(f"Loss value: {loss_val}")
```

In contrast to the previous example, this code omits the `global_step` argument from the `apply_gradients()` call. This is the correct usage for TensorFlow 2.x. The optimizer now manages the step count internally when calling `apply_gradients()`. This means that within a training loop you do not have to pass a global_step variable when applying the gradient. If you need to track the global step outside the optimizer scope, it can be retrieved from the optimizers property as `optimizer.iterations` which is updated with every call to `apply_gradients()`.

**Code Example 3: Using the `minimize()` Function**

```python
import tensorflow as tf

# Assume model and loss are defined elsewhere
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)

    optimizer.minimize(loss, model.trainable_variables, tape=tape) # Using minimize, implicit step management
    return loss

# Example usage (simplified)
images = tf.random.normal((32, 10))
labels = tf.random.normal((32, 10))
loss_val = train_step(images, labels)
print(f"Loss value: {loss_val}")
```

This last example demonstrates the simplest way to manage gradients within TensorFlow 2.x: using `optimizer.minimize()`. Here, the `minimize()` function internally handles the gradient computation, the gradient application, and the global step increment in one operation. Using `minimize` avoids both passing the `global_step` variable manually and calling `apply_gradients`. The `tape` object is passed for automatic differentiation as it is in examples 1 and 2. In many cases, this approach will provide easier implementation of a training process compared to manual gradient application.

To resolve the issue, the primary step is to remove any occurrences of `global_step` parameter from calls to `apply_gradients()`. If step tracking is essential, you should use the `.iterations` property of the optimizer as a counter. If you are using custom callbacks, for example, and need the optimizer step within your callback, this property will return an integer representing the amount of times that `apply_gradients()` has been called. The alternative, as shown above, is to utilize `optimizer.minimize()`. When doing so, the `apply_gradients` method is called internally by the `minimize` method and the updates to the global step are implicitly managed. This is the recommended method for training deep learning models using TensorFlow 2.x.

Regarding resources, the official TensorFlow documentation is the primary source of truth and is regularly updated. Specifically, reviewing the sections on custom training loops and optimizers will prove helpful. Additionally, studying the examples included within the source code repositories for popular TensorFlow tutorials and examples can provide practical demonstrations of best practices for managing gradients and the training step count. Consulting books and articles dedicated to advanced topics in TensorFlow and deep learning are also highly beneficial. These sources will provide more in-depth discussion of underlying principles and best practices for developing robust model training pipelines. Finally, scrutinize the release notes for TensorFlow, specifically the changes made during the transition from TF 1.x to TF 2.x, as this would highlight major API adjustments that may cause issues when migrating between different versions.
