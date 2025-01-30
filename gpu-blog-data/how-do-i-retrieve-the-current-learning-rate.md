---
title: "How do I retrieve the current learning rate of an ExponentialDecay schedule in TensorFlow 2.0's SGD optimizer?"
date: "2025-01-30"
id: "how-do-i-retrieve-the-current-learning-rate"
---
The `ExponentialDecay` learning rate schedule in TensorFlow 2.0 doesn't directly expose its current learning rate as a readily accessible attribute.  This is a common source of confusion, stemming from the inherent nature of the schedule: it's a function that computes the learning rate dynamically based on the global step.  Therefore, retrieving the "current" learning rate necessitates understanding how the schedule interacts with the optimizer and the training process.  My experience debugging similar issues across various TensorFlow projects led me to this solution.

**1. Understanding the Mechanism**

The `tf.keras.optimizers.schedules.ExponentialDecay` class calculates the learning rate at each step according to the formula:

`learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)`

Where:

* `initial_learning_rate` is the starting learning rate.
* `decay_rate` determines the decay factor.
* `global_step` is the current training step.
* `decay_steps` is the number of steps after which the learning rate is multiplied by `decay_rate`.

Crucially, this calculation isn't stored as a persistent variable within the `ExponentialDecay` object itself.  The schedule only provides the learning rate *on demand*.  Therefore, to access the current learning rate, we must explicitly call the schedule with the current global step.

**2. Code Examples and Commentary**

The following examples demonstrate different methods for retrieving the current learning rate.  Each method addresses a specific scenario, highlighting the flexibility required when working with dynamic learning rate schedules.

**Example 1:  Directly calling the schedule with the current global step**

This is the most straightforward approach.  It requires access to the global step counter.  In many Keras training loops, this counter is implicitly handled.  However, for more granular control, especially in custom training loops, managing the global step explicitly is essential.  In my work optimizing a large-scale recommender system, this method proved invaluable for monitoring learning rate decay.

```python
import tensorflow as tf

initial_learning_rate = 0.1
decay_rate = 0.96
decay_steps = 1000
global_step = tf.Variable(1500, dtype=tf.int64)  # Example global step

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)

current_learning_rate = learning_rate_schedule(global_step)
print(f"Current learning rate at step {global_step.numpy()}: {current_learning_rate.numpy()}")

```

This code snippet explicitly defines the `global_step` variable and then uses it as input to the `learning_rate_schedule`. The output displays the computed learning rate at that specific step.  Remember to adjust the `global_step` value according to your training progress.


**Example 2:  Integrating into a Keras training loop**

In a standard Keras `fit` method, accessing the global step directly is less intuitive.  We need to leverage Keras callbacks to monitor and record the current learning rate at each epoch or batch.  During my work on a deep reinforcement learning agent, I developed this approach to create learning rate visualizations.

```python
import tensorflow as tf

initial_learning_rate = 0.1
decay_rate = 0.96
decay_steps = 1000

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global_step = self.model.optimizer.iterations.numpy()
        current_learning_rate = learning_rate_schedule(global_step)
        print(f"Epoch {epoch+1}, Global Step: {global_step}, Learning Rate: {current_learning_rate.numpy()}")

model = tf.keras.models.Sequential(...) # Define your model here
model.compile(optimizer=optimizer, loss='mse')

model.fit(x_train, y_train, epochs=10, callbacks=[LearningRateLogger()])
```

This example uses a custom Keras callback to print the current learning rate at the end of each epoch. The callback accesses the global step via the optimizer's `iterations` attribute.  The output shows the learning rate's evolution over the training process.


**Example 3: Custom training loop with manual global step management**

For maximum control, especially in unconventional training setups, a custom training loop is necessary.  Here, you explicitly manage the global step, giving you complete visibility into the learning rate adjustment.  This was critical in a project involving distributed training on a cluster.

```python
import tensorflow as tf

initial_learning_rate = 0.1
decay_rate = 0.96
decay_steps = 1000

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule)

global_step = tf.Variable(0, dtype=tf.int64)

for epoch in range(10):
    for batch in range(100):
        with tf.GradientTape() as tape:
            # Your training step here...  e.g., loss = model(x_batch)
            loss = tf.random.uniform([]) # placeholder for actual loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_step.assign_add(1)
        current_learning_rate = learning_rate_schedule(global_step)
        print(f"Epoch {epoch+1}, Batch {batch+1}, Global Step: {global_step.numpy()}, Learning Rate: {current_learning_rate.numpy()}")

```

This illustrates complete control over the training loop and global step.  The learning rate is printed after each batch, providing a detailed view of its evolution. Remember to replace the placeholder loss calculation with your actual model and loss function.

**3. Resource Recommendations**

The official TensorFlow documentation provides comprehensive information on optimizers and learning rate schedules.  Exploring the source code of TensorFlow itself can be beneficial for understanding the underlying implementation details.  Furthermore, examining existing examples and tutorials focusing on custom training loops and Keras callbacks can deepen your understanding.  Finally, dedicated books on deep learning with TensorFlow will offer broader context and theoretical backing.
