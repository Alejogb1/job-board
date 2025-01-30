---
title: "How can I log scalar metrics to TensorBoard from a Keras subclassed model's training loop?"
date: "2025-01-30"
id: "how-can-i-log-scalar-metrics-to-tensorboard"
---
TensorBoard, a visualization toolkit within TensorFlow, directly integrates with Keras training processes through callbacks, making scalar metric logging from custom training loops a nuanced, yet achievable task. The primary challenge lies in ensuring proper data transmission during the model’s manual training regime when not using `model.fit()`. The `tf.summary.scalar` function, used in conjunction with a `tf.summary.FileWriter` instantiated for the training directory, provides the mechanism for data recording. I've personally encountered this when implementing a variational autoencoder with a custom loss function and required detailed per-epoch reconstruction error tracking, as well as observing the KL divergence component during each training step.

The core principle involves creating a `tf.summary.FileWriter` associated with a specific log directory. This writer acts as a sink for summary data, specifically scalar values in this case. Within your custom training loop, after calculating the desired scalar metric, such as the loss value, you employ `tf.summary.scalar` to generate a summary. Importantly, you must provide the writer with this summary using `writer.flush()` at the end of each epoch to make the data visible in TensorBoard. The critical aspect is to maintain this writer’s instance across training epochs so it can aggregate summaries progressively. The `tf.summary.scalar` function requires a tag, which acts as a unique identifier for the metric within TensorBoard, and the actual scalar value, which represents the calculated measurement during the current training step or epoch.

To illustrate this, consider a scenario where you're tracking training and validation loss for a basic regression model within a subclassed model:

**Example 1: Basic Loss Logging**

```python
import tensorflow as tf
import numpy as np

class RegressionModel(tf.keras.Model):
  def __init__(self, units=10):
    super(RegressionModel, self).__init__()
    self.dense = tf.keras.layers.Dense(units)
    self.out = tf.keras.layers.Dense(1)

  def call(self, x):
    x = self.dense(x)
    return self.out(x)


# Simulate data
num_samples = 1000
input_dim = 5
X_train = np.random.rand(num_samples, input_dim).astype(np.float32)
y_train = np.random.rand(num_samples, 1).astype(np.float32)
X_val = np.random.rand(200, input_dim).astype(np.float32)
y_val = np.random.rand(200, 1).astype(np.float32)

# Hyperparameters
epochs = 10
learning_rate = 0.01
log_dir = "./logs"

# Create model and optimizer
model = RegressionModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()


# Create FileWriter for writing log
train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")
val_summary_writer = tf.summary.create_file_writer(log_dir + "/val")


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def val_step(x,y):
    y_pred = model(x)
    loss = loss_fn(y, y_pred)
    return loss

for epoch in range(epochs):
    total_train_loss = 0
    for step in range(X_train.shape[0]//32): # Simplified training loop

       batch_x = X_train[step * 32 : (step+1) * 32]
       batch_y = y_train[step * 32 : (step+1) * 32]
       loss = train_step(batch_x, batch_y)
       total_train_loss += loss


    avg_train_loss = total_train_loss / (X_train.shape[0]//32)

    total_val_loss = 0
    for step in range(X_val.shape[0]//32):

        batch_x = X_val[step * 32 : (step+1) * 32]
        batch_y = y_val[step * 32 : (step+1) * 32]
        loss = val_step(batch_x, batch_y)
        total_val_loss += loss
    avg_val_loss = total_val_loss / (X_val.shape[0]//32)

    with train_summary_writer.as_default():
       tf.summary.scalar("loss", avg_train_loss, step = epoch)
    with val_summary_writer.as_default():
        tf.summary.scalar("loss", avg_val_loss, step = epoch)

    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
```
In this example, distinct file writers, `train_summary_writer` and `val_summary_writer`, are created for train and validation logs. Inside the training loop, following the calculation of the average loss per epoch,  `tf.summary.scalar` records this loss with the tag “loss” and the epoch number as the step, ensuring incremental reporting to the associated TensorBoard log file. Using `with writer.as_default()` is key for the subsequent call to `tf.summary.scalar` to function as it specifies the target FileWriter.

Now consider tracking additional metrics, not just the loss. For example, imagine we want to track the mean absolute error in addition to the loss.  We’d simply add additional calls to the `tf.summary.scalar` function as illustrated below:

**Example 2: Multiple Scalar Metrics**

```python
import tensorflow as tf
import numpy as np


class RegressionModel(tf.keras.Model):
  def __init__(self, units=10):
    super(RegressionModel, self).__init__()
    self.dense = tf.keras.layers.Dense(units)
    self.out = tf.keras.layers.Dense(1)

  def call(self, x):
    x = self.dense(x)
    return self.out(x)


# Simulate data
num_samples = 1000
input_dim = 5
X_train = np.random.rand(num_samples, input_dim).astype(np.float32)
y_train = np.random.rand(num_samples, 1).astype(np.float32)
X_val = np.random.rand(200, input_dim).astype(np.float32)
y_val = np.random.rand(200, 1).astype(np.float32)

# Hyperparameters
epochs = 10
learning_rate = 0.01
log_dir = "./logs"

# Create model and optimizer
model = RegressionModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()
mae_fn = tf.keras.metrics.MeanAbsoluteError()


# Create FileWriter for writing log
train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")
val_summary_writer = tf.summary.create_file_writer(log_dir + "/val")


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    mae_fn.update_state(y,y_pred)
    return loss

@tf.function
def val_step(x,y):
    y_pred = model(x)
    loss = loss_fn(y, y_pred)
    mae_fn.update_state(y, y_pred)
    return loss

for epoch in range(epochs):
    total_train_loss = 0
    mae_fn.reset_state() # reset metrics for each epoch
    for step in range(X_train.shape[0]//32):
       batch_x = X_train[step * 32 : (step+1) * 32]
       batch_y = y_train[step * 32 : (step+1) * 32]
       loss = train_step(batch_x, batch_y)
       total_train_loss += loss


    avg_train_loss = total_train_loss / (X_train.shape[0]//32)
    train_mae = mae_fn.result()

    total_val_loss = 0
    mae_fn.reset_state()
    for step in range(X_val.shape[0]//32):
        batch_x = X_val[step * 32 : (step+1) * 32]
        batch_y = y_val[step * 32 : (step+1) * 32]
        loss = val_step(batch_x, batch_y)
        total_val_loss += loss

    avg_val_loss = total_val_loss / (X_val.shape[0]//32)
    val_mae = mae_fn.result()

    with train_summary_writer.as_default():
        tf.summary.scalar("loss", avg_train_loss, step = epoch)
        tf.summary.scalar("mae", train_mae, step= epoch)
    with val_summary_writer.as_default():
        tf.summary.scalar("loss", avg_val_loss, step = epoch)
        tf.summary.scalar("mae", val_mae, step = epoch)


    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train MAE: {train_mae:.4f} Val Loss: {avg_val_loss:.4f}, Val MAE {val_mae:.4f}")
```

Here, both train and validation mean absolute errors are calculated and logged alongside the loss values, each identified by distinct tags (“loss” and “mae”).  The `tf.keras.metrics.MeanAbsoluteError()` metric is employed to provide that metric and the metric is reset per epoch with `reset_state()`. This example shows how you can simply extend logging to track multiple metrics.

The key part is ensuring you call the `tf.summary.scalar` method for each metric you need, all within the context of a `FileWriter` object.

Finally, consider a more complex scenario involving a metric calculated inside a gradient context during training.  In this case, we can calculate the variance of the gradients per layer. We might be interested in whether this value diminishes over training.

**Example 3:  Metric Calculated Inside Gradient Context**

```python
import tensorflow as tf
import numpy as np

class RegressionModel(tf.keras.Model):
  def __init__(self, units=10):
    super(RegressionModel, self).__init__()
    self.dense = tf.keras.layers.Dense(units)
    self.out = tf.keras.layers.Dense(1)

  def call(self, x):
    x = self.dense(x)
    return self.out(x)


# Simulate data
num_samples = 1000
input_dim = 5
X_train = np.random.rand(num_samples, input_dim).astype(np.float32)
y_train = np.random.rand(num_samples, 1).astype(np.float32)
X_val = np.random.rand(200, input_dim).astype(np.float32)
y_val = np.random.rand(200, 1).astype(np.float32)

# Hyperparameters
epochs = 10
learning_rate = 0.01
log_dir = "./logs"

# Create model and optimizer
model = RegressionModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()


# Create FileWriter for writing log
train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    grad_variances = []
    for grad in gradients:
        if grad is not None:
          grad_variances.append(tf.math.reduce_variance(grad))

    return loss, grad_variances

for epoch in range(epochs):
    total_train_loss = 0
    grad_variances_list = [] # track variance

    for step in range(X_train.shape[0]//32):
        batch_x = X_train[step * 32 : (step+1) * 32]
        batch_y = y_train[step * 32 : (step+1) * 32]
        loss, grad_variances = train_step(batch_x, batch_y)
        total_train_loss += loss
        grad_variances_list.append(grad_variances)



    avg_train_loss = total_train_loss / (X_train.shape[0]//32)
    avg_grad_variances = tf.reduce_mean(tf.stack(grad_variances_list), axis=0) # average per layer across batches
    with train_summary_writer.as_default():
      tf.summary.scalar("loss", avg_train_loss, step = epoch)
      for i, var in enumerate(avg_grad_variances):
        tf.summary.scalar(f"grad_var_layer_{i}", var, step=epoch) # Add variance of gradients per layer
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Grad Variances: {avg_grad_variances}")
```
This example demonstrates a scenario where a derived metric, the gradient variance per layer, is computed *during* the gradient computation stage. The values are collected over batches and then averaged across each layer before being passed to the writer via `tf.summary.scalar`, illustrating how one can track arbitrary metrics from a custom model's training loop in TensorBoard.

For further learning, I recommend exploring the official TensorFlow documentation related to the `tf.summary` module, specifically the `tf.summary.scalar`, `tf.summary.FileWriter`, and `tf.summary.create_file_writer` methods. Additionally, reviewing the Keras documentation concerning custom training loops can solidify understanding of how these integrate. Practical examples are available on the official TensorFlow GitHub repository in the form of notebooks and tutorials which demonstrate usage in broader contexts such as variational autoencoders or GANs. I also find it helpful to consult the TensorFlow website for updates on the API if using the latest versions.
