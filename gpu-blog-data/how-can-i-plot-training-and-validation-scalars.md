---
title: "How can I plot training and validation scalars together in Keras TensorBoard?"
date: "2025-01-30"
id: "how-can-i-plot-training-and-validation-scalars"
---
The primary challenge in visualizing Keras training and validation metrics within TensorBoard lies in correctly structuring your model’s training loop and logging relevant scalar values, ensuring that both sets are distinctly tagged and can be easily displayed. It's a common issue, one I've encountered many times while optimizing deep learning models for image segmentation and NLP tasks. The key is understanding that TensorBoard leverages distinct log directories for training and validation data. We must ensure we're writing to the appropriate log directory in each phase of the model's cycle.

The standard Keras `fit` method, when provided with validation data, automatically handles logging to both training and validation directories. However, for scenarios with custom training loops or specific logging requirements beyond the default, we must implement these log writes manually. I've often seen this become necessary when implementing techniques like adversarial training or using multiple optimizers within a single training loop, scenarios where the standard `fit` method becomes restrictive.

Here's how we can achieve clear separation and parallel plotting using Keras and the `tf.summary` API within a custom training loop:

1.  **Setting up Summary Writers**: We need two separate `tf.summary.FileWriter` instances, one for training logs and the other for validation logs. These writers will handle writing the scalar values to the correct TensorBoard log directories. I typically instantiate these writers at the beginning of my training script, often encapsulating this logic in a dedicated logging function or class. This setup ensures the training and validation data are kept distinct in the TensorBoard visualizations.

2.  **Writing Scalars During Training and Validation**: During the training loop, at the end of each epoch (or at regular intervals), I calculate training metrics (loss, accuracy, etc.) and use `tf.summary.scalar` to write these to the training summary writer. Crucially, the 'name' parameter provided to this method must be unique within the training context; this becomes important in TensorBoard to identify and group data. Then, similarly, during the validation phase (after a training epoch), I compute validation metrics using the validation dataset and write to the validation summary writer with appropriate, distinct names. This step ensures that the training and validation metrics can be plotted side-by-side in TensorBoard. I generally prefer calculating metrics after all batches are completed for each epoch, which I've found provides a more stable evaluation.

3.  **Flushing Writers**: It’s important to `flush` both the training and validation writers after each epoch's logging, ensuring the data is written to disk. This is a crucial step that I've often seen people overlook, leading to missing information in TensorBoard or only seeing the latest metric. Failure to flush the writer will keep the log data buffered in memory, preventing it from being accessible in TensorBoard.

Here's a first example demonstrating this principle with a simplified training loop, using a `tf.keras.Model` for classification:

```python
import tensorflow as tf
import numpy as np

# Sample data (replace with your actual dataset)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
X_val = np.random.rand(200, 10)
y_val = np.random.randint(0, 2, 200)

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()
train_metric = tf.keras.metrics.BinaryAccuracy()
val_metric = tf.keras.metrics.BinaryAccuracy()
# Setup Summary Writers
train_log_dir = 'logs/train'
val_log_dir = 'logs/validation'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)


def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(y, logits)
    return loss


def val_step(x,y):
  logits = model(x, training=False)
  val_metric.update_state(y, logits)


epochs = 10
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
      batch_loss = train_step(x_batch, y_batch)
      epoch_loss += batch_loss

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', epoch_loss/(step+1), step=epoch)
        tf.summary.scalar('accuracy', train_metric.result(), step=epoch)
    train_metric.reset_states()
    train_summary_writer.flush()

    for val_x_batch, val_y_batch in val_dataset:
       val_step(val_x_batch, val_y_batch)
    with val_summary_writer.as_default():
      tf.summary.scalar('accuracy', val_metric.result(), step=epoch)
    val_metric.reset_states()
    val_summary_writer.flush()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(step+1):.4f}, Train Acc: {train_metric.result():.4f}, Validation Acc: {val_metric.result():.4f}")
```

This code demonstrates the basic pattern: separate writers, logging within appropriate blocks, and using distinct names. When you run TensorBoard pointing to the parent logs directory ('logs' in this case), you will see separate 'loss' and 'accuracy' plots, with distinct lines for training and validation. I have found that keeping metrics clear with this structure makes it easier to diagnose training or overfitting issues.

For more detailed analysis, like when exploring different model layers or internal activation behavior, I've found it very useful to log custom scalar statistics that would be difficult to visualize any other way. For example, the mean and standard deviation of activations can reveal when a specific layer is becoming unstable during training.

Here's a code snippet illustrating how I would add such custom logging of a layer's activations:

```python
# ... (previous code, model defined)...

def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(y, logits)
    activation = model.layers[0](x, training=True)
    return loss, activation

for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
      batch_loss, activation = train_step(x_batch, y_batch)
      epoch_loss += batch_loss

      mean_activation = tf.reduce_mean(activation)
      std_activation = tf.math.reduce_std(activation)

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', epoch_loss/(step+1), step=epoch)
      tf.summary.scalar('accuracy', train_metric.result(), step=epoch)
      tf.summary.scalar('layer1_mean_activation', mean_activation, step=epoch)
      tf.summary.scalar('layer1_std_activation', std_activation, step=epoch)

    train_metric.reset_states()
    train_summary_writer.flush()

    # ... (validation step as in the previous example) ...
```

Here I've extracted the output of the first layer `model.layers[0]` and logged the mean and standard deviation of the output. In the TensorBoard visualization these extra metrics would show as separate plots. It is crucial that the names of the log scalars are unique in order to avoid overwriting plots and create a good level of granularity when debugging models. I find it a very useful practice to log the mean/std of gradients as well.

Lastly, you can also use `tf.summary.histogram` to visualize the distribution of gradients or weights across the layers, giving you further insight. When diagnosing issues with gradients exploding or vanishing this histogram functionality provides a crucial tool. This logging can be easily added to the custom training loop, after gradient computation, as shown in this example:

```python
# ... (previous code, model defined)...

def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(y, logits)
    return loss, gradients


for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
      batch_loss, grads = train_step(x_batch, y_batch)
      epoch_loss += batch_loss

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', epoch_loss/(step+1), step=epoch)
      tf.summary.scalar('accuracy', train_metric.result(), step=epoch)
      for i, grad in enumerate(grads):
          tf.summary.histogram(f'layer_{i}_gradients', grad, step=epoch)

    train_metric.reset_states()
    train_summary_writer.flush()
    # ... (validation step as before)...
```

Here, I've looped through the gradients and logged each with a unique name, which allows us to observe the distribution of the gradient magnitudes across the layers.

Regarding resources, the official TensorFlow documentation offers a comprehensive guide to TensorBoard and the `tf.summary` API. I have also found several deep learning books, particularly those that delve into custom training loops, to provide useful examples. Additionally, I have found that exploring open-source model training repositories on GitHub, where the log infrastructure is well implemented, helps understand real-world implementations of custom logging. Finally, the Keras documentation has many examples of `callback` implementations, which can be a more structured alternative to custom training loop logging, and it may be a starting point for complex applications. Through these resources, one can gain a deeper understanding of not just logging but the entire training process.
