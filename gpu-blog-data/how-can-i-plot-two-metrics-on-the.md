---
title: "How can I plot two metrics on the same TensorBoard graph in TensorFlow 2?"
date: "2025-01-30"
id: "how-can-i-plot-two-metrics-on-the"
---
The core challenge in visualizing multiple metrics concurrently in TensorBoard stems from how TensorFlow’s summary writing API handles different data types and how it organizes information within the log directories. Understanding this fundamental architecture is key to effectively plotting metrics together.

My experience, spanning several machine learning projects focused on complex model training, repeatedly highlighted the need for concise, comparative visualizations of diverse performance measures. I frequently encountered situations where simply observing training loss was insufficient to debug issues or evaluate model efficacy; a simultaneous view of accuracy and other relevant metrics was paramount. TensorFlow 2, in particular, requires explicit handling to ensure that distinct scalar summaries are logged and displayed on a unified graph. This is not an automatic behavior and necessitates a deliberate approach within your training loop.

The primary mechanism for writing metrics to TensorBoard lies in the `tf.summary` API, specifically functions like `tf.summary.scalar()`. Each call to this function, with a unique name, registers a distinct scalar value under the specified tag and step. The key to plotting multiple metrics lies in consistently using the same step within each summary call, and employing distinct names for the scalar metrics. If summaries are written with different steps, TensorBoard interprets them as disparate datasets, creating separate visualizations. For instance, if 'loss' is logged at step ‘1’ and 'accuracy' is logged at step ‘2’, they will not appear on the same plot.

To achieve a combined graph, I ensure that within my training loop, all metrics that I want to visualize together are written within the scope of a single training step iteration. The process typically involves calculating each metric, using `tf.summary.scalar()` to log each to the TensorBoard writer with consistent step tracking, and flushing the summary writer at appropriate intervals to update visualizations.

Let’s illustrate this with code. Below, I'll outline a basic example that includes the necessary structure:

```python
import tensorflow as tf

# Setup for demo
log_dir = 'logs/fit'
summary_writer = tf.summary.create_file_writer(log_dir)

# Simplified training loop with synthetic data
num_epochs = 10
num_batches = 100
batch_size = 32

@tf.function
def train_step(batch_data, batch_labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = tf.random.normal(shape=(batch_size, 1)) # Placeholder
        loss = tf.reduce_mean(tf.square(predictions - batch_labels))
    gradients = tape.gradient(loss, [tf.Variable(tf.random.normal(shape=(1,1)))]) # Placeholder
    optimizer.apply_gradients(zip(gradients,[tf.Variable(tf.random.normal(shape=(1,1)))])) # Placeholder

    accuracy = tf.reduce_mean(tf.cast(tf.abs(predictions - batch_labels) < 0.5, tf.float32)) # Placeholder metric

    return loss, accuracy


# Training process
optimizer = tf.keras.optimizers.Adam()
global_step = tf.Variable(0, dtype=tf.int64)
for epoch in range(num_epochs):
    for batch in range(num_batches):
        #Synthetic training data
        batch_data = tf.random.normal(shape=(batch_size, 10))
        batch_labels = tf.random.normal(shape=(batch_size, 1))
        loss, accuracy = train_step(batch_data, batch_labels, optimizer)

        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=global_step)
            tf.summary.scalar('accuracy', accuracy, step=global_step)
        global_step.assign_add(1)
    print(f'Epoch: {epoch+1}, Loss: {loss}, Accuracy: {accuracy}')
```

In the above example, the key element is the use of the `global_step` variable that is incremented after each training step. Both the ‘loss’ and ‘accuracy’ are written to the summary writer using the same `global_step` value, thus they will be aligned on the TensorBoard plot. I use `summary_writer.as_default()` context manager to make sure summaries go to our defined location. This ensures both metrics are associated with a singular 'x-axis' in tensorboard, allowing for direct observation of trends and relationships.

Building on this, we can introduce a validation phase and track the same metrics:

```python
import tensorflow as tf

# Setup
log_dir = 'logs/fit_val'
summary_writer = tf.summary.create_file_writer(log_dir)

# Simplified validation calculation (same data setup)
@tf.function
def validation_step(batch_data, batch_labels):
    predictions = tf.random.normal(shape=(batch_size, 1)) # Placeholder
    loss = tf.reduce_mean(tf.square(predictions - batch_labels))
    accuracy = tf.reduce_mean(tf.cast(tf.abs(predictions - batch_labels) < 0.5, tf.float32))
    return loss, accuracy


# Combined training and validation loop
num_epochs = 10
num_batches = 100
batch_size = 32
optimizer = tf.keras.optimizers.Adam()
global_step = tf.Variable(0, dtype=tf.int64)

for epoch in range(num_epochs):
    for batch in range(num_batches):
        #Synthetic training data
        batch_data = tf.random.normal(shape=(batch_size, 10))
        batch_labels = tf.random.normal(shape=(batch_size, 1))
        # Training step
        with tf.GradientTape() as tape:
            predictions = tf.random.normal(shape=(batch_size, 1)) # Placeholder
            loss = tf.reduce_mean(tf.square(predictions - batch_labels))
        gradients = tape.gradient(loss, [tf.Variable(tf.random.normal(shape=(1,1)))]) # Placeholder
        optimizer.apply_gradients(zip(gradients,[tf.Variable(tf.random.normal(shape=(1,1)))])) # Placeholder
        accuracy = tf.reduce_mean(tf.cast(tf.abs(predictions - batch_labels) < 0.5, tf.float32))

        with summary_writer.as_default():
            tf.summary.scalar('training_loss', loss, step=global_step)
            tf.summary.scalar('training_accuracy', accuracy, step=global_step)

        # Validation step
        if batch % 10 == 0: # Simplified validation frequency
             batch_data_val = tf.random.normal(shape=(batch_size, 10)) # Synthetic validation data
             batch_labels_val = tf.random.normal(shape=(batch_size, 1))
             val_loss, val_accuracy = validation_step(batch_data_val, batch_labels_val)

             with summary_writer.as_default():
                 tf.summary.scalar('validation_loss', val_loss, step=global_step)
                 tf.summary.scalar('validation_accuracy', val_accuracy, step=global_step)
        global_step.assign_add(1)

    print(f'Epoch: {epoch+1}, Training Loss: {loss}, Validation Loss: {val_loss}')
```
In this iteration, I've introduced separate summary logging for both training and validation metrics.  I've also maintained `global_step` which ensures all four metrics (training/validation loss and accuracy) are synchronized along the x-axis of the TensorBoard graph.  The validation summaries are recorded only every 10th training step to reduce visual clutter, while ensuring enough data is available to visualize the trends.

Finally, consider the scenario where we want to track a more complex metric, such as F1-Score. This can be implemented, though require a bit more work, since `tf.summary.scalar()` only takes a scalar and not a tensor:

```python
import tensorflow as tf
# Setup
log_dir = 'logs/fit_f1'
summary_writer = tf.summary.create_file_writer(log_dir)

#Simplified f1 calculation using placeholders, not optimized for real-world use
@tf.function
def calculate_f1(true_labels, predictions):
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(true_labels == 1, predictions >= 0.5),tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(predictions >= 0.5, tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(true_labels == 1, tf.float32))

    precision = true_positives / (predicted_positives + 1e-7) # Add small value to prevent 0 division
    recall = true_positives / (actual_positives + 1e-7)

    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1


# Training process with f1 logging
num_epochs = 10
num_batches = 100
batch_size = 32
optimizer = tf.keras.optimizers.Adam()
global_step = tf.Variable(0, dtype=tf.int64)

for epoch in range(num_epochs):
    for batch in range(num_batches):
         #Synthetic training data
        batch_data = tf.random.normal(shape=(batch_size, 10))
        batch_labels = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=2, dtype=tf.int32) #Simulate class labels
        batch_labels = tf.cast(batch_labels, dtype=tf.float32) #Convert to float for use in loss

        # Training step
        with tf.GradientTape() as tape:
            predictions = tf.random.normal(shape=(batch_size, 1)) # Placeholder
            loss = tf.reduce_mean(tf.square(predictions - batch_labels))
        gradients = tape.gradient(loss, [tf.Variable(tf.random.normal(shape=(1,1)))]) # Placeholder
        optimizer.apply_gradients(zip(gradients,[tf.Variable(tf.random.normal(shape=(1,1)))])) # Placeholder

        accuracy = tf.reduce_mean(tf.cast(tf.abs(predictions - batch_labels) < 0.5, tf.float32))

        f1_score = calculate_f1(batch_labels, predictions)

        with summary_writer.as_default():
            tf.summary.scalar('training_loss', loss, step=global_step)
            tf.summary.scalar('training_accuracy', accuracy, step=global_step)
            tf.summary.scalar('f1_score', f1_score, step=global_step)

        global_step.assign_add(1)
    print(f'Epoch: {epoch+1}, Loss: {loss}, F1: {f1_score}')
```
In this final example, a function `calculate_f1` encapsulates the logic for F1-Score calculation. It is called within the training loop, and the resultant scalar is logged, again using `global_step` to ensure alignment with the other logged metrics on the TensorBoard graph.  This approach ensures that even complex, derived metrics can be incorporated into your comparative visualizations.

For further exploration of best practices related to TensorFlow summary logging, the official TensorFlow documentation contains detailed guidance on the `tf.summary` API.  Additionally, I recommend investigating examples within the TensorFlow models repository to study how large projects structure and use TensorBoard in complex training scenarios. Finally, books dedicated to practical machine learning with TensorFlow can often offer additional nuanced understanding of how visualization can be integrated into the full development process.
