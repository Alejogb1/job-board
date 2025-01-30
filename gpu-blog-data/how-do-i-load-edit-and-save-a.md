---
title: "How do I load, edit, and save a TensorFlow checkpoint?"
date: "2025-01-30"
id: "how-do-i-load-edit-and-save-a"
---
The key to effectively managing TensorFlow models across training sessions resides in checkpointing, a mechanism that serializes the model's variables to disk. This allows for the resumption of training, the reuse of pre-trained weights, and the assessment of a model's state at any point during its lifecycle. My experience developing several convolutional neural networks for image recognition has made a firm grasp of checkpointing essential, and I can outline the process in detail.

The primary components for this process are the `tf.train.Checkpoint` object, which manages the variables to be saved, and the `tf.train.CheckpointManager`, which orchestrates the saving and restoring of checkpoints. The checkpoint itself is not a single file, but a collection of files containing the model's graph structure and the numerical values of its variables. These files are stored in a specified directory, creating a record of the training state that can be loaded later.

The first critical step is defining what to save. This involves creating a `tf.train.Checkpoint` instance, providing the variables to manage. These variables typically include the model itself, optimizers, and any other relevant quantities such as learning rates or global step counters. The model, if inheriting from `tf.keras.Model`, will automatically have its trainable variables included. For non-keras models, the variables can be manually assigned using object properties or a dictionary. After establishing the `Checkpoint`, a `CheckpointManager` is initiated. This manager takes the checkpoint, the directory path, and optionally, parameters to control the number of retained checkpoints. It maintains a history of checkpoints and removes older ones, preventing excessive disk usage.

To save the model state, the `CheckpointManager`'s `save()` method is invoked. This method creates a new checkpoint directory, serializes the variables, and writes them to this directory. The checkpoint manager maintains a history so that the user can easily revert to previous states. Restoring the model's variables is achieved with `CheckpointManager`'s `restore()` method. This method will load the last saved checkpoint if no checkpoint is specified or a specific one if path provided. After this, the model state is identical to when the checkpoint was saved. Note that the restored values will only apply to the variables managed by the provided checkpoint instance. It's vital to instantiate the model with the same architecture before restoring the checkpoint. If that has not been done, restoration will have no impact on the current variables. 

Here are three detailed code examples illustrating these steps, along with commentary.

**Example 1: Basic Checkpointing with a Keras Model**

This example demonstrates how to checkpoint a `tf.keras.Model` along with its optimizer, a common practice during training.

```python
import tensorflow as tf
import os

# Define a simple Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Define an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define a checkpoint and a checkpoint manager
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_directory = './training_checkpoints_example_1'
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

# Save the initial checkpoint (before training). Useful for initializations.
checkpoint_manager.save()

# Mimic training: Update model variables
x = tf.random.normal(shape=(10,784))
y = tf.random.normal(shape=(10,10))
with tf.GradientTape() as tape:
    output = model(x)
    loss = tf.reduce_mean((output - y)**2)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
# Save a checkpoint after one gradient update (i.e. "after training")
checkpoint_manager.save()

# Simulate restoring. First load the latest checkpoint.
checkpoint_manager.restore_or_initialize()

# Optionally load an older specific checkpoint using its path
specific_checkpoint_path = checkpoint_manager.latest_checkpoint
checkpoint.restore(specific_checkpoint_path)

print('Checkpoint Saved and Restored. Note that the file structure will contain "ckpt-1" and "ckpt-2" or similar.')
print(f'Model weights after restore (first layer):\n{model.trainable_variables[0][0,:5]}')

#Clean the saved checkpoint folder
os.system(f'rm -rf {checkpoint_directory}')
```

This example first constructs a Keras sequential model and optimizer. A `tf.train.Checkpoint` includes both. The `CheckpointManager` stores checkpoints in the designated directory, keeping only the three latest. I included a manual gradient update to simulate training. `checkpoint_manager.save()` stores a checkpoint *before* and *after* the training simulation. Then `checkpoint_manager.restore_or_initialize()` is used to load the most recent checkpoint. Finally, I showed how a specific checkpoint can be restored using `checkpoint.restore()`. Note, this will restore to the initial state of when the checkpoint was saved, meaning after this operation, the model's state is at the stage when the checkpoint was saved. To illustrate, I print the model’s weights. The directory structure created during this process typically contains "ckpt-1", "ckpt-2" and their related files, not just the specified directory in the code. Finally, the folder is deleted.

**Example 2: Checkpointing Custom Variables Outside Keras**

This example shows how to include variables from a non-Keras context within the checkpointing process.

```python
import tensorflow as tf
import os

# Define a variable outside Keras
my_variable = tf.Variable(tf.random.normal(shape=(5,)), name="my_var")

# Define a simple custom non-keras model (for the sake of completeness, not necessary)
class CustomLinearModel:
    def __init__(self, input_shape):
        self.weights = tf.Variable(tf.random.normal(shape=(input_shape, 10)), name='model_weights')
        self.biases = tf.Variable(tf.zeros(shape=(10,)), name='model_biases')
        
    def __call__(self, x):
        return tf.matmul(x, self.weights) + self.biases

custom_model = CustomLinearModel(784)


# Define a checkpoint and a checkpoint manager, note that we need to provide the non-keras variables manually
checkpoint = tf.train.Checkpoint(my_variable=my_variable, model=custom_model)
checkpoint_directory = './training_checkpoints_example_2'
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

# Save the initial state
checkpoint_manager.save()

# Update variables
my_variable.assign(my_variable * 2)
x = tf.random.normal(shape=(10,784))
with tf.GradientTape() as tape:
    output = custom_model(x)
    loss = tf.reduce_mean((output - tf.zeros_like(output))**2)
grads = tape.gradient(loss, [custom_model.weights, custom_model.biases])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer.apply_gradients(zip(grads, [custom_model.weights, custom_model.biases]))

# Save the checkpoint after manual update
checkpoint_manager.save()

# Restore the most recent checkpoint
checkpoint_manager.restore_or_initialize()

print('Checkpoint Saved and Restored with custom variables')
print(f'custom_variable after restore: {my_variable}')

#Clean the saved checkpoint folder
os.system(f'rm -rf {checkpoint_directory}')
```

This example introduces a TensorFlow variable directly, without using Keras structures. A custom non-keras model structure is introduced. These variables are included in the `tf.train.Checkpoint` using keyword arguments. After saving and modifying these variables, the code restores them using the checkpoint manager, demonstrating the ability to manage arbitrary TensorFlow variables. I print the value of the variable after restoration. The file structure follows a similar pattern as the previous example.

**Example 3: Manual Step Management and Restoration**

This final example highlights manually saving the training step and then restoring it, which is important when resuming training from a specific point.

```python
import tensorflow as tf
import os

# Define a Keras Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
global_step = tf.Variable(0, dtype=tf.int64, name='global_step') # added a global step
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_step)
checkpoint_directory = './training_checkpoints_example_3'
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

# Save initial checkpoint
checkpoint_manager.save()

# Simulate training loop
for _ in range(5):
    global_step.assign_add(1)
    x = tf.random.normal(shape=(10,784))
    y = tf.random.normal(shape=(10,10))
    with tf.GradientTape() as tape:
        output = model(x)
        loss = tf.reduce_mean((output - y)**2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if global_step % 2 == 0:
        checkpoint_manager.save() # Save every 2nd training step

# restore the most recent step
checkpoint_manager.restore_or_initialize()
print('Checkpoint Saved and Restored, restoring at step:', int(global_step)) # The value should match the last saved value

#Clean the saved checkpoint folder
os.system(f'rm -rf {checkpoint_directory}')
```

This example introduces a global training step counter. The counter is included in the `tf.train.Checkpoint`. Then, a short training loop updates the counter. Checkpoints are saved at an interval of 2 updates. After training, `checkpoint_manager.restore_or_initialize()` loads the latest checkpoint, including the value of the step counter. The step number is printed after restoration. The file structure generated follows the same convention as the other examples, and it is then deleted.

For further understanding and more detailed information on TensorFlow checkpointing, consult the official TensorFlow documentation, specifically the sections on `tf.train.Checkpoint`, `tf.train.CheckpointManager`, and related model saving APIs. The TensorFlow tutorials and guides on the official website also provide context and best practices in model checkpointing. These resources offer a comprehensive exploration of both basic and advanced strategies, supplementing my experience-based approach.
