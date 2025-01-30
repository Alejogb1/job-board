---
title: "How can CNN training learning curves be saved, recovered, and updated if a server crashes?"
date: "2025-01-30"
id: "how-can-cnn-training-learning-curves-be-saved"
---
Convolutional Neural Network (CNN) training involves computationally intensive processes that often span considerable time. Therefore, reliable saving and recovery mechanisms for learning curves and model parameters are essential when dealing with potential server crashes. A failure to implement these mechanisms correctly can lead to wasted resources and impede the entire research process.

Saving the learning curve, which typically tracks metrics such as loss and accuracy across training epochs, is crucial for performance analysis and hyperparameter tuning. This information, alongside the model's weights, forms a checkpoint. The approach taken to create, save, and reload these checkpoints directly impacts the resilience of the training procedure. A basic approach is to periodically save this information to persistent storage. This usually involves using a callback mechanism, within your framework of choice, that triggers the saving operation. This ensures that if a server crashes, the training can be resumed from the last saved state. The saved learning curve data allows you to visualize the progress of the training before the crash, and to monitor the effects of the recovery.

The learning curve is not merely a single file, but rather a growing collection of metrics over epochs. Therefore, it needs a structure that can efficiently manage this evolving dataset. I've personally found that combining the saving of model weights with the learning curve data in a structured manner helps reduce redundancy and keeps the whole process more coherent. Consider the following examples.

**Example 1: Basic Checkpointing with TensorFlow Keras**

In TensorFlow’s Keras API, a `ModelCheckpoint` callback can automate the process of saving model weights based on conditions like validation loss or epoch number. This alone is not enough to save learning curves. We have to save the history object separately in the `on_epoch_end` hook. This saves the trained weights after each epoch. While we're doing this, we can also log our learning curve data with that same `on_epoch_end` hook.

```python
import tensorflow as tf
import numpy as np
import json

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss'):
        super(CustomCallback, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.history_data = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
          logs = {}

        #Save Weights using the default checkpoint provided
        filepath_format = self.filepath + "weights-{epoch:02d}-{val_loss:.2f}.h5"
        self.model.save_weights(filepath_format.format(epoch=epoch, **logs))

        # Save the learning curve (metrics)
        current_data = {"epoch": epoch}
        current_data.update(logs)
        self.history_data.append(current_data)
        with open(self.filepath + "history.json", "w") as f:
            json.dump(self.history_data, f, indent=4)


# Assume 'model' is defined elsewhere
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a dummy dataset
x_train = np.random.rand(100,28,28,1)
y_train = np.random.randint(0,10,(100,))
x_val = np.random.rand(50,28,28,1)
y_val = np.random.randint(0,10,(50,))
checkpoint_filepath = './training_checkpoints/' #Directory to store all files

custom_callback = CustomCallback(checkpoint_filepath)

#Train the model, passing in the callback
model.fit(x_train, y_train, epochs=2, validation_data = (x_val, y_val), callbacks = [custom_callback])

```

In this example, the `CustomCallback` class saves both the model weights at each epoch and the training history into a JSON file. This simple design is adequate, but I've found that saving only the best weights by validation loss with early stopping often saves space and speeds the process up.

**Example 2: Best-Weight Checkpointing with Early Stopping**

```python
import tensorflow as tf
import numpy as np
import json
import os

class BestCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss'):
        super(BestCheckpointCallback, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best_val = float('inf')  # Initialize with positive infinity for val_loss
        self.history_data = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
          logs = {}

        # Save learning curve
        current_data = {"epoch": epoch}
        current_data.update(logs)
        self.history_data.append(current_data)
        with open(os.path.join(self.filepath, "history.json"), "w") as f:
             json.dump(self.history_data, f, indent=4)

        current_val = logs.get(self.monitor)
        if current_val is not None and current_val < self.best_val:
            self.best_val = current_val
            filepath_format = os.path.join(self.filepath, "best_weights.h5")
            self.model.save_weights(filepath_format)


    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        print("Best val:",self.best_val)

# Assume 'model' is defined elsewhere
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a dummy dataset
x_train = np.random.rand(100,28,28,1)
y_train = np.random.randint(0,10,(100,))
x_val = np.random.rand(50,28,28,1)
y_val = np.random.randint(0,10,(50,))
checkpoint_filepath = './training_checkpoints/' #Directory to store all files

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False)
best_checkpoint = BestCheckpointCallback(checkpoint_filepath)

#Train the model, passing in the callback
model.fit(x_train, y_train, epochs=10, validation_data = (x_val, y_val), callbacks = [best_checkpoint, early_stop])
```

Here, only the weights that correspond to the lowest validation loss are saved. Along with those, the history is still logged. In this particular example, `EarlyStopping` monitors the validation loss as well and prematurely ends training if it stops improving. While this has the benefit of saving storage, it does not guarantee the recovery from the last epoch trained before a crash if training is still progressing.

**Example 3: Loading and Resuming Training**

To resume training, the checkpoint must be loaded. This involves loading both the model weights and the history. Here’s a revised code snippet for that:

```python
import tensorflow as tf
import numpy as np
import json
import os

def resume_training(model, checkpoint_filepath, resume_epoch=None):
     # load history data
    history_path = os.path.join(checkpoint_filepath, "history.json")
    with open(history_path, "r") as f:
         loaded_history_data = json.load(f)

    last_epoch = len(loaded_history_data) - 1 # Get the last epoch of training
    if resume_epoch is not None and resume_epoch < last_epoch:
          print("Warning: Using provided epoch instead of recovered epoch")
          last_epoch = resume_epoch

    print("Resuming from Epoch:", last_epoch)

    # Load Model weights
    if last_epoch >= 0:
      filepath_format = os.path.join(checkpoint_filepath, "best_weights.h5")
      model.load_weights(filepath_format)

    #Reconfigure the callback to continue recording data
    resume_callback = BestCheckpointCallback(checkpoint_filepath)
    resume_callback.history_data = loaded_history_data


    return last_epoch, resume_callback

# Assume 'model' is defined elsewhere
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a dummy dataset
x_train = np.random.rand(100,28,28,1)
y_train = np.random.randint(0,10,(100,))
x_val = np.random.rand(50,28,28,1)
y_val = np.random.randint(0,10,(50,))
checkpoint_filepath = './training_checkpoints/' #Directory to store all files
max_epochs = 10

#To resume, pass in the model and checkpoint filepath
last_epoch, callback = resume_training(model, checkpoint_filepath)

#Train the model again
# We need to adjust the epochs parameter so that we keep training from where we left off
model.fit(x_train, y_train, initial_epoch = last_epoch, epochs = max_epochs, validation_data = (x_val, y_val), callbacks = [callback])
```

Here, the `resume_training` function reloads the history and the best weights, and also properly sets the `BestCheckpointCallback`. Using the `initial_epoch` parameter in the fit method, we can start from the last checkpointed epoch. The `resume_epoch` parameter is included to provide a fallback should we need to resume from something different than the last epoch.

When building this, I encountered multiple issues related to correctly maintaining the order of the history data. Loading the history data into the callback before the next epoch is essential for continuity of the logging process.  Also, remember that the file paths used to save the weights must be consistent with what's loaded.

For further learning, resources on TensorFlow's official documentation for callbacks and model saving are a great start. Furthermore, exploring the specifics of model checkpointing for your respective deep learning framework can significantly enhance the reliability of your training workflow. Beyond the framework-specific documentation, consider reading articles and blog posts on general deep learning training best practices, which often cover checkpointing strategies. Books focusing on best practices in practical machine learning workflows and software engineering also often provide guidance on these areas.
