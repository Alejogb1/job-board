---
title: "How can I plot TensorFlow model history from a checkpoint using Matplotlib?"
date: "2025-01-30"
id: "how-can-i-plot-tensorflow-model-history-from"
---
TensorFlow's ModelCheckpoint callback, while crucial for saving model weights, does not inherently preserve the training history across multiple runs of the same model if the training is interrupted and resumed from a checkpoint. This presents a challenge when visualizing the training process using Matplotlib. I've frequently encountered this issue when dealing with long-running deep learning training sessions, often requiring bespoke solutions for proper monitoring and analysis. My approach centers on extracting history information during training and leveraging that data for subsequent plotting.

The core issue is that the `fit()` method of a TensorFlow model returns a history object only for the current training session. If training halts and resumes using checkpointed weights, the subsequent `fit()` call generates a new history object, potentially overwriting previous metrics. Therefore, the crucial step involves manually logging training metrics throughout the training process and then loading both the model from the checkpoint and the logged history data separately. This enables Matplotlib-based visualization without losing valuable training information.

Here's a strategy using callbacks and a simple file saving system, followed by a post-training data merging and plotting implementation:

**Phase 1: Logging Training Metrics during training**

Instead of relying solely on the history object from a single `fit()` call, I'll create a custom callback to save the metrics per epoch. The approach here is to accumulate the metric information (e.g., loss, accuracy, and their validation counterparts) and write these values to a data file. A convenient data format is JSON, which allows for simple structured data representation.

```python
import tensorflow as tf
import json
import os

class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath="training_history.json"):
        super().__init__()
        self.filepath = filepath
        self.history = []

        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                try:
                  self.history = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Existing history file corrupted, starting new history")


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history.append(logs)
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=4)

# Example Usage during training
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint/model.ckpt',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

history_callback = HistoryCallback(filepath="training_history.json")

# Sample Data - replace with your actual data.
import numpy as np
x_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 2, 1000)
x_val = np.random.rand(200, 784)
y_val = np.random.randint(0, 2, 200)


model.fit(x_train, y_train,
          epochs=5,
          validation_data=(x_val,y_val),
          callbacks=[checkpoint_callback, history_callback])

```

In this code snippet, a custom `HistoryCallback` subclass is introduced. Upon initialization, it attempts to load any existing history from the specified `filepath`. Then, after each epoch completes, it appends the metrics to the internal `self.history` list and saves that data back into the JSON file. During training, both the model checkpoint callback and this custom history callback are used. Consequently, the history will be persisted across training session regardless of interruptions. The existing history is also gracefully handled when present to avoid data loss, which in my experience is critical.

**Phase 2:  Loading Checkpoint and Merging History for Plotting**

Now, after the model has been trained (potentially across multiple runs, saving checkpoints), the next step is to load the checkpoint and history. I'll then extract the relevant metrics and generate visualizations. The advantage here is that I'm not relying on the `fit()` method's return. I am working with the saved information.

```python
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import numpy as np


def plot_training_history(history_path, checkpoint_path):
    with open(history_path, 'r') as f:
        history = json.load(f)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    # Load the latest weights
    model.load_weights(checkpoint_path)
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    epochs = len(history)

    if epochs == 0:
        print("No training history available.")
        return

    loss = [entry['loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]
    accuracy = [entry['accuracy'] for entry in history]
    val_accuracy = [entry['val_accuracy'] for entry in history]


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.show()


#Example Usage:
plot_training_history("training_history.json", "model_checkpoint/model.ckpt")

```

Here, a `plot_training_history` function takes the history and checkpoint paths as inputs.  It loads the recorded history from the JSON file and loads the model architecture, then loads the checkpointed weights into it. From the combined information, it creates two subplots, plotting the loss and accuracy for the training and validation sets against each epoch. Note that the model is re-created prior to loading the saved weights; only weights are saved, not the entire model structure. The use of `plt.tight_layout()` is also crucial to ensure no labels overlap in the created charts.

**Phase 3: Handling Checkpoints During Multi-Run Training**

A common scenario is when model training occurs over multiple runs and you're not always sure what point of the training process you stopped at.  The previous examples do work correctly if the training is stopped and resumed, but in my experience, a good implementation includes explicit handling of the potential for restarting training that has previously been interrupted.

```python
import tensorflow as tf
import json
import os
import numpy as np

def continue_training_from_checkpoint(model, training_data, validation_data, epochs,
                                     checkpoint_path = 'model_checkpoint/model.ckpt',
                                     history_path="training_history.json"):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    history_callback = HistoryCallback(filepath=history_path)

    model.fit(training_data[0],training_data[1],
              epochs=epochs,
              validation_data=validation_data,
              callbacks=[checkpoint_callback, history_callback],
              initial_epoch=len(history_callback.history)
    )


# Example Usage
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 2, 1000)
x_val = np.random.rand(200, 784)
y_val = np.random.randint(0, 2, 200)
training_data = (x_train, y_train)
validation_data = (x_val, y_val)


continue_training_from_checkpoint(model, training_data, validation_data, epochs=5)

# Example for subsequent training
continue_training_from_checkpoint(model, training_data, validation_data, epochs=5)
plot_training_history("training_history.json", 'model_checkpoint/model.ckpt')
```

The updated example introduces `continue_training_from_checkpoint`. It determines the number of epochs already run by reading the previously saved history data and passes that information to the fit method using the `initial_epoch` argument. The code will load weights and metrics from any prior training, making it easy to train over multiple sessions.

**Resource Recommendations**

For further understanding and exploration, the official TensorFlow documentation on `tf.keras.callbacks.ModelCheckpoint` and creating custom callbacks provides a good foundation. The Matplotlib library documentation is essential for creating more tailored visualizations. Additionally, the JSON library documentation provides the knowledge needed for saving and loading data effectively.  I recommend a focus on understanding the core APIs and developing a working solution instead of relying on overly complex or external packages.
