---
title: "How to resume Keras neural network training from a specific epoch?"
date: "2025-01-30"
id: "how-to-resume-keras-neural-network-training-from"
---
Resuming Keras neural network training from a specific epoch is a common requirement when dealing with time-consuming model training or when needing to adjust parameters mid-process. In my experience, this is often necessitated by unexpected system interruptions, hyperparameter tuning experiments that require restarting from a previous state, or even the need to fine-tune a previously trained model on new data. The Keras API, built on TensorFlow or other backends, provides robust mechanisms for achieving this by leveraging model saving and loading functionalities alongside the concept of a training state.

The fundamental principle is to save both the model architecture and the learned weights (i.e., the model state) at regular intervals, typically at the end of each epoch or after a specific number of training steps. This saved state acts as a snapshot, enabling you to restart training at that precise point later on. Keras implements this through callbacks, objects that execute actions at various stages during training, particularly the ModelCheckpoint callback. This callback monitors metrics and saves the model at the end of each epoch or at specific intervals if configured to do so. When resuming, you load the model weights from a saved file (typically an HDF5 file, with a .h5 extension) and then call the `fit` method on the loaded model. Crucially, you will need to manually specify the starting epoch for the resumed training. By doing this, the `fit` method will begin updating weights starting with the next epoch, effectively picking up where it left off. The optimizer state is also persisted, so learning rate schedules, momentum terms, and other optimizer parameters are correctly loaded.

Here are three examples demonstrating how to resume Keras training from a particular epoch, highlighting different nuances:

**Example 1: Basic Resumption with ModelCheckpoint**

This example shows a basic setup, saving the model at the end of each epoch and then resuming from a specified saved epoch.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define a simple model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

# Initial training for 5 epochs, saving at end of each epoch
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint/epoch_{epoch:02d}.h5',
    save_freq='epoch'
)

history = model.fit(X_train, y_train, epochs=5, callbacks=[checkpoint_callback])


# Simulate stopping training after epoch 3
# Load model from epoch 3
loaded_model = keras.models.load_model('model_checkpoint/epoch_03.h5')

# Resume training from epoch 3 for 5 more epochs
resume_history = loaded_model.fit(X_train, y_train,
                                  initial_epoch=3,
                                  epochs=8)  # Trains for 5 more epochs from epoch 3, up to epoch 8
```

In this example, the `ModelCheckpoint` callback saves model weights into different files named based on the current epoch number. This aids in easily identifying where to load the weights from. After the initial training, the model is loaded from the checkpoint corresponding to epoch 3 using `keras.models.load_model()`, and then `fit()` is called again, setting `initial_epoch` to 3 and the total `epochs` to 8, effectively resuming training from the fourth epoch (epochs start with 0). The learning rate and other optimizer parameters are implicitly loaded along with the model state. The file naming convention, with placeholders such as `{epoch:02d}`, allows us to control file organization in a robust way.

**Example 2: Resuming with a Single Saved Model and Overwriting Strategy**

This example demonstrates using a single saved model file and overwriting it at each epoch. This approach saves disk space but will only allow resuming from the last saved state.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define the same model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)


# Single model checkpoint
checkpoint_path = 'model_checkpoint/single_model.h5'
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',
    save_best_only=False,  # Important: disable save best only for this example
    overwrite=True
)

history = model.fit(X_train, y_train, epochs=5, callbacks=[checkpoint_callback])

# Load the single saved model
loaded_model = keras.models.load_model(checkpoint_path)

# Resume training for 5 epochs, total epochs will be 10
resume_history = loaded_model.fit(X_train, y_train,
                                  initial_epoch=5,
                                  epochs=10)
```

Here, `save_best_only` is explicitly set to `False` and `overwrite` to `True` in the `ModelCheckpoint` callback. The `save_best_only` flag, when set to `True`, only saves the model if the monitored metric improves. Since we want to save the model at every epoch in this scenario, we set it to `False`.  The model weights at the end of each epoch will overwrite the previous state on disk. The training will then continue from where it left off by setting the `initial_epoch` to the epoch at which training was interrupted. The flexibility of the `ModelCheckpoint` allows for custom saving behavior to fit various scenarios.

**Example 3: Resuming with Custom Checkpoint Logic**

This example demonstrates saving/loading with a custom logic in case you need to persist more than model weights, for example, a learning rate that changed during training or the epoch number itself.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

# Define the same model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

checkpoint_dir = "custom_checkpoint"

def save_checkpoint(epoch, model, optimizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.h5")
    model.save_weights(model_path)

    # Save additional states (e.g., optimizer state, current epoch)
    checkpoint_data = {
        "epoch": epoch,
        "learning_rate": optimizer.learning_rate.numpy()
    }
    with open(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.json"), 'w') as f:
         json.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_dir, epoch, model, optimizer):
   model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.h5")
   model.load_weights(model_path)

   with open(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.json"), 'r') as f:
        checkpoint_data = json.load(f)

   optimizer.learning_rate.assign(checkpoint_data["learning_rate"])

   return  checkpoint_data["epoch"]


# Train for 5 epochs, with a custom save function
for epoch in range(5):
   history = model.fit(X_train, y_train, epochs=1, verbose=0)
   save_checkpoint(epoch, model, optimizer, checkpoint_dir)
   print(f"Finished Epoch {epoch}")



# Resume Training from epoch 3
resume_epoch = load_checkpoint(checkpoint_dir, 3, model, optimizer) + 1


resume_history = model.fit(X_train, y_train,
                                  initial_epoch=resume_epoch,
                                  epochs=10)
```

In this example, we defined custom `save_checkpoint` and `load_checkpoint` functions that handle saving not only the model's weights but also the learning rate of the optimizer and the current epoch number in a separate JSON file. This demonstrates how to extend basic checkpointing to include more information that may be necessary for specific use cases. The `load_checkpoint` function then loads the model's weights as well as restoring the optimizer learning rate and then returns the loaded epoch + 1 for a correct `initial_epoch` parameter during `fit` call. This is a more advanced way of checkpointing and it gives full control over what is persisted between training runs.

For further exploration of model checkpointing and related Keras features, I recommend exploring the official Keras documentation, specifically the sections covering the `ModelCheckpoint` callback, model saving and loading, and training workflows. Textbooks and tutorials on deep learning using Keras and TensorFlow often include discussions on these topics that may provide more practical context. Finally, research articles detailing specific training techniques or strategies may provide deeper insights into best practices for large-scale model training and recovery. Understanding these resources will solidify a foundation on which more advanced techniques can be built.
