---
title: "How to selectively log metrics during Keras/Tensorflow model training?"
date: "2025-01-30"
id: "how-to-selectively-log-metrics-during-kerastensorflow-model"
---
During the development of a complex deep learning model for image segmentation – a project I led involving a UNet architecture with multiple custom loss functions – I encountered the common problem of excessively verbose logging. Standard Keras callbacks, while useful, often produce a torrent of metrics during each training epoch, making it difficult to pinpoint areas needing attention. This experience highlighted the need for a granular approach to metric logging, enabling selective output based on specific conditions or phases of training. Implementing such control requires understanding how Keras callbacks operate and how to leverage them for custom behavior.

The core challenge lies in the fact that Keras callbacks, by design, execute a predefined set of actions on each batch or epoch. The `on_batch_end` and `on_epoch_end` methods, for example, fire regardless of internal conditions within the training loop. Therefore, achieving selective logging requires injecting conditional logic into these methods, which is facilitated through custom callback classes inheriting from `keras.callbacks.Callback`. The key here is accessing the `logs` dictionary provided within callback methods, which contains the current state of metrics, and controlling print or logging statements accordingly.

Furthermore, effective metric logging often depends on the *type* of metric being tracked. A loss value might be relevant at every epoch, while a more specialized metric, such as the Intersection over Union (IoU) in my segmentation project, might be more useful at the end of each epoch after sufficient convergence. Additionally, it’s beneficial to selectively log during early epochs to monitor convergence or during later epochs to observe fine-tuning, thereby avoiding excessive terminal clutter. The ability to selectively log also proves crucial when dealing with multiple models being trained simultaneously on a single GPU or cluster where log management can become complicated very quickly. This allows for better focus and understanding of each model's training progression without sifting through unnecessary output.

Let's examine a few code examples to illustrate this process.

**Example 1: Selective Epoch-Based Logging**

This example demonstrates logging only training loss and validation accuracy, and only after a specified initial number of epochs.

```python
import tensorflow as tf
import keras
from keras.callbacks import Callback
import numpy as np

class SelectiveEpochLogger(Callback):
    def __init__(self, start_epoch=5):
        super().__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            print(f"Epoch {epoch + 1}: Training Loss = {logs['loss']:.4f}, Validation Accuracy = {logs['val_accuracy']:.4f}")


# Sample Model Setup
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Add validation data for accurate demo.
val_X = np.random.rand(50, 10)
val_y = np.random.randint(0, 2, 50)

# Initiate the custom callback
selective_logger = SelectiveEpochLogger(start_epoch=3)

model.fit(X, y, epochs=10, batch_size=32, validation_data=(val_X, val_y), callbacks=[selective_logger], verbose = 0)
```

In this code, the `SelectiveEpochLogger` callback is initialized with `start_epoch=3`. During the first two epochs, no output will be generated. Starting from epoch three, the callback will print the training loss and validation accuracy. This demonstrates the basic control of logging based on a simple epoch threshold. The `verbose=0` argument in `model.fit` is added to remove the standard logging so the custom logs can be more easily seen.

**Example 2: Selective Metric-Based Logging within a Custom Callback**

Building upon the first example, this one allows logging based on a *specific metric value*. In this case, the code monitors validation loss and logs an output *only if it decreases* compared to the previous epoch.

```python
import tensorflow as tf
import keras
from keras.callbacks import Callback
import numpy as np

class ImprovedValidationLogger(Callback):
    def __init__(self):
        super().__init__()
        self.previous_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.previous_val_loss:
            print(f"Epoch {epoch + 1}: Validation Loss improved from {self.previous_val_loss:.4f} to {current_val_loss:.4f}")
        self.previous_val_loss = current_val_loss if current_val_loss is not None else self.previous_val_loss

# Sample Model Setup (same as previous)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Add validation data for accurate demo.
val_X = np.random.rand(50, 10)
val_y = np.random.randint(0, 2, 50)

# Initiate the custom callback
validation_logger = ImprovedValidationLogger()

model.fit(X, y, epochs=10, batch_size=32, validation_data=(val_X, val_y), callbacks=[validation_logger], verbose=0)

```
Here, the `ImprovedValidationLogger` stores the previous validation loss and prints only if there is an improvement. A key point to observe is the use of `logs.get('val_loss')` which gracefully handles scenarios where validation data is not supplied, avoiding `KeyError`. This emphasizes the importance of checking the availability of metrics within the `logs` dictionary, particularly during validation.

**Example 3: Batch-level logging with conditional frequency**

This third example demonstrates batch-level logging, displaying the training loss only every N batches.

```python
import tensorflow as tf
import keras
from keras.callbacks import Callback
import numpy as np

class BatchStepLogger(Callback):
    def __init__(self, log_frequency=10):
        super().__init__()
        self.log_frequency = log_frequency
        self.batch_count = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.log_frequency == 0:
            print(f"Batch {batch+1}: Training Loss = {logs['loss']:.4f}")

# Sample Model Setup (same as previous)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Initiate the custom callback
batch_logger = BatchStepLogger(log_frequency=5)

model.fit(X, y, epochs=2, batch_size=32, callbacks=[batch_logger], verbose=0)

```

The `BatchStepLogger` keeps track of the batch number and, when the batch count is a multiple of `log_frequency`, it displays the training loss. This type of log is especially helpful for debugging or inspecting metrics at the batch level during training. This is particularly beneficial when training larger models or in scenarios requiring close examination of mini-batch behavior.

These examples illustrate the power and flexibility of custom Keras callbacks for selective metric logging. It's essential to understand that a callback is a class with event handling methods. The `logs` dictionary is the key to accessing the metric values collected by the model. Leveraging the conditions and logging based on the various training events or metric levels is a powerful way to control the output stream.

For further exploration of related topics, I would recommend consulting these resources:

1.  Keras documentation on callbacks: This is the primary resource for understanding callback functionality and the available methods. A deep dive into the API reveals possibilities not covered in tutorials.
2.  Advanced Keras examples: There are several examples available within the Keras Github repository and elsewhere that demonstrates building complex callbacks that manipulate model training parameters or collect custom metrics.
3.  Tensorflow guide on writing custom callbacks: The Tensorflow website’s guide covers the underlying callback mechanism in detail which allows for deeper understanding.

In conclusion, selective logging is not merely an aesthetic improvement. It is a practical necessity for handling complex model training scenarios. The ability to conditionally display metric outputs reduces information overload and allows for efficient debugging and monitoring. Through custom callbacks, a nuanced level of control can be achieved, catering to a wide range of machine learning needs and workflows.
