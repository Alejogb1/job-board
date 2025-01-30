---
title: "How to display console messages during TensorFlow 2 training in a Jupyter Notebook?"
date: "2025-01-30"
id: "how-to-display-console-messages-during-tensorflow-2"
---
TensorFlow 2's training loop, when executed within a Jupyter Notebook, can sometimes exhibit a delay or even suppress console output, obscuring crucial debugging and progress information.  This stems from how Jupyter handles standard output redirection and TensorFlow's internal logging mechanisms. I've experienced this directly while fine-tuning several transformer models, where silent training runs made it difficult to track validation loss and identify potential overfitting issues in real-time. To effectively address this, one needs to understand and manipulate TensorFlow's logging behavior in conjunction with Jupyter’s execution environment.

The primary issue is that TensorFlow, by default, routes its training updates and logging to Python's `logging` module. While this is a robust system, Jupyter captures and buffers these outputs, often delaying or omitting their display in the notebook output cell. The buffered output is usually flushed only after the training loop finishes or during specific notebook operations, thus hindering immediate feedback. To circumvent this, the approach requires modifying how TensorFlow's logging is handled or, alternatively, implementing a custom callback that explicitly prints status messages.

The most basic solution is to directly access and modify the root logger using Python's `logging` module. By setting the logging level and adding a suitable handler, we can force TensorFlow's messages to be printed immediately to the console, bypassing Jupyter's buffered handling. Below is an initial example illustrating this approach:

```python
import tensorflow as tf
import logging

# Configure the root logger
logging.getLogger().setLevel(logging.INFO) # Or logging.DEBUG for more detailed output
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

# Dummy model and training data
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train = tf.random.normal(shape=(1000, 784))
y_train = tf.random.uniform(shape=(1000, 1), maxval=2, dtype=tf.int32)

# Training
model.fit(x_train, y_train, epochs=5)
```

In this first example, we're taking direct control of the global logging mechanism. The `logging.getLogger()` method retrieves the root logger. Setting the level to `logging.INFO` or `logging.DEBUG` determines what level of detail is shown in the output. By adding a `StreamHandler`, we direct log messages to the standard output, bypassing Jupyter's buffering. This directly impacts output during training, providing a much better sense of the learning process. Note the inclusion of a `Formatter` which can make the output easier to parse by including timestamps and log levels. This method, while effective, can produce verbose output, depending on the TensorFlow operations occurring.

A second, more granular approach involves creating a custom Keras callback. Keras callbacks provide a mechanism to intervene at various points in the training loop (start of an epoch, end of a batch, etc.). By defining a custom callback, we can explicitly print information related to loss or other metrics directly to the console. This method provides more fine-grained control over what and when information is displayed. Consider this implementation:

```python
import tensorflow as tf

class ConsoleLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}")

# Dummy model and training data (as in the previous example)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
x_train = tf.random.normal(shape=(1000, 784))
y_train = tf.random.uniform(shape=(1000, 1), maxval=2, dtype=tf.int32)

# Training with the custom callback
model.fit(x_train, y_train, epochs=5, callbacks=[ConsoleLoggerCallback()])
```

The `ConsoleLoggerCallback` above extends `tf.keras.callbacks.Callback` and overrides the `on_epoch_end` method. At the end of each training epoch, this custom callback extracts metrics such as loss and accuracy from the `logs` dictionary and prints them to the console using an f-string, formatted for better readability. The key benefit of this method lies in its customization; it is possible to tailor it to display any metric or information pertinent to a particular training session, unlike the blanket solution from the first example.

Finally, in scenarios involving distributed training or more complex models,  a combination of logging modification and custom callbacks proves to be most effective.  The previous example was designed to work in a non-distributed setting, and could be modified to work in such a scenario. This usually also involves using the `tf.distribute` API. Here is an example with a custom logging callback for distributed training, in this instance just demonstrating print out, rather than logging to file:

```python
import tensorflow as tf

class DistributedConsoleLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
           logs = {}
        with self.strategy.scope():
           print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}")

# Dummy model and training data (as in previous examples)
strategy = tf.distribute.MirroredStrategy() # Or any other distribution strategy
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


x_train = tf.random.normal(shape=(1000, 784))
y_train = tf.random.uniform(shape=(1000, 1), maxval=2, dtype=tf.int32)

# Training with distributed strategy and custom callback
model.fit(x_train, y_train, epochs=5, callbacks=[DistributedConsoleLoggerCallback(strategy)])
```

This example introduces `tf.distribute.MirroredStrategy`. With the model defined in the distribution scope, and the callback leveraging it for output, the logging is forced to execute in a distributed context. While output might be somewhat jumbled by concurrent executions in each replica, this method ensures that the information is displayed, regardless. This approach is the most robust for complex projects where standard logging and callbacks alone may not suffice. It directly addresses the challenges of delayed and buffered output in multi-replica training scenarios. It is important to note that depending on the distribution strategy used, output from each replica might be intermingled. Techniques like logging to a shared file might be required in production scenarios.

For further study, consider exploring the official TensorFlow documentation, specifically sections relating to: custom callbacks, logging module configuration, and the `tf.distribute` API. Resources such as *Deep Learning with Python* by Chollet and any of the textbooks from the Deep Learning specialization on Coursera offer detailed explanations about implementing custom callbacks. Understanding these resources will improve the ability to effectively monitor and control training behavior within Jupyter Notebooks. These approaches, gained through both theoretical understanding and practical troubleshooting, provide a clear path for resolving issues with console output during TensorFlow training within a Jupyter environment.
