---
title: "How to suppress TensorFlow INFO messages about missing checkpoint paths?"
date: "2025-01-30"
id: "how-to-suppress-tensorflow-info-messages-about-missing"
---
TensorFlow's verbosity, particularly regarding checkpoint paths during model loading, can clutter logs and hinder readability in production environments. I’ve encountered this repeatedly while deploying models trained on distributed clusters, where the absence of a specific checkpoint file on a worker node is a common, benign occurrence that doesn't impede the primary training process. These `INFO` messages, while potentially helpful during debugging, can become a distracting noise in stable applications.

The core problem stems from TensorFlow’s logging infrastructure, which defaults to a relatively high verbosity level, often emitting messages that, while technically informative, are not actionable in many use cases. Specifically, when loading a model, TensorFlow frequently checks for various checkpoint files. If a file is not found, a potentially misleading `INFO` message is generated even if the model load is ultimately successful due to other available checkpoints or alternative loading mechanisms. It is not an error but an informational message, however, the frequency of these messages creates log noise.

The solution lies in adjusting TensorFlow’s logging level. TensorFlow employs a mechanism for controlling the verbosity of its output, and this can be manipulated through environment variables or programmatically within the application code itself. We are not attempting to mask errors, simply to suppress the low-level informational output that is deemed unnecessary for the operational context. Suppressing the output is done by setting the logging level above `INFO`.

I've found that there are a couple of ways to achieve this, but the most robust and portable method is to configure the logging level before TensorFlow is imported. Here is the explanation and rationale for my method.

First, the ideal approach is to use the `os` module in Python, allowing you to adjust the environment variable that dictates TensorFlow’s log level. This ensures that the logging configuration is effective across all TensorFlow operations throughout the process, avoiding the need to manually adjust logging at multiple points. The environment variable we need is `TF_CPP_MIN_LOG_LEVEL`. Assigning the integer value `1` to it effectively sets the logging level to WARNING, suppressing all messages with a level of INFO or lower. We are not setting it to `2` to suppress `WARNING` messages as `WARNING` messages can be useful. Setting it to `0` would display all messages which is the default, and setting it to `3` will suppress errors which should never be suppressed.

This method works by modifying the environment before TensorFlow imports, meaning that the logging level will persist. Once the logging is suppressed, we should never see `INFO` messages again when dealing with checkpoint paths. The advantage of this method is that it’s global for all TensorFlow operations, does not require manual logging setups and is easily portable. It can also be configured from a deployment script.

Below is an example demonstrating this approach.

```python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Sets the logging level to WARNING

import tensorflow as tf

# Now any INFO messages, including those about missing checkpoint paths, will be suppressed.
# Rest of your TensorFlow code
# Example usage of loading a checkpoint if available
try:
    model = tf.keras.models.load_model('my_model')
    print("Model loaded successfully")
except:
    print("Model load failed")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

```

In this code example, the environment variable `TF_CPP_MIN_LOG_LEVEL` is set to '1' using the `os` module. This step is done *before* TensorFlow is imported, ensuring the desired logging level is applied during the library’s initialization phase. Afterwards, the `tf` module is imported, and from now on all low-level `INFO` messages from TensorFlow are suppressed. If a model load fails, as demonstrated in the `try` `except` block, the messages will be suppressed and the default code in the except block will execute as if a model is not loaded. The code example attempts to load a model and if that fails, a new model is initialized. Without logging suppression, an `INFO` message will be displayed if the model file is not present, which we no longer see with this suppression method.

Another approach involves directly using TensorFlow’s logging functions, although I find it less consistent and more verbose than the first approach. It modifies the global logging level to only display errors and warnings, therefore suppressing less significant messages like the checkpoint ones. This requires a call to the `tf.get_logger().setLevel()` function, which must be set using the correct logging level.

Below is an example showing how to do it using the `tf.get_logger().setLevel()` approach.

```python
import tensorflow as tf

tf.get_logger().setLevel('WARNING')

# Now any INFO messages, including those about missing checkpoint paths, will be suppressed.
# Rest of your TensorFlow code

# Example usage of loading a checkpoint if available
try:
    model = tf.keras.models.load_model('my_model')
    print("Model loaded successfully")
except:
    print("Model load failed")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
```

In this code snippet, the logging level is configured directly through the TensorFlow logger. We call `tf.get_logger()` to retrieve the default logger and then set its level to `'WARNING'`. Any logging message of level `INFO` and lower, including the messages about missing checkpoints paths will then be suppressed. The rest of the code remains the same, so that even if a model is not loaded, the low-level `INFO` messages will be suppressed, demonstrating the functionality of the second approach. While this method achieves the same end result as the first, it is less reliable as it can be overridden later in the code. If another library or a different call also modifies the global logging level, the setting might be overwritten. This is why I prefer the environment variable based method.

Finally, I’ll also include an example of setting the log level via the `absl` library (part of TensorFlow's ecosystem), although I find the first approach more succinct. This method utilizes the `absl.logging` module. This is more verbose than the previous example, and also less portable as it relies on specific TensorFlow modules.

```python
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.WARNING)

# Now any INFO messages, including those about missing checkpoint paths, will be suppressed.
# Rest of your TensorFlow code

# Example usage of loading a checkpoint if available
try:
    model = tf.keras.models.load_model('my_model')
    print("Model loaded successfully")
except:
    print("Model load failed")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
```

Here, the `absl.logging.set_verbosity()` method is utilized, and the verbosity level is set to `logging.WARNING`. This effectively sets the minimum level for logging messages, therefore suppressing the same `INFO` level messages as the previous two approaches. The rest of the code remains the same, and the functionality is the same as the previous methods of suppressing log messages. While functional, the first approach using `os.environ` is preferable because of its portability and robustness.

For further information and details on how TensorFlow handles logging, refer to the official TensorFlow documentation. There are sections dedicated to controlling verbosity, which provide deeper insight into the logging hierarchy and available options. Also, checking the `absl` library documentation will assist in understanding the last example with `absl.logging`. Lastly, the general documentation for the Python `os` module will be very useful when dealing with environment variables in Python.
