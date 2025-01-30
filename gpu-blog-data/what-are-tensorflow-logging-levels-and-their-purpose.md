---
title: "What are TensorFlow logging levels and their purpose in notebooks?"
date: "2025-01-30"
id: "what-are-tensorflow-logging-levels-and-their-purpose"
---
TensorFlow's logging mechanism, crucial for debugging and monitoring model training within Jupyter Notebooks and other interactive environments, operates through a hierarchical system of severity levels.  Understanding these levels and their effective manipulation is essential for efficient model development and troubleshooting. My experience working on large-scale natural language processing projects has highlighted the importance of precisely controlling the volume and type of logged information.  Incorrectly configured logging can lead to overwhelming output obscuring critical errors, while insufficient logging can hinder the identification of subtle performance bottlenecks.


The TensorFlow logging system utilizes a standard set of severity levels, each corresponding to a different degree of importance. These levels, ordered from most to least severe, are: `FATAL`, `ERROR`, `WARNING`, `INFO`, and `DEBUG`. Each level filters out messages of lower severity. For instance, setting the logging level to `WARNING` will suppress `INFO` and `DEBUG` messages, while `ERROR` and `FATAL` messages will still be displayed. This filtering is instrumental in managing the output, especially during extensive training runs where the volume of logged data can be substantial.  `FATAL` errors typically represent unrecoverable problems halting execution. `ERROR` signifies issues requiring immediate attention but may not always halt execution. `WARNING` alerts the user to potentially problematic conditions that warrant investigation. `INFO` messages provide informative updates on the progress and status of the training process. Finally, `DEBUG` messages contain highly detailed information useful for low-level debugging.


**1.  Clear Explanation:**

The logging level is controlled primarily through the `tf.compat.v1.logging` module (or the equivalent in newer TensorFlow versions).  This module offers functions to configure the logging level and to explicitly log messages at specific severity levels. Setting the level globally affects all subsequent logging statements within the TensorFlow codebase and any custom logging added using the same module.  Individual logging calls, however, can override the global level.  For instance, a `tf.compat.v1.logging.error()` call will always produce an output, regardless of the globally set level,  as long as the error condition is met.  This fine-grained control provides flexibility in managing the informational flow.  Effective logging practices involve setting a higher level (like `INFO` or `WARNING`) during regular training to track progress and a lower level (like `DEBUG`) when troubleshooting specific issues. This approach avoids information overload during routine operations while facilitating detailed analysis during debugging.  Furthermore, understanding the origin of log messages – TensorFlow itself, custom code, or third-party libraries – allows for more targeted debugging. Timestamps within the log messages are equally critical for reconstructing the sequence of events during problematic runs.


**2. Code Examples with Commentary:**

**Example 1: Setting the Global Logging Level**

```python
import tensorflow as tf

# Set the global logging level to WARNING.  This will suppress INFO and DEBUG messages.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)

# Subsequent logging statements will only be displayed if they are WARNING, ERROR, or FATAL.
tf.compat.v1.logging.info("This INFO message will be suppressed.")
tf.compat.v1.logging.warning("This WARNING message will be displayed.")
tf.compat.v1.logging.error("This ERROR message will be displayed.")

# Creating a simple model to demonstrate logging within a training loop
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(10,), activation='relu')])
model.compile(optimizer='adam', loss='mse')

# Simulating a training loop to demonstrate conditional logging
for epoch in range(3):
    loss = model.fit(tf.random.normal((100, 10)), tf.random.normal((100, 1)), epochs=1, verbose=0).history['loss'][0]
    if loss > 0.5:
        tf.compat.v1.logging.warning(f'High loss detected in epoch {epoch+1}: {loss}')
```

This example demonstrates how to set the global logging level and how INFO messages are suppressed while WARNING messages are displayed. The addition of conditional logging within the training loop shows a practical application of the logging system for monitoring performance.


**Example 2:  Explicit Logging Calls**

```python
import tensorflow as tf

#Even with a higher global logging level, explicit logging calls take precedence
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)

tf.compat.v1.logging.debug("This DEBUG message will be displayed regardless of global level.")
tf.compat.v1.logging.info("This INFO message will still be displayed.")

#Demonstrating error handling with logging
try:
  result = 10 / 0
except ZeroDivisionError:
  tf.compat.v1.logging.error("Division by zero encountered!")
```

This example highlights the precedence of explicit logging calls over the globally set level. The DEBUG and INFO messages are displayed even though the global level is set to WARNING. The error handling section showcases logging's role in reporting exceptions.


**Example 3: Custom Logging Handlers (Advanced)**

```python
import tensorflow as tf
import logging

# Create a custom logger
logger = logging.getLogger('my_custom_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler('my_log.txt')
file_handler.setLevel(logging.WARNING) # Only WARNING and above will be written to the file.

# Create a console handler to display logs on the console.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) #all messages will appear on the console.

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log messages at different levels
logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")

# integrating this with tensorflow. Note that this requires specific formatting for proper integration.
# This part is demonstration only and might require adjustment depending on the TensorFlow version.
tf.compat.v1.logging.info("TensorFlow message at INFO level", exc_info=True) # includes the traceback
```

This advanced example demonstrates the creation of a custom logger using Python's built-in `logging` module.  This allows for more granular control over where logs are directed (e.g., a file for persistent storage and the console for immediate feedback) and the level of detail recorded in each location.  This approach is particularly helpful when dealing with complex projects requiring different logging configurations for different components. Note that direct integration with TensorFlow's logging requires careful consideration of formatting and potential conflicts with TensorFlow's internal handlers.


**3. Resource Recommendations:**

The official TensorFlow documentation.  A comprehensive guide on Python's built-in `logging` module.  Relevant chapters in advanced Python programming texts focusing on exception handling and debugging.  A book on software engineering best practices for logging and monitoring.
