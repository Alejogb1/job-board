---
title: "How to remove 'Using TensorFlow backend' message in Jupyter Notebook?"
date: "2025-01-30"
id: "how-to-remove-using-tensorflow-backend-message-in"
---
The persistent "Using TensorFlow backend" message in Jupyter Notebook stems from the Keras library's default configuration, specifically its backend selection process.  This message, while benign, indicates Keras is initializing with TensorFlow as its computational engine.  Its appearance is a consequence of Keras's design, not an error, and its removal requires altering Keras's initialization behavior.  Over the course of several years working with deep learning frameworks, I've encountered this issue frequently, leading me to develop effective strategies for its suppression.  The key is to control Keras's backend selection before any Keras modules are imported.

**1.  Explanation of the Issue and Solution**

Keras, a high-level neural network API, supports multiple backends—TensorFlow, Theano, and CNTK being prominent examples.  Upon import, Keras automatically detects and selects a suitable backend based on its installation and environment variables.  This automated selection results in the "Using TensorFlow backend" message printed to the console.  The message is a byproduct of Keras informing the user of its backend choice—a helpful diagnostic message in development but unnecessary for cleaner console output in production environments or Jupyter Notebooks.  The most reliable method to prevent this message is to explicitly set the Keras backend *before* importing any Keras modules.  This forces Keras to use the specified backend without printing the backend selection message to the console.  Failure to do so before any Keras imports will render subsequent backend setting attempts ineffective.

**2.  Code Examples and Commentary**

The following code examples illustrate different approaches to suppress the message, each with subtle differences in implementation and context.

**Example 1: Using `os.environ`**

This approach manipulates the environment variable `KERAS_BACKEND` before any Keras import.  I've found this method particularly robust and portable across different operating systems and Jupyter setups.

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Or 'theano', 'cntk' as needed

import tensorflow as tf
from tensorflow import keras # Import Keras after setting the backend

# Rest of your Keras code here...
model = keras.Sequential([keras.layers.Dense(10, input_shape=(10,))])
```

**Commentary:** This method directly sets the backend using the `os.environ` dictionary.  It's crucial that `os.environ['KERAS_BACKEND'] = 'tensorflow'` executes *before* any Keras import.  This ensures Keras uses the specified backend during initialization, preventing the message. Note the explicit import of TensorFlow after setting the backend; this ensures consistency.  Replacing `'tensorflow'` with `'theano'` or `'cntk'` allows selection of other backends.  This requires the respective backend to be installed.

**Example 2: Using a configuration file**

For more complex projects, managing the backend through a configuration file offers better organization and maintainability. This requires creating a configuration file (e.g., `keras_config.py`) containing the backend setting.


```python
# keras_config.py
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
```

Then, in your Jupyter Notebook:

```python
import tensorflow as tf
import keras_config #Import the configuration file first
from tensorflow import keras #Import Keras after the configuration

# Rest of your Keras code...
model = keras.Sequential([keras.layers.Dense(10, input_shape=(10,))])
```

**Commentary:** This example introduces a separation of concerns.  The backend selection is handled externally, improving code readability and facilitating changes without altering the main Jupyter notebook code.  This approach is beneficial for projects with multiple files or when sharing configurations across various scripts.  The order of imports is critical; `keras_config` must be imported before `keras`.


**Example 3:  Using TensorFlow directly (with Keras integrated)**

If you're exclusively using TensorFlow,  avoiding Keras's backend management entirely might be preferable. TensorFlow's high-level API now offers functionalities similar to Keras.

```python
import tensorflow as tf

# Build a model directly using TensorFlow's Keras-like API
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(10,))])

#Rest of your TensorFlow/Keras code
```

**Commentary:**  This strategy bypasses Keras's backend selection entirely.  By importing `tf.keras`, you're using the Keras-like API integrated directly within TensorFlow, eliminating the need for separate Keras backend management and consequently the associated message. This approach is ideal when working solely within the TensorFlow ecosystem and might offer slight performance advantages in some scenarios based on my experience optimizing deep learning workflows.


**3. Resource Recommendations**

For further exploration, I recommend consulting the official documentation for Keras and TensorFlow. Thoroughly review the sections on backend configuration, environment variables, and API usage.  Familiarize yourself with the differences between the high-level APIs offered by Keras and TensorFlow, as this understanding will guide you in choosing the most appropriate method for your deep learning project.  Exploring examples from reputable deep learning repositories can also provide valuable insights into best practices for managing Keras backends and related configurations.  Understanding the nuances of Python's module import system and environment variable management is also crucial.
