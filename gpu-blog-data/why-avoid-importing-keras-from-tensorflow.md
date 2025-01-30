---
title: "Why avoid importing Keras from TensorFlow?"
date: "2025-01-30"
id: "why-avoid-importing-keras-from-tensorflow"
---
Importing Keras directly from TensorFlow, while seemingly convenient, introduces potential versioning conflicts and limits the flexibility that a standalone Keras library offers. My experience maintaining large-scale machine learning pipelines across various teams has consistently shown that decoupling these components leads to more robust and manageable codebases.

The core issue stems from how TensorFlow and Keras have evolved. Keras, originally a high-level API intended to be backend-agnostic, was subsequently integrated into TensorFlow as `tf.keras`. While this inclusion made Keras readily available within the TensorFlow ecosystem, it also tethered the Keras version to the specific TensorFlow release. This creates a challenge because Keras as a standalone library often receives updates and bug fixes independently of the core TensorFlow package. When using `tf.keras`, you're locked into the Keras version dictated by your TensorFlow installation, potentially missing out on improvements or encountering issues that have been resolved in a newer standalone Keras release.

Furthermore, the independent Keras library offers greater control over backends. While TensorFlow is a common backend, Keras also supports JAX and other platforms. By importing Keras directly (e.g., `import keras`), you retain the ability to switch backends by configuring the Keras environment. This is crucial in research contexts where experimentation across hardware accelerators or different numerical backends is necessary. Relying on `tf.keras` binds you to the TensorFlow ecosystem and limits your ability to easily port models or workloads to other environments.

Let's consider a practical example. Suppose you initially developed a model using TensorFlow 2.8, which bundled Keras 2.8. If you later need to incorporate a new feature introduced in Keras 2.10, you might find that upgrading TensorFlow is overly cumbersome, especially if other parts of your pipeline depend on the older TensorFlow version. Directly importing Keras would allow you to upgrade only Keras, mitigating this dependency issue.

To illustrate, here's a scenario demonstrating the flexibility gained by importing Keras independently:

```python
# Example 1: Using standalone Keras
import keras
from keras.layers import Dense
from keras.models import Sequential

# Define a model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(100,)))
model.add(Dense(1, activation='sigmoid'))
# Model is defined using standalone Keras, independent of TensorFlow

# Check the backend. It may or may not be TensorFlow
print(f"Keras backend: {keras.backend.backend()}")


# Dummy data for demonstration
import numpy as np
X = np.random.rand(100, 100)
y = np.random.randint(0, 2, 100)


# Compile and fit the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Evaluate the model
_, accuracy = model.evaluate(X,y, verbose = 0)
print(f"Model accuracy: {accuracy}")

# The model is now using whatever Keras is configured to use as a backend.
```

In the above example, we import `keras` directly. This provides the advantage of defining a model architecture without an explicit dependency on a particular version of TensorFlow. This approach, after configuring the backend, can use TensorFlow, JAX, or other compatible environments.

Now, consider the alternative, using `tf.keras`:

```python
# Example 2: Using tf.keras

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define a model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(100,)))
model.add(Dense(1, activation='sigmoid'))

# The model is defined using tf.keras and bound to TensorFlow

print(f"Keras backend: {tf.keras.backend.backend()}")

# Dummy data for demonstration
import numpy as np
X = np.random.rand(100, 100)
y = np.random.randint(0, 2, 100)


# Compile and fit the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Evaluate the model
_, accuracy = model.evaluate(X,y, verbose = 0)
print(f"Model accuracy: {accuracy}")

# This model is explicitly using TensorFlow as its backend.
```

In this case, the Keras API is accessed through `tf.keras`, inherently coupling the model definition to the TensorFlow version. If you are using TensorFlow, this may be fine, but you are losing the decoupling and flexibility benefits we have discussed earlier. This becomes a hurdle when working with heterogeneous environments or when trying to use a standalone Keras feature not yet available in the `tf.keras` distribution. While the core functionality might seem identical, the dependency is the critical difference. The `tf.keras` implementation is effectively a submodule of TensorFlow, whereas the standalone Keras is an independent entity, which promotes modularity.

As a more advanced example, consider code that takes advantage of Keras Callbacks. The flexibility remains when using standalone keras, even when you want to use specific TensorFlow functionality:

```python
# Example 3: Standalone Keras, using TensorFlow callbacks

import keras
from keras.layers import Dense
from keras.models import Sequential

import tensorflow as tf

# Define a model using standalone Keras

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(100,)))
model.add(Dense(1, activation='sigmoid'))


# Create a custom tensorflow callback
class CustomTensorflowCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(f'Custom TensorFlow callback called at epoch: {epoch}')


# Create an instance of the custom callback and use it
custom_callback = CustomTensorflowCallback()

# Dummy data for demonstration
import numpy as np
X = np.random.rand(100, 100)
y = np.random.randint(0, 2, 100)


# Compile and fit the model, using the custom callback

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[custom_callback])

# Evaluate the model
_, accuracy = model.evaluate(X,y, verbose = 0)
print(f"Model accuracy: {accuracy}")

# The model is again using a standalone Keras, with a custom Tensorflow callback
# demonstrating the freedom to include functionalities from both libraries.

```

In this third example, the flexibility of the decoupled Keras library is preserved, while still allowing the user to utilize components from the TensorFlow library, namely the callback API. We can use the power of TensorFlow when required, without creating hard dependencies that can limit development.

The distinction between importing `keras` and `tf.keras` lies primarily in the degree of dependency and control over the Keras library. While `tf.keras` provides a convenient shortcut, it comes at the cost of flexibility, which can become a significant impediment during large-scale development and deployment. Direct `keras` import, conversely, empowers you to manage dependencies, choose the desired backend, and utilize the latest Keras features without being constrained by TensorFlow's release cycle.

To deepen your understanding of these libraries, I suggest exploring the official Keras documentation, which offers extensive guides and examples on model construction and backend configurations. Additionally, reviewing articles and tutorials discussing best practices in deep learning framework management will further highlight the advantages of maintaining independent and modular packages. Finally, examining the source code of both Keras and TensorFlow on their respective repositories can provide granular insights into the differences between the standalone Keras API and its `tf.keras` counterpart.
