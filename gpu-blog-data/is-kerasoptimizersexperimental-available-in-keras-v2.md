---
title: "Is `keras.optimizers.experimental` available in Keras v2?"
date: "2025-01-30"
id: "is-kerasoptimizersexperimental-available-in-keras-v2"
---
The `keras.optimizers.experimental` module was introduced with the Keras 3 API, and therefore it does not exist within the Keras v2 ecosystem. My experience migrating models from TensorFlow 2.x (which utilized Keras v2 as its primary API) to Keras 3 and standalone Keras has highlighted this specific discrepancy. This migration involved numerous adjustments to namespaces, and the optimizer module was a particularly prominent area requiring rework.

Keras v2's optimizers are accessed directly through the `keras.optimizers` module. This module contains implementations of core optimization algorithms like SGD, Adam, and RMSprop, along with various extensions and techniques such as learning rate scheduling and gradient clipping. These are accessible and usable within the Keras v2 API, providing a rich set of tools for training neural networks. The key distinction is that Keras v2 operates within the framework of TensorFlow 2.x, leveraging its backend implementations, whereas Keras 3 is designed to be a multi-backend framework (TensorFlow, JAX, PyTorch). The experimental module within Keras 3 represents a forward-looking approach to optimizer development, allowing for more direct contributions and testing of newer methods without being constrained by the backward compatibility concerns of stable APIs.

The presence of the `experimental` module in Keras 3 signals the introduction of new optimizers and related features that are still undergoing development and testing. These may not have the same level of stability or widespread adoption as the core optimizers, and they are typically subject to potential changes or removals in future updates. This experimental aspect permits a more agile development cycle and encourages community participation in the evolution of these tools. For a Keras v2 project, directly referencing the `experimental` module is an error that results in an `AttributeError`, as the API simply does not expose such a module.

Let's examine code examples that illustrate these differences. First, a Keras v2 scenario, showcasing the standard way to use optimizers:

```python
import tensorflow as tf

# Assuming TensorFlow 2.x where Keras is keras v2
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Assume 'x_train' and 'y_train' are defined elsewhere for training
# model.fit(x_train, y_train, epochs=10)

print(f"Optimizer type: {type(optimizer)}")
```

In the above example, I demonstrate the typical workflow within Keras v2. `tf.keras.optimizers.Adam` is directly imported and initialized. It's then passed to the `model.compile` method. This is the standard way to configure an optimizer in the TensorFlow 2.x/Keras v2 environment. The output, confirming the type of object, will be something similar to `<class 'keras.optimizers.adam.Adam'>`, emphasizing its location within the `keras.optimizers` module. There is no reference to any `experimental` namespace, as this feature was not part of the library’s architecture at that time.

Now, let's explore a hypothetical Keras 3 code example that *would* use the experimental module (if it were in v2, which, as previously discussed, it is not):

```python
import keras
import keras.optimizers.experimental as kex

# Assume Keras 3 (or standalone Keras)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = kex.Lion(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Assume 'x_train' and 'y_train' are defined elsewhere for training
# model.fit(x_train, y_train, epochs=10)

print(f"Optimizer type: {type(optimizer)}")
```

This code would only work within the Keras 3 context, or a Keras environment that specifically includes the `keras.optimizers.experimental` module, showcasing the namespace difference. Attempting this code within a Keras v2 environment would lead to an `AttributeError` upon trying to import or use `keras.optimizers.experimental`. The `Lion` optimizer within this namespace is an example of what Keras 3 is making more accessible, but which was not available in earlier versions. The type output here would display as something like `<class 'keras.optimizers.experimental.lion.Lion'>`.

Finally, if I were to attempt using `keras.optimizers.experimental` within Keras v2 (i.e., TensorFlow 2.x), I would encounter an error. The following snippet shows such an attempt and the error that would arise:

```python
import tensorflow as tf

try:
  import tensorflow.keras.optimizers.experimental as kex  # This will fail in v2
  print("Success!") #This should not print
except AttributeError as e:
  print(f"Error: {e}")
```
Running this code in a Keras v2/TensorFlow 2.x environment will not result in "Success!" being printed, but will instead produce the `AttributeError` message. The traceback will explicitly show that the `experimental` module is not present within the `tf.keras.optimizers` structure. This further emphasizes that the experimental module and its associated optimizers are exclusive to Keras 3 and later versions, or other environments that have explicitly made them accessible. It's imperative to check the specific Keras version being used to know if a module is present.

To further educate oneself on the differences between the versions of Keras, I recommend consulting the official Keras documentation for the API differences, specifically focusing on the changes introduced in Keras 3. The TensorFlow documentation also provides details on integrating Keras, but with the caveat that this generally applies to the Keras v2 implementation integrated within TensorFlow 2.x. Furthermore, exploring blog posts and technical articles that have discussed Keras 2 to Keras 3 migrations can offer valuable real-world experiences and practical guidance. Examining source code directly is a powerful method of learning; I have often looked at the specific implementation of the optimizers within the Keras repositories (either on GitHub or wherever the projects are hosted) to understand more deeply the architectural differences. For example, examining commit history relating to changes in the optimizer sub-packages can provide insight on when certain namespaces and structures were introduced. These sources, in combination, provide a robust learning path regarding API evolution.
