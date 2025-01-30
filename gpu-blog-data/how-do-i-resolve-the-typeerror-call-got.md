---
title: "How do I resolve the 'TypeError: call() got an unexpected keyword argument 'is_training'' error when migrating TensorFlow code from version 1.1.0 to 2.2.0?"
date: "2025-01-30"
id: "how-do-i-resolve-the-typeerror-call-got"
---
The `TypeError: call() got an unexpected keyword argument 'is_training'` encountered during TensorFlow 1.1.0 to 2.2.0 migration stems from a fundamental shift in the framework's approach to training and inference.  TensorFlow 1.x relied heavily on `tf.placeholder` and explicit session management, often using boolean flags like `is_training` to control behavior within custom layers or models.  TensorFlow 2.x, however, embraces eager execution and the `tf.keras` API, rendering such explicit flags largely obsolete.  My experience in migrating large-scale image recognition models highlighted this transition as a common source of errors.

The core issue is that TensorFlow 2.x's `tf.keras.layers.Layer` class, unlike its 1.x counterpart, doesn't inherently expect an `is_training` argument within its `call()` method.  The training/inference distinction is now primarily handled through the `tf.function` decorator, leveraging automatic control dependencies and the `tf.GradientTape` context manager.  Therefore, the solution involves refactoring code to leverage TensorFlow 2.x's built-in mechanisms for managing training and inference phases.

**1. Explanation:**

The `is_training` flag in TensorFlow 1.x was typically used to conditionally execute different operations depending on whether the model was in training or inference mode.  Common scenarios included:

* **Batch Normalization:**  Parameters were updated during training but held constant during inference.
* **Dropout:** Dropout layers were active during training but inactive during inference.
* **Conditional computations:** Specific parts of a network could be bypassed during inference to optimize performance.

In TensorFlow 2.x, these behaviors are mostly handled automatically.  Batch normalization layers automatically switch to inference mode when `training=False` is implicitly passed during inference. Dropout layers similarly behave accordingly.  Conditional operations are best managed using `tf.cond` or the equivalent functionality within `tf.function`s.


**2. Code Examples with Commentary:**

**Example 1:  Refactoring a Custom Layer with Batch Normalization:**

```python
# TensorFlow 1.x code (problematic)
import tensorflow as tf1

class MyLayer(tf1.keras.layers.Layer):
  def __init__(self):
    super(MyLayer, self).__init__()
    self.bn = tf1.keras.layers.BatchNormalization()

  def call(self, inputs, is_training):
    x = self.bn(inputs, training=is_training)
    return x

# TensorFlow 2.x refactored code
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyLayer, self).__init__()
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=None): #training argument is now optional
    x = self.bn(inputs, training=training)
    return x

# Usage: the training flag is handled automatically by tf.keras.Model.fit()
model = tf.keras.Sequential([MyLayer()])
model.compile(...)
model.fit(...)
```

This example demonstrates how the explicit `is_training` argument is removed. TensorFlow 2.x automatically manages the training state for the batch normalization layer based on the model's execution context.


**Example 2: Managing Dropout with `tf.function`:**

```python
# TensorFlow 1.x code (problematic)
import tensorflow as tf1

def my_model(inputs, is_training):
  x = tf1.keras.layers.Dropout(0.5)(inputs, training=is_training)
  return x

# TensorFlow 2.x refactored code
import tensorflow as tf

@tf.function
def my_model(inputs, training=True): # training defaults to True
  x = tf.keras.layers.Dropout(0.5)(inputs, training=training)
  return x

# Usage:  the training state is determined by how the function is called.
inference_output = my_model(inputs, training=False) # Inference mode
training_output = my_model(inputs) # Training mode
```

Here, the `tf.function` decorator allows for automatic handling of training and inference states.  The `training` argument provides explicit control when needed.

**Example 3: Conditional Computation using `tf.cond`:**

```python
# TensorFlow 1.x code (problematic)
import tensorflow as tf1

def my_model(inputs, is_training):
  if is_training:
    x = tf1.layers.dense(inputs, 64)
    x = tf1.layers.dense(x, 10)
  else:
    x = tf1.layers.dense(inputs, 10) # Simpler inference path
  return x


# TensorFlow 2.x refactored code
import tensorflow as tf

@tf.function
def my_model(inputs, training=True):
  x = tf.cond(training,
               lambda: tf.keras.Sequential([tf.keras.layers.Dense(64), tf.keras.layers.Dense(10)])(inputs),
               lambda: tf.keras.layers.Dense(10)(inputs))
  return x
```

This example shows how conditional logic is restructured using `tf.cond` within a `tf.function`, avoiding direct reliance on the `is_training` flag. The `tf.cond` function provides conditional execution based on the `training` variable.

**3. Resource Recommendations:**

The official TensorFlow 2.x migration guide.  The TensorFlow API documentation, specifically focusing on `tf.keras.layers.Layer`, `tf.function`, `tf.GradientTape`, and `tf.cond`.  A well-structured tutorial on building custom layers and models in TensorFlow 2.x.  Examining example code repositories featuring TensorFlow 2.x projects.  Reviewing documentation on the changes in batch normalization and dropout layers between TensorFlow 1.x and 2.x.


By carefully addressing the underlying changes in how TensorFlow manages training and inference, and by employing the recommended techniques,  the `TypeError: call() got an unexpected keyword argument 'is_training'` can be effectively resolved, ensuring a smooth and successful migration of existing TensorFlow 1.x code to the more modern and efficient TensorFlow 2.x framework.  My personal experience demonstrates that a thorough understanding of these concepts is crucial for avoiding similar issues during large-scale model migrations.
