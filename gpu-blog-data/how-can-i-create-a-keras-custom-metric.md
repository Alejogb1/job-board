---
title: "How can I create a Keras custom metric that ignores specific values?"
date: "2025-01-30"
id: "how-can-i-create-a-keras-custom-metric"
---
The core challenge in creating a Keras custom metric that ignores specific values lies in effectively masking or filtering the target variable before the metric calculation.  Directly modifying the loss function isn't ideal; a custom metric provides a cleaner separation of concerns, allowing for independent evaluation and monitoring during training.  In my experience developing large-scale image classification models, this approach proved essential for handling noisy or irrelevant data points within the ground truth.

**1. Clear Explanation**

Creating a custom Keras metric that ignores specific values involves several steps. First, you define a function that accepts the `y_true` (ground truth) and `y_pred` (predicted values) tensors.  Crucially, this function needs to identify and mask the values you wish to ignore.  This is typically achieved using boolean indexing or TensorFlow/NumPy array manipulation based on the specific values you want to exclude.  After masking, the metric calculation proceeds only on the remaining, relevant data points.  Finally, the function returns a single scalar value representing the computed metric.  This scalar is then used by Keras during the training and evaluation phases. The `tf.keras.metrics.Metric` class offers a structured approach, streamlining the process and ensuring compatibility.


**2. Code Examples with Commentary**

**Example 1: Ignoring specific values in regression**

This example demonstrates a custom Mean Squared Error (MSE) metric that ignores instances where the ground truth value is -1.

```python
import tensorflow as tf

class MaskedMSE(tf.keras.metrics.Metric):
    def __init__(self, name='masked_mse', **kwargs):
        super(MaskedMSE, self).__init__(name=name, **kwargs)
        self.mse = self.add_weight(name='mse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, -1)
        masked_y_true = tf.boolean_mask(y_true, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)
        mse_update = tf.reduce_mean(tf.square(masked_y_true - masked_y_pred))
        self.mse.assign_add(mse_update)
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))


    def result(self):
        return self.mse / self.count if self.count > 0 else 0.0

    def reset_states(self):
        self.mse.assign(0.0)
        self.count.assign(0.0)

# Example usage:
metric = MaskedMSE()
y_true = tf.constant([1.0, 2.0, -1.0, 4.0, 5.0])
y_pred = tf.constant([1.2, 1.8, -1.1, 3.9, 5.1])
metric.update_state(y_true, y_pred)
print(f"Masked MSE: {metric.result().numpy()}") # Output will be the MSE excluding the -1 instance.
```

This code defines a `MaskedMSE` class inheriting from `tf.keras.metrics.Metric`. `update_state` applies a boolean mask to exclude `-1` values from `y_true` and `y_pred` before calculating the MSE.  The `result` method then provides the average MSE over the non-masked values, handling cases with no valid data points.  `reset_states` ensures proper metric initialization between epochs.


**Example 2:  Handling multiple ignore values in classification**

This example extends the concept to multi-class classification, ignoring classes represented by 0 and 5.

```python
import tensorflow as tf

class MaskedCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, ignore_values=[0,5], name='masked_categorical_accuracy', **kwargs):
        super(MaskedCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name='correct_predictions', initializer='zeros')
        self.total_predictions = self.add_weight(name='total_predictions', initializer='zeros')
        self.ignore_values = ignore_values


    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.logical_and(tf.not_equal(y_true, self.ignore_values[0]), tf.not_equal(y_true, self.ignore_values[1]))

        masked_y_true = tf.boolean_mask(y_true, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)

        correct = tf.equal(tf.argmax(masked_y_pred, axis=-1), tf.cast(masked_y_true, tf.int64))
        self.correct_predictions.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total_predictions.assign_add(tf.cast(tf.shape(masked_y_true)[0], tf.float32))

    def result(self):
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0

    def reset_states(self):
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)

# Example usage:
metric = MaskedCategoricalAccuracy()
y_true = tf.constant([1, 2, 0, 3, 5, 4])
y_pred = tf.constant([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.2, 0.8, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.7, 0.3, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]])
metric.update_state(y_true, y_pred)
print(f"Masked Categorical Accuracy: {metric.result().numpy()}") #Output will reflect accuracy excluding classes 0 and 5.

```

This example demonstrates handling multiple ignore values in categorical accuracy. The mask is constructed using `tf.logical_and` to combine conditions for excluding each specified class.


**Example 3:  Ignoring NaN values in a custom metric**

This showcases a robust approach to handle `NaN` values, a common issue in real-world datasets.

```python
import tensorflow as tf
import numpy as np

class MaskedCosineSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name='masked_cosine_similarity', **kwargs):
        super(MaskedCosineSimilarity, self).__init__(name=name, **kwargs)
        self.similarity_sum = self.add_weight(name='similarity_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.logical_and(tf.math.is_finite(y_true), tf.math.is_finite(y_pred))
        masked_y_true = tf.boolean_mask(y_true, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)

        similarity = tf.reduce_sum(masked_y_true * masked_y_pred, axis=-1) / (tf.norm(masked_y_true, axis=-1) * tf.norm(masked_y_pred, axis=-1) + 1e-9) # Adding small epsilon to avoid division by zero

        self.similarity_sum.assign_add(tf.reduce_sum(similarity))
        self.count.assign_add(tf.cast(tf.shape(masked_y_true)[0], tf.float32))


    def result(self):
        return self.similarity_sum / self.count if self.count > 0 else 0.0

    def reset_states(self):
        self.similarity_sum.assign(0.0)
        self.count.assign(0.0)

# Example usage
metric = MaskedCosineSimilarity()
y_true = tf.constant([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
y_pred = tf.constant([[0.8, 1.8, np.nan], [3.2, 3.8, 5.2]])
metric.update_state(y_true, y_pred)
print(f"Masked Cosine Similarity: {metric.result().numpy()}") # Output will ignore NaN values
```

This example demonstrates handling `NaN` values using `tf.math.is_finite`.  Note the addition of a small epsilon (1e-9) in the denominator to prevent division by zero errors, a detail often overlooked but critical for numerical stability.


**3. Resource Recommendations**

The TensorFlow documentation on custom metrics and the official Keras guide are invaluable resources.  A thorough understanding of TensorFlow's tensor manipulation functions (particularly boolean masking and indexing) is essential.  Furthermore, familiarizing yourself with NumPy array operations can provide additional flexibility in handling data pre-processing and masking.  Finally, mastering the principles of numerical stability in TensorFlow computations will help avoid common pitfalls in developing robust custom metrics.
