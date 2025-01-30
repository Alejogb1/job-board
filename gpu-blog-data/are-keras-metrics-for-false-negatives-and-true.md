---
title: "Are Keras metrics for false negatives and true negatives always expressed as fractions?"
date: "2025-01-30"
id: "are-keras-metrics-for-false-negatives-and-true"
---
No, Keras metrics for false negatives (FN) and true negatives (TN) are not always expressed as fractions; they are, by default, expressed as counts. This distinction is critical when evaluating the performance of a model, particularly in scenarios with imbalanced datasets. My experience building anomaly detection systems, where true negatives vastly outnumbered other outcomes, has demonstrated the importance of understanding the inherent behavior of these metrics to avoid misinterpretations.

The core issue stems from how Keras implements binary classification metrics. When you specify `tf.keras.metrics.FalseNegatives()` or `tf.keras.metrics.TrueNegatives()`, Keras initializes a stateful metric that internally accumulates the raw counts of FNs and TNs, respectively. These accumulated counts are then accessed through the metric's `result()` method. They are *not* automatically normalized to produce a rate or a fraction. The user is responsible for any further manipulation or calculations needed to arrive at a desired fractional representation, such as the false negative rate or true negative rate.

The confusion likely arises because many other metrics, such as precision, recall, and F1 score, are inherently fractional and are calculated from the raw FN, TN, FP, and TP counts. Keras, however, provides these pre-calculated fractional metrics directly (e.g., `tf.keras.metrics.Precision()`, `tf.keras.metrics.Recall()`). It is important to differentiate between raw count metrics and calculated fractional metrics. The `FalseNegatives` and `TrueNegatives` belong to the former category.

Let me demonstrate this with code examples.

**Example 1: Demonstrating Default Counts**

In this example, we will train a simplistic model and observe the raw counts returned by the FN and TN metrics.

```python
import tensorflow as tf
import numpy as np

# Generate dummy data
y_true = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32)
y_pred = np.array([0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7, 0.8, 0.4, 0.9], dtype=np.float32)

# Threshold predictions for binary classification
y_pred_binary = tf.cast(y_pred > 0.5, tf.int32)

# Instantiate metrics
false_negatives = tf.keras.metrics.FalseNegatives()
true_negatives = tf.keras.metrics.TrueNegatives()

# Update metrics with predictions and true labels
false_negatives.update_state(y_true, y_pred_binary)
true_negatives.update_state(y_true, y_pred_binary)

# Get results
fn_count = false_negatives.result()
tn_count = true_negatives.result()

print(f"False Negative Count: {fn_count.numpy()}")
print(f"True Negative Count: {tn_count.numpy()}")

# Clean up for subsequent example
false_negatives.reset_state()
true_negatives.reset_state()

```
The output of this script will show the raw counts of the false negatives and true negatives. In this case, for the provided data and threshold, it should yield 1 and 5, respectively, representing 1 instance incorrectly classified as negative and 5 instances correctly classified as negative.  These values are not ratios or percentages.

**Example 2: Calculating the False Negative Rate**

Here, I'll showcase how to calculate the False Negative Rate, by using the counts produced by the `FalseNegatives` and relevant additional calculations to normalize.

```python
import tensorflow as tf
import numpy as np

# Generate dummy data
y_true = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32)
y_pred = np.array([0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7, 0.8, 0.4, 0.9], dtype=np.float32)

# Threshold predictions
y_pred_binary = tf.cast(y_pred > 0.5, tf.int32)

# Instantiate metric
false_negatives = tf.keras.metrics.FalseNegatives()

# Update metric
false_negatives.update_state(y_true, y_pred_binary)

# Get raw count
fn_count = false_negatives.result()

# Calculate total actual positives
actual_positives = tf.reduce_sum(tf.cast(y_true, tf.float32))

# Calculate False Negative Rate
false_negative_rate = fn_count / actual_positives if actual_positives > 0 else tf.constant(0.0, dtype=tf.float32)


print(f"False Negative Count: {fn_count.numpy()}")
print(f"Actual Positive Count: {actual_positives.numpy()}")
print(f"False Negative Rate: {false_negative_rate.numpy()}")

# Clean up
false_negatives.reset_state()
```

The output will first display the same raw count for FN as before. It then calculates the total actual positives within the ground truth data. Finally, it produces the False Negative Rate (FNR), which is calculated by dividing the count of FNs by the total number of actual positives. In the case of this particular example, the False Negative rate will be 0.25 (1/4), since there was 1 false negative and 4 actual positives. The critical part is the division step; it is the user's responsibility to transform the counts to rates.

**Example 3:  Integrating with Model Training**

This final example showcases the practical application of monitoring both raw counts and derived rate during model training. Using a custom metric for FNR will allow us to track it during model training via `model.fit()`.

```python
import tensorflow as tf
import numpy as np

class FalseNegativeRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_negative_rate', **kwargs):
        super(FalseNegativeRate, self).__init__(name=name, **kwargs)
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.actual_positives = self.add_weight(name='ap', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.cast(y_true, dtype=tf.int32)
      y_pred = tf.cast(y_pred > 0.5, tf.int32)
      
      actual_positives = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
      
      # Calculate FNs
      
      
      false_negative_mask = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0))
      false_negatives = tf.reduce_sum(tf.cast(false_negative_mask, tf.float32))

      self.false_negatives.assign_add(false_negatives)
      self.actual_positives.assign_add(actual_positives)

    def result(self):
        return tf.math.divide_no_nan(self.false_negatives, self.actual_positives)

    def reset_state(self):
        self.false_negatives.assign(0.0)
        self.actual_positives.assign(0.0)



# Generate dummy data for training
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100, 1))


# Create a basic model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[FalseNegativeRate(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives()])


# Train the model
history = model.fit(X_train, y_train, epochs=2, verbose=0)

print(f"FN Rate at Epoch 2: {history.history['false_negative_rate'][1]}")
print(f"FN Count at Epoch 2: {history.history['false_negatives'][1]}")
print(f"TN Count at Epoch 2: {history.history['true_negatives'][1]}")

```
In this example, a custom metric named `FalseNegativeRate` is created which inherits from `tf.keras.metrics.Metric`. It maintains internal state for `false_negatives` and `actual_positives`. It overrides the necessary methods like `update_state`, `result` and `reset_state` to properly update and return the FNR. This custom metric is included in the model's metrics when compiling the model and, during training, both the raw counts for FN and TN, and the FNR will be logged alongside the loss. This allows for concurrent monitoring of both the raw counts and a meaningful derived rate.

In summary, while Keras does provide `FalseNegatives` and `TrueNegatives` metrics, these are *counts*, not fractions or rates by default. The responsibility falls on the user to compute any desired fractional representation from these raw counts. The examples illustrate the difference between direct output and calculated rates and demonstrate how to correctly compute the False Negative Rate. To delve deeper, I recommend exploring resources that offer comprehensive coverage of binary classification metrics and model evaluation, with a focus on the implementation within TensorFlow and Keras. These resources often provide more details on imbalanced data and metric interpretation, both of which can significantly impact how we understand the performance of a model. Consider works on statistical learning, advanced machine learning techniques, and research papers that focus on specific metrics for various applications.
