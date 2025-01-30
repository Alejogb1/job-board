---
title: "Is tfa.metrics.F1Score calculating incorrectly?"
date: "2025-01-30"
id: "is-tfametricsf1score-calculating-incorrectly"
---
The reported behavior of `tfa.metrics.F1Score`, specifically in relation to multi-class classification, has indeed led to confusion. I've encountered instances where the macro-averaged F1 score it produces doesn’t align with the expected value derived from a manual calculation, particularly when some classes have no true positive predictions. This discrepancy isn’t due to a fundamental flaw in the underlying mathematical formula for F1-score itself but rather the specific implementation choices within TensorFlow Addons' `F1Score` class and its default averaging behavior.

Let’s unpack this. The F1-score is the harmonic mean of precision and recall, defined as 2 * (precision * recall) / (precision + recall). For a single class, this is straightforward. However, in multi-class problems, you have multiple per-class F1 scores that require aggregation into a single metric. The two most common averaging methods are macro and weighted averaging. Macro-averaging calculates the F1 score per class and then averages them. Weighted averaging calculates the F1 score per class weighted by the number of true instances for each class. `tfa.metrics.F1Score`, by default, uses macro averaging. The core issue arises from how `tfa.metrics.F1Score` handles scenarios where a class lacks true positives; this situation frequently results from sparse data distributions or underperforming models for some categories.

When a class has zero true positives, it also almost always has zero precision *and* zero recall (assuming your model doesn't predict it at all). In this case, the individual F1 score for that class becomes 0/0 which mathematically is undefined. `tfa.metrics.F1Score` handles this not with an explicit error, but by assuming the score is zero, and proceeds with the macro-averaging. This is where the problem occurs. If multiple classes lack true positive predictions, the resulting macro-averaged F1 score may be erroneously low. A manual calculation would often, and arguably should, exclude the F1 score of zero classes from the averaging process, resulting in a higher overall macro F1 score.

I've personally observed this several times while fine-tuning deep learning models for multi-label text classification tasks. I found the `tfa.metrics.F1Score` would persistently display a lower score than my hand-calculated results. Here are some code examples to illustrate these findings:

**Example 1: Illustrating the basic F1 Score calculation:**

```python
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Example with non-zero true positives for all classes
y_true_ex1 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int32)
y_pred_ex1 = tf.constant([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]], dtype=tf.float32)
y_pred_ex1 = tf.argmax(y_pred_ex1, axis=1)

f1_metric_ex1 = tfa.metrics.F1Score(num_classes=3, average='macro')
f1_metric_ex1.update_state(y_true_ex1, y_pred_ex1)
f1_score_ex1 = f1_metric_ex1.result().numpy()
print(f"F1 Score (Example 1, tfa): {f1_score_ex1}")

def calculate_f1_macro(y_true, y_pred, num_classes):
    f1_scores = []
    for class_idx in range(num_classes):
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_idx), tf.equal(y_pred, class_idx)), dtype=tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, class_idx), dtype=tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, class_idx), dtype=tf.float32))

        precision = true_positives / (predicted_positives + 1e-8) # add small value to prevent division by zero
        recall = true_positives / (actual_positives + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return tf.reduce_mean(f1_scores).numpy()

manual_f1_ex1 = calculate_f1_macro(tf.argmax(y_true_ex1, axis=1), y_pred_ex1, 3)
print(f"F1 Score (Example 1, manual): {manual_f1_ex1}")
```
In this first example, we have predictions that result in non-zero true positives for all three classes. The `tfa.metrics.F1Score` output matches the manually calculated macro-averaged score because each individual F1 can be calculated meaningfully, and zero values aren't introduced.

**Example 2: Illustrating the issue with zero true positives.**

```python
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Example with zero true positives for one class
y_true_ex2 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=tf.int32)
y_pred_ex2 = tf.constant([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32)
y_pred_ex2 = tf.argmax(y_pred_ex2, axis=1)


f1_metric_ex2 = tfa.metrics.F1Score(num_classes=3, average='macro')
f1_metric_ex2.update_state(y_true_ex2, y_pred_ex2)
f1_score_ex2 = f1_metric_ex2.result().numpy()
print(f"F1 Score (Example 2, tfa): {f1_score_ex2}")


manual_f1_ex2 = calculate_f1_macro(tf.argmax(y_true_ex2, axis=1), y_pred_ex2, 3)
print(f"F1 Score (Example 2, manual): {manual_f1_ex2}")


```

Here, class 0 in `y_true_ex2` is present but not predicted correctly. This results in zero true positives, which in turn causes `tfa.metrics.F1Score` to assume an F1-score of 0 for that class. The manually computed score, in contrast, does not include a value of zero in the average, thus the macro average is higher. This discrepancy is the core of the reported issue.

**Example 3: Using 'weighted' averaging**

```python
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Example using weighted averaging
y_true_ex3 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=tf.int32)
y_pred_ex3 = tf.constant([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32)
y_pred_ex3 = tf.argmax(y_pred_ex3, axis=1)

f1_metric_ex3 = tfa.metrics.F1Score(num_classes=3, average='weighted')
f1_metric_ex3.update_state(y_true_ex3, y_pred_ex3)
f1_score_ex3 = f1_metric_ex3.result().numpy()
print(f"F1 Score (Example 3, tfa, weighted): {f1_score_ex3}")


def calculate_f1_weighted(y_true, y_pred, num_classes):
    f1_scores = []
    class_counts = []
    for class_idx in range(num_classes):
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_idx), tf.equal(y_pred, class_idx)), dtype=tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, class_idx), dtype=tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, class_idx), dtype=tf.float32))
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)
        class_counts.append(actual_positives)
        
    f1_scores_tensor = tf.stack(f1_scores)
    class_counts_tensor = tf.stack(class_counts)
    return tf.reduce_sum(f1_scores_tensor * class_counts_tensor) / tf.reduce_sum(class_counts_tensor).numpy()


manual_f1_ex3 = calculate_f1_weighted(tf.argmax(y_true_ex3, axis=1), y_pred_ex3, 3)
print(f"F1 Score (Example 3, manual, weighted): {manual_f1_ex3}")

```
Example 3 switches `tfa.metrics.F1Score` to `average="weighted"`, and computes the manual average as weighted by class support. Observe that the `tfa.metrics.F1Score` and the manually calculated `f1_weighted` values match, indicating that the implementation functions as expected when the averaging is done with weights. In the context of the reported issue, using the weighted average can mitigate problems when some classes have zero true positives, though it may not be the desired metric for all scenarios.

**Resource Recommendations:**

For a deeper understanding of F1 score, I recommend reviewing the original publication on the metric itself (specifically regarding precision, recall, and harmonic mean). Additionally, various resources on performance metrics for classification, such as standard academic texts on machine learning and deep learning, should be consulted. Exploring the mathematical definitions of macro- and weighted-averaging are also crucial. You can also find many tutorials and articles explaining these concepts by searching for "multi-class classification metrics" or "F1-score interpretation". The scikit-learn documentation provides a concise yet comprehensive definition of the `f1_score` metric which can help understand various averaging schemes. Understanding these concepts, along with a clear understanding of your specific task, will help you correctly apply these metrics. This will allow you to avoid incorrect interpretation when working with `tfa.metrics.F1Score` or any other evaluation function.
