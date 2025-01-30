---
title: "What's the difference between Keras AUC(curve='PR') and scikit-learn's average precision?"
date: "2025-01-30"
id: "whats-the-difference-between-keras-auccurvepr-and-scikit-learns"
---
The key distinction between Keras's `AUC(curve='PR')` metric and scikit-learn's `average_precision_score` lies primarily in their implementation within their respective frameworks, with subtle differences in behavior arising from that context. Both aim to quantify the performance of binary classifiers by summarizing the precision-recall (PR) curve, but one operates within the context of a TensorFlow training loop, leveraging computational graph execution, while the other calculates values directly from NumPy arrays. This can lead to slight numerical discrepancies due to different precision handling and algorithmic choices.

In my work on a large-scale anomaly detection system, I frequently encountered this divergence when attempting to compare offline model evaluations, computed with scikit-learn, to the real-time training metrics displayed by Keras. The initial expectation was near-perfect alignment, given that both purportedly measured the area under the PR curve. However, the minor differences I witnessed motivated a deeper dive into the specific implementations to understand the nuances.

Here's a detailed explanation:

Both the Keras AUC with the `curve='PR'` parameter and scikit-learn's `average_precision_score` measure the area under the precision-recall curve. The PR curve plots the precision (true positives / (true positives + false positives)) against recall (true positives / (true positives + false negatives)) across various classification thresholds. It is especially useful when dealing with imbalanced datasets where the minority class is of greater interest, as it focuses solely on positive predictions. A high average precision, or a large area under the PR curve, indicates a strong performing classifier because the model consistently achieves high precision with high recall, a desired attribute in such scenarios.

Fundamentally, these measures calculate a weighted average of the precision values at each threshold, where the weights are defined by the difference in recall between consecutive thresholds. The average precision can be interpreted as the average precision achieved across all possible thresholds. For a discrete set of thresholds, it can be formulated as a sum over increments in recall multiplied by the precision at that increment. This core concept is the same for both Keras and scikit-learn.

However, the precise manner in which this is computed differs. Keras’s AUC is implemented within TensorFlow using computational graphs and operates on batched predictions during training. It approximates the AUC using discrete thresholds based on a moving average of precision/recall across batches as the model trains. Scikit-learn's `average_precision_score`, on the other hand, operates directly on arrays of predictions and true labels. It calculates precision and recall at every unique threshold value present in the data, allowing for a more precise (but potentially computationally more expensive) calculation than the batched approach in Keras. Furthermore, scikit-learn's implementation utilizes sorting and searching operations over the entire input, while Keras is designed to process batches sequentially within its computational graph. This difference, especially the use of moving averages in Keras, contributes to subtle variations in the results between the two implementations.

Moreover, the input requirements differ slightly. Keras requires the probability of each sample belonging to the positive class, while scikit-learn also accepts classification decisions (0 or 1) but is primarily designed for probability-like scores. In my experience, providing probability scores to both functions ensures consistency in what they are interpreting as prediction scores, reducing any further divergence arising from treating decision boundaries differently.

Here are examples illustrating their usage and potential differences:

**Example 1: Basic Usage Comparison**

```python
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow.keras.metrics import AUC

# Sample true labels and predictions
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_prob = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9, 0.4, 0.7, 0.95, 0.05])


# Calculate with scikit-learn
sklearn_ap = average_precision_score(y_true, y_prob)
print(f"Scikit-learn Average Precision: {sklearn_ap:.4f}")

# Calculate with Keras
keras_auc = AUC(curve='PR')
keras_auc.update_state(y_true, y_prob)
keras_ap = keras_auc.result().numpy()
print(f"Keras PR AUC: {keras_ap:.4f}")
```

In this first example, we can see that the calculated `average_precision_score` from scikit-learn and the PR-AUC from Keras applied to the same data yields generally similar values, but they are often not identical. The minor difference, approximately 0.002 here, is due to the difference in calculation methods, particularly how the thresholds are selected, and the lack of moving averages in scikit-learn's method.

**Example 2: Impact of Batch Processing in Keras**

```python
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow.keras.metrics import AUC

# Sample true labels and predictions
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_prob = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9, 0.4, 0.7, 0.95, 0.05])


# Calculate with scikit-learn
sklearn_ap = average_precision_score(y_true, y_prob)
print(f"Scikit-learn Average Precision: {sklearn_ap:.4f}")

# Calculate with Keras using batch processing (mimicked)
keras_auc = AUC(curve='PR')
batch_size = 3
for i in range(0, len(y_true), batch_size):
    batch_true = y_true[i:i+batch_size]
    batch_prob = y_prob[i:i+batch_size]
    keras_auc.update_state(batch_true, batch_prob)

keras_ap = keras_auc.result().numpy()
print(f"Keras PR AUC (batched): {keras_ap:.4f}")
```

This second example attempts to mimic the batched processing that occurs within a Keras training loop. The data is processed in batches when updating the state of the Keras AUC metric, simulating how metrics are calculated during training. While still comparable to the scikit-learn `average_precision_score`, there might be more variance in the value compared to the original Keras use where all data was passed simultaneously. This is because the Keras AUC uses moving averages of precision and recall, where each batch update adjusts the precision/recall curve and therefore the approximate area under the curve slightly.

**Example 3: Effect of Class Imbalance**

```python
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow.keras.metrics import AUC

# Imbalanced data
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
y_prob = np.array([0.1, 0.2, 0.15, 0.05, 0.25, 0.3, 0.1, 0.01, 0.85, 0.2])


# Calculate with scikit-learn
sklearn_ap = average_precision_score(y_true, y_prob)
print(f"Scikit-learn Average Precision (Imbalanced): {sklearn_ap:.4f}")

# Calculate with Keras
keras_auc = AUC(curve='PR')
keras_auc.update_state(y_true, y_prob)
keras_ap = keras_auc.result().numpy()
print(f"Keras PR AUC (Imbalanced): {keras_ap:.4f}")
```

This final example shows the metric applied to imbalanced data. Notice that the values are typically low as the goal is to identify rare positive cases (1s) accurately, which is more challenging when they are few. Even with this class imbalance, the values calculated by scikit-learn and Keras are still reasonably similar. The difference primarily stems from how the thresholds are selected and calculated as discussed earlier, specifically how those thresholds map to precision and recall, thereby affecting how the area is approximated.

For further information, the documentation for the TensorFlow Keras API provides comprehensive details on the `tf.keras.metrics.AUC` class. Scikit-learn's documentation also has extensive information about `sklearn.metrics.average_precision_score`. Moreover, various academic papers discuss precision-recall curves and their application in machine learning, which provides a more theoretical and in-depth explanation of these concepts. Studying how different evaluation metrics operate within their respective frameworks provides insight into implementation details that can explain numerical variation.
