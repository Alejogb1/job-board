---
title: "How does torchmetrics behave with one-hot encoded values?"
date: "2025-01-30"
id: "how-does-torchmetrics-behave-with-one-hot-encoded-values"
---
One crucial aspect of leveraging TorchMetrics effectively, particularly within classification tasks, lies in understanding its handling of one-hot encoded inputs. Many metrics within this library are designed to work seamlessly with either integer labels or probability distributions outputted directly from a model. However, scenarios often arise where the true labels are represented as one-hot vectors, necessitating a clear understanding of the internal mechanisms at play to avoid unexpected behavior. Based on my experience fine-tuning various deep learning models, failing to account for this nuanced interaction can lead to inaccurate performance assessments during training.

The core functionality within TorchMetrics concerning one-hot encoded inputs revolves around its ability to identify and handle these vectors by converting them to a compatible representation, typically via an `argmax` operation on the one-hot vector. When metric updates are performed, TorchMetrics generally expects integer-based label inputs that correspond to class indices, especially in multi-class settings. If you provide it with one-hot vectors, the framework recognizes this input structure. Instead of calculating incorrect scores, it interprets the provided one-hot representation to select the index of the maximum value. This effectively mimics a conversion from a one-hot encoded vector to the associated class integer index that represents the actual ground truth for comparison with model outputs. This conversion is essential for metrics designed to operate on class indices, like `Accuracy`, `Precision`, `Recall`, and `F1Score`. This inherent adaptation simplifies the process of metric computation, eliminating the need for explicit, manual conversions by the user before metric updates.

Let's delve into some practical examples to illustrate this behavior more clearly. Consider a situation where a model outputs a probability vector and the ground truth is represented as a one-hot encoded vector.

**Example 1: Basic Accuracy Calculation**

Here's an example where we compute accuracy with one-hot labels. This example highlights the internal behavior.

```python
import torch
import torchmetrics

# Initialize a basic accuracy metric.
accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)

# Model output probabilities for a batch of 2.
probabilities = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])

# Corresponding true labels encoded as one-hot vectors.
one_hot_labels = torch.tensor([[0, 1, 0], [1, 0, 0]])

# Update the metric with the probabilities and one-hot labels.
accuracy_metric.update(probabilities, one_hot_labels)

# Get computed accuracy.
accuracy = accuracy_metric.compute()

print(f"Accuracy: {accuracy}") # Output: Accuracy: 1.0
```

In this example, `probabilities` are model outputs and `one_hot_labels` represents the ground truth. The accuracy metric, despite receiving one-hot encoded vectors, correctly infers the class index (0, 1, 2) corresponding to the true labels, thanks to the implicit conversion using `argmax`. As such, the metric returns an accuracy of 1.0, because the predicted class (argmax of probabilities) matches the correct class (argmax of one-hot). This indicates that internal mechanism in TorchMetrics is successfully handling the one-hot format. Note that `task='multiclass'` is needed if we are inputting probabilities, rather than a simple classification prediction.

**Example 2: Precision Calculation with One-Hot Targets**

Now let's consider a case with precision calculation where predictions can span different classes.

```python
import torch
import torchmetrics

# Initialize the precision metric.
precision_metric = torchmetrics.Precision(task='multiclass', num_classes=3)

# Model output probabilities for a batch of 3.
probabilities = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])

# Corresponding true labels encoded as one-hot vectors.
one_hot_labels = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

# Update precision using one-hot labels.
precision_metric.update(probabilities, one_hot_labels)

# Get the computed precision.
precision = precision_metric.compute()

print(f"Precision: {precision}") # Output: Precision: 1.0
```

Here, the precision metric is used. The same principles hold, where `one_hot_labels` is intelligently converted to its corresponding class index prior to calculating precision against the predicted class index (argmax of probabilities). Again, we observe that despite giving it one-hot encoded labels, the metric behaves correctly. This time, because all three class predictions match their respective one-hot labels, precision is 1.0.

**Example 3: F1 Score and handling multiple batches**

Let's see how this behaves over multiple batches, using a different metric, the F1-score.

```python
import torch
import torchmetrics

# Initialize the f1 score.
f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=3)

# Batch 1 model predictions and one-hot labels.
probs_batch1 = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]])
one_hot_labels_batch1 = torch.tensor([[0, 1, 0], [1, 0, 0]])

# Update metric with the first batch.
f1_metric.update(probs_batch1, one_hot_labels_batch1)

# Batch 2 model predictions and one-hot labels.
probs_batch2 = torch.tensor([[0.2, 0.1, 0.7], [0.3, 0.6, 0.1]])
one_hot_labels_batch2 = torch.tensor([[0, 0, 1], [0, 1, 0]])

# Update metric with the second batch.
f1_metric.update(probs_batch2, one_hot_labels_batch2)

# Compute F1 Score after processing both batches.
f1_score = f1_metric.compute()

print(f"F1 Score: {f1_score}") # Output: F1 Score: 1.0
```

In this example, we show how the metric aggregates information over multiple batches to provide the final score. The F1 score correctly accounts for both batches and returns 1.0, again demonstrating the automatic one-hot encoding handling of TorchMetrics over multiple update steps.

In each of these examples, we have demonstrated that TorchMetrics can accept one-hot encoded targets without requiring preprocessing. This intelligent handling prevents errors and streamlines development by avoiding tedious manual conversion steps before each metric update. The library automatically translates the one-hot encoded data to an integer representation based on `argmax`.

For those seeking further knowledge, I suggest exploring the official documentation of TorchMetrics, which offers detailed explanations and examples of all metrics, including those discussed above. Specifically, examining the source code for metric classes and their associated `update` method can offer greater clarity on their internal workings. Textbooks on deep learning and classification tasks can provide a foundational context for better understanding the implications and usage of these metrics. Finally, researching the mathematics behind performance metrics like `Precision`, `Recall`, and `F1-Score` can provide essential insights into their meaning and interpretation. These resources together provide a deeper understanding of how TorchMetrics functions with one-hot encoded vectors and also solidifies your grasp on machine learning performance analysis.
