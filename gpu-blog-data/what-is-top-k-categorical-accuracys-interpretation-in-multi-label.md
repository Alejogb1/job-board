---
title: "What is top-k categorical accuracy's interpretation in multi-label classification?"
date: "2025-01-30"
id: "what-is-top-k-categorical-accuracys-interpretation-in-multi-label"
---
Top-k categorical accuracy in multi-label classification presents a nuanced interpretation compared to its single-label counterpart.  The key distinction lies in how we define "correct" predictions when a single instance can belong to multiple categories simultaneously.  My experience developing multi-label image classifiers for satellite imagery analysis highlighted this subtlety; simply counting perfectly predicted label sets proved insufficient for evaluating model performance in scenarios where partial matches were valuable.

Top-k accuracy addresses this limitation.  Instead of requiring a perfect match between the predicted and true label sets, it measures the proportion of instances where at least *k* of the predicted top-k labels are correct.  This accounts for situations where the model might correctly identify a significant portion of the relevant labels, even if it misses some or includes a few incorrect ones. The parameter *k* provides a tunable level of stringency; a higher *k* demands a greater overlap between predicted and true labels for a successful prediction.

Let's consider the formal definition:  Given an instance with a true label set *Y* and a predicted label set *P*, ordered by confidence score, the prediction is deemed correct if  | *Y* ∩ *P<sub>k</sub>* | ≥ *k*, where *P<sub>k</sub>* is the set of the top *k* labels in *P*.  Top-k accuracy is then the ratio of correctly predicted instances to the total number of instances in the evaluation set.

This differs markedly from standard multi-label accuracy which demands  *Y* == *P* for a correct classification.  For instance, if an image contains three labels (cloud, building, road) and the model predicts (cloud, building, water), standard accuracy would mark it as incorrect. However, with top-k=2, this would be classified as correct if 'cloud' and 'building' were the top two predictions. This provides a more forgiving and often more realistic assessment of the model's capabilities, particularly when the task is challenging or the labels are inherently ambiguous.

The choice of *k* itself is crucial.  A smaller *k* will yield a higher accuracy, while a larger *k* reflects a more rigorous evaluation. The optimal *k* depends heavily on the specific application. In my satellite imagery work, a *k* of 2 or 3 proved reasonable, capturing the essence of the key features of a given image while avoiding over-lenient evaluation.  Using a *k* equivalent to the average number of labels per instance becomes akin to standard multi-label accuracy.


Now, let's explore this with code examples, using Python and NumPy:

**Example 1: Basic Top-k Accuracy Calculation**

```python
import numpy as np

def top_k_accuracy(true_labels, predicted_labels, k):
    """Calculates top-k categorical accuracy.

    Args:
        true_labels: A NumPy array of true label sets (lists or arrays).
        predicted_labels: A NumPy array of predicted label sets (lists or arrays).  Assumed to be sorted by confidence.
        k: The value of k for top-k accuracy.

    Returns:
        The top-k accuracy as a float.
    """
    correct_predictions = 0
    for i in range(len(true_labels)):
        top_k_predictions = predicted_labels[i][:k]
        if len(set(true_labels[i]) & set(top_k_predictions)) >= k:
            correct_predictions += 1
    return correct_predictions / len(true_labels)

true_labels = np.array([[1, 2, 3], [4, 5], [1, 3, 5]])
predicted_labels = np.array([[1, 2, 4], [5, 4, 3], [1, 5, 3]])
k = 2
accuracy = top_k_accuracy(true_labels, predicted_labels, k)
print(f"Top-{k} accuracy: {accuracy}")
```

This function directly implements the definition outlined above. It iterates through each instance, finds the intersection between true and top-k predicted labels, and checks if the intersection size meets the *k* threshold.


**Example 2:  Using Scikit-learn's `accuracy_score` for binary relevance**

```python
import numpy as np
from sklearn.metrics import accuracy_score

def top_k_accuracy_sklearn(true_labels, predicted_probabilities, k, num_classes):
    """Calculates top-k accuracy using scikit-learn for binary relevance.

    Args:
        true_labels: A NumPy array of binary label vectors (one-hot encoded).
        predicted_probabilities:  A NumPy array of predicted probabilities for each class.
        k: The value of k.
        num_classes: The total number of classes.

    Returns:
        The top-k accuracy.
    """
    top_k_predictions = np.argsort(predicted_probabilities, axis=1)[:, -k:]
    binary_top_k_predictions = np.zeros_like(predicted_probabilities)
    for i in range(len(top_k_predictions)):
        binary_top_k_predictions[i, top_k_predictions[i]] = 1
    return accuracy_score(true_labels, binary_top_k_predictions)

true_labels = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
predicted_probabilities = np.array([[0.8, 0.9, 0.2], [0.1, 0.7, 0.3], [0.9, 0.1, 0.8]])
k = 2
num_classes = 3
accuracy = top_k_accuracy_sklearn(true_labels, predicted_probabilities, k, num_classes)
print(f"Top-{k} accuracy (Scikit-learn): {accuracy}")

```

This example leverages Scikit-learn's `accuracy_score` function after converting predicted probabilities to binary predictions based on the top-k classes. This approach is more efficient for larger datasets. Note that this method requires one-hot encoding of the true labels.


**Example 3: Handling Imbalanced Datasets using Weighted Top-k Accuracy**

```python
import numpy as np

def weighted_top_k_accuracy(true_labels, predicted_labels, k, class_weights):
    """Calculates weighted top-k accuracy, addressing class imbalance.

    Args:
        true_labels: A NumPy array of true label sets.
        predicted_labels: A NumPy array of predicted label sets.
        k: The value of k.
        class_weights: A dictionary mapping class labels to their weights.

    Returns:
        The weighted top-k accuracy.
    """
    total_weight = 0
    correct_weight = 0
    for i in range(len(true_labels)):
        top_k_predictions = predicted_labels[i][:k]
        instance_weight = sum([class_weights[label] for label in true_labels[i]])
        if len(set(true_labels[i]) & set(top_k_predictions)) >= k:
            correct_weight += instance_weight
        total_weight += instance_weight
    return correct_weight / total_weight

true_labels = np.array([[1, 2, 3], [4, 5], [1, 3, 5]])
predicted_labels = np.array([[1, 2, 4], [5, 4, 3], [1, 5, 3]])
k = 2
class_weights = {1: 0.5, 2: 1, 3: 0.8, 4: 1.2, 5: 0.7}
accuracy = weighted_top_k_accuracy(true_labels, predicted_labels, k, class_weights)
print(f"Weighted Top-{k} accuracy: {accuracy}")

```

This function introduces class weights to address potential class imbalances.  In scenarios where certain classes are far more frequent than others, a standard top-k accuracy might be misleading. This weighted version provides a more nuanced evaluation by considering the relative importance of different classes.


For further exploration, I recommend studying texts on multi-label classification evaluation metrics, specifically focusing on the theoretical underpinnings of various accuracy measures and their practical implications.  A solid grounding in probability and statistics will also prove invaluable in understanding and applying these techniques effectively.  Additionally, exploring advanced machine learning libraries' documentation will showcase efficient implementations of these calculations.
