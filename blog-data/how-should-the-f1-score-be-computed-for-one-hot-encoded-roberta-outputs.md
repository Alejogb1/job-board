---
title: "How should the F1 score be computed for one-hot encoded Roberta outputs?"
date: "2024-12-16"
id: "how-should-the-f1-score-be-computed-for-one-hot-encoded-roberta-outputs"
---

Let’s tackle this. I've seen this particular issue crop up multiple times, particularly when fine-tuning transformer models like Roberta for tasks involving categorical classification. It’s not as straightforward as it might first appear, especially when dealing with one-hot encoded outputs. A typical gotcha is thinking you can directly apply a binary F1 calculation. Instead, we need to approach it with a clear understanding of how to handle multi-class predictions.

My team and I faced this head-on a few years back while working on a sentiment analysis project. We had a Roberta model, meticulously trained, spitting out one-hot encoded vectors for each sentiment class: positive, negative, and neutral. Initially, we naively computed F1 scores as if they were binary problems. The resulting scores were, shall we say, less than informative – often giving deceptively high or low values. After much investigation, debugging, and a few whiteboard sessions, we finally settled on a procedure that I found consistently reliable.

The central issue stems from the fact that a one-hot encoded output isn’t inherently a binary classification. It represents a probability distribution across multiple classes, and each position in the one-hot vector corresponds to a specific class's activation. We aren’t comparing “positive” vs “not positive,” but rather trying to accurately predict one of *n* classes, where *n* is the total number of distinct classes in your data set. Therefore, computing a standard F1 score in its binary form does not accurately reflect the performance across the various classes.

To do this correctly, we need to decompose the evaluation into a *per-class* calculation, then compute the macro- or micro-average F1 score as needed, or potentially consider a weighted average based on the class support in your data. Here's how I've approached it in practice:

1. **Prediction Conversion:** Your Roberta model outputs a probability distribution for each class. The one-hot encoding is usually the desired format for the model's *target* during training, but the actual prediction may exist as a vector of floating point values. To find the predicted class for the metric evaluation, you first need to find the class with the highest probability. This is done by identifying the index of the highest value in the output vector for each sample.

2.  **Per-Class Precision and Recall:** For each class, calculate its precision and recall. Precision for a class is the number of true positives divided by the total predicted positives. Recall for a class is the number of true positives divided by the total actual positives. Think of true positives as when the predicted class matches the ground-truth class for an instance.

3.  **Per-Class F1 Score:** With per-class precision and recall in hand, the F1 score for each class can be calculated using the standard formula: `2 * (precision * recall) / (precision + recall)`.

4.  **Aggregation:** Finally, we need a single score to represent the model's overall performance. I've found macro-averaging to be the most informative for most classification scenarios. A macro-average calculates the F1 score for each class and then takes the arithmetic average of all the class-wise F1 scores. This treats all classes equally, regardless of their prevalence in the data. The alternative, micro-averaging, calculates overall precision and recall across all classes and then uses those values to compute the F1 score. This approach is suitable when you are primarily interested in overall accuracy and does not emphasize individual classes. There’s also weighted average, which weights the F1 score by the class support in your data, and which is another valid method if the class imbalances are significant.

Now, let's look at a few code examples in python, using `sklearn` for calculating the metrics, and assuming we have a `y_true` with the integer representations of each class, and the `y_pred` being the predicted probabilities.

**Example 1: Macro-averaged F1 score**

```python
import numpy as np
from sklearn.metrics import f1_score

def calculate_macro_f1(y_true, y_pred_probs):
    """Calculates macro-averaged F1 score for one-hot encoded outputs.

    Args:
        y_true: Ground truth labels as integers
        y_pred_probs: Predicted probabilities (e.g., from Roberta output) as a numpy array.

    Returns:
        The macro-averaged F1 score.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)  # convert probabilities to predicted classes
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


# Example usage:
y_true = np.array([0, 1, 2, 0, 1, 2]) # ground truth classes
y_pred_probs = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.2, 0.6]
]) # predicted probabilities
macro_f1 = calculate_macro_f1(y_true, y_pred_probs)
print(f"Macro-averaged F1 score: {macro_f1}")

```

**Example 2: Micro-averaged F1 score**

```python
import numpy as np
from sklearn.metrics import f1_score

def calculate_micro_f1(y_true, y_pred_probs):
    """Calculates micro-averaged F1 score for one-hot encoded outputs.

    Args:
        y_true: Ground truth labels as integers
        y_pred_probs: Predicted probabilities (e.g., from Roberta output) as a numpy array.

    Returns:
        The micro-averaged F1 score.
    """
    y_pred = np.argmax(y_pred_probs, axis=1) # convert probabilities to predicted classes
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1


# Example usage:
y_true = np.array([0, 1, 2, 0, 1, 2])  # ground truth classes
y_pred_probs = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.2, 0.6]
]) # predicted probabilities
micro_f1 = calculate_micro_f1(y_true, y_pred_probs)
print(f"Micro-averaged F1 score: {micro_f1}")

```

**Example 3: Weighted F1 score**

```python
import numpy as np
from sklearn.metrics import f1_score

def calculate_weighted_f1(y_true, y_pred_probs):
     """Calculates weighted F1 score for one-hot encoded outputs.

    Args:
        y_true: Ground truth labels as integers
        y_pred_probs: Predicted probabilities (e.g., from Roberta output) as a numpy array.

    Returns:
        The weighted F1 score.
    """
    y_pred = np.argmax(y_pred_probs, axis=1) # convert probabilities to predicted classes
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

# Example usage:
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 0])  # ground truth classes
y_pred_probs = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.2, 0.6],
    [0.9, 0.05, 0.05],
    [0.85, 0.10, 0.05],
    [0.9, 0.05, 0.05]
]) # predicted probabilities
weighted_f1 = calculate_weighted_f1(y_true, y_pred_probs)
print(f"Weighted F1 score: {weighted_f1}")

```

In each example, we convert the probability predictions to class predictions by taking `argmax` across the appropriate axis. Then the appropriate `average` setting is applied to the `f1_score` function.

For deepening your understanding, I highly recommend reviewing the material on multi-class classification evaluation. A key text is *'Pattern Recognition and Machine Learning'* by Christopher Bishop. The section covering classification evaluation metrics provides excellent background. Additionally, the *'Handbook of Research on Machine Learning Applications'* edited by Hassan and Ghali, dedicates considerable space to performance evaluation in multi-class settings. Reading these resources helped me formulate the approach detailed above and avoid the common pitfalls I've observed.

In closing, correctly computing the F1 score with one-hot encoded outputs requires careful consideration of the multi-class nature of the problem. Avoid binary approximations, use per-class calculations, and then aggregate with an appropriate averaging method. The resulting metrics will be far more meaningful and help guide your model improvements more effectively.
