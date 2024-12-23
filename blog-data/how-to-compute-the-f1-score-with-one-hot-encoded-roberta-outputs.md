---
title: "How to compute the F1 score with one-hot encoded Roberta outputs?"
date: "2024-12-23"
id: "how-to-compute-the-f1-score-with-one-hot-encoded-roberta-outputs"
---

Let's tackle the intricacies of calculating the F1 score when dealing with one-hot encoded Roberta outputs. I remember a project a few years back where we were fine-tuning a Roberta model for multi-label text classification – it was crucial then, as it is now, to accurately measure performance, and the F1 score is a cornerstone metric in these scenarios. Unlike binary or multi-class classification with single predicted labels, one-hot encoding for multi-label scenarios introduces a bit of a wrinkle into the process, so let’s delve into it.

The core challenge is that Roberta, after a linear layer, typically outputs logits (raw scores), not directly the one-hot encoded predictions we need for the F1 score. We'll need to convert these logits into probabilities using a sigmoid function, then apply a threshold to create the predicted labels, which we then can compare with the true labels also represented in one-hot encoding.

Here's a breakdown of the necessary steps, along with code examples to make it concrete.

First, understand that the F1 score, at its heart, is the harmonic mean of precision and recall. Precision tells us what proportion of our predicted positives are actually true positives; recall, conversely, captures what proportion of the actual positives we correctly identified. In mathematical terms:

*   **Precision** = True Positives / (True Positives + False Positives)
*   **Recall** = True Positives / (True Positives + False Negatives)
*   **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)

For multi-label, one-hot encoded scenarios, this calculation isn't as simple as counting across single labels. We compute these metrics on a *per-label basis*, then average or aggregate them. The two most common aggregation strategies are: *macro-averaging*, which calculates the metric for each label individually and then takes the average, and *micro-averaging*, which calculates the overall true positives, false positives, and false negatives across all labels.

Let's look at code demonstrating how to do this in Python using `torch` and `sklearn`.

**Example 1: Converting Logits to Predictions**

This snippet focuses on converting the logits from Roberta into one-hot encoded predictions.

```python
import torch
import torch.nn.functional as F
import numpy as np

def logits_to_onehot(logits, threshold=0.5):
    """Converts Roberta logits to one-hot encoded predictions.

    Args:
        logits (torch.Tensor): The output logits from Roberta.
        threshold (float): The threshold to use for binarization.

    Returns:
        torch.Tensor: One-hot encoded predictions.
    """
    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).int()
    return predictions

# Example Usage
logits = torch.randn(10, 5) # 10 samples, 5 labels
predictions = logits_to_onehot(logits)
print("Predictions (One-Hot Encoded):\n", predictions)
```

In this example, we're using the sigmoid function to convert logits to probabilities. These probabilities indicate how confident the model is for each label, ranging from 0 to 1. We then apply a threshold (0.5 is common) to binarize these predictions, effectively converting the probabilities into a one-hot vector. This threshold is a hyperparameter you can adjust based on the specific requirements of your task.

**Example 2: Macro-Averaged F1 Score**

Now, let's show the calculation of the macro-averaged F1 score.

```python
from sklearn.metrics import f1_score

def macro_f1_score(true_labels, predictions):
    """Calculates the macro-averaged F1 score.

    Args:
        true_labels (torch.Tensor): True one-hot encoded labels.
        predictions (torch.Tensor): Predicted one-hot encoded labels.

    Returns:
        float: Macro-averaged F1 score.
    """
    true_labels_np = true_labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    f1 = f1_score(true_labels_np, predictions_np, average='macro', zero_division=0)
    return f1

# Example Usage
true_labels = torch.randint(0, 2, (10, 5)) # Example true labels
f1_macro = macro_f1_score(true_labels, predictions)
print("Macro-averaged F1 Score:", f1_macro)
```

Here, `sklearn.metrics.f1_score` is used, specifically setting the `average` parameter to `'macro'`. It does the label-wise calculations, and then the F1 scores from each label are averaged for a final result. The `zero_division=0` parameter ensures that if a label has zero true positives, we don’t get a division-by-zero error; instead, its f1 score will be set to zero which makes sense from an F1 score calculation point of view.

**Example 3: Micro-Averaged F1 Score**

Finally, let’s calculate the micro-averaged F1 score.

```python
def micro_f1_score(true_labels, predictions):
    """Calculates the micro-averaged F1 score.

    Args:
        true_labels (torch.Tensor): True one-hot encoded labels.
        predictions (torch.Tensor): Predicted one-hot encoded labels.

    Returns:
        float: Micro-averaged F1 score.
    """
    true_labels_np = true_labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    f1 = f1_score(true_labels_np, predictions_np, average='micro', zero_division=0)
    return f1

# Example Usage
f1_micro = micro_f1_score(true_labels, predictions)
print("Micro-averaged F1 Score:", f1_micro)
```

The only change from the macro-averaged case is the `average` parameter set to `'micro'`.  This computes the overall true positive, false positive, and false negative counts across *all* label predictions, then computes the precision and recall from these counts, and, lastly, calculates a single, global F1 score.

In my experience, deciding between macro- and micro-averaging is contextual. If all labels are equally important in your task, macro-averaging is often preferred. If instead, the dataset is imbalanced and the goal is to optimize the overall precision/recall across all labels together, then micro-averaging might be more relevant, although you could consider weighting each label in the macro average instead.

For further exploration into the theoretical underpinnings of these metrics and best practices, I'd recommend consulting the following resources. Specifically for F1 score calculation, you will find a solid explanations in the foundational text *“Pattern Recognition and Machine Learning”* by Christopher M. Bishop, especially the chapters discussing evaluation metrics in classification problems. For understanding metrics more generally, *“Information Retrieval: Implementing and Evaluating Search Engines”* by Stefan Büttcher, Charles L. A. Clarke, and Gordon V. Cormack is highly insightful, particularly on the trade-offs between different evaluation methodologies. Finally, exploring the sklearn documentation itself on classification metrics and their average parameters will be extremely helpful. By thoroughly understanding these concepts, you will be well equipped to select the most appropriate approach to evaluate your model’s performance.
