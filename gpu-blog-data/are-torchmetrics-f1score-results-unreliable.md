---
title: "Are torchmetrics' F1Score results unreliable?"
date: "2025-01-30"
id: "are-torchmetrics-f1score-results-unreliable"
---
The F1 score, a harmonic mean of precision and recall, is a crucial metric for evaluating the performance of classification models, especially when dealing with imbalanced datasets. My experience using `torchmetrics` in complex medical image analysis, however, has highlighted specific scenarios where its reported `F1Score` can appear misleading if not carefully interpreted. This isn't an indictment of the library itself, but rather a consequence of the metric's sensitivity to implementation details and potential for misapplication. A deeper look reveals that apparent discrepancies often stem from differences in averaging strategies, handling of edge cases (e.g., zero true positives), and the subtleties of per-class vs. micro/macro averaged computation.

The primary challenge is understanding the nuances within `torchmetrics`'s `F1Score` class and how those choices align with the specific evaluation goals. The `average` parameter is central to this. The default is `binary`, suitable for two-class problems. However, when working with multi-class settings, options like `macro`, `micro`, and `weighted` become relevant. `Macro` averaging calculates the F1 score for each class independently and then averages those scores. This treats all classes equally, irrespective of their size. `Micro` averaging aggregates the true positives, false positives, and false negatives across all classes, effectively computing a single F1 score for all classes combined. This is influenced more by the majority class when class imbalance is prevalent. `Weighted` averaging is similar to `macro`, but weights each class’s score by its support (number of true instances). A mismatch between the intended averaging type and the application context can lead to misleading conclusions about the model's actual performance across all classes.

The library also offers distinct functionalities for per-class F1 scores. Accessing these individual scores can provide a more granular picture of performance, pinpointing classes that struggle and informing debugging strategies. A common pitfall, especially when dealing with multi-label classification (where each instance can belong to multiple classes), is not explicitly specifying the correct `multilabel` argument. If instances can belong to more than one class, failing to set `multilabel=True` will lead to incorrect F1 score calculation. This misinterpretation will result in significantly lower, and thus, seemingly poor, performance metrics that do not reflect actual model ability. Furthermore, I have observed that even with appropriate parameter selection, situations with zero true positives in a given class can result in a NaN F1 score, or a 0 score, depending on the library version and selected parameters. This happens due to dividing by zero during calculation of precision or recall. Such values can distort the aggregated F1 results when using `macro` or `weighted` averaging if not properly addressed, requiring careful handling within the evaluation pipeline.

Let’s examine specific code examples that illustrate these nuances.

**Example 1: Multi-Class Classification with Incorrect Averaging**

```python
import torch
from torchmetrics import F1Score

# Example predictions and targets
preds = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
target = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])

# Using default 'binary' average (incorrect for this multi-class case)
f1_binary = F1Score()
f1_binary_score = f1_binary(preds.argmax(dim=1), target.argmax(dim=1))
print(f"Binary F1 Score: {f1_binary_score}")

# Using 'macro' average
f1_macro = F1Score(average="macro", num_classes=3)
f1_macro_score = f1_macro(preds.argmax(dim=1), target.argmax(dim=1))
print(f"Macro F1 Score: {f1_macro_score}")

# Using 'micro' average
f1_micro = F1Score(average="micro", num_classes=3)
f1_micro_score = f1_micro(preds.argmax(dim=1), target.argmax(dim=1))
print(f"Micro F1 Score: {f1_micro_score}")

# Get per-class scores
f1_per_class = F1Score(average=None, num_classes=3)
f1_per_class_scores = f1_per_class(preds.argmax(dim=1), target.argmax(dim=1))
print(f"Per-Class F1 Scores: {f1_per_class_scores}")
```

Here, it is evident that the `binary` average misrepresents the situation, treating the task as multiple one-vs-rest binary problems. The `macro` and `micro` scores provide differing perspectives; in this case, `micro` is inflated due to the dataset's balanced class distribution. The per-class scores highlight that class 0 has a slightly lower F1 than the rest. Failing to align the averaging with the problem at hand would lead to inaccurate interpretations of the model's performance.

**Example 2: Handling Multi-Label Data**

```python
import torch
from torchmetrics import F1Score

# Example multi-label predictions and targets
preds_multilabel = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
target_multilabel = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 1]])

# Incorrectly using F1Score without multilabel argument
f1_wrong = F1Score(num_classes=3)
f1_wrong_score = f1_wrong(preds_multilabel, target_multilabel)
print(f"Incorrect F1 Score (No multilabel): {f1_wrong_score}")

# Correctly using F1Score with multilabel = True
f1_correct = F1Score(num_classes=3, multilabel=True)
f1_correct_score = f1_correct(preds_multilabel, target_multilabel)
print(f"Correct F1 Score (multilabel=True): {f1_correct_score}")
```

This example demonstrates the significant impact of the `multilabel` parameter. By neglecting to specify `multilabel=True`, the metric treats the task as multi-class, resulting in an incorrect F1 score. The correct calculation, with `multilabel=True`, provides an appropriate evaluation of performance in this multi-label scenario.

**Example 3: Zero True Positives**

```python
import torch
from torchmetrics import F1Score

# Example with zero true positives for class 0
preds_zero_tp = torch.tensor([1, 2, 1, 2])
target_zero_tp = torch.tensor([2, 2, 1, 2]) # No prediction matches target in class 0

f1_zero = F1Score(average="macro", num_classes=3)
f1_zero_score = f1_zero(preds_zero_tp, target_zero_tp)
print(f"F1 Score with zero true positives: {f1_zero_score}")

f1_zero_weighted = F1Score(average="weighted", num_classes=3)
f1_zero_weighted_score = f1_zero_weighted(preds_zero_tp, target_zero_tp)
print(f"Weighted F1 Score with zero true positives: {f1_zero_weighted_score}")

f1_zero_micro = F1Score(average="micro", num_classes=3)
f1_zero_micro_score = f1_zero_micro(preds_zero_tp, target_zero_tp)
print(f"Micro F1 Score with zero true positives: {f1_zero_micro_score}")

f1_zero_per_class = F1Score(average=None, num_classes=3)
f1_zero_per_class_score = f1_zero_per_class(preds_zero_tp, target_zero_tp)
print(f"Per-class F1 score: {f1_zero_per_class_score}")
```

In this example, class 0 has zero true positives which will produce a NaN F1 score when computed separately. When this situation occurs in per-class results it is vital to handle these values appropriately, depending on the specific averaging being applied. In this example, `macro` computes F1 on a per-class basis, so a zero true-positive case in any of the per-class computations will result in a `nan` value that will impact the overall `macro` average result. Alternatively, `micro` aggregation, computes all the metrics and then the mean for the entire set, thus mitigating the issue.

In summary, `torchmetrics`’ `F1Score` is a reliable tool but is susceptible to misuse if its underlying mechanics are not thoroughly understood. Selection of the correct averaging strategy for the given task, accurate handling of the `multilabel` setting, careful interpretation of per-class scores, and explicit consideration of zero true positive occurrences are critical for ensuring the validity and interpretability of the metric. The observed issues often do not represent a flaw in `torchmetrics`' design but are rather the result of a nuanced evaluation context and highlight the need for meticulous application of any metric.

For a deeper understanding, I recommend reviewing materials focusing on classification performance metrics, particularly precision, recall, and the F1 score's properties. Explore literature covering the concepts of micro, macro, and weighted averaging. Furthermore, consulting resources discussing the intricacies of multi-label and multi-class classification will prove beneficial. Also, carefully examine the specific documentation and code examples provided for the `torchmetrics` library itself to understand how the internal calculations for `F1Score` are performed and to identify any changes across versions.
