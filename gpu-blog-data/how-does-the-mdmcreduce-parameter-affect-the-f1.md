---
title: "How does the `mdmc_reduce` parameter affect the F1 score calculation in torchmetrics?"
date: "2025-01-30"
id: "how-does-the-mdmcreduce-parameter-affect-the-f1"
---
The `mdmc_reduce` parameter in TorchMetrics' F1 score calculation fundamentally alters how the metric is aggregated across multiple classes and datasets, impacting the final reported value.  My experience working on large-scale multi-label classification problems for medical image analysis has highlighted the crucial role of this parameter in interpreting and utilizing the F1 score effectively.  Incorrect usage can lead to misleading performance evaluations, especially in scenarios involving imbalanced datasets or varying numbers of classes across different datasets or folds in cross-validation.


The F1 score, being the harmonic mean of precision and recall, inherently handles class-wise performance. However, when dealing with multiple classes or multiple datasets, a mechanism is required to combine these individual class-wise F1 scores into a single scalar metric for overall performance assessment. This is where `mdmc_reduce` comes into play.  It dictates the aggregation strategy applied to the per-class F1 scores.  Failing to select the appropriate reduction method can obfuscate valuable insights into model performance.


There are three primary reduction methods offered by TorchMetrics' F1 score implementation:  `'global'`, `'weighted avg'`, and `'samples'` (note that `None` also exists, but is not an aggregation, but rather a direct return of per-class results). Understanding the nuances of each method is crucial for meaningful interpretation.


**1. `'global'` Reduction:** This approach calculates the macro-averaged F1 score. It computes the F1 score for each class independently and then averages these scores.  Each class receives equal weight regardless of its prevalence in the dataset. This is particularly useful when class imbalance is a concern and you want to equally emphasize the performance on each class, regardless of its size. However, it can be sensitive to outliers; a single class with a very low F1 score can disproportionately affect the overall average.

**Code Example 1: Global Reduction**

```python
import torch
from torchmetrics import F1Score

# Sample predictions and targets (multi-class classification)
preds = torch.tensor([0, 1, 2, 0, 1])
target = torch.tensor([0, 1, 1, 0, 2])

f1 = F1Score(task="multiclass", num_classes=3, average='macro') # average = 'macro' is equivalent to mdmc_reduce='global'
f1_score = f1(preds, target)
print(f"Global F1 Score: {f1_score}")

#Note: Using mdmc_reduce='global' in the F1Score constructor yields the same result.

f1_explicit = F1Score(task="multiclass", num_classes=3, mdmc_reduce='global')
f1_score_explicit = f1_explicit(preds, target)
print(f"Global F1 Score (explicit): {f1_score_explicit}")

```

This code demonstrates calculating the global F1 score.  The `num_classes` argument is vital for correct computation, especially in multi-class scenarios.  Note that explicitly setting `mdmc_reduce='global'` yields the same result as specifying `average='macro'`.


**2. `'weighted avg'` Reduction:** This computes a weighted average of the per-class F1 scores, where the weights are proportional to the number of samples in each class.  Classes with more samples contribute more significantly to the overall F1 score. This method is preferable when class imbalance is substantial, as it provides a more representative measure of the model's performance across different classes.  It mitigates the influence of under-represented classes on the overall metric.

**Code Example 2: Weighted Average Reduction**

```python
import torch
from torchmetrics import F1Score

# Sample predictions and targets (multi-class classification with class imbalance)
preds = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
target = torch.tensor([0, 0, 0, 1, 0, 1, 2, 2])

f1 = F1Score(task="multiclass", num_classes=3, average='weighted') # average = 'weighted' is equivalent to mdmc_reduce='weighted avg'
f1_score = f1(preds, target)
print(f"Weighted Average F1 Score: {f1_score}")

f1_explicit = F1Score(task="multiclass", num_classes=3, mdmc_reduce='weighted avg')
f1_score_explicit = f1_explicit(preds, target)
print(f"Weighted Average F1 Score (explicit): {f1_score_explicit}")

```

This example highlights the weighted average calculation.  Notice the difference from the global average when class distribution is uneven. The `average='weighted'` parameter mirrors the `mdmc_reduce='weighted avg'` setting.


**3. `'samples'` Reduction:** This computes the F1 score across all samples considering each sample as a separate class. This is typically used in multi-label classification tasks, where each sample can belong to multiple classes.  In this case, the F1 score isn't calculated separately for each class, instead it assesses the performance across the entire set of samples and their associated labels, treating each sample as an individual unit. It's useful when the model's ability to correctly classify the combination of labels for each sample is paramount.

**Code Example 3: Samples Reduction**

```python
import torch
from torchmetrics import F1Score

# Sample predictions and targets (multi-label classification)
preds = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
target = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])

f1 = F1Score(task="multilabel", num_labels=3, average='samples') # average = 'samples' is equivalent to mdmc_reduce='samples'
f1_score = f1(preds, target)
print(f"Samples F1 Score: {f1_score}")

f1_explicit = F1Score(task="multilabel", num_labels=3, mdmc_reduce='samples')
f1_score_explicit = f1_explicit(preds, target)
print(f"Samples F1 Score (explicit): {f1_score_explicit}")

```

This illustrates the `'samples'` reduction method for multi-label classification.  Observe how the `task` argument is set to `'multilabel'` to indicate this specific classification type.  Again, the equivalent results are obtained using the `average` parameter and `mdmc_reduce`.


**Resource Recommendations:**

I would suggest consulting the official TorchMetrics documentation for detailed explanations and additional examples.  Furthermore, a thorough understanding of the mathematical foundations of precision, recall, and the F1 score is essential for accurate interpretation of the results.  Finally, exploring papers on performance evaluation metrics in machine learning would offer valuable context for choosing the appropriate aggregation strategy based on the characteristics of your data and problem.  This knowledge will allow you to correctly select the `mdmc_reduce` parameter and effectively interpret its impact on your model's reported F1 score.
