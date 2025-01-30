---
title: "How can I use PyTorch Lightning's accuracy metric while ignoring specific classes?"
date: "2025-01-30"
id: "how-can-i-use-pytorch-lightnings-accuracy-metric"
---
The challenge with applying a standard accuracy metric in multi-class classification often arises when certain classes are irrelevant to the evaluation, or when class imbalances significantly skew the global performance figure. This is particularly pertinent in my work with medical image segmentation where, for example, a background class might dominate the pixel count, rendering overall accuracy less informative than metrics focused on specific organ segmentation. PyTorch Lightning provides a robust framework, but its direct metric integration does not natively support class exclusion. Instead, we leverage its flexibility to construct a custom accuracy metric that meets this need.

The core idea is to modify how accuracy is calculated. Instead of considering all classes in the true and predicted labels, we must filter out those we wish to ignore prior to comparison. This necessitates access to the predicted class labels, the ground truth class labels, and a defined set of classes to exclude. The fundamental workflow remains consistent: we accumulate the predicted and true values, then compute the metric at the end of an epoch. The alteration lies specifically in the accumulation step. Instead of directly using the raw predictions and true labels, we select only the relevant subset.

To implement this using PyTorch Lightning, we don't override the `Accuracy` metric from the `torchmetrics` package; rather, we build upon its principles. We'll inherit from `torchmetrics.Metric` and implement the `update` and `compute` methods, but crucially, within the `update` method, we will introduce the filtering logic. The metric stores all the filtered predictions and ground truth data during each training/validation batch for subsequent computation.

Below are three distinct code examples, showing progressively more robust implementations to handle varying data formats.

**Example 1: Simple Class Exclusion with Pre-Processed Tensors**

This initial example demonstrates the core logic for filtering when we already have the predictions and ground truth labels as tensors of class IDs.

```python
import torch
import torchmetrics

class FilteredAccuracy(torchmetrics.Metric):
    def __init__(self, ignore_classes: list, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_classes = ignore_classes
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Assuming preds and target are tensors of class IDs (long dtype)
        mask = torch.ones_like(target, dtype=torch.bool)
        for ignore_class in self.ignore_classes:
           mask = mask & (target != ignore_class)

        filtered_preds = preds[mask]
        filtered_target = target[mask]

        self.correct += (filtered_preds == filtered_target).sum()
        self.total += filtered_target.numel()

    def compute(self):
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)

# Example Usage:
predictions = torch.tensor([0, 1, 2, 3, 1, 2, 0, 4])
ground_truth = torch.tensor([0, 1, 2, 0, 1, 3, 0, 4])
ignore_list = [0, 3]

metric = FilteredAccuracy(ignore_classes=ignore_list)
metric.update(predictions, ground_truth)
accuracy = metric.compute()
print(f"Accuracy ignoring classes {ignore_list}: {accuracy}")

```

*   **Explanation:**  The `FilteredAccuracy` class inherits from `torchmetrics.Metric`, initializing with a list of classes to exclude. `update` performs a boolean mask filtering, comparing each element in the `target` to ignore classes, resulting in an element-wise `True/False` tensor; the `mask` is then used to index the `preds` and `target`.  The total number of correct predictions and the total valid elements are stored within the `self.correct` and `self.total` variables. The `compute` method calculates and returns the accuracy.

*   **Use Case:** This is suitable when your predictions and ground truths are already provided as single class IDs. This is the direct equivalent of a model directly outputting argmax predictions and ground truths being discrete class IDs.

**Example 2: Handling Model Output Probabilities**

Often, model outputs are not class IDs but class probabilities. This example adds functionality to extract the predicted class using `argmax` on the probabilities.

```python
import torch
import torchmetrics

class FilteredAccuracyProbs(torchmetrics.Metric):
    def __init__(self, ignore_classes: list, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_classes = ignore_classes
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Assuming preds are probabilities, target is tensor of class IDs
        predicted_classes = torch.argmax(preds, dim=-1) # get predicted class id
        mask = torch.ones_like(target, dtype=torch.bool)
        for ignore_class in self.ignore_classes:
           mask = mask & (target != ignore_class)

        filtered_preds = predicted_classes[mask]
        filtered_target = target[mask]


        self.correct += (filtered_preds == filtered_target).sum()
        self.total += filtered_target.numel()

    def compute(self):
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)


# Example Usage:
probabilities = torch.tensor([
    [0.1, 0.8, 0.05, 0.05],
    [0.2, 0.3, 0.4, 0.1],
    [0.9, 0.05, 0.02, 0.03],
    [0.01, 0.01, 0.01, 0.97]
])

ground_truth = torch.tensor([1, 2, 0, 3])
ignore_list = [0, 3]


metric = FilteredAccuracyProbs(ignore_classes=ignore_list)
metric.update(probabilities, ground_truth)
accuracy = metric.compute()

print(f"Accuracy ignoring classes {ignore_list}: {accuracy}")
```
*   **Explanation:** This version expects the model output `preds` as a tensor of probabilities per class. Thus, we introduce `torch.argmax(preds, dim=-1)` to retrieve the predicted class ID before applying the filtering and accuracy calculation, otherwise, the functionality is identical to the previous example.
*   **Use Case:**  This is highly common in classification tasks where the model outputs a vector of probabilities representing class scores, requiring an argmax for discrete predictions. This reflects the common situation where the raw output of a model is probabilities and it needs to be converted to a discrete prediction.

**Example 3: Handling Batched Data**

The previous examples handle single instances, but in practice, training data is processed in batches. This example includes proper batch handling.

```python
import torch
import torchmetrics

class FilteredAccuracyBatched(torchmetrics.Metric):
    def __init__(self, ignore_classes: list, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_classes = ignore_classes
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Assuming preds are probabilities, target is tensor of class IDs
        predicted_classes = torch.argmax(preds, dim=-1) # get predicted class id
        batch_size = target.shape[0]
        for i in range(batch_size):
            mask = torch.ones_like(target[i], dtype=torch.bool)
            for ignore_class in self.ignore_classes:
                mask = mask & (target[i] != ignore_class)


            filtered_preds = predicted_classes[i][mask]
            filtered_target = target[i][mask]

            self.correct += (filtered_preds == filtered_target).sum()
            self.total += filtered_target.numel()

    def compute(self):
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)

# Example Usage:
batch_probabilities = torch.tensor([
    [[0.1, 0.8, 0.05, 0.05], [0.2, 0.3, 0.4, 0.1]],
    [[0.9, 0.05, 0.02, 0.03], [0.01, 0.01, 0.01, 0.97]]
])

batch_ground_truth = torch.tensor([[1, 2], [0, 3]])
ignore_list = [0, 3]


metric = FilteredAccuracyBatched(ignore_classes=ignore_list)
metric.update(batch_probabilities, batch_ground_truth)
accuracy = metric.compute()

print(f"Accuracy ignoring classes {ignore_list}: {accuracy}")
```

*   **Explanation:** This expands the previous example by iterating through each instance of the batch, applying the filtering to individual samples, before finally accumulating the results. The crucial element is the batch dimension handling, ensuring that metrics are calculated on samples instead of across the entire batch as a single entity.
*   **Use Case:** This is the practical version used within a PyTorch Lightning training loop, as the data is passed in batches.

For robust, production-ready implementations, consult the `torchmetrics` documentation regarding `dist_sync_on_step` for multi-GPU training and validation. It's also useful to experiment with different filtering techniques. Instead of iterating through the `ignore_classes`, a `torch.isin` check can potentially be faster for larger exclusion lists if applicable. Also, consider adding checks for input tensor shapes and types within the `update` method for more robust behavior.

For further information, familiarize yourself with the PyTorch documentation on tensor manipulation, specifically indexing and masking. Also, studying the `torchmetrics` library documentation, especially the section describing custom metrics is essential. These provide detailed guides on creating metrics, ensuring they are correctly integrated within the PyTorch ecosystem. Finally, research best practices in multi-class classification metrics to understand the implications of excluding classes on performance measurement. Understanding the limitations and biases introduced by class exclusions is paramount when performing data analysis.
