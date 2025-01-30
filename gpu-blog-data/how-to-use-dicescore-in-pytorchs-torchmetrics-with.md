---
title: "How to use `dice_score()` in PyTorch's `torchmetrics` with correct 'preds' and 'target' inputs?"
date: "2025-01-30"
id: "how-to-use-dicescore-in-pytorchs-torchmetrics-with"
---
The `dice_score` function within the `torchmetrics` library calculates the Dice coefficient, a statistic that quantifies the similarity between two sets of data, commonly used for evaluating the performance of segmentation models. Incorrect interpretation of the `preds` and `target` input requirements will lead to inaccurate, often dramatically deflated, scores, misguiding model development. It’s critical to understand their format and ensure alignment with your specific task.

I've repeatedly encountered situations in my work on medical image segmentation where initial `dice_score` results were misleading simply due to incorrect data formatting. For instance, an image segmentation task might output probabilities per class for each pixel, while the `dice_score` function requires a particular encoding to function correctly. `torchmetrics` offers flexibility, but this requires a solid understanding of input expectations.

The crux of the problem lies in the fact that `dice_score`, as implemented in `torchmetrics`, typically operates on *integer encoded* data for `target` and the correspondingly formatted, often *binary* for `preds` when dealing with multi-class segmentation problems. It does not, in its typical usage with multi-class labels, directly process one-hot encoded probabilities for the `preds`, or class probabilities. It expects the `target` to be a tensor of integers representing class labels and the `preds` tensor to be either integer labels or, if probabilities are initially output, binarized or argmaxed labels before being used. For binary segmentation, the `preds` will typically contain probabilities which do not need further processing before being used by `dice_score`.

Let’s break this down. `dice_score` calculates:

Dice =  2 * |X ∩ Y| / (|X| + |Y|)

Where X and Y represent the sets of elements belonging to the foreground in the `preds` and `target` tensors respectively, and |...| denotes cardinality of the set. This calculation is performed per class when dealing with multiclass tasks, and averaged across classes if the average is requested.

The `target` tensor should hold the ground-truth segmentation masks as integer class labels. Each unique integer value in the tensor represents a distinct class. For example, for a three-class problem, a `target` tensor would contain the values `0`, `1`, and `2`, representing background, object class A, and object class B respectively. It is not a one-hot encoded tensor and must have dimensions aligned with how the `preds` are prepared.

The `preds` tensor, in its typical usage with multi-class labels, must be in a similar integer format after any required post-processing. For a model that outputs probabilities across classes, you’ll first use the `argmax` operation along the class dimension to get the most likely class prediction at each spatial location. The result of `argmax` is then what will become the input to `dice_score`. For binary segmentation the output is typically a probability of the pixel representing the foreground which can be directly used by `dice_score`.

Here are examples to demonstrate the differences in handling both binary and multiclass inputs:

**Example 1: Binary Segmentation**

```python
import torch
from torchmetrics.functional import dice_score

# Example binary prediction probabilities
preds = torch.tensor([0.2, 0.8, 0.1, 0.9, 0.6, 0.4, 0.7, 0.3])
preds = preds.reshape(1, 1, 2, 4) # Reshaped to (batch, channels, height, width)


# Example binary ground truth
target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
target = target.reshape(1, 1, 2, 4) # Reshaped to (batch, channels, height, width)

# Calculate Dice score
dice = dice_score(preds, target)
print(f"Binary Dice Score: {dice.item()}")
```

In this scenario, the `preds` are kept as probabilities and directly provided to the function. The `target` has only two values (0 and 1) and represents the ground truth labels, and the function will correctly compute the Dice score between them. Note that both tensors are shaped appropriately for typical PyTorch image operations.

**Example 2: Multi-class Segmentation (Incorrect)**

```python
import torch
from torchmetrics.functional import dice_score

# Example multi-class prediction probabilities (3 classes)
preds = torch.tensor([
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
    [0.7, 0.1, 0.2],
    [0.1, 0.3, 0.6]
])
preds = preds.reshape(1, 4, 3, 1) # Reshape to batch, height, channels, width

# Example multi-class target ground truth (3 classes)
target = torch.tensor([0, 2, 0, 1])
target = target.reshape(1, 1, 4, 1) # Reshape to batch, channels, height, width

#This is incorrect since the 'preds' contain probabilities

# Calculate Dice score (will produce incorrect results)
dice = dice_score(preds, target, average="macro")

print(f"Multi-class Dice Score (Incorrect): {dice.item()}")
```

This snippet shows an **incorrect** way of using `dice_score` with multi-class data. The `preds` are probability scores, not class labels, and should have been converted using an `argmax` operation before being given to the function. This will produce very poor scores, often close to zero.

**Example 3: Multi-class Segmentation (Corrected)**

```python
import torch
from torchmetrics.functional import dice_score

# Example multi-class prediction probabilities (3 classes)
preds = torch.tensor([
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
    [0.7, 0.1, 0.2],
    [0.1, 0.3, 0.6]
])
preds = preds.reshape(1, 4, 3, 1) # Reshape to batch, height, channels, width

# Example multi-class target ground truth (3 classes)
target = torch.tensor([0, 2, 0, 1])
target = target.reshape(1, 1, 4, 1) # Reshape to batch, channels, height, width

# Convert probabilities to class predictions using argmax
preds_classes = torch.argmax(preds, dim=2).to(torch.int64)
print(preds_classes)

# Calculate Dice score
dice = dice_score(preds_classes, target, average="macro")
print(f"Multi-class Dice Score (Corrected): {dice.item()}")
```

This is the correct approach. We take the `argmax` across the class dimension of `preds` (dim=2), which converts probabilities into the most likely class labels. This newly prepared `preds_classes` tensor is then provided to `dice_score` along with the correctly formatted `target` which contains integer class labels. Note that the average value calculated will be the macro average (equal average of all per-class Dice scores).

In summary, the core of using `dice_score` correctly lies in ensuring that your `preds` and `target` tensors are appropriately formatted based on whether you are performing binary or multiclass segmentation. For multi-class tasks, ensure that the `target` contains integer labels and the `preds` are class indices obtained by taking the argmax of your model output. For binary segmentation, the `target` contains the two classes using integer labels, and `preds` represents the probability of the foreground class. Always check your data’s shape and type before feeding it to `dice_score` to avoid obtaining deceptively low scores, which can lead to false conclusions about your model’s performance.

For further understanding, I recommend consulting the official PyTorch documentation, along with more general resources related to performance metrics in segmentation tasks. Also of great use would be articles, tutorials or books that describe the process of working with multi-class data using `argmax` and related strategies. Focus specifically on how `argmax` is used in both classification and segmentation for correct evaluation. Pay close attention to the shapes of tensors throughout the evaluation pipeline. Finally, familiarize yourself with the concept of macro averaging as used in the `torchmetrics` library for multi-class segmentation tasks.
