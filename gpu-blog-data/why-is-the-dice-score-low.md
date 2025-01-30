---
title: "Why is the dice score low?"
date: "2025-01-30"
id: "why-is-the-dice-score-low"
---
The consistently low Dice score, particularly in image segmentation tasks, typically stems from a misalignment between the predicted and ground truth masks rather than a systemic flaw in the model's overall ability to classify pixels. This misalignment can manifest as subtle shifts, inaccurate boundaries, or a tendency to over- or under-segment. I've encountered this often enough in medical imaging projects, and it frequently involves more nuanced problems than simply “poor model training.” It’s critical to inspect the specific failure modes to identify the root cause.

Fundamentally, the Dice score, or F1 score when applied to binary classification like segmentation masks, assesses the overlap between two sets—in this case, the predicted segmentation mask and the ground truth mask. The formula is 2 * |A ∩ B| / (|A| + |B|), where A is the predicted mask, B is the ground truth, and | | represents the size of the set (typically measured as pixel count). This formula penalizes both false positives (predicted pixels outside the ground truth) and false negatives (ground truth pixels missed by the prediction). A low score signifies a significant mismatch in terms of this overlap, not just a general inability of the model to “recognize” objects.

Several factors contribute to this misalignment, and they often operate in conjunction. One common issue is imprecise labeling in the ground truth data. Medical images, for example, frequently require manual annotation by experts. The human element introduces variations; even minor inconsistencies in how a boundary is drawn can translate into noticeable score drops, especially if the boundary region represents a significant percentage of the total object pixels. These labeling variations might be within acceptable clinical bounds but can be problematic for precise overlap metrics like the Dice coefficient.

Another critical contributor is the inherent limitation of the model itself. If the model’s capacity is insufficient, it might fail to learn complex boundary structures or subtle texture differences that define objects of interest. This is not simply about the quantity of data. Models with poorly designed architecture, like those with limited receptive fields, or inadequate regularization, might struggle to form crisp segmentations even when the training dataset is expansive. Conversely, an over-complex model might overfit to subtle training variations, which in turn hinders generalization to unseen data.

Class imbalance, a common condition in segmentation tasks, is another factor. For example, if target objects occupy only a small fraction of the image, the model could become biased towards predicting background. This will manifest as undershooting during segmentation. While a model might accurately identify *some* pixels belonging to the target object, the overall Dice score will be penalized if it is not capturing most of them or is generating spurious areas outside the actual object. To mitigate the effects, I've implemented weighted loss functions or sampling strategies to give higher significance to minority classes (in our case, regions belonging to the object).

Finally, during inference, the model might encounter data that is significantly different from the training set in terms of image characteristics (e.g. brightness, contrast, noise). This shift in data distribution frequently reduces Dice scores. The model that is meticulously trained on a dataset taken with a specific medical device could show poor performance on the data gathered from another medical instrument. Thus, thorough preprocessing during data acquisition and augmentation during training are paramount.

Let’s illustrate these points with code. The first example explores the impact of class imbalance. Imagine a scenario where we have a very small region to segment.

```python
import numpy as np

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8) #Adding small value to prevent zero division.


# Scenario with severe class imbalance and only partial overlap.
ground_truth = np.zeros((100, 100), dtype=int)
ground_truth[40:60, 40:60] = 1  # Small square in the center
predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[45:65, 45:65] = 1 # Prediction partially shifted

print(f"Dice score with slight offset and imbalance: {dice_coefficient(ground_truth, predicted_mask):.4f}")

predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[40:60, 40:60] = 1 # Perfect match
print(f"Dice score with perfect match: {dice_coefficient(ground_truth, predicted_mask):.4f}")
```

This code demonstrates that even with a significant overlap, the overall Dice score is comparatively low because the size of the object (here a square) is small compared to the whole image, and a slight misalignment substantially penalizes the score. In comparison, a perfect match gives a higher Dice score.

The next code block considers the impact of an incorrect edge prediction. Here, we assume that the model has correctly predicted the general area, but it is over-segmenting the boundaries of the object slightly.

```python
# Scenario with edge over-segmentation
ground_truth = np.zeros((100, 100), dtype=int)
ground_truth[20:80, 20:80] = 1
predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[18:82, 18:82] = 1 # Prediction is slightly larger.

print(f"Dice score with over-segmentation: {dice_coefficient(ground_truth, predicted_mask):.4f}")


predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[20:80, 20:80] = 1 # Exact Match
print(f"Dice score with exact segmentation: {dice_coefficient(ground_truth, predicted_mask):.4f}")
```

Even a small over-segmentation, with the core area largely covered, still leads to a reduced Dice score because the overlap is less than perfect. This highlights how sensitive the Dice score is to boundary accuracy and false positives.

Finally, a third example examines the effect of under-segmentation. Here, we assume the model predicts only a subset of the true object, an issue I've observed with more complex model architectures with limited training data.

```python
# Scenario with under-segmentation
ground_truth = np.zeros((100, 100), dtype=int)
ground_truth[20:80, 20:80] = 1
predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[30:70, 30:70] = 1 # Prediction is smaller

print(f"Dice score with under-segmentation: {dice_coefficient(ground_truth, predicted_mask):.4f}")

predicted_mask = np.zeros((100, 100), dtype=int)
predicted_mask[20:80, 20:80] = 1 # Exact match
print(f"Dice score with exact segmentation: {dice_coefficient(ground_truth, predicted_mask):.4f}")
```

This code illustrates that missing sections of the target object, leading to under-segmentation, significantly lowers the Dice coefficient. This often happens when the model fails to learn enough about the object's characteristics.

In summary, understanding a low Dice score requires careful analysis of the specific ways the predicted masks differ from the ground truth masks. It's not simply about a model’s raw predictive power but about its capability to produce accurate spatial overlap. My process for addressing this involves a multi-pronged approach starting with a thorough examination of the training and validation datasets to ensure label consistency and quality. Data augmentation can help the model generalize to variations in data while ensuring that data-centric issues are resolved before optimizing the model architecture. Regularization techniques and model capacity analysis have proven to be quite helpful in limiting over-fitting and under-fitting issues. Finally, exploring different loss functions and optimization strategies that better address class imbalance issues are equally important to achieve acceptable performance.

For further reading, I suggest exploring textbooks and technical reports on the specific topics of medical image analysis and computer vision. Papers and books focused on semantic segmentation techniques, evaluation metrics, and model optimization are invaluable. A comprehensive understanding of common issues specific to these domains is essential to improve performance in complex image segmentation tasks.
