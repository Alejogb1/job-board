---
title: "How do I calculate metrics for multi-label edge segmentation in PyTorch?"
date: "2025-01-30"
id: "how-do-i-calculate-metrics-for-multi-label-edge"
---
Multi-label edge segmentation, unlike its single-label counterpart, necessitates careful consideration of the partially correct predictions arising from the potential presence of multiple edge types within a single pixel. Standard binary metrics, such as Intersection over Union (IoU) or F1 score, when applied naively across all labels, fail to capture the nuanced performance of such models. A more appropriate approach involves evaluating metrics *per label* and subsequently aggregating these values into a single, comprehensive performance score, or examining individual label performance to understand areas of model strength and weakness.

During my time working on a medical image analysis project involving the detection of anatomical structures’ boundaries, I encountered the exact challenge of multi-label edge segmentation. We were tasked with delineating multiple distinct tissue interfaces – muscle/bone, fat/muscle, and so forth – from MRI scans. Simple pixel-wise comparisons against ground truth segmentation masks proved inadequate as edge pixels often simultaneously belonged to several categories. For example, a single pixel at the edge of a muscle might also be at the edge of a fat tissue, requiring a prediction of both edge types to be considered correct.

To quantify the model's performance, I used a combination of per-label metrics followed by averaging across labels. This approach involved three key steps: calculating the confusion matrix for *each* edge type, deriving label-specific IoU scores, and finally computing an average across all label IoUs as a summary metric.

The core process hinges on generating a confusion matrix for each label independently. In multi-label segmentation, this implies that a pixel is considered as positive if the ground truth indicates the presence of that specific edge and its corresponding prediction also indicates the presence of that specific edge. Similarly, a pixel is considered as a true negative if it’s absent in both the ground truth and prediction for the current edge type. False positives are pixels where that edge type is predicted but is not actually present, and false negatives are pixels where the edge type is present but was not predicted.

```python
import torch
import numpy as np

def calculate_confusion_matrix(pred_mask, gt_mask, num_labels):
    """Calculates confusion matrices for each label.

    Args:
        pred_mask (torch.Tensor): Predicted segmentation mask [batch, height, width].
        gt_mask (torch.Tensor): Ground truth segmentation mask [batch, height, width, num_labels].
        num_labels (int): Number of edge labels.

    Returns:
        torch.Tensor: Stack of confusion matrices [num_labels, 2, 2].
    """
    batch_size = pred_mask.size(0)
    height = pred_mask.size(1)
    width = pred_mask.size(2)
    confusion_matrices = torch.zeros((num_labels, 2, 2), dtype=torch.int64)

    for label_idx in range(num_labels):
        gt_mask_label = gt_mask[:, :, :, label_idx].bool()
        pred_mask_label = pred_mask == (label_idx + 1)  # Assuming labels start from 1
        for batch_idx in range(batch_size):
            for i in range(height):
                for j in range(width):
                  if gt_mask_label[batch_idx, i, j]:
                    if pred_mask_label[batch_idx, i, j]:
                      confusion_matrices[label_idx, 1, 1] += 1  # True Positive
                    else:
                      confusion_matrices[label_idx, 1, 0] += 1 # False Negative
                  else:
                     if pred_mask_label[batch_idx, i, j]:
                       confusion_matrices[label_idx, 0, 1] += 1 # False Positive
                     else:
                        confusion_matrices[label_idx, 0, 0] += 1 # True Negative

    return confusion_matrices
```

In this function, `pred_mask` contains the predicted labels, assuming an integer representation for different labels (e.g., 1 for edge type 1, 2 for edge type 2, etc.). The `gt_mask` is a tensor of shape \[batch, height, width, num\_labels] where each channel represents the presence or absence of a specific edge type at every pixel.  I iterate through each pixel in the batch, check if that specific label is present in the ground truth and if it is also predicted by the model for that pixel. The confusion matrix values are accumulated, representing true positives, false positives, false negatives, and true negatives for each specific edge label.

Following confusion matrix computation, label-specific IoU scores can be generated:

```python
def calculate_iou_per_label(confusion_matrices):
    """Calculates IoU for each label given the confusion matrices.

     Args:
        confusion_matrices (torch.Tensor): Stack of confusion matrices [num_labels, 2, 2].

    Returns:
        torch.Tensor: IoU scores for each label [num_labels].
    """
    num_labels = confusion_matrices.size(0)
    iou_scores = torch.zeros(num_labels)
    for label_idx in range(num_labels):
        tp = confusion_matrices[label_idx, 1, 1]
        fp = confusion_matrices[label_idx, 0, 1]
        fn = confusion_matrices[label_idx, 1, 0]
        iou = tp / (tp + fp + fn + 1e-8) #Adding epsilon to avoid division by 0
        iou_scores[label_idx] = iou
    return iou_scores
```

This function takes the previously computed confusion matrices and calculates IoU for each label independently, handling potential divide by zero issues with a small epsilon.

Lastly, to provide a single metric for overall performance, an average IoU can be calculated from per-label IoUs:

```python
def calculate_average_iou(iou_scores):
    """Calculates the average IoU across all labels.

    Args:
        iou_scores (torch.Tensor): IoU scores for each label [num_labels].

    Returns:
        float: Average IoU score.
    """
    return iou_scores.mean().item()
```

This function calculates and returns the mean of the per-label IoU scores, providing a single performance metric. I found it useful to inspect each label IoU independently to determine which edges the model performed best and worst on, as this guides the refinement of training or preprocessing strategies.

A few practical considerations warrant attention. First, during the development of the code, I encountered issues with computational overhead when working with very large image resolutions.  In such scenarios, it is beneficial to consider sub-sampling, or calculating metrics over patches rather than full images during the validation process. Second, it’s crucial to be aware of class imbalance in multi-label segmentation datasets. The presence of certain edge types might be significantly rarer than others. In my experience, a weighted average based on label frequency can provide a more robust performance assessment by preventing common classes from dominating the evaluation. Finally, it’s valuable to augment traditional metrics such as IoU with others, like Dice scores or Hausdorff distance (when shape and boundary precision are crucial), in order to obtain a more complete understanding of segmentation quality.

For deeper understanding of segmentation evaluation practices, I recommend referring to research publications on image segmentation metrics, specifically those dealing with medical image analysis.  Reviewing relevant articles on multi-label classification and its evaluation will further clarify nuances in the topic. Additionally, consult the PyTorch documentation, paying special attention to its implementation of basic tensor operations, especially those useful for comparison and logical indexing. Finally, examining open-source repositories featuring segmentation models, with a focus on the evaluation methods, offers invaluable real-world perspective. Combining this knowledge with targeted practical experimentation is the key for a robust metric implementation.
