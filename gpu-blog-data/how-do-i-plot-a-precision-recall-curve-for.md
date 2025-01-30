---
title: "How do I plot a precision-recall curve for a segmentation model?"
date: "2025-01-30"
id: "how-do-i-plot-a-precision-recall-curve-for"
---
Precision-recall curves offer a crucial performance assessment for segmentation models, particularly when dealing with imbalanced datasets where overall accuracy might be misleading. Their construction involves systematically varying the decision threshold of the model's output and subsequently calculating precision and recall values at each threshold. Understanding this process is essential for selecting an appropriate operational point for your model.

The underlying concept relies on the model producing a probability map indicating the likelihood of each pixel belonging to a specific class (e.g., the object of interest). Instead of directly outputting a binary mask (segmented or not segmented), most deep learning segmentation models generate these continuous probability maps. Creating a precision-recall curve requires converting this continuous output into binary predictions using a series of thresholds. For each threshold, we classify a pixel as belonging to the object if its probability exceeds that threshold. The process then involves calculating precision and recall using the obtained binary predictions in comparison with ground truth labels. The precision is calculated as the ratio of true positives (pixels correctly classified as object) to the total number of pixels predicted as object (true positives + false positives). Recall, on the other hand, is calculated as the ratio of true positives to the total number of actual object pixels (true positives + false negatives).

A critical aspect of this process is the choice of thresholds. In practice, thresholds usually range from 0 to 1, covering the entire probability space. The precision and recall values are then plotted against each other, creating the precision-recall curve. The shape of the curve provides valuable insights: an ideal model would achieve high precision and high recall simultaneously, represented by a curve that hugs the top-right corner of the plot. Curves closer to the top-right region generally indicate better performance.

Now, let's consider a practical example. I have worked on a project involving a U-Net architecture for segmenting lung nodules from CT scans. The output of my U-Net is a probability map where each pixel value indicates the likelihood of that pixel belonging to a nodule. To generate the precision-recall curve, the following steps are crucial, usually achieved through a dedicated evaluation script:

First, the model predictions and corresponding ground truth masks must be loaded. Suppose we are working with a dataset composed of 100 images. Let's assume that `model_predictions` represents the probability maps obtained from the model for those images and `ground_truth_masks` holds the corresponding binary ground truth masks for the nodules. Both can be stored as NumPy arrays. I also assume the functions `calculate_precision` and `calculate_recall` are already defined according to their mathematical definitions.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

def calculate_precision(y_true, y_pred):
  """Calculates precision for binary classification.

  Args:
    y_true: Ground truth labels (binary).
    y_pred: Predicted labels (binary).

  Returns:
      Precision value.
  """
  return precision_score(y_true.flatten(), y_pred.flatten())

def calculate_recall(y_true, y_pred):
  """Calculates recall for binary classification.

  Args:
    y_true: Ground truth labels (binary).
    y_pred: Predicted labels (binary).

  Returns:
      Recall value.
  """
  return recall_score(y_true.flatten(), y_pred.flatten())

#Assume model_predictions and ground_truth_masks are loaded NumPy arrays
def generate_precision_recall_curve(model_predictions, ground_truth_masks):
  thresholds = np.linspace(0, 1, 100)
  precisions = []
  recalls = []

  for threshold in thresholds:
    binary_predictions = (model_predictions > threshold).astype(int)
    precisions_at_threshold = []
    recalls_at_threshold = []

    for i in range(len(model_predictions)):
       precision = calculate_precision(ground_truth_masks[i], binary_predictions[i])
       recall = calculate_recall(ground_truth_masks[i], binary_predictions[i])
       precisions_at_threshold.append(precision)
       recalls_at_threshold.append(recall)


    precisions.append(np.mean(precisions_at_threshold))
    recalls.append(np.mean(recalls_at_threshold))


  return precisions, recalls, thresholds
```

In the above example, we iterate through a range of thresholds from 0 to 1. At each threshold, we generate binary predictions by setting all probabilities above the threshold to 1, otherwise 0. Subsequently, we compute precision and recall for each image, then average these values across all images, appending them to `precisions` and `recalls` respectively.

Secondly, after obtaining the precision and recall values, these are often used to derive the area under the precision-recall curve (AUPRC), a single metric that quantifies the overall performance of the segmentation model.

```python
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def plot_precision_recall_curve(precisions, recalls, thresholds):
  auc_score = auc(recalls, precisions)
  plt.plot(recalls, precisions, marker='.')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title(f'Precision-Recall Curve (AUC = {auc_score:.2f})')
  plt.grid(True)
  plt.show()
  return auc_score

precisions, recalls, thresholds = generate_precision_recall_curve(model_predictions, ground_truth_masks)
auc_score = plot_precision_recall_curve(precisions, recalls, thresholds)

print(f"Area Under the Precision Recall Curve: {auc_score}")
```

This code snippet demonstrates how to leverage the `auc` function from `sklearn.metrics` to calculate the AUPRC and plotting the curve using matplotlib. The returned `auc_score` provides a convenient single number that summarizes the performance of the segmentation model.

Finally, to make the process computationally feasible, particularly for larger datasets, it's beneficial to vectorize operations. This reduces the computational overhead, enhancing the efficiency of precision-recall curve construction. In this case, the per-image iteration can be removed, and we can reshape model predictions and ground truth masks into a single large matrix.

```python
def generate_precision_recall_curve_vectorized(model_predictions, ground_truth_masks):
  thresholds = np.linspace(0, 1, 100)
  precisions = []
  recalls = []
  # Reshape to combine all the images into a single array
  model_predictions_flat = np.concatenate([x.flatten() for x in model_predictions])
  ground_truth_masks_flat = np.concatenate([x.flatten() for x in ground_truth_masks])

  for threshold in thresholds:
    binary_predictions = (model_predictions_flat > threshold).astype(int)
    precision = calculate_precision(ground_truth_masks_flat, binary_predictions)
    recall = calculate_recall(ground_truth_masks_flat, binary_predictions)
    precisions.append(precision)
    recalls.append(recall)

  return precisions, recalls, thresholds
```

The `generate_precision_recall_curve_vectorized` function showcases this approach. By flattening all image predictions and ground truth masks into single arrays before the loop, we can leverage `calculate_precision` and `calculate_recall` more efficiently, as each call now operates on the entire dataset instead of processing images individually, making the processing faster. The plot_precision_recall_curve function remains the same for vectorized and non-vectorized implementations.

In conclusion, plotting precision-recall curves for segmentation models involves systematically varying thresholds applied to the output probability maps, calculating precision and recall at each threshold, and finally plotting these values against each other. The vectorization provides an efficient approach for large datasets. The AUPRC provides a comprehensive score that condenses this curve into a single numeric evaluation.

For further reading, I would recommend exploring materials focusing on performance evaluation for binary classification models. Resources outlining common metrics such as precision, recall, and the F1-score are invaluable. Additionally, any text focusing on the analysis of imbalanced datasets provides a strong context for precision-recall curves, which are typically more suitable than accuracy metrics when dealing with such datasets. Finally, exploring materials specific to performance analysis in medical image analysis will offer valuable context within that specialized domain.
