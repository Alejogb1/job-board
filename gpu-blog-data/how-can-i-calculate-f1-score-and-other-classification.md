---
title: "How can I calculate F1-score and other classification metrics for a Faster R-CNN object detection model in PyTorch?"
date: "2025-01-30"
id: "how-can-i-calculate-f1-score-and-other-classification"
---
The inherent challenge with evaluating Faster R-CNN and similar object detection models stems from their dual-output nature: they predict both bounding boxes and class labels. Unlike single-output classification models where metrics like accuracy are directly applicable, object detection requires metrics that consider spatial overlap and classification correctness simultaneously. This often leads to the need to aggregate results across multiple detections within a single image.

I've frequently encountered this issue while developing custom defect detection systems, where pinpointing the location of a flaw is as critical as classifying its type. To address this, I’ve found it essential to decompose the problem into its component parts before calculating a global F1-score. Specifically, we need to first establish the correspondence between predicted and ground-truth boxes based on intersection-over-union (IoU), and then use these correspondences to calculate traditional classification metrics like precision, recall, and ultimately, F1-score.

Let's begin with a clear explanation of the steps involved. The process typically involves: 1) Calculating IoU between each predicted box and every ground truth box; 2) Assigning predictions to ground truths based on the IoU threshold and avoiding multiple predictions being assigned to the same ground truth; 3) Determining true positives, false positives, and false negatives based on those assignments; 4) Calculating precision and recall, and finally deriving F1-score. Note that there’s no one single F1-score value per image for object detection. Instead, it’s often necessary to compute class-wise F1 scores and report their average to obtain a final value, or more typically to look at average precision (AP) which also considers precision/recall curve.

Here's a breakdown with code examples. First, consider a helper function to calculate the IoU:

```python
import torch

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (torch.Tensor): A tensor representing bounding box 1 [x1, y1, x2, y2].
        box2 (torch.Tensor): A tensor representing bounding box 2 [x1, y1, x2, y2].

    Returns:
        float: The IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0
```

This `calculate_iou` function, which I often include in my detection utilities, takes two bounding boxes, represented as [x1, y1, x2, y2] coordinates, and computes their intersection over union. This forms the fundamental building block for matching predicted boxes with ground truth. Note that I have implemented this with `torch.Tensor` which is needed when working with a PyTorch model.

The next crucial step is to assign predictions to ground truths based on the calculated IoUs. We aim to maximize the matches, with each ground truth only corresponding to at most one prediction. Here’s how one might achieve this:

```python
def match_predictions(predictions, ground_truths, iou_threshold):
    """Matches predictions to ground truths based on IoU.

    Args:
        predictions (list of dict): List of prediction dictionaries, each containing 'boxes', 'labels'.
        ground_truths (list of dict): List of ground truth dictionaries, each containing 'boxes', 'labels'.
        iou_threshold (float): The IoU threshold for matching.

    Returns:
        list: A list of tuples (prediction_index, ground_truth_index) that indicate matched pairs
    """

    matches = []
    used_ground_truths = set()

    for p_idx, prediction in enumerate(predictions):
        best_iou = 0
        best_gt_idx = None

        for gt_idx, ground_truth in enumerate(ground_truths):
            if gt_idx in used_ground_truths:
                continue
            for p_box in prediction['boxes']:
                for gt_box in ground_truth['boxes']:
                    iou = calculate_iou(p_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
        
        if best_iou > iou_threshold:
            matches.append((p_idx, best_gt_idx))
            used_ground_truths.add(best_gt_idx)

    return matches
```

This `match_predictions` function takes lists of predictions and ground truths, iterates over the prediction boxes, computing IoU for each against all available ground truths, and finally assigns matches if they cross the pre-defined IoU threshold. Note that this example code is assuming that all the boxes, predictions, and ground truths are in the same image. In a more realistic scenario, you would need to have image specific labels and predictions, and iterate over each image.

Finally, with matches established, we can determine true positives (TP), false positives (FP), and false negatives (FN) and calculate precision, recall and the F1-score. Here’s an implementation:

```python
def calculate_f1_score(predictions, ground_truths, iou_threshold):
   """Calculates the F1-score for object detection.

    Args:
        predictions (list of dict): List of prediction dictionaries, each containing 'boxes', 'labels'.
        ground_truths (list of dict): List of ground truth dictionaries, each containing 'boxes', 'labels'.
        iou_threshold (float): The IoU threshold for matching.

    Returns:
        dict: A dictionary containing 'f1_scores', 'precision', 'recall' for each class
    """

   matches = match_predictions(predictions, ground_truths, iou_threshold)

   true_positives = {}
   false_positives = {}
   false_negatives = {}
   all_labels = set()

   for p in predictions:
       for label in p['labels']:
           all_labels.add(label)

   for gt in ground_truths:
      for label in gt['labels']:
           all_labels.add(label)

   for label in all_labels:
        true_positives[label] = 0
        false_positives[label] = 0
        false_negatives[label] = 0


   matched_predictions = [p_idx for p_idx, _ in matches]
   for p_idx, g_idx in matches:
         pred_label = predictions[p_idx]['labels'][0] # Assume only one label per box for simplicity
         gt_label = ground_truths[g_idx]['labels'][0] # Assume only one label per box for simplicity

         if pred_label == gt_label:
           true_positives[pred_label] += 1
         else:
             false_positives[pred_label] += 1


   for p_idx, prediction in enumerate(predictions):
         if p_idx not in matched_predictions:
             for label in prediction['labels']:
                 false_positives[label] += 1

   matched_ground_truths = [g_idx for _, g_idx in matches]
   for g_idx, ground_truth in enumerate(ground_truths):
        if g_idx not in matched_ground_truths:
            for label in ground_truth['labels']:
                false_negatives[label] += 1
   
   results = {}
   for label in all_labels:
        precision = true_positives[label] / (true_positives[label] + false_positives[label] + 1e-8)
        recall = true_positives[label] / (true_positives[label] + false_negatives[label] + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        results[label] = {'f1_score': f1, 'precision': precision, 'recall': recall}
   return results
```

This `calculate_f1_score` function consolidates the matching process, determines the classification counts for TP, FP, and FN, and computes class-specific precision, recall and F1 scores.  I have used a small 1e-8 constant to avoid division-by-zero error.  Note the current code is making a lot of assumptions, most notably that there is only one box per prediction and one box per ground truth, and there is only one label per box, which is unlikely in real world datasets.

For effective implementation and deeper understanding, I'd recommend consulting resources such as the COCO evaluation metrics documentation and any introductory text on performance metrics used in classification.  Also explore papers discussing specific object detection evaluation techniques, which often describe average precision (AP) and mean average precision (mAP) as more robust metrics to F1 score, as it accounts for different thresholds and considers the precision recall curve. Finally, understanding the specific intricacies of your dataset and the task you're solving, as well as considering edge cases, is necessary to develop robust and accurate evaluation mechanisms.
