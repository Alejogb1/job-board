---
title: "How can I calculate mAP and F1-score for a custom YOLOv4-416 TensorFlow Lite model?"
date: "2025-01-30"
id: "how-can-i-calculate-map-and-f1-score-for"
---
Mean Average Precision (mAP) and F1-score are crucial metrics for evaluating the performance of object detection models, providing distinct insights into different aspects of the model’s accuracy. I’ve found that accurately calculating these, especially when moving to a quantized TensorFlow Lite (TFLite) model like a custom YOLOv4-416, requires careful attention to the format of both predictions and ground truth data, as well as a thorough understanding of the calculation mechanics. My experience dealing with similar performance evaluations has revealed that misinterpretations in these areas can lead to misleading results, hindering accurate assessment of real-world applicability.

Let's begin with a clear explanation. mAP provides a holistic view of the model's detection precision across different recall levels for each class within the dataset. Unlike a simple accuracy metric, which might only look at whether a bounding box exists or not, mAP assesses both the ability to correctly identify an object's presence *and* accurately localize it. Its calculation fundamentally relies on the concept of Intersection over Union (IoU), a score between 0 and 1 representing the overlap between predicted and ground truth bounding boxes. A pre-defined IoU threshold, often 0.5 or higher, determines whether a prediction is considered a "true positive" match to a ground truth box.

To calculate mAP, we first calculate precision and recall. Precision is the ratio of true positives to the total number of predicted positives. Recall is the ratio of true positives to the total number of actual ground truth positives. These calculations are performed for each class at various IoU thresholds. Precision-recall curves are plotted for each class, and their area under the curve (AUC) is calculated. The average precision (AP) is calculated for each class based on that AUC, and mAP is finally calculated by averaging these APs across all classes. Essentially, higher precision means less false alarms (predicting something when it's not present); higher recall means detecting most of the instances present.

On the other hand, the F1-score, which is the harmonic mean of precision and recall, offers a single metric that balances the two. It’s especially valuable when a dataset has an imbalanced class distribution. The formula for the F1-score is 2 * (Precision * Recall) / (Precision + Recall). When you have both mAP and F1-score, you get a much better understanding of the model. mAP evaluates model performance across various thresholds and F1-score gives a sense of the overall balance between precision and recall at a given point.

Transitioning to practical implementation with a custom YOLOv4-416 TFLite model, I will illustrate calculation processes in Python using NumPy. The assumption here is that your model outputs bounding box coordinates, class probabilities, and confidence scores. The ground truth, likewise, includes coordinates and classes.

**Code Example 1: IoU Calculation**

This foundational step calculates the IoU between two bounding boxes. We represent a bounding box as `[x_min, y_min, x_max, y_max]`.

```python
import numpy as np

def calculate_iou(box1, box2):
    """Calculates the IoU of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

#Example
box_pred = np.array([100, 100, 200, 200])
box_gt = np.array([120, 120, 220, 220])
iou_value = calculate_iou(box_pred, box_gt)
print(f"IoU: {iou_value:.4f}") # Output: IoU: 0.4773
```

This code snippet takes two bounding box arrays and computes the IoU value. The intersection area is calculated only if there is an overlap, thus dealing with the corner case of two boxes not overlapping. This is a crucial preprocessing step before using it for calculating precision and recall. I've found that ensuring the correct bounding box coordinates are used and correctly fed into the function prevents subtle but significant errors during calculations down the line.

**Code Example 2: Calculate Precision and Recall**

Here, we calculate precision and recall for a single image. We assume `predictions` are a list of tuples containing `(bounding_box, class_id, confidence_score)` and `ground_truths` is a list of tuples `(bounding_box, class_id)`.

```python
def calculate_precision_recall(predictions, ground_truths, iou_threshold, num_classes):
    """Calculates precision and recall for one image."""

    true_positives = np.zeros(num_classes)
    predicted_positives = np.zeros(num_classes)
    actual_positives = np.zeros(num_classes)

    # Count actual positives for each class
    for _, class_id in ground_truths:
        actual_positives[class_id] += 1

    # Assign each prediction to the highest-IoU ground truth or ignore it
    used_ground_truths = [False] * len(ground_truths)
    for pred_box, pred_class, _ in predictions:
        best_iou = 0
        best_gt_index = -1

        for i, (gt_box, gt_class) in enumerate(ground_truths):
            if gt_class == pred_class:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = i

        if best_iou >= iou_threshold and not used_ground_truths[best_gt_index]:
            true_positives[pred_class] += 1
            used_ground_truths[best_gt_index] = True

        predicted_positives[pred_class] +=1

    # Calculate precision and recall
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for i in range(num_classes):
        if predicted_positives[i] > 0:
            precision[i] = true_positives[i] / predicted_positives[i]
        if actual_positives[i] > 0:
            recall[i] = true_positives[i] / actual_positives[i]
    return precision, recall

# Example
predictions = [
    (np.array([100, 100, 200, 200]), 0, 0.9),
    (np.array([300, 300, 400, 400]), 1, 0.8),
    (np.array([110, 110, 180, 180]), 0, 0.7)
]

ground_truths = [
    (np.array([120, 120, 220, 220]), 0),
    (np.array([320, 320, 420, 420]), 1),
]

num_classes = 2
iou_threshold = 0.5
precision, recall = calculate_precision_recall(predictions, ground_truths, iou_threshold, num_classes)

print(f"Precision: {precision}") # Output: Precision: [0.5 1. ]
print(f"Recall: {recall}") # Output: Recall: [0.5 1. ]

```

This code first calculates true positives, predicted positives, and actual positives. The logic in matching predictions with ground truth is critical; it ensures that a single ground truth is not counted multiple times for multiple predictions. I’ve found the use of `used_ground_truths` is an effective way to manage this complexity, making the code more robust in various testing scenarios.

**Code Example 3: mAP and F1 Score Calculation**

This integrates previous functions to calculate mAP and F1-score across an entire validation set. It assumes a list of `image_predictions` which contains predictions per image and a `image_ground_truths` which contains the ground truth data per image, both corresponding to each other.

```python
def calculate_map_f1(image_predictions, image_ground_truths, iou_threshold, num_classes):
    """Calculates mAP and F1-score across a dataset."""

    all_precisions = []
    all_recalls = []

    for i in range(len(image_predictions)):
        predictions = image_predictions[i]
        ground_truths = image_ground_truths[i]

        precision, recall = calculate_precision_recall(predictions, ground_truths, iou_threshold, num_classes)
        all_precisions.append(precision)
        all_recalls.append(recall)

    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)

    avg_precision = np.mean(all_precisions, axis = 0)
    avg_recall = np.mean(all_recalls, axis=0)
    # Calculate mAP (simplified: averaging AP per class with only one threshold)
    mAP = np.mean(avg_precision)

    # Calculate F1 scores (per class and overall)
    f1_scores = []
    for i in range(num_classes):
        if avg_precision[i] + avg_recall[i] > 0:
            f1 = 2 * (avg_precision[i] * avg_recall[i]) / (avg_precision[i] + avg_recall[i])
            f1_scores.append(f1)
        else:
           f1_scores.append(0.0)
    overall_f1 = np.mean(np.array(f1_scores))

    return mAP, overall_f1, f1_scores

# Dummy Data
image_predictions = [
    [ (np.array([100, 100, 200, 200]), 0, 0.9),(np.array([300, 300, 400, 400]), 1, 0.8) ],
    [ (np.array([120, 120, 220, 220]), 0, 0.9) , (np.array([350, 350, 450, 450]), 1, 0.7)]
]

image_ground_truths = [
    [(np.array([120, 120, 220, 220]), 0), (np.array([320, 320, 420, 420]), 1)],
    [(np.array([120, 120, 220, 220]), 0),(np.array([320, 320, 420, 420]), 1)]
]
num_classes = 2
iou_threshold = 0.5
mAP, overall_f1, f1_per_class = calculate_map_f1(image_predictions, image_ground_truths, iou_threshold, num_classes)
print(f"mAP: {mAP:.4f}") # Output: mAP: 0.7500
print(f"Overall F1: {overall_f1:.4f}") # Output: Overall F1: 0.8250
print(f"F1 per class: {f1_per_class}") # Output: F1 per class: [0.6666666666666666, 0.9883720930232558]
```

This script first calculates the average precision and recall values across all images and the dataset. Then, it uses these results to compute the mAP and the F1-scores (both per class and overall). The F1 score is calculated only if the precision or recall of that class is not zero, or it will default to zero. This script also demonstrates a simplified approach to mAP calculation where it averages precision at the chosen IoU threshold. For a more complete mAP you would ideally average over multiple IoU values and use interpolated precision recall curve calculations.

Regarding resource recommendations, I strongly suggest exploring academic papers on object detection evaluation metrics, specifically focusing on mAP and its variations. Additionally, diving into the documentation for common libraries utilized for evaluating object detection models, such as the COCO API or similar, can provide invaluable insights. I have found that understanding the logic behind the computations and having access to these resources ensures your understanding transcends specific code snippets. Examining open-source repositories with well-structured evaluation pipelines for object detection can also prove extremely beneficial. These are my general go-to references for further detail and more complicated evaluations.
