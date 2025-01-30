---
title: "How can I calculate precision and recall for a trained TensorFlow object detection model?"
date: "2025-01-30"
id: "how-can-i-calculate-precision-and-recall-for"
---
Object detection model evaluation, specifically concerning precision and recall, necessitates a thorough understanding of ground truth bounding boxes and their model-predicted counterparts. These metrics, essential for characterizing a model’s detection capabilities, are calculated based on the intersection over union (IoU) between these two sets of bounding boxes, and subsequently considering true positives, false positives, and false negatives. My experience, gained during several years developing computer vision applications for robotic platforms, underscores the criticality of this evaluation phase, especially in safety-critical deployments where accurate object detection is paramount.

Precision quantifies the proportion of predicted bounding boxes that are actually relevant objects, while recall quantifies the proportion of actual objects in the scene that the model successfully identified. Achieving high values in both metrics is usually the primary objective. The calculation process is structured in several stages: first, we compute the IoU for every combination of predicted and ground truth boxes; next, we classify each prediction as either a true positive (TP), false positive (FP), or a false negative (FN) depending on the IoU and detection thresholds; finally, we summarize these classifications to derive precision and recall scores. The inherent complexity stems from handling scenarios where multiple predictions may overlap a single ground truth, requiring careful assignment logic to prevent double counting.

The initial step, IoU calculation, is foundational. Consider bounding boxes represented by their coordinates: (x_min, y_min, x_max, y_max). The IoU, given two bounding boxes B1 and B2, is determined by the following formula: IoU(B1, B2) = Area(B1 ∩ B2) / Area(B1 ∪ B2). Geometrically, the intersection is the area of the overlapping region, while the union is the combined area of both boxes. This is a key step because, without accurate IoU computation, the evaluation metrics would become meaningless.

Here's a Python code example for calculating the IoU between two bounding boxes:

```python
import numpy as np

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: A tuple or list (x_min, y_min, x_max, y_max) representing the first bounding box.
        box2: A tuple or list (x_min, y_min, x_max, y_max) representing the second bounding box.

    Returns:
        The IoU value (float) or 0 if no overlap.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    if intersection_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

# Example Usage
box_a = (10, 10, 100, 100)
box_b = (50, 50, 150, 150)
iou_score = calculate_iou(box_a, box_b)
print(f"IoU between box_a and box_b is: {iou_score:.2f}")
```

In this code example, `calculate_iou` function takes two bounding boxes as input, computes the coordinates of the intersection rectangle, calculates intersection and union areas, and returns the IoU. The `max(0, ...)` ensures a zero intersection if the boxes don't overlap, preventing errors. The example demonstrates how to call the function and print the resulting IoU value.

Once IoU is computed for all pairs of predicted and ground truth boxes, a threshold, generally set around 0.5, determines whether a prediction is a true positive. If the IoU is above this threshold, and assuming each ground truth can be assigned at most one prediction, it's classified as a true positive. If no prediction for a ground truth box achieves an IoU above the threshold, that ground truth is considered a false negative. Predictions with an IoU below the threshold, or unmatched predictions, become false positives. The assignment process becomes more nuanced when dealing with multiple predictions for each object, where a greedy approach using the highest IoU score is often used.

The next code snippet exemplifies the classification of predicted boxes into true positives, false positives, and false negatives, given a set of bounding boxes and an IoU threshold:

```python
def classify_detections(predictions, ground_truths, iou_threshold):
    """Classifies detections into true positives, false positives, and false negatives.

    Args:
        predictions: A list of lists, each inner list being [confidence, x_min, y_min, x_max, y_max].
        ground_truths: A list of lists, each inner list being [x_min, y_min, x_max, y_max].
        iou_threshold: The IoU threshold for determining true positives.

    Returns:
      A tuple: (true_positives, false_positives, false_negatives) - lists of indices
    """
    true_positives = []
    false_positives = []
    false_negatives = list(range(len(ground_truths))) # Initially assume all GTs are false negatives
    
    assigned_ground_truths = [False] * len(ground_truths)

    for i, prediction in enumerate(predictions):
        best_iou = 0
        best_match = -1
        
        for j, gt in enumerate(ground_truths):
            iou = calculate_iou(prediction[1:], gt) # Use only bounding box coords, not confidence
            if iou > best_iou:
                best_iou = iou
                best_match = j

        if best_iou >= iou_threshold:
             if not assigned_ground_truths[best_match]:
                 true_positives.append(i)
                 assigned_ground_truths[best_match] = True
                 if best_match in false_negatives:
                     false_negatives.remove(best_match)

             else:
                 false_positives.append(i)
        else:
           false_positives.append(i)
    
    return true_positives, false_positives, false_negatives


#Example Usage
predictions_example = [[0.9, 10, 10, 100, 100], [0.7, 50, 50, 150, 150], [0.6, 200,200,300,300]]
ground_truths_example = [[15, 15, 110, 110], [45, 45, 160, 160], [400, 400, 500, 500]]
iou_threshold_example = 0.5

true_pos, false_pos, false_neg = classify_detections(predictions_example, ground_truths_example, iou_threshold_example)

print(f"True Positives: {true_pos}")
print(f"False Positives: {false_pos}")
print(f"False Negatives: {false_neg}")
```

This function, `classify_detections`, iterates through all predictions, calculating the best IoU match for each against all ground truths, applying assignment rules (a prediction matched to a GT is a TP), and accounting for duplicate assignments by tracking used GTs.  False negatives are assigned to unmatched ground truths in an initialized list. The example demonstrates that indices for each class are provided after classification.  This step transforms the geometric problem of bounding box overlap into a set of class labels, which can then be summarized to produce evaluation metrics.

With true positives, false positives, and false negatives calculated, precision and recall are computed as follows: Precision = TP / (TP + FP) and Recall = TP / (TP + FN). Precision represents the proportion of correct positive identifications out of all positive identifications (i.e. including those that are incorrect), while recall represents the proportion of correctly identified positives out of all existing positive ground truths. A high precision, coupled with a high recall, signifies a robust model. It is important to note that precision and recall can be summarized for a specific threshold value and averaged over different threshold values to produce mean average precision, but this is beyond the scope of the response.

Finally, let us generate these evaluation metrics with this final code example:

```python
def calculate_precision_recall(true_positives, false_positives, false_negatives, ground_truth_count):
    """Calculates precision and recall.

    Args:
        true_positives: A list of true positive indices.
        false_positives: A list of false positive indices.
        false_negatives: A list of false negative indices.
         ground_truth_count: Count of all ground truths.

    Returns:
        A tuple: (precision, recall). Both are floats, or None if invalid counts.
    """
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)


    if tp_count + fp_count == 0:
        precision = 0.0 #Avoid division by zero
    else:
        precision = tp_count / (tp_count + fp_count)

    if ground_truth_count == 0:
      recall = 0.0
    elif tp_count + fn_count == 0: #Avoid division by zero
      recall = 0.0
    else:
       recall = tp_count / (tp_count + fn_count)
    
    return precision, recall


#Example Usage (using the outputs from previous example)

precision, recall = calculate_precision_recall(true_pos, false_pos, false_neg, len(ground_truths_example))
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

This `calculate_precision_recall` function takes the TP, FP, FN, and the number of ground truth boxes as input and calculates precision and recall based on the previously discussed formulas, while adding clauses to avoid divide by zero errors when no predictions have been made, or ground truths are absent.  The example shows how these metrics are calculated using the outputs from the previous example.

For further exploration of object detection evaluation metrics, consider referencing resources discussing the Pascal VOC Challenge evaluation criteria. Research materials related to mean average precision (mAP) will also provide additional depth of knowledge. The COCO dataset documentation and associated evaluation metrics are beneficial for those pursuing more contemporary methodologies.
