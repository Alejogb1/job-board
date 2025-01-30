---
title: "What are the main errors in object detection evaluation models?"
date: "2025-01-30"
id: "what-are-the-main-errors-in-object-detection"
---
Object detection evaluation metrics, while seemingly straightforward, are frequently misinterpreted, leading to inaccurate assessments of model performance. I’ve seen this firsthand, troubleshooting issues in our in-house surveillance system, where initial metrics suggested a highly accurate model when, in reality, it struggled with specific object classes and occlusions. The key issue stems from a reliance on overall averages, which often obscure critical failure points. The primary errors can be categorized into issues with metric selection, inappropriate thresholding, and insufficient consideration of class imbalances.

The most common error involves focusing solely on the mean Average Precision (mAP) as the defining metric, especially when the underlying task involves diverse object classes. While mAP provides a holistic view of performance, it averages the Average Precision (AP) across all classes. This can be misleading because a model might achieve high AP for easily detectable objects but perform poorly on those that are more difficult or rare. For example, if we had a system trained to detect cars, pedestrians, and bicycles, a high mAP might mask the fact that it's highly accurate at detecting cars but struggles with the smaller, more variable appearance of bicycles. The individual AP values per class, which highlight these discrepancies, are often overlooked, leading to overly optimistic interpretations. To address this, a thorough evaluation includes analysis of per-class AP, not just the overall average.

Another significant error occurs in the handling of intersection-over-union (IoU) thresholds. IoU measures the overlap between a predicted bounding box and the ground truth bounding box. A high IoU indicates a good detection. However, evaluation typically applies a single fixed IoU threshold (e.g., 0.5 or 0.75) to determine if a prediction is a true positive or false positive. This is problematic because different detection tasks might require different IoU thresholds to be considered "good". Furthermore, the threshold isn't uniformly appropriate across different object classes. Consider that a detection of a small object like a license plate may need a higher IoU to be considered accurate, while a detection of a large object such as a truck might be acceptable with a lower IoU value.

The chosen IoU threshold has a direct impact on the evaluation results. When set too low, a model with imprecise bounding boxes might still appear effective, masking its deficiencies. Conversely, if set too high, even relatively accurate detections could be classified as false positives, leading to a pessimistic view of performance. I have often found that evaluating detection models with IoU thresholds varying across different object classes and then calculating average precision across these thresholds, known as Average Precision across IoUs, provides a more robust picture of model performance.

Finally, ignoring class imbalance leads to skewed evaluation results, particularly when certain classes appear significantly more frequently in the training or validation sets. Models will often learn to favor the majority class. This bias inflates the overall metrics, even if performance on less frequent or underrepresented classes is poor. For instance, in an image dataset containing many car images but few motorbike images, a model might achieve high AP for car detection, while motorbikes are hardly ever correctly localized. Relying solely on mAP, as it is usually calculated, without regard for the imbalanced class distributions will provide an overly optimistic picture of model performance. A proper analysis would require techniques like weighted AP or evaluating with a rebalanced dataset to see how the model performs without the bias of an uneven class distribution.

To illustrate these points, let's consider a few code examples using a fictional evaluation script, keeping in mind these are simplifications for illustrative purposes:

**Example 1: Single IoU Threshold Evaluation**

```python
def evaluate_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Simple evaluation using a single IoU threshold.
    Assumes each box is represented as [x1, y1, x2, y2].
    Returns the calculated precision and recall.
    """

    tp, fp, fn = 0, 0, 0 # True Positive, False Positive, False Negative

    for gt_box in gt_boxes:
        found_match = False
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box) # Function to calculate IoU
            if iou >= iou_threshold:
                tp += 1
                found_match = True
                pred_boxes.remove(pred_box) # Remove to prevent matching same box
                break
        if not found_match:
            fn += 1
    
    fp = len(pred_boxes) # Any remaining pred_boxes are false positives

    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    return precision, recall

# Example usage
gt_boxes_car = [[100,100,200,200],[300,300,400,400]]
pred_boxes_car = [[110,110,210,210],[320,320,420,420]]
precision, recall = evaluate_detections(gt_boxes_car, pred_boxes_car, iou_threshold=0.5)
print(f"Precision: {precision}, Recall: {recall}")

gt_boxes_bicycle = [[50,50,80,80], [250,250,300,300]]
pred_boxes_bicycle = [[52,52,78,78]] # One missed detection
precision, recall = evaluate_detections(gt_boxes_bicycle, pred_boxes_bicycle, iou_threshold=0.5)
print(f"Precision: {precision}, Recall: {recall}")
```

This simplified example showcases a naive evaluation approach with a single IoU threshold. While useful as a basic test, it doesn’t provide insights into varying IoUs or per-class performance differences.

**Example 2: Per-Class AP Calculation (Conceptual)**

```python
def calculate_per_class_ap(ground_truth, predictions, iou_threshold=0.5):
    """
    Conceptual example to highlight class wise calculation.
    Assumes ground_truth and predictions are lists of dictionaries.
    Each dictionary includes:
      - "boxes": list of bounding boxes
      - "class": str representing object class.
    Returns a dict containing AP for each class.
    """
    class_aps = {}
    all_classes = set([item['class'] for item in ground_truth] + [item['class'] for item in predictions])

    for class_name in all_classes:
        gt_boxes_for_class = [item['boxes'] for item in ground_truth if item['class'] == class_name]
        pred_boxes_for_class = [item['boxes'] for item in predictions if item['class'] == class_name]
        if not gt_boxes_for_class:
          print(f"No ground truth for {class_name}")
          class_aps[class_name] = 0
          continue
        if not pred_boxes_for_class:
            print(f"No predictions for {class_name}")
            class_aps[class_name] = 0
            continue
        
        precision, recall = evaluate_detections(gt_boxes_for_class, pred_boxes_for_class, iou_threshold)
        # Simplified AP calculation. In reality should be interpolated precision-recall curve
        class_aps[class_name] = calculate_ap_from_pr(precision, recall) #Function to calculate AP

    return class_aps

#Example usage
ground_truth = [
    {'class': 'car', 'boxes': [[100, 100, 200, 200]]},
    {'class': 'bicycle', 'boxes': [[50,50,80,80]]}
]

predictions = [
    {'class': 'car', 'boxes': [[110, 110, 210, 210]]},
    {'class': 'bicycle', 'boxes': [[52,52,78,78]]}
]
class_aps = calculate_per_class_ap(ground_truth, predictions, iou_threshold=0.5)
print(f"Per class Average Precision: {class_aps}")
```

This snippet emphasizes the importance of calculating AP separately for each class rather than relying on a single mAP value. It highlights that class-specific performance can vary drastically.

**Example 3: Varying IoU thresholds**

```python
def calculate_ap_across_ious(ground_truth, predictions, iou_thresholds):
    """
    Calculates AP at multiple IoU thresholds and averages.
    Assumes ground_truth and predictions have the format in the previous example
    """
    aps_across_ious = []
    for iou_threshold in iou_thresholds:
        class_aps = calculate_per_class_ap(ground_truth, predictions, iou_threshold)
        aps_across_ious.append(sum(class_aps.values())/len(class_aps))
    
    return sum(aps_across_ious)/len(aps_across_ious)


iou_thresholds = [0.5, 0.75, 0.9]
average_ap_across_ious = calculate_ap_across_ious(ground_truth, predictions, iou_thresholds)
print(f"Average AP across different IoUs: {average_ap_across_ious}")
```

This demonstrates that running evaluations across a range of thresholds creates a more accurate picture of the robustness of the model rather than relying on a single cutoff.

For further study, I would recommend researching literature on object detection metrics, specifically focusing on the COCO evaluation metrics, which provide guidance on more accurate performance analysis, and exploring the concepts of interpolated average precision. The works of Everingham et al. on the PASCAL VOC challenge, as well as the evaluation methodology in the Microsoft COCO dataset, also offer good foundational knowledge. Furthermore, research on handling class imbalance in computer vision is beneficial. Exploring various sampling techniques, such as oversampling and undersampling, and their application to object detection performance evaluation, would also deepen understanding.
