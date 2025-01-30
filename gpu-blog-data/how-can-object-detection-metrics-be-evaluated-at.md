---
title: "How can object detection metrics be evaluated at the class level using the GCP AI Platform API in Python?"
date: "2025-01-30"
id: "how-can-object-detection-metrics-be-evaluated-at"
---
Evaluating object detection model performance at the class level necessitates a nuanced approach beyond simple overall metrics.  My experience integrating custom object detectors with the GCP AI Platform Prediction API highlighted the critical need for granular evaluation, particularly when dealing with imbalanced datasets or models exhibiting class-specific weaknesses.  This requires careful handling of prediction outputs and leveraging the flexibility of Python's data manipulation libraries.


**1.  Clear Explanation:**

The GCP AI Platform Prediction API returns detection results as JSON objects, each containing bounding boxes, confidence scores, and class labels.  Simple aggregate metrics like mean Average Precision (mAP) mask individual class performance.  To assess class-specific performance, we must process these results to compute metrics for each class independently.  This typically involves:

* **Filtering Predictions:** Isolating predictions for a specific class.
* **Ground Truth Matching:** Associating predictions with corresponding ground truth annotations.  This involves considering Intersection over Union (IoU) threshold to determine true positives, false positives, and false negatives.
* **Metric Calculation:** Computing precision, recall, F1-score, and potentially other relevant metrics (e.g., average precision at different IoU thresholds) for each class.

The process relies on the structure of both the prediction output and the ground truth data.  Consistency in data formatting is crucial for automated processing.  I found that creating a custom Python class to represent the detection results and ground truth data significantly improved code readability and maintainability.


**2. Code Examples with Commentary:**

**Example 1:  Custom Data Structure and Prediction Parsing:**

```python
import json

class DetectionResult:
    def __init__(self, image_id, detections):
        self.image_id = image_id
        self.detections = detections  # List of dictionaries (each with 'class_id', 'bbox', 'score')

def parse_prediction(prediction_json):
    """Parses GCP AI Platform prediction JSON into DetectionResult objects."""
    predictions = json.loads(prediction_json)
    image_id = predictions['image_id']  # Assumes 'image_id' is present
    detections = predictions['detections']
    return DetectionResult(image_id, detections)


# Example usage:
prediction_json = '{"image_id": "image1", "detections": [{"class_id": 1, "bbox": [0.1, 0.2, 0.3, 0.4], "score": 0.9}, {"class_id": 2, "bbox": [0.5, 0.6, 0.7, 0.8], "score": 0.7}]}'
result = parse_prediction(prediction_json)
print(result.detections) # Output: list of detections for image1
```

This example defines a `DetectionResult` class to encapsulate prediction data and a function `parse_prediction` to process the JSON output from the GCP AI Platform API.  This structured approach simplifies subsequent processing.  Note that the assumption of `image_id` and `detections` keys in the JSON needs to be adapted based on the API's actual output structure.  Error handling (e.g., for missing keys) should be added for robust production code.

**Example 2:  Ground Truth Matching and Metric Calculation for a Single Class:**

```python
import numpy as np

def calculate_class_metrics(predictions, ground_truth, class_id, iou_threshold=0.5):
    """Calculates precision and recall for a given class."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through predictions for the specified class
    for prediction in predictions:
      if prediction['class_id'] == class_id:
        best_iou = 0
        for gt in ground_truth:
          if gt['class_id'] == class_id:
            iou = calculate_iou(prediction['bbox'], gt['bbox'])
            best_iou = max(best_iou, iou)
        if best_iou >= iou_threshold:
          true_positives += 1
        else:
          false_positives += 1

    #Count false negatives (ground truths without matching predictions)
    for gt in ground_truth:
        if gt['class_id'] == class_id:
            found_match = False
            for prediction in predictions:
                if prediction['class_id'] == class_id and calculate_iou(prediction['bbox'], gt['bbox']) >= iou_threshold:
                    found_match = True
                    break
            if not found_match:
                false_negatives +=1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def calculate_iou(bbox1, bbox2):
    #Helper function to calculate Intersection over Union
    # Assumes bbox format: [xmin, ymin, xmax, ymax]
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou
```

This example demonstrates the core logic of matching predictions to ground truth and calculating precision and recall for a single class. The `calculate_iou` function is a helper to compute the IoU.  This function assumes a specific format for bounding boxes (`[xmin, ymin, xmax, ymax]`).  Adaptation might be needed depending on the format used in your ground truth data and API output.

**Example 3:  Iterating Through Classes and Aggregating Results:**

```python
def evaluate_class_metrics(predictions, ground_truths, class_ids, iou_threshold=0.5):
    results = {}
    for class_id in class_ids:
        class_predictions = [p for p in predictions if p['class_id'] == class_id]
        class_ground_truths = [gt for gt in ground_truths if gt['class_id'] == class_id]
        precision, recall = calculate_class_metrics(class_predictions, class_ground_truths, class_id, iou_threshold)
        results[class_id] = {'precision': precision, 'recall': recall}
    return results


# Example Usage (assuming predictions and ground_truths are lists of dictionaries):
class_ids = [1, 2, 3] #List of the classes to evaluate
results = evaluate_class_metrics(predictions, ground_truths, class_ids)
print(results)
```

This example shows how to iterate through a list of class IDs, applying the per-class metric calculation from Example 2.  The function returns a dictionary containing precision and recall for each class.


**3. Resource Recommendations:**

For in-depth understanding of object detection metrics, I recommend studying relevant publications on the topic of evaluation metrics in object detection.  Thorough familiarity with NumPy for efficient array operations is essential.  Additionally, mastering JSON manipulation in Python is crucial for effective handling of API responses.  Finally, a strong understanding of the underlying principles of precision, recall, F1-score, and Intersection over Union is fundamental to interpreting the results.  Familiarity with Python's standard library, particularly with the `json` module is also critical.
