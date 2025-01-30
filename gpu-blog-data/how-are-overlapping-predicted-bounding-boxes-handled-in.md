---
title: "How are overlapping predicted bounding boxes handled in ObjectDetectionEvaluator() output?"
date: "2025-01-30"
id: "how-are-overlapping-predicted-bounding-boxes-handled-in"
---
Bounding box overlap in object detection, a common occurrence, is primarily addressed by Non-Maximum Suppression (NMS) within the evaluation process, which is implicitly handled when using `ObjectDetectionEvaluator()`. This process aims to select the most confident and accurate bounding box for each detected object while discarding highly overlapping, redundant predictions. Understanding how NMS operates within the evaluator is crucial for interpreting evaluation metrics correctly and for tuning model performance. The evaluator itself doesn't expose NMS as a parameter; rather, it is a fundamental component applied to the raw output of your object detection model before evaluation metrics are calculated.

I've spent considerable time debugging evaluation pipelines in various object detection projects, and while the specifics can vary depending on the underlying library, the core logic remains consistent. Before I describe the NMS process embedded in `ObjectDetectionEvaluator()`, let's establish a clear understanding of the general workflow. Your model will generate a set of predicted bounding boxes, each with an associated confidence score and a predicted class label. These bounding boxes are represented by coordinates (typically x_min, y_min, x_max, y_max) relative to the input image. Before these boxes are evaluated against the ground truth, they are often subjected to NMS. The underlying evaluator, whether in TensorFlow, PyTorch, or other frameworks, applies NMS implicitly, as this process is essential to calculate precision and recall correctly.

The core idea behind NMS is iterative suppression. Let's consider a single object class to keep the explanation clear. First, predicted boxes are sorted by their confidence scores in descending order. The box with the highest score is kept as the initial detection. Then, we iterate through the remaining boxes. For each remaining box, we calculate its intersection over union (IoU) with the current highest scoring box. If the calculated IoU exceeds a predefined threshold, the current box is suppressed (discarded), as this indicates significant overlap with a more confident prediction. This process is repeated, selecting the next highest scoring box and suppressing any highly overlapping boxes, until all candidate boxes are either selected or suppressed. This iterative procedure leads to a final set of non-overlapping bounding boxes, ready to be compared to ground truth.

Now, consider how this directly relates to the output of `ObjectDetectionEvaluator()`. When you feed the evaluator your model’s raw bounding box predictions and corresponding ground truth bounding boxes, the evaluator does not calculate metrics on your model’s raw predictions directly. Internally, NMS is applied to these raw predictions based on the specific implementation of the evaluator within the utilized framework. It generates a refined set of predicted bounding boxes, without exposing these internal steps. Therefore, the precision, recall, and other metrics calculated by the evaluator are computed based on this *refined*, non-overlapping set of predictions, not the initial, potentially redundant bounding boxes.

Let’s look at some conceptual code examples to illustrate the impact of NMS. These are Python based pseudocode using NumPy array, simulating how NMS would affect bounding boxes within the evaluator's implicit operation.

**Example 1: Illustrating NMS Functionality**

```python
import numpy as np

def compute_iou(box1, box2):
    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x1_max = min(box1[2], box2[2])
    y1_max = min(box1[3], box2[3])
    intersection_area = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - intersection_area
    if union_area == 0:
        return 0  # Handles cases of no overlap.
    return intersection_area / union_area

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        keep_boxes.append(boxes[current_index])
        sorted_indices = sorted_indices[1:]
        indices_to_remove = []
        for i, index in enumerate(sorted_indices):
            if compute_iou(boxes[current_index], boxes[index]) > iou_threshold:
                indices_to_remove.append(i)
        sorted_indices = np.delete(sorted_indices, indices_to_remove)
    return np.array(keep_boxes)


boxes = np.array([[10, 10, 100, 100],  # Box 1, High Confidence
                 [20, 20, 110, 110],  # Box 2, Medium Confidence, Overlaps Box 1
                 [150, 150, 250, 250], # Box 3, Low Confidence
                 [160, 160, 240, 240]])# Box 4, Lower Confidence, Overlaps Box 3

scores = np.array([0.9, 0.7, 0.4, 0.2])
filtered_boxes = non_max_suppression(boxes,scores)
print("Original Boxes:\n",boxes)
print("NMS Filtered Boxes:\n",filtered_boxes)

```
This code explicitly applies NMS to sample boxes. Box 1 is kept, while Box 2, overlapping and having lower confidence, is removed. The same logic applies to Box 3 and 4 resulting in NMS eliminating the lower confidence box.

**Example 2: Simplified NMS within Evaluation**
```python
import numpy as np
from collections import namedtuple

# Simplified data structure for a predicted box
BoxPrediction = namedtuple('BoxPrediction', ['box', 'score', 'class_id'])

def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):

    # Simplified NMS (as seen in example 1)
    def apply_nms(predictions, iou_threshold):
      #Assume the same non_max_suppression function as in Example 1 is available
       boxes = [prediction.box for prediction in predictions]
       scores = [prediction.score for prediction in predictions]
       keep_boxes = non_max_suppression(np.array(boxes), np.array(scores),iou_threshold)
       filtered_predictions = []
       for original_pred in predictions:
        for k_box in keep_boxes:
             if np.array_equal(original_pred.box,k_box):
               filtered_predictions.append(original_pred)
       return filtered_predictions



    # NMS is implicitly applied during evaluation.
    filtered_predictions = apply_nms(predictions,iou_threshold)

    # In reality, the evaluator would then calculate metrics such as precision, recall, etc.
    # This is a simplified representation.
    print("Filtered Predictions after NMS:")
    for prediction in filtered_predictions:
        print(f"Box: {prediction.box}, Score: {prediction.score}, Class ID: {prediction.class_id}")

# Example Predicted Bounding Boxes
pred_boxes = [
  BoxPrediction(box=np.array([10, 10, 100, 100]), score=0.9, class_id=1),
  BoxPrediction(box=np.array([20, 20, 110, 110]), score=0.7, class_id=1),
  BoxPrediction(box=np.array([150, 150, 250, 250]), score=0.8, class_id=2),
  BoxPrediction(box=np.array([160, 160, 240, 240]), score=0.6, class_id=2)

]

# Simulate ground truth - Not actually used for NMS
ground_truth_boxes = []

evaluate_detections(pred_boxes, ground_truth_boxes)


```
Here we define a simplified evaluation function that simulates the application of NMS before metrics are calculated. Although not calculating metrics, this example clarifies that NMS is applied as an intermediate step before evaluation metrics. This code example also showcases that predictions are associated to class labels.
**Example 3: Impact of IoU Threshold**
```python
import numpy as np
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        keep_boxes.append(boxes[current_index])
        sorted_indices = sorted_indices[1:]
        indices_to_remove = []
        for i, index in enumerate(sorted_indices):
            if compute_iou(boxes[current_index], boxes[index]) > iou_threshold:
                indices_to_remove.append(i)
        sorted_indices = np.delete(sorted_indices, indices_to_remove)
    return np.array(keep_boxes)

boxes = np.array([[10, 10, 100, 100],
                 [20, 20, 110, 110],
                 [150, 150, 250, 250],
                 [160, 160, 240, 240]])
scores = np.array([0.9, 0.7, 0.4, 0.2])


# NMS with a high threshold
filtered_boxes_high_threshold = non_max_suppression(boxes, scores, iou_threshold=0.8)
print("Filtered Boxes High Threshold (IoU = 0.8):\n",filtered_boxes_high_threshold)
# NMS with a low threshold
filtered_boxes_low_threshold = non_max_suppression(boxes, scores, iou_threshold=0.1)
print("Filtered Boxes Low Threshold (IoU = 0.1):\n",filtered_boxes_low_threshold)
```
This final example demonstrates the importance of the NMS `iou_threshold`. With a higher threshold, fewer boxes are suppressed (potentially retaining redundant boxes). A lower threshold results in more aggressive suppression (possibly discarding some valid boxes). The default value of this parameter in `ObjectDetectionEvaluator()` of the underlying frameworks is often a practical value, such as 0.5.

For further reading on object detection and evaluation, I would recommend the following materials. Start by exploring research papers in venues like CVPR and ICCV, which frequently publish foundational works in the area. Look specifically for papers discussing evaluation metrics like Average Precision and the underlying algorithms for object detection. A good practical resource would be documentation from popular machine learning libraries like TensorFlow, PyTorch, and Detectron2, which delve into specific implementation details. Finally, there are many good computer vision textbooks that delve into the theoretical background of these processes.
In summary, the `ObjectDetectionEvaluator()` implicitly handles overlapping bounding boxes via Non-Maximum Suppression. It processes your raw model output, applying NMS, and provides evaluation metrics calculated on the resulting non-overlapping box predictions. Understanding this implicit operation is crucial for interpreting results and tuning object detection models correctly. The key is to remember that the metrics you observe are based on this post-processed data, not the initial raw outputs.
