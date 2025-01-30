---
title: "Why does the TensorFlow Object Detection API's mAP evaluation exhibit erratic behavior?"
date: "2025-01-30"
id: "why-does-the-tensorflow-object-detection-apis-map"
---
The erratic mean Average Precision (mAP) behavior during evaluation of models trained with the TensorFlow Object Detection API often stems from a confluence of factors related to the inherent complexity of object detection evaluation and specific implementation details within the API. Primarily, mAP’s sensitivity to variations in the detection threshold, box IoU (Intersection over Union) threshold, and the distribution of object instances within the dataset contribute significantly to this observed instability.

I’ve encountered this frequently, particularly during a project involving aerial imagery analysis where inconsistencies in object size and perspective across training and validation sets often resulted in substantial mAP fluctuations between training epochs. The API itself, while powerful, is not immune to these inherent vulnerabilities. Specifically, two critical areas within the evaluation pipeline impact the mAP score's stability: the NMS (Non-Maximum Suppression) algorithm and the manner in which the API handles bounding box overlap during evaluation.

Let’s first consider NMS. This crucial step is applied post-detection to eliminate redundant bounding boxes predicted for the same object. The NMS threshold determines how tightly bounding boxes must overlap to be considered the same prediction, and this has a direct effect on the resulting precision and recall used to calculate the mAP. When dealing with a dense scenario involving numerous object detections, a slightly changed NMS threshold will affect the number of true positives, false positives and false negatives drastically. This effect can be further amplified if one is using an object detector that produces less confident or more scattered results, increasing the rate of false positives. The mAP score can easily fluctuate significantly if the NMS parameters are not tuned specifically to each dataset. The default parameters within the API's configuration can frequently fall short of optimal performance in varied or niche datasets.

The second, and equally important factor, is the bounding box overlap handling. mAP calculations depend on evaluating the correctness of bounding box predictions by using an Intersection over Union threshold. A prediction is a considered true positive if the IoU with a ground truth box is above that given IoU threshold. Again, the default values provided within the API’s configuration files are not guaranteed to be optimal across different datasets. Changes to the number of true positives and false positives, and hence precision and recall, will cause mAP to vary. Further, the process of matching predictions to ground-truth boxes becomes more intricate as the number of detections rises and is subject to implementation quirks. A particularly troublesome edge case arises when the API struggles to resolve ambiguous matches between multiple predicted boxes and multiple ground-truth boxes, especially in scenarios with overlapping objects. Minor changes in model training, parameters or even the augmentation method applied to each training epoch can lead to slightly different detection positions and sizes, with the potential for different matching outcomes in the overlapping areas, thus changing the mAP. In summary, both NMS and box IoU thresholds introduce significant variability in the evaluation process, which might become especially apparent when training progress is measured every few epochs.

To illustrate these points, I'll share three code snippets demonstrating how these factors can impact the evaluation process. The first example shows a simple evaluation loop, modified to demonstrate how a changing NMS threshold affects detected objects. I am using hypothetical functions, as access to the inner functions of the API is limited for demonstration purposes:

```python
def evaluate_with_nms(detections, gt_boxes, iou_threshold=0.5, nms_threshold=0.5):
    """Evaluates detections using given IoU and NMS thresholds."""
    nms_detections = apply_nms(detections, nms_threshold) # hypothetical NMS function
    true_positives, false_positives, false_negatives = calculate_tp_fp_fn(nms_detections, gt_boxes, iou_threshold) # hypothetical function
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    ap = calculate_average_precision(precision, recall)
    return ap

# Example usage:

# Assume detections and gt_boxes are loaded
detections = [
     {'box': [0.1, 0.2, 0.3, 0.4], 'score': 0.9},
     {'box': [0.12, 0.22, 0.33, 0.44], 'score': 0.85}, # slightly overlapping
     {'box': [0.5, 0.6, 0.7, 0.8], 'score': 0.7} # distinct object
    ]
gt_boxes = [{'box': [0.1, 0.2, 0.3, 0.4]}, {'box':[0.5, 0.6, 0.7, 0.8]}]

ap_nms_05 = evaluate_with_nms(detections, gt_boxes, nms_threshold=0.5)
ap_nms_02 = evaluate_with_nms(detections, gt_boxes, nms_threshold=0.2)

print(f"mAP with NMS 0.5: {ap_nms_05}")
print(f"mAP with NMS 0.2: {ap_nms_02}")
```

In this example, observe how a change in the NMS threshold (from 0.5 to 0.2) drastically affects the average precision by altering the number of considered true positives. A lower NMS threshold will result in the second overlapping box not being suppressed, and likely to be counted as false positive. Note that the precision calculated at the end of the function is a simplified representation of mAP calculation, but illustrates the effect on average precision.

Next, let's examine the sensitivity of mAP to the IoU threshold. A higher IoU threshold will penalize bounding box predictions that are not highly overlapping with the ground truth boxes, while a lower IoU threshold allows for more variability.

```python
def evaluate_with_iou(detections, gt_boxes, iou_threshold=0.5, nms_threshold=0.5):
    """Evaluates detections with given IoU threshold."""
    nms_detections = apply_nms(detections, nms_threshold)
    true_positives, false_positives, false_negatives = calculate_tp_fp_fn(nms_detections, gt_boxes, iou_threshold)
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    ap = calculate_average_precision(precision, recall)
    return ap

detections = [
    {'box': [0.15, 0.25, 0.35, 0.45], 'score': 0.9},  # Slightly off
    {'box': [0.5, 0.6, 0.7, 0.8], 'score': 0.7} # good box
    ]
gt_boxes = [{'box': [0.1, 0.2, 0.3, 0.4]}, {'box':[0.5, 0.6, 0.7, 0.8]}]

ap_iou_05 = evaluate_with_iou(detections, gt_boxes, iou_threshold=0.5)
ap_iou_07 = evaluate_with_iou(detections, gt_boxes, iou_threshold=0.7)

print(f"mAP with IoU 0.5: {ap_iou_05}")
print(f"mAP with IoU 0.7: {ap_iou_07}")
```

Here, even with the same detections, a change in the IoU threshold causes a change in the number of true positives. The detection with `[0.15, 0.25, 0.35, 0.45]` box may be counted as a true positive when the threshold is set to 0.5 but is a false positive when it's 0.7. This example highlights that the mAP is dependent on this threshold, and it's important to be aware of its value when using the mAP score for comparison.

Finally, let’s consider that small variations in training can lead to changes in bounding box predictions. In turn, these minor changes might produce fluctuating mAP results. This can often be observed between training epochs.

```python
import random

def simulate_model_drift(detections, max_variation=0.02):
  """Simulates minor changes in box predictions."""
  new_detections = []
  for detection in detections:
      box = detection['box']
      new_box = [coord + random.uniform(-max_variation, max_variation) for coord in box]
      new_detections.append({'box': new_box, 'score': detection['score']})
  return new_detections

def evaluate_with_drift(detections, gt_boxes, iou_threshold=0.5, nms_threshold=0.5):
  """Evaluates detections, simulating model drift."""
  nms_detections = apply_nms(detections, nms_threshold)
  true_positives, false_positives, false_negatives = calculate_tp_fp_fn(nms_detections, gt_boxes, iou_threshold)
  precision = calculate_precision(true_positives, false_positives)
  recall = calculate_recall(true_positives, false_negatives)
  ap = calculate_average_precision(precision, recall)
  return ap

detections = [
    {'box': [0.15, 0.25, 0.35, 0.45], 'score': 0.9},
    {'box': [0.5, 0.6, 0.7, 0.8], 'score': 0.7}
  ]
gt_boxes = [{'box': [0.1, 0.2, 0.3, 0.4]}, {'box':[0.5, 0.6, 0.7, 0.8]}]

ap_before_drift = evaluate_with_drift(detections, gt_boxes)
drifted_detections = simulate_model_drift(detections)
ap_after_drift = evaluate_with_drift(drifted_detections, gt_boxes)

print(f"mAP before drift: {ap_before_drift}")
print(f"mAP after drift: {ap_after_drift}")
```
Here, I simulate slight variations in the bounding box positions, leading to altered mAP scores. It is important to keep in mind that in every training epoch small shifts in detections are expected and that will inevitably change the mAP.

For those looking to further investigate this, I recommend focusing on the detailed documentation of the TensorFlow Object Detection API. Pay close attention to the configuration parameters governing the NMS algorithm and IoU threshold settings in the evaluation section. Additionally, exploring academic papers discussing the nuances of object detection evaluation and alternative metrics beyond mAP can be beneficial. Finally, examining other implementations of mAP calculation (outside of the API) can illuminate the subtleties of its implementation. Experimenting with modified NMS and IoU thresholds should provide an empirical insight into their effect on the detection evaluation.
