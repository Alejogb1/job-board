---
title: "How can I efficiently access scores, labels, and ground truth counts from a TensorFlow Object Detection model?"
date: "2025-01-30"
id: "how-can-i-efficiently-access-scores-labels-and"
---
TensorFlow Object Detection models, especially after inference, often present the challenge of extracting granular results beyond the visual bounding boxes. Direct access to scores, class labels, and particularly the count of true positives within the predicted outputs necessitates a clear understanding of the model's output tensors and appropriate post-processing techniques.

My experience working with custom object detection pipelines, including training on datasets like COCO and PASCAL VOC, has frequently required this level of detail. While visualizing bounding boxes provides immediate feedback, detailed evaluation and analysis of model performance depend heavily on programmatically accessing these specific data points.

The core of the issue lies in the output format of the TensorFlow Object Detection API. The model, after forwarding an input image, returns a dictionary containing multiple tensors. The most relevant tensors for this task are generally named: 'detection_boxes', 'detection_scores', 'detection_classes', and often a 'num_detections' tensor. The 'detection_boxes' tensor holds normalized coordinates for each detected object, while 'detection_scores' contains the confidence score for each of these detections. The 'detection_classes' tensor provides the predicted class index corresponding to each detected box. Crucially, the 'num_detections' tensor indicates the valid number of detected boxes, as the tensors are often padded to a fixed size to optimize GPU utilization.

Accessing these tensors involves processing the output dictionary after running inference. Specifically, you would extract these tensors, remove the padding using the information provided in the 'num_detections' tensor, and then manipulate them as needed. Extracting ground truth counts, however, is more nuanced because the model does not directly return this information. Instead, this typically involves comparing the model's predictions with the ground truth annotations provided within your dataset format. This often requires a custom function, tailored to your dataset's particular ground truth structure, that performs Intersection over Union (IoU) analysis and other relevant metrics.

Here are three code examples illustrating this process:

**Example 1: Extracting Scores, Labels, and Boxes**

```python
import tensorflow as tf
import numpy as np

def process_detection_output(output_dict):
  """
    Extracts scores, labels, and boxes from a TensorFlow Object Detection model output.

    Args:
        output_dict: A dictionary containing the output tensors from a TF object detection model.

    Returns:
        A tuple containing NumPy arrays: scores, class labels, and bounding box coordinates.
  """

  num_detections = int(output_dict['num_detections'][0])
  detection_boxes = output_dict['detection_boxes'][0][:num_detections].numpy()
  detection_classes = output_dict['detection_classes'][0][:num_detections].numpy().astype(np.int32)
  detection_scores = output_dict['detection_scores'][0][:num_detections].numpy()

  return detection_scores, detection_classes, detection_boxes


# Simulated output dictionary from a model
output = {
    'detection_boxes': tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]]]), # padded
    'detection_scores': tf.constant([[0.9, 0.8, 0.01]]),# padded
    'detection_classes': tf.constant([[1, 2, 3]],dtype=tf.int64), # padded
    'num_detections': tf.constant([2]),
}

scores, classes, boxes = process_detection_output(output)
print("Scores:", scores)
print("Classes:", classes)
print("Boxes:", boxes)
```

*   **Commentary:** This code defines a function `process_detection_output` that takes the model's output dictionary as input. It extracts the relevant tensors, utilizes the 'num\_detections' tensor to remove padded values, converts the tensors to NumPy arrays, and converts class indices to integers. The example usage simulates a typical output dictionary, showing the function's application and the resultant arrays. Note the array slicing `[:num_detections]` which removes the padding.

**Example 2: Extracting Ground Truth from a Sample Annotation Format**

```python
import numpy as np

def calculate_true_positives(detection_boxes, detection_classes, ground_truth_boxes, ground_truth_classes, iou_threshold=0.5):
  """
    Calculates the number of true positives based on IoU with ground truth annotations.

    Args:
        detection_boxes: NumPy array of predicted bounding box coordinates.
        detection_classes: NumPy array of predicted class labels.
        ground_truth_boxes: NumPy array of ground truth bounding box coordinates.
        ground_truth_classes: NumPy array of ground truth class labels.
        iou_threshold: IoU threshold for determining a true positive.

    Returns:
        The number of true positives.
  """

  true_positives = 0
  if len(detection_boxes) == 0:
     return 0

  for i, pred_box in enumerate(detection_boxes):
      pred_class = detection_classes[i]
      for j, gt_box in enumerate(ground_truth_boxes):
        gt_class = ground_truth_classes[j]

        if pred_class != gt_class:
            continue  #Skip comparisons where the classes do not match.

        intersection_area = calculate_intersection_area(pred_box, gt_box)
        union_area = calculate_union_area(pred_box, gt_box)
        iou = intersection_area / union_area if union_area > 0 else 0.0
        if iou >= iou_threshold:
          true_positives += 1
          break  # Avoid counting multiple matches for one GT

  return true_positives

def calculate_intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection_area

def calculate_union_area(box1, box2):
  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
  intersection_area = calculate_intersection_area(box1,box2)
  return box1_area + box2_area - intersection_area


# Example usage with normalized coordinates
predicted_boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2,0.2,0.5,0.4]])
predicted_classes = np.array([1, 2, 1])
ground_truth_boxes = np.array([[0.12, 0.22, 0.32, 0.42], [0.55, 0.65, 0.75, 0.85]])
ground_truth_classes = np.array([1, 2])

true_positives_count = calculate_true_positives(predicted_boxes, predicted_classes, ground_truth_boxes, ground_truth_classes)
print("Number of True Positives:", true_positives_count)
```

*   **Commentary:** This example demonstrates how to calculate the number of true positives by comparing predicted bounding boxes and classes with ground truth data. It uses the Intersection over Union (IoU) metric and a threshold. The function iterates through predicted boxes, compares them to the ground truths, and counts a match if their IoU is above the defined threshold *and* their classes match. The example input simulates predicted and ground truth data in normalized format, along with basic functions for area and intersection calculation. The `break` statement prevents double counting.

**Example 3: Batch Processing Multiple Images**

```python
import tensorflow as tf
import numpy as np

def process_batch_detections(batch_output_dict, batch_ground_truth_boxes, batch_ground_truth_classes):
  """
      Processes a batch of detection outputs, calculating true positives for each image.

      Args:
        batch_output_dict: Dictionary of model outputs for the batch.
        batch_ground_truth_boxes: A list containing ground truth boxes for each image.
        batch_ground_truth_classes: A list containing ground truth classes for each image.

      Returns:
        A NumPy array of true positive counts for each image in the batch.
  """
  batch_true_positives = []

  for i in range(int(batch_output_dict['num_detections'].shape[0])):
        num_detections = int(batch_output_dict['num_detections'][i])
        detection_boxes = batch_output_dict['detection_boxes'][i][:num_detections].numpy()
        detection_classes = batch_output_dict['detection_classes'][i][:num_detections].numpy().astype(np.int32)

        ground_truth_boxes = batch_ground_truth_boxes[i]
        ground_truth_classes = batch_ground_truth_classes[i]


        true_positives = calculate_true_positives(detection_boxes, detection_classes,
                                                 ground_truth_boxes, ground_truth_classes)
        batch_true_positives.append(true_positives)
  return np.array(batch_true_positives)

#Simulated Batch output example
batch_output = {
    'detection_boxes': tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]],
                                    [[0.3,0.3,0.6,0.6],[0.8,0.8,0.9,0.9],[0.0,0.0,0.0,0.0]]]),
    'detection_scores': tf.constant([[0.9, 0.8, 0.01],[0.7,0.6,0.02]]),
    'detection_classes': tf.constant([[1, 2, 3],[2,1,3]], dtype = tf.int64),
    'num_detections': tf.constant([2,2])
}
batch_gt_boxes = [np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
                   np.array([[0.4, 0.4, 0.6, 0.6], [0.7,0.7,0.8,0.8]]) ]
batch_gt_classes = [np.array([1,2]), np.array([2,1])]


true_positives_batch = process_batch_detections(batch_output, batch_gt_boxes, batch_gt_classes)
print("True Positives per image:", true_positives_batch)
```

*   **Commentary:** This function extends the previous examples to handle batched model outputs. It iterates through the batch dimension, processes each image individually, calls the `calculate_true_positives` function on each image, and compiles the individual counts into a batch array. This example showcases how to apply the principles shown before to multiple images, which is particularly relevant when evaluating a model's performance over an entire dataset.

For further learning, I would recommend exploring resources directly from TensorFlow. The TensorFlow Object Detection API’s official documentation provides extensive information about the output formats and the process of evaluating object detection models. The source code of libraries like TensorFlow Models, specifically those related to object detection, can be very helpful. Finally, academic research papers focusing on object detection evaluation metrics (e.g., mAP calculation) will also prove invaluable when building advanced custom evaluation pipelines. These resources, while technical, offer a deep understanding of these concepts.
