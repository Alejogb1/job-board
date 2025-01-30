---
title: "How do I calculate Intersection over Union (IoU) for multi-labeled bounding boxes in TensorFlow 2.x?"
date: "2025-01-30"
id: "how-do-i-calculate-intersection-over-union-iou"
---
Calculating Intersection over Union (IoU) for multi-labeled bounding boxes in TensorFlow 2.x requires a nuanced approach, diverging significantly from the single-label case. Specifically, when a single bounding box can be associated with multiple object categories, the traditional IoU calculation for one-to-one matching falls short. I encountered this challenge while working on a project involving aerial image analysis where multiple types of structures could exist within a single bounding box, necessitating an IoU calculation that considered all object labels present within. The core problem is that a naive application of the standard IoU formula, using only one label as a reference, would lead to a misrepresentation of the overlap between predicted and ground truth boxes when multiple labels are in play.

The solution involves computing IoU *per class* and then considering how to aggregate these per-class values into a single, meaningful score. I've discovered that a direct global IoU across all labels doesn't often provide the desired behavior, as it can mask performance issues in specific classes. Therefore, I implemented an approach that explicitly considers each label and leverages vectorized operations in TensorFlow for efficiency.

Fundamentally, we’re aiming to determine the degree of overlap between predicted bounding boxes and their corresponding ground truth bounding boxes, taking into account that a single box can have multiple associated labels. This means for each predicted box, we have a vector of predicted labels, and similarly for each ground truth box. The calculation proceeds as follows:

1.  **Bounding Box Coordinates:** Assume each bounding box is represented by `[y_min, x_min, y_max, x_max]` normalized coordinates between 0 and 1. We must have both predicted bounding boxes and ground truth bounding boxes in this format.

2.  **Label Representation:** We represent the labels associated with bounding boxes using one-hot encoded vectors (or a multi-hot encoded vector when one box has multiple labels). For instance, if we have three classes, `[0, 1, 0]` represents class 1 being present.

3.  **Per-Class IoU:** For each class, we filter the predicted bounding boxes and ground truth bounding boxes to only consider the boxes that contain that specific class label. Then we calculate the classic IoU for those.

4.  **IoU per Box:** Within each of the filtered lists of bounding boxes for a given class, we must correctly match the bounding boxes together to calculate IoU. These lists may not be the same length. Usually, a matrix of IoUs will be produced, and from this matrix one of a variety of matching strategies can be employed.

5.  **Aggregation:** Finally, we have to decide how to aggregate this per-class per-box IoU data into a meaningful score. Options include taking the mean IoU per class, the mean average IoU, or a more complicated method.

Here's the first example that lays the groundwork for the per-class IoU calculation:

```python
import tensorflow as tf

def box_area(boxes):
    """Calculates the area of bounding boxes.

    Args:
        boxes: A tensor of shape [N, 4] representing bounding boxes
        in the format [y_min, x_min, y_max, x_max].

    Returns:
        A tensor of shape [N] containing the area of each box.
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min))

def intersection_area(boxes1, boxes2):
    """Calculates the intersection area of two sets of bounding boxes.

    Args:
        boxes1: A tensor of shape [N, 4].
        boxes2: A tensor of shape [M, 4].

    Returns:
        A tensor of shape [N, M] containing the intersection area
        between each pair of boxes.
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(boxes1, 4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(boxes2, 4, axis=1)

    max_min_y = tf.maximum(y_min1, tf.transpose(y_min2))
    max_min_x = tf.maximum(x_min1, tf.transpose(x_min2))
    min_max_y = tf.minimum(y_max1, tf.transpose(y_max2))
    min_max_x = tf.minimum(x_max1, tf.transpose(x_max2))

    intersection_height = tf.maximum(0.0, min_max_y - max_min_y)
    intersection_width = tf.maximum(0.0, min_max_x - max_min_x)

    return tf.squeeze(intersection_height * intersection_width)

def iou(boxes1, boxes2):
    """Calculates the Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1: A tensor of shape [N, 4].
        boxes2: A tensor of shape [M, 4].

    Returns:
        A tensor of shape [N, M] containing the IoU of each pair.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    intersection = intersection_area(boxes1, boxes2)

    return intersection / (tf.expand_dims(area1,axis=-1) + tf.expand_dims(area2, axis=0) - intersection + 1e-8) # 1e-8 avoids division by 0
```

This initial code block provides three crucial functions: `box_area`, `intersection_area`, and `iou`. These implement fundamental bounding box operations essential for the next stage. `box_area` simply computes the area of a bounding box; `intersection_area` finds the intersection area between all pairs of boxes provided; and `iou` uses these two functions to calculate IoU between all pairs of bounding boxes. It also includes a small value to avoid division-by-zero issues.

Building on this, the next example demonstrates how to filter boxes by class and calculate per-class IoU using the functions defined previously:

```python
def calculate_per_class_iou(predicted_boxes, predicted_labels, ground_truth_boxes, ground_truth_labels, num_classes):
    """Calculates per-class IoU for multi-labeled bounding boxes.

    Args:
        predicted_boxes: A tensor of shape [P, 4] representing predicted bounding boxes.
        predicted_labels: A tensor of shape [P, C] representing predicted labels (one-hot).
        ground_truth_boxes: A tensor of shape [G, 4] representing ground truth bounding boxes.
        ground_truth_labels: A tensor of shape [G, C] representing ground truth labels (one-hot).
        num_classes: The number of classes in our dataset.

    Returns:
        A list of tensors containing each class IoU matrix.  Each matrix will
        be of size [P, G] where entries correspond to the IoU between a given predicted and ground truth box.
    """
    iou_per_class = []
    for class_index in range(num_classes):
        pred_box_indices_for_class = tf.where(predicted_labels[:, class_index] > 0)[:, 0]
        gt_box_indices_for_class = tf.where(ground_truth_labels[:, class_index] > 0)[:, 0]

        pred_boxes_for_class = tf.gather(predicted_boxes, pred_box_indices_for_class)
        gt_boxes_for_class = tf.gather(ground_truth_boxes, gt_box_indices_for_class)

        #Avoid calculation if boxes are missing for a given class
        if tf.size(pred_boxes_for_class) == 0 or tf.size(gt_boxes_for_class) == 0:
             iou_matrix = tf.zeros((tf.shape(pred_boxes_for_class)[0], tf.shape(gt_boxes_for_class)[0]), dtype=tf.float32)
        else:
            iou_matrix = iou(pred_boxes_for_class, gt_boxes_for_class)

        iou_per_class.append(iou_matrix)

    return iou_per_class
```

This function `calculate_per_class_iou` iterates through each class, filters both the predicted and ground truth boxes to only include those with the class label, and then proceeds with calculating and storing the per-class IoU matrix. It checks the shape of the filtered boxes and returns a zero IoU matrix when either the set of predicted or ground truth boxes are empty. This structure allows us to apply different metrics over each class, instead of over all labels in a given bounding box.

Finally, this example illustrates a simple aggregation strategy: the average of maximum IoU values per predicted box, per class. This is a common approach for object detection tasks:

```python
def calculate_aggregated_iou(predicted_boxes, predicted_labels, ground_truth_boxes, ground_truth_labels, num_classes):
    """Calculates the aggregated IoU for all classes.

    Args:
        predicted_boxes: A tensor of shape [P, 4].
        predicted_labels: A tensor of shape [P, C].
        ground_truth_boxes: A tensor of shape [G, 4].
        ground_truth_labels: A tensor of shape [G, C].
        num_classes: The number of classes in our dataset.

    Returns:
        A tensor containing the final aggregated IoU metric.
    """
    iou_per_class = calculate_per_class_iou(predicted_boxes, predicted_labels, ground_truth_boxes, ground_truth_labels, num_classes)
    mean_max_iou_per_class = []

    for iou_matrix in iou_per_class:
        if tf.size(iou_matrix) > 0:
            max_iou = tf.reduce_max(iou_matrix, axis=1) #Max IoU per prediction for a class.
            mean_max_iou_per_class.append(tf.reduce_mean(max_iou)) #Mean of Max IoUs for a class
        else:
             mean_max_iou_per_class.append(tf.constant(0.0, dtype=tf.float32))


    return tf.reduce_mean(mean_max_iou_per_class) #Mean of Mean-Max IoUs across all classes.
```

This `calculate_aggregated_iou` function calls `calculate_per_class_iou` and then iterates through the resulting per-class IoU matrices. For each class, it finds the maximum IoU value *per predicted box* and then takes the average of those maximums. Finally, it calculates the mean of all these class-specific average IoUs. It handles cases where no bounding boxes are detected for a given class by skipping calculation and appending a 0.0 for that class.

For further study, I suggest reviewing the TensorFlow Object Detection API, specifically its implementation of metrics. Also, exploring literature on evaluation metrics for multi-label object detection, found in various computer vision research papers, will improve understanding of the range of options available, and the pros and cons of each. Finally, research on optimal matching between bounding boxes (e.g., the Hungarian Algorithm or greedy methods) can be essential, since the method described here does not explicitly match boxes. Instead it considers the IoU between all possible box pairs, which can be a useful approach in many cases.
