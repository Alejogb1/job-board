---
title: "How can YOLO bounding boxes be calculated in ONNX?"
date: "2025-01-30"
id: "how-can-yolo-bounding-boxes-be-calculated-in"
---
The conversion of a YOLO (You Only Look Once) object detection model to the ONNX (Open Neural Network Exchange) format, while facilitating cross-platform deployment, necessitates a post-processing step to interpret its raw output into meaningful bounding boxes. The ONNX model, after performing its forward pass, typically produces a tensor containing information about object presence, class probabilities, and bounding box parameters, all encoded in a format specific to YOLO's architecture. Extracting usable bounding boxes requires understanding this format and implementing the reverse calculation of YOLO's encoding.

I've worked extensively with various YOLO versions converted to ONNX, and the fundamental challenge remains consistent: decoding the output tensor back into interpretable coordinates. The core problem stems from YOLO's design, which predicts bounding box parameters relative to grid cells and anchor boxes. The ONNX output, representing these relative values, needs to be transformed to represent absolute coordinates in the image. This process involves several steps, which I'll detail and illustrate with Python code examples.

The general structure of the output tensor varies slightly between YOLO versions (v3, v4, v5, etc.), but it fundamentally encodes the following:

1.  **Grid Cell Information:** The image is divided into a grid, and each grid cell is responsible for predicting objects whose centers fall within that cell.
2.  **Anchor Boxes:** Predefined bounding box shapes (anchor boxes) are used to help the model predict bounding box sizes more efficiently. For each grid cell, multiple anchor boxes are considered.
3.  **Bounding Box Parameters:** For each anchor box at each grid cell, the model predicts a set of values that modify the anchor box's dimensions and position: center offsets (tx, ty), width and height scaling factors (tw, th), and an object confidence score.

The transformation process, after receiving the raw ONNX output, can be broken down into these stages:

1.  **Tensor Reshaping:** The raw output tensor is reshaped into a more manageable form. This often involves splitting the tensor into channels corresponding to each bounding box parameter (tx, ty, tw, th, objectness, class probabilities). The exact shape depends on the model's architecture.
2.  **Applying Sigmoid Functions:** The center offsets (tx, ty) and the object confidence score are typically passed through a sigmoid function. This constrains these values to a range between 0 and 1, which represents a percentage offset within the grid cell and the likelihood of an object being present, respectively.
3.  **Exponential Scaling:** The width and height scaling factors (tw, th) are often exponentiated. This converts these scale factors to absolute width and height adjustments that are then applied to anchor box dimensions.
4.  **Calculating Absolute Center Coordinates:** Using the sigmoid-modified offsets (sx, sy) and the grid cell coordinates, the absolute center coordinates of each predicted bounding box are calculated. This is often done by adding the sigmoid output scaled by grid dimensions to the offset of grid centers.
5.  **Calculating Absolute Width and Height:** The exponentially scaled width and height factors are multiplied by the corresponding anchor box dimensions to derive the absolute width and height of the bounding box.
6.  **Class Scores:** Class probabilities are extracted and usually passed through a softmax function (if they are raw logits), or the provided probabilities are used directly.
7.  **Filtering:**  Bounding boxes with low objectness scores are discarded.
8. **Non-Maximum Suppression (NMS):** This is a critical step to remove redundant detections, where multiple bounding boxes detect the same object. NMS selects the bounding box with the highest confidence score from a set of overlapping boxes.

Let's look at specific code examples.

**Example 1: Reshaping and Sigmoid Application (Generic Example)**

```python
import numpy as np

def process_yolo_output(output, grid_size, anchors, num_classes):
    """
    Processes the raw ONNX output from a generic YOLO model.

    Args:
    output (np.ndarray): Raw output tensor from the ONNX model.
    grid_size (tuple): Height and Width of the feature map.
    anchors (np.ndarray): Anchor box dimensions.
    num_classes (int): Number of object classes.

    Returns:
    np.ndarray: An array of bounding boxes and their attributes.
    """
    height, width = grid_size
    num_anchors = len(anchors)

    # Assuming output shape is (batch, height * width * num_anchors, 5+num_classes)
    output = output.reshape(1, height, width, num_anchors, 5+num_classes)

    # Initialize array to store results
    processed_boxes = np.zeros((height * width * num_anchors, 5+num_classes))
    idx=0

    for h in range(height):
        for w in range(width):
            for a in range(num_anchors):
               # Extract values (tx, ty, tw, th, objectness, classes)
               bbox_data = output[0, h, w, a, :]

               tx, ty, tw, th, obj_score = bbox_data[0:5]
               class_probs = bbox_data[5:]

               # Apply sigmoid to center offsets and objectness score
               sx = 1.0 / (1.0 + np.exp(-tx)) #sigmoid
               sy = 1.0 / (1.0 + np.exp(-ty)) #sigmoid
               obj_score = 1.0 / (1.0 + np.exp(-obj_score)) #sigmoid

               processed_boxes[idx,0:5] = np.array([sx, sy, tw, th, obj_score])
               processed_boxes[idx,5:] = class_probs
               idx +=1


    return processed_boxes

# Example usage:
output_tensor = np.random.rand(1, 13 * 13 * 3, 85) #dummy data: 13x13 grid, 3 anchors, 80 classes
anchors_ex = np.array([[10, 13], [16, 30], [33, 23]])
processed_output_ex = process_yolo_output(output_tensor, (13, 13), anchors_ex, 80)
print("Shape of processed output example:", processed_output_ex.shape)
```

This example demonstrates the initial processing of the output tensor. It reshapes the output tensor based on the feature map dimensions and applies the sigmoid function to center offsets and objectness scores. This is a critical step to get the bounding box parameters into the correct ranges. It's important to understand that output tensor shapes may vary depending on the specific model. Here, I assume the output is already flattened for simplicity. The example simulates data using random numbers, and the dimensions are specific for demonstration purpose.

**Example 2: Calculating Absolute Bounding Box Coordinates**

```python
def calculate_absolute_boxes(processed_boxes, grid_size, anchors, image_size):
    """
    Calculates the absolute bounding box coordinates.

    Args:
    processed_boxes (np.ndarray): Processed output from process_yolo_output.
    grid_size (tuple): Grid size of feature map.
    anchors (np.ndarray): Anchor box dimensions.
    image_size (tuple): Height and width of the input image.

    Returns:
    np.ndarray: Absolute bounding box coordinates and confidence.
    """

    height, width = grid_size
    img_height, img_width = image_size
    num_anchors = len(anchors)

    absolute_boxes = np.zeros((len(processed_boxes), 6)) #x1, y1, x2, y2, objectness, class_id


    idx = 0
    for h in range(height):
        for w in range(width):
            for a in range(num_anchors):
                sx, sy, tw, th, obj_score = processed_boxes[idx,0:5]
                class_probs = processed_boxes[idx,5:]
                # Get anchor box dimensions for current anchor
                anchor_w, anchor_h = anchors[a]

                # Calculate bounding box center coordinates
                cx = (w + sx) / width
                cy = (h + sy) / height

                # Calculate bounding box width and height
                bw = anchor_w * np.exp(tw) / img_width
                bh = anchor_h * np.exp(th) / img_height

                # Convert center coordinates, width, and height to absolute (x1, y1, x2, y2) format
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                class_id = np.argmax(class_probs)
                absolute_boxes[idx] = np.array([x1*img_width, y1*img_height, x2*img_width, y2*img_height, obj_score, class_id])

                idx += 1


    return absolute_boxes


# Example usage:
img_size_ex = (416, 416)  # Example image size
absolute_boxes_ex = calculate_absolute_boxes(processed_output_ex, (13, 13), anchors_ex, img_size_ex)
print("Shape of calculated absolute boxes example:", absolute_boxes_ex.shape)
```

This function builds upon the previous example, taking the output, and transforming relative parameters into absolute bounding box coordinates within the image. The scaling based on grid cell indices, feature map size, and anchor box dimensions is demonstrated, converting normalized coordinates to real image coordinates. The result is a set of bounding box coordinates in pixel space and confidence scores. This also include class_id based on class_probs.

**Example 3: Non-Maximum Suppression (Simplified Example)**

```python
def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Applies simplified Non-Maximum Suppression.

    Args:
    boxes (np.ndarray): Absolute bounding boxes with objectness (x1, y1, x2, y2, obj_score, class_id).
    iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: The indices of the selected boxes.
    """
    def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area_box1 + area_box2 - intersection
            return intersection / union if union > 0 else 0

    # Sort boxes based on objectness scores in descending order
    sorted_indices = np.argsort(boxes[:, 4])[::-1]

    selected_indices = []
    suppressed_indices = set()

    for i in sorted_indices:
        if i not in suppressed_indices:
            selected_indices.append(i)
            for j in sorted_indices:
                if j != i and j not in suppressed_indices:
                    if iou(boxes[i], boxes[j]) > iou_threshold and boxes[i,5] == boxes[j,5]:
                        suppressed_indices.add(j)

    return selected_indices


# Example usage:
nms_boxes_ex = non_max_suppression(absolute_boxes_ex)
print("Shape of selected boxes after NMS:", len(nms_boxes_ex))
```

This example demonstrates a simplified version of Non-Maximum Suppression. I've prioritized clarity over a highly optimized implementation, illustrating the core concept of filtering out overlapping detections based on their Intersection over Union (IoU) scores. The implementation compares bounding boxes with same class_id.

For further exploration and a deeper understanding, I suggest studying the original YOLO papers as a primary resource. Specifically, the papers on YOLOv3, YOLOv4, and YOLOv5 are crucial. Additionally, research the ONNX official documentation, especially the specifications for custom operators, since some post-processing steps may be performed within the ONNX graph using custom operations. Finally, examining open-source implementations of YOLO-to-ONNX converters is invaluable for understanding practical implementations. Specifically look into implementations of post-processing operations to decode boxes. This includes understanding how the specific anchoring system is handled. By reviewing these resources, a comprehensive understanding of bounding box calculations with ONNX can be developed.
