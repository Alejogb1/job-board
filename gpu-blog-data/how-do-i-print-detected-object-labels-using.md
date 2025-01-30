---
title: "How do I print detected object labels using the TensorFlow Object Detection API?"
date: "2025-01-30"
id: "how-do-i-print-detected-object-labels-using"
---
The TensorFlow Object Detection API, while robust for model training and evaluation, requires careful navigation to extract and display predicted object labels post-inference. The core issue often stems from the API's output format, which provides numerical class IDs rather than human-readable labels. To bridge this gap, we need to utilize the provided label map and construct a lookup mechanism.

Here's a breakdown of how I've addressed this in various projects. The typical workflow involves these steps after performing inference: 1) accessing the 'detection_classes' tensor from the output dictionary, 2) loading the label map, and 3) converting the numerical class IDs to their corresponding string labels for display. The output 'detection_classes' tensor contains floating-point numbers representing class indices (starting at 1), not the raw string labels. I've found that relying directly on these raw numbers often leads to incorrect labeling if not handled properly.

**1. Explanation of the Core Process**

The TensorFlow Object Detection API utilizes a protobuf-based label map file (often named 'labelmap.pbtxt') during training. This file contains a mapping between numerical class IDs (integers) and their corresponding string labels. This mapping is vital for interpreting the network's output. The 'detection_classes' tensor produced after running inference contains numerical values directly referencing indices in this label map, where each index starts at 1, reflecting the internal indexing of the 'item' entries within the protocol buffer. The inference output includes other tensors like 'detection_scores' and 'detection_boxes' that describe confidence and location of the bounding boxes, but we focus specifically on class labeling in this discussion.

The fundamental steps for achieving printable labels are: first, extract the 'detection_classes' output tensor as a NumPy array; second, parse the labelmap file, typically containing the class `id` as an integer and the associated `name` as a string. This parsed data will be structured into a lookup dictionary. Finally, iterate through the predicted class indices, using this lookup to obtain the corresponding readable labels. This approach ensures that, regardless of the underlying training configuration, the labels are retrieved accurately by directly referencing the provided label mapping.

**2. Code Examples with Commentary**

The following examples demonstrate various approaches I've used, each building upon the core concepts described above.

**Example 1: Simple Label Lookup**

This first snippet demonstrates basic label extraction assuming you have already run inference and possess the resulting output dictionary as `output_dict`, and that the path to the label map file is stored in the variable `label_map_path`.

```python
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

def load_label_map(label_map_path):
  """Loads label map from a pbtxt file. Returns a dictionary of ID to name."""
  with open(label_map_path, 'r') as f:
    label_map_string = f.read()
  label_map = text_format.Parse(label_map_string,
      tf.compat.v1.gfile.Proto('object_detection.protos.StringIntLabelMap'))
  label_map_dict = {}
  for item in label_map.item:
    label_map_dict[item.id] = item.name
  return label_map_dict

def extract_labels(output_dict, label_map_path):
  """Extracts class labels from inference output."""
  label_map = load_label_map(label_map_path)
  detection_classes = output_dict['detection_classes'].numpy().astype(np.int32) #Cast to Int for index lookup
  detection_scores = output_dict['detection_scores'].numpy() # Keep for score filtering if desired
  labels = [label_map[class_id] for class_id in detection_classes] #lookup class string names
  return labels
  
# Example Usage (assume 'output_dict' is already defined):
# labels = extract_labels(output_dict, 'path/to/labelmap.pbtxt')
# print(labels)
```

This example establishes a basic lookup system by loading the pbtxt file into a dictionary. It then iterates through the detected classes to extract labels for all detected objects. Note that this assumes a consistent format of the labelmap. This particular implementation relies on exact match lookup within the extracted labelmap for the corresponding indices derived from the inference output. It's also designed to be modular to aid reuse and is robust enough to handle commonly structured labelmaps.

**Example 2: Filtering by Confidence Score**

This expands upon Example 1 by integrating a score threshold for filtering out low-confidence detections before extracting the labels, a common requirement in practical application.

```python
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

def load_label_map(label_map_path):
    """Loads label map from a pbtxt file. Returns a dictionary of ID to name."""
    with open(label_map_path, 'r') as f:
        label_map_string = f.read()
    label_map = text_format.Parse(label_map_string,
        tf.compat.v1.gfile.Proto('object_detection.protos.StringIntLabelMap'))
    label_map_dict = {}
    for item in label_map.item:
      label_map_dict[item.id] = item.name
    return label_map_dict

def extract_labels_with_threshold(output_dict, label_map_path, min_score_thresh = 0.5):
    """Extracts class labels from inference output, filtering by confidence score."""
    label_map = load_label_map(label_map_path)
    detection_classes = output_dict['detection_classes'].numpy().astype(np.int32)
    detection_scores = output_dict['detection_scores'].numpy()
    valid_detections = detection_scores >= min_score_thresh
    filtered_classes = detection_classes[valid_detections]
    labels = [label_map[class_id] for class_id in filtered_classes]
    return labels
# Example Usage
# filtered_labels = extract_labels_with_threshold(output_dict, 'path/to/labelmap.pbtxt', min_score_thresh=0.6)
# print(filtered_labels)
```

Here, we added a `min_score_thresh` parameter. Only the objects with scores equal to or higher than the threshold have their corresponding labels added to the result.  This addresses a common use-case for ignoring potentially false detections by incorporating score thresholding before extracting class labels. This avoids unnecessary computation and provides the most relevant information based on predefined confidence criteria.

**Example 3: Handling Missing Labels**

Occasionally, label maps may not contain IDs that are predicted. This code snippet adds a layer of robustness by checking if an ID exists within the label map, using a generic 'Unknown' label if it's missing.

```python
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

def load_label_map(label_map_path):
    """Loads label map from a pbtxt file. Returns a dictionary of ID to name."""
    with open(label_map_path, 'r') as f:
        label_map_string = f.read()
    label_map = text_format.Parse(label_map_string,
        tf.compat.v1.gfile.Proto('object_detection.protos.StringIntLabelMap'))
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.id] = item.name
    return label_map_dict

def extract_labels_with_fallback(output_dict, label_map_path):
    """Extracts class labels, handling missing IDs."""
    label_map = load_label_map(label_map_path)
    detection_classes = output_dict['detection_classes'].numpy().astype(np.int32)
    labels = []
    for class_id in detection_classes:
        if class_id in label_map:
            labels.append(label_map[class_id])
        else:
            labels.append("Unknown")
    return labels

# Example Usage
# robust_labels = extract_labels_with_fallback(output_dict, 'path/to/labelmap.pbtxt')
# print(robust_labels)
```

The primary change is within the label extraction loop. Before appending a label to the results, it checks if the `class_id` is present in the `label_map` dictionary. If the `class_id` is not found, a string 'Unknown' is assigned instead, preventing an error and providing a graceful failure case when an unexpected output is generated. This is useful for model development phases where the label map may not be fully congruent with the latest trained model output.

**3. Resource Recommendations**

For further exploration, I recommend the following sources:

*   **TensorFlow Object Detection API Documentation**: This is the authoritative reference and provides comprehensive documentation on its various components. The sections on model building, inference, and data formats are particularly relevant.
*   **Official TensorFlow Tutorials**: The TensorFlow website often has hands-on tutorials specific to the object detection API which can be helpful in understanding the core processes. Exploring the tutorials, especially those addressing model evaluation and post-processing provides further context and real-world examples.
*   **Open Source Implementations**: Examining example implementations of the Object Detection API can reveal efficient ways to handle common problems. Projects often include code snippets and demonstrations to use the API with a greater degree of specific examples. Pay attention to how labelmaps and post-processing logic is handled in practical contexts.

In summary, extracting readable labels after object detection using the TensorFlow Object Detection API requires understanding the output tensor formats and the label map. By loading and using a custom label lookup function, one can accurately translate predicted class indices into their respective string labels. Employing techniques like score-based filtering and fallback handling enhances the robustness of such label extraction mechanisms.
