---
title: "How to load and use a saved TensorFlow 2 object detection model for prediction?"
date: "2025-01-30"
id: "how-to-load-and-use-a-saved-tensorflow"
---
The core challenge in leveraging a pre-trained TensorFlow 2 object detection model for prediction lies not just in loading the model, but in understanding and correctly configuring the pre- and post-processing steps inherent to the model's architecture.  My experience working on several industrial-scale defect detection projects underscored the criticality of meticulous handling of input tensors and output parsing.  Neglecting these often leads to incorrect or nonsensical predictions, despite a seemingly successful model load.

**1. Clear Explanation:**

Loading and utilizing a saved TensorFlow 2 object detection model involves several distinct phases.  Firstly, the model itself, typically saved as a SavedModel directory, needs to be loaded using the `tf.saved_model.load` function. This function returns a `tf.saved_model.load_options` which in turn can be used to run inference. However, the crucial steps are the pre-processing of input images to match the model's expected input format and post-processing of the model's raw output to extract meaningful bounding boxes and class labels.  The specifics of these pre- and post-processing steps depend heavily on the specific model architecture used (e.g., SSD, Faster R-CNN, EfficientDet).  The model's documentation should explicitly specify these requirements.  Common pre-processing steps include resizing, normalization, and potentially color space conversion.  Post-processing typically involves applying non-maximum suppression (NMS) to filter out overlapping bounding boxes and mapping the raw output scores to class probabilities.

Furthermore, the model's configuration file (usually a `.config` file or similar) is often necessary to understand the specific input shapes, class labels, and other hyperparameters vital for accurate prediction.  Ignoring this file may lead to the model's input requirements being mismatched with the supplied image, resulting in errors.


**2. Code Examples with Commentary:**

**Example 1: Basic Model Loading and Prediction (Faster R-CNN)**

This example demonstrates loading a Faster R-CNN model and performing a basic prediction.  Assume the model is saved in `'saved_model'` directory.

```python
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.saved_model.load('saved_model')

# Sample image (replace with your actual image loading and preprocessing)
image_np = np.zeros((640, 640, 3), dtype=np.uint8) # Example: 640x640 RGB image

# Perform prediction (assuming model expects a single image as input)
detections = model(image_np[np.newaxis, ...])

# Access relevant fields from the output.  The specific names may vary depending on the model
detection_boxes = detections['detection_boxes'][0]
detection_classes = detections['detection_classes'][0]
detection_scores = detections['detection_scores'][0]
num_detections = detections['num_detections'][0]

print("Detection Boxes:", detection_boxes)
print("Detection Classes:", detection_classes)
print("Detection Scores:", detection_scores)
print("Number of Detections:", num_detections)
```

**Commentary:**  This example assumes a straightforward model output structure. In practice, the output tensor names might differ, requiring consultation of the model's documentation.  The crucial aspect is the `[np.newaxis, ...]` which adds a batch dimension to the image as most models expect a batch of images as input even for a single inference.  Error handling for missing keys or incorrect tensor shapes should be added for production-level code.

**Example 2:  Pre-processing and Post-processing (SSD)**

This expands on the previous example, incorporating explicit pre-processing and post-processing steps frequently used for models like SSD.

```python
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# ... (Model loading as in Example 1) ...

# Pre-processing: Resize and normalize
image_np = load_image("myimage.jpg") # Placeholder image loading
image_np = tf.image.resize(image_np, (300, 300)) #Resize to the model's input size
image_np = image_np / 255.0  # Normalize

# Perform prediction
detections = model(image_np[np.newaxis, ...])

# Post-processing: Non-Maximum Suppression (NMS)
#  Assume a function 'perform_nms' exists which implements NMS based on detection_boxes and detection_scores
boxes, scores, classes, num_detections = perform_nms(detections['detection_boxes'][0], detections['detection_scores'][0], detections['detection_classes'][0], detections['num_detections'][0], iou_threshold=0.5, score_threshold=0.5)

# Load class labels from label map
category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt', use_display_name=True)

# Visualization (optional)
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    boxes,
    classes.astype(np.int32),
    scores,
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

# Further processing of boxes, scores, and classes
```

**Commentary:** This example includes placeholder functions for image loading, `perform_nms` (which should be implemented using TensorFlow or a similar library), and label map loading. The `visualization_utils` are from the TensorFlow Object Detection API and require installation (`pip install tf-models-official`). The pre-processing step explicitly resizes and normalizes the image.  The post-processing step applies NMS to refine detection results.  The class labels are loaded to map numerical class IDs to meaningful names.


**Example 3: Handling Multiple Images (Batch Processing)**

Efficient processing of multiple images involves feeding a batch of images to the model simultaneously.

```python
import tensorflow as tf
import numpy as np

# ... (Model loading as in Example 1) ...

# Load multiple images (replace with your image loading logic)
images = []
for i in range(5):  #Process 5 images
    image_np = load_image(f'image_{i}.jpg')
    image_np = tf.image.resize(image_np, (300, 300))
    image_np = image_np / 255.0
    images.append(image_np)

images_batch = np.stack(images)

# Perform prediction on the batch
detections = model(images_batch)

# Post-processing (looping through batch elements)
for i in range(images_batch.shape[0]):
    boxes, scores, classes, num_detections = perform_nms(detections['detection_boxes'][i], detections['detection_scores'][i], detections['detection_classes'][i], detections['num_detections'][i], iou_threshold=0.5, score_threshold=0.5)
    # ... further processing for each image ...

```

**Commentary:** This example showcases batch processing for improved efficiency.  The `np.stack` function combines individual images into a single tensor that the model can process. The post-processing loop iterates through each element of the batch, extracting relevant detection data for each image independently.


**3. Resource Recommendations:**

The official TensorFlow Object Detection API documentation.  TensorFlow's core documentation, covering model loading and tensor manipulation.  A comprehensive textbook on deep learning, particularly chapters on object detection and model deployment.  Finally, research papers describing the specific object detection model architecture used.  Understanding the architecture details is paramount for proper pre- and post-processing.
