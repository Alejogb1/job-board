---
title: "How can I obtain object locations and prediction scores using Keras?"
date: "2025-01-30"
id: "how-can-i-obtain-object-locations-and-prediction"
---
Specifically, focus on use cases with object detection models.

Deep learning models trained for object detection, unlike simple classification models, output not just a class label but also the bounding box coordinates defining the object's location within an image and a confidence score indicating the model's certainty about the prediction. Accessing these values programmatically in Keras requires understanding the specific output structure of the model and then parsing this structure appropriately. From my experience deploying several object detection systems, I’ve found that interpreting these outputs correctly is essential for downstream tasks such as tracking, annotation, and analysis.

The output format of an object detection model is highly dependent on the architecture and its training regime. Models built using the Keras framework, especially those leveraging pre-trained backbones, often employ formats aligned with common bounding box representations. Specifically, the output often takes the form of a tensor, where the final dimension encodes the predicted bounding boxes, their corresponding classes, and the confidence scores. The precise order and arrangement within this dimension, however, varies. A typical arrangement includes box coordinates (x1, y1, x2, y2, or cx, cy, width, height), a class probability distribution, and an objectness score. The objectness score represents the probability that a box contains any object at all. The class probabilities, when present, then refine this with the probability of belonging to a specific category. This structured output enables us to pinpoint regions of interest and associate a prediction with each. Extracting this information involves slicing the tensor and interpreting the indices accordingly.

It is crucial to remember that the output from the raw model frequently requires post-processing steps such as non-maximum suppression (NMS) to eliminate redundant bounding boxes and apply a confidence threshold. NMS essentially removes bounding boxes with significant overlap and lower confidence scores within the same class. Failing to implement these procedures correctly can result in duplicate or false positive detections.

Here are three practical examples detailing how to extract object locations and prediction scores from a Keras object detection model:

**Example 1: Extracting Bounding Boxes and Scores from a Simple Model Output**

Assume a model, named `detector_model`, outputs a tensor of shape `(batch_size, num_detections, 6)`, where the last dimension contains `[x1, y1, x2, y2, score, class_id]`. This assumes a bounding box defined by top-left and bottom-right corners, followed by a detection confidence score and class identifier, and no objectness score.

```python
import tensorflow as tf
import numpy as np

# Assume detector_model is already loaded or created
# Mock output for demonstration
batch_size = 1
num_detections = 5
mock_output = tf.random.uniform(shape=(batch_size, num_detections, 6)) # Simulate bounding boxes, scores, class_id
detector_output = mock_output

for batch_idx in range(detector_output.shape[0]):
    batch_detections = detector_output[batch_idx]

    for detection_idx in range(batch_detections.shape[0]):
        detection = batch_detections[detection_idx]
        x1, y1, x2, y2, score, class_id = detection.numpy()
        print(f"Detection {detection_idx}:")
        print(f"  Bounding Box: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"  Score: {score:.4f}")
        print(f"  Class ID: {int(class_id)}")

```

In this example, I iterate through each detection in the batch, extract the bounding box coordinates (x1, y1, x2, y2), confidence score, and class ID using simple array indexing and convert the elements to numpy for readability. This is a fundamental approach suitable when the model output is already structured directly for bounding boxes with class IDs. Here, `mock_output` simulates model output. In a real system, `detector_output` would be the output from a `detector_model.predict(input_image)`.

**Example 2: Handling Output with Objectness and Class Probabilities**

Consider a model output tensor of shape `(batch_size, num_boxes, num_classes + 5)`, where the last dimension represents `[x_center, y_center, width, height, objectness, class_1, class_2,...,class_n]`. This includes objectness score and class probabilities, requiring us to select the class with the maximum probability.

```python
import tensorflow as tf
import numpy as np

# Assume detector_model is already loaded or created
# Mock output for demonstration
batch_size = 1
num_boxes = 5
num_classes = 3
mock_output = tf.random.uniform(shape=(batch_size, num_boxes, num_classes + 5)) # Simulate bounding boxes, scores, class_probs
detector_output = mock_output

confidence_threshold = 0.5 # Filter out low confidence detections

for batch_idx in range(detector_output.shape[0]):
    batch_detections = detector_output[batch_idx]

    for detection_idx in range(batch_detections.shape[0]):
      detection = batch_detections[detection_idx]
      x_center, y_center, width, height, objectness, *class_probabilities = detection.numpy()

      if objectness > confidence_threshold: # Only proceed if there's a high likelihood of an object

        class_probabilities = np.array(class_probabilities)
        class_id = np.argmax(class_probabilities)
        score = class_probabilities[class_id] # Score is the class probability


        # Convert cx,cy,w,h to x1,y1,x2,y2 if needed
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        print(f"Detection {detection_idx}:")
        print(f"  Bounding Box: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"  Score: {score:.4f}")
        print(f"  Class ID: {int(class_id)}")

```

In this instance, I filter detections by applying a confidence threshold to the objectness score, then identify the most probable class using `argmax` on the class probability array. I also include code that converts center-based bounding boxes to coordinate-based bounding boxes for completeness.

**Example 3: Incorporating Non-Maximum Suppression**

Here, I show how to perform NMS to remove overlapping bounding boxes with lower confidence scores. I’ll reuse the output structure of example 2, but focus on the postprocessing steps. TensorFlow offers a convenient method to handle this.

```python
import tensorflow as tf
import numpy as np

# Assume detector_model is already loaded or created
# Mock output for demonstration
batch_size = 1
num_boxes = 5
num_classes = 3
mock_output = tf.random.uniform(shape=(batch_size, num_boxes, num_classes + 5)) # Simulate bounding boxes, scores, class_probs
detector_output = mock_output

confidence_threshold = 0.5
iou_threshold = 0.4 # Intersection over Union threshold

for batch_idx in range(detector_output.shape[0]):
    batch_detections = detector_output[batch_idx]
    
    # Preprocess data and store boxes, scores and classes separately.
    boxes = []
    scores = []
    classes = []
    for detection_idx in range(batch_detections.shape[0]):
      detection = batch_detections[detection_idx]
      x_center, y_center, width, height, objectness, *class_probabilities = detection.numpy()

      if objectness > confidence_threshold:
        class_probabilities = np.array(class_probabilities)
        class_id = np.argmax(class_probabilities)
        score = class_probabilities[class_id]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes.append([y1, x1, y2, x2])
        scores.append(score)
        classes.append(class_id)


    # Convert lists to tensor format for NMS operation
    boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores_tensor = tf.convert_to_tensor(scores, dtype=tf.float32)
    classes_tensor = tf.convert_to_tensor(classes, dtype = tf.int32)


    # Apply non-maximum suppression
    selected_indices = tf.image.non_max_suppression(boxes_tensor, scores_tensor, max_output_size=len(boxes), iou_threshold=iou_threshold)

    # Loop through the selected indices
    for i in selected_indices:
      x1, y1, x2, y2 = boxes_tensor[i].numpy()
      score = scores_tensor[i].numpy()
      class_id = classes_tensor[i].numpy()
      print(f"Detection {i}:")
      print(f"  Bounding Box: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
      print(f"  Score: {score:.4f}")
      print(f"  Class ID: {int(class_id)}")
```

This example demonstrates an essential post-processing step. I first filter by objectness scores as before. Then,  I prepare the bounding box coordinates, scores and class IDs for NMS by converting to tensors. The `tf.image.non_max_suppression` function does the heavy lifting of removing overlapping detections based on an IOU threshold, returning only indices of the selected boxes. The remaining code then parses these selected detections in the way detailed previously.

For anyone building object detection systems, I strongly advise familiarizing yourself with resources that dive deeper into object detection architectures and common output formats. Texts focusing on computer vision with deep learning will offer insights into the theoretical background. A good sourcebook on the use of TensorFlow will be essential for the practical aspects. Furthermore, research papers introducing specific models (e.g. YOLO, SSD, Faster R-CNN) should be consulted to grasp architecture specifics. Examining model implementations within open-source projects can also improve understanding and provide valuable insights. Finally, I would recommend investigating any documentation or examples that come with pre-trained object detection models or detection modules.
