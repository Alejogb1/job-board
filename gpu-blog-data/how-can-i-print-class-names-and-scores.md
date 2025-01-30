---
title: "How can I print class names and scores in TensorFlow Object Detection API?"
date: "2025-01-30"
id: "how-can-i-print-class-names-and-scores"
---
The TensorFlow Object Detection API stores class information as numerical indices, necessitating a mapping to human-readable labels for practical use during model evaluation or inference. Accessing these labels and corresponding scores within the API’s output requires an understanding of the prediction dictionaries returned by the model.

**Explanation of the Process**

The Object Detection API, after performing inference with a trained model, returns a dictionary of tensors. This dictionary contains crucial data regarding detected objects within an image, including bounding boxes, classification scores, and class indices. The class indices, integer values starting typically from zero, refer to the order of labels defined during model training. Crucially, these indices do not directly represent the human-readable class names, such as “cat,” “dog,” or “car.”

To obtain the corresponding class names, a label map is required. This label map is usually a protocol buffer file (`.pbtxt`) that stores a mapping between the numerical class index and its corresponding string label. This file was created during the model training setup and needs to be loaded and parsed for use during inference. Without loading and utilizing this label map, only the numerical indices, lacking the intended semantic information, are available.

The typical workflow involves the following steps: 1) Loading the exported inference graph which includes the model and its parameters; 2) Loading the label map protocol buffer; 3) Preparing the input image; 4) Feeding it to the model; 5) Retrieving the output prediction dictionary; and 6) Accessing and decoding class indices and scores using the loaded label map. Specifically, the output dictionary contains tensors like ‘detection_classes’ (containing class indices), ‘detection_scores’ (containing confidence scores), and ‘detection_boxes’ (containing bounding box coordinates). To print class names, we need to map the values of 'detection_classes' tensor using the label map and associate the resulting names with their corresponding scores from 'detection_scores' tensor.

The score values contained in the 'detection_scores' tensor represent a confidence value for each detected bounding box and its corresponding class. These scores typically range from 0 to 1, with higher values indicating higher confidence in the detection and classification accuracy. Generally, it is common to filter detected boxes and their scores based on a confidence threshold. This means setting a minimum score for a detection to be considered valid, and thus printed.

**Code Examples with Commentary**

Example 1 demonstrates the core process of loading the necessary components and accessing the detection outputs:

```python
import tensorflow as tf
from object_detection.utils import label_map_util
import numpy as np
import cv2

# 1. Path to the exported inference graph
PATH_TO_FROZEN_GRAPH = 'path/to/your/frozen_inference_graph.pb'

# 2. Path to the label map
PATH_TO_LABELS = 'path/to/your/label_map.pbtxt'

# 3. Load the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# 4. Load the label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# 5. Create a session and perform inference
with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        # Get input and output tensors
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


        # Load an example image (replace with actual image loading)
        image = cv2.imread('path/to/your/test_image.jpg')
        image_np = np.expand_dims(image, axis=0) # Expand for batched processing

        # Perform detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np})

        # Convert numpy tensors to usable format
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # The printing happens later in the next example.
```

In this first example, the code performs all the necessary loading, imports, and sets the stage to conduct inference. It loads the frozen graph, the label map, gets the input and output tensors, loads an image and finally performs the object detection task. This example does not extract human-readable labels yet, focusing on obtaining the raw model output. The use of `tf.compat.v1` is essential here because the Object Detection API, depending on its version, might require compatibility with Tensorflow 1.x API.

The second example shows how the class indices are mapped to class names using the loaded `category_index` and applies a score threshold:

```python
# Continuation from previous code block.
        # Iterate over detections and print those exceeding a threshold
        min_score_thresh = .5 # Minimum confidence for detection

        for i in range(boxes.shape[0]):
            if scores[i] > min_score_thresh:
              class_id = classes[i]
              class_name = category_index[class_id]['name']
              score = scores[i]
              print(f"Class: {class_name}, Score: {score:.4f}")
```

This segment builds upon the results from Example 1. A `min_score_thresh` value filters out detection boxes with low confidence. The code then iterates over the detected objects, retrieves the integer class id from `classes` tensor, utilizes the `category_index` to extract the human-readable `class_name`, and prints both the `class_name` and the corresponding confidence `score` with four decimal precision. This is how a label map connects the numerical index and the name, which makes the results understandable.

Finally, here's a third example to demonstrate how these classes could be integrated into a visual output:
```python
# Continuation from the previous code blocks.
        # Draw bounding boxes and labels
        image_with_detections = image.copy() # Create a copy so we don't modify original image
        for i in range(boxes.shape[0]):
          if scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = boxes[i]
            im_height, im_width, _ = image.shape
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            class_id = classes[i]
            class_name = category_index[class_id]['name']
            score = scores[i]

            cv2.rectangle(image_with_detections, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image_with_detections, label, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        cv2.imshow("Object Detection Results", image_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```
This final code block now takes the detected objects and overlays the bounding boxes, class names, and scores on the original image and displays it. This example shows the practicality of converting the raw tensor outputs to usable, understandable results by incorporating them into visualizations. This illustrates one potential next step for users after generating printed outputs.

**Resource Recommendations**

For an in-depth understanding of the TensorFlow Object Detection API, I recommend consulting the official TensorFlow documentation, which includes guides on training models, exporting inference graphs, and using the utilities. The API’s GitHub repository contains numerous examples, including Jupyter Notebook tutorials, which provide practical demonstrations of many functionalities.

Additionally, various online courses and tutorials cover object detection concepts and how to utilize them with TensorFlow. These resources often offer hands-on practice with sample datasets, solidifying understanding of the concepts and process. Further, examining other open-source projects that use the Object Detection API can give more insight into real-world application.
