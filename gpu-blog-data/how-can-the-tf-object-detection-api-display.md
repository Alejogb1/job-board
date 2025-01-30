---
title: "How can the TF Object Detection API display object names to the console?"
date: "2025-01-30"
id: "how-can-the-tf-object-detection-api-display"
---
The TensorFlow Object Detection API, while primarily geared towards visual output, does not inherently offer direct console printing of detected object names during inference. This is by design; the API is structured to return bounding box coordinates and class IDs, not human-readable labels as a default. To achieve console display of object names, one must augment the inference process with label map translation.

My experience building real-time object detection systems for industrial automation has made this a common requirement. We needed detailed logs of detected objects to fine-tune our models and diagnose intermittent performance issues, thus prompting the need for console output. The core issue stems from the way the API handles classifications: it returns numerical class indices, which must be mapped back to the string representations contained within a label map. The label map is crucial as it’s the bridge that translates the numeric class ID from the prediction output to its corresponding object name.

The standard API workflow involves loading a trained model, preprocessing input images, feeding them to the model for inference, and interpreting the output tensors. These tensors consist of bounding boxes (often normalized), class scores, and class indices. The absence of object names stems from how these indices are used internally within the TF framework.

To display object names in the console, the necessary steps involve retrieving the detection result, parsing the returned class IDs, and using the label map data to perform the required look-up operation. These steps must occur after the inference step is completed. The code implementation can utilize Python's native data structures, specifically dictionaries, to efficiently perform the mapping between class index and label.

Here is an example of how one can achieve this with a loaded TensorFlow detection model:

```python
import tensorflow as tf
import numpy as np

# Assume model is loaded as 'model' and label_map_path points to a valid pbtxt file
def load_label_map(label_map_path):
    """Loads label map from pbtxt file and returns a dictionary."""
    label_map = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            if 'id:' in line:
                id_ = int(line.split(': ')[1])
            elif 'name:' in line:
                name = line.split(': ')[1].strip().replace("'", "")
                label_map[id_] = name
    return label_map

def display_detections(detection_result, label_map):
    """Parses the detection output and displays object names."""
    num_detections = int(detection_result.pop('num_detections'))
    detections = {key: detection_result[key][:num_detections].numpy() for key in detection_result}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    for i in range(num_detections):
      class_id = detections['detection_classes'][i]
      score = detections['detection_scores'][i]
      if score > 0.5: # Confidence Thresholding
        object_name = label_map.get(class_id, 'Unknown')
        print(f"Detected: {object_name}, Confidence: {score:.4f}")


# Example usage
label_map_path = "path/to/your/label_map.pbtxt"
label_map = load_label_map(label_map_path)
# Replace 'input_tensor' with your image data.
input_tensor = tf.random.uniform(shape=[1,640,640,3], dtype=tf.float32) #Placeholder data
detections = model(input_tensor)
display_detections(detections, label_map)
```

This code snippet first loads the label map by parsing the `.pbtxt` file, extracting both the numerical class IDs and the corresponding string names. The `load_label_map` function does this and creates a dictionary that is used in `display_detections`. Then `display_detections` processes the detection result, iterates through each detected object above a threshold of 0.5, looks up the class name via the `label_map` dictionary, and outputs the name along with its confidence score to the console. The initial `input_tensor` is merely placeholder example input. In any real scenario, image preprocessing will need to occur and replace this with your processed input.

The label map is often a `.pbtxt` file and might resemble the following structure:
```
item {
  id: 1
  name: 'person'
}
item {
  id: 2
  name: 'car'
}
item {
  id: 3
  name: 'bicycle'
}
...
```

A more complex example showcases how to process multiple images in a batch, which is a common optimization technique when running inference on GPUs or TPUs:

```python
import tensorflow as tf
import numpy as np

def load_label_map(label_map_path):
    """Loads label map from pbtxt file and returns a dictionary."""
    label_map = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            if 'id:' in line:
                id_ = int(line.split(': ')[1])
            elif 'name:' in line:
                name = line.split(': ')[1].strip().replace("'", "")
                label_map[id_] = name
    return label_map

def display_batch_detections(detection_result, label_map):
    """Parses batch detection output and displays object names."""
    num_detections = int(detection_result.pop('num_detections')[0]) #num detections in entire batch, not per image.
    detections = {key: detection_result[key][:num_detections].numpy() for key in detection_result}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Number of detections is across the batch, not per image
    i = 0
    while i < num_detections:
        class_id = detections['detection_classes'][i]
        score = detections['detection_scores'][i]
        if score > 0.5: # Confidence thresholding
            object_name = label_map.get(class_id, 'Unknown')
            print(f"Detected: {object_name}, Confidence: {score:.4f}, Batch ID: {i//(detections['detection_classes'].shape[0]//detection_result['num_detections'].shape[1])}")
        i+=1


# Example usage
label_map_path = "path/to/your/label_map.pbtxt"
label_map = load_label_map(label_map_path)
# Replace 'input_tensor' with your image batch data.
input_tensor = tf.random.uniform(shape=[4,640,640,3], dtype=tf.float32) #Place holder data for batch size 4
detections = model(input_tensor)
display_batch_detections(detections, label_map)
```

This adaptation of the previous example is made for handling batched inference scenarios. The main changes are that the `num_detections` variable is accessed using `[0]` since it’s a tensor representing all detections in the batch rather than a single scalar. The output also includes which batch ID the detection pertains to, which is not a straightforward mapping; it relies on the `detection_classes` shape and the shape of `num_detections` tensor to derive this. It shows an important consideration in batch inference: detected objects from a single image are not contiguous.

Additionally, this can be made more robust to scenarios where a label map may be missing entries by adding error handling:

```python
import tensorflow as tf
import numpy as np

def load_label_map(label_map_path):
    """Loads label map from pbtxt file and returns a dictionary."""
    label_map = {}
    try:
        with open(label_map_path, 'r') as f:
            for line in f:
                if 'id:' in line:
                    id_ = int(line.split(': ')[1])
                elif 'name:' in line:
                    name = line.split(': ')[1].strip().replace("'", "")
                    label_map[id_] = name
    except FileNotFoundError:
        print(f"Error: Label map file not found at {label_map_path}")
        return None
    return label_map

def display_detections(detection_result, label_map):
    """Parses the detection output and displays object names."""
    if not label_map:
        print("Error: Label map not loaded. Cannot display object names.")
        return
    num_detections = int(detection_result.pop('num_detections'))
    detections = {key: detection_result[key][:num_detections].numpy() for key in detection_result}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    for i in range(num_detections):
      class_id = detections['detection_classes'][i]
      score = detections['detection_scores'][i]
      if score > 0.5: # Confidence Thresholding
        object_name = label_map.get(class_id, 'Unknown')
        print(f"Detected: {object_name}, Confidence: {score:.4f}")


# Example usage
label_map_path = "path/to/your/label_map.pbtxt"
label_map = load_label_map(label_map_path)
if label_map is not None: #Ensure label map exists.
    # Replace 'input_tensor' with your image data.
    input_tensor = tf.random.uniform(shape=[1,640,640,3], dtype=tf.float32) #Placeholder data
    detections = model(input_tensor)
    display_detections(detections, label_map)

```

This version adds error handling using a `try-except` block to catch `FileNotFoundError` when the label map file does not exist. It also adds a check to ensure `label_map` is not `None` before the program proceeds with inference and output. Such precautions help ensure a robust and resilient system. These are simple error handlers, and more complex systems would require more sophisticated error recovery mechanisms.

For further study, I recommend reviewing the official TensorFlow documentation for the Object Detection API which explains the underlying mechanisms. Additionally, exploring examples on GitHub and similar platforms can provide concrete implementations and practical guidance.  Consultation of the core TensorFlow API documentation is also advisable for a more detailed understanding of the tensor operations involved. These resources are useful for building robust and reliable object detection applications. Understanding the data structures returned from the inference stage as well as the model configurations is essential for successfully displaying the object names in console.
