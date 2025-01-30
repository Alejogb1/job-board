---
title: "How do I extract class and bounding box coordinates from YOLOv5 ONNX predictions?"
date: "2025-01-30"
id: "how-do-i-extract-class-and-bounding-box"
---
YOLOv5's ONNX export facilitates efficient inference outside the PyTorch ecosystem, but extracting meaningful data from the raw output requires careful understanding of the model's architecture and the ONNX runtime's interface.  My experience deploying YOLOv5 models in production environments has highlighted the need for robust, error-handled post-processing of these predictions.  The key lies in understanding the structure of the ONNX output tensor, which represents class probabilities and bounding box parameters in a specific format.  This format consistently reflects the number of detection classes and the number of bounding boxes predicted per image.

**1.  Explanation of the ONNX Output Tensor Structure**

The ONNX runtime's inference result for a YOLOv5 model typically returns a single tensor.  This tensor's dimensions are determined by the model configuration; however, they follow a consistent pattern.  The dimensions usually follow this pattern:  `[batch_size, num_detections, 6]`, where `batch_size` is the number of input images, `num_detections` is the maximum number of bounding boxes the model predicts per image, and `6` represents the data for each detection:

* **Element 0:** Confidence score (probability that a detection represents an object).
* **Element 1:** Class ID (index of the predicted class from the model's class list).
* **Element 2:** x-coordinate of the center of the bounding box (normalized to image width).
* **Element 3:** y-coordinate of the center of the bounding box (normalized to image height).
* **Element 4:** width of the bounding box (normalized to image width).
* **Element 5:** height of the bounding box (normalized to image height).

It's crucial to remember that the `num_detections` dimension might contain predictions with low confidence scores, representing potential false positives.  Filtering these low-confidence predictions is a necessary post-processing step.  Furthermore, normalized coordinates need to be converted back to pixel coordinates using the image's width and height.

**2. Code Examples with Commentary**

The following examples demonstrate extraction and processing in Python using `onnxruntime`.  I'll assume the necessary libraries are already installed (`onnxruntime`, `numpy`).  Error handling is deliberately included to ensure robustness.

**Example 1: Basic Extraction and Filtering**

This example focuses on extracting and filtering predictions, demonstrating the core process:

```python
import onnxruntime as ort
import numpy as np

def process_yolo_predictions(onnx_path, image_path, confidence_threshold=0.5):
    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        img = # Load and preprocess image here (e.g., using OpenCV)  - omitted for brevity

        # Perform inference
        output = sess.run(None, {input_name: img})[0]

        # Filter predictions based on confidence threshold
        filtered_predictions = output[output[:, 0] >= confidence_threshold]

        return filtered_predictions
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

predictions = process_yolo_predictions("path/to/yolov5.onnx", "path/to/image.jpg")
if predictions is not None:
    print(predictions)
```

**Example 2: Conversion to Pixel Coordinates**

This expands on Example 1, transforming normalized coordinates into pixel coordinates:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

def process_yolo_predictions_with_coordinates(onnx_path, image_path, confidence_threshold=0.5):
    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        img = np.array(Image.open(image_path))
        img_height, img_width = img.shape[:2]  # Get image dimensions
        img = # preprocess image as required - omitted for brevity
        output = sess.run(None, {input_name: img})[0]

        filtered_predictions = output[output[:, 0] >= confidence_threshold]

        # Convert normalized coordinates to pixel coordinates
        x_centers = (filtered_predictions[:, 2] * img_width).astype(int)
        y_centers = (filtered_predictions[:, 3] * img_height).astype(int)
        widths = (filtered_predictions[:, 4] * img_width).astype(int)
        heights = (filtered_predictions[:, 5] * img_height).astype(int)

        # Calculate bounding box corners
        x_min = x_centers - (widths // 2)
        y_min = y_centers - (heights // 2)
        x_max = x_min + widths
        y_max = y_min + heights

        #Combine results into a structured array
        return np.column_stack((filtered_predictions[:,1], x_min, y_min, x_max, y_max))

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

predictions = process_yolo_predictions_with_coordinates("path/to/yolov5.onnx", "path/to/image.jpg")
if predictions is not None:
  print(predictions)

```

**Example 3: Class Names and Bounding Box Visualization**

This example incorporates class labels and demonstrates a basic visualization (requires `matplotlib`):

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_detections(onnx_path, image_path, confidence_threshold=0.5, class_names=["class1", "class2", ...]): # Replace with your class names
    try:
        # ... (Inference and coordinate conversion from Example 2) ...

        fig, ax = plt.subplots(1)
        img = Image.open(image_path)
        ax.imshow(img)

        for detection in predictions:
            class_id, x_min, y_min, x_max, y_max = detection
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, class_names[int(class_id)], color='r')

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

visualize_detections("path/to/yolov5.onnx", "path/to/image.jpg", class_names=['person', 'car', 'bicycle'])
```


**3. Resource Recommendations**

The ONNX Runtime documentation is essential.  Understanding NumPy array manipulation is also vital for efficient processing of the prediction tensor.  Furthermore, a comprehensive guide on image processing and preprocessing techniques would be beneficial, especially regarding the specific requirements of YOLOv5.  Finally, the original YOLOv5 repository provides valuable insights into the model's architecture and expected outputs.  Proper attention to these resources will drastically improve your understanding and ability to work with YOLOv5 ONNX models.
