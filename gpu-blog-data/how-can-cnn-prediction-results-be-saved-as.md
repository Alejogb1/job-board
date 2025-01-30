---
title: "How can CNN prediction results be saved as images?"
date: "2025-01-30"
id: "how-can-cnn-prediction-results-be-saved-as"
---
Saving CNN prediction results as images requires careful consideration of the output format and the desired visual representation.  My experience in developing medical image analysis pipelines highlighted the importance of a robust and flexible solution, capable of handling diverse prediction types.  The key is to understand that the raw output of a CNN is typically a numerical array, representing probabilities or feature maps, which must be transformed into a visually interpretable format.


**1.  Explanation:**

The process involves several stages. First, we obtain the CNN's prediction, which may be a class probability vector (for classification tasks) or a feature map (for segmentation or object detection).  Then, we design a visualization scheme appropriate for the type of prediction.  For classification, a simple bar chart or a text overlay on the input image might suffice.  For segmentation, a color-coded overlay is commonly used, mapping predicted classes to specific colors.  Object detection necessitates drawing bounding boxes around detected objects, accompanied by class labels and confidence scores.  Finally, we leverage libraries like Matplotlib, OpenCV, or Pillow to render the visualized prediction data as an image file (e.g., PNG, JPG). The choice of library depends on efficiency requirements and desired features.  In my work processing large datasets of high-resolution medical scans, OpenCV's performance advantages were crucial.


**2. Code Examples:**

**Example 1: Classification Task Visualization**

This example shows how to visualize a classification result by overlaying the predicted class label and confidence score onto the input image.  I've used this approach extensively in my work on classifying microscopic images of cancerous cells.


```python
import cv2
import numpy as np

def visualize_classification(image_path, prediction, class_labels):
    """Visualizes classification prediction on an image.

    Args:
        image_path: Path to the input image.
        prediction: A NumPy array representing class probabilities.
        class_labels: A list of class labels corresponding to the prediction.
    """
    image = cv2.imread(image_path)
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    label = f"{class_labels[predicted_class]}: {confidence:.2f}"

    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("classification_result.png", image)


#Example Usage
image_path = "input_image.jpg"
prediction = np.array([0.1, 0.8, 0.1]) #Example prediction: high probability for class 1
class_labels = ["Class A", "Class B", "Class C"]
visualize_classification(image_path, prediction, class_labels)
```

This function takes the image path, prediction array, and class labels as input. It identifies the class with the highest probability, calculates confidence, and overlays this information onto the image using OpenCV's `putText` function before saving it as "classification_result.png".


**Example 2: Semantic Segmentation Visualization**

This example demonstrates visualizing semantic segmentation results, a technique I often applied when segmenting brain tumors in MRI scans.  A color map is crucial here to represent different segmented regions.


```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(segmentation_map, color_map):
    """Visualizes a semantic segmentation map.

    Args:
        segmentation_map: A NumPy array representing the segmentation map.
        color_map: A NumPy array representing the color map.  Shape should be (num_classes, 3)
    """
    colored_segmentation = color_map[segmentation_map]
    plt.imshow(colored_segmentation.astype(np.uint8))
    plt.axis('off')
    plt.savefig("segmentation_result.png")

#Example Usage
segmentation_map = np.random.randint(0, 3, size=(100,100)) #Example segmentation map
color_map = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) # Red, Green, Blue for 3 classes
visualize_segmentation(segmentation_map, color_map)
```

This function uses Matplotlib to create a colored image from the segmentation map and a predefined color map.  The `astype(np.uint8)` conversion ensures correct image format.  The `axis('off')` removes axis ticks and labels from the saved image.



**Example 3: Object Detection Visualization**

For object detection, bounding boxes and labels need to be drawn.  This example mirrors the techniques I employed in a project identifying various types of vehicles in satellite imagery.


```python
import cv2

def visualize_object_detection(image_path, detections):
    """Visualizes object detection results.

    Args:
        image_path: Path to the input image.
        detections: A list of dictionaries, each containing 'bbox' (bounding box coordinates), 'class_label', and 'confidence'.
    """
    image = cv2.imread(image_path)
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class_label']}: {detection['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite("detection_result.png", image)


# Example Usage
image_path = "input_image.jpg"
detections = [
    {'bbox': (100, 100, 200, 200), 'class_label': 'Car', 'confidence': 0.95},
    {'bbox': (300, 150, 400, 250), 'class_label': 'Truck', 'confidence': 0.80}
]
visualize_object_detection(image_path, detections)

```

This function iterates through the detection results, drawing bounding boxes and labels using OpenCV functions. The coordinates (x1, y1, x2, y2) define the box's top-left and bottom-right corners.


**3. Resource Recommendations:**

For in-depth understanding of CNN architectures and implementation, consult established textbooks on deep learning and computer vision.  Explore the documentation of relevant Python libraries such as NumPy, Matplotlib, OpenCV, and Pillow.  Furthermore, reviewing research papers on visualization techniques specific to your task (classification, segmentation, or object detection) will provide valuable insights and potentially enhance your visualization strategies.  Finally, I highly recommend examining the source code of existing open-source projects involving image processing and CNN visualization, as this provides practical examples and code patterns.
