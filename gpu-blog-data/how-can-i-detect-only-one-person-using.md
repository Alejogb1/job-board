---
title: "How can I detect only one person using imageai.Detection?"
date: "2025-01-30"
id: "how-can-i-detect-only-one-person-using"
---
The core challenge in using imageai.Detection to identify a *single* person lies in understanding its output format and implementing logic to filter results. The library, by default, returns a list of detections, potentially containing multiple humans in a given image, alongside various other object classes. Therefore, achieving single person detection necessitates post-processing the raw output.

My experience, particularly in embedded systems where limited computational resources demand precise detection, has shown the necessity for tailored processing beyond the standard examples. I initially encountered this issue when developing a real-time occupancy monitor utilizing a camera feed. The goal was to trigger an alert when *exactly one* person was present, not when a crowd appeared.

The fundamental steps involve:

1. **Performing Detection:** Employing `imageai.Detection` with a pretrained model (e.g., YOLOv3, RetinaNet) to obtain raw detection results.

2. **Filtering by Class:** Extracting only the detections classified as 'person'. This eliminates non-human objects, narrowing down the possible detections.

3. **Counting Valid Detections:** Determining the number of identified 'person' instances within the filtered results.

4. **Conditional Logic:** Implementing logic based on the count. The desired outcome is to proceed only if the count is exactly 1.

The `detectObjectsFromImage` method, without modification, simply returns a list. Each item in this list represents a detection. Therefore, extracting the required information requires inspecting the dictionaries within the list. The dictionaries contain keys like 'name' (the class detected), 'box_points' (bounding box coordinates) and 'percentage_probability' (detection confidence).

Here are three code examples illustrating the process, progressing in complexity:

**Example 1: Basic Person Detection and Count**

```python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.jpg"), output_image_path=os.path.join(execution_path, "image_output.jpg"), minimum_percentage_probability=30)

person_count = 0
for detection in detections:
    if detection['name'] == 'person':
        person_count += 1

print(f"Number of people detected: {person_count}")
if person_count == 1:
    print("Exactly one person detected.")
```

This initial example demonstrates the fundamental extraction and counting of 'person' detections. The detector is instantiated with a local model path (it is assumed that the 'yolov3.pt' model file exists within the current working directory) and an image named 'image.jpg' is used as input. All detected objects with a confidence over 30% are stored in the `detections` variable.  The code iterates through the returned detections, and for every one classified as 'person', the `person_count` is incremented. A simple conditional statement then outputs whether exactly one person was detected. While functional, this approach lacks robustness as it makes no checks on the validity of the loaded model or input image.

**Example 2: Error Handling and Bounding Box Access**

```python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

try:
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
    detector.loadModel()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.jpg"), output_image_path=os.path.join(execution_path, "image_output.jpg"), minimum_percentage_probability=30)
except Exception as e:
    print(f"Error during detection: {e}")
    exit()

person_detections = [detection for detection in detections if detection['name'] == 'person']
person_count = len(person_detections)


if person_count == 1:
    bounding_box = person_detections[0]['box_points']
    print(f"Exactly one person detected with bounding box: {bounding_box}")
else:
    print(f"Number of people detected: {person_count}")
```

This second example introduces error handling using `try-except` blocks, addressing potential issues during model loading and image detection. Additionally, it demonstrates how to retrieve bounding box coordinates if a single person is detected. A list comprehension is used to filter out the people detections from the list of all detections. Once that has completed, the code checks if the length of `person_detections` is 1 and prints the bounding box coordinates if this is the case.  This example is more robust in handling potential issues and provides additional data extraction capabilities.

**Example 3: Confidence Thresholding and Rejection Criteria**

```python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

try:
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
    detector.loadModel()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.jpg"), output_image_path=os.path.join(execution_path, "image_output.jpg"), minimum_percentage_probability=50)
except Exception as e:
    print(f"Error during detection: {e}")
    exit()

person_detections = [detection for detection in detections if detection['name'] == 'person' and detection['percentage_probability'] >= 70]

person_count = len(person_detections)


if person_count == 1:
    print("Exactly one person detected with high confidence.")
elif person_count > 1:
    print("Multiple people detected, rejecting.")
else:
    print("No high confidence person detections.")
```

This final example extends the previous ones by incorporating confidence thresholds. Here, only detections with a 'person' label *and* a probability of 70% or higher are included in the `person_detections` list. The `detectObjectsFromImage` function is also passed a `minimum_percentage_probability` of 50. This provides additional control over the quality of detections. This version also shows a more elaborate response in cases where there is not exactly one person detected.  This final iteration emphasizes the importance of confidence filtering when performing image detection.

In my experiences, adjusting the `minimum_percentage_probability` during the initial detection phase and implementing a second filtering stage based on the probability of the person detection helps to improve results. The optimal configuration often depends on the specifics of the model and the image conditions. While the library provides defaults, fine-tuning the probability thresholds based on the application is a standard practice.

For further study, I recommend consulting the official ImageAI documentation for detailed information on model loading, input/output formats, and detection parameters. Additionally, explore resources detailing common object detection metrics like precision, recall, and F1-score. This will aid in optimizing detection parameters. Lastly, a review of fundamental Python programming principles, such as list comprehensions, error handling, and data structure manipulation is always valuable. This knowledge is crucial for post-processing the output. These resources will provide a thorough foundation for effectively utilizing and customizing `imageai.Detection` for specific use cases like the one discussed.
