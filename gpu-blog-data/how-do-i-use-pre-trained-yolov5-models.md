---
title: "How do I use pre-trained YOLOv5 models?"
date: "2025-01-30"
id: "how-do-i-use-pre-trained-yolov5-models"
---
YOLOv5, standing for You Only Look Once version 5, is a state-of-the-art object detection model known for its speed and accuracy. Its pre-trained weights offer a significant advantage, eliminating the need for extensive training from scratch, especially beneficial when computational resources are limited. I've leveraged these pre-trained models extensively in several projects, from real-time drone surveillance to automated quality inspection systems, and I've found understanding their implementation to be relatively straightforward, albeit with important nuances.

The core of utilizing a pre-trained YOLOv5 model centers around loading the model architecture with its pre-existing learned weights and subsequently processing new input images to generate bounding boxes and associated class probabilities. The process generally follows these steps: 1) setting up the environment, ensuring necessary dependencies are installed; 2) loading the model with desired weights; 3) loading and preparing input images or video; 4) passing the input data through the model; 5) processing the output into useful information; and 6) visualizing or utilizing the results.

The primary library used for this purpose is PyTorch, and the process is facilitated by the official YOLOv5 repository, typically accessed via `git clone`. The environment requires installing PyTorch alongside other dependencies listed in the repository’s requirements file. Once the environment is set, accessing the pre-trained models is as simple as referencing them by name within the PyTorch environment; the official repository handles the download automatically. The models come pre-trained on the COCO dataset, consisting of 80 object classes. It’s important to note that while retraining on custom data can improve performance on specialized applications, the pre-trained COCO weights often provide reasonable results in many general object detection scenarios out-of-the-box.

Let’s examine specific code examples to illustrate the usage.

**Example 1: Basic Inference on a Single Image**

This example demonstrates the fundamental process of loading a YOLOv5 model and running inference on a single image.

```python
import torch

# Load the pre-trained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image for inference
img_path = 'path/to/your/image.jpg' # Replace with your actual image path
results = model(img_path)

# Print the results
print(results.pandas().xyxy[0]) # Print detections as pandas DataFrames

# Save the image with drawn bounding boxes
results.save(save_dir="output_images")
```
Here, the `torch.hub.load` function loads the 'yolov5s' model, the smallest version, from the official Ultralytics repository. `pretrained=True` ensures the pre-trained weights are also loaded. This loads the model into the CPU by default; for GPU usage, you'd need to ensure your PyTorch installation supports it, and you would transfer the model to the GPU using `.to('cuda')`. The image is loaded from the provided `img_path` and passed to the model, generating output which is stored in the `results` object. The `results.pandas().xyxy[0]` extracts bounding box information and class probabilities into a Pandas DataFrame for easy manipulation. This can be customized based on user need. Additionally, the `results.save` line generates an image with drawn bounding boxes based on the identified objects.

**Example 2: Inference with Custom Confidence Threshold**

Object detection involves trade-offs between precision and recall. Sometimes, default thresholds for confidence scores can lead to either excessive false positives or the omission of some true objects. This example shows how to control that using the `conf` parameter.

```python
import torch

# Load the pre-trained YOLOv5m model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Load an image for inference
img_path = 'path/to/your/image.jpg' # Replace with your actual image path

# Perform inference with a custom confidence threshold (e.g., 0.5)
results = model(img_path, conf=0.5)

# Print the results
print(results.pandas().xyxy[0])

# Save the image with drawn bounding boxes
results.save(save_dir="output_images")
```
In this case, we use the 'yolov5m' model, which is larger and typically more accurate than 'yolov5s', at the expense of increased computation time. The critical addition here is the `conf=0.5` argument within the model call. This tells the model to only return detections where the confidence score is greater than or equal to 0.5. Experimentation is often needed to find an appropriate threshold depending on application goals. Lowering the threshold increases recall (fewer objects missed), but increases false positives. Raising the threshold reduces false positives, but increases missed detections.

**Example 3: Inference on a Video Stream**

Often, object detection is applied to video. This example shows how to apply a YOLOv5 model to every frame in a video and store the results in a list of frame detections.

```python
import torch
import cv2

# Load the pre-trained YOLOv5l model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Load a video file
video_path = 'path/to/your/video.mp4' # Replace with your actual video path
cap = cv2.VideoCapture(video_path)

frame_detections = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on current frame
    results = model(frame)

    # Store the frame detections
    frame_detections.append(results.pandas().xyxy[0])


    # Optionally, you could also display the video with bounding boxes in real-time here using cv2.imshow
    # The bounding box drawing would be handled in similar fashion to the single image examples


cap.release()

# Process the stored frame_detections
# For instance, you could print the detection for each frame
for i, detections in enumerate(frame_detections):
  print(f"Frame {i}: \n {detections}")

```
Here we demonstrate processing a video using `cv2.VideoCapture`. The video is loaded, and each frame is processed individually by the YOLOv5 model. The results are extracted using `results.pandas().xyxy[0]` and appended to the `frame_detections` list. You can choose to display bounding boxes on a video stream in real time here; for brevity, we are not implementing real-time drawing here. After processing, the `frame_detections` list contains the detection results for each frame.

These examples illustrate the core elements of utilizing pre-trained YOLOv5 models. While I have presented the simplest approaches, several other options exist, such as manipulating the data loading process for improved performance or employing custom data augmentation techniques to further refine results. Additionally, you can further optimize detection by filtering based on the identified class, for example only looking for pedestrians.

For those seeking deeper understanding, I recommend exploring the official YOLOv5 documentation available on the Ultralytics GitHub repository. The repository provides thorough explanations of the model architecture, training methodologies, and code examples. Research papers detailing the evolution of YOLO (You Only Look Once) object detectors can provide a foundational understanding of their core mechanisms, allowing you to effectively integrate them into different applications. Consider also consulting practical guides and blog posts detailing real-world applications of YOLOv5. The PyTorch official documentation is also an invaluable resource, as it is the backbone for all code. These resources should allow you to confidently begin implementing these pre-trained models into your applications.
