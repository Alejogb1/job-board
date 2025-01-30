---
title: "How do I obtain a predicted image from a YOLOv5 model?"
date: "2025-01-30"
id: "how-do-i-obtain-a-predicted-image-from"
---
The core task of obtaining a predicted image from a YOLOv5 model involves processing an input image through the model's network, interpreting the resulting tensor outputs to identify bounding boxes and associated class probabilities, and then visualizing these detections on the original image. This isn't a straightforward "predict and display" function but a sequence of operations requiring careful handling of the model's output.

First, let's outline the general process. We begin with a loaded YOLOv5 model – I’ll assume it has been trained and is accessible, perhaps through PyTorch’s `torch.hub`. The input to the model is a preprocessed image, typically resized and normalized, represented as a PyTorch tensor. The output, also a tensor, holds the raw predictions: bounding box coordinates, objectness scores (how likely the box contains *any* object), and class probabilities. These raw outputs need to be filtered based on objectness and class probability thresholds to remove low-confidence detections. Then, a process called Non-Maximum Suppression (NMS) is applied to eliminate overlapping bounding boxes that may have detected the same object multiple times. Finally, the remaining bounding boxes are drawn onto a copy of the original input image, often with class labels and confidence scores for clarity.

My experience has shown this to be a critical step in any real-world deployment of a YOLOv5 model. A poorly handled prediction pipeline can result in significant performance degradation, such as false positives, missed detections, or significantly delayed inference.

Here’s a breakdown, with illustrative Python examples:

**Example 1: Raw Model Output and Preprocessing**

This code snippet demonstrates the initial steps: loading the model, preprocessing the input image, and obtaining the raw prediction tensor.

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Load a pretrained YOLOv5 model (adjust as needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # Put model in evaluation mode

def preprocess_image(image_path, img_size=640):
    """Preprocesses an image for YOLOv5 input."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Example usage
image_path = "test_image.jpg"
img_tensor = preprocess_image(image_path)

with torch.no_grad(): # Disable gradient calculation for inference
    outputs = model(img_tensor)

print(f"Shape of raw model output: {outputs.shape}") # Print shape of the tensor
print(f"Model Output Data type: {outputs.dtype}")  # Print the data type of output
```

Here, I use `torch.hub.load` to load a pre-trained `yolov5s` model. This is crucial for not starting from scratch, providing reasonable accuracy and speed. The `preprocess_image` function performs several key operations. First, it reads the image using PIL, converts it to RGB, resizes it to a square resolution (typically 640x640), transforms it to a tensor, and normalizes it. Normalization to mean and standard deviation is essential as it aligns the input data with how the model was trained. The `unsqueeze(0)` adds a batch dimension since the model expects a batch of images, even if it is a single image. Finally, the `model` call with a `torch.no_grad()` context performs inference and `outputs` holds the raw, unprocessed prediction. The prints provide useful information: the raw model output shape typically follows the format (batch_size, number of detections, 85), where 85 includes 4 bounding box coordinates (x_center, y_center, width, height), an objectness score, and 80 class probabilities for the standard COCO dataset, and the type is usually `torch.float32`.

**Example 2: Filtering and Non-Maximum Suppression**

This section illustrates how to filter low-confidence detections and apply Non-Maximum Suppression using `utils.general.non_max_suppression`.

```python
from utils.general import non_max_suppression

def filter_detections(outputs, conf_thres=0.25, iou_thres=0.45):
    """Filters detections using confidence and NMS."""

    # Apply Non-Maximum Suppression (NMS)
    filtered_outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=iou_thres)
    #The output of non_max_suppression is a list of tensors (each tensor corresponds to an image from batch)

    return filtered_outputs

# Example Usage
conf_thres = 0.4
iou_thres = 0.5

filtered_outputs = filter_detections(outputs, conf_thres, iou_thres)

if filtered_outputs[0] is not None: # Check if detections found
    print(f"Filtered detections shape: {filtered_outputs[0].shape}")
    print(f"Number of Detections After filtering: {len(filtered_outputs[0])}")
else:
    print("No detections found after filtering")

```

Here, `non_max_suppression` from YOLOv5's `utils.general` module does the heavy lifting. It takes the raw model output (`outputs`), applies confidence thresholding by rejecting bounding boxes where objectness score is below `conf_thres`, and then applies NMS, removing overlapping bounding boxes based on an intersection-over-union threshold of `iou_thres`. I typically adjust these thresholds based on the application and dataset; for instance, a higher `conf_thres` is often beneficial in applications needing very low false positive rates. It's important to note that the output of `non_max_suppression` is a *list of tensors,* where each tensor contains all the detections for the corresponding image from the batch. In this case, since I used one input image, the list will contain only one tensor, accessible via `filtered_outputs[0]`. The `if` check avoids issues when no detections are found for a specific image, which is quite common in real scenarios.

**Example 3: Bounding Box Visualization**

Finally, this shows how to draw bounding boxes on the original image, using the filtered detections.

```python
import cv2
def draw_bounding_boxes(image_path, filtered_outputs, class_names=None):
    """Draws bounding boxes on an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2 reads images in BGR format
    # check for detections
    if filtered_outputs[0] is not None:

        detections = filtered_outputs[0].cpu().numpy() # move to cpu if needed and convert to numpy

        for *xyxy, conf, cls in detections: # Iterating over detections for a single image
            xyxy = np.array(xyxy).astype(int)

            x1, y1, x2, y2 = xyxy
            label = str(int(cls)) if class_names is None else class_names[int(cls)]
            text = f"{label} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #convert back to BGR for saving

        cv2.imwrite("output_image.jpg", image)
        print("Output image saved")
    else:
        print("No detections to draw.")

# Example usage
class_names = model.names
draw_bounding_boxes(image_path, filtered_outputs, class_names)
```

This function opens the original image using OpenCV, which I've found to be more convenient for drawing. The filtered detections from the previous step are processed one by one. I move the tensors to cpu if necessary, and convert them to numpy for convenient access. The bounding box coordinates are unpacked from the `xyxy` format, scaled to integers, and used to draw rectangles. I include a confidence score and class label on each box. `model.names` provides an accessible list of class names, which are highly recommended for use rather than class id numbers. Finally, the modified image is saved to disk. Using a dedicated drawing function, versus trying to do it all within the core loop, aids greatly in code readability and troubleshooting.

In summary, obtaining predicted images involves more than a simple model invocation. Preprocessing, filtering, NMS, and visualization are critical parts of a functional pipeline. The example code illustrates how these pieces are connected, and these strategies have proven consistent across numerous projects where I've used YOLOv5 models. For further study, exploring the official YOLOv5 documentation, PyTorch tutorials, and OpenCV resources regarding image processing are highly recommended. It is also useful to review the work of other practitioners through code repositories on platforms like GitHub to observe different styles and optimization techniques.
