---
title: "How can I export YOLOv5 inferred images as an annotated dataset?"
date: "2025-01-30"
id: "how-can-i-export-yolov5-inferred-images-as"
---
Exporting YOLOv5 inferences directly as an annotated dataset requires a multi-step process, as the model outputs detection data rather than a formatted annotation file. I've encountered this exact problem several times while automating visual inspection pipelines, and the solution involves processing the model's output to match an expected annotation format, typically a text-based format like YOLO's, or JSON for broader compatibility. This process involves iterating through each detected object, extracting its bounding box, and then relating that information back to the original image.

Fundamentally, YOLOv5 outputs a list of detections for each image, where each detection typically consists of the bounding box coordinates (x_center, y_center, width, height), the confidence score, and the class label. To transform these into usable annotations, we need to: 1) normalize the bounding box coordinates according to the image dimensions, if the model output is in pixel space; 2) transform those normalized coordinates back into pixel space for output if needed, while maintaining the correct ratio; and 3) write the processed data into a file for each image, in the desired format.

The choice of output format is crucial. For continued use within YOLOv5 training, the standard format is a .txt file per image where each line represents a single detection, formatted as: `<class_id> <x_center> <y_center> <width> <height>`, where the coordinates are normalized between 0 and 1 relative to the image's width and height. For use in other annotation tools or for more general purposes, JSON provides a structured approach. JSON allows for more fields such as image paths and metadata.

Here's a Python-based approach, demonstrating common data transformations and file writing operations.

**Example 1: Exporting as YOLOv5 .txt files**

This script takes a dictionary of YOLOv5 inference outputs, along with a dictionary mapping image file paths to their respective dimensions. For each image it generates a .txt file with annotations.

```python
import os

def export_yolov5_txt(detections, image_dimensions, output_dir):
    """
    Exports YOLOv5 detections to text files in the YOLOv5 format.

    Args:
        detections (dict): A dictionary where keys are image file paths, and values
                        are lists of detections. Each detection is assumed to be a list
                        or tuple: [class_id, x_center, y_center, width, height, confidence]
        image_dimensions (dict): A dictionary where keys are image file paths, and
                               values are tuples of (width, height).
        output_dir (str): The directory to save the text files to.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path, dets in detections.items():
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        output_file = os.path.join(output_dir, filename)

        with open(output_file, 'w') as f:
            image_w, image_h = image_dimensions[image_path]
            for det in dets:
                class_id, x_center, y_center, width, height, _ = det  #unpack the detection

                # Normalize coordinates if they are in pixels
                x_center_norm = x_center / image_w
                y_center_norm = y_center / image_h
                width_norm = width / image_w
                height_norm = height / image_h

                # Format and write to file
                f.write(f"{int(class_id)} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
```

In this example, the function `export_yolov5_txt` iterates through each image's detections. It assumes that the detection data includes a confidence score, which is not used for the annotation file but is typically included in inference output. The function normalizes the bounding box coordinates using the provided image dimensions and then writes each detection to a new .txt file. These normalized coordinates are a critical aspect for the YOLOv5 training procedure.

**Example 2: Exporting as JSON files**

This function produces a JSON file containing the annotation data, where each entry corresponds to an image and its detections. This is typically better for applications outside YOLOv5's native processing.

```python
import json
import os

def export_yolov5_json(detections, image_dimensions, output_dir):
    """
    Exports YOLOv5 detections to a JSON file.

    Args:
        detections (dict): A dictionary where keys are image file paths, and values
                        are lists of detections. Each detection is assumed to be a list
                        or tuple: [class_id, x_center, y_center, width, height, confidence]
        image_dimensions (dict): A dictionary where keys are image file paths, and
                               values are tuples of (width, height).
        output_dir (str): The directory to save the json file to.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_data = {"images": []}
    for image_path, dets in detections.items():
        image_w, image_h = image_dimensions[image_path]
        image_data = {
                "file_name": image_path,
                "annotations": []
            }

        for det in dets:
            class_id, x_center, y_center, width, height, _ = det

            # Normalize coordinates if they are in pixels
            x_center_norm = x_center / image_w
            y_center_norm = y_center / image_h
            width_norm = width / image_w
            height_norm = height / image_h

            annotation = {
                "class_id": int(class_id),
                "x_center": x_center_norm,
                "y_center": y_center_norm,
                "width": width_norm,
                "height": height_norm
                }
            image_data["annotations"].append(annotation)
        json_data["images"].append(image_data)

    output_file = os.path.join(output_dir, "annotations.json")

    with open(output_file, "w") as f:
       json.dump(json_data, f, indent=4)
```

The `export_yolov5_json` function structures the data into a format where each image is an entry containing its file path and a list of annotations. It similarly normalizes coordinates if they are in pixel units. The output is a single JSON file containing annotations for all images. The use of indent=4 in `json.dump` improves readability.

**Example 3: Pixel Bounding Box Export (Optional)**

This optional function shows the process of exporting bounding boxes in pixel coordinates for visualization. This demonstrates that while the model is trained on normalized values, we can re-calculate their pixel representation by reversing the normalization step in Example 1. This is useful if you want to create bounding boxes for displaying over images.

```python
import os
import cv2

def export_yolov5_pixel_bboxes(detections, image_dimensions, output_dir):
   """
    Exports YOLOv5 detections to image files with drawn bounding boxes.

    Args:
        detections (dict): A dictionary where keys are image file paths, and values
                        are lists of detections. Each detection is assumed to be a list
                        or tuple: [class_id, x_center, y_center, width, height, confidence]
        image_dimensions (dict): A dictionary where keys are image file paths, and
                               values are tuples of (width, height).
        output_dir (str): The directory to save the images to.
    """

   if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   for image_path, dets in detections.items():
       image = cv2.imread(image_path)
       image_w, image_h = image_dimensions[image_path]
       for det in dets:
           class_id, x_center_norm, y_center_norm, width_norm, height_norm, _ = det # normalized values
           # Convert from normalized to pixel coordinates
           x_center = int(x_center_norm * image_w)
           y_center = int(y_center_norm * image_h)
           width = int(width_norm * image_w)
           height = int(height_norm * image_h)

           # Calculate top-left corner of bounding box for drawing
           x1 = int(x_center - width / 2)
           y1 = int(y_center - height / 2)
           x2 = int(x_center + width / 2)
           y2 = int(y_center + height / 2)

           # Draw the bounding box on the image (example, color and line thickness are changeable)
           cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

       filename = os.path.basename(image_path)
       output_path = os.path.join(output_dir, filename)
       cv2.imwrite(output_path, image)

```
In the `export_yolov5_pixel_bboxes` function, bounding box coordinates are first normalized and then converted back to pixel space before being drawn onto the image using `cv2.rectangle`. The center and bounding box width and height are used to draw a rectangle on the original image. The altered images are written to the `output_dir`.

To summarize, exporting YOLOv5 inferences to a usable annotation format involves: 1) correctly extracting the bounding box, class, and confidence information from the model's output, 2) normalizing (or pixel-shifting) the bounding box coordinates based on the image size; and 3) writing the data to a structured file format, such as .txt files or a JSON file. The choice of method depends heavily on your downstream application.

For anyone exploring this further, I recommend researching object detection evaluation metrics like mean Average Precision (mAP) to measure the quality of your annotations. Further knowledge of COCO and Pascal VOC annotation formats can be beneficial for working with datasets from multiple sources. Additionally, examining libraries like OpenCV for image manipulation or the json library for structured file creation can deepen your understanding of working with annotation files. Consider consulting tutorials on data loading and augmentation practices within PyTorch as well. These resources will provide a comprehensive context for further development.
