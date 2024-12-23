---
title: "How can ImageAI be used for predicting multiple images?"
date: "2024-12-23"
id: "how-can-imageai-be-used-for-predicting-multiple-images"
---

Alright,  It’s not uncommon to need to process a batch of images rather than just one, and ImageAI provides the necessary mechanisms to do just that—though perhaps not always in the most obvious way. Over the years, I've encountered situations where single-image processing was a bottleneck. Switching to batch processing with ImageAI significantly improved throughput in a couple of projects, most notably when I was working on an automated quality control system for a manufacturing line. The system had to classify hundreds of images per minute, and the naive loop-based processing was, frankly, pathetic.

The key concept to understand is that while ImageAI’s high-level API is designed for ease of use, that simplicity doesn't hinder its batch processing capabilities. We often see newcomers attempt to run the same prediction loop multiple times, feeding a single image at a time. That method works, but it’s far from efficient. The real power comes from leveraging ImageAI's underlying frameworks like TensorFlow or PyTorch efficiently. Instead of individual predictions, we can prepare lists of images and let the backend handle the prediction in a batch.

Let’s dive into how you’d actually achieve this with some code snippets. The core idea revolves around preparing your data appropriately and then using the object detection, classification, or other prediction functions with the prepared data. I’ll focus on object detection here, but similar principles apply to other tasks.

**Example 1: Basic Batch Object Detection**

This example demonstrates how to perform object detection on multiple image files stored in a folder. We load all images, place their file paths into a list, and then use ImageAI's detector function with this list.

```python
from imageai.Detection import ObjectDetection
import os

# Location of your pre-trained model (update as needed)
MODEL_PATH = "path/to/your/pretrained-yolov3.h5"
IMAGE_DIR = "path/to/your/images/folder"
OUTPUT_DIR = "path/to/your/output/folder"

# Initialize the detector
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()

# Get a list of all image files in the directory
image_files = [os.path.join(IMAGE_DIR, f)
               for f in os.listdir(IMAGE_DIR)
               if os.path.isfile(os.path.join(IMAGE_DIR, f))
               and f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Detect objects in all images
detections = detector.detectObjectsFromImage(input_image=image_files,
                                            output_image_path=OUTPUT_DIR,
                                            minimum_percentage_probability=30)


# Process the returned results (a list of dictionaries)
for i, image_detection in enumerate(detections):
    print(f"Detections for Image {i+1}:")
    for detection in image_detection:
       print(f"  {detection['name']} : {detection['percentage_probability']:.2f}%  , at position {detection['box_points']}")
```

In this example, `input_image` is given a list of image file paths, allowing the underlying model to process these images in batches. `output_image_path` then determines where the processed images with bounding boxes are saved, again, in a batch. Note that if you do not need to save the output images, simply do not provide a path and only the detections will be returned. This will significantly speed up your process.

**Example 2: Processing Images from Memory (Bytes)**

Sometimes, your images aren't necessarily on disk but might be in memory, perhaps fetched from a database, or retrieved through an API call. In such instances, the file path approach wouldn't work. Here, we will demonstrate how to process images directly from their byte representations. This is particularly useful in scenarios where disk I/O becomes a bottleneck.

```python
from imageai.Detection import ObjectDetection
import io
from PIL import Image
import requests

# Model path and other configuration stays the same
MODEL_PATH = "path/to/your/pretrained-yolov3.h5"

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()


# Fetch a list of image URLs, assume you already have this
image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.png", "https://example.com/image3.jpeg"]
image_bytes_list = []

#Fetch bytes for each URL
for url in image_urls:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_bytes = response.content
        image_bytes_list.append(image_bytes)


# Process all images in bytes using the detector.detectObjectsFromImage function.
# we will set the "input_type" to "array". By default this parameter is "file"
detections = detector.detectObjectsFromImage(input_image=image_bytes_list,
                                        output_image_path="output_images",
                                        minimum_percentage_probability=30,
                                        input_type="array",
                                        output_type="array")
# Process detections, similar to example 1
for i, image_detection in enumerate(detections):
    print(f"Detections for Image {i+1}:")
    for detection in image_detection:
       print(f"  {detection['name']} : {detection['percentage_probability']:.2f}%  , at position {detection['box_points']}")
```

Here, we retrieve image data as bytes using `requests`, convert them to the format expected by `ImageAI` (bytes) and then we are able to use the `detector.detectObjectsFromImage` function with the parameter input_type set to `array`. This time, we also set `output_type` to array because we are not using image file paths. The output is then a list of the processed images in bytes together with detections. Again, note that output images are not necessary, you can avoid producing the images.

**Example 3: Batch prediction on preprocessed data**

Sometimes you have the images already loaded, processed, and maybe you are performing some additional steps. It can be the case that you cannot pass filepaths or bytes. In this case, you can pass a list of numpy arrays.

```python
from imageai.Detection import ObjectDetection
import numpy as np
from PIL import Image
import requests
import cv2

# Model path and other configuration stays the same
MODEL_PATH = "path/to/your/pretrained-yolov3.h5"

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()


# Fetch a list of image URLs, assume you already have this
image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.png", "https://example.com/image3.jpeg"]
image_array_list = []

#Fetch bytes for each URL, then convert to numpy array
for url in image_urls:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        image_array_list.append(image)


# Process all images as numpy arrays using the detector.detectObjectsFromImage function.
# we will set the "input_type" to "array"
detections = detector.detectObjectsFromImage(input_image=image_array_list,
                                        output_image_path="output_images",
                                        minimum_percentage_probability=30,
                                        input_type="array",
                                        output_type="array")

# Process detections, similar to example 1
for i, image_detection in enumerate(detections):
    print(f"Detections for Image {i+1}:")
    for detection in image_detection:
       print(f"  {detection['name']} : {detection['percentage_probability']:.2f}%  , at position {detection['box_points']}")
```

Here, we perform a similar process to Example 2, but we go one step further and convert the PIL image to a numpy array using `np.array(image)`. This allows us to pass our data to the `detector.detectObjectsFromImage` with `input_type="array"`. This gives you full flexibility to use ImageAI with any existing image processing pipeline.

**Technical Notes and Further Reading**

Batch processing in deep learning often relies on techniques like vectorization and parallel processing offered by underlying frameworks such as TensorFlow or PyTorch. ImageAI leverages these under the hood when you pass lists of images or byte representations. Batch sizes are determined by how much your GPU memory can handle. Start with smaller batch sizes to avoid memory errors, and increase them gradually until you observe a decrease in processing speed due to memory constraints. Experimentation is key.

For a deeper dive, I strongly recommend the following resources:

1.  **"Deep Learning with Python" by François Chollet:** This book offers a fantastic introduction to deep learning concepts using Keras, which forms the backend for some ImageAI models. Understanding Keras and how it manages batching will help you get more insight into how ImageAI works under the hood.
2.  **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This resource is invaluable for exploring the practical aspects of deep learning. It provides detailed explanations on how TensorFlow and Keras handle batch processing, along with optimizations you can leverage.
3.  **TensorFlow and PyTorch official documentation:** The official documentation for both libraries is crucial for anyone serious about deep learning. Pay close attention to sections on data loading and processing.
4.  **Research papers on object detection:** If you want to get in-depth about the model that powers ImageAI, I would recommend checking the original publications of models like YOLO, SSD, and RetinaNet.

In summary, ImageAI provides a robust way to conduct batch predictions. By properly structuring your input data and using the correct parameters, you can efficiently process multiple images, leading to significant performance improvements in your applications. As with all things in engineering, start simple and refine. Good luck.
