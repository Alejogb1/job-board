---
title: "How can YOLOv5 be used to determine if detected objects are identical?"
date: "2025-01-30"
id: "how-can-yolov5-be-used-to-determine-if"
---
YOLOv5, while excellent at object *detection*, fundamentally outputs bounding box coordinates and associated class probabilities, not object *identification*. Determining if detected objects are identical requires a separate process leveraging the feature embeddings that can be extracted from YOLOv5's intermediate layers. My experience building a vision-based inspection system for a manufacturing line required this very capability; simply detecting multiple instances of, say, a screw, was insufficient. I needed to ascertain if these screws were the same type, and ultimately, if they were identical in the image’s perspective.

The core principle involves extracting feature maps from a selected layer within the YOLOv5 architecture, then using these to generate a compact, numerical representation (an embedding) for each detected object. These embeddings, often obtained through techniques like average pooling, provide a high-dimensional feature space representation of the object. Objects visually similar will have embeddings that are closer in this space, according to a selected similarity metric, commonly cosine similarity or Euclidean distance. This approach doesn’t rely on image-level comparisons; it analyzes the learned feature patterns for the detected regions of interest.

Here’s a breakdown of how to achieve this, assuming a PyTorch-based YOLOv5 implementation:

1.  **Model Modification:** The first step involves modifying the YOLOv5 model to expose the feature maps of an intermediate layer (usually a convolutional layer). I typically insert a "hook" to retrieve these features during the forward pass. In practice, the `model.model[-1].m` layer has worked well because it’s late in the network, providing rich contextual information, though this can vary depending on the specific model size and configuration. We specifically target the features outputted after a particular convolution for better performance.

2.  **Bounding Box Alignment:** The extracted features are spatially aligned with each object bounding box. The YOLOv5 output provides the predicted box coordinates and these must be used to extract the relevant portions from the feature map. The box coordinates might need to be scaled up since feature maps are usually smaller than input images. We extract the region of interest (ROI) features of each box.

3.  **Embedding Generation:** Once ROIs are extracted, they can be processed to generate an embedding. This commonly involves spatial average pooling on the extracted ROIs followed by optional normalization. The embedding condenses the feature map into a fixed-length vector, suitable for comparison.

4.  **Similarity Calculation:** Finally, the similarity or distance between embeddings is calculated. Cosine similarity, being relatively robust to scaling differences in feature vectors, often performs better than Euclidean distance, especially with deep network features. Objects with embeddings displaying high similarity scores would be regarded as identical. It's important to choose an appropriate threshold for similarity when deciding if the objects are similar or not, and these must be empirically determined.

Here are three illustrative code examples:

**Example 1: Feature Extraction and ROI cropping:**

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Assuming yolov5 model is loaded in 'model'
# Example model loading (replace with actual loading procedure)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model.eval()

def get_roi_features(image_path, model, target_layer=-1):

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(model.device)
    
    feature_outputs = {}
    
    def hook(module, input, output):
        feature_outputs["output"] = output.detach() # Using dict to store the features
        
    handle = model.model[target_layer].register_forward_hook(hook)

    with torch.no_grad():
        detections = model(input_tensor)[0] # Model inference
        handle.remove()

    # Filter detections by confidence.
    detections = detections[detections[:, 4] > 0.5] # Filtering by confidence

    cropped_feature_maps = []
    original_image_width, original_image_height = image.size

    if detections.shape[0] == 0:
        return []

    for detection in detections:

        x_min, y_min, x_max, y_max, _ , _= detection.cpu().numpy()
        
        # Scaling box coordinates
        x_min = int(x_min * original_image_width / 640)
        y_min = int(y_min * original_image_height / 640)
        x_max = int(x_max * original_image_width / 640)
        y_max = int(y_max * original_image_height / 640)
        

        # Feature map from the hook.
        features = feature_outputs["output"][0]

        # Feature map resize.
        feature_width = features.shape[2]
        feature_height = features.shape[1]
        
        #Scaling box coordinates
        x_min_feature = int(x_min * feature_width / original_image_width)
        y_min_feature = int(y_min * feature_height / original_image_height)
        x_max_feature = int(x_max * feature_width / original_image_width)
        y_max_feature = int(y_max * feature_height / original_image_height)

        #Crop Features from the feature map
        cropped_features = features[:,y_min_feature:y_max_feature, x_min_feature:x_max_feature]
        cropped_feature_maps.append(cropped_features.cpu())

    return cropped_feature_maps

# Example Usage
image_path = "example.jpg" # Replace with your image
cropped_features_list = get_roi_features(image_path, model)
if cropped_features_list:
    print("Number of cropped feature maps:", len(cropped_features_list))
else:
    print("No objects detected")
```

*Commentary:* This snippet demonstrates how to modify the YOLOv5 model to extract feature maps from a particular layer using a forward hook, specifically for the `-1` index (last layer). It processes an input image, executes a forward pass, and then uses bounding boxes from the detection to crop the feature map according to original image coordinates. It also scales the output bounding boxes and feature maps appropriately before cropping to ensure correct spatial alignment.

**Example 2: Embedding generation using average pooling:**

```python
def generate_embeddings(cropped_features_list):

    embeddings = []

    for features in cropped_features_list:
      # Perform global average pooling.
      embedding = torch.mean(features, dim=[-2, -1])  
      # Normalize the embedding.
      embedding = torch.nn.functional.normalize(embedding, p=2, dim=1) # Added L2 norm
      embeddings.append(embedding.cpu())
    return embeddings
  
# Usage (assuming cropped_features_list from example 1)
embeddings_list = generate_embeddings(cropped_features_list)

if embeddings_list:
    print("Number of embeddings:", len(embeddings_list))
    print("Shape of the embedding:", embeddings_list[0].shape)
else:
    print("No embeddings generated")
```
*Commentary:* This function takes the list of cropped feature maps as an argument and performs average pooling to transform them into vector embeddings. It also includes an L2 normalization which is very important to get better results. Each vector corresponds to an individual detected object in the image.

**Example 3: Similarity comparison with Cosine Similarity:**

```python
import torch.nn.functional as F

def compare_embeddings(embeddings_list):
    num_embeddings = len(embeddings_list)

    if num_embeddings <= 1:
        print("Not enough objects to compare.")
        return

    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):

            similarity_score = F.cosine_similarity(embeddings_list[i], embeddings_list[j])
            print(f"Similarity between object {i+1} and object {j+1}: {similarity_score.item()}")


# Usage (assuming embeddings_list from example 2)
compare_embeddings(embeddings_list)
```

*Commentary:* This function demonstrates pairwise comparison of generated embeddings using cosine similarity. It iterates through all pairs of detected objects and prints their similarity scores, enabling identification of similar or identical objects in an image.

These code snippets provide a basic framework. In a production system, further considerations include batch processing for performance, fine-tuning the model using a specific dataset of relevant objects and more robust feature extraction methods. Experimenting with different network layers for feature extraction, various pooling strategies, and normalization techniques is necessary for optimal performance.

For further learning, resources dedicated to Deep Learning with PyTorch, particularly those focused on computer vision, are vital. Consult textbooks and research papers addressing metric learning and feature similarity. Specific knowledge on convolutional neural network architectures (CNNs) and techniques like feature visualization, are extremely useful. Additionally, review materials focusing on object detection with YOLO models in PyTorch for deeper understanding on implementation. Finally, understanding principles of distance metrics and their properties is crucial in evaluating the output similarity.
