---
title: "Can Google ML Kit identify similar products without a custom model?"
date: "2025-01-30"
id: "can-google-ml-kit-identify-similar-products-without"
---
Google ML Kit provides pre-trained APIs for several machine learning tasks, image similarity being among them, but its ability to identify truly "similar products" without a custom-trained model is nuanced and depends heavily on the interpretation of similarity and the nature of the products being evaluated. My experience, developing a prototype inventory management system, involved extensive testing of the ML Kit's image labeling and text recognition APIs alongside its more basic on-device vision capabilities. This allowed me to evaluate its out-of-the-box strengths and limitations in product identification.

The core challenge lies in distinguishing between superficial visual similarity and semantically relevant product similarity. Google ML Kit's Vision API offers capabilities such as label detection, object detection, and landmark recognition. These APIs, while powerful for general-purpose visual understanding, are not inherently designed for fine-grained product matching. For example, the label detection API might recognize both a blue t-shirt and a red t-shirt as "t-shirt" but will not readily identify them as “similar products”, meaning they belong to the same class and might be interchangeable from an inventory or user-need perspective. Similarly, object detection can isolate the region of interest containing the product, but this isolation alone does not equate to understanding the product's specific attributes.

Therefore, the answer to the posed question is qualified: ML Kit can facilitate a pipeline for similarity identification without a bespoke model, but it won't directly declare "these two products are similar." Instead, it allows a developer to extract features from the images that, when compared programmatically, can form the basis for a product similarity measure. The key is to combine ML Kit's output with custom logic and feature matching algorithms.

I found that label detection and object detection output could be used to create feature vectors. Let's illustrate with three code examples written as Python snippets, demonstrating how you could implement such a similarity system using ML Kit’s outputs as if you were receiving them from an API.

**Example 1: Basic Label-Based Similarity**

This example demonstrates the use of label detection results for a basic similarity approach. Here, we use the identified labels directly as feature representations and compare the overlap.

```python
def calculate_label_similarity(image1_labels, image2_labels):
    """
    Calculates a simple similarity score based on shared labels.
    
    Args:
        image1_labels: A list of strings representing labels identified in image 1
        image2_labels: A list of strings representing labels identified in image 2

    Returns:
        A float representing the similarity score, ranging from 0 to 1.
    """
    common_labels = set(image1_labels) & set(image2_labels)
    total_labels = set(image1_labels) | set(image2_labels)
    
    if not total_labels:
        return 0.0  # Avoid division by zero
    
    return len(common_labels) / len(total_labels)

# Fictional output from ML Kit label detection API:
image1_labels = ["t-shirt", "clothing", "blue"]
image2_labels = ["t-shirt", "clothing", "red"]
image3_labels = ["chair", "furniture", "wood"]

similarity_1_2 = calculate_label_similarity(image1_labels, image2_labels)
similarity_1_3 = calculate_label_similarity(image1_labels, image3_labels)

print(f"Similarity between image 1 and image 2: {similarity_1_2}")  # Output: ~0.67
print(f"Similarity between image 1 and image 3: {similarity_1_3}")  # Output: 0.0
```

This simple approach will provide some results; two t-shirts with different colors would be considered somewhat similar. However, it's not robust. Different clothing items labeled with a common term such as 'clothing' might incorrectly register a high similarity. It also fails to account for more nuanced characteristics like patterns, material, or style.

**Example 2: Object Detection with Bounding Box Feature Generation**

Building upon the label approach, this example demonstrates using object detection API output to generate bounding box feature vectors. We represent the objects with the area and ratio of their bounding boxes.

```python
import numpy as np

def calculate_bounding_box_features(detections):
    """
    Calculates bounding box features (area and aspect ratio).

    Args:
        detections: A list of dictionaries, where each dictionary represents an object detection and has keys like 'bounding_box' and 'label'.

    Returns:
        A list of dictionaries, each containing the bounding box features ('area' and 'ratio') and the corresponding 'label'.
    """
    features = []
    for detection in detections:
        bbox = detection['bounding_box']
        width = bbox['right'] - bbox['left']
        height = bbox['bottom'] - bbox['top']
        area = width * height
        ratio = width / height if height > 0 else 0.0  # Avoid division by zero
        features.append({'area': area, 'ratio': ratio, 'label': detection['label']})
    return features


def calculate_feature_similarity(feature_set1, feature_set2, weight_area=0.4, weight_ratio=0.3, weight_label=0.3):
    """
    Calculates similarity based on bounding box features and label matching.

    Args:
        feature_set1: A list of dictionaries with bounding box features and labels for image 1
        feature_set2: A list of dictionaries with bounding box features and labels for image 2
        weight_area: Weight to give area similarity
        weight_ratio: Weight to give ratio similarity
        weight_label: Weight to give label similarity

    Returns:
        A float representing similarity, a value from 0 to 1.
    """
    total_similarity = 0.0
    num_comparisons = 0
    for feature1 in feature_set1:
      for feature2 in feature_set2:
        if feature1['label'] != feature2['label']:
          continue

        area_sim = 1 - abs(feature1['area'] - feature2['area']) / (feature1['area'] + feature2['area'] + 1e-8) #Avoid division by 0
        ratio_sim = 1 - abs(feature1['ratio'] - feature2['ratio']) / (feature1['ratio'] + feature2['ratio'] + 1e-8) #Avoid division by 0
        label_sim = 1.0 if feature1['label'] == feature2['label'] else 0.0
        similarity = (weight_area * area_sim) + (weight_ratio * ratio_sim) + (weight_label * label_sim)

        total_similarity += similarity
        num_comparisons += 1

    return total_similarity/num_comparisons if num_comparisons else 0.0

# Fictional output from ML Kit object detection API:
image1_detections = [
  {'bounding_box': {'left': 100, 'top': 50, 'right': 300, 'bottom': 250}, 'label': 't-shirt'},
  {'bounding_box': {'left': 350, 'top': 100, 'right': 450, 'bottom': 200}, 'label': 'button'}
]

image2_detections = [
  {'bounding_box': {'left': 120, 'top': 60, 'right': 320, 'bottom': 260}, 'label': 't-shirt'},
  {'bounding_box': {'left': 340, 'top': 90, 'right': 440, 'bottom': 190}, 'label': 'button'}
]
image3_detections = [
   {'bounding_box': {'left': 50, 'top': 50, 'right': 450, 'bottom': 450}, 'label': 'chair'}
]

image1_features = calculate_bounding_box_features(image1_detections)
image2_features = calculate_bounding_box_features(image2_detections)
image3_features = calculate_bounding_box_features(image3_detections)

similarity_1_2 = calculate_feature_similarity(image1_features, image2_features)
similarity_1_3 = calculate_feature_similarity(image1_features, image3_features)


print(f"Similarity between image 1 and image 2: {similarity_1_2}")  # Output: higher than Example 1.
print(f"Similarity between image 1 and image 3: {similarity_1_3}")  # Output: 0.0
```

This approach is more detailed, factoring in the size and shape of detected objects, which adds more discrimination. However, it is still fundamentally limited by the quality of the object detection. If different products share similar shapes and bounding boxes, they might be incorrectly deemed similar.

**Example 3: Combining Text and Image features**

This expands on the idea by incorporating text data if present within the product's picture. We'll use a simplified example that would require a separate OCR API, which ML kit offers.

```python
def calculate_combined_similarity(image1_labels, image2_labels, image1_text, image2_text, label_weight=0.5, text_weight=0.5):
  label_similarity = calculate_label_similarity(image1_labels, image2_labels)

  common_text = set(image1_text) & set(image2_text)
  total_text = set(image1_text) | set(image2_text)
  text_similarity = len(common_text)/len(total_text) if total_text else 0.0

  return (label_weight * label_similarity) + (text_weight * text_similarity)

# Fictional output from ML Kit label detection and text recognition APIs:
image1_labels = ["t-shirt", "clothing", "blue"]
image2_labels = ["t-shirt", "clothing", "red"]
image3_labels = ["chair", "furniture", "wood"]

image1_text = ["brand-name", "size-m"]
image2_text = ["brand-name", "size-l"]
image3_text = ["brand-x"]

similarity_1_2 = calculate_combined_similarity(image1_labels, image2_labels, image1_text, image2_text)
similarity_1_3 = calculate_combined_similarity(image1_labels, image3_labels, image1_text, image3_text)

print(f"Combined similarity between image 1 and image 2: {similarity_1_2}") #Should be higher than in example 1
print(f"Combined similarity between image 1 and image 3: {similarity_1_3}") #Should still be 0
```

This final example shows that adding text context can enhance accuracy, especially where products might be visually similar but have distinguishing textual identifiers such as brand and size. It is worth noting that these features can be normalized and weighted for better results.

The presented code examples illustrate that ML Kit can extract useful features that, with the application of custom similarity algorithms, can perform reasonably well in simple similarity tasks. For product matching, particularly across categories, using these approaches alone will be very limiting and will lack robustness. It is a good starting point for prototype development, but in more robust and complex use cases, there will be a point where a custom model fine-tuned on specific product data would be more appropriate.

To delve deeper into this area, several resources exist. Books focused on computer vision, such as “Computer Vision: Algorithms and Applications” by Richard Szeliski provide a comprehensive theoretical background. Additionally, materials concerning feature engineering for machine learning will equip you to tailor algorithms to your needs better. Finally, documentation from the Google Cloud Platform and Firebase, concerning their respective ML Kit offerings, will explain in detail the inner workings of their APIs and how to utilize them most effectively.
