---
title: "How can I increase the number of bounding boxes per class?"
date: "2025-01-30"
id: "how-can-i-increase-the-number-of-bounding"
---
Increasing the number of bounding boxes per class, particularly when dealing with object detection datasets, frequently involves a balance between data augmentation, synthetic data generation, and strategies to address class imbalance. My experience developing models for automated defect detection in manufacturing has provided insights into various effective approaches, and the trade-offs involved.

The initial challenge often stems from insufficient real-world instances of certain object classes, leading to models that struggle to generalize effectively. This is especially true in scenarios where one class is prevalent, while another is rarer, creating an imbalance that biases model training. Increasing the bounding box count addresses this, but simply replicating existing boxes is ineffective; it does not introduce variability necessary for robust model learning. Instead, we must consider methods that create new, relevant bounding boxes while keeping the dataset consistent.

One fundamental technique to augment bounding box count is through *geometric transformations*. These transformations manipulate existing images and bounding boxes in predictable ways, creating new training samples. Simple examples include scaling, rotation, and translation. When applied carefully, these methods maintain the core attributes of an object, allowing the model to learn invariant features across slight variations. However, the extent to which these can be applied without rendering the bounding boxes invalid needs careful thought. Transformations exceeding what is realistically possible in the dataset should be avoided.

Another technique is *image manipulation*. These go beyond simple geometric changes. These include adding noise, adjusting brightness, changing contrast, and introducing blurs. These techniques introduce variability that can improve the model’s robustness to image distortions. When applying these, it is paramount to consider how the manipulations impact the visibility of the objects. Bounding boxes need to remain accurately placed over the relevant object even after these changes. Additionally, one should also be mindful of how these manipulations may interact with existing class characteristics in the dataset.

Beyond these augmentation techniques, there are several data synthesis strategies. Consider *copy-pasting objects*, which involves cutting an object from one image, using its associated bounding box, and pasting it onto another image. This is best applied when objects can plausibly be overlaid on an alternate background. However, it also introduces the challenge of correctly adjusting the bounding box. Further, the placement needs to avoid obvious visual artefacts. A more sophisticated technique is *generative adversarial networks (GANs)*. Properly trained, a GAN can produce entirely new images, including objects along with their bounding boxes. These provide a potentially endless source of augmented bounding boxes, but the process requires careful control to maintain fidelity to the original data.

Here are some coded examples demonstrating common methods for augmenting the number of bounding boxes per class:

**Example 1: Geometric Augmentation using OpenCV (Python)**

```python
import cv2
import numpy as np

def apply_geometric_augmentation(image, bbox, scale_factor=1.2, angle=10):
    """Applies scaling and rotation to an image and its bounding box.

    Args:
        image (numpy.ndarray): Input image.
        bbox (list): Bounding box coordinates in [x_min, y_min, x_max, y_max] format.
        scale_factor (float): Scaling factor.
        angle (int): Rotation angle in degrees.

    Returns:
        tuple: Augmented image and transformed bounding box.
    """

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    scale_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    augmented_image = cv2.warpAffine(image, scale_matrix, (width, height))

    # Convert bounding box coordinates to homogenous coordinates
    x_min, y_min, x_max, y_max = bbox
    bbox_points = np.array([[x_min, y_min, 1], [x_max, y_min, 1], [x_max, y_max, 1], [x_min, y_max, 1]])

    # Transform bounding box points using the rotation matrix
    transformed_points = (scale_matrix @ bbox_points.T).T
    transformed_x_min = min(transformed_points[:, 0])
    transformed_x_max = max(transformed_points[:, 0])
    transformed_y_min = min(transformed_points[:, 1])
    transformed_y_max = max(transformed_points[:, 1])

    transformed_bbox = [int(transformed_x_min), int(transformed_y_min), int(transformed_x_max), int(transformed_y_max)]
    return augmented_image, transformed_bbox

# Example usage
image = cv2.imread("sample_image.jpg")  # Load your sample image here
bbox = [50, 50, 200, 200] # Define a bounding box on sample image
augmented_image, transformed_bbox = apply_geometric_augmentation(image, bbox)

cv2.imshow("Augmented Image", augmented_image)
cv2.rectangle(augmented_image, (transformed_bbox[0], transformed_bbox[1]), (transformed_bbox[2], transformed_bbox[3]), (0, 255, 0), 2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
This Python example, using OpenCV, demonstrates how to scale and rotate an image along with its bounding box. The `cv2.warpAffine` function applies the transformation. The associated bounding box points are converted into homogenous coordinates, transformed by the same matrix, and then projected back. This is critical for maintaining bounding box accuracy.

**Example 2: Image Manipulation using Pillow (Python)**

```python
from PIL import Image, ImageEnhance
import random

def apply_image_manipulation(image, brightness_factor=0.8, contrast_factor=1.2):
  """Applies random brightness and contrast adjustments to an image.

  Args:
      image (PIL.Image): Input image.
      brightness_factor (float): Brightness adjustment factor.
      contrast_factor (float): Contrast adjustment factor.

  Returns:
    PIL.Image: Augmented image.
  """

  enhancer_brightness = ImageEnhance.Brightness(image)
  augmented_image = enhancer_brightness.enhance(brightness_factor)

  enhancer_contrast = ImageEnhance.Contrast(augmented_image)
  augmented_image = enhancer_contrast.enhance(contrast_factor)

  return augmented_image

# Example usage
image = Image.open("sample_image.jpg") # Load your sample image here
brightness = random.uniform(0.5, 1.5)
contrast = random.uniform(0.7, 1.3)

augmented_image = apply_image_manipulation(image, brightness_factor = brightness, contrast_factor = contrast)
augmented_image.show()
```
This example utilizes the Pillow library for image manipulations. The code adjusts the brightness and contrast of an image. Random variation in parameters improves the overall diversity. The bounding boxes in this example do not require adjustment as the base image itself is being manipulated.

**Example 3: Basic Copy-Paste using OpenCV (Python)**

```python
import cv2
import numpy as np

def copy_paste_object(background_image, object_image, object_bbox, target_coords):
    """Pastes an object onto a background image.

    Args:
        background_image (numpy.ndarray): Image to paste onto.
        object_image (numpy.ndarray): Image containing the object.
        object_bbox (list): Bounding box coordinates of the object [x_min, y_min, x_max, y_max].
        target_coords (tuple): x and y coordinates of the top-left corner where to paste the object on the background.

    Returns:
        tuple: Image with pasted object, bounding box in new coordinates.
    """

    x_min, y_min, x_max, y_max = object_bbox
    object_roi = object_image[y_min:y_max, x_min:x_max]

    # Extract dimensions for proper placement
    height, width = object_roi.shape[:2]
    target_x, target_y = target_coords

    # Ensure the placement is valid given background dimensions
    background_height, background_width = background_image.shape[:2]

    # Check if the object exceeds boundaries of the background image.
    if target_x + width > background_width or target_y + height > background_height:
       print ("Placement exceeds background boundaries.")
       return background_image, None # Return the background as is.

    background_image[target_y:target_y + height, target_x:target_x + width] = object_roi

    # Adjust the bounding box for the pasted region.
    pasted_bbox = [target_x, target_y, target_x + width, target_y + height]
    return background_image, pasted_bbox

# Example usage
background_image = cv2.imread("background_image.jpg")  # Load background image.
object_image = cv2.imread("object_image.jpg")  # Load object image.
object_bbox = [20, 30, 100, 110]  # Bounding box of object in object_image.
target_coords = (150, 180)  # Coordinates to paste on the background image.

pasted_image, new_bbox = copy_paste_object(background_image, object_image, object_bbox, target_coords)

if new_bbox:
    cv2.rectangle(pasted_image, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (0, 255, 0), 2)
cv2.imshow("Pasted Image", pasted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
This example illustrates a very basic method to copy and paste an object from one image to another. The code extracts a region of interest (ROI) and places it within a different image, recalculating and producing a new bounding box. Error handling is added to prevent placement exceeding image boundaries. Note that no consideration is made for perspective changes, blending, or other advanced blending techniques that could make the pasted object more realistic.

For further exploration and deeper understanding, resources on advanced data augmentation techniques using libraries such as Albumentations, as well as documentation of relevant libraries like OpenCV, Pillow and TensorFlow's image preprocessing, are good starting points. Furthermore, materials related to GANs and their application in image synthesis can provide additional insight into generating entirely synthetic data, which often involves a more advanced approach. Understanding strategies for dealing with imbalanced data, as covered in many machine learning textbooks and online course materials, will contribute to more effective results when using these techniques. Finally, investigating the specific methods employed within popular object detection frameworks, such as TensorFlow Object Detection API or Detectron2, can provide additional context and coding inspiration.
