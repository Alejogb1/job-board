---
title: "How can I dynamically adjust image annotations to match photo transformations (rotation/crop) in Python?"
date: "2024-12-23"
id: "how-can-i-dynamically-adjust-image-annotations-to-match-photo-transformations-rotationcrop-in-python"
---

, let's delve into this. Image annotation adjustment based on transformations—it’s a challenge I've definitely faced, particularly during a project involving automated document processing years ago. We had to rotate scanned documents frequently, and keeping the bounding boxes of extracted text accurately aligned was a crucial, and sometimes frustrating, aspect. So, how can we tackle this dynamically using Python? The key is understanding matrix transformations and applying them appropriately to annotation coordinates.

Fundamentally, when you rotate or crop an image, you're essentially applying geometric transformations. These transformations can be represented mathematically using matrices. For example, rotation around a center point and scaling operations can be combined into a single affine transformation matrix. We need to compute this matrix for each transformation applied to the image and then apply it inversely to the annotation coordinates. Let's break it down into steps.

First, let's consider rotation. When an image rotates, any annotated point (x, y) moves according to the angle and center of rotation. The rotation matrix around the center point (cx, cy) is:

`[ cos(θ)   -sin(θ)   cx*(1 - cos(θ)) + cy*sin(θ) ]`
`[ sin(θ)    cos(θ)   cy*(1 - cos(θ)) - cx*sin(θ) ]`
`[    0        0                 1                ]`

Where θ is the angle of rotation in radians.

Next, for cropping, we're essentially translating and scaling a portion of the image. Cropping can be defined by a bounding box that outlines the new image area. To match the new coordinates, we’ll need to apply a similar transformation approach, but with a matrix incorporating the cropping offset and potentially a scaling factor if the crop region doesn’t exactly match the original image size.

Here's how I've handled this in a practical setting, and how we can codify it with Python, using libraries like `numpy` for calculations and `PIL` (Pillow) for image transformations.

**Example 1: Rotation Adjustment**

```python
import numpy as np
from PIL import Image

def rotate_annotations(annotations, angle, image_size, center=None):
    """
    Adjusts annotation coordinates based on image rotation.

    Args:
      annotations: List of annotations, where each is [x1, y1, x2, y2]
      angle: Angle of rotation in degrees.
      image_size: Tuple (width, height) of the original image.
      center: Optional tuple (cx, cy) for the center of rotation, defaults to image center.

    Returns:
        Adjusted list of annotations.
    """

    angle_rad = np.radians(angle)
    width, height = image_size
    if center is None:
       center_x, center_y = width / 2, height / 2
    else:
       center_x, center_y = center

    rotation_matrix = np.array([
      [np.cos(angle_rad), -np.sin(angle_rad), center_x * (1 - np.cos(angle_rad)) + center_y * np.sin(angle_rad)],
      [np.sin(angle_rad), np.cos(angle_rad),  center_y * (1 - np.cos(angle_rad)) - center_x * np.sin(angle_rad)],
      [0, 0, 1]
    ])

    adjusted_annotations = []
    for x1, y1, x2, y2 in annotations:
        points = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]).T
        transformed_points = np.dot(rotation_matrix, points)
        transformed_x = transformed_points[0, :].round().astype(int)
        transformed_y = transformed_points[1, :].round().astype(int)
        adjusted_x1, adjusted_x2 = min(transformed_x), max(transformed_x)
        adjusted_y1, adjusted_y2 = min(transformed_y), max(transformed_y)
        adjusted_annotations.append([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2])

    return adjusted_annotations


# Example Usage:
image = Image.new('RGB', (100, 100), color = 'red')
original_annotations = [[20, 20, 80, 80]] # bounding box
rotated_angle = 45
rotated_image = image.rotate(rotated_angle)
adjusted_boxes = rotate_annotations(original_annotations, rotated_angle, image.size)
print("Rotated boxes:", adjusted_boxes)

```

In this example, I define a function `rotate_annotations` which takes annotations, rotation angle, image size and optionally rotation center and it returns the transformed annotations. Notice how each corner of the annotation is rotated separately and bounding box is reconstructed from the min/max. The use of numpy allows for clear and concise matrix operations.

**Example 2: Cropping Adjustment**

```python
import numpy as np
from PIL import Image

def crop_annotations(annotations, crop_box, original_size):
    """
    Adjusts annotation coordinates based on image cropping.

    Args:
        annotations: List of annotations, where each is [x1, y1, x2, y2]
        crop_box: Tuple (left, top, right, bottom) of the crop region.
        original_size: Tuple (width, height) of the original image.

    Returns:
      Adjusted list of annotations.
    """
    left, top, right, bottom = crop_box
    adjusted_annotations = []
    for x1, y1, x2, y2 in annotations:
        adjusted_x1 = max(0, x1 - left)
        adjusted_y1 = max(0, y1 - top)
        adjusted_x2 = min(right-left, x2 - left)
        adjusted_y2 = min(bottom-top, y2 - top)
        adjusted_annotations.append([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2])
    return adjusted_annotations

# Example Usage:
image = Image.new('RGB', (100, 100), color = 'green')
original_annotations = [[20, 20, 80, 80]]
crop_region = (10, 10, 90, 90)
cropped_image = image.crop(crop_region)
adjusted_boxes = crop_annotations(original_annotations, crop_region, image.size)
print("Cropped boxes:", adjusted_boxes)
```
Here, `crop_annotations` is shown, which offsets and adjusts annotation coordinates based on the crop region's coordinates. It clamps coordinate values so they don't fall outside of the cropped boundaries. It avoids the complexities of matrices because the translation is straightforward.

**Example 3: Combining Rotation and Cropping**
In a real scenario, you’ll likely need a combination of both. Here's a short snippet combining the two:

```python
import numpy as np
from PIL import Image

def adjust_annotations(annotations, angle, crop_box, image_size, center=None):

   rotated_annotations = rotate_annotations(annotations, angle, image_size, center)
   adjusted_annotations = crop_annotations(rotated_annotations, crop_box, image_size)
   return adjusted_annotations


# Example Usage
image = Image.new('RGB', (100, 100), color = 'blue')
original_annotations = [[20, 20, 80, 80]]
rotated_angle = 30
crop_region = (10, 10, 90, 90)
rotated_image = image.rotate(rotated_angle)
cropped_image = rotated_image.crop(crop_region)
adjusted_boxes = adjust_annotations(original_annotations, rotated_angle, crop_region, image.size)
print("Final adjusted boxes:", adjusted_boxes)
```

In this last example, we have a combined function `adjust_annotations` which applies rotation first, followed by cropping, to illustrate the typical transformation pipeline.

For a deeper dive, I highly recommend looking into the following resources:

1.  **"Multiple View Geometry in Computer Vision" by Richard Hartley and Andrew Zisserman:** This book is foundational for understanding the mathematical underpinnings of geometric transformations, including how matrices are used in image processing. It covers affine transformations in significant detail, which are central to this problem.

2.  **"Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods:** This is another excellent resource that delves into image transformations from a more signal-processing perspective. It provides a very good mathematical background and explanations of image processing techniques.

3.  **OpenCV documentation:** While we didn't directly use OpenCV here, it’s crucial to be aware of the functionalities. The documentation on geometric transformations and matrix manipulation is comprehensive.

My experience has shown me that clarity in transformations is vital. Using the right matrices, understanding their sequence, and ensuring that annotations are properly adjusted is crucial for robust and reliable automated processes dealing with visual data. Remember to test thoroughly, especially edge cases with extreme rotations or crops. It is essential to consider the overall transformation pipeline and use matrix operations correctly for consistently accurate results. With a careful implementation, you can achieve reliable and dynamically adjusted annotations that stay accurately aligned with your image transformations.
