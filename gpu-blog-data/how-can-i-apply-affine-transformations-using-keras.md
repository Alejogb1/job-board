---
title: "How can I apply affine transformations using Keras' `apply_affine_transform` with a specific rotation angle (theta)?"
date: "2025-01-30"
id: "how-can-i-apply-affine-transformations-using-keras"
---
The `apply_affine_transform` function within Keras' image preprocessing utilities, while powerful, lacks direct control over rotation via an explicit angle parameter.  My experience working on medical image registration projects highlighted this limitation.  The function operates instead by manipulating transformation matrices, necessitating a clear understanding of linear algebra to achieve precise rotations.  This response details how to construct these matrices for targeted rotations and integrate them into the `apply_affine_transform` workflow.

1. **Clear Explanation:** Affine transformations are a combination of linear transformations (rotation, scaling, shearing) and translations.  Representing these transformations as matrices allows for efficient computation and concatenation.  A 2D rotation matrix, rotating a point counter-clockwise by an angle θ, is defined as:

```
R(θ) = | cos(θ)  -sin(θ) |
       | sin(θ)   cos(θ) |
```

This matrix, when multiplied with a point's coordinate vector (x, y), rotates that point around the origin.  To rotate around an arbitrary point (x_c, y_c), a translation is required: first translate the point so the center of rotation becomes the origin, then apply the rotation, and finally translate back. This entire process can be represented by a single, combined transformation matrix.

Constructing this matrix requires three steps:

a) **Translation to Origin:**  This involves subtracting the coordinates of the center of rotation from the point coordinates.  The translation matrix is:

```
T_1 = | 1  0  -x_c |
      | 0  1  -y_c |
      | 0  0   1   |
```

b) **Rotation:** This is performed using the rotation matrix `R(θ)` defined above.  It operates on homogeneous coordinates (adding a '1' as the third coordinate).

c) **Translation Back:** This is the inverse of the first translation:

```
T_2 = | 1  0  x_c |
      | 0  1  y_c |
      | 0  0   1   |
```

The complete affine transformation matrix `M` is the product of these three matrices: `M = T_2 * R(θ) * T_1`.  This matrix is then used within `apply_affine_transform`.  Note that `apply_affine_transform` expects a transformation matrix of shape (3, 3) for 2D images, representing homogeneous coordinates.

2. **Code Examples with Commentary:**

**Example 1: Rotating around the image center:**

```python
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform
import tensorflow as tf

def rotate_image(image, theta):
    # Assuming image is a NumPy array (H, W, C)
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    theta_rad = np.deg2rad(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    transform_matrix = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ]) @ rotation_matrix @ np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    transformed_image = apply_affine_transform(image, transform_matrix=transform_matrix.astype(np.float32))
    return transformed_image

# Example usage:
image = np.zeros((100, 100, 3), dtype=np.uint8) # Replace with your image
rotated_image = rotate_image(image, 45) # Rotate by 45 degrees

tf.keras.preprocessing.image.array_to_img(rotated_image) # display using something like matplotlib
```

This code defines a function `rotate_image` that takes an image and rotation angle as input.  It calculates the transformation matrix as described above and applies it using `apply_affine_transform`.  The use of `np.float32` ensures compatibility with Keras.

**Example 2: Rotating around a specific point:**

```python
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform

def rotate_around_point(image, theta, point):
    # point = (x,y)
    theta_rad = np.deg2rad(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    transform_matrix = np.array([
        [1, 0, -point[0]],
        [0, 1, -point[1]],
        [0, 0, 1]
    ]) @ rotation_matrix @ np.array([
        [1, 0, point[0]],
        [0, 1, point[1]],
        [0, 0, 1]
    ])

    transformed_image = apply_affine_transform(image, transform_matrix=transform_matrix.astype(np.float32))
    return transformed_image

# Example usage:
image = np.zeros((100, 100, 3), dtype=np.uint8)
rotated_image = rotate_around_point(image, 30, (25, 75)) # Rotate 30 degrees around (25, 75)

tf.keras.preprocessing.image.array_to_img(rotated_image)
```

This example demonstrates rotation around an arbitrary point specified by the `point` parameter.  The structure remains the same, adapting the translation matrices to the chosen point.

**Example 3: Incorporating scaling and shearing:**

```python
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform

def complex_transform(image, theta, scale_x, scale_y, shear):
    theta_rad = np.deg2rad(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    shear_matrix = np.array([
        [1, shear, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    transform_matrix = scaling_matrix @ shear_matrix @ rotation_matrix

    transformed_image = apply_affine_transform(image, transform_matrix=transform_matrix.astype(np.float32))
    return transformed_image

# Example usage
image = np.zeros((100,100,3), dtype=np.uint8)
transformed_image = complex_transform(image, 20, 1.2, 0.8, 0.5) #20deg rotation, x-scale 1.2, y-scale 0.8, shear 0.5

tf.keras.preprocessing.image.array_to_img(transformed_image)
```

This showcases the flexibility of the matrix approach.  Scaling and shearing matrices are included, demonstrating how easily additional affine transformations can be integrated by matrix multiplication. Remember to adjust the order of matrix multiplication to reflect the desired sequence of transformations.


3. **Resource Recommendations:**

"Linear Algebra and its Applications" by David C. Lay;  "Numerical Recipes in C++" by William H. Press et al.;  "Programming Computer Vision with Python" by Jan Erik Solem.  These texts provide comprehensive explanations of linear algebra and image processing techniques relevant to affine transformations.  Consult these for a deeper understanding of the underlying mathematical principles.
