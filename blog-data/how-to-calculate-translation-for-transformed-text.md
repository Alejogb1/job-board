---
title: "How to Calculate translation for transformed text?"
date: "2024-12-14"
id: "how-to-calculate-translation-for-transformed-text"
---

alright, so you're tackling the classic transformed text translation problem. i've been there, spent more than a few late nights staring at screens trying to get this to work. it's not exactly a walk in the park but it's very doable if you understand the fundamentals. let's break it down, i'll give you my take based on some painful experiences i've had over the years, specifically focusing on a few different scenarios.

first off, what do we mean by "transformed text"? well it could be anything from simple scaling and rotation to more complex perspective transformations or even non-linear warps. at its core, we need to figure out how to map points from the original text's coordinate space into the transformed text's coordinate space. and then, inversely, go back the other way. this mapping, for many transformations can be done via matrix algebra. i once thought it would be easier using polar coordinate system with radial distortion, after spending a weekend i realized that a normal matrix operation would have done the trick faster and better.

i remember back in my early days working on a project that involved image manipulation for a scanning device. the scanned text came in at an angle, slightly distorted, a common occurrence. i thought that i could avoid matrix math, which i had a very weak understanding of at the time, using a simple iterative solution. oh boy i was wrong, it took me a week to get something working that would just break on edge cases, that could be easily handled by a matrix transformation. let me tell you, that was a painful lesson in understanding the beauty of linear algebra.

let's start with the simplest case, an affine transformation. this includes translation, rotation, scaling, and shearing. these can all be combined into a single 3x3 matrix (assuming 2d space, we use a homogeneous coordinate representation). the first 2x2 part is for scale, rotation, shear. The last row is always (0, 0, 1) and the last column is a translation vector. the top two rows of the last column hold the tx and ty of the translation.

```python
import numpy as np

def affine_transform_matrix(tx=0, ty=0, scale_x=1, scale_y=1, rotation_degrees=0, shear_x=0, shear_y=0):
    rotation_rad = np.radians(rotation_degrees)
    cos_val = np.cos(rotation_rad)
    sin_val = np.sin(rotation_rad)

    transform_matrix = np.array([
        [scale_x * cos_val - shear_y * sin_val, scale_x * -sin_val - shear_y * cos_val, tx],
        [scale_y * shear_x * cos_val + scale_y * sin_val, scale_y * -shear_x * sin_val + scale_y * cos_val, ty],
        [0, 0, 1]
    ])
    return transform_matrix

def apply_transform(points, matrix):
    transformed_points = []
    for x, y in points:
        point_vector = np.array([x, y, 1])
        transformed_vector = np.dot(matrix, point_vector)
        transformed_points.append((transformed_vector[0], transformed_vector[1]))
    return transformed_points

#example usage
translation_matrix = affine_transform_matrix(tx=10, ty=20)
points = [(10,10),(20,20)]
transformed_points = apply_transform(points, translation_matrix)
print(transformed_points)

scale_matrix = affine_transform_matrix(scale_x=2, scale_y=2)
transformed_points = apply_transform(points,scale_matrix)
print(transformed_points)

rotation_matrix = affine_transform_matrix(rotation_degrees=45)
transformed_points = apply_transform(points,rotation_matrix)
print(transformed_points)

combined_matrix = affine_transform_matrix(tx=10, ty=20, scale_x=2, scale_y=2, rotation_degrees=45, shear_x=0.5, shear_y=0.2)
transformed_points = apply_transform(points,combined_matrix)
print(transformed_points)
```

in the code above i've shown you a basic python example that computes the affine transform matrix and applies it to a set of coordinates. keep in mind that to revert this transformation, you'll need to compute the inverse of the transform matrix. a simple `np.linalg.inv(matrix)` will do the trick. the main idea here is that the transform can be represented as a single matrix, and you can compose them by matrix multiplication, so first transform multiplied by second transformation and so on.

but what if the transformation is non-linear? well, then things get a little more hairy. a common example would be a perspective transform, often used to correct images of text taken at an angle. perspective transforms, while still linear in homogeneous space, require a 4x4 matrix for 3d and a 3x3 for 2d. it is no longer affine. for this type of transform, you'll need to know at least four pairs of matching points: one in the original image and another in the transformed one. if you are working with images, you would use a computer vision library for this.

i had an interesting experience with a project involving analyzing old scanned documents. we had a ton of historical records that were warped and had perspective distortion, and our task was to extract data from the text. i initially tried to do that by approximating the perspective transform with multiple affine transforms but the results were just not accurate enough. eventually i switched to a proper perspective matrix and used opencv for calculating it, it was the only way.

```python
import cv2
import numpy as np

def perspective_transform_matrix(src_points, dst_points):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def apply_perspective_transform(points, matrix):
    points = np.float32(points)
    transformed_points = cv2.perspectiveTransform(np.array([points]), matrix)
    return [(p[0][0], p[0][1]) for p in transformed_points]

#example
src_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
dst_points = [(10, 10), (120, 20), (110, 110), (5, 95)]
matrix = perspective_transform_matrix(src_points, dst_points)
points = [(20,20),(50,50),(75,25)]
transformed_points = apply_perspective_transform(points, matrix)
print(transformed_points)

```
the code above shows a simple usage of perspective transforms. you should have the four points for the original image and four matching points for the transformed image, after that you can call `cv2.getPerspectiveTransform` which would give you the transformation matrix. using `cv2.perspectiveTransform` you can transform a set of points.

now, let's dive into a case where the transformation is completely arbitrary and not easily represented by a matrix. this is where things can get really complex and you have to use interpolation. i once had to work with a text display that had this crazy non-linear distortion that looked like some sort of fisheye effect, but more complex. it wasn't a simple mathematical formula and i had to go through trial and error on getting a grid deformation map and it was quite a task, but i think it gave me a solid understanding of this process.

for this scenario you will need to generate a grid of points in the original space and calculate the corresponding points in the transformed space. after this you can use this information to find the position of other points in the transformation. you can do that by interpolation. there are different methods, linear interpolation being the simplest and most commonly used.

```python
import numpy as np
from scipy.interpolate import interp2d

def calculate_grid(original_grid_size, transform_function):
    x_orig = np.linspace(0, 1, original_grid_size[0])
    y_orig = np.linspace(0, 1, original_grid_size[1])
    grid_x, grid_y = np.meshgrid(x_orig, y_orig)
    transformed_grid_x, transformed_grid_y = [], []

    for y_idx, y in enumerate(y_orig):
      transformed_grid_x_row, transformed_grid_y_row = [], []
      for x_idx, x in enumerate(x_orig):
        transformed_x, transformed_y = transform_function(x, y)
        transformed_grid_x_row.append(transformed_x)
        transformed_grid_y_row.append(transformed_y)
      transformed_grid_x.append(transformed_grid_x_row)
      transformed_grid_y.append(transformed_grid_y_row)

    return grid_x, grid_y, np.array(transformed_grid_x), np.array(transformed_grid_y)


def apply_non_linear_transform(points, grid_x, grid_y, transformed_grid_x, transformed_grid_y):
    interpolated_x_func = interp2d(grid_x, grid_y, transformed_grid_x, kind='linear')
    interpolated_y_func = interp2d(grid_x, grid_y, transformed_grid_y, kind='linear')
    transformed_points = []

    for x, y in points:
      x_out = interpolated_x_func(x, y)[0]
      y_out = interpolated_y_func(x, y)[0]
      transformed_points.append((x_out, y_out))
    return transformed_points

# example
def example_transform(x, y):
    # this is just an example, it can be any random transformation
    x_out = x + 0.1 * np.sin(y * 10)
    y_out = y + 0.1 * np.cos(x * 10)
    return x_out, y_out

original_grid_size = (10, 10) # larger grid means more accuracy and higher cost
grid_x, grid_y, transformed_grid_x, transformed_grid_y = calculate_grid(original_grid_size, example_transform)
points = [(0.2,0.2), (0.5,0.5), (0.8,0.8)]
transformed_points = apply_non_linear_transform(points, grid_x, grid_y, transformed_grid_x, transformed_grid_y)
print(transformed_points)

```

this snippet above provides an implementation for a general transformation using the scipy library. the grid represents a transformation map between original space to distorted space and it uses a linear interpolation for points outside the grid. this is a powerful technique for complex transformations. you need to have a way of calculating the new positions of your original grid, after that the interpolation takes care of the rest. there are more precise interpolation methods, like cubic interpolation, that can be used if a linear interpolation is not giving you a good enough result.

the key takeaway here is that, for transformed text, you always need to determine the mapping between your original and transformed space. whether it is through matrix math, computer vision functions or more complex interpolations, it's all about defining that relationship.

as for resources, i can suggest, for linear algebra i highly recommend "linear algebra and its applications" by gilbert strang. it covers all the fundamentals really well and is a classic in the field. for computer vision concepts, “computer vision: algorithms and applications” by richard szeliski is an amazing text that goes into deep details about most algorithms in the computer vision field. if you need more details about numerical analysis the book "numerical recipes" can give you an insight in most numerical methods and how to calculate them, its more of a reference than a theoretical book. these are not introductory books, so depending on your previous knowledge, you may need a lighter read before delving into these.

and lastly a piece of advice from my past, always start with the simplest solution and move up in complexity only when needed, and make sure you are using the tools that are available, no need to reinvent the wheel, unless its a square one, that would be funny, but very inefficient.
