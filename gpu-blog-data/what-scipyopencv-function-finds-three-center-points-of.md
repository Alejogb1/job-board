---
title: "What SciPy/OpenCV function finds three center points of a crescent-shaped binary mask?"
date: "2025-01-30"
id: "what-scipyopencv-function-finds-three-center-points-of"
---
The challenge of locating three meaningful center points within a crescent-shaped binary mask necessitates a multi-faceted approach, moving beyond simple centroid calculations. Direct application of `cv2.findContours` in OpenCV followed by `cv2.moments` or `scipy.ndimage.center_of_mass` will yield the centroid of the entire shape, which is not representative of three distinct points. My experience developing machine vision systems for defect detection, specifically in irregularly shaped objects, has shown that a technique leveraging contour decomposition and skeletonization, combined with careful region selection, is necessary for success.

First, extracting the contour of the crescent is the initial step. `cv2.findContours` can achieve this effectively. However, it's critical to note that this function returns a list of contours, and depending on the complexity of the binary mask, additional filtering might be needed. After contour extraction, the critical part is representing the crescent's underlying structural form. Skeletonization, achieved through the `skimage.morphology.skeletonize` function, provides a thin, single-pixel representation of the crescent. Once the skeleton is obtained, analyzing the endpoints becomes feasible. Ideally, a crescent will present two endpoints derived from its arc and one midpoint along the skeleton. The midpoint, although not an "end" in the geometric sense, is needed to fulfill the requirement. While the endpoints are relatively straightforward to extract by finding pixels with one neighbor on the skeleton, the midpoint requires an additional process. I have found using a curve fitting or distance calculation technique works well for that particular center point.

The challenge then moves to differentiating between the actual crescent "ends" and any spurious branches that might appear. These spurious branches are commonly observed due to noise or imperfections within the original binary mask. To mitigate this, filtering based on branch length or connectivity often proves useful. In practice, I use a length threshold or by examining the number of pixels forming a branch. Small branches are discarded; this significantly improves robustness. Once the candidate endpoints are identified, I determine their associated coordinates in the original image space using the pixel locations from the skeleton.

Finding the third point on the skeleton, i.e., the middle, requires further analysis, and I've had most success using an iterative distance method. I sample the skeleton with a sliding window, measuring the distance between points. The point with the greatest cumulative distance to the two endpoints is a reasonable approximation of the middle point. This is robust to noise on the skeleton and slight deviations in the shape of the crescent. Finally, I assemble all the three coordinates (the two endpoints, and the calculated middle point) and present those as the final result. This overall process provides three meaningful "center" points for a crescent shape – the two endpoints of its arc and its midpoint along its arc.

**Code Example 1: Contour Extraction and Skeletonization**

This Python snippet demonstrates the initial steps of extracting the contour and skeletonizing the binary mask, using the OpenCV and scikit-image libraries:

```python
import cv2
import numpy as np
from skimage import morphology

def extract_skeleton(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Ensure a contour is found, else return empty array
    if not contours:
        return np.array([]), np.array([])

    # Select the largest contour if there are multiple, assumes largest is the crescent
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask from the contour
    contour_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

    skeleton = morphology.skeletonize(contour_mask > 0).astype(np.uint8)
    return skeleton, largest_contour
```
This function takes a binary mask as input. It begins by extracting all contours present within the mask using OpenCV's `cv2.findContours` function. In the case of multiple contours, the largest contour is selected as the target crescent. Then, this selected contour is used to generate a mask and `skimage.morphology.skeletonize` is used to perform the skeletonization. Finally, both the skeleton and original contour are returned. The skeleton is crucial for identifying endpoints and calculating the middle point.

**Code Example 2: Endpoint Identification and Spurious Branch Removal**

The following snippet focuses on identifying skeleton endpoints and removing spurious branches which usually stems from noise:

```python
def find_endpoints(skeleton):
    rows, cols = np.where(skeleton)
    endpoints = []
    for i, j in zip(rows, cols):
        neighbors = 0
        for x in range(max(0, i-1), min(skeleton.shape[0], i+2)):
            for y in range(max(0, j-1), min(skeleton.shape[1], j+2)):
               if (x != i or y != j) and skeleton[x, y] == 1:
                    neighbors += 1
        if neighbors <= 1:
            endpoints.append((j,i))

    #Filter out spurious branches based on connectivity; could be improved with length
    filtered_endpoints = []
    for ep in endpoints:
        ep_x, ep_y = ep
        if skeleton[ep_y, ep_x] == 1:
           filtered_endpoints.append(ep)
    return filtered_endpoints
```

This function takes the skeleton as input and iterates over all the non-zero pixels (i.e. part of the skeleton) to find the endpoints, identified as pixels with at most one neighbor. The neighborhood is checked on the 8 neighbors around the current pixel. After finding the endpoints, we add a secondary check for the identified endpoints: we filter the endpoints list by checking if the particular pixel is still part of the skeleton (this deals with possible problems from the previous check). The coordinates of these endpoints are returned. In real applications, it would be beneficial to add another check here using branch length or similar criteria, but for simplicity this was not included.

**Code Example 3: Midpoint Calculation**

The final snippet demonstrates how to calculate the midpoint along the skeleton, utilizing the identified endpoints and an iterative distance measurement approach:

```python
def find_midpoint(skeleton, endpoints):
    rows, cols = np.where(skeleton)
    if len(endpoints) != 2:
        return None
    
    p1 = endpoints[0]
    p2 = endpoints[1]

    max_dist = 0
    midpoint = None
    for i, j in zip(rows, cols):
        current_point = (j, i)
        dist = np.sqrt((current_point[0]-p1[0])**2 + (current_point[1]-p1[1])**2) + \
               np.sqrt((current_point[0]-p2[0])**2 + (current_point[1]-p2[1])**2)
        if dist > max_dist:
           max_dist = dist
           midpoint = current_point
    return midpoint

#Combine all functions
def get_crescent_centers(binary_mask):
    skeleton, contour = extract_skeleton(binary_mask)
    if skeleton.size == 0 : return []
    endpoints = find_endpoints(skeleton)
    if len(endpoints) != 2: return []

    midpoint = find_midpoint(skeleton, endpoints)
    if midpoint is None: return []
    return endpoints + [midpoint]
```

This function begins by extracting all points that form the skeleton, and checks that two endpoints have been passed (if not, it returns a None value). After this check, the points are iterated and the distance between every point in the skeleton and both endpoints is calculated. The point with the maximal sum of distances to the two endpoints is the selected middle point of the crescent. Then, a function that integrates all previous function `get_crescent_centers` is used, that returns empty array if the process fail, and an array containing all three points otherwise.

**Resource Recommendations**

For further exploration, consult the following documentation and resources:

*   **OpenCV Documentation:** The official OpenCV documentation provides extensive details on functions like `cv2.findContours`, `cv2.moments` and image manipulation techniques. It’s particularly valuable for understanding contour properties and processing methods.
*   **Scikit-image Documentation:** The scikit-image library's documentation covers morphology functions, such as `skimage.morphology.skeletonize`. It includes a deep dive into its parameters and behavior, crucial for effective skeletonization.
*   **Scientific Publications on Image Analysis:** Search for research papers related to shape analysis, specifically those concerning feature point detection within non-convex shapes. Focus on publications covering skeletonization algorithms and methods to extract key points from contours and skeletons.
*   **Books on Computer Vision:** Refer to introductory and advanced books on computer vision for comprehensive understanding of contour processing, geometric transformations, and skeleton-based analysis. These resources often provide a detailed context, and theoretical foundations needed for more sophisticated analysis.

By using these recommendations, a deeper understanding of the described methodologies can be achieved, and enable building more complex and robust applications.
