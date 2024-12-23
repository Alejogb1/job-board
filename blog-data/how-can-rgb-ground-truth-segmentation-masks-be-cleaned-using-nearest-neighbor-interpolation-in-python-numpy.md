---
title: "How can RGB ground truth segmentation masks be cleaned using nearest neighbor interpolation in Python NumPy?"
date: "2024-12-23"
id: "how-can-rgb-ground-truth-segmentation-masks-be-cleaned-using-nearest-neighbor-interpolation-in-python-numpy"
---

, let’s tackle this. I recall dealing with a particularly noisy dataset of satellite imagery a few years back. The initial RGB ground truth segmentation masks were, to put it mildly, riddled with artifacts. We needed a way to smooth those edges and fill in small gaps reliably. Nearest neighbor interpolation, while simple, proved to be a surprisingly effective tool when used thoughtfully within a NumPy environment. It’s definitely not the fanciest algorithm we have, but its efficiency and ease of implementation made it ideal for our processing pipeline.

The core idea with nearest neighbor interpolation for segmentation masks is straightforward: for any pixel that needs to be filled or smoothed, we assign it the value of the closest available, valid pixel. In the context of a segmentation mask, this means finding the nearest labeled region and extending its boundary a little bit. It's not going to reconstruct complex shapes, but it’s excellent for removing small isolated errors, filling little holes, or correcting rough edges.

Let's break down how I'd typically approach this using NumPy and a little bit of SciPy for good measure. Imagine we've loaded our mask into a NumPy array, and for clarity, let's assume it's a 3-channel array (RGB) representing different object classes. First, we have to identify the locations where we need to perform the interpolation. Often, this is where the mask isn't fully defined, perhaps due to imperfections in the labeling process. These locations will typically have a predefined 'invalid' value (often a zero or a specific placeholder).

Here's the Python code snippet to identify invalid pixels, and the valid locations using NumPy array slicing and boolean indexing, an area of NumPy manipulation that's a cornerstone to any data scientist's workflow.

```python
import numpy as np
from scipy.spatial import KDTree

def find_invalid_pixels(mask, invalid_value=0):
  """Identifies coordinates of invalid pixels in a mask.

    Args:
        mask (np.ndarray): The input mask, likely a 3D RGB array.
        invalid_value (int, optional): The value indicating invalid pixels. Defaults to 0.

    Returns:
        np.ndarray: A 2D array of [row, col] coordinates of invalid pixels.
    """
  invalid_indices = np.where(np.all(mask == invalid_value, axis=-1))
  return np.array(list(zip(invalid_indices[0], invalid_indices[1])))

def find_valid_pixels(mask, invalid_value=0):
  """Identifies coordinates of valid pixels in a mask.

    Args:
        mask (np.ndarray): The input mask, likely a 3D RGB array.
        invalid_value (int, optional): The value indicating invalid pixels. Defaults to 0.

    Returns:
        np.ndarray: A 2D array of [row, col] coordinates of valid pixels.
    """
  valid_mask = np.any(mask != invalid_value, axis=-1)
  valid_indices = np.where(valid_mask)
  return np.array(list(zip(valid_indices[0], valid_indices[1])))


#Example Usage
if __name__ == '__main__':
    mask_shape = (100,100,3)
    test_mask = np.zeros(mask_shape,dtype=np.uint8)
    #Create a small filled-in square for testing
    test_mask[20:80,20:80,:] = [255,0,0]
    invalid_pixels = find_invalid_pixels(test_mask)
    valid_pixels = find_valid_pixels(test_mask)
    print(f"Number of Invalid Pixels: {invalid_pixels.shape[0]}")
    print(f"Number of Valid Pixels: {valid_pixels.shape[0]}")

```

This first snippet highlights using `np.where` in combination with boolean masks, which is crucial. Once we've identified our 'invalid' regions, the next step is to find our 'valid' regions that can serve as interpolation sources. Now, we transition into using a `KDTree` from SciPy for efficient nearest neighbor searching. This structure allows us to quickly find the closest valid pixel for each invalid one.

Here's the crucial snippet of code showing how we perform the actual nearest neighbor interpolation:

```python
from scipy.spatial import KDTree
import numpy as np

def nearest_neighbor_interpolation(mask, invalid_value=0):
    """Performs nearest neighbor interpolation on a mask.

    Args:
        mask (np.ndarray): The input mask, likely a 3D RGB array.
        invalid_value (int, optional): The value indicating invalid pixels. Defaults to 0.

    Returns:
        np.ndarray: The interpolated mask.
    """
    invalid_pixels = find_invalid_pixels(mask, invalid_value)
    valid_pixels = find_valid_pixels(mask, invalid_value)


    if not valid_pixels.size:
      return mask #Nothing to interpolate, return original mask
    kdtree = KDTree(valid_pixels)

    interpolated_mask = np.copy(mask) #Create a copy so we don't mutate original

    for invalid_pixel in invalid_pixels:
        distance, index = kdtree.query(invalid_pixel)
        nearest_valid_pixel = valid_pixels[index]
        interpolated_mask[invalid_pixel[0], invalid_pixel[1]] = mask[nearest_valid_pixel[0], nearest_valid_pixel[1]]
    return interpolated_mask

# Example Usage
if __name__ == '__main__':
    mask_shape = (100,100,3)
    test_mask = np.zeros(mask_shape,dtype=np.uint8)
    #Create a small filled-in square for testing
    test_mask[20:80,20:80,:] = [255,0,0]
    #Add some invalid pixels (holes)
    test_mask[30:35, 30:35] = 0
    interpolated_mask = nearest_neighbor_interpolation(test_mask)
    print("Interpolation Complete!")
    #You would typically visualize or save the result at this point
    #e.g.: plt.imshow(interpolated_mask)
    #      plt.show()
```

The code above defines a function which performs the nearest neighbor interpolation.  The function constructs a `KDTree` using valid pixel locations. Then, for every invalid pixel, we use the `KDTree` to query and find the nearest valid pixel coordinates, finally, using these coordinates, the invalid pixel is set to the color of the nearest valid pixel. This fills in the invalid regions with the color of their neighbors.  Finally, the modified mask is returned.

One crucial consideration is how you might want to adapt this for very sparse masks, or where there are large gaps. In such cases, a single nearest neighbor approach might propagate artifacts if you don’t have enough densely clustered valid pixels in the original mask. A good approach in these scenarios, would be a combination with another method, such as a simple morphological operations like dilation first to expand the labels, followed by the nearest neighbor interpolation to smooth things out.

Finally, here’s an extension that includes a dilation step, as described.  This is important because while nearest-neighbor does well with small, isolated holes, it doesn't handle larger empty areas well. Dilation is an effective preprocessing step. This snippet includes dilation by using the `binary_dilation` function from SciPy's `ndimage` module. The function first dilates the valid regions, expanding them and reducing the 'holes,' then performs nearest neighbor interpolation on the expanded regions.

```python
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation

def dilated_nearest_neighbor_interpolation(mask, invalid_value=0, dilation_iterations=2):
    """Performs nearest neighbor interpolation after dilation on a mask.

    Args:
        mask (np.ndarray): The input mask, likely a 3D RGB array.
        invalid_value (int, optional): The value indicating invalid pixels. Defaults to 0.
        dilation_iterations (int, optional): Number of times to dilate the mask

    Returns:
        np.ndarray: The interpolated mask.
    """
    
    valid_mask = np.any(mask != invalid_value, axis=-1) # Create a boolean mask of valid pixels
    dilated_mask = binary_dilation(valid_mask,iterations=dilation_iterations).astype(np.uint8)
    dilated_mask_rgb = np.stack([dilated_mask,dilated_mask,dilated_mask], axis=-1) #Stack to make into 3D RGB equivalent
    
    temp_mask = mask.copy()
    #Replace original mask with dilated mask only in areas where it's newly valid.
    temp_mask[dilated_mask_rgb == 1] = [invalid_value,invalid_value,invalid_value]
    

    return nearest_neighbor_interpolation(temp_mask,invalid_value) #Use our existing interpolation on the dilated areas.

# Example usage:
if __name__ == '__main__':
    mask_shape = (100,100,3)
    test_mask = np.zeros(mask_shape,dtype=np.uint8)
    #Create a filled in square for testing
    test_mask[20:80,20:80,:] = [255,0,0]
    #Add some invalid pixels (larger hole)
    test_mask[30:45, 30:45] = 0

    dilated_interpolated_mask = dilated_nearest_neighbor_interpolation(test_mask)
    print("Dilated Interpolation Complete!")
    #You'd typically display or save the mask here
    #e.g.: plt.imshow(dilated_interpolated_mask)
    #      plt.show()
```

For anyone wanting a deeper dive into the algorithms and theory, I’d highly recommend “Digital Image Processing” by Rafael C. Gonzalez and Richard E. Woods. It’s a classic for a reason. Also, the SciPy documentation itself is fantastic for details on `KDTree` and `ndimage`. Another book worth checking out is “Programming Computer Vision with Python” by Jan Erik Solem, as it offers a more hands-on approach. When using techniques like nearest-neighbor interpolation, it's often not about finding a magical algorithm, but understanding the limitations of each method, and how it can be combined effectively with other methods to achieve the desired result. This approach provides robust and effective solutions in a practical context.
