---
title: "minimum filter in image processing algorithm?"
date: "2024-12-13"
id: "minimum-filter-in-image-processing-algorithm"
---

so you're asking about a minimum filter in image processing right Been there done that trust me I've wrestled with this beast more times than I care to admit

So basically a minimum filter is a type of image processing operation that goes through an image pixel by pixel It doesn't care about the color in this context it operates on the pixel values in its neighborhood it looks around a selected pixel and finds the minimum pixel value within a predefined area and then it replaces the original pixel value with this minimum value So you can think of it as a kind of 'darkening' or 'erosion' operation which is useful in many contexts usually to reduce noise to make things more homogeneous It's not about averaging or blurring that’s a different thing this is about finding the darkest spot and making the center pixel match that darkest spot

I remember when I first started getting into image processing I was working on this project a long time ago back in University when I was experimenting with some segmentation algorithm I had some images with noise issues making segmentation hard and I thought some kind of filter would help I first thought about blur but it made the edges fade and was bad for my specific case after looking for other alternatives I came across with the minimum filter and it was perfect it removed that pepper like noise I was dealing with which made the borders of the objects way easier to detect

Let’s break down a bit how it works it is pretty simple First you need a neighborhood also called a kernel which is like a sliding window that you move over each pixel in the image this neighborhood is usually a small square a 3x3 or 5x5 matrix is common but you can play with the sizes and it is one of the main parameters in the filter. For each pixel you select the neighborhood and find the minimum value inside it and then you replace the central pixel of that neighborhood with the minimum value you've found I know it’s probably easier to understand with code so let me give you an example in Python using NumPy since that’s what most people use for scientific stuff

```python
import numpy as np

def minimum_filter(image, kernel_size):
    height, width = image.shape
    pad_size = kernel_size // 2 # how much to pad on each side

    # Zero-pad the image to handle boundary pixels
    padded_image = np.pad(image, pad_size, mode='constant')

    output_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Extract the neighborhood using array slicing
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            # Find minimum pixel value in the kernel
            min_pixel = np.min(neighborhood)
            
            # Set current pixel to the minimum value found
            output_image[i, j] = min_pixel
    
    return output_image

# Example use
if __name__ == '__main__':
  example_image = np.array([
      [10, 20, 30, 40, 50],
      [15, 5, 25, 35, 45],
      [20, 30, 40, 6, 55],
      [25, 35, 45, 55, 65],
      [30, 40, 50, 60, 70]
  ], dtype=np.uint8)

  filtered_image = minimum_filter(example_image, 3) # 3x3 kernel

  print("Original Image:")
  print(example_image)
  print("\nFiltered Image:")
  print(filtered_image)

```

This is a basic implementation it's not the most efficient code but it gets the job done you see the padding on the borders is important or you will have strange effects on the borders because you won't have enough pixels to create the neighborhood we just apply a constant padding which in this case will be of zeros also if you work with color images you would apply this function in each of the channels. For a gray image is as simple as that though

And if you wanted a more optimized implementation there are other ways to do it For example you can avoid looping by using the `scipy.ndimage.minimum_filter` library which is a way faster implementation of the same idea here’s an example:

```python
import numpy as np
from scipy.ndimage import minimum_filter

def minimum_filter_scipy(image, kernel_size):
    filtered_image = minimum_filter(image, size=kernel_size)
    return filtered_image

# Example use
if __name__ == '__main__':
  example_image = np.array([
      [10, 20, 30, 40, 50],
      [15, 5, 25, 35, 45],
      [20, 30, 40, 6, 55],
      [25, 35, 45, 55, 65],
      [30, 40, 50, 60, 70]
  ], dtype=np.uint8)

  filtered_image = minimum_filter_scipy(example_image, 3) # 3x3 kernel

  print("Original Image:")
  print(example_image)
  print("\nFiltered Image:")
  print(filtered_image)
```
This way the code is more concise since the function already does all of the necessary heavy lifting but the underlying logic is exactly the same it still needs the same concepts of neighborhood and minimum value search

Sometimes instead of a square neighborhood you might need a different shape for the kernel say a cross or a circle shaped one which brings us to the concept of the structuring element that is more complex than a simple square neighborhood and if you are using that you will need to use a different approach but it is usually something like this:
```python
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.morphology import disk
import matplotlib.pyplot as plt

def minimum_filter_scipy_structured(image, radius):
    # Create a disk-shaped structuring element
    selem = disk(radius)

    # Perform binary erosion (works as min filter on grayscale images using morphological operation)
    filtered_image = binary_erosion(image, selem).astype(image.dtype)
    return filtered_image

# Example use
if __name__ == '__main__':
    example_image = np.array([
    [10, 20, 30, 40, 50],
    [15, 5, 25, 35, 45],
    [20, 30, 40, 6, 55],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70]
    ], dtype=np.uint8)
    
    radius = 1 # Radius for the disk structuring element

    filtered_image = minimum_filter_scipy_structured(example_image, radius) # 3x3 kernel
    print("Original Image:")
    print(example_image)
    print("\nFiltered Image:")
    print(filtered_image)
```

This shows you how to use different kernels by applying what is known as morphological erosion which is actually a minimum filter but using the concept of structured element that allows the use of different forms. Note that morphological operations use binary images but they will work on greyscale images too as a minimum filter. The result depends on the form of the structure element in this case a circle and its radius which will determine the overall neighborhood size. 

Now remember it's not a magical fix-all for image issues It has its limitations and can sometimes distort features or even make things worse if the kernel is too big or the image has structures that look similar to noise always experiment with different parameters and see what fits best your images and your needs also always experiment a lot

Finally if you're looking for further reading I would suggest looking into:

*   **"Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods:** This is a classic textbook in the field of image processing it covers everything in deep detail including morphological operations like erosion which are used as a minimum filter with custom kernels
*   **"Computer Vision Algorithms and Applications" by Richard Szeliski:** This book has a broad overview of image processing algorithms including minimum and maximum filters among many other useful techniques.
*   **Scientific papers on morphological image analysis:** Search on google scholar for papers on morphological image processing operations and you'll find tons of details on minimum and maximum filters or how to design structuring elements that might work for a very specific application you have in mind.

I hope this helps you in your endeavor if you have any other questions please feel free to ask I might or might not have an answer and if that’s the case don’t worry someone else here will definitely know how to answer.
