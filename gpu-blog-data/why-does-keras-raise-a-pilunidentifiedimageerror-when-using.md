---
title: "Why does Keras raise a PIL.UnidentifiedImageError when using a non-Pillow image library?"
date: "2025-01-30"
id: "why-does-keras-raise-a-pilunidentifiedimageerror-when-using"
---
The core issue arises from Keras's reliance on Pillow (PIL) as its default image processing backend when loading images via functions like `keras.utils.load_img` or when implicitly working with image data loaders. When a non-Pillow library such as OpenCV or Scikit-image is employed to read an image file, the resulting data structure—typically a NumPy array—does not conform to the expected data type or internal representation that Keras functions are designed to accept, thus leading to the `PIL.UnidentifiedImageError`. This error signals that the Keras image handling routines cannot interpret the input as a valid image format according to Pillow’s specifications.

I encountered this firsthand during a project involving a large image dataset where, for preprocessing speed and features, I opted to use OpenCV to handle image reading and resizing operations. Initially, I loaded the images using OpenCV, converted the resultant NumPy arrays to the standard image format expected by Keras (i.e., height, width, channels, data type) and then attempted to use them as inputs to Keras layers. This produced intermittent failures. The underlying problem was not the shape or data type of the array itself. Instead, Keras's `load_img` function and many internal image handling mechanisms expect to be passed either a file path that it can interpret using Pillow or a Pillow Image object, not a NumPy array, even if that array contains pixel data in the correct format. When a non-Pillow generated NumPy array is passed, Keras attempts to interpret the structure using Pillow’s file format handling code which fails, resulting in this error.

The problem surfaces subtly as often the array looks "correct" from a user perspective. The pixel data are typically represented as NumPy arrays with integer (e.g. `uint8`) or floating-point data types, suitable for image representation. However, Pillow’s internal functions expect either a direct byte stream from a file (that it can parse) or the special `PIL.Image` object, which encapsulates additional metadata and format information. When a NumPy array, created with other libraries such as OpenCV, is passed to Keras image handling routines, these functions mistakenly treat it as a file path or file-like object, leading to the `PIL.UnidentifiedImageError` when Pillow cannot identify the file format (or in this case the data type) at the provided location.

To illustrate this, consider the following code examples:

**Example 1: Direct Keras image load resulting in the error**

```python
import cv2
import keras
import numpy as np

# Load image using OpenCV
img_cv = cv2.imread('example.jpg')

# Attempt to use the OpenCV image with Keras (will throw the error)
try:
  img_keras = keras.utils.load_img(img_cv)
except Exception as e:
  print(f"Error encountered: {e}")
```

In this example, OpenCV reads an image from the file `example.jpg`. The `cv2.imread` function returns a NumPy array representing the image data.  When that array is passed directly to `keras.utils.load_img`, Keras incorrectly assumes the array is either a file path or a PIL image object. Since the array is neither, Pillow’s image loading routines fail with the `PIL.UnidentifiedImageError`. It highlights the problem:  Keras's API is expecting specific input types which are different from the structures provided by non-Pillow image handling libraries. The error is not rooted in the image data itself, but in how it's being interpreted by Keras.

**Example 2: Correct Usage with Pillow**

```python
from PIL import Image
import keras
import numpy as np

# Load image using Pillow
img_pil = Image.open('example.jpg')

# Use the PIL image with Keras (works correctly)
img_keras = keras.utils.load_img(img_pil)

# Convert to numpy array for further processing
img_arr = keras.utils.img_to_array(img_keras)

print(f"Keras image format: {type(img_keras)}")
print(f"Numpy array shape: {img_arr.shape}")
```

Here, the image is loaded using `PIL.Image.open`. The returned `PIL.Image` object is a data structure that Keras functions can interpret directly because it's a known data type for the image loader. The Keras function `keras.utils.load_img` correctly identifies the input and processes the image, resulting in a PIL image object. Finally, `keras.utils.img_to_array` can be used to convert the PIL image object into the desired numpy array format if further numerical processing is required. This illustrates the proper use of the Keras image loading routine with Pillow.

**Example 3: Converting OpenCV to Pillow and then to Keras**

```python
import cv2
from PIL import Image
import keras
import numpy as np

# Load image using OpenCV
img_cv = cv2.imread('example.jpg')
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # convert BGR to RGB

# Convert OpenCV array to Pillow Image
img_pil = Image.fromarray(img_cv)


#Use the pillow image in Keras load_img
img_keras = keras.utils.load_img(img_pil)

# Convert to numpy array for further processing
img_arr = keras.utils.img_to_array(img_keras)

print(f"Keras image format: {type(img_keras)}")
print(f"Numpy array shape: {img_arr.shape}")

```

This example provides a direct solution to the identified problem. First, we load the image using OpenCV. OpenCV returns the pixel data in BGR order (blue, green, red). Since Pillow works with RGB order, a conversion needs to be performed to ensure color accuracy. The `cv2.cvtColor` function performs the color channel conversion. Then, we leverage the `PIL.Image.fromarray` function to convert the OpenCV-generated NumPy array into a PIL Image object. This newly created Pillow object can now be correctly processed by `keras.utils.load_img`. This demonstrates the conversion process required for proper usage when starting with non-Pillow image data.

The correct approach when utilizing non-Pillow image libraries like OpenCV is to explicitly convert the result into a PIL Image object before using any Keras image related functions. Alternatively, the user could pass image file paths to Keras, which will then load the images internally using PIL. The most direct way to avoid this error is to either: a) use Pillow to load images from the start; or b) when using other libraries, convert the resulting NumPy arrays into Pillow image objects before passing them to Keras for image handling. This conversion provides the required metadata and internal structure that Keras image processing routines expect.

For additional learning, consult documentation for the Pillow library, particularly the `PIL.Image` module and its image loading methods and image format specifications. Familiarize oneself with the functions for converting between different image data representations, notably how to create a `PIL.Image` object from a NumPy array using `Image.fromarray()`. Further understanding can be gleaned by inspecting the Keras documentation, particularly sections related to image preprocessing, image data loading using generators, and image augmentation, with attention to expected input data types. Research into the Keras source code could also prove beneficial, specifically the parts that handle image data inputs.
