---
title: "Does ImageDataGenerator flow_from_directory incorrectly resize images?"
date: "2024-12-23"
id: "does-imagedatagenerator-flowfromdirectory-incorrectly-resize-images"
---

Right then, let’s tackle this. It’s a question that’s probably caused a few raised eyebrows over the years, and frankly, I’ve seen my share of confusion around it. The query, specifically, centers on whether keras’ `ImageDataGenerator.flow_from_directory` function actually resizes images as expected, and the short, somewhat unsatisfying answer is: it largely does, but with some important caveats that can easily lead to misinterpretations. It's not a case of it being “incorrect” per se, but rather a matter of understanding the underlying mechanics and its implications on your image data pipeline.

Let me recount a particularly memorable project where this reared its head. Back in my days working on a plant disease detection system, we initially saw wildly inconsistent results. Images were fed in from various sources, and while `flow_from_directory` seemed convenient, we were seeing a performance drop on even the most basic CNN model. After much head-scratching, and, I must confess, a few hours combing through library source code, we realized that the resize operation, while present, wasn’t the end of the story. The crux of it lay in how this resizing is internally conducted alongside image augmentation, and the specific interpolation methods employed.

Here's a clearer picture of what's happening. `flow_from_directory` reads images from your specified directory structure, automatically labeling them based on subdirectory names. This part works as advertised. The resizing part occurs during image loading as specified by the `target_size` parameter. The crucial part that is sometimes overlooked is how these resized images are then augmented, which can affect the perceived quality of the resized images. By default, the resize will use a bilinear interpolation, but other interpolation methods are available to adjust quality and the speed of the process.

The perceived "incorrectness" largely comes from two primary sources. First, the interpolation method; secondly the understanding of what exactly is a “target size”. We might expect that the `target_size=(256, 256)` will uniformly resize all of our images to that size without any issues. However, a rectangular image, let’s say `(1024, 512)` resized to `(256,256)` needs more complex transformations than just resizing. The transformation is conducted through the function skimage.transform.resize (which uses `order=1` by default, which means bilinear resizing).

Let’s explore this with examples:

**Example 1: Basic Resizing with Bilinear Interpolation**

Let’s start with a simple demonstration. We'll create an `ImageDataGenerator` object with a `target_size` of `(150, 150)`, and then examine the shapes of the images after loading:

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Create a dummy directory structure and image files
os.makedirs("test_images/class_a", exist_ok=True)
os.makedirs("test_images/class_b", exist_ok=True)

# Creating dummy images with varying sizes
img1 = Image.new('RGB', (100, 200), color='red')
img2 = Image.new('RGB', (400, 300), color='green')
img3 = Image.new('RGB', (500, 500), color='blue')

img1.save("test_images/class_a/img1.png")
img2.save("test_images/class_b/img2.png")
img3.save("test_images/class_b/img3.png")

# Create an ImageDataGenerator with a specified target size
datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directory
image_generator = datagen.flow_from_directory(
    'test_images',
    target_size=(150, 150),
    batch_size=3,
    class_mode='categorical' # Adjust based on task
)

# Get the first batch of images
images, labels = next(image_generator)

# Print the shape of the first image in the batch
print(f"Image shape after resizing: {images[0].shape}")
print(f"Class labels: {labels}")

# clean up files
os.remove("test_images/class_a/img1.png")
os.remove("test_images/class_b/img2.png")
os.remove("test_images/class_b/img3.png")
os.rmdir("test_images/class_a")
os.rmdir("test_images/class_b")
os.rmdir("test_images")

```

This snippet creates a dummy structure with three images of varying sizes, loads them using `flow_from_directory`, and then prints the shape of the resized images, verifying that the resizing to (150,150) actually did happen. Notice that all images in a batch after `flow_from_directory` have the shape (150,150,3). In most cases, this is exactly what you would expect.

**Example 2: Impact of Different Interpolation Methods**

Now let's investigate the effect of different interpolation methods. `flow_from_directory` doesn't expose a direct parameter to control the interpolation. Instead, it relies on the underlying `skimage.transform.resize` function, which defaults to bilinear interpolation, but we can manually apply resizing and compare it to bilinear resizing.

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize as skresize
import os
from PIL import Image
import matplotlib.pyplot as plt

# Create a dummy directory structure and image files
os.makedirs("test_images_interpolation/class_a", exist_ok=True)

# Create a dummy image
original_img = Image.new('RGB', (400, 300), color='yellow')
original_img.save("test_images_interpolation/class_a/image.png")


datagen = ImageDataGenerator(rescale=1./255)
image_generator = datagen.flow_from_directory(
    'test_images_interpolation',
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Resizing using flow_from_directory (bilinear by default)
resized_image_by_flow, labels = next(image_generator)

# Resizing using skimage.transform.resize with different interpolation methods
pil_image = Image.open("test_images_interpolation/class_a/image.png")
numpy_image = np.array(pil_image)
resized_image_bicubic = skresize(numpy_image,(150,150),order=3,preserve_range=True, channel_axis=2).astype(np.uint8)


# Convert for display
resized_image_by_flow_disp = (resized_image_by_flow[0] * 255).astype(np.uint8)
pil_resized_image_by_flow = Image.fromarray(resized_image_by_flow_disp)
pil_resized_image_bicubic = Image.fromarray(resized_image_bicubic)


# Display the resized images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(pil_resized_image_by_flow)
axes[0].set_title("Resized by flow (Bilinear)")
axes[1].imshow(pil_resized_image_bicubic)
axes[1].set_title("Resized by Skimage (Bicubic)")
plt.show()

# clean up files
os.remove("test_images_interpolation/class_a/image.png")
os.rmdir("test_images_interpolation/class_a")
os.rmdir("test_images_interpolation")
```

This example demonstrates that while the default bilinear interpolation may be acceptable in many cases, you have to explicitly use an image processing library, such as skimage, to utilize other types of interpolations, such as bicubic. The quality difference can become noticeable, especially if you have high frequency details in your images.

**Example 3: The 'Target Size' Interpretation**

Lastly, let’s think about the interpretation of the `target_size`. The `target_size` argument does not guarantee that your image will be resized into that shape by preserving the aspect ratio. The resizing process will modify the size of an input image, regardless of the input's aspect ratio. That means the same image of shape (x,y,3), after `flow_from_directory` is applied, will have a target shape of (`target_size[0]`,`target_size[1]`,3), no matter what the original aspect ratio of (x,y).

Here's a conceptual snippet:

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Create a dummy directory structure and image files
os.makedirs("test_images_aspect/class_a", exist_ok=True)

# Creating dummy image with a rectangular shape
img4 = Image.new('RGB', (600, 200), color='purple')
img4.save("test_images_aspect/class_a/img4.png")


datagen = ImageDataGenerator(rescale=1./255)
image_generator = datagen.flow_from_directory(
    'test_images_aspect',
    target_size=(100, 100),
    batch_size=1,
    class_mode='categorical' # Adjust based on task
)

images, labels = next(image_generator)
print(f"Resized image shape: {images[0].shape}")


# clean up files
os.remove("test_images_aspect/class_a/img4.png")
os.rmdir("test_images_aspect/class_a")
os.rmdir("test_images_aspect")

```
This will still give you an image of the shape (100,100,3). There are other image preprocessing utilities in Keras that could be used to maintain the aspect ratio, such as `tf.image.resize_with_pad` or `tf.image.resize`, however, `flow_from_directory` does not support these methods out of the box. This is something that can be confusing if you expect your resizing to also preserve aspect ratio in cases where an image is rectangular rather than square.

In summary, `flow_from_directory` does resize images as instructed via the `target_size` parameter. However, the subtle aspects, such as default interpolation (bilinear) and the non-preservation of aspect ratio, could lead to the perception that it's “incorrect” or “not what you expected." I've seen firsthand how not understanding these details can skew results during training.

If you want to dive deeper, I recommend the following: for a general understanding of digital image processing, *Digital Image Processing* by Rafael C. Gonzalez and Richard E. Woods is an excellent starting point; for understanding the mathematics behind image resizing, and specifically different interpolation methods, have a look at the relevant sections of *Computer Graphics: Principles and Practice* by James D. Foley, Andries van Dam, Steven K. Feiner, and John F. Hughes. You should also consider carefully reading the source code for `ImageDataGenerator` and `skimage.transform.resize`, particularly if you need to fully grasp the implementation details. In addition, explore resources on practical aspects of image preprocessing for deep learning, such as “A Gentle Introduction to Image Preprocessing for Deep Learning” by Jason Brownlee, published on Machine Learning Mastery. This knowledge will help you make informed choices regarding your image preprocessing pipeline.
