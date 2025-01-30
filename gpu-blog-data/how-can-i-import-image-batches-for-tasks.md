---
title: "How can I import image batches for tasks other than classification?"
date: "2025-01-30"
id: "how-can-i-import-image-batches-for-tasks"
---
Batching image data for non-classification tasks, such as semantic segmentation, object detection, or image generation, requires a careful approach distinct from simple classification pipelines. The primary divergence stems from the necessity to maintain spatial correspondence between inputs and outputs, alongside handling diverse label formats and, in some cases, variable-sized images. My experience developing a depth estimation model highlighted these challenges, requiring a custom data loading strategy to efficiently feed the network with batches of aligned image and depth maps.

A foundational issue is that unlike classification where the target is a single class label, these tasks typically involve dense pixel-wise outputs or bounding box coordinates. Consequently, simple one-hot encoding or label lists become inadequate. We need to ensure each batch contains correctly formatted inputs and corresponding outputs that the model can learn from. This involves more than just reading images; it requires preprocessing steps tailored to the specific task and the output representation.

For example, consider a semantic segmentation problem. Here, the goal is to assign each pixel a class label (e.g., road, building, tree). We can't merely feed the network batches of raw images and a corresponding list of class labels. We require a method to transform each input image into a batch of images, and simultaneously transform the corresponding pixel-wise label map into a batch of target maps, maintaining alignment. If we have images of shape `(H, W, 3)` and pixel labels ranging from 0 to `C-1`, the training batch for a segmentation model, in general, comprises input tensors with shape `(B, H, W, 3)` and target tensors with shape `(B, H, W, C)` (for one-hot encoding), where `B` is the batch size.

Another challenge arises when working with variable-sized input images. Deep learning models require input tensors with fixed dimensions. While we can resize all images to a uniform size, this can introduce distortions, particularly in tasks where finer details are crucial. Techniques like padding, cropping, or a combination of both, are often used to maintain the aspect ratio while ensuring all images within a batch have the same shape. This becomes more intricate when handling associated ground truth data as we need to perform similar transformations, keeping alignment in mind.

Below are three code examples, using Python and NumPy, illustrating common approaches, assuming the availability of image loading functionality using libraries such as `PIL` or `opencv`. In a real-world deep learning scenario, these Numpy based loading procedures are normally superseded with optimized tensor-based loading approaches using libraries such as TensorFlow, PyTorch, or Jax. However, it serves the purpose of illustrating underlying concepts here.

**Example 1: Batching with Resize and One-Hot Encoding for Segmentation**

This example demonstrates creating batches from a list of image file paths and a corresponding list of segmentation mask file paths. It resizes all images and masks to a common size and converts the segmentation masks to a one-hot encoded format.

```python
import numpy as np
from PIL import Image

def create_segmentation_batch(image_paths, mask_paths, batch_size, target_size, num_classes):
    """Creates batches of images and segmentation masks.

    Args:
        image_paths: A list of image file paths.
        mask_paths: A list of corresponding mask file paths.
        batch_size: The batch size.
        target_size: A tuple (height, width) for resizing.
        num_classes: The number of classes in segmentation.

    Returns:
        A tuple of (batch_images, batch_masks) as numpy arrays.
    """
    batch_images = []
    batch_masks = []
    for i in range(batch_size):
        img_path = image_paths[i]
        mask_path = mask_paths[i]

        # Load and resize the image
        img = Image.open(img_path).resize(target_size)
        img_arr = np.asarray(img) / 255.0 # Normalize to [0,1]
        batch_images.append(img_arr)

        # Load, resize and one-hot encode the mask
        mask = Image.open(mask_path).resize(target_size, Image.NEAREST)
        mask_arr = np.asarray(mask, dtype=np.int32)

        one_hot_mask = np.eye(num_classes)[mask_arr]
        batch_masks.append(one_hot_mask)

    return np.stack(batch_images, axis=0), np.stack(batch_masks, axis=0)

# Example usage
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
mask_paths = ["mask1.png", "mask2.png", "mask3.png", "mask4.png"]
batch_size = 4
target_size = (256, 256)
num_classes = 5

batch_images, batch_masks = create_segmentation_batch(image_paths, mask_paths, batch_size, target_size, num_classes)

print("Image Batch Shape:", batch_images.shape)
print("Mask Batch Shape:", batch_masks.shape)
```
This example demonstrates loading images and their associated masks, resizing them, and converting the masks to a one-hot representation. The normalization step on images is standard, and the use of `Image.NEAREST` preserves discrete classes during resizing of the mask images.

**Example 2: Batching with Padding for Object Detection**

This example shows how to create a batch of images and bounding box annotations for an object detection task, utilizing padding to ensure all images have the same dimensions while retaining the aspect ratio. The bounding boxes are also adjusted accordingly.

```python
import numpy as np
from PIL import Image

def create_detection_batch(image_paths, bbox_annotations, batch_size, target_size):
    """Creates batches of images and bounding box annotations using padding.

    Args:
        image_paths: A list of image file paths.
        bbox_annotations: A list of bounding boxes (x_min, y_min, x_max, y_max) for each image
        batch_size: The batch size.
        target_size: A tuple (target_height, target_width) for the padded output.

    Returns:
       A tuple of (batch_images, batch_bboxes) as numpy arrays
    """

    batch_images = []
    batch_bboxes = []

    for i in range(batch_size):
      img_path = image_paths[i]
      bbox = np.array(bbox_annotations[i]) # assumes bounding box info are a list of lists
      img = Image.open(img_path)
      img_width, img_height = img.size

      target_height, target_width = target_size

      # Calculate padding required
      pad_h = max(0, target_height - img_height)
      pad_w = max(0, target_width - img_width)
      pad_top = pad_h // 2
      pad_bottom = pad_h - pad_top
      pad_left = pad_w // 2
      pad_right = pad_w - pad_left

      # Pad the image
      padded_img = Image.new('RGB', (img_width+pad_left+pad_right,img_height+pad_top+pad_bottom),(0,0,0))
      padded_img.paste(img, (pad_left,pad_top))
      padded_img_arr = np.asarray(padded_img) / 255.0
      batch_images.append(padded_img_arr)

      # Adjust the bounding box to the padded image
      x_min, y_min, x_max, y_max = bbox
      x_min += pad_left
      x_max += pad_left
      y_min += pad_top
      y_max += pad_top
      batch_bboxes.append(np.array([x_min, y_min, x_max, y_max]))
      

    return np.stack(batch_images, axis=0), np.stack(batch_bboxes, axis=0)

# Example Usage:
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
bbox_annotations = [[20, 30, 100, 150], [50, 60, 180, 200], [10, 20, 120, 180], [40, 50, 160, 220]]
batch_size = 4
target_size = (300, 300)

batch_images, batch_bboxes = create_detection_batch(image_paths, bbox_annotations, batch_size, target_size)

print("Padded Image Batch Shape:", batch_images.shape)
print("Bounding Box Batch Shape:", batch_bboxes.shape)
```
Here, padding with zeros (`(0,0,0)` for RGB images) is employed, adjusting the coordinates of bounding box accordingly. This keeps the aspect ratio of each input image without warping the shape, while making all images in a batch have same dimensions.

**Example 3: Batching for Image Generation with Augmentation**

This example illustrates a simple technique of augmenting images on the fly, combining it with random cropping to create batches with variability for tasks like GAN training.

```python
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random

def create_generation_batch(image_paths, batch_size, target_size):
  """Creates batches of images with on-the-fly random augmentations.

  Args:
    image_paths: A list of image file paths.
    batch_size: The batch size.
    target_size: A tuple (height, width) for cropping.

  Returns:
      A numpy array of shape (batch_size, height, width, 3)
  """

  batch_images = []
  for i in range(batch_size):
    img_path = random.choice(image_paths)
    img = Image.open(img_path)
    
    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    brightness_factor = random.uniform(0.8, 1.2)
    img = enhancer.enhance(brightness_factor)

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    contrast_factor = random.uniform(0.8, 1.2)
    img = enhancer.enhance(contrast_factor)

    # Random flips
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    if random.random() < 0.5:
        img = ImageOps.flip(img)

    #Random crop
    w, h = img.size
    target_h, target_w = target_size

    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)

    img = img.crop((x,y, x+target_w, y+target_h))

    img_arr = np.asarray(img) / 255.0
    batch_images.append(img_arr)

  return np.stack(batch_images, axis=0)

image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
batch_size = 4
target_size = (256, 256)

batch_images = create_generation_batch(image_paths, batch_size, target_size)
print("Image Batch Shape:", batch_images.shape)
```
This example shows data augmentation is applied during batch loading, and the images are randomly cropped to a desired target size. These simple augmentation strategies can improve model robustness and generalization.

For further study on efficient data handling, I recommend exploring resources on TensorFlow Data API and PyTorch's DataLoaders. These provide powerful and optimized means to load and preprocess batches of data, incorporating techniques like prefetching and multithreading. Also, I advise researching best practices for image resizing, and image augmentation techniques that are specific to your needs. Investigating best practices in batching will greatly improve both the efficiency and performance of your deep learning models.
