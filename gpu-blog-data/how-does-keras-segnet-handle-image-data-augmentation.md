---
title: "How does Keras SegNet handle image data augmentation with ImageDataGenerator for training?"
date: "2025-01-30"
id: "how-does-keras-segnet-handle-image-data-augmentation"
---
SegNet, a deep learning architecture for semantic image segmentation, inherently benefits from data augmentation to improve generalization and reduce overfitting, especially when training with limited data.  The `ImageDataGenerator` class within Keras provides the primary mechanism for applying these augmentations during training.  The generator does not directly alter the original image files stored on disk; instead, it performs on-the-fly transformations each time it provides a batch of data to the training process. This dynamic approach is crucial, particularly with high-resolution images or large datasets, as it circumvents the need to pre-compute and store augmented versions of all images. This dynamic application of augmentation both saves storage and offers a far greater diversity in the training data.

The core of augmentation via Keras's `ImageDataGenerator` lies in specifying transformation parameters when instantiating the class.  These parameters detail the types of augmentations desired and their respective magnitudes.  Crucially, for semantic segmentation tasks such as those often addressed by SegNet, we typically need paired augmentations. This means that whatever transformation is applied to the input image should be identically applied to the corresponding segmentation mask. The `ImageDataGenerator` class is, by itself, not directly aware of segmentation masks and requires an adaptation, typically with the usage of separate generators for the input image and the label mask, seeded with identical random states.

I've personally trained several SegNet models using Keras, and from my experience, carefully selecting the right augmentation techniques and magnitudes is a key step for improving model performance.  Insufficient augmentation will not sufficiently regularize the model, potentially leading to overfitting to specifics of the training dataset. Overly aggressive augmentation can introduce artificial artifacts or transformations that are not representative of real-world data, confusing the model during learning. The balance is found by judicious parameter tuning and monitoring of validation metrics throughout the process.  The critical consideration is always that the augmentations are applied realistically and preserve core semantic information.

Here are some practical examples of how to use `ImageDataGenerator` for training SegNet, paying particular attention to paired image and mask augmentation. The assumption in all examples is that there exist two directories, say `images` and `masks`, with images in the image directory matching their label masks in the corresponding locations in the mask directory.

**Example 1: Basic Augmentations with Matching Seed**

This first example demonstrates several basic but powerful augmentations: rescaling, rotation, and horizontal flipping. The `rescale` argument normalizes pixel values to the [0, 1] range. `rotation_range` allows for random rotations, and `horizontal_flip` introduces mirrored versions of images. Importantly, the same random seed is used for both the image and mask generator, guaranteeing a corresponding augmentation on each pair.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

seed = 42  # Arbitrary, but consistently used for pairing

# Image Augmentation Generator
image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'  # Avoid black regions by filling with nearest pixel
)


# Mask Augmentation Generator - should use the same arguments and seed as image_datagen
mask_datagen = ImageDataGenerator(
    rescale=1./255, # Scale as well to ensure consistent scale and prevent errors later.
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare Generators
image_generator = image_datagen.flow_from_directory(
    'images',
    target_size=(256, 256), # Resize all images to a consistent size
    batch_size=32,
    class_mode=None,
    seed=seed # Use a seed for augmentation consistency across samples in each batch.
)

mask_generator = mask_datagen.flow_from_directory(
    'masks',
    target_size=(256, 256), # Use the same resizing and seed.
    batch_size=32,
    class_mode=None,
    seed=seed
)


# Combined Generator to allow yield to give tuples.
def combined_generator(img_gen, mask_gen):
    while True:
        img_batch = img_gen.next()
        mask_batch = mask_gen.next()
        yield img_batch, mask_batch # Yield pairs of corresponding image and mask batches.


# Initialize combined generator.
train_generator = combined_generator(image_generator, mask_generator)
```

In this example, a `combined_generator` is necessary to synchronize and output image-mask pairs for use in the SegNet model's `fit()` function. It is not sufficient to pass the separate generators, as the model requires an iterable yielding `(images, masks)` tuples. The `flow_from_directory` method expects images to be in a directory structure with subfolders for each class. Since we are using it for masks, we can simply add all masks in the same folder. The `class_mode=None` argument prevents unwanted categorical label encoding. A critical point here is that we rescale pixel values of the masks by `1./255` as well. This normalizes masks to float data type from an initial integer (0-255) format which is sometimes used, thereby preventing errors further down in the process.

**Example 2: Adding Shear and Zoom**

This example expands upon the basic augmentations, adding shear and zoom capabilities. These can be beneficial for simulating perspective distortions and varying object scales in the data, respectively.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


seed = 123  # Another arbitrary seed, also used for consistent pairing.

# Augmented Image Data Generator.
image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2, # add shear.
    zoom_range=0.2,  # add zoom
    fill_mode='nearest'
)

# Augmented Mask Data Generator
mask_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

image_generator = image_datagen.flow_from_directory(
    'images',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,
    seed=seed
)

mask_generator = mask_datagen.flow_from_directory(
    'masks',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,
    seed=seed
)


def combined_generator(img_gen, mask_gen):
    while True:
        img_batch = img_gen.next()
        mask_batch = mask_gen.next()
        yield img_batch, mask_batch


train_generator = combined_generator(image_generator, mask_generator)
```

The addition of `shear_range` and `zoom_range` is straightforward; the key remains the use of identical seeds in each of the `ImageDataGenerator` instances. By adding these augmentation methods the model becomes more robust to changes in viewpoint and scale within the data, increasing its ability to generalize to new, unseen instances.

**Example 3: Using a Custom Function**

The  `ImageDataGenerator` class also allows for a custom preprocessing function, offering even more flexibility. Here's an example where we perform an additional color jitter. While Keras offers inbuilt color augmentations as well, this example shows one way in which these can be implemented using a custom function.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

seed = 777 # Yet another arbitrary seed for paired consistency.


def custom_preprocessing(img):
  img = tf.image.random_brightness(img, max_delta=0.2)
  img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
  img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
  img = tf.image.random_hue(img, max_delta=0.1)
  return img



image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    preprocessing_function=custom_preprocessing,
    fill_mode='nearest'

)

mask_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_generator = image_datagen.flow_from_directory(
    'images',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,
    seed=seed
)


mask_generator = mask_datagen.flow_from_directory(
    'masks',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,
    seed=seed
)


def combined_generator(img_gen, mask_gen):
    while True:
        img_batch = img_gen.next()
        mask_batch = mask_gen.next()
        yield img_batch, mask_batch


train_generator = combined_generator(image_generator, mask_generator)
```
Here, the `custom_preprocessing` function alters the input image's brightness, contrast, saturation, and hue using methods from `tensorflow.image`.  The important detail is that this custom function is called *before* any other augmentation is applied. Notice that the mask does not get this custom augmentation because it does not make sense to alter the labels. The custom function takes the image array as input, and returns an altered version.  This demonstrates how the `ImageDataGenerator` class can be easily extended via this `preprocessing_function` to accommodate complex augmentations or preprocessing routines, opening up additional options beyond those immediately offered by the parameters within the class.

For more detailed theoretical information regarding data augmentation techniques in deep learning, numerous academic texts and research papers on image processing are available from major scientific publishers. Specifically, reviewing research publications in journals focusing on computer vision and pattern recognition will offer in-depth theoretical foundations for the practical augmentation strategies shown in the provided code examples. Additionally, official Keras documentation provides exhaustive details about the functionalities and configurable options of the `ImageDataGenerator` class. Finally, online resources pertaining to deep learning and computer vision offer numerous case studies and practical examples that demonstrate the application of image augmentation in similar scenarios to what I've described.
