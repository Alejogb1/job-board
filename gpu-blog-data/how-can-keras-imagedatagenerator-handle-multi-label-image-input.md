---
title: "How can Keras ImageDataGenerator handle multi-label image input and output?"
date: "2025-01-30"
id: "how-can-keras-imagedatagenerator-handle-multi-label-image-input"
---
Multi-label image classification, where each image can belong to multiple categories, requires careful handling of data during training in deep learning models. Keras' `ImageDataGenerator`, designed primarily for single-label image classification or image augmentation, needs adaptation to effectively process multi-label scenarios. I've encountered this challenge numerous times when working with medical image datasets where pathologies are often overlapping, requiring the network to predict multiple diagnoses.

The core issue stems from the fact that `ImageDataGenerator` typically expects a single label per image, represented either as an integer index in the case of categorical classification, or as a single numerical value for regression. Multi-label classification requires a binary vector as the target, where each position in the vector corresponds to a distinct label, and a value of 1 indicates the presence of that label, while 0 signifies absence. `ImageDataGenerator`, in its default configuration, does not directly generate these vectors from file names or directories, requiring a custom approach to integrate with the generator's workflow.

Specifically, the challenge lies in the `flow_from_directory` method, which infers labels from the subdirectory structure. This method is unsuitable for multi-label problems. Furthermore, while `flow` method could theoretically work with pre-generated labels, it limits the data preprocessing flexibility provided by `ImageDataGenerator`. Thus, we need a tailored method to provide correct input to our model.

The solution involves creating a custom generator that wraps `ImageDataGenerator` and produces both the images and corresponding multi-hot encoded labels. This essentially involves two main steps. Firstly, parsing the file names to extract the relevant labels and create a multi-hot encoded target vector. Secondly, using the base generator’s `flow` method to handle all augmentations, while providing it with our extracted labels.

Here’s how this works in practice. I’ll illustrate with three scenarios, each building upon the last to demonstrate increasing complexity.

**Scenario 1: Simple Multi-label Classification with a Fixed Label Set**

In this scenario, labels are explicitly embedded in file names, such as "image_cat_dog.jpg". I’ve found this to be a common pattern in datasets where labels are already assigned via file annotations.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def multilabel_data_generator(image_dir, label_map, batch_size, image_size=(256,256)):
    datagen = ImageDataGenerator(rescale=1./255, # basic augmentation, change if needed
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)


    image_filenames = os.listdir(image_dir)

    def generator():
        while True:
            batch_images = []
            batch_labels = []
            for _ in range(batch_size):
                image_filename = np.random.choice(image_filenames)
                image_path = os.path.join(image_dir, image_filename)
                image = tf.keras.utils.load_img(image_path, target_size=image_size)
                image = tf.keras.utils.img_to_array(image)


                labels_from_filename = image_filename.split('_')[1:-1] # split and remove image id and file suffix

                label_vector = np.zeros(len(label_map), dtype=np.float32)
                for label_str in labels_from_filename:
                    if label_str in label_map:
                        label_index = label_map.index(label_str)
                        label_vector[label_index] = 1.0


                batch_images.append(image)
                batch_labels.append(label_vector)


            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield (batch_images, batch_labels)

    return generator()
# Example Usage
image_dir = 'data/images'  # Place your images here
label_map = ['cat', 'dog', 'bird', 'fish'] # All possible labels, ordered
batch_size = 32
image_size = (128,128)
multi_label_gen = multilabel_data_generator(image_dir, label_map, batch_size, image_size)

# Now you can use multi_label_gen with model.fit using steps_per_epoch
```

This example demonstrates a foundational custom generator. The `multilabel_data_generator` function accepts the image directory, a predefined list of labels, batch size, and image size. It then creates an inner generator function that uses a random sampling scheme and performs the following tasks: read the image file and convert it into an array. It parses the filename to extract multi-labels, converting them into a one-hot encoded NumPy vector. Finally, it packages the batch image arrays and batch one-hot encoded target vectors. This solution uses standard python for looping and list manipulation, with `tensorflow.keras` to load the images.

**Scenario 2: Multi-label Classification with a Label File**

In some situations, labels are not directly encoded in file names, but instead provided in a separate file. I've often seen this in medical image data, where annotation files link image names to diagnosis codes. This scenario requires modification to read from such files.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def multilabel_data_generator_with_csv(image_dir, labels_csv, label_map, batch_size, image_size=(256,256)):
    datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

    labels_df = pd.read_csv(labels_csv)
    labels_df.set_index('filename', inplace=True)

    image_filenames = labels_df.index.tolist()

    def generator():
        while True:
            batch_images = []
            batch_labels = []
            for _ in range(batch_size):
                image_filename = np.random.choice(image_filenames)
                image_path = os.path.join(image_dir, image_filename)
                image = tf.keras.utils.load_img(image_path, target_size=image_size)
                image = tf.keras.utils.img_to_array(image)

                labels_for_image = labels_df.loc[image_filename].tolist()
                label_vector = np.zeros(len(label_map), dtype=np.float32)
                for index,label_str in enumerate(label_map):
                  if labels_for_image[index] == 1: # Assuming csv has binary 1 or 0 to label
                    label_vector[index]=1

                batch_images.append(image)
                batch_labels.append(label_vector)

            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield (batch_images, batch_labels)


    return generator()

# Example Usage
image_dir = 'data/images'
labels_csv = 'data/labels.csv'  # csv with 'filename' column and columns for labels eg 'cat', 'dog' with binary value
label_map = ['cat', 'dog', 'bird', 'fish']
batch_size = 32
image_size = (128,128)
multi_label_gen_csv = multilabel_data_generator_with_csv(image_dir, labels_csv, label_map, batch_size, image_size)
```

This refined solution uses Pandas to load label information from a CSV file, providing increased robustness in label handling. `labels.csv` is assumed to contain a 'filename' column corresponding to the image filenames, and columns for each label with a binary 0 or 1 value. The generator now retrieves the corresponding labels using the dataframe, enabling flexibility in multi-label dataset definition.

**Scenario 3: Integrating with Augmentation and Efficient Batch Loading**

While the previous examples worked, they lacked actual augmentation. We can efficiently integrate `ImageDataGenerator` for augmentation and improve generator efficiency.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def multilabel_data_generator_augmented(image_dir, labels_csv, label_map, batch_size, image_size=(256,256)):
    datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

    labels_df = pd.read_csv(labels_csv)
    labels_df.set_index('filename', inplace=True)

    image_filenames = labels_df.index.tolist()

    def generator():
        while True:
            batch_filenames = np.random.choice(image_filenames, size=batch_size, replace=False)
            batch_images = []
            batch_labels = []

            for image_filename in batch_filenames:
              image_path = os.path.join(image_dir, image_filename)
              image = tf.keras.utils.load_img(image_path, target_size=image_size)
              image = tf.keras.utils.img_to_array(image)
              batch_images.append(image)

              labels_for_image = labels_df.loc[image_filename].tolist()
              label_vector = np.zeros(len(label_map), dtype=np.float32)
              for index, label_str in enumerate(label_map):
                if labels_for_image[index] == 1:
                  label_vector[index] = 1
              batch_labels.append(label_vector)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            flow_generator = datagen.flow(batch_images, batch_labels, batch_size=batch_size)
            yield next(flow_generator)


    return generator()


# Example Usage
image_dir = 'data/images'
labels_csv = 'data/labels.csv'
label_map = ['cat', 'dog', 'bird', 'fish']
batch_size = 32
image_size = (128,128)
multi_label_gen_augmented = multilabel_data_generator_augmented(image_dir, labels_csv, label_map, batch_size, image_size)
```

This version has significantly improved efficiency. It first selects a batch of filenames. Subsequently, images and labels are loaded for this batch. Finally, `ImageDataGenerator.flow` is used to apply augmentations to images, returning an augmented image and label batch for each iteration. This prevents having to load, and then augment, each image individually. It utilizes the highly efficient Keras’ `ImageDataGenerator` to improve performance and prevent duplicate augmentation.

For further learning and understanding, I would recommend studying these resources: The official Keras documentation, which offers extensive examples for ImageDataGenerator; relevant blog posts and articles on medium that discuss data loading techniques for image classification; and advanced deep learning books covering custom data loading pipelines, such as those for dealing with unbalanced datasets. Deep learning courses, especially those focusing on practical computer vision applications, can also help solidify these concepts. These resources provide both the theoretical background and practical application needed for tackling complex image-based machine learning problems.
