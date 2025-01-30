---
title: "How can TensorFlow image augmentation ensure each class has the same number of images?"
date: "2025-01-30"
id: "how-can-tensorflow-image-augmentation-ensure-each-class"
---
A common challenge in image classification tasks arises when datasets exhibit class imbalances, where certain categories possess significantly more training samples than others. Directly training a model on such imbalanced data often leads to biased performance, favoring the overrepresented classes. While various techniques exist to address this problem, using TensorFlow's image augmentation capabilities within a targeted sampling strategy can effectively equalize per-class sample counts during training. My experience working on a defect detection system for industrial components highlighted this issue firsthand. Our initial dataset comprised hundreds of images of 'normal' parts, but only a few dozen representing specific types of defects. Without intervention, the model consistently struggled to accurately classify defects.

The core principle for equalization using TensorFlow is to dynamically augment images from underrepresented classes during training. Instead of simply applying augmentations randomly to the whole dataset, we focus augmentation on classes with fewer samples. This means we modify the sampling process within the training loop to ensure that, in expectation, each batch contains an equal proportion of examples from each class, or a predefined ratio based on desired augmentation factor for individual classes. This approach is more memory-efficient and adaptive than pre-augmenting the entire dataset, especially when dealing with large datasets. We avoid creating static copies of augmented images, and only generate them on-the-fly as required.

To implement this in TensorFlow, you need to manage the dataset loading, augmentation, and batching in a controlled manner. The key is to maintain class-specific indices or iterators, then sample from those, applying augmentations when sampling from underrepresented groups. This can involve custom data loading functions that provide batch-sized examples, along with selective augmentation based on the target class.

Here are three code examples illustrating different approaches, with commentary:

**Example 1: Basic Custom Data Generator with Class-Specific Sampling**

This example presents a simplified generator to illustrate the core sampling logic. It relies on in-memory lists of images and their labels. This approach works well with datasets small enough to fit in memory but is not suitable for handling very large datasets.

```python
import tensorflow as tf
import numpy as np

def custom_data_generator(images, labels, batch_size, augmentations_per_class, class_names):
    """
    A simple generator for balanced class sampling with augmentation.

    Args:
        images (list): A list of file paths for images.
        labels (list): A list of corresponding labels.
        batch_size (int): Size of each batch.
        augmentations_per_class (dict): Augmentation factor for each class.
        class_names (list): Names of all classes.

    Yields:
        tuple: Batch of images and their corresponding labels
    """
    class_indices = {c: np.where(np.array(labels) == c)[0] for c in class_names}
    num_classes = len(class_names)
    min_samples = min([len(idx) for idx in class_indices.values()])
    class_iters = {c: iter(idx) for c, idx in class_indices.items()}

    while True:
        batch_images = []
        batch_labels = []
        for c in class_names:
            for _ in range(batch_size // num_classes): # Fixed amount of samples for each class per batch
                try:
                     index = next(class_iters[c])
                except StopIteration:
                     class_iters[c] = iter(class_indices[c]) # Reset iterator
                     index = next(class_iters[c])


                image_path = images[index]
                image = tf.io.read_file(image_path)
                image = tf.io.decode_jpeg(image, channels=3)
                image = tf.image.convert_image_dtype(image, tf.float32)


                if augmentations_per_class[c] > 1:
                        for _ in range(augmentations_per_class[c]-1):
                                augmented_image = tf.image.random_flip_left_right(image)
                                augmented_image = tf.image.random_brightness(augmented_image,max_delta=0.2)
                                batch_images.append(augmented_image)
                                batch_labels.append(c)
                batch_images.append(image)
                batch_labels.append(c)

        batch_images = tf.stack(batch_images)
        batch_labels = tf.convert_to_tensor(batch_labels)
        yield batch_images, batch_labels


# Example Usage (assuming you have lists 'images_list', 'labels_list'):
#  class_names = list(set(labels_list))
#  augment_per_class= { class_name : 2 if labels_list.count(class_name) < min(list(map(labels_list.count,set(labels_list))))*2 else 1  for class_name in class_names }
#  train_dataset= tf.data.Dataset.from_generator(lambda: custom_data_generator(images_list,labels_list,batch_size,augment_per_class, class_names ),
#                                              output_signature=(tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
#                                                                tf.TensorSpec(shape=(None,), dtype=tf.int32)))

```

*   **Commentary:** The `custom_data_generator` iterates through each class, using class iterators, and loads images based on their indices. Augmentation is applied based on a dictionary where each class has a augmentation factor. If the iterator is exhausted, it's reset, allowing for indefinite iteration. The batch size is divided by number of classes to ensure equal representation, with the rest being added by the augmentations.

**Example 2: TensorFlow Dataset API with Imbalanced Dataset Transformation**

This example leverages the TensorFlow Dataset API for loading images from file paths and applies an imbalanced data transformation at the dataset level.

```python
import tensorflow as tf
import numpy as np

def load_and_preprocess_image(image_path, label):
    """ Loads, decodes, and preprocesses a single image. """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment_image(image, label, augment_factor):
        images = [image]
        for _ in range(augment_factor-1):
           augmented_image = tf.image.random_flip_left_right(image)
           augmented_image = tf.image.random_brightness(augmented_image,max_delta=0.2)
           images.append(augmented_image)
        return tf.stack(images) , tf.repeat(label, augment_factor)


def create_balanced_dataset(images_list, labels_list, batch_size,augmentations_per_class , class_names):

    """Creates a TensorFlow dataset with class-balanced sampling."""
    dataset = tf.data.Dataset.from_tensor_slices((images_list,labels_list ))
    dataset = dataset.map(load_and_preprocess_image)

    def augment_wrapper(image, label):
         augment_factor = augmentations_per_class[tf.get_static_value(label).decode('utf-8')]
         images,labels=augment_image(image,label,augment_factor)
         return tf.data.Dataset.from_tensor_slices((images,labels))

    dataset = dataset.flat_map(augment_wrapper)

    dataset = dataset.shuffle(buffer_size=len(images_list)*2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Example Usage (assuming you have lists 'images_list', 'labels_list'):
#  class_names = list(set(labels_list))
#  augment_per_class= { class_name : 2 if labels_list.count(class_name) < min(list(map(labels_list.count,set(labels_list))))*2 else 1  for class_name in class_names }
#  train_dataset = create_balanced_dataset(images_list, labels_list, batch_size,augment_per_class , class_names)
```

*   **Commentary:** The `create_balanced_dataset` function sets up the dataset pipeline. The `augment_image` function does selective augmentations based on the augmentation factor of the class.  `flat_map` flattens the dataset and is followed by shuffling, batching, and prefetching, ensuring efficient data loading. This approach moves the core logic into the dataset's map operations.

**Example 3:  Class Weighted Loss**

While not strictly an augmentation method, using class-weighted loss functions is a powerful complementary technique when dealing with class imbalance. Below is a way to use this function alongside the previous code.

```python
import tensorflow as tf
from sklearn.utils import class_weight

def weighted_categorical_crossentropy(y_true, y_pred, weights):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(weights))
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * tf.gather(weights, tf.argmax(y_true, axis=1))
    return tf.reduce_mean(loss)


def calculate_class_weights(labels):
  """Calculates class weights using sklearn.utils.class_weight"""
  class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
  return dict(zip(np.unique(labels),class_weights))

# Example Usage (assuming you have lists 'images_list', 'labels_list'):
#  class_names = list(set(labels_list))
#  augment_per_class= { class_name : 2 if labels_list.count(class_name) < min(list(map(labels_list.count,set(labels_list))))*2 else 1  for class_name in class_names }
#  train_dataset = create_balanced_dataset(images_list, labels_list, batch_size,augment_per_class , class_names)
#  class_weights = calculate_class_weights(labels_list)

#  model.compile(optimizer='adam',
#              loss= lambda y_true,y_pred:weighted_categorical_crossentropy(y_true,y_pred,tf.convert_to_tensor(list(class_weights.values()))),
#              metrics=['accuracy'])
```

*   **Commentary:** This function uses `sklearn.utils.class_weight` to compute class weights and the `weighted_categorical_crossentropy` to incorporate these weights into the loss calculations. This penalizes misclassifications of under-represented classes more, effectively balancing their contribution during training. This is used alongside the augmentation methods above, providing a comprehensive approach to handling class imbalance.

**Resource Recommendations:**

For deeper understanding of the topics covered, I would recommend exploring the TensorFlow documentation. The sections covering the `tf.data` API, image manipulation functions within `tf.image`, and custom training loops will be particularly useful. Additionally, research papers on data augmentation techniques for image classification provide insightful perspectives on selecting appropriate transformations. Textbooks on machine learning and deep learning often include sections on addressing class imbalance, which might give a broader overview of the topic. For specific problems, the scikit-learn library also provides very good tools, especially `sklearn.utils.class_weight`. Experimentation and validation are key to adapting the techniques to any specific dataset.
