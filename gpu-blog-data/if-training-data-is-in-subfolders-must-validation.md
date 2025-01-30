---
title: "If training data is in subfolders, must validation data also be in subfolders?"
date: "2025-01-30"
id: "if-training-data-is-in-subfolders-must-validation"
---
No, validation data does not inherently need to be organized into subfolders mirroring the structure of the training data, although doing so can streamline certain workflows and maintain consistency. The crucial factor is that both datasets, regardless of folder organization, are properly labeled and that the data loading mechanisms within the training pipeline can locate and utilize them correctly. I've encountered scenarios where differing folder structures for training and validation data required substantial modifications to data loading logic, a situation easily avoided with some upfront planning.

The core issue hinges on the separation of the training and validation sets, and how the training pipeline interprets these datasets. The training dataset is used to adjust the model's internal parameters, while the validation set is employed to assess the model's performance on unseen data during training, allowing us to detect overfitting and make informed decisions about hyperparameter tuning or architectural modifications. The organization of data within folders is a practical matter, and its relevance stems primarily from how libraries and frameworks handle data loading.

Specifically, many machine learning libraries, such as TensorFlow or PyTorch, offer data loading utilities that can operate effectively with diverse directory structures, as long as the paths to the image or feature data and their associated labels are defined correctly. These utilities often provide flexible mechanisms to map filenames to labels and even allow for specifying custom logic for loading different formats. It's not the subfolder structure itself that enforces proper dataset split; it’s the proper identification of what data belongs in training and what data belongs in validation. We might, for instance, store images in one folder, labels in another, and still achieve proper dataset partitioning.

The common practice of using mirrored subfolder structures often arises when dealing with image classification tasks where each subfolder represents a unique class. In this configuration, the image filenames can be programmatically mapped to labels based on the folder in which they reside. However, such uniformity isn't mandatory. As long as the code can ascertain which input belongs to which class, the structural layout becomes secondary to the overall logic.

Let's illustrate this with a few code examples using Python with TensorFlow to demonstrate different approaches:

**Example 1: Mirrored Subfolders for Training and Validation**

This first scenario utilizes a standard setup where training and validation data mirror each other, each containing subfolders for different classes.

```python
import tensorflow as tf
import os

# Define paths
train_dir = 'path/to/train_data'
val_dir = 'path/to/val_data'

# Create image datasets with mirrored structure
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    seed=42
)


# Print class names to confirm correct loading
print("Training Class Names:", train_ds.class_names)
print("Validation Class Names:", val_ds.class_names)


# Example of iteration
for images, labels in train_ds.take(1):
   print("Training batch shape:", images.shape, labels.shape)
for images, labels in val_ds.take(1):
   print("Validation batch shape:", images.shape, labels.shape)


```
In this example, `image_dataset_from_directory` automatically infers labels from the subfolder names. Crucially, both the `train_ds` and `val_ds` are built identically regarding subfolder organization, leveraging this structure to determine labels without any manual assignment. The important takeaway is that `image_dataset_from_directory` expects this arrangement by default.

**Example 2: Differing Subfolders, Explicit Mapping**

Here, we illustrate a case with a different folder organization. The training data retains the class-based subfolders but the validation images are located in a flat folder with a separate labels CSV file.

```python
import tensorflow as tf
import pandas as pd
import os


# Define paths
train_dir = 'path/to/train_data'
val_image_dir = 'path/to/val_data_images'
val_labels_file = 'path/to/val_labels.csv'


# Function to load validation data with explicit label mapping
def load_val_dataset(image_dir, labels_file, image_size, batch_size):
    labels_df = pd.read_csv(labels_file)
    image_paths = [os.path.join(image_dir, filename) for filename in labels_df['filename']]
    labels = labels_df['label'].values

    def process_path(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Adjust format accordingly
        image = tf.image.resize(image, image_size)
        return image


    val_images_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(process_path)
    val_labels_ds = tf.data.Dataset.from_tensor_slices(labels).map(tf.one_hot, num_classes=len(set(labels)))  # Adjust num_classes
    val_ds = tf.data.Dataset.zip((val_images_ds, val_labels_ds)).batch(batch_size)

    return val_ds


# Create training dataset using similar to prior example
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)


val_ds = load_val_dataset(val_image_dir, val_labels_file, (224,224), 32)


# Print example of validation dataset
for images, labels in val_ds.take(1):
     print("Validation batch shape:", images.shape, labels.shape)

```
In this example, the validation data's labels are read from a CSV file, requiring us to construct a custom `tf.data.Dataset` pipeline that explicitly maps image filenames to their respective labels. It demonstrates how different structures can coexist with the use of a custom data loading process. The training set is loaded using the convenience of `image_dataset_from_directory` whilst validation loading is explicitly coded.

**Example 3: No Subfolders, Explicit Loading**

Lastly, we consider a case where neither training nor validation data are organized in subfolders. All images reside in a single folder each with separate label CSV files.

```python
import tensorflow as tf
import pandas as pd
import os

# Define paths
train_image_dir = 'path/to/train_data_images'
train_labels_file = 'path/to/train_labels.csv'
val_image_dir = 'path/to/val_data_images'
val_labels_file = 'path/to/val_labels.csv'


# Function to load a dataset with explicit label mapping
def load_dataset(image_dir, labels_file, image_size, batch_size):
    labels_df = pd.read_csv(labels_file)
    image_paths = [os.path.join(image_dir, filename) for filename in labels_df['filename']]
    labels = labels_df['label'].values

    def process_path(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3) # Adjust format accordingly
        image = tf.image.resize(image, image_size)
        return image

    images_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(process_path)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels).map(tf.one_hot, num_classes=len(set(labels)))
    ds = tf.data.Dataset.zip((images_ds, labels_ds)).batch(batch_size)

    return ds


# Create the training and validation datasets
train_ds = load_dataset(train_image_dir, train_labels_file, (224,224), 32)
val_ds = load_dataset(val_image_dir, val_labels_file, (224,224), 32)

# Print example of training dataset
for images, labels in train_ds.take(1):
     print("Training batch shape:", images.shape, labels.shape)

# Print example of validation dataset
for images, labels in val_ds.take(1):
     print("Validation batch shape:", images.shape, labels.shape)

```
Here, both training and validation datasets are constructed via the custom `load_dataset` function. This function reads the image paths and labels from CSV files and creates the tensorflow dataset by manually loading each image and label using the dataframe. The point is that we do not require subfolders for either. This example exhibits that, beyond the convenience of `image_dataset_from_directory`, any data structure can be handled as long as we map the filenames to the labels manually.

In conclusion, while subfolders can simplify the workflow and provide a standard organization that works seamlessly with many utilities, they are not compulsory for validation data. The real requirement is a consistent and correct way of associating inputs with their corresponding labels, regardless of the physical storage structure. The key takeaway is to understand the data loading mechanics of your tools and align your data structure with those expectations. When your structures deviate from default assumptions, you must implement the proper loading routines as seen in the later examples.

For further understanding of data loading and manipulation in machine learning, I recommend exploring the documentation for TensorFlow Datasets and PyTorch DataLoaders. Additionally, tutorials focusing on data loading strategies and best practices, such as those found on machine learning course websites or blogs, provide additional insights. Finally, a review of best practices in project structure and data organization for machine learning can also be beneficial.
