---
title: "How can I combine data from two image folders into a single dataset using flow_from_directory?"
date: "2025-01-26"
id: "how-can-i-combine-data-from-two-image-folders-into-a-single-dataset-using-flowfromdirectory"
---

Combining data from two image folders into a single dataset using `flow_from_directory` in Keras requires a structured approach since the function is inherently designed to generate batches from a single root directory. My experience, especially during a past project involving multi-source medical imaging, highlighted the necessity of carefully managing this process to ensure data integrity and efficient model training. The key fact here is that `flow_from_directory` assumes a class-based folder structure within the root directory. To accommodate data from multiple folders, one must first consolidate the data while maintaining class labels, then use `flow_from_directory` on the combined structure.

The primary challenge involves ensuring that images from both source folders are correctly labeled and accessible to the Keras data generator. Specifically, `flow_from_directory` relies on the directory structure to infer labels, expecting subdirectories within the root directory to represent classes. Therefore, directly providing two independent image folders to a single `flow_from_directory` call is not feasible. Instead, I generally employ a two-step process: first, creating a unified directory structure that incorporates data from both folders; second, feeding this unified structure to `flow_from_directory`.

Here’s a breakdown of the process with illustrative examples:

**Step 1: Consolidate Data and Structure Folders**

Assume I have two folders: `source_folder_1` and `source_folder_2`.  Each folder contains subfolders representing different classes (e.g., 'cat', 'dog', or 'normal', 'abnormal'). The goal is to create a new `unified_data_folder` containing these images from both sources, with labels preserved through the folder structure. This involves programmatically copying the image files from the source folders into the correct class directories within `unified_data_folder`. The code must handle the existing class labels and transfer the files appropriately. Here's a Python code snippet demonstrating this:

```python
import os
import shutil

def consolidate_image_folders(source_folders, unified_data_folder):
    if not os.path.exists(unified_data_folder):
        os.makedirs(unified_data_folder)

    for source_folder in source_folders:
        for class_name in os.listdir(source_folder):
            class_path = os.path.join(source_folder, class_name)
            if os.path.isdir(class_path):
                unified_class_path = os.path.join(unified_data_folder, class_name)
                if not os.path.exists(unified_class_path):
                    os.makedirs(unified_class_path)
                for image_name in os.listdir(class_path):
                   image_path = os.path.join(class_path, image_name)
                   if os.path.isfile(image_path):
                       shutil.copy2(image_path, unified_class_path)

# Example usage:
source_folders = ['source_folder_1', 'source_folder_2']
unified_data_folder = 'unified_data_folder'
consolidate_image_folders(source_folders, unified_data_folder)
```

This code iterates through each source folder and its subfolders (representing classes). It creates corresponding class subfolders in the `unified_data_folder` and copies each image file while preserving metadata (using `shutil.copy2`). I've used `shutil.copy2` instead of `shutil.copy` as it maintains the original metadata of the file, which is essential for certain workflows. The `os.path.isfile` check is vital to exclude any potential directories that could be mistaken as image files.

**Step 2: Using `flow_from_directory` on the Consolidated Data**

Now that I have a unified directory structure with correctly placed images, I can readily use `flow_from_directory`. It’s important to define the batch size and target image size to fit the input requirements of the model. The class mode parameter needs to match the classification setup. For example, for binary classification, it’s set to `binary`. For multi-class classification, it should be set to `categorical` or `sparse`, depending on the label encoding. Here is a code example for binary classification:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Example parameters
image_size = (256, 256)
batch_size = 32

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = data_generator.flow_from_directory(
    unified_data_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = data_generator.flow_from_directory(
    unified_data_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
```

In this example, `ImageDataGenerator` is instantiated for preprocessing, which includes rescaling image pixel values. The `flow_from_directory` method is used twice: once for training and once for validation. The `subset` parameter separates the data for training and validation, using a 20% split specified in the `ImageDataGenerator` initialization. Crucially, I've incorporated `shuffle = True` for the training set generator to help prevent overfitting. For the validation generator I have `shuffle=False` to have consistent validation data sets every epoch.

**Step 3: Data Augmentation and Advanced Scenarios**

While the previous examples establish the basic functionality, adding data augmentation often improves model generalization. Further, there are instances where handling class imbalances is paramount. Therefore, I use another example to show these enhancements:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

image_size = (256, 256)
batch_size = 32
data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = data_generator.flow_from_directory(
    unified_data_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
validation_generator = data_generator.flow_from_directory(
    unified_data_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Class balancing (if needed):
if train_generator.class_indices: #only attempt to balance if there are classes present
    class_labels = np.concatenate([np.argmax(y, axis=1) for x, y in train_generator], axis=0)
    class_weights = compute_class_weight('balanced', classes = np.unique(class_labels), y= class_labels)
    class_weight_dict = dict(zip(np.unique(class_labels), class_weights))
else:
    class_weight_dict = None #no classes, so no class weights.
```

Here, I’ve augmented the `ImageDataGenerator` with rotation, shifts, shearing, zoom, and flips, all of which enhance the model’s robustness. I am using `categorical` class mode, assuming this is multi-class classification. Furthermore, if there is class imbalance, I calculate the class weights using `sklearn.utils.class_weight.compute_class_weight`. Note, this requires us to iterate through the training generator once to gather all class labels, before training and pass those labels in as the y-values. The `class_weight_dict` can then be passed into the training loop during `model.fit`. The `if train_generator.class_indices:` prevents errors if no classes exist in the data.

**Resource Recommendations**

For deeper understanding of Keras data preprocessing and handling, I strongly advise reviewing the official Keras API documentation for `ImageDataGenerator` and `flow_from_directory`.  Additionally, several textbooks covering deep learning with Python provide comprehensive information on data handling strategies and best practices.  I also suggest exploring introductory texts on image processing, which detail image manipulation techniques that are closely connected with data augmentation. The documentation of any python libraries mentioned in the examples, like `os`, `shutil`, or `scikit-learn` is very valuable as well. Finally, tutorials and examples covering data loading and augmentation in machine learning will further enhance your skills.
