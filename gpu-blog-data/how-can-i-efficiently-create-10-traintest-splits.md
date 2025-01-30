---
title: "How can I efficiently create 10% train/test splits of image data using ImageDataGenerator?"
date: "2025-01-30"
id: "how-can-i-efficiently-create-10-traintest-splits"
---
Splitting image data into train and test sets is crucial for developing robust machine learning models, and efficient handling of this split, especially with large datasets, is paramount. Using `ImageDataGenerator` for this purpose requires a specific understanding of its capabilities. I've frequently encountered situations in my work where improper splitting led to biased model evaluations, highlighting the importance of proper methodology. The core issue isn't the `ImageDataGenerator` itself, but rather, how its image loading and augmentation capabilities can be leveraged alongside a suitable data splitting strategy. Specifically, `ImageDataGenerator` alone doesn't directly manage the split. It requires an external mechanism for controlling which files go to each set. Here’s how it’s done.

**Explanation of Data Splitting Strategy with ImageDataGenerator**

The `ImageDataGenerator` class in Keras (or TensorFlow) is primarily designed for real-time data augmentation and batch loading, not for explicit dataset splitting. It reads image files from a directory, applies transformations, and yields batches. To achieve a 10% test split, the key is first separating file paths before feeding them into `ImageDataGenerator`. There are essentially two common ways to accomplish this: manual file path manipulation or leveraging tools like `train_test_split` from scikit-learn. My preferred method involves `train_test_split` because it provides a concise, readable, and more robust way to handle splits, especially when dealing with stratified or non-uniform distributions of data.

The process begins with acquiring a list of all image file paths in your target directory. These paths are then used to create corresponding labels if needed (for supervised learning tasks). This list of file paths and labels is then passed to `train_test_split` which, based on the specified `test_size` (0.1 in your case) and `random_state` for reproducibility, creates two sets: one for training and one for testing. These sets contain lists of file paths instead of actual image data.

Once the file paths are split, we utilize the `flow_from_dataframe` method of `ImageDataGenerator`. This method accepts a Pandas DataFrame that specifies the file paths and the corresponding class labels, which it then uses to load images. By feeding the training and testing file paths with their respective labels into separate `flow_from_dataframe` calls, we generate distinct `ImageDataGenerator` instances for train and test data.

Important note: Because we are splitting the file paths themselves, our image augmentation settings are applied separately to the two data sets. It is common to *only* apply augmentation to the training set to avoid introducing misleading variability into the test set. If using the `flow_from_directory` method, the method of splitting described above is still valid; instead of passing the dataframes in, the method expects the path to the training and test data folders. This approach is less adaptable in my experience for complex data configurations and requires a more rigid directory structure.

**Code Examples with Commentary**

Here are three code examples illustrating variations of this data splitting strategy.

**Example 1: Basic Split using `train_test_split` with Directory Structure**

This example assumes that the dataset is in a flat directory (e.g. no class-specific subdirectories) and image labels need to be derived from the file names.

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
data_dir = "path/to/your/images" # Replace with your image data folder
test_split_ratio = 0.1
random_seed = 42

# Get all file paths and create labels
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
labels = [os.path.basename(f).split("_")[0] for f in file_paths]  # Example label creation from file name, modify as needed
df = pd.DataFrame({'filename': file_paths, 'label': labels})

# Split data
train_df, test_df = train_test_split(df, test_size=test_split_ratio, random_state=random_seed, stratify=df['label'])

# ImageDataGenerators with image size specification
image_size = (224, 224) # Specify the target image size
batch_size = 32 # Specify batch size

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )


test_datagen = ImageDataGenerator(rescale=1./255) # Test data should have no augmentation


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Ensure consistent ordering during evaluation
)

print(f"Training batches: {len(train_generator)}, Testing batches: {len(test_generator)}")
```

**Commentary:** This code demonstrates a basic implementation using `train_test_split`. The labels are assumed to be available by splitting the file names, and stratified splitting is used to ensure the data set's classes are equally represented in both training and testing. `flow_from_dataframe` allows for direct loading of images based on the provided dataframe and target sizes. Notably, the training data augmentation is detailed, while the test data avoids any augmentation.

**Example 2: Split with Image Data and Different Label Handling**

This code demonstrates label handling when the label is stored in a separate CSV and a more general approach to label handling.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
data_dir = "path/to/your/images" # Replace with your image data folder
labels_csv = "path/to/labels.csv" # Replace with the path to your labels csv
test_split_ratio = 0.1
random_seed = 42
image_size = (224, 224)
batch_size = 32

# Load labels, assuming image file paths are in a column 'filename'
df = pd.read_csv(labels_csv)
df['filename'] = data_dir + "/" + df['filename']

# Split data
train_df, test_df = train_test_split(df, test_size=test_split_ratio, random_state=random_seed) # Can add stratify=df['label_column'] if required

# ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )


test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label_column',  # Replace 'label_column' with the name of the label column in your CSV
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label_column', # Replace 'label_column' with the name of the label column in your CSV
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print(f"Training batches: {len(train_generator)}, Testing batches: {len(test_generator)}")
```

**Commentary:** This variant demonstrates using a CSV file to link filenames to labels, which is frequently seen in many machine learning tasks. The code uses `pd.read_csv` to parse the provided CSV and then constructs the correct file paths for each entry. The `y_col` attribute is now parameterized to accept an arbitrary label column from the CSV.

**Example 3: Split with Direct Directory Input and Minimal Augmentation**

This example focuses on utilizing `flow_from_directory` and shows a simpler implementation, but requires more structured directory structure.

```python
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
data_dir = "path/to/your/images" # Replace with your image data folder
test_split_ratio = 0.1
random_seed = 42
image_size = (224, 224)
batch_size = 32
train_dir = "path/to/train_directory" # Path to generated train directory
test_dir = "path/to/test_directory" # Path to generated test directory


# Function to create directories and move files into it.
def create_directories_and_move_files(directory_list, file_paths):
    for label in directory_list:
        os.makedirs(os.path.join(directory_list[label], label), exist_ok=True)
    for file in file_paths:
        label = os.path.basename(file).split("_")[0] # Modify to match the correct labelling
        target_dir = os.path.join(directory_list[label], label)
        shutil.move(file, target_dir)
    
# Get all file paths and create labels
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
labels = [os.path.basename(f).split("_")[0] for f in file_paths]  # Example label creation from file name, modify as needed
df = pd.DataFrame({'filename': file_paths, 'label': labels})

# Split data
train_df, test_df = train_test_split(df, test_size=test_split_ratio, random_state=random_seed, stratify=df['label'])

train_dirs = {label:train_dir for label in train_df['label'].unique()}
test_dirs = {label:test_dir for label in test_df['label'].unique()}

# Creating the train data directories and moving the corresponding filepaths
create_directories_and_move_files(train_dirs, train_df['filename'])
create_directories_and_move_files(test_dirs, test_df['filename'])


# ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
print(f"Training batches: {len(train_generator)}, Testing batches: {len(test_generator)}")
```
**Commentary:** This example focuses on using `flow_from_directory`. This approach requires moving files into corresponding train/test folders. Although this is often not desired it does allow the user to use `flow_from_directory`. This implementation includes minimal augmentation to focus on splitting. It relies on a folder structure where each class is in its own subdirectory.

**Resource Recommendations**

To deepen your understanding of the underlying concepts, I recommend exploring the following areas through available textbooks, online tutorials, or research publications:

*   **Scikit-learn documentation:** Familiarize yourself with `train_test_split` functionality for flexible dataset splitting. Understanding options like stratification and random seed management is critical.

*   **TensorFlow/Keras documentation:**  Review the documentation for `ImageDataGenerator` and its various methods such as `flow_from_dataframe` and `flow_from_directory`. Pay specific attention to the parameters for augmentation, rescaling, and batch size.
*   **Pandas documentation:** Learn about DataFrame manipulation techniques, which are essential when working with file lists and labels extracted from CSV files.
*   **General Data Augmentation Techniques:** While the code shows examples, understanding the logic of data augmentation, such as transformations in spatial, color, or illumination spaces, is essential to choose the correct parameters when optimizing your model.

In summary, splitting data for train/test using `ImageDataGenerator` demands a clear understanding of data partitioning. Directly splitting the file paths using `train_test_split` and then creating specific generators using `flow_from_dataframe` offers a flexible and reliable method. The `flow_from_directory` is also useful provided there is a suitable folder structure. Remember that `ImageDataGenerator` itself doesn't directly split data, but it is a powerful tool when paired with a proper splitting strategy.
