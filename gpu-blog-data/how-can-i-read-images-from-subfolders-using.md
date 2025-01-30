---
title: "How can I read images from subfolders using Keras' `flow_from_dataframe`?"
date: "2025-01-30"
id: "how-can-i-read-images-from-subfolders-using"
---
Data pipelines for deep learning, especially when working with images, often require flexibility in how data is structured on disk.  Keras' `ImageDataGenerator` and its associated `flow_from_dataframe` method, while powerful, can sometimes be less intuitive to use when images reside in subfolders within a parent directory.  My experience building several image classification models has shown me that directly utilizing `flow_from_dataframe` to pull images from subfolders requires a specific data frame structure and awareness of how the method interprets file paths.

The primary challenge arises because `flow_from_dataframe` expects a DataFrame with at least two columns: one containing the file paths relative to a specified directory and another containing class labels, typically represented as strings or integers. It does not inherently "understand" a nested directory structure as labels, requiring instead a flattened representation. The subfolder structure is a convenience for organization, but the model does not understand it without proper mapping of the structure.

To successfully read images from subfolders, I have found it necessary to prepare the DataFrame to meet `flow_from_dataframe`'s expectations. Specifically, we must enumerate the image paths within the subfolders and create a corresponding column with the subfolder name as the label, then supply the parent folder to the function. This involves programmatic file system traversal and data frame construction using a suitable library like `pandas`. I've used several approaches to create such DataFrames depending on my context, sometimes from existing metadata and other times building them from scratch.

Here are three code examples, each addressing a slightly different scenario, with detailed comments:

**Example 1:  Simple Subfolder Classification (Manual DataFrame Creation)**

This scenario is for a classification problem where each subfolder directly represents a class. Here’s how I usually start with a minimal dataset:

```python
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory structure:
# parent_folder/
#  class_a/
#    image1.jpg
#    image2.jpg
#  class_b/
#    image3.jpg
#    image4.jpg

def create_dataframe_from_subfolders(parent_folder):
    """
    Constructs a DataFrame suitable for flow_from_dataframe from subfolders.

    Args:
        parent_folder (str): The path to the parent folder containing class subfolders.

    Returns:
        pandas.DataFrame: A DataFrame with 'filename' and 'label' columns.
    """
    data = []
    for class_name in os.listdir(parent_folder):
        class_path = os.path.join(parent_folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # filter image types
                    full_path = os.path.join(class_name, filename) # relative path
                    data.append({'filename': full_path, 'label': class_name})

    return pd.DataFrame(data)


if __name__ == '__main__':
    parent_directory = "parent_folder" # replace with actual path
    df = create_dataframe_from_subfolders(parent_directory)

    # Image augmentation parameters
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Create the image iterator
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=parent_directory, # provide the parent folder
        x_col="filename",
        y_col="label",
        target_size=(224, 224), # Adjust to model's expected input
        class_mode='categorical',
        batch_size=32,
        subset="training"
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=parent_directory,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        subset="validation"
    )
    # Example of reading a few batches (would continue for training)
    train_batch_x, train_batch_y = next(train_generator)
    print(f"Train batch X shape: {train_batch_x.shape}")
    print(f"Train batch Y shape: {train_batch_y.shape}")
```

This code defines a function, `create_dataframe_from_subfolders`, which traverses the directory structure, storing relative file paths and class names in a list of dictionaries. It then builds a Pandas DataFrame from that list. When using `flow_from_dataframe`, it's important to provide the `parent_directory` parameter, and in the DataFrame, the 'filename' column should be the relative paths *within* that parent directory, not the absolute path on the filesystem. This avoids issues with incorrect file loading during training.

**Example 2:  Multi-class Classification with Pre-existing Metadata**

In some scenarios, you might already have a CSV or similar data file that maps images to labels. Here's how I manage this scenario, assuming my CSV file contains the absolute path to the image and its corresponding label:

```python
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example CSV structure (absolute paths):
# file_path,label
# /path/to/parent_folder/class_a/image1.jpg,class_a
# /path/to/parent_folder/class_b/image2.jpg,class_b
# /path/to/parent_folder/class_a/image3.jpg,class_a


def transform_dataframe_for_flow(df, parent_folder):
    """
    Transforms a DataFrame with absolute paths to work with flow_from_dataframe.

    Args:
        df (pandas.DataFrame): DataFrame with 'file_path' and 'label' columns.
        parent_folder (str): The parent folder used as the base path.

    Returns:
        pandas.DataFrame: A DataFrame with 'filename' and 'label' columns suitable for flow_from_dataframe.
    """
    df['filename'] = df['file_path'].str.replace(parent_folder + "/", "", regex=False)
    return df[['filename', 'label']]


if __name__ == '__main__':
    metadata_file = "image_metadata.csv" # replace with path to csv
    parent_directory = "/path/to/parent_folder" # replace with actual path

    metadata_df = pd.read_csv(metadata_file)
    transformed_df = transform_dataframe_for_flow(metadata_df, parent_directory)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=transformed_df,
        directory=parent_directory,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        subset="training"
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=transformed_df,
        directory=parent_directory,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        subset="validation"
    )

    # Example usage
    train_batch_x, train_batch_y = next(train_generator)
    print(f"Train batch X shape: {train_batch_x.shape}")
    print(f"Train batch Y shape: {train_batch_y.shape}")

```
Here,  the key function `transform_dataframe_for_flow` extracts the relative path needed by `flow_from_dataframe` from the provided absolute paths. Providing the correct parent path and using relative paths in the dataframe ensures that `flow_from_dataframe` finds the images correctly.

**Example 3:  Regression (No Subfolders as Class Labels)**

In a regression task, the subfolders themselves do not represent classes. Labels must come from a metadata file and associated with filenames. Let’s suppose the metadata file contains the filename (no path) and the regression target:

```python
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Example metadata.csv structure:
# filename,regression_target
# image1.jpg,1.23
# image2.jpg,4.56
# image3.jpg,7.89


def create_dataframe_for_regression(parent_folder, metadata_file):
    """
    Creates a DataFrame for regression tasks using metadata file

    Args:
        parent_folder: Path to the parent folder containing image subfolders
        metadata_file: Path to the CSV metadata file.

    Returns:
        pandas.DataFrame: DataFrame with image filenames relative to parent folder and labels
    """
    metadata_df = pd.read_csv(metadata_file)
    file_paths = []
    for subdir in os.listdir(parent_folder):
       subdir_path = os.path.join(parent_folder, subdir)
       if os.path.isdir(subdir_path):
         for filename in os.listdir(subdir_path):
           if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
             file_paths.append(os.path.join(subdir, filename))

    # Combine with metadata on filename
    df = pd.DataFrame({'filename':file_paths})
    df = df.merge(metadata_df, on="filename")
    return df


if __name__ == '__main__':
    parent_directory = "parent_folder" # replace with actual path
    metadata_file = "metadata.csv" # replace with actual path to the metadata
    df = create_dataframe_for_regression(parent_directory, metadata_file)


    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=parent_directory,
        x_col="filename",
        y_col="regression_target",
        target_size=(224, 224),
        class_mode='raw', # Use 'raw' for regression
        batch_size=32,
        subset="training"
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=parent_directory,
        x_col="filename",
        y_col="regression_target",
        target_size=(224, 224),
        class_mode='raw',
        batch_size=32,
        subset="validation"
    )

     # Example usage
    train_batch_x, train_batch_y = next(train_generator)
    print(f"Train batch X shape: {train_batch_x.shape}")
    print(f"Train batch Y shape: {train_batch_y.shape}")
```

In this case, `create_dataframe_for_regression` constructs the DataFrame ensuring that the metadata is correctly matched with image paths, using 'raw' class mode for regression, not relying on subfolders as classes. This code joins the relative image paths with the regression targets from the metadata file, which is critical for this kind of problem.

For further study, I recommend exploring these resources (no URLs) : the Pandas documentation, the Keras documentation for ImageDataGenerator, and general Python file system and path manipulation tutorials. Mastering these concepts allows more efficient data pipelines when working with image data in deep learning.
