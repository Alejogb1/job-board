---
title: "How can validation sets be implemented using `BatchDataset`?"
date: "2025-01-30"
id: "how-can-validation-sets-be-implemented-using-batchdataset"
---
A critical challenge when training machine learning models lies in accurately gauging their ability to generalize to unseen data, thereby preventing overfitting. Using a `BatchDataset` alongside a validation set is crucial for this purpose. The fundamental concept revolves around splitting available data into distinct training and validation portions. The training data informs the model's learning, while the validation set provides an unbiased estimate of the model's performance on new data, effectively enabling early stopping and hyperparameter tuning.

Implementing a validation set with a `BatchDataset` typically involves first preparing your data and then leveraging the functionalities offered by the framework, often TensorFlow or PyTorch, to split it into two `BatchDataset` instances. The process often benefits from data shuffling prior to splitting to ensure representative samples are present in both datasets. We use these separated datasets to train and monitor the performance concurrently, allowing us to make informed decisions during model optimization.

The first step I've found beneficial is consistently using a configurable random seed. This practice guarantees reproducibility in data splitting. Consider the following scenario: I have a large dataset stored as a set of CSV files, each containing numeric feature data and a corresponding label. I must create a validation set using TensorFlow's `tf.data.Dataset` to create the batch and facilitate further steps.

```python
import tensorflow as tf
import numpy as np

def create_dataset(filepaths, batch_size, shuffle_buffer_size, seed=42, validation_split=0.2):
    """
    Creates training and validation datasets from filepaths.

    Args:
        filepaths: List of filepaths containing data.
        batch_size: Integer, the batch size.
        shuffle_buffer_size: Integer, size of shuffle buffer.
        seed: Integer, random seed for shuffling.
        validation_split: Float, percentage of data to be used for validation.

    Returns:
        train_dataset, validation_dataset: Batched training and validation datasets.
    """

    all_data = []
    all_labels = []

    for filepath in filepaths:
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        all_data.append(data[:, :-1]) #Assuming last column is the label
        all_labels.append(data[:, -1:])
    
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)

    dataset_size = len(all_data)
    validation_size = int(dataset_size * validation_split)

    data_indices = np.arange(dataset_size)
    np.random.seed(seed)
    np.random.shuffle(data_indices)

    validation_indices = data_indices[:validation_size]
    train_indices = data_indices[validation_size:]
    
    validation_data = all_data[validation_indices]
    validation_labels = all_labels[validation_indices]
    train_data = all_data[train_indices]
    train_labels = all_labels[train_indices]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(shuffle_buffer_size, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset


# Example usage:
file_paths = ['data_file_0.csv', 'data_file_1.csv', 'data_file_2.csv']  
batch_size = 32
shuffle_buffer_size = 1000
train_dataset, validation_dataset = create_dataset(file_paths, batch_size, shuffle_buffer_size)
```

In the above example, the `create_dataset` function first loads data from the specified files, assuming comma-separated values and that the last column represents the label. The data are then concatenated. The function subsequently calculates split indices and creates distinct training and validation datasets by slicing based on these indices.  Shuffling the data using a set seed beforehand is essential for creating non-biased data sets. Using `from_tensor_slices` creates a dataset from the numpy arrays, which is then shuffled, batched and prefetched for increased training speed. Note that the validation dataset is only batched and prefetched to reduce computational overhead. The function returns two batched `tf.data.Dataset` objects, allowing for efficient iterative access to data.

While the previous approach works well with data loaded into memory, working with large datasets may require on-demand loading. In this scenario we must leverage TensorFlow’s functionalities to read and create a dataset of the files themselves. This approach is common in image processing applications where datasets are often larger than available memory.

```python
import tensorflow as tf
import os

def create_image_dataset(image_dir, batch_size, image_size, shuffle_buffer_size, validation_split=0.2, seed=42):
    """
    Creates training and validation datasets from an image directory.

    Args:
        image_dir: String, path to directory containing images in subfolders by class.
        batch_size: Integer, batch size.
        image_size: Tuple, image dimension in (height, width).
        shuffle_buffer_size: Integer, the size of shuffle buffer.
        validation_split: Float, percentage of data to be used for validation.
        seed: Integer, random seed for shuffling.

    Returns:
        train_dataset, validation_dataset: Batched training and validation datasets.
    """
   
    all_files = tf.data.Dataset.list_files(os.path.join(image_dir, '*/*'), shuffle=True, seed=seed)
    dataset_size = tf.data.experimental.cardinality(all_files).numpy()
    validation_size = int(dataset_size * validation_split)
    
    validation_files = all_files.take(validation_size)
    train_files = all_files.skip(validation_size)


    def process_image(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Adjust as needed for PNG, etc.
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0 # Normalize pixel values
        label = tf.strings.split(file_path, os.path.sep).to_tensor()[-2] # Class names are subfolder names
        label = tf.strings.to_number(label, out_type=tf.int32) # Assumes class subfolders are labeled with a numerical ID
        return image, label

    train_dataset = train_files.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_dataset = validation_files.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset

# Example usage:
image_dir = './images'
batch_size = 32
image_size = (128, 128)
shuffle_buffer_size = 1000
train_dataset, validation_dataset = create_image_dataset(image_dir, batch_size, image_size, shuffle_buffer_size)
```

In this image processing scenario, the function uses `tf.data.Dataset.list_files` to create a dataset of all image files and then splits it into training and validation portions. The `process_image` function is then used to read and decode images on demand, while it also uses the file structure to get class labels. This approach is better suited to very large image datasets, as it reduces the amount of memory required at any given time. Again the datasets are properly shuffled, batched and prefetched for optimal performance.

While the above approaches are sufficient for many use cases, consider a more complex situation where the data loading and preprocessing is computationally intensive. In this case it is important to decouple the data loading from the actual training pipeline, using a custom `tf.data.Dataset` subclass, allowing for maximum control over data reading and processing, potentially leveraging parallel processing at various stages.

```python
import tensorflow as tf
import os
import numpy as np

class CustomDataset(tf.data.Dataset):
    def __init__(self, filepaths, batch_size, shuffle_buffer_size, transform=None, validation_split=0.2, seed=42):
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.transform = transform
        self.validation_split = validation_split
        self.seed = seed
        self.dataset_size = 0 
        self.data, self.labels = self._load_data()


    def _load_data(self):
        """
        Load and preprocess data from filepaths.

        Returns:
            data, labels: all data and labels as a numpy array
        """

        all_data = []
        all_labels = []

        for filepath in self.filepaths:
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            all_data.append(data[:, :-1])
            all_labels.append(data[:, -1:])

        all_data = np.concatenate(all_data)
        all_labels = np.concatenate(all_labels)

        self.dataset_size = len(all_data)

        return all_data, all_labels

    
    def _generate(self, indices):
        """
        Generates data tuples from the given indices using data and labels

        Args:
            indices: integer array of indices to use

        Returns:
            Tuple: numpy array of the data and label
        """
        data = self.data[indices]
        labels = self.labels[indices]
        
        if self.transform is not None:
             data = self.transform(data)
        
        return data, labels


    def get_dataset(self):
        """
        Creates training and validation datasets

        Returns:
            train_dataset, validation_dataset: Batched training and validation datasets.
        """
        validation_size = int(self.dataset_size * self.validation_split)
        
        data_indices = np.arange(self.dataset_size)
        np.random.seed(self.seed)
        np.random.shuffle(data_indices)

        validation_indices = data_indices[:validation_size]
        train_indices = data_indices[validation_size:]

        train_dataset = tf.data.Dataset.from_tensor_slices(train_indices)
        train_dataset = train_dataset.map(lambda i : tf.py_function(self._generate,[i], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.shuffle_buffer_size, seed=self.seed).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_tensor_slices(validation_indices)
        validation_dataset = validation_dataset.map(lambda i : tf.py_function(self._generate,[i], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    
        return train_dataset, validation_dataset
    


# Example usage:
file_paths = ['data_file_0.csv', 'data_file_1.csv', 'data_file_2.csv']
batch_size = 32
shuffle_buffer_size = 1000
dataset = CustomDataset(file_paths, batch_size, shuffle_buffer_size)
train_dataset, validation_dataset = dataset.get_dataset()

```

This `CustomDataset` subclass demonstrates the usage of a custom class that implements the `tf.data.Dataset` interface, encapsulating both data loading and preprocessing. The `_generate` function in this example utilizes `tf.py_function` which allows for arbitrary python code to be used within a TensorFlow graph, allowing for a wide range of custom functions to be used in the data processing pipeline. It has similar functionality to the first example but uses an object-oriented paradigm to maintain state, which is useful for more complex setups. The `get_dataset` function uses a similar approach to create the train and validation datasets.

For further exploration on data processing with TensorFlow, examine the official TensorFlow documentation for `tf.data`. Also, research advanced data loading strategies like `tf.data.experimental.make_csv_dataset`. For PyTorch equivalents, refer to the `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` classes, in addition to their tutorials on data loading for specific use cases, such as image datasets. Furthermore, a deep understanding of data loading and preprocessing paradigms will help in dealing with complex data sources.
