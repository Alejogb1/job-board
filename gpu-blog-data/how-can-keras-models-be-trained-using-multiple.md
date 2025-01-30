---
title: "How can Keras models be trained using multiple NumPy files?"
date: "2025-01-30"
id: "how-can-keras-models-be-trained-using-multiple"
---
Training Keras models with data split across multiple NumPy files is a common scenario, especially when dealing with large datasets that exceed available memory. This approach leverages efficient data handling and preprocessing using NumPy while retaining Keras's model building and training capabilities. The primary challenge lies in providing Keras's training API a data stream rather than a single monolithic array. I’ve personally encountered this situation several times when training models on medical imaging data, where each scan generated a distinct NumPy array; merging everything into one giant file was impractical and computationally costly.

The solution rests on custom Keras data generators. A generator, in this context, is a Python function that yields batches of data on demand. Keras's `fit` method accepts a generator as its `x` (input data) argument, bypassing the need to load all training data into memory simultaneously. This enables efficient memory management and training scalability. Essentially, we create a class that inherits from Keras’s `tf.keras.utils.Sequence` class which provides some boilerplate for sequence data handling that works well with Keras training.

The core functionality of this generator is to iterate through our available NumPy files, load them sequentially, and yield batches of the contained data. This involves maintaining an index of files, reading the files, potentially augmenting data, splitting them into inputs and outputs (if applicable), and organizing these into batches. The design also needs to be mindful of shuffling to prevent the model from learning sequence biases from the order of the files. Additionally, the generator needs to implement `__len__` for Keras’s fit function to determine the number of batches in an epoch, and `__getitem__` to actually retrieve data batches during training.

Here’s the first code example illustrating this:

```python
import numpy as np
import tensorflow as tf
import os

class NumpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, batch_size, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        # Total batches = total number of data entries, regardless of splitting
        num_entries = 0
        for file_path in self.file_paths:
          data = np.load(file_path)
          num_entries += data.shape[0]
        return int(np.floor(num_entries / self.batch_size))

    def __getitem__(self, index):
        # This method loads and yields one batch from files.
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        x_batch_list = []
        y_batch_list = []
        # Accumulate the data needed
        for file_idx in batch_indexes:
          data = np.load(self.file_paths[file_idx])
          x_batch_list.extend(data[:, :-1])  # All but the last column is input.
          y_batch_list.extend(data[:, -1])  # Last column is output/label.

        x_batch = np.array(x_batch_list)
        y_batch = np.array(y_batch_list)
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
```

This `NumpyDataGenerator` assumes that your NumPy files contain both features and labels, with labels as the last column in the array. The initializer takes a list of file paths, the batch size, and an option for shuffling.  `__len__` counts the total items by loading each file to get size, and dividing by the batch size to find number of batches per epoch.  The `__getitem__` reads the batch sized indexes, loads the data for each index, and forms the batched arrays of inputs `x` and outputs `y`, before returning them. `on_epoch_end` shuffles the file indexes when an epoch completes to ensure varied data presentation. This example directly loads each file at each call, which can be inefficient if the same files are used often; more robust implementations use a cache system.

Let’s consider the case where input and label are split into separate files. This requires the generator to manage two lists of files, aligning the respective inputs and outputs. This is common with large image datasets that often have the corresponding masks in distinct files. The code would need adaptation to load each input and output file pair.

```python
import numpy as np
import tensorflow as tf
import os

class SeparateNumpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_file_paths, label_file_paths, batch_size, shuffle=True):
        self.input_file_paths = input_file_paths
        self.label_file_paths = label_file_paths
        assert len(self.input_file_paths) == len(self.label_file_paths), "Input and label file lists must match in length"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.input_file_paths))
        self.on_epoch_end()

    def __len__(self):
        # Total batches = total number of data entries, regardless of splitting
        num_entries = 0
        for file_path in self.input_file_paths:
          data = np.load(file_path)
          num_entries += data.shape[0]
        return int(np.floor(num_entries / self.batch_size))

    def __getitem__(self, index):
        # This method loads and yields one batch from files.
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        x_batch_list = []
        y_batch_list = []
        for file_idx in batch_indexes:
          x_data = np.load(self.input_file_paths[file_idx])
          y_data = np.load(self.label_file_paths[file_idx])
          x_batch_list.extend(x_data)
          y_batch_list.extend(y_data)

        x_batch = np.array(x_batch_list)
        y_batch = np.array(y_batch_list)
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
```

In `SeparateNumpyDataGenerator`, we now receive separate lists of paths for inputs and labels. Crucially, we ensure the number of input files is the same as the number of label files. In `__getitem__`,  we load matching `x` and `y` files from the given file indexes. The rest of the logic remains the same, but it now operates on paired sets of files instead of single files.

Finally, an important consideration is data augmentation.  The generator approach enables dynamic, on-the-fly augmentation rather than pre-processing all NumPy files. This helps minimize memory usage since augmented versions of the files are not stored in additional files. For example, we could add random rotations or translations to images within the `__getitem__` method.

```python
import numpy as np
import tensorflow as tf
import os
from scipy.ndimage import rotate

class AugmentedNumpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, batch_size, shuffle=True, augmentation_prob=0.5):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation_prob = augmentation_prob
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        num_entries = 0
        for file_path in self.file_paths:
          data = np.load(file_path)
          num_entries += data.shape[0]
        return int(np.floor(num_entries / self.batch_size))


    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        x_batch_list = []
        y_batch_list = []

        for file_idx in batch_indexes:
          data = np.load(self.file_paths[file_idx])
          x_data = data[:, :-1]
          y_data = data[:, -1]

          for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]

            if np.random.random() < self.augmentation_prob:
                # This will only work for 2D data.
                angle = np.random.uniform(-20, 20)
                x = rotate(x.reshape(int(np.sqrt(x.size)), int(np.sqrt(x.size))), angle, reshape=False).flatten() # Assuming images are square.
            x_batch_list.append(x)
            y_batch_list.append(y)


        x_batch = np.array(x_batch_list)
        y_batch = np.array(y_batch_list)
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
```
Here, `AugmentedNumpyDataGenerator` adds a `augmentation_prob` argument to control how often augmentation is applied.  Inside `__getitem__`, each element is potentially augmented via a rotation with `scipy.ndimage.rotate` (assuming input data represents 2D images). The augmentation needs to be tailored to the specific dataset and task.

For further exploration of data loading and processing, consider consulting official TensorFlow documentation regarding the `tf.data` API which provides efficient mechanisms for data pipelines, including file-based access and augmentation.  Also, the Keras documentation provides thorough information on data generators and the `Sequence` class. Libraries like Scikit-image offer a wide range of image processing and augmentation techniques. Understanding how data is handled in Python (e.g., generators, iterators, list comprehensions) also greatly assists in optimizing these implementations.
