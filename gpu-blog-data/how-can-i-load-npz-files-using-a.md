---
title: "How can I load .npz files using a Keras custom data generator?"
date: "2025-01-30"
id: "how-can-i-load-npz-files-using-a"
---
Compressed NumPy archives, denoted by the `.npz` extension, present a particular challenge when integrated with Keras’ data generation framework because they inherently bundle multiple arrays within a single file, necessitating a preprocessing step to extract and yield the data correctly. This deviates from typical image-based scenarios where a file usually maps directly to a single input. My experience training custom models for spectral analysis applications has frequently involved handling pre-computed data stored in `.npz` format, making this a common workflow for my projects. I'll describe how to create a Keras data generator that appropriately handles this.

The core issue lies in the fact that a typical Keras generator, built using `tf.keras.utils.Sequence`, expects each yield to represent a single batch of data—usually a tuple of input features and target variables. However, `.npz` files often contain multiple named arrays, not necessarily organized in a way that aligns directly with training inputs and outputs. Therefore, the key process is to load the `.npz` file, extract the appropriate arrays, and then structure them into the batches expected by the Keras model. This is achieved by subclassing `tf.keras.utils.Sequence` and overriding its methods. The `__len__` method specifies the number of batches per epoch, and `__getitem__` dictates how each batch is generated, forming the heart of the data loading logic.

The first step is to establish a structure to manage the file paths and specify which arrays from the `.npz` files will be used as inputs and which as targets, if applicable. This can be accomplished with a constructor in our custom generator class that accepts lists of filepaths, along with strings denoting the names of the input and target arrays inside those files. The crucial section is the `__getitem__` method where NumPy's `np.load` is called on the appropriate file, followed by retrieval of specified arrays by name. These arrays are then combined to form the training batch. This entire process needs to be done inside this generator, with no changes made to the training loop. Let me provide an example:

```python
import numpy as np
import tensorflow as tf

class NPZSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, input_array_names, target_array_names=None, batch_size=32):
        self.file_paths = file_paths
        self.input_array_names = input_array_names
        self.target_array_names = target_array_names
        self.batch_size = batch_size
        self.num_samples = len(file_paths)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)
        batch_paths = self.file_paths[start:end]

        batch_inputs = []
        batch_targets = [] if self.target_array_names else None
        for path in batch_paths:
            loaded_data = np.load(path)
            # Handle multiple input arrays
            current_inputs = [loaded_data[name] for name in self.input_array_names]
            # Convert single input case into a list
            if len(current_inputs) == 1:
              current_inputs = current_inputs[0]
            batch_inputs.append(current_inputs)

            if self.target_array_names:
                current_targets = [loaded_data[name] for name in self.target_array_names]
                if len(current_targets) == 1:
                   current_targets = current_targets[0]
                batch_targets.append(current_targets)


        if len(batch_inputs) > 1 and isinstance(batch_inputs[0], list):
            batch_inputs = [np.stack(inp_list) for inp_list in zip(*batch_inputs)]
        else:
            batch_inputs = np.stack(batch_inputs)
        if self.target_array_names:
           if len(batch_targets) > 1 and isinstance(batch_targets[0], list):
               batch_targets = [np.stack(tar_list) for tar_list in zip(*batch_targets)]
           else:
               batch_targets = np.stack(batch_targets)
           return batch_inputs, batch_targets
        return batch_inputs
```

In this first example, I've implemented a flexible generator that can accommodate multiple inputs as well as single or multiple outputs. Inside the `__init__` method, the necessary paths to the files, the input array names, output array names, and batch size are taken as inputs and stored as members. The `__len__` method returns the number of batches per epoch given the batch size and number of files. `__getitem__` is where the `.npz` file is loaded using `np.load`, the requested arrays are extracted using list comprehensions, and then the arrays are batched using the function `np.stack`. A check is also included to see if the inputs and outputs are a list of arrays, and the output is restructured if that is the case. If target array names are provided, then the function returns the input and output arrays, otherwise only the input arrays.

This is a common structure for handling multiple arrays from an `.npz` file, but there can be situations where you'll need more complex transformations. For example, you might want to combine or preprocess input and target arrays from each `.npz` file before they are yielded to the model. Here is an example demonstrating this.

```python
import numpy as np
import tensorflow as tf

class PreprocessedNPZSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, input_array_names, target_array_names, batch_size=32, transformation_fn=None):
        self.file_paths = file_paths
        self.input_array_names = input_array_names
        self.target_array_names = target_array_names
        self.batch_size = batch_size
        self.num_samples = len(file_paths)
        self.transformation_fn = transformation_fn

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)
        batch_paths = self.file_paths[start:end]

        batch_inputs = []
        batch_targets = []

        for path in batch_paths:
            loaded_data = np.load(path)
            inputs = [loaded_data[name] for name in self.input_array_names]
            targets = [loaded_data[name] for name in self.target_array_names]
            if self.transformation_fn:
                transformed_input, transformed_target = self.transformation_fn(inputs, targets)
                batch_inputs.append(transformed_input)
                batch_targets.append(transformed_target)
            else:
                batch_inputs.append(np.concatenate(inputs, axis=-1))
                batch_targets.append(np.concatenate(targets, axis=-1))

        batch_inputs = np.stack(batch_inputs)
        batch_targets = np.stack(batch_targets)

        return batch_inputs, batch_targets
```

Here, I've added a `transformation_fn` argument to the constructor. This enables us to execute an arbitrary transformation on the data extracted from each `.npz` file. By default, if no function is provided, the extracted arrays are concatenated along the last axis. This example shows how a preprocessing function can be injected into our data generator, letting us use custom transformations on the data before it is passed to the model.

In cases where not all the data for a single batch comes from a single `.npz` file, further customization is needed. Let's imagine the case where batches are derived from sequences of data stored in `.npz` files: each file contains a segment of a longer sequence, and we want to combine segments from different files into a single batch.

```python
import numpy as np
import tensorflow as tf

class SequenceNPZSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, input_array_name, target_array_name, batch_size=32, sequence_length=10):
        self.file_paths = file_paths
        self.input_array_name = input_array_name
        self.target_array_name = target_array_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.data_indices = self._generate_data_indices()
        self.num_samples = len(self.data_indices)

    def _generate_data_indices(self):
        data_indices = []
        for file_idx, path in enumerate(self.file_paths):
            loaded_data = np.load(path)
            input_data = loaded_data[self.input_array_name]
            num_sequences = max(0, input_data.shape[0] - self.sequence_length + 1)
            for sequence_idx in range(num_sequences):
                data_indices.append((file_idx, sequence_idx))
        return data_indices

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)
        batch_indices = self.data_indices[start:end]

        batch_inputs = []
        batch_targets = []

        for file_idx, seq_idx in batch_indices:
          loaded_data = np.load(self.file_paths[file_idx])
          inputs = loaded_data[self.input_array_name][seq_idx: seq_idx+self.sequence_length]
          targets = loaded_data[self.target_array_name][seq_idx: seq_idx+self.sequence_length]
          batch_inputs.append(inputs)
          batch_targets.append(targets)

        batch_inputs = np.stack(batch_inputs)
        batch_targets = np.stack(batch_targets)
        return batch_inputs, batch_targets
```

Here, the data indices are generated to store both file locations and sequence starting positions of the data, effectively creating a flattened list of the sequence beginnings. The generator now iterates through this flattened list, extracting sequence segments from files and generating the input and output batch. This addresses a specific scenario where batch creation involves sequences across files, making the generator more adaptable.

For further exploration, I highly recommend consulting the official TensorFlow documentation for `tf.keras.utils.Sequence`, which explains the lifecycle and requirements of a data generator object in detail. Also, reviewing examples in books on deep learning applications that handle time series data will be valuable for understanding the generation of sequence data, since this often involves data stored in formats other than standard images, such as numpy arrays. Furthermore, practical exploration of NumPy's array manipulation functions, particularly `np.load`, `np.stack` and `np.concatenate` will greatly increase your familiarity with handling data from `.npz` files. Finally, examining real-world data generation pipelines, as presented in open source deep learning projects can show you advanced techniques for handling diverse data formats. These will offer a solid foundational grasp for creating more advanced, efficient, and robust data loading strategies.
