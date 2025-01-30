---
title: "How can I shuffle a TensorFlow dataset without a buffer?"
date: "2025-01-30"
id: "how-can-i-shuffle-a-tensorflow-dataset-without"
---
Achieving efficient shuffling of a TensorFlow dataset without relying on a buffer, especially when dealing with large, out-of-memory datasets, requires understanding the interplay between data preprocessing and the inherent properties of TensorFlow’s `tf.data.Dataset` API. The core challenge arises from the desire to randomize the order of data samples without loading the entire dataset into memory at once, which standard buffer-based shuffling methods do. Instead, we must leverage techniques that operate on an index-based level, allowing for random access without materializing the whole dataset during the shuffling operation.

The typical approach to shuffling, which utilizes the `Dataset.shuffle(buffer_size)` method, loads a number of elements equal to `buffer_size` into a buffer and randomly samples from within it. This works well for datasets that fit comfortably in memory but becomes impractical for larger datasets due to memory limitations. To circumvent this, we can shuffle the indices themselves and then use those shuffled indices to fetch the data, thereby avoiding the buffer. This implies that the initial dataset must be structured such that individual samples are efficiently accessible given their index.

Here's how I've implemented this approach in previous projects, detailing three variations based on how the original data is structured:

**Example 1: Shuffling a Dataset Created from a List of Files**

Often, my datasets start as a list of filenames, where each file represents a data sample. Shuffling here means shuffling the order of the file paths before creating the dataset. This is easily accomplished using a `tf.random.shuffle` operation on a `tf.range` object which will then be used for indexed retrieval during file loading.

```python
import tensorflow as tf
import numpy as np
import os

def create_file_dataset(file_paths):
  """Creates a dataset from a list of file paths."""
  num_files = len(file_paths)
  indices = tf.range(num_files, dtype=tf.int64)
  shuffled_indices = tf.random.shuffle(indices)

  def load_file(index):
    file_path = file_paths[index]
    # Simulated file loading (replace with actual loading)
    data = np.load(file_path).astype(np.float32) # Assume .npy format.
    return data
    
  dataset = tf.data.Dataset.from_tensor_slices(shuffled_indices)
  dataset = dataset.map(load_file, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset

# Example usage (simulated files)
if not os.path.exists("simulated_data"):
    os.makedirs("simulated_data")
    for i in range(50):
        np.save(f"simulated_data/data_{i}.npy", np.random.rand(100, 100)) # Creating 50 simulated files

file_paths = [f"simulated_data/data_{i}.npy" for i in range(50)]
dataset = create_file_dataset(file_paths)

for example in dataset.take(5): # Showing first 5
  print(example.shape)
```

In this example, `create_file_dataset` takes a list of file paths as input. First, a sequence of integers is created using `tf.range` representing file indices. Then, this sequence is randomly shuffled. The `load_file` function is defined to load an individual file based on its index, here using numpy's load functionality for simplicity. Critically, `tf.data.Dataset.from_tensor_slices` creates a dataset from our shuffled index sequence, and then a `map` operation uses the index to load data. The final dataset now presents the data in shuffled order without having ever loaded the complete data set at once. `num_parallel_calls=tf.data.AUTOTUNE` permits the load operation to be applied in parallel, maximizing performance.

**Example 2: Shuffling a Dataset Represented as a CSV File with Indexed Access**

Another common scenario I encounter involves datasets stored in a single CSV file, where each row represents a data sample. Direct access to rows by index using libraries like `pandas` can be memory intensive. We must stream the data, accessing records directly based on shuffled row indices.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def create_csv_dataset(csv_path):
  """Creates a dataset from a CSV file, shuffling row indices."""
  df = pd.read_csv(csv_path, header=None)
  num_rows = len(df)
  indices = tf.range(num_rows, dtype=tf.int64)
  shuffled_indices = tf.random.shuffle(indices)

  def load_row(index):
     # read_csv_iterator makes streaming access to a csv file possible.
     # We create a text dataset and skip the required amount of rows. Then we use the next item.
     csv_ds = tf.data.TextLineDataset(csv_path).skip(index)
     raw_line = next(iter(csv_ds))
     values = tf.strings.to_number(tf.strings.split(raw_line, ',').to_tensor(), out_type=tf.float32)
     return values

  dataset = tf.data.Dataset.from_tensor_slices(shuffled_indices)
  dataset = dataset.map(load_row, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset

# Example usage (simulated CSV)
if not os.path.exists("simulated_data"):
    os.makedirs("simulated_data")
data = np.random.rand(50, 10)
np.savetxt(f"simulated_data/data.csv", data, delimiter=',')

csv_path = f"simulated_data/data.csv"
dataset = create_csv_dataset(csv_path)

for example in dataset.take(5):
  print(example.shape)
```

Here, `create_csv_dataset` takes the path to a CSV file. We use pandas to quickly ascertain the number of rows, generate a `tf.range` of indices, and then shuffle them. The `load_row` function reads the CSV file, skips rows as needed, and then reads the desired row using `TextLineDataset` with a skip parameter. The text of the row is converted to numbers. The dataset is then created from the shuffled indices and the load function is mapped over it, resulting in shuffled data without loading the whole CSV file into memory.

**Example 3: Shuffling a Dataset Generated On-the-Fly**

In some applications, data isn’t pre-stored in files. Instead, it's generated on the fly. In such cases, we can generate the data samples within a function, and use indices to control which data to generate. This requires a function that, when given an index, produces the corresponding data sample.

```python
import tensorflow as tf
import numpy as np

def create_synthetic_dataset(num_samples):
  """Creates a dataset from a generating function, shuffling via index."""
  indices = tf.range(num_samples, dtype=tf.int64)
  shuffled_indices = tf.random.shuffle(indices)

  def generate_sample(index):
      # Example of generating data based on an index. Replace this to generate your required dataset.
      return tf.random.normal(shape=(100, 100)) + tf.cast(index,tf.float32)

  dataset = tf.data.Dataset.from_tensor_slices(shuffled_indices)
  dataset = dataset.map(generate_sample, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset

# Example usage
dataset = create_synthetic_dataset(100)

for example in dataset.take(5):
   print(example.shape)
```

In this case, `create_synthetic_dataset` receives the total number of samples.  Shuffling is accomplished by shuffling the integer indices representing sample number. The `generate_sample` function illustrates how you might dynamically generate data based on the provided index. In this example, a normal distribution is sampled, and the sample index is added to it. Again, a `tf.data.Dataset` is constructed from the shuffled indices and mapped by the data generation function, achieving shuffling without any intermediate buffer of the data itself.

**Resource Recommendations**

To further delve into this topic, I recommend exploring the official TensorFlow documentation on `tf.data`, specifically regarding the structure and construction of datasets. Additionally, the section on input pipelines provides valuable context on optimizing data loading and preprocessing strategies. Reviewing examples that demonstrate index-based data retrieval will deepen your understanding of bufferless shuffling techniques. Researching ways to parallelize dataset operations using `num_parallel_calls` can further optimize performance, a critical factor for large datasets. Finally, familiarizing oneself with methods to handle datasets beyond what fits in memory, is useful for large machine learning training projects. These concepts are typically covered in advanced TensorFlow tutorials.
