---
title: "How can I convert a NumPy memmap of non-image numeric data to TFRecord format for training?"
date: "2025-01-30"
id: "how-can-i-convert-a-numpy-memmap-of"
---
Direct memory-mapped access to large datasets, specifically with NumPy's `memmap`, provides a performant alternative to loading entire datasets into RAM. Converting these `memmap` arrays of numeric data, rather than image data, directly into TFRecord format can significantly optimize TensorFlow training pipelines by enabling efficient streaming from disk without full dataset materialization.

I’ve encountered this challenge frequently during large-scale time-series analysis where datasets exceeded available RAM. The bottleneck wasn’t data processing in TensorFlow, but rather the inefficient loading mechanism from disk. The key is understanding how `memmap` enables direct byte-level access to disk files and translating that into the binary format required by TFRecords.

A TFRecord file stores data as a sequence of binary records, each record typically consisting of a single training example. These records are themselves serialized TensorFlow `tf.train.Example` protocol buffers. The process involves iterating through your `memmap` array, constructing a corresponding `tf.train.Example` for each row (or a logical chunk of your data), and then writing these serialized protocol buffers to a TFRecord file.

Here's a detailed breakdown of the process, incorporating my practical experience with this method.

**1. Defining Feature Descriptions:**

The first step is to specify the data structure using `tf.io.FixedLenFeature`. This informs TensorFlow how to interpret the data in each record. In my experience, this is often the most crucial step for ensuring data compatibility down the line. Given the numerical nature of your data, you will usually use `tf.float32`, `tf.int64`, or similar depending on data types. Assuming your memmap contains rows of floating-point numbers with a fixed number of columns, you can define the feature description as a dictionary specifying the data type and shape. For example, for a memmap with 10 columns of float32 data:

```python
import numpy as np
import tensorflow as tf

def create_feature_description(num_columns):
    feature_description = {
        'data': tf.io.FixedLenFeature([num_columns], tf.float32)
    }
    return feature_description
```

This ensures TensorFlow understands how many values to expect and their data type within each serialized record.

**2.  Creating `tf.train.Example` Protocol Buffers:**

Next, you need a function to convert each row of your `memmap` into a `tf.train.Example` message. This involves wrapping the NumPy array row within a TensorFlow `tf.train.BytesList`. The important aspect here is encoding the NumPy array to a raw byte string before inclusion in `tf.train.BytesList`. This ensures data fidelity during the serialization process. I’ve found that the explicit byte encoding step often fixes subtle issues with differing data types during deserialization.

```python
def create_example(row):
    feature = {
        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row.tobytes()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
```

Notice the use of `row.tobytes()`. This method explicitly converts the NumPy array row into a byte string, which is the only type accepted by the `tf.train.BytesList`. This raw byte representation is essential for preserving data integrity.

**3. Writing Data to TFRecord:**

The final step involves reading your memmap, processing each row, and writing the corresponding `tf.train.Example` to the TFRecord file. I recommend using `tf.io.TFRecordWriter` to handle the lower-level file handling. The key performance optimization here is the direct access provided by memmaps. The data isn’t loaded into RAM during this process.

```python
def write_memmap_to_tfrecord(memmap_path, tfrecord_path, num_columns):
    mm = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(np.memmap(memmap_path, dtype=np.float32, mode='r').shape[0],num_columns))
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for row in mm:
            example = create_example(row)
            writer.write(example.SerializeToString())
    del mm
```
The `del mm` line is critical as it closes the memory map file. Failure to close it can cause issues with file locking, and other processes may fail to access it.

**4. Reading Data from TFRecord (For Verification):**

To verify your TFRecord, a simple reading function can be implemented. This demonstrates the reverse operation and shows that the data stored in TFRecord can be parsed into the original format.  When debugging issues, I found it extremely beneficial to immediately test that I can correctly load data from the TFRecord after creation, which avoids chasing issues in the training loop.

```python
def read_tfrecord(tfrecord_path, num_columns):
    feature_description = create_feature_description(num_columns)
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        parsed_example['data'] = tf.io.decode_raw(parsed_example['data'], out_type=tf.float32)
        return parsed_example

    parsed_dataset = dataset.map(_parse_function)
    for example in parsed_dataset:
      print(example['data'])
      break
```

This example demonstrates parsing the binary bytes read from the TFRecord using `tf.io.decode_raw` and reshaping the resulting tensor to its original form.

**Code Example (Comprehensive):**

Combining the previous snippets into a coherent example:
```python
import numpy as np
import tensorflow as tf

def create_feature_description(num_columns):
    feature_description = {
        'data': tf.io.FixedLenFeature([num_columns], tf.float32)
    }
    return feature_description

def create_example(row):
    feature = {
        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row.tobytes()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_memmap_to_tfrecord(memmap_path, tfrecord_path, num_columns):
    mm = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(np.memmap(memmap_path, dtype=np.float32, mode='r').shape[0],num_columns))
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for row in mm:
            example = create_example(row)
            writer.write(example.SerializeToString())
    del mm

def read_tfrecord(tfrecord_path, num_columns):
    feature_description = create_feature_description(num_columns)
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        parsed_example['data'] = tf.io.decode_raw(parsed_example['data'], out_type=tf.float32)
        return parsed_example

    parsed_dataset = dataset.map(_parse_function)
    for example in parsed_dataset:
      print(example['data'])
      break

# Example usage:
num_rows = 1000
num_columns = 10
memmap_file = "my_data.dat"
tfrecord_file = "my_data.tfrecord"

# Create a dummy memmap file
dummy_data = np.random.rand(num_rows,num_columns).astype(np.float32)
mm_dummy = np.memmap(memmap_file, dtype=np.float32, mode='w+', shape=(num_rows, num_columns))
mm_dummy[:] = dummy_data[:]
del mm_dummy

# Convert from memmap to TFRecord
write_memmap_to_tfrecord(memmap_file, tfrecord_file, num_columns)

# Read TFRecord to verify functionality
read_tfrecord(tfrecord_file, num_columns)
```

**Resource Recommendations:**

To further explore this topic, focus on these specific areas. Firstly, consult the official TensorFlow documentation detailing the `tf.data` API. Pay close attention to the functionality related to TFRecordDataset. Understanding the specifics of `tf.io.FixedLenFeature` and the broader capabilities of protocol buffers is also crucial. Secondly, thoroughly review the NumPy documentation, specifically regarding memory mapping with `np.memmap` and data type conversions (e.g., `tobytes`). Lastly, explore general guides and tutorials focused on optimizing TensorFlow data pipelines, specifically focusing on techniques for large datasets and efficient streaming, as these best practices will be directly applicable to the approach described here.
