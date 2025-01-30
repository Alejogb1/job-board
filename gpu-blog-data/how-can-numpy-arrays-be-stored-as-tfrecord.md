---
title: "How can NumPy arrays be stored as TFRecord files?"
date: "2025-01-30"
id: "how-can-numpy-arrays-be-stored-as-tfrecord"
---
The efficient storage and retrieval of numerical data, particularly when integrated with TensorFlow workflows, often necessitates the use of TFRecord files. NumPy arrays, a fundamental data structure in scientific computing, can be seamlessly serialized and stored within TFRecord files, leveraging the protocol buffer mechanism at its core. I have implemented this pattern extensively in large-scale machine learning pipelines, and I’ve refined an approach that avoids common pitfalls regarding memory management and data type compatibility.

First, the fundamental challenge is converting the multi-dimensional NumPy array into a serializable byte string before TFRecord storage. The inherent structure of a NumPy array, including its data type, dimensions, and the actual numerical values, needs to be preserved during serialization and reconstructed accurately during retrieval. Directly writing the raw memory representation of the array can be problematic due to endianness and data type inconsistencies across different systems or environments. I’ve found a more robust approach involves encoding the NumPy array and its metadata using TensorFlow's `tf.train.Example` protocol buffer message.

A `tf.train.Example` is a key-value container where values are represented as `tf.train.Feature` instances. We can store a NumPy array as a single `bytes_list` feature, provided we first convert the NumPy array into a byte string. The steps involve: (1) recording the array's shape to reconstruct the dimensions upon retrieval, (2) converting the NumPy array to a byte string using its `tobytes()` method, and (3) storing the shape as an `int64_list` feature and the byte representation as a `bytes_list` feature within the `tf.train.Example`. Then we serialize the `tf.train.Example` message and write it into a TFRecord file using a `tf.io.TFRecordWriter`.

Retrieval reverses this process. We use a `tf.data.TFRecordDataset` to read from the TFRecord file. Inside of a `map` operation we will parse individual `tf.train.Example` messages using the inverse of the schema we established during writing, using `tf.io.parse_single_example`. We then reconstruct the NumPy array using `np.frombuffer` and reshape it with the shape stored earlier. This process ensures that the data is accurately represented after serialization and deserialization. I’ve noticed that proper schema definition is critical; failure to match the read/write schema leads to errors that are often difficult to debug.

Below are three code examples demonstrating this. The first shows how to serialize a single NumPy array and write it to a TFRecord file. The second demonstrates reading it back, and the third illustrates batch storage and retrieval.

**Example 1: Writing a Single NumPy Array to TFRecord**

```python
import numpy as np
import tensorflow as tf

def serialize_numpy_array(array):
    """Serializes a NumPy array into a tf.train.Example."""
    array_shape = array.shape
    array_bytes = array.tobytes()
    feature = {
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=array_shape)),
        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_bytes])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_numpy_to_tfrecord(numpy_array, filename):
    """Writes a NumPy array to a TFRecord file."""
    serialized_array = serialize_numpy_array(numpy_array)
    with tf.io.TFRecordWriter(filename) as writer:
        writer.write(serialized_array)

if __name__ == '__main__':
    test_array = np.random.rand(10, 10, 3).astype(np.float32)
    write_numpy_to_tfrecord(test_array, 'test.tfrecord')
    print("Array saved to test.tfrecord")
```

In this example, `serialize_numpy_array` takes a NumPy array, gets its shape and converts the array into a byte string. We then create `tf.train.Example` using a dictionary defining each field's feature type; then the whole message is serialized for storage. The `write_numpy_to_tfrecord` function handles the actual writing to the specified TFRecord file. I’ve repeatedly relied on this function as a basis for the more complex serialization procedures.

**Example 2: Reading a Single NumPy Array from TFRecord**

```python
import numpy as np
import tensorflow as tf

def parse_single_example(serialized_example):
    """Parses a serialized tf.train.Example back to a NumPy array."""
    feature_description = {
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    shape = example['shape'].numpy()
    data = example['data'].numpy()
    array = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return array

def read_numpy_from_tfrecord(filename):
    """Reads a NumPy array from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = dataset.map(parse_single_example)
    return next(iter(parsed_dataset))

if __name__ == '__main__':
    restored_array = read_numpy_from_tfrecord('test.tfrecord')
    print("Restored array shape:", restored_array.shape)
    print("Restored array dtype:", restored_array.dtype)
    # For testing, print comparison of small slices
    original_test_array = np.random.rand(10, 10, 3).astype(np.float32)
    write_numpy_to_tfrecord(original_test_array, 'test.tfrecord')
    restored_array = read_numpy_from_tfrecord('test.tfrecord')
    print("Are first 3x3 slices equal:", np.allclose(original_test_array[:3, :3, :],restored_array[:3, :3, :]))
```

This example shows `parse_single_example` which defines the expected structure using `FixedLenFeature`, retrieves the serialized data, reconstructs the NumPy array and returns it. The function `read_numpy_from_tfrecord` creates a TFRecord dataset and uses the parse function to transform the messages to NumPy arrays. I often employ the iterator method to access the data, which is generally efficient for small data sets or for debugging. The final part of the code illustrates how to reproduce the example and verify the restoration is correct.

**Example 3: Writing and Reading Batches of NumPy Arrays**

```python
import numpy as np
import tensorflow as tf

def serialize_batch_numpy_array(arrays):
  """Serializes a batch of NumPy arrays into a list of tf.train.Example."""
  serialized_examples = []
  for array in arrays:
      array_shape = array.shape
      array_bytes = array.tobytes()
      feature = {
          'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=array_shape)),
          'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_bytes])),
      }
      example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
      serialized_examples.append(example_proto.SerializeToString())
  return serialized_examples

def write_batch_numpy_to_tfrecord(numpy_arrays, filename):
    """Writes a batch of NumPy arrays to a TFRecord file."""
    serialized_arrays = serialize_batch_numpy_array(numpy_arrays)
    with tf.io.TFRecordWriter(filename) as writer:
        for serialized_array in serialized_arrays:
            writer.write(serialized_array)

def parse_batch_example(serialized_example):
    """Parses a serialized tf.train.Example back to a NumPy array."""
    feature_description = {
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    shape = example['shape'].numpy()
    data = example['data'].numpy()
    array = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return array


def read_batch_numpy_from_tfrecord(filename):
    """Reads a batch of NumPy arrays from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = dataset.map(parse_batch_example)
    return list(parsed_dataset.as_numpy_iterator())


if __name__ == '__main__':
    batch_size = 5
    test_arrays = [np.random.rand(10, 10, 3).astype(np.float32) for _ in range(batch_size)]
    write_batch_numpy_to_tfrecord(test_arrays, 'test_batch.tfrecord')
    restored_arrays = read_batch_numpy_from_tfrecord('test_batch.tfrecord')
    print(f"Restored {len(restored_arrays)} arrays of shape {restored_arrays[0].shape}")
    print("Are first slices equal:", np.allclose(test_arrays[0][:3,:3,:], restored_arrays[0][:3,:3,:]))
    print("Are second slices equal:", np.allclose(test_arrays[1][:3,:3,:], restored_arrays[1][:3,:3,:]))
```
This example shows how to write and read multiple NumPy arrays using very similar techniques as the first example, except the data is batched for both reading and writing. The writing is done one by one in a loop in `write_batch_numpy_to_tfrecord`. And the reading is done via a `list` which makes use of a `numpy_iterator`. This approach is useful when storing several arrays separately.

In my experience, careful consideration of data types, shapes and the serialization/deserialization process are crucial for successful data handling in production environments.  When dealing with different data types (e.g., integers, doubles), adapt the parsing step to correctly reconstruct the appropriate NumPy array type using `dtype`. Remember to consistently update both serialization and deserialization code for any schema changes.

For more detailed explanations and alternative approaches to data serialization with TensorFlow, consult the official TensorFlow documentation, especially the sections on `tf.data` and `tf.train.Example`. Additionally, explore the NumPy documentation regarding the use of `tobytes` and `frombuffer`, as these are frequently involved in the process. Finally, exploring open-source projects on GitHub that use TensorFlow datasets and TFRecord will often provide pragmatic approaches and best practices.
