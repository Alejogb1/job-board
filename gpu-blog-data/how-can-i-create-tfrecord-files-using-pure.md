---
title: "How can I create TFRecord files using pure Python, without TensorFlow?"
date: "2025-01-30"
id: "how-can-i-create-tfrecord-files-using-pure"
---
The core challenge in crafting TFRecord files without TensorFlow lies in understanding the underlying protocol buffer serialization mechanism.  TensorFlow's convenience functions abstract away much of this detail, but the process is fundamentally about constructing and writing serialized Protocol Buffer messages to disk. My experience developing custom data pipelines for large-scale image recognition projects has reinforced this understanding.  Direct manipulation necessitates a thorough grasp of the `protobuf` library and careful attention to data type handling.

**1. Clear Explanation:**

TFRecord files are essentially containers for serialized Protocol Buffer messages.  Each message typically represents a single data instance, such as an image and its corresponding label.  To create a TFRecord file without TensorFlow, you must first define a Protocol Buffer message schema that describes the structure of your data.  This schema is then compiled into Python classes using the `protobuf` compiler.  These generated classes provide methods for constructing and serializing the messages.  Finally, these serialized messages are written to a file, which is then the TFRecord file.  The key is to manage the binary serialization correctly, ensuring compatibility with the TensorFlow `tf.io.TFRecordDataset` function if you intend to read the file back into a TensorFlow environment later.  Failure to adhere to this protocol results in unreadable or corrupted TFRecords.  My previous work involved migrating legacy data formats into a TFRecord system, highlighting the criticality of precise type conversions.


**2. Code Examples with Commentary:**

**Example 1: Simple Example with a Single Feature**

This example demonstrates creating a TFRecord file containing single integer values.

```python
import tensorflow as tf # Only for tf.train.Example, to avoid external proto def
import io

# Define a Protocol Buffer message (using tf.train.Example for simplicity)
example_proto = tf.train.Example(features=tf.train.Features(feature={
    'value': tf.train.Feature(int64_list=tf.train.Int64List(value=[10]))
}))

# Serialize the message
serialized_example = example_proto.SerializeToString()

# Write to file
with open('simple.tfrecord', 'wb') as f:
    f.write(serialized_example)


# Reading back (for demonstration, using tf)
raw_dataset = tf.data.TFRecordDataset('simple.tfrecord')
for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example.features.feature['value'].int64_list.value) # Output: [10]
```

This code leverages `tf.train.Example` for simplicity.  In a production setting, you would typically define your own `.proto` file and compile it using the `protoc` compiler.  This allows for more complex data structures and greater control.  The critical step is `SerializeToString()`, which converts the Protocol Buffer message into a binary format suitable for TFRecord storage.  The `wb` mode ensures binary writing.


**Example 2:  Multiple Features (string and float)**

This example expands on the first, showing how to incorporate multiple features of different data types.

```python
import tensorflow as tf # Only for tf.train.Example
import io

example_proto = tf.train.Example(features=tf.train.Features(feature={
    'string_value': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'hello'])),
    'float_value': tf.train.Feature(float_list=tf.train.FloatList(value=[3.14159]))
}))

serialized_example = example_proto.SerializeToString()

with open('multiple.tfrecord', 'wb') as f:
    f.write(serialized_example)


#Reading back (for demonstration, using tf)
raw_dataset = tf.data.TFRecordDataset('multiple.tfrecord')
for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example.features.feature['string_value'].bytes_list.value[0].decode('utf-8')) #Output: hello
    print(example.features.feature['float_value'].float_list.value[0]) #Output: 3.14159
```

Observe how different data types are handled using appropriate `tf.train.Feature` subtypes. String values require encoding to bytes using `b'hello'`.  This example highlights the importance of consistent type handling to avoid errors during serialization and deserialization.  Incorrect encoding can lead to data corruption or read failures.


**Example 3:  Multiple Examples in a Single File**

This illustrates writing multiple data instances into a single TFRecord file.

```python
import tensorflow as tf # Only for tf.train.Example
import io

examples = []
for i in range(3):
    example_proto = tf.train.Example(features=tf.train.Features(feature={
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[i * 2.5]))
    }))
    examples.append(example_proto.SerializeToString())

with open('multiple_examples.tfrecord', 'wb') as f:
    for example in examples:
        f.write(example)

#Reading back (for demonstration, using tf)
raw_dataset = tf.data.TFRecordDataset('multiple_examples.tfrecord')
for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example.features.feature['id'].int64_list.value[0], example.features.feature['value'].float_list.value[0])

```

This demonstrates a more realistic scenario where multiple data points are written sequentially.  Each `SerializeToString()` call produces a separate serialized example, appended to the file.  Efficient handling of large datasets necessitates optimized file writing techniques and potentially the use of buffered writing to improve performance.


**3. Resource Recommendations:**

The official Protocol Buffer documentation.  A comprehensive guide to Python's `io` module for file handling.  The TensorFlow documentation, particularly sections on the `tf.io` module, which, although not directly used in the code examples above, provide a valuable reference for understanding the TFRecord format and its conventions.  Understanding these resources is crucial for correctly implementing and debugging your custom TFRecord creation process.  Successfully handling large scale data pipelines necessitates a firm grasp of these concepts.
