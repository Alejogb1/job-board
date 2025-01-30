---
title: "How can I read TFRecords containing lists of variable-length data using the TensorFlow Dataset API?"
date: "2025-01-30"
id: "how-can-i-read-tfrecords-containing-lists-of"
---
TFRecords often encode sequences of varying lengths, a common challenge when dealing with time-series or text data. The key to effectively handling these variable-length sequences within the TensorFlow Dataset API lies in understanding how to properly parse the serialized data and subsequently construct a usable dataset. The standard `tf.io.parse_single_example` function, while useful for fixed-length features, proves insufficient for nested structures like lists. I've encountered this frequently in my work with recurrent models and have refined a process that reliably addresses this.

**Understanding the Problem**

The core challenge originates from the way TFRecords serialize data. A single record represents a serialized dictionary of `tf.train.Feature` objects. Each feature holds a value that can be one of three types: bytes lists (`bytes_list`), float lists (`float_list`), or int64 lists (`int64_list`). When you have variable-length sequences, they're typically stored as byte strings which require additional parsing on the receiving end to retrieve their original data. Standard functions like `tf.io.parse_single_example` expect a fixed shape for the feature values, creating difficulties when attempting to process records that, for instance, may contain different numbers of word embeddings. Consequently, we must employ a more tailored approach, parsing the byte strings and reconstructing the nested data structures within them.

**The Solution: Custom Parsing**

The recommended approach involves two main steps: First, we define feature specifications that denote our serialized data, and second, we create a parsing function that takes the serialized record as input and converts it into a structured dictionary compatible with the TensorFlow Dataset API.

**Step 1: Feature Specification**

The feature specification defines how our data is organized in the TFRecord. For variable-length lists, we must use `tf.io.FixedLenFeature([], dtype=tf.string)`, which will read the raw bytes of the list. I typically define this specification as a dictionary:

```python
import tensorflow as tf

def create_feature_description():
  """Defines the feature specification for our TFRecord data."""
  feature_description = {
      'id': tf.io.FixedLenFeature([], tf.int64),
      'sequence': tf.io.FixedLenFeature([], tf.string),
  }
  return feature_description
```

In this example, I assume the TFRecord contains an integer identifier and a serialized string representing our variable-length list. The key here is defining the 'sequence' feature using `tf.io.FixedLenFeature([], tf.string)`, which indicates the data is a single, variable-length byte string. Note that even if our list is composed of integers or floating point numbers, it should be encoded as bytes for variable length storage within a `tf.train.Example`.

**Step 2: The Parsing Function**

The most crucial part is the parsing function. It takes the serialized record and the feature specification, applies the parsing using `tf.io.parse_single_example`, and then further processes the raw byte strings to reconstitute the variable-length data.

```python
def parse_example(serialized_example, feature_description, sequence_dtype):
  """Parses the TFRecord and handles the variable-length sequence."""
  parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

  # Decode the byte string, assuming a specific data type.
  sequence_bytes = parsed_example['sequence']

  # This part is where the logic to decode the specific sequence is added.
  # This approach works best if you've pre-serialized your sequence data.
  # Here we assume the list was stored as a series of tf.float32 values.
  sequence = tf.io.decode_raw(sequence_bytes, sequence_dtype)

  # Optionally, specify a known shape after decoding.
  # We don't know the shape but if you're dealing with a fixed
  # maximum, you could reshape here.
  # sequence = tf.reshape(sequence, [-1,embedding_dimension])

  parsed_example['sequence'] = sequence

  return parsed_example
```

The `parse_example` function first applies `tf.io.parse_single_example`, using the feature description we created. Then, it extracts the raw byte string for the 'sequence' feature. We then decode the bytes into a Tensor of the appropriate data type. In this specific example, I'm assuming it represents a list of `tf.float32` values. In a real-world situation, the decoding step would depend on how the lists were serialized. It's common to store lists as raw bytes of data, sometimes using `struct.pack` in python. For example, if your original data was a python `list` of `float`, you might have converted it to a bytestring with `struct.pack('f'*len(my_list), *my_list)`, and similarly when reading the bytes you'd use `struct.unpack('f'*len(my_bytes)//4, my_bytes)` after reading the raw bytes.

**Code Example 1: A Simple Integer List**

Let's consider a situation where each sequence is a variable-length list of integers. We'll demonstrate the entire process from TFRecord creation to parsing within a dataset.

```python
import numpy as np
import struct

def create_tfrecord_with_int_sequences(filename):
  """Creates a TFRecord containing lists of integers."""
  with tf.io.TFRecordWriter(filename) as writer:
      for i in range(3):
          sequence = np.random.randint(0, 10, size=np.random.randint(1, 5)) # variable-length integer list
          serialized_sequence = struct.pack('i'*len(sequence), *sequence)
          example = tf.train.Example(features=tf.train.Features(feature={
              'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
              'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_sequence])),
          }))
          writer.write(example.SerializeToString())

  return filename

def read_tfrecord_with_int_sequences(filename):
  feature_description = create_feature_description()

  def parsing_func(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
      sequence_bytes = parsed_example['sequence']
      sequence = tf.io.decode_raw(sequence_bytes, tf.int32)
      parsed_example['sequence'] = sequence
      return parsed_example

  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(parsing_func)

  return dataset

temp_filename = 'test.tfrecord'
create_tfrecord_with_int_sequences(temp_filename)
dataset = read_tfrecord_with_int_sequences(temp_filename)

for example in dataset.take(3):
    print(f"ID: {example['id'].numpy()}, Sequence: {example['sequence'].numpy()}")
```
In this example, we generated example sequences with varying numbers of integers, packed them into byte strings, and stored them in the TFRecord. We then demonstrate reading these records, decoding the integers. Notice that we still perform the parsing on the bytestring, just as we did in the parsing example before.

**Code Example 2: Variable Length Float Embeddings**

Let's examine a more practical example involving variable length lists of float vectors. I often encounter this when working with NLP models where each sequence is variable number of embedding vectors.
```python
import numpy as np
import struct

def create_tfrecord_with_float_embeddings(filename, embedding_dimension):
  """Creates a TFRecord containing lists of float embeddings."""
  with tf.io.TFRecordWriter(filename) as writer:
      for i in range(3):
          num_vectors = np.random.randint(1, 5)
          sequence = np.random.rand(num_vectors, embedding_dimension).astype(np.float32)
          serialized_sequence = b"".join([struct.pack('f'*embedding_dimension, *vec) for vec in sequence])
          example = tf.train.Example(features=tf.train.Features(feature={
              'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
              'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_sequence])),
          }))
          writer.write(example.SerializeToString())

  return filename

def read_tfrecord_with_float_embeddings(filename, embedding_dimension):
  feature_description = create_feature_description()
  def parsing_func(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
      sequence_bytes = parsed_example['sequence']
      sequence_length = tf.size(sequence_bytes) // (4 * embedding_dimension)
      sequence = tf.io.decode_raw(sequence_bytes, tf.float32)
      sequence = tf.reshape(sequence, [sequence_length, embedding_dimension])
      parsed_example['sequence'] = sequence
      return parsed_example

  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(parsing_func)
  return dataset

embedding_dimension = 16
temp_filename = 'embeddings.tfrecord'
create_tfrecord_with_float_embeddings(temp_filename, embedding_dimension)
dataset = read_tfrecord_with_float_embeddings(temp_filename, embedding_dimension)

for example in dataset.take(3):
    print(f"ID: {example['id'].numpy()}, Sequence Shape: {example['sequence'].shape.as_list()}")
```
Here we demonstrate how each sequence is itself a list of embeddings. We first convert each embedding vector to a bytestring using `struct.pack`, and then concatenate these bytes into a bytestring representing the entire sequence. When decoding we can infer the number of vectors given the length of the bytestring and the embedding dimension.

**Code Example 3: Nested Structure**

To further illustrate, we can consider a nested structure, where a sequence consists of multiple sub-sequences each with its own variable length. This demonstrates the power of working with nested bytestrings.
```python
import numpy as np
import struct

def create_tfrecord_with_nested_sequences(filename):
  """Creates a TFRecord containing nested lists of integers."""
  with tf.io.TFRecordWriter(filename) as writer:
      for i in range(3):
          num_sub_sequences = np.random.randint(1, 4)
          serialized_sub_sequences = b""
          for _ in range(num_sub_sequences):
              sub_sequence = np.random.randint(0, 10, size=np.random.randint(1, 5)) # variable-length integer list
              serialized_sub_sequence = struct.pack('i'*len(sub_sequence), *sub_sequence)
              serialized_sub_sequences += serialized_sub_sequence
          example = tf.train.Example(features=tf.train.Features(feature={
              'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
              'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_sub_sequences])),
          }))
          writer.write(example.SerializeToString())
  return filename

def read_tfrecord_with_nested_sequences(filename):
    feature_description = create_feature_description()

    def parsing_func(serialized_example):
        parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
        sequence_bytes = parsed_example['sequence']
        
        # Assuming we know we are encoding integers, decode the bytes of the entire sequence
        sequence = tf.io.decode_raw(sequence_bytes, tf.int32)
        
        # Now, you need logic to split the integers into sub-sequences if needed.
        # For demonstration, this just shows the integers
        parsed_example['sequence'] = sequence
        return parsed_example

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parsing_func)

    return dataset
temp_filename = 'nested_sequences.tfrecord'
create_tfrecord_with_nested_sequences(temp_filename)
dataset = read_tfrecord_with_nested_sequences(temp_filename)

for example in dataset.take(3):
    print(f"ID: {example['id'].numpy()}, Sequence: {example['sequence'].numpy()}")
```

This example extends the method to handling arbitrarily nested structures. Note that the complexity here isn't in reading the raw bytes, it is in correctly extracting the correct number of elements given the raw bytestring. In reality the structure could be defined, and more complicated parsing operations performed in the `parsing_func`.

**Resource Recommendations**

To further enhance your understanding and practical skills with TFRecords and the Dataset API, I suggest reviewing the official TensorFlow documentation on `tf.data.Dataset` and `tf.io`.  Explore tutorials on creating and reading TFRecords with varying data types. Pay close attention to examples involving `tf.io.decode_raw`, as this is central to the process of converting byte strings to usable tensors. Consider looking into libraries that may have been created to simplify this process.

In summary, working with variable-length lists within TFRecords necessitates understanding how to utilize the `tf.io.FixedLenFeature` for byte strings, followed by careful parsing using `tf.io.decode_raw` and reshaping.  The examples above, derived from my experiences with a variety of projects, provide a solid basis for adapting this technique to your specific use cases. Remember to always encode your variable-length data as raw bytes before storing within your TFRecord and then carefully decode them on the read side.
