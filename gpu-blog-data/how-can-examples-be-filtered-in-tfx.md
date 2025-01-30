---
title: "How can examples be filtered in TFX?"
date: "2025-01-30"
id: "how-can-examples-be-filtered-in-tfx"
---
In TensorFlow Extended (TFX), efficient and targeted example filtering is crucial for various stages of the machine learning pipeline, from data validation to model training and evaluation. TFX utilizes Apache Beam for its distributed data processing, and therefore leverages Beam's robust filtering capabilities. The core mechanism involves applying a Beam `Filter` transform, typically using a lambda function or a callable object defined using the `tf.train.Example` protocol buffer's field names as inputs, allowing for flexible and performant conditional selection of examples.

The foundational element is the `beam.Filter` transform, which, when applied to a PCollection of `tf.train.Example` protocol buffers, returns a new PCollection containing only the examples that satisfy the specified filtering criteria. This filtering can be based on the values of features within the `Example`, presence or absence of features, or derived values computed during the filtering process. The filtering operations are carried out in a distributed manner by Apache Beam’s runners (e.g., DirectRunner, DataflowRunner), ensuring scalability for large datasets. In practice, I've frequently encountered scenarios requiring filters based on data quality checks, targeted training subsets, and specific experimental control conditions during modeling iterations.

Let's explore this with concrete examples.

**Example 1: Filtering Based on a Numerical Feature**

Imagine I'm working on a dataset for house price prediction. One of the features, let's say `square_footage`, is crucial. However, I observed that some entries with unusually low `square_footage` values are likely data entry errors. I need to exclude these entries. Here’s how I would implement such a filter:

```python
import apache_beam as beam
import tensorflow as tf
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils.dsl_utils import external_input

# Dummy function to simulate the previous TFX component output.
def _create_example_gen_output(path_to_data):
  output = standard_artifacts.Examples()
  output.uri = path_to_data
  output.split_names = artifact_utils.encode_split_names(['train'])
  return output

def _create_dummy_data(data_path):
  # Simulate example data in tfrecords.
  writer = tf.io.TFRecordWriter(data_path + '/data.tfrecord')
  for i in range(100):
      example = tf.train.Example(features=tf.train.Features(feature={
        'square_footage': tf.train.Feature(int64_list=tf.train.Int64List(value=[i * 10 + 100])),
        'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        'price': tf.train.Feature(float_list=tf.train.FloatList(value=[100000.0 + i*1000.0]))
      }))
      writer.write(example.SerializeToString())
  writer.close()

def filter_examples_by_numerical_feature(examples_input):
    """Filters examples based on the 'square_footage' feature."""

    filtered_examples = (
        examples_input
        | 'ReadTfRecords' >> beam.io.ReadFromTFRecord(file_pattern=examples_input.uri + '/*')
        | 'ParseTfExample' >> beam.Map(tf.io.parse_single_example,
                                        features={
                                             'square_footage': tf.io.FixedLenFeature([], dtype=tf.int64),
                                             'bedrooms': tf.io.FixedLenFeature([], dtype=tf.int64),
                                             'price': tf.io.FixedLenFeature([], dtype=tf.float32)
                                         })
        | 'FilterLowSquareFootage' >> beam.Filter(lambda example: example['square_footage'] > 200)
        | 'SerializeToExample' >> beam.Map(lambda example: tf.train.Example(features=tf.train.Features(feature={
          'square_footage': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['square_footage']])),
          'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['bedrooms']])),
          'price': tf.train.Feature(float_list=tf.train.FloatList(value=[example['price']]))
        })).SerializeToString())
    )
    return filtered_examples

if __name__ == '__main__':
  import tempfile
  temp_dir = tempfile.mkdtemp()
  data_path = temp_dir + '/my_examples'
  _create_dummy_data(data_path)
  example_gen_output = _create_example_gen_output(data_path)
  with beam.Pipeline() as pipeline:
    filtered_data = filter_examples_by_numerical_feature(example_gen_output)
    # Write filtered data
    filtered_data | 'WriteFilteredTfRecords' >> beam.io.WriteToTFRecord(
        file_path_prefix=temp_dir + '/filtered_examples/data',
        file_name_suffix='.tfrecord')
  print("Filtered data written to ", temp_dir + '/filtered_examples')

```
In this code: First, a dummy directory with synthetic data is created. The function `filter_examples_by_numerical_feature` performs the filtering. We read the tfrecords, parse the example to be able to access the features. We then employ `beam.Filter` with a lambda function: `lambda example: example['square_footage'] > 200`. This lambda receives parsed examples as input dictionaries. It selects only examples where the `square_footage` is strictly greater than 200. The filtered examples are then written back to a new directory.

**Example 2: Filtering Based on a String Feature**

Consider another scenario.  I'm working with a dataset that contains a `city` feature, a string indicating the city each house belongs to. I need to filter out data for a particular city, “Springfield,” perhaps for regional model training or specific experimentation. Here’s the implementation:

```python
import apache_beam as beam
import tensorflow as tf
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils.dsl_utils import external_input

# Dummy function to simulate the previous TFX component output.
def _create_example_gen_output(path_to_data):
  output = standard_artifacts.Examples()
  output.uri = path_to_data
  output.split_names = artifact_utils.encode_split_names(['train'])
  return output

def _create_dummy_data(data_path):
  # Simulate example data in tfrecords.
  writer = tf.io.TFRecordWriter(data_path + '/data.tfrecord')
  cities = ['Springfield', 'Shelbyville', 'Capital City']
  for i in range(100):
      example = tf.train.Example(features=tf.train.Features(feature={
        'city': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cities[i % len(cities)].encode('utf-8')])),
        'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        'price': tf.train.Feature(float_list=tf.train.FloatList(value=[100000.0 + i*1000.0]))
      }))
      writer.write(example.SerializeToString())
  writer.close()

def filter_examples_by_string_feature(examples_input):
    """Filters examples to exclude those with the 'city' feature equal to "Springfield"."""

    filtered_examples = (
        examples_input
        | 'ReadTfRecords' >> beam.io.ReadFromTFRecord(file_pattern=examples_input.uri + '/*')
        | 'ParseTfExample' >> beam.Map(tf.io.parse_single_example,
                                        features={
                                             'city': tf.io.FixedLenFeature([], dtype=tf.string),
                                             'bedrooms': tf.io.FixedLenFeature([], dtype=tf.int64),
                                             'price': tf.io.FixedLenFeature([], dtype=tf.float32)
                                         })
        | 'FilterSpringfield' >> beam.Filter(lambda example: example['city'] != b'Springfield')
        | 'SerializeToExample' >> beam.Map(lambda example: tf.train.Example(features=tf.train.Features(feature={
          'city': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['city']])),
          'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['bedrooms']])),
          'price': tf.train.Feature(float_list=tf.train.FloatList(value=[example['price']]))
        })).SerializeToString())
    )
    return filtered_examples

if __name__ == '__main__':
  import tempfile
  temp_dir = tempfile.mkdtemp()
  data_path = temp_dir + '/my_examples'
  _create_dummy_data(data_path)
  example_gen_output = _create_example_gen_output(data_path)
  with beam.Pipeline() as pipeline:
    filtered_data = filter_examples_by_string_feature(example_gen_output)
    # Write filtered data
    filtered_data | 'WriteFilteredTfRecords' >> beam.io.WriteToTFRecord(
        file_path_prefix=temp_dir + '/filtered_examples/data',
        file_name_suffix='.tfrecord')
  print("Filtered data written to ", temp_dir + '/filtered_examples')

```

Similar to the first example, dummy data is generated. Within the `filter_examples_by_string_feature` function, after parsing the examples, `beam.Filter` applies the lambda function: `lambda example: example['city'] != b'Springfield'`. Since string features are read as byte strings, 'Springfield' is also expressed as a byte string using `b'Springfield'`. Examples whose `city` feature does not match “Springfield” are retained, and subsequently serialized back into TFRecord format.

**Example 3: Filtering Based on Presence of a Feature**

In data cleaning, handling incomplete data is common. Let’s assume some records may lack a critical feature, let’s say `garage_size`. I need to filter out examples missing this feature. This requires checking feature existence within each example. Here's how it's done:

```python
import apache_beam as beam
import tensorflow as tf
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils.dsl_utils import external_input

# Dummy function to simulate the previous TFX component output.
def _create_example_gen_output(path_to_data):
  output = standard_artifacts.Examples()
  output.uri = path_to_data
  output.split_names = artifact_utils.encode_split_names(['train'])
  return output

def _create_dummy_data(data_path):
  # Simulate example data in tfrecords.
  writer = tf.io.TFRecordWriter(data_path + '/data.tfrecord')
  for i in range(100):
      feature_dict = {
          'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
          'price': tf.train.Feature(float_list=tf.train.FloatList(value=[100000.0 + i*1000.0]))
      }
      if i % 3 != 0:
        feature_dict['garage_size'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[i % 5]))
      example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

      writer.write(example.SerializeToString())
  writer.close()

def filter_examples_with_feature(examples_input):
    """Filters examples to keep those with the 'garage_size' feature present."""

    filtered_examples = (
        examples_input
        | 'ReadTfRecords' >> beam.io.ReadFromTFRecord(file_pattern=examples_input.uri + '/*')
        | 'ParseTfExample' >> beam.Map(tf.io.parse_single_example,
                                        features={
                                             'garage_size': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                                             'bedrooms': tf.io.FixedLenFeature([], dtype=tf.int64),
                                             'price': tf.io.FixedLenFeature([], dtype=tf.float32)
                                         })
        | 'FilterGarageSizePresent' >> beam.Filter(lambda example: example['garage_size'] != -1)
        | 'SerializeToExample' >> beam.Map(lambda example: tf.train.Example(features=tf.train.Features(feature={
          'garage_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['garage_size']])),
          'bedrooms': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['bedrooms']])),
          'price': tf.train.Feature(float_list=tf.train.FloatList(value=[example['price']]))
        })).SerializeToString())
    )
    return filtered_examples

if __name__ == '__main__':
  import tempfile
  temp_dir = tempfile.mkdtemp()
  data_path = temp_dir + '/my_examples'
  _create_dummy_data(data_path)
  example_gen_output = _create_example_gen_output(data_path)
  with beam.Pipeline() as pipeline:
    filtered_data = filter_examples_with_feature(example_gen_output)
    # Write filtered data
    filtered_data | 'WriteFilteredTfRecords' >> beam.io.WriteToTFRecord(
        file_path_prefix=temp_dir + '/filtered_examples/data',
        file_name_suffix='.tfrecord')
  print("Filtered data written to ", temp_dir + '/filtered_examples')
```

Here, within the dummy data generation, the `garage_size` feature is present for some records but missing for others. The key insight is within the `ParseTfExample` step: the `garage_size` feature is provided a `default_value` of -1, allowing us to handle missing features gracefully. Then, `beam.Filter` uses the condition `lambda example: example['garage_size'] != -1`. Examples having a `garage_size` feature (thus not having its default value of -1), are retained for the next stage.

In addition to these examples, it’s possible to combine multiple filtering conditions using boolean operators (`and`, `or`). Furthermore, custom callable functions, rather than lambda expressions, can be used to encapsulate more complex filtering logic. This allows for greater flexibility and maintainability of the filtering rules.

For further study, I recommend consulting the Apache Beam documentation, focusing on the `beam.Filter` transform, alongside the TFX documentation, especially components related to data transformations like the Transform component. Reviewing TensorFlow’s protocol buffer documentation, particularly the `tf.train.Example` structure is also beneficial.  Examining the source code of relevant TFX components can further solidify understanding of the implemented filtering methodologies. Examining examples of large-scale data pipelines developed with TFX and Beam can also provide additional practical insights.
