---
title: "How to create TFRecord batches for TensorFlow training?"
date: "2025-01-30"
id: "how-to-create-tfrecord-batches-for-tensorflow-training"
---
TensorFlow’s TFRecord format offers an efficient method for storing sequence of data, and proper batching is vital for effective training of deep learning models. The efficiency comes from the ability of TFRecords to hold large datasets of different input types in a single file. Batching further optimizes this by creating groups of examples for model training at once, mitigating the input bottleneck. I've spent considerable time refining data pipelines for image recognition and natural language processing projects, where meticulous batching of TFRecords was the cornerstone for performant models.

At its core, creating TFRecord batches involves combining the capabilities of TensorFlow's `tf.data` API with the file reading functionality tailored for TFRecords. The process breaks down into a series of distinct steps, each crucial for creating a reliable and optimized data feed to our TensorFlow models.

First, we need to define how we read individual data points from the TFRecord. TFRecords, unlike simple text files, are serialized records with a specific schema that needs to be deserialized correctly. We define a function that can parse an example from the TFRecord file. This parser function is essential since the information stored within the record is binary. It decodes the byte strings into usable tensors based on your defined feature structure. It does this by using a map of feature keys and their corresponding data types using `tf.io.FixedLenFeature`. This step is critical as it ensures that the data read matches the expectations of your model architecture.

Second, we use the `tf.data.TFRecordDataset` to create a dataset object from a TFRecord file, or a list of TFRecord files. This object allows us to apply transformations and operations on the data in an efficient, asynchronous way. The dataset represents an iterable collection of parsed elements from your TFRecord.

Third, applying transformations like shuffling and repeating can significantly improve model training and stability. We shuffle the dataset with `dataset.shuffle()` to reduce the effects of ordered input data. Shuffling must be done prior to batching, or training stability and effectiveness will decrease because, if data is ordered according to the original feature within the record, our models will see a skewed distribution of values during training. We then typically repeat the data with `dataset.repeat()`. This operation repeats the dataset indefinitely (or a specific number of times) to allow more training iterations to occur.

Lastly, we batch the dataset using `dataset.batch()`. This is where the groups of examples are formed. You specify the batch size here, determining how many training examples are grouped into a single batch. After batching, the data is ready to be used for model training. The following sections describe specific code implementations and commentary.

Here’s a practical code example detailing this process. This example focuses on image data and assumes the feature dictionary consists of the "image" and "label" keys.

```python
import tensorflow as tf

def parse_tfrecord_example(example_proto):
  """Parses a single TFRecord example."""
  feature_description = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64),
  }
  parsed_example = tf.io.parse_single_example(example_proto, feature_description)
  image = tf.io.decode_jpeg(parsed_example['image'], channels=3) # Assuming JPEG images
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = parsed_example['label']
  return image, label


def create_batched_dataset(tfrecord_files, batch_size, shuffle_buffer_size):
  """Creates a batched dataset from TFRecord files."""
  dataset = tf.data.TFRecordDataset(tfrecord_files)
  dataset = dataset.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  return dataset

# Example Usage
tfrecord_files = ['/path/to/example1.tfrecord', '/path/to/example2.tfrecord']
batch_size = 32
shuffle_buffer_size = 1000
batched_dataset = create_batched_dataset(tfrecord_files, batch_size, shuffle_buffer_size)

# To iterate over the batches
for images, labels in batched_dataset.take(10):  # Take 10 batches
    print("Batch shape:", images.shape, labels.shape)
```
In this first example, `parse_tfrecord_example` defines the feature schema and parses the images and labels. Crucially, within this function, we decode the byte string representation of the image using `tf.io.decode_jpeg` and convert them to a float32 representation with `tf.image.convert_image_dtype`. This ensures they are ready for use with TensorFlow models. The `create_batched_dataset` function takes a list of file paths, batch size, and shuffle buffer size as input. We also add `prefetch()` at the end. This asynchronous process overlaps data loading and processing with the model's training loop, effectively utilizing resources and accelerating the training cycle. The `take(10)` method is used to limit iterations for demonstrative purposes.

The next example demonstrates a slightly more complex scenario where images are resized as part of the data preparation process within the parsing function, and adds optional data augmentation, which has proven highly valuable in image analysis:

```python
import tensorflow as tf

def parse_and_preprocess(example_proto, image_size=(224, 224), augment=False):
    """Parses, preprocesses, and optionally augments a single TFRecord example."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, image_size)

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)

    label = parsed_example['label']
    return image, label

def create_batched_dataset_with_preprocess(tfrecord_files, batch_size, shuffle_buffer_size, image_size=(224, 224), augment=False):
  """Creates a batched dataset with preprocessing."""
  dataset = tf.data.TFRecordDataset(tfrecord_files)
  dataset = dataset.map(lambda x: parse_and_preprocess(x, image_size, augment), num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  return dataset

# Example Usage
tfrecord_files = ['/path/to/example1.tfrecord', '/path/to/example2.tfrecord']
batch_size = 32
shuffle_buffer_size = 1000
image_size = (256,256)
augment_data = True
batched_dataset = create_batched_dataset_with_preprocess(tfrecord_files, batch_size, shuffle_buffer_size, image_size, augment_data)

for images, labels in batched_dataset.take(10):
    print("Batch shape:", images.shape, labels.shape)

```
In this second example, we introduce `tf.image.resize` to set a standard image size across the dataset. This is useful when input data has varying shapes. We also demonstrate optional data augmentation, including random horizontal flips and brightness adjustments. The `augment` parameter controls whether this augmentation is applied, which is useful for training datasets. We use a lambda function as a convenience for a cleaner calling of the map function. This allows greater flexibility during data input.

Finally, this third example moves to a different domain of text data processing. It shows how to deal with variable length text sequences, using padding as a way to batch sequences of different length:

```python
import tensorflow as tf
import numpy as np

def parse_text_example(example_proto):
    """Parses a single TFRecord example containing text data."""
    feature_description = {
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    text = parsed_example['text']
    label = parsed_example['label']
    return text, label

def preprocess_text(text, vocab_size, max_length=50):
    """Preprocesses text into integer sequences and pads them."""
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length = max_length)
    tokenizer.adapt(tf.data.Dataset.from_tensor_slices([text]))
    sequence = tokenizer(text)
    return sequence

def create_batched_text_dataset(tfrecord_files, batch_size, shuffle_buffer_size, vocab_size, max_length = 50):
  """Creates a batched dataset for text data from TFRecord files."""
  dataset = tf.data.TFRecordDataset(tfrecord_files)
  dataset = dataset.map(parse_text_example, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.map(lambda text, label : (preprocess_text(text, vocab_size, max_length), label) , num_parallel_calls = tf.data.AUTOTUNE)
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat()
  dataset = dataset.padded_batch(batch_size, padded_shapes=([max_length],[]))
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  return dataset


# Example Usage
tfrecord_files = ['/path/to/text_example1.tfrecord', '/path/to/text_example2.tfrecord']
batch_size = 32
shuffle_buffer_size = 1000
vocab_size = 10000 # Define vocab size
max_length = 50
batched_dataset = create_batched_text_dataset(tfrecord_files, batch_size, shuffle_buffer_size, vocab_size, max_length)

for sequences, labels in batched_dataset.take(10):
  print("Batch shape:", sequences.shape, labels.shape)
```
In this third example, we've introduced a `TextVectorization` layer. This layer maps strings to numerical sequences. We use `padded_batch()` which will pad out sequences of varying lengths to a consistent size, determined by our `max_length` parameter. This batching method allows us to effectively use text data with recurrent models or transformer architectures. In contrast to the fixed input shape used in our prior examples, the padding provided by the `padded_batch()` method makes batching of non-uniform sequence lengths possible.

For additional learning, I'd recommend exploring the official TensorFlow documentation on `tf.data` and the TFRecord format. Reading through tutorials demonstrating different input data types and preprocessing methods would also be beneficial. The "Effective TensorFlow" book also covers advanced data preprocessing, while the Stanford CS230 Deep Learning course also provides great practical guides.
