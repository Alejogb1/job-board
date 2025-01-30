---
title: "How do I specify image shape when reading and writing TFRecords with TensorFlow?"
date: "2025-01-30"
id: "how-do-i-specify-image-shape-when-reading"
---
The correct handling of image shape metadata within TFRecords is crucial for efficient and error-free TensorFlow pipelines, often proving a major source of frustration if not carefully addressed. Directly embedding the image dimensions, particularly height and width, as features in each TFRecord example is essential, rather than relying solely on implicit assumptions or static configuration outside the record itself.

My experience has shown that problems arise when the preprocessing stage or model input layers assume a fixed shape, while the actual images within the TFRecord vary. This mismatch commonly manifests as a dimension-related error during data loading or model training. Therefore, a robust approach involves consistently storing the image's shape alongside the raw image data within the TFRecord itself. This approach empowers the TensorFlow data pipeline to accommodate images of differing dimensions dynamically, preventing downstream issues related to shape mismatches.

The fundamental concept is to encode both the image's raw byte string and the height and width as distinct features within each TFRecord example. During the reading process, these shape features are then utilized to appropriately decode and reshape the images. The common mistake of assuming every image has the same size can lead to hard-to-debug errors, especially when dealing with datasets collected from diverse sources or through processes that do not enforce a strict image shape.

Here's how this process typically unfolds with a working example:

**Encoding Images into TFRecords**

To write image data and their corresponding shape information, a `tf.train.Example` is created containing these three features: the raw image data (as a string), height, and width (both as integers). This example is then serialized and written to a TFRecord file. It’s important to use fixed-length integer features for height and width, allowing direct access without requiring any extra parsing overhead.

```python
import tensorflow as tf
import numpy as np
import os

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(image_paths, tfrecord_path):
    """Creates a TFRecord file from a list of image paths."""
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image_path in image_paths:
          try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            height, width, _ = image.shape
            image_raw = tf.io.serialize_tensor(image).numpy()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(image_raw)
              }))
            writer.write(example.SerializeToString())
          except tf.errors.InvalidArgumentError as e:
              print(f"Skipping image {image_path} due to decoding error: {e}")

if __name__ == '__main__':
  # Create dummy images
  os.makedirs("dummy_images", exist_ok=True)
  img1 = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
  img2 = np.random.randint(0, 256, size=(150, 150, 3), dtype=np.uint8)
  img3 = np.random.randint(0, 256, size=(200, 100, 3), dtype=np.uint8)
  tf.keras.utils.save_img("dummy_images/image1.jpg", img1)
  tf.keras.utils.save_img("dummy_images/image2.jpg", img2)
  tf.keras.utils.save_img("dummy_images/image3.jpg", img3)
  image_paths = ["dummy_images/image1.jpg", "dummy_images/image2.jpg", "dummy_images/image3.jpg"]
  tfrecord_file = 'images.tfrecord'
  create_tfrecord(image_paths, tfrecord_file)
```
This code first defines helper functions to create TensorFlow features of type `bytes` or `int64`. The `create_tfrecord` function iterates through the given list of image paths, reads the image, decodes it, extracts the dimensions, serializes the tensor data, creates the features for the example, and then writes the example to a TFRecord file. The `try...except` block catches and logs image decoding errors to prevent a full process failure. Example dummy images are created and saved to disk to be used during the record creation.

**Reading Images from TFRecords**

The reading process involves parsing the TFRecord file, retrieving the raw image data, and subsequently reshaping the image based on the stored height and width values. `tf.io.parse_single_example` is used to parse individual TFRecord examples based on a defined feature description. The raw image data is then deserialized to obtain a tensor, and finally reshaped using the retrieved height and width. This way, every image is properly shaped regardless of its original dimensions.

```python
def parse_tfrecord(example):
  """Parses a TFRecord example into image and shape."""
  feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  parsed_example = tf.io.parse_single_example(example, feature_description)
  height = tf.cast(parsed_example['height'], tf.int32)
  width = tf.cast(parsed_example['width'], tf.int32)
  image_raw = parsed_example['image_raw']
  image = tf.io.parse_tensor(image_raw, out_type=tf.uint8)
  image = tf.reshape(image, tf.stack([height, width, 3])) # Reshape based on parsed dimensions
  return image

def load_tfrecord_dataset(tfrecord_path):
    """Loads images from a TFRecord file into a TensorFlow dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    return dataset

if __name__ == '__main__':
  tfrecord_file = 'images.tfrecord'
  dataset = load_tfrecord_dataset(tfrecord_file)

  for i, image in enumerate(dataset):
     print(f"Image {i+1} Shape: {image.shape}")
```
This snippet includes `parse_tfrecord`, which utilizes `tf.io.parse_single_example` with the correct feature specification to retrieve the height, width, and raw image data. The raw image is then deserialized using `tf.io.parse_tensor` and reshaped according to the parsed dimensions. `load_tfrecord_dataset` then uses this function to create a TensorFlow dataset. The main section then iterates through the loaded images and prints out their shapes, demonstrating that each image has been appropriately reshaped according to its actual original dimensions.

**Handling Different Image Types**

While these examples use RGB JPEG images, this approach can easily be generalized to other image types (e.g., PNG, grayscale) or to alternative data representations (e.g., serialized numerical data) by adapting the encoding and decoding procedures. The following example demonstrates handling serialized floating point pixel data.

```python
import tensorflow as tf
import numpy as np
import os

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord_float(data_arrays, tfrecord_path):
    """Creates a TFRecord file from a list of numpy arrays of floats."""
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for data_array in data_arrays:
            height, width = data_array.shape
            data_raw = tf.io.serialize_tensor(data_array).numpy()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'data_raw': _bytes_feature(data_raw)
            }))
            writer.write(example.SerializeToString())

def parse_tfrecord_float(example):
    """Parses a TFRecord example into array and shape."""
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'data_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    height = tf.cast(parsed_example['height'], tf.int32)
    width = tf.cast(parsed_example['width'], tf.int32)
    data_raw = parsed_example['data_raw']
    data = tf.io.parse_tensor(data_raw, out_type=tf.float32)
    data = tf.reshape(data, tf.stack([height, width]))
    return data

def load_tfrecord_dataset_float(tfrecord_path):
    """Loads float arrays from a TFRecord file into a TensorFlow dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_float)
    return dataset

if __name__ == '__main__':
    # Create dummy data
    data1 = np.random.rand(100, 200).astype(np.float32)
    data2 = np.random.rand(150, 150).astype(np.float32)
    data3 = np.random.rand(200, 100).astype(np.float32)
    data_arrays = [data1, data2, data3]
    tfrecord_file = 'float_data.tfrecord'
    create_tfrecord_float(data_arrays, tfrecord_file)
    dataset = load_tfrecord_dataset_float(tfrecord_file)

    for i, data in enumerate(dataset):
      print(f"Data {i+1} Shape: {data.shape}")
```
In this example, the methods `create_tfrecord_float` and `parse_tfrecord_float` have been defined to handle floating point arrays instead of images. The key change is the usage of `tf.float32` when deserializing the data and removing the channel information when reshaping, since the data is now 2D. The loading process is identical to the image-based one. This example demonstrates that the strategy of embedding shape metadata in TFRecords is adaptable to different data types.

For further learning, I recommend exploring the official TensorFlow documentation on TFRecords and the `tf.data` API, as well as studying more complex pipeline designs that handle variable sequence lengths with padding, alongside the image shape. A comprehensive understanding of these features enables building data pipelines that are resilient to variations in input data. Also research strategies to efficiently leverage the TFRecord format for distributed training or data parallel execution across GPUs/TPUs. Using the guidance here as a baseline and following these recommendations will ensure an understanding of how to effectively handle image shapes when reading and writing TFRecords with TensorFlow.
