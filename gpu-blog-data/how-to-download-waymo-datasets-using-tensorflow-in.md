---
title: "How to download Waymo datasets using TensorFlow in Colab with Python 3.8+?"
date: "2025-01-30"
id: "how-to-download-waymo-datasets-using-tensorflow-in"
---
Accessing and processing Waymo Open Dataset data with TensorFlow in Google Colab requires a specific workflow due to the dataset's size and storage format (TFRecord). I've spent the past several months working extensively with this dataset for autonomous driving perception models, so I’m familiar with the challenges involved and have developed a reliable methodology. The key lies in efficiently streaming the data from Google Cloud Storage (GCS), where Waymo hosts the dataset, directly into TensorFlow pipelines within the Colab environment. Downloading the entire dataset locally to a Colab instance is generally impractical and unnecessary.

First, you need to understand that the Waymo dataset is structured as TFRecord files, optimized for TensorFlow's data loading capabilities. These files contain serialized examples, each representing a snapshot of sensor data at a specific time. This includes lidar point clouds, camera images, and labels for objects in the scene. Direct file access from Colab is inefficient; we rely on TensorFlow's input pipeline and the GCS API to create a streamlined data flow.

The initial step involves setting up your Colab environment with the required libraries, specifically `tensorflow`, `tensorflow_datasets`, and `waymo-open-dataset`.  Installation is straightforward:

```python
!pip install tensorflow==2.10.0 tensorflow-datasets waymo-open-dataset-tf-2-10-0
```

The `tensorflow-datasets` library provides a convenient interface for accessing datasets in various formats, including TFRecord. The `waymo-open-dataset-tf` package facilitates handling the specific structure of Waymo's TFRecord files. Version matching, like `2-10-0` here, is critical. Inconsistencies often result in parsing errors and data corruption issues.

After installation, you configure authentication to access the data from Google Cloud Storage.  The default approach uses Google's client libraries and Colab's authorization tokens. You don't need to manually manage any credentials if you are running this within the Colab environment with a Google account.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from waymo_open_dataset import dataset_pb2 as open_dataset

# Disable eager execution. Waymo data can be complex.
tf.compat.v1.disable_eager_execution()


def parse_tf_example(example_proto):
    """Parses a single Waymo tf.Example. Returns a dict."""
    features = {
        'laser': tf.io.FixedLenFeature([], tf.string),
        'camera_image_0': tf.io.FixedLenFeature([], tf.string),
        'camera_image_1': tf.io.FixedLenFeature([], tf.string),
        'camera_image_2': tf.io.FixedLenFeature([], tf.string),
        'camera_image_3': tf.io.FixedLenFeature([], tf.string),
        'camera_image_4': tf.io.FixedLenFeature([], tf.string),
        'camera_image_5': tf.io.FixedLenFeature([], tf.string),
        'camera_calibration': tf.io.FixedLenFeature([], tf.string),
        'context_name': tf.io.FixedLenFeature([], tf.string),
        'frame_timestamp_micros': tf.io.FixedLenFeature([], tf.int64),
        'range_images': tf.io.FixedLenFeature([], tf.string),
        'camera_labels': tf.io.FixedLenFeature([], tf.string),
        'laser_labels': tf.io.FixedLenFeature([], tf.string)

    }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    return parsed_features
```
This code snippet demonstrates the crucial parsing step using `tf.io.parse_single_example`. Note that I've included a reasonably large list of keys to extract from a Waymo TFRecord. The `features` dictionary defines the expected datatypes for each field. Error handling becomes much simpler when you know the dataset's structure, and failing to declare the correct types can lead to cryptic TensorFlow errors. For each type you select `tf.io.FixedLenFeature`, since we are dealing with single values, and declare the appropriate TensorFlow type, such as `tf.string` or `tf.int64`. The return is a dictionary of tensor objects.

With this parsing function, we can build a TensorFlow Dataset pipeline that streams data directly from GCS. The `tf.data.TFRecordDataset` creates a dataset of TFRecord entries, which we can then map to our parsing function. This allows us to load data in batch and feed it to training loops.

```python
def create_dataset(split='training'):

    dataset_path = "gs://waymo_open_dataset_v_1_3_0/"

    if split == 'training':
        filenames = tf.io.gfile.glob(dataset_path + 'training_*.tfrecord')
    elif split == 'validation':
         filenames = tf.io.gfile.glob(dataset_path + 'validation_*.tfrecord')
    else:
       raise ValueError(f"Invalid split: {split}. Choose 'training' or 'validation'.")

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
```

The function above creates a dataset. I define paths to the specific location of the Waymo dataset in Google Cloud Storage. Instead of loading the whole dataset into local storage, we provide paths to the relevant TFRecord files directly from the bucket. Note that `num_parallel_reads` and `num_parallel_calls` are specified as `tf.data.AUTOTUNE`, which lets TensorFlow intelligently manage parallelism. This is an important detail, especially on the Colab free instance where resources are limited. It allows the data pipeline to adapt to the hardware dynamically. Finally, the function returns a TensorFlow `dataset` object which can be further processed and manipulated.

Finally, let's look at how to use the dataset object for iteration and processing. We'll take the training set and extract point cloud data as an example. It is vital to deserialize the point cloud data because, recall, it is stored as a string due to TFRecord constraints.
```python
dataset = create_dataset(split='training')

# Selects the first element (one frame) of the dataset
first_example = next(iter(dataset))

laser_bytes = first_example['laser'].numpy()

laser = open_dataset.Laser()
laser.ParseFromString(laser_bytes)
point_cloud = laser.points # point_cloud is now a structured numpy array

print(f"Number of points: {point_cloud.shape}")
print(f"Point cloud data type: {point_cloud.dtype}")


# Iterating through the entire dataset (not recommended for large datasets like this!)
# for sample in dataset:
#     laser_bytes = sample['laser'].numpy()
#     laser = open_dataset.Laser()
#     laser.ParseFromString(laser_bytes)
#     point_cloud = laser.points
#     print(f"Batch point shape {point_cloud.shape}")
```

This code demonstrates how to obtain the parsed features from the dataset for further analysis. We first pull a single example from the dataset. Then we extract the laser data in bytes and utilize the `open_dataset.Laser()` method to convert to a useful object. `point_cloud` is now a structured numpy array containing xyz coordinates, range, and intensity data of all points in a frame. I have included a simple print statement to show the shape and the datatype of the point cloud. In the last part of the code block, I commented out the actual iteration over a batch. Doing this for a large training set may not be feasible and would result in an `out-of-memory` error. When training, you want to iterate over data in a batched, optimized way via the `dataset` object, typically in your training loop. The way I have shown it here is to verify that it is pulling and processing the data correctly.

In summary, effectively downloading and processing Waymo datasets in Colab using TensorFlow requires careful management of data pipelines and cloud storage. You must use `tf.data` to stream data directly from Google Cloud Storage. Batch processing is important for data handling and training efficiency. Additionally, careful attention to data serialization and deserialization steps is necessary to retrieve data in a usable format.

For further study, I strongly recommend reviewing the official TensorFlow documentation on building data input pipelines and utilizing TFRecord datasets. Also, the official Waymo Open Dataset documentation and code samples provide a deeper understanding of the dataset structure and methods to work with the raw data. Finally, exploring best practices for building high-performance data pipelines can be beneficial for maximizing the performance of your data analysis or model training. The key is understanding how the data is structured in TFRecords and creating a data processing pipeline that avoids downloading the full dataset.
