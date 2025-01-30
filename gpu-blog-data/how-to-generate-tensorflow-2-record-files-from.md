---
title: "How to generate TensorFlow 2 .record files from labelImg XML (Pascal VOC format)?"
date: "2025-01-30"
id: "how-to-generate-tensorflow-2-record-files-from"
---
Directly converting Pascal VOC XML annotations into TensorFlow TFRecord files requires parsing the XML structure, extracting bounding box coordinates and class labels, and serializing this data into a format TensorFlow can efficiently read for training object detection models. This process involves several key steps that, if mishandled, can lead to data loading errors or mismatched labels during training.

My experience with numerous object detection projects revealed the importance of rigorous data preprocessing and the benefits of using TFRecords. TFRecords offer optimized I/O operations when training models with large datasets. I've found, especially with high-resolution images, that directly reading images and their corresponding XML annotations during training drastically bottlenecks the training pipeline. Therefore, I’ve standardized on a conversion to TFRecord as an initial step in my projects.

The core task is to transform the label information contained in each XML file into a data structure digestible by the TensorFlow `tf.train.Example` protocol buffer. These `tf.train.Example` protocol buffers are the building blocks of the TFRecord files. Each `tf.train.Example` will represent a single image and its associated annotation data. The process usually follows this sequence:

1.  **XML Parsing:** For each image, use a library such as `xml.etree.ElementTree` to parse the corresponding Pascal VOC XML annotation file. This parsing extracts the image filename, dimensions (width and height), and bounding box annotations with class labels.

2.  **Data Extraction and Normalization:** Extract the bounding box coordinates (xmin, ymin, xmax, ymax) from the XML data, representing them as floating-point numbers normalized to the range \[0, 1] relative to the image dimensions. Also extract the object class labels from the XML and convert them either to integers or one-hot encoded vectors depending on the downstream model architecture.

3.  **TF.train.Example Construction:** Create a `tf.train.Example` protocol buffer, including the image data, the normalized bounding boxes, the class labels, and any other necessary data, serialized as TensorFlow features. I serialize the raw image bytes instead of the path; this approach allows for better flexibility with data loading strategies.

4.  **TFRecord Writing:** Serialize the constructed `tf.train.Example` protocol buffers using a `tf.io.TFRecordWriter` into one or more TFRecord files. I generally use a sharded approach, breaking the full dataset into multiple files, which simplifies parallel reading during model training.

Let's examine some code examples.

**Code Example 1: Parsing the XML and Extracting Data**

This example focuses on parsing a single XML file and extracting the critical information.

```python
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf

def parse_xml(xml_path, class_map):
  tree = ET.parse(xml_path)
  root = tree.getroot()
  filename = root.find('filename').text
  size = root.find('size')
  width = int(size.find('width').text)
  height = int(size.find('height').text)
  bboxes = []
  labels = []
  for obj in root.findall('object'):
    label = obj.find('name').text
    if label not in class_map:
        continue
    label_id = class_map[label]

    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    xmin = xmin / width
    ymin = ymin / height
    xmax = xmax / width
    ymax = ymax / height

    bboxes.append([xmin, ymin, xmax, ymax])
    labels.append(label_id)
  
  return filename, (height, width), np.array(bboxes,dtype=np.float32), np.array(labels,dtype=np.int64)
  
if __name__ == "__main__":
  class_map = {'cat': 0, 'dog': 1, 'bird': 2} # example class map
  
  #Example usage
  # Assuming an XML file exists named example.xml
  filename, image_dim, bboxes, labels  = parse_xml("example.xml",class_map)
  print(f"filename: {filename}")
  print(f"image dimensions: {image_dim}")
  print(f"bounding boxes: {bboxes}")
  print(f"labels: {labels}")
```

This code initializes a dictionary that maps class labels to integer IDs, which is crucial for numerical representation. The `parse_xml` function uses `xml.etree.ElementTree` to efficiently read and extract data from the XML format.  Bounding boxes are normalized relative to the image size. Crucially, it also handles the case where an annotation belongs to a class not present in the specified `class_map`, preventing unexpected errors during TFRecord creation.

**Code Example 2: Creating the `tf.train.Example` and Serializing Features**

This example outlines the creation of the TF `tf.train.Example` and illustrates how to serialize the various data components (image bytes, normalized bounding box coordinates, and class labels).

```python
def create_tf_example(image_bytes, image_dim, bboxes, labels):
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_dim[0]])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_dim[1]])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes[:, 0])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes[:, 1])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes[:, 2])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes[:, 3])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":
    #Example usage
    image_path = "test_image.jpg"  
    image = tf.io.read_file(image_path)
    image_bytes = image.numpy()
    
    dummy_dim = (200,300)
    dummy_boxes = np.array([[0.1,0.1,0.3,0.3], [0.6,0.6,0.9,0.9]], dtype=np.float32)
    dummy_labels = np.array([0,1], dtype=np.int64)
    
    example = create_tf_example(image_bytes, dummy_dim, dummy_boxes, dummy_labels)
    print(example)
```

The `create_tf_example` function constructs a dictionary that will be used to generate a `tf.train.Example`. Note how the bounding box coordinates and class labels are formatted, requiring proper conversions to TensorFlow’s `float_list` and `int64_list` feature types, respectively. Importantly, the image data is serialized as `bytes_list`.  I’ve observed that using `tf.io.read_file` to obtain the raw bytes prevents the common issue of accidentally using the file path within the TFRecord file.

**Code Example 3: Writing the TFRecord File**

Finally, this example demonstrates how to use `tf.io.TFRecordWriter` to write the `tf.train.Example` objects into the TFRecord file.

```python
def write_tfrecord(tfrecord_path, examples):
  with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

if __name__ == "__main__":
    #Example usage
    image_path = "test_image.jpg"  
    image = tf.io.read_file(image_path)
    image_bytes = image.numpy()
    
    dummy_dim = (200,300)
    dummy_boxes = np.array([[0.1,0.1,0.3,0.3], [0.6,0.6,0.9,0.9]], dtype=np.float32)
    dummy_labels = np.array([0,1], dtype=np.int64)
    
    example = create_tf_example(image_bytes, dummy_dim, dummy_boxes, dummy_labels)
    
    write_tfrecord("example.tfrecord", [example])

```

The `write_tfrecord` function handles the actual writing of the TFRecord data to disk. It iterates through a list of `tf.train.Example` and serializes each, ensuring each entry represents a single annotated image.  Using a context manager with the `tf.io.TFRecordWriter` object simplifies resource handling and prevents issues with incomplete writes.

In practice, the process involves iterating over each XML annotation file, pairing it with the corresponding image, and using the `parse_xml`, `create_tf_example`, and `write_tfrecord` to construct the TFRecord. To improve performance I will often use multiprocessing to parse and encode the TF examples and use a separate process to write to disk as writing can become a bottleneck.

For further study on this topic, I recommend reviewing the TensorFlow documentation on data loading and TFRecords. Additionally, exploring example implementations of object detection training pipelines on platforms like GitHub provides a hands-on approach. Researching common pitfalls in data preprocessing, such as inconsistent image size handling or incorrect label assignments, is also beneficial. Finally, reading materials covering the theory and practical implications of protocol buffers can deepen the understanding of TFRecords. These actions will help form a comprehensive knowledge of best practices when utilizing TFRecords.
