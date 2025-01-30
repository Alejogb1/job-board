---
title: "How can I convert TFRecords to COCO format?"
date: "2025-01-30"
id: "how-can-i-convert-tfrecords-to-coco-format"
---
Working extensively with large image datasets, I've encountered the need to bridge the gap between TensorFlow's TFRecord format and the Common Objects in Context (COCO) annotation format quite often. TFRecords, designed for efficient data storage and processing within TensorFlow pipelines, are optimized for ingestion, but lack the human-readability and interchangeability of COCO JSON files, often favored for training object detection models in frameworks outside of TensorFlow and general dataset visualization. The conversion, therefore, becomes a crucial step in a flexible machine learning workflow.

Essentially, the conversion process involves extracting data from the serialized TFRecord files and restructuring it into the required COCO JSON schema. This means parsing the encoded examples within TFRecords which typically include image data and bounding box information, often as serialized TF.Example protobuf messages. These messages need to be deserialized, the image data decoded, and the annotations extracted. These annotations, usually stored as a combination of bounding box coordinates, class labels, and possibly additional attributes, then need to be transformed into the COCO's structured representation. This COCO representation is essentially nested dictionaries or JSON data following the COCO format's specifications for images, categories, and annotations.

The complexity arises from several factors. First, TFRecords are binary files, not directly human-readable, and require knowledge of the underlying protobuf schema used during their creation. Second, TFRecords may contain varied data structures, especially when incorporating diverse training data. Third, the COCO format has specific requirements, including standardized key names, precise data types, and defined structures for various annotation types like bounding boxes, segmentation masks, and keypoints. The conversion therefore demands careful parsing and translation of the data while also ensuring that the COCO output is valid and conforms to all specifications.

Here's how I've handled this in practice, using Python, TensorFlow, and JSON libraries.

**Example 1: Basic Conversion with Bounding Boxes**

This example focuses on the core scenario: converting bounding boxes from TFRecords to COCO format. I'm assuming the TFRecord examples store images as byte strings and bounding boxes as four floating-point values – `ymin`, `xmin`, `ymax`, `xmax` – that are normalized between 0 and 1. The TFRecord is assumed to contain examples for only one category which corresponds to the category with id '1' in the COCO format.

```python
import tensorflow as tf
import json
import os
from PIL import Image
import io

def tfrecord_to_coco_basic(tfrecord_path, output_json_path, image_dir):
    """
    Converts a TFRecord to COCO JSON format (basic bounding boxes).
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    annotation_id = 1
    image_id = 1
    for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord_path):
        example_proto = tf.train.Example()
        example_proto.ParseFromString(example)

        image_bytes = example_proto.features.feature['image/encoded'].bytes_list.value[0]
        ymin = example_proto.features.feature['image/object/bbox/ymin'].float_list.value[0]
        xmin = example_proto.features.feature['image/object/bbox/xmin'].float_list.value[0]
        ymax = example_proto.features.feature['image/object/bbox/ymax'].float_list.value[0]
        xmax = example_proto.features.feature['image/object/bbox/xmax'].float_list.value[0]

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        filename = f"image_{image_id}.jpg"
        image_path = os.path.join(image_dir, filename)
        image.save(image_path)

        coco_image = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_data["images"].append(coco_image)

        bbox_width = (xmax - xmin) * width
        bbox_height = (ymax - ymin) * height
        bbox_xmin = xmin * width
        bbox_ymin = ymin * height

        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [bbox_xmin, bbox_ymin, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }

        coco_data["annotations"].append(coco_annotation)
        annotation_id += 1
        image_id +=1


    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
  tfrecord_file = "example.tfrecord"  # Replace with path to your TFRecord file
  output_json = "output.json"  # Path to output COCO JSON file
  image_storage_path = "images/" # Path to store the extracted images
  os.makedirs(image_storage_path, exist_ok=True) # Create the image storage path if it doesn't exist

  tfrecord_to_coco_basic(tfrecord_file, output_json, image_storage_path)
  print(f"COCO JSON written to {output_json}")
```

In this script, `tfrecord_to_coco_basic` function reads a TFRecord file, deserializes the encoded examples, extracts image and bounding box data, and transforms it to the appropriate format. Specifically, images are extracted and saved into a provided image folder. Bounding box coordinates are converted from normalized values to absolute pixel values and inserted into the `bbox` key in the COCO annotation. The image and annotation dictionaries are populated and stored inside the `coco_data` dictionary which is output as a JSON file.

**Example 2: Handling Multiple Categories**

This example demonstrates how to handle multiple object categories. This will require information on the class labels, assumed to be stored in each example. Here, I'm assuming `image/object/class/label` contains an integer that corresponds to an object class category.

```python
import tensorflow as tf
import json
import os
from PIL import Image
import io


def tfrecord_to_coco_multicategory(tfrecord_path, output_json_path, image_dir, category_map):
    """
     Converts a TFRecord with multiple categories to COCO JSON format.
    """

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for id, name in category_map.items()]
    }

    annotation_id = 1
    image_id = 1

    for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord_path):
        example_proto = tf.train.Example()
        example_proto.ParseFromString(example)

        image_bytes = example_proto.features.feature['image/encoded'].bytes_list.value[0]
        ymin = example_proto.features.feature['image/object/bbox/ymin'].float_list.value[0]
        xmin = example_proto.features.feature['image/object/bbox/xmin'].float_list.value[0]
        ymax = example_proto.features.feature['image/object/bbox/ymax'].float_list.value[0]
        xmax = example_proto.features.feature['image/object/bbox/xmax'].float_list.value[0]
        class_label = int(example_proto.features.feature['image/object/class/label'].int64_list.value[0])

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        filename = f"image_{image_id}.jpg"
        image_path = os.path.join(image_dir, filename)
        image.save(image_path)


        coco_image = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_data["images"].append(coco_image)


        bbox_width = (xmax - xmin) * width
        bbox_height = (ymax - ymin) * height
        bbox_xmin = xmin * width
        bbox_ymin = ymin * height
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_label,
            "bbox": [bbox_xmin, bbox_ymin, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }

        coco_data["annotations"].append(coco_annotation)
        annotation_id += 1
        image_id+=1
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
    tfrecord_file = "example.tfrecord" # Path to the TFRecord file
    output_json = "output_multi.json" # Path to the output COCO JSON file
    image_storage_path = "images/"  #Path to store the extracted images
    os.makedirs(image_storage_path, exist_ok=True)  # Create the image storage path if it doesn't exist

    category_map = {1: 'cat', 2: 'dog', 3: 'bird'} # Example category map

    tfrecord_to_coco_multicategory(tfrecord_file, output_json, image_storage_path, category_map)
    print(f"COCO JSON written to {output_json}")
```

This version, `tfrecord_to_coco_multicategory`, utilizes a `category_map` (a dictionary) to define the association between integer class labels and category names. These categories are added into the `coco_data` dictionary. For each annotation, it extracts the integer class label using the `class_label` from the TFRecord file and uses this value as the `category_id` in the COCO JSON. This results in an output JSON formatted correctly to handle multiple categories in COCO.

**Example 3: Handling Variable Object Counts**

In a more complex scenario, each image within a TFRecord might contain a variable number of objects. This requires parsing bounding boxes as lists. The following example demonstrates such a process, assuming the TFRecord contains lists of bounding box and class labels.

```python
import tensorflow as tf
import json
import os
from PIL import Image
import io


def tfrecord_to_coco_variable(tfrecord_path, output_json_path, image_dir, category_map):
    """
     Converts a TFRecord with variable object counts to COCO JSON format.
    """

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for id, name in category_map.items()]
    }
    annotation_id = 1
    image_id = 1

    for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord_path):
        example_proto = tf.train.Example()
        example_proto.ParseFromString(example)

        image_bytes = example_proto.features.feature['image/encoded'].bytes_list.value[0]
        ymin_list = example_proto.features.feature['image/object/bbox/ymin'].float_list.value
        xmin_list = example_proto.features.feature['image/object/bbox/xmin'].float_list.value
        ymax_list = example_proto.features.feature['image/object/bbox/ymax'].float_list.value
        xmax_list = example_proto.features.feature['image/object/bbox/xmax'].float_list.value
        class_labels = example_proto.features.feature['image/object/class/label'].int64_list.value

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        filename = f"image_{image_id}.jpg"
        image_path = os.path.join(image_dir, filename)
        image.save(image_path)

        coco_image = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_data["images"].append(coco_image)

        for i in range(len(ymin_list)):
            ymin = ymin_list[i]
            xmin = xmin_list[i]
            ymax = ymax_list[i]
            xmax = xmax_list[i]
            class_label = int(class_labels[i])

            bbox_width = (xmax - xmin) * width
            bbox_height = (ymax - ymin) * height
            bbox_xmin = xmin * width
            bbox_ymin = ymin * height

            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_label,
                "bbox": [bbox_xmin, bbox_ymin, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_annotation)
            annotation_id += 1
        image_id+=1
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
    tfrecord_file = "example.tfrecord" # Path to the TFRecord file
    output_json = "output_variable.json" # Path to the output COCO JSON file
    image_storage_path = "images/"  # Path to store the extracted images
    os.makedirs(image_storage_path, exist_ok=True)  # Create the image storage path if it doesn't exist
    category_map = {1: 'person', 2: 'car', 3: 'bicycle'} # Example category map

    tfrecord_to_coco_variable(tfrecord_file, output_json, image_storage_path, category_map)
    print(f"COCO JSON written to {output_json}")
```

In `tfrecord_to_coco_variable`, bounding box coordinates and class labels are extracted as lists from the TFRecord example. The function then iterates through these lists, generating a new COCO annotation for each object in the image, and saving the results to a JSON file. This handles the scenario where images have a varying number of annotations.

For further development and understanding of TFRecord to COCO conversions, I recommend exploring several resources. For TensorFlow, the official documentation is essential for understanding TFRecord usage and the `tf.train.Example` format. Detailed guides on working with TensorFlow datasets and data pipelines can also be invaluable. To better understand the COCO format, its specifications, including the details of the JSON schema and data types can be found in its official documentation.  Furthermore, the Python json library, and its documentation, is crucial for writing the COCO annotations in valid JSON. Finally, the PIL (Pillow) library is key for handling the image processing requirements.  By understanding these tools, you can adapt these examples for more complex use cases such as handling segmentations and keypoints.
