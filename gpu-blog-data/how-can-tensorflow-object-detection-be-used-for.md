---
title: "How can TensorFlow Object Detection be used for a single class?"
date: "2025-01-30"
id: "how-can-tensorflow-object-detection-be-used-for"
---
TensorFlow Object Detection, while powerful for multi-class scenarios, can be effectively adapted for single-class object detection, significantly simplifying model training and inference. My experience building automated inspection systems involved precisely this use case: identifying defects on a manufacturing line, where the sole target was a single 'defect' class. I'll detail how to achieve this, focusing on streamlining the process for a single object type, and share coding examples.

The key modification lies in configuring the model and training data to recognize only one class, which simplifies the complexity normally associated with multi-class detection. We’re essentially collapsing all potential categories into a single ‘positive’ class, against which we train the model to differentiate from a ‘background’ or ‘negative’ class. This alteration impacts the training data preparation and the model’s configuration.

Initially, the training data must be annotated to support a single class. Instead of providing bounding boxes with multiple class labels, all instances of the object you wish to detect are labeled with the same single, consistent identifier. For example, in a scenario detecting ‘car’ presence, all car bounding boxes would be labeled ‘car’, rather than separate ‘sedan’, ‘SUV’, or ‘truck’ classifications. This reduces the ambiguity and classification work required by the model during training.

The primary change in the configuration occurs within the model configuration file, generally a `.config` file used with TensorFlow Object Detection. This file specifies the model architecture, training parameters, and data paths. For single-class detection, modify the `num_classes` parameter to 1. This setting informs the model that it only needs to predict the probability for a single class label and the background.

Further optimization is achieved by adjusting evaluation metrics. When configured for multiple classes, the model calculates metrics like Mean Average Precision (mAP) across all classes. For single-class detection, you primarily need to track the Average Precision (AP), precision, recall, and F1-score directly relevant to your one class. The model’s evaluation metrics can be streamlined in training to focus on the performance around the single target. During training, visualizing the loss curve will be particularly helpful as it should only reflect the error for classifying between the singular object class and the background.

Here are some code examples illustrating these changes:

**Example 1: Modifying the Configuration File (`.config`)**

This is a snippet from a TensorFlow Object Detection model configuration file showing the critical modification for `num_classes`. We’ll assume the model has been pre-selected such as a SSD or Faster R-CNN model. The exact structure may vary, but the key concept remains the same: modifying `num_classes` in the appropriate sections.

```python
# ... (other config parameters) ...

model {
  ssd {
    num_classes: 1  # Set the number of classes to 1
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    # ... (other model specific parameters) ...
  }
}

train_config {
  batch_size: 32
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03
          total_steps: 4000
          warmup_learning_rate: 0.01
          warmup_steps: 500
        }
      }
      momentum_optimizer_value: 0.9
    }
  }
  # ... (other training parameters) ...
}
```
*Commentary:* The crucial line here is `num_classes: 1`. This line tells the underlying model architecture to only consider one class in its predictions. This change cascades through the model’s architecture and training regime, drastically simplifying the problem and leading to more targeted learning. You’ll typically find this parameter within the `model` block of the configuration.

**Example 2: Preparing the TFRecord Dataset**

The TensorFlow Object Detection API requires data in TFRecord format. This snippet illustrates how one would define a single class label during the creation of such a TFRecord file using a Python script. Assume that `example_data` contains bounding box coordinates, the image path, etc.

```python
import tensorflow as tf

def create_tf_example(example_data, class_name = "target_object"):
    #Unpack example_data into features
    image_path, xmins, xmaxs, ymins, ymaxs = example_data

    # Convert image to bytes
    image_bytes = tf.io.read_file(image_path)
    image_format = 'jpeg'  # Or png/other format
    width = 800 #or get the width from the image
    height = 600 #or get the height from the image

    # Create the TF Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes.numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format.encode('utf-8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_name.encode('utf-8')] * len(xmins))),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1] * len(xmins))),
    }))
    return tf_example

# Example usage - write tf_example
with tf.io.TFRecordWriter("single_class.record") as writer:
    # Example data - replace with your data
    example_data = ("images/image1.jpg", [0.2,0.6], [0.4,0.8],[0.1,0.5],[0.3,0.7] )
    tf_example = create_tf_example(example_data)
    writer.write(tf_example.SerializeToString())
```
*Commentary:* The important section is `'image/object/class/label'`. Instead of needing a series of different integers corresponding to different classes, we use '1' for all bounding boxes. Also the `image/object/class/text` assigns the single string class to all bounding boxes as well. This reflects our simplification of the problem to a singular object detection task. The class_name argument can be set to the proper name to assist in future identification of the classes. The same process is repeated for all images that contain the object. This TFRecord file will be used for training.

**Example 3: Inference Code with a Single Class**

The following example demonstrates how to load the trained single-class model, and interpret its output. The loaded model will output detections with a single class.

```python
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
detection_model = tf.saved_model.load('path/to/saved_model')

# Load and preprocess the image
image = cv2.imread('path/to/test_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = np.expand_dims(image_rgb, 0)
input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

# Perform inference
detections = detection_model(input_tensor)

# Extract detections (adjust indices based on the model output)
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy()

# Filter results based on confidence score (optional)
threshold = 0.5
for i in range(len(boxes)):
    if scores[i] > threshold:
        # Process the detected object
        ymin, xmin, ymax, xmax = boxes[i]
        h,w,_=image.shape
        ymin, xmin, ymax, xmax = int(ymin*h),int(xmin*w),int(ymax*h),int(xmax*w)
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        print("Object detected with score", scores[i])
        cv2.putText(image, f'Object: {scores[i]:.2f}', (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Display or save the image with detections
cv2.imshow("Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
*Commentary:* The code loads the model, preprocesses an image, performs inference, and processes the resulting bounding boxes and scores. Note that the `classes` output will always return a class ID of 0 (assuming the class has the id of 1) since the model only has to differentiate between the singular positive class and background. In this single-class scenario, we are primarily interested in the bounding box locations and the confidence score indicating whether or not the object is present.

For further learning, I recommend reviewing official TensorFlow Object Detection tutorials, paying special attention to sections on dataset preparation and model configuration. The TensorFlow documentation itself offers invaluable details on the specific architecture parameters and their impact. Practical experience can also be obtained using open-source datasets where one can practice training a model to detect one object from the dataset. Finally, the various blog posts and technical articles in the field offer insights and examples not explicitly covered in the official documentation. These resource types provide the foundation required to further solidify this knowledge.
