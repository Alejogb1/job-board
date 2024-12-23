---
title: "How to export SSD-Mobilenet TFLite model with bounding box decoding but without NMS using TensorFlow Object Detection API?"
date: "2024-12-23"
id: "how-to-export-ssd-mobilenet-tflite-model-with-bounding-box-decoding-but-without-nms-using-tensorflow-object-detection-api"
---

Okay, let's tackle this. From my experience, exporting a tflite model with bounding box decoding but without non-maximum suppression (nms) from the TensorFlow Object Detection API is a fairly common requirement, especially when fine-tuning for edge deployments where you might want to handle nms on the device itself for greater control or latency reasons. It's not as straightforward as just flipping a switch, but with a few tweaks to the export process, it's definitely achievable. I remember vividly an early project where I was working on a custom drone navigation system. We needed a very streamlined pipeline, and moving nms onto the embedded processor was crucial. We faced some similar challenges then, and that experience definitely shaped my approach here.

The core issue boils down to how the TensorFlow Object Detection API's export process is configured. By default, it tends to package up the whole shebang including nms. To extract just the raw bounding box outputs, we need to modify the export pipeline. Specifically, the problem lies in the post-processing stage defined within the export process. Instead of relying on the standard `tf.function` decorated inference functions that handle bounding box decoding *and* nms, we have to construct a custom post-processing stage that isolates the raw outputs of the box regression and classification heads.

First off, a crucial aspect is understanding the structure of the output tensors that ssd_mobilenet provides before nms. Typically, you’ll have at least two key tensors: one for the class scores (or probabilities), and another for bounding box coordinates. In tflite, these are usually found after the last convolutional or fully connected layer just before any post processing. The shape for the class scores might look like `[1, num_anchors, num_classes]` and for the bounding boxes `[1, num_anchors, 4]` where num_anchors are the number of default boxes generated for the feature maps (dependent on the model configuration). This might vary based on the exact architecture configuration and the export parameters you used, so it's essential to verify the output tensor names and shapes after the initial export.

To get to this point without nms, you'll want to look into modifying the exporter's pipeline config, usually a `.config` protobuf file. Instead of the standard `tf.saved_model.save` functionality, you need to create an alternative model construction method that cuts the process short. Let me demonstrate how that might look with a few illustrative code examples:

**Example 1: Extracting raw outputs with a custom function**

```python
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection import exporter
from google.protobuf import text_format

def export_tflite_without_nms(pipeline_config_path, trained_checkpoint_dir, output_dir):
  """Exports tflite model without nms."""

  with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
    pipeline_config = text_format.Parse(f.read(),
                                       exporter.pipeline_pb2.TrainEvalPipelineConfig())
  model_config = pipeline_config.model
  detection_model = model_builder.build(
      model_config=model_config, is_training=False)

  checkpoint = tf.train.Checkpoint(model=detection_model)
  checkpoint.restore(tf.train.latest_checkpoint(trained_checkpoint_dir)).expect_partial()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32)])
  def inference_fn(image):
    image_tensor = tf.cast(image, dtype=tf.float32)
    features, preprocessed_image = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(features, preprocessed_image)
    raw_box_outputs = prediction_dict['raw_detection_boxes']
    raw_class_outputs = prediction_dict['raw_detection_scores']
    return raw_class_outputs, raw_box_outputs


  tf.saved_model.save(inference_fn, output_dir,
                       signatures={"serving_default": inference_fn.get_concrete_function()})

  converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
  tflite_model = converter.convert()
  with open(output_dir + "/model_without_nms.tflite", 'wb') as f:
    f.write(tflite_model)
  print(f"TFLite model saved to: {output_dir}/model_without_nms.tflite")


if __name__ == '__main__':
  pipeline_config_path = 'path/to/your/pipeline.config' # replace with actual config
  trained_checkpoint_dir = 'path/to/your/checkpoint' # replace with actual checkpoint dir
  output_dir = 'path/to/output_directory' # replace with desired output
  export_tflite_without_nms(pipeline_config_path, trained_checkpoint_dir, output_dir)

```

In this example, the crucial part is the `inference_fn` decorator. Inside it, we explicitly return the `raw_detection_boxes` and `raw_detection_scores` tensors which usually are available inside the `prediction_dict`. We're effectively sidestepping the typical bounding box decoding and nms step.

**Example 2: Using a `tf.Module` approach for exporting**

```python
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection import exporter
from google.protobuf import text_format

class NoNMSModel(tf.Module):
    def __init__(self, pipeline_config_path, trained_checkpoint_dir):
      super(NoNMSModel, self).__init__()
      with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
          pipeline_config = text_format.Parse(f.read(),
                                              exporter.pipeline_pb2.TrainEvalPipelineConfig())
      model_config = pipeline_config.model
      self.detection_model = model_builder.build(
          model_config=model_config, is_training=False)

      checkpoint = tf.train.Checkpoint(model=self.detection_model)
      checkpoint.restore(tf.train.latest_checkpoint(trained_checkpoint_dir)).expect_partial()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32)])
    def inference(self, image):
      image_tensor = tf.cast(image, dtype=tf.float32)
      features, preprocessed_image = self.detection_model.preprocess(image_tensor)
      prediction_dict = self.detection_model.predict(features, preprocessed_image)
      raw_box_outputs = prediction_dict['raw_detection_boxes']
      raw_class_outputs = prediction_dict['raw_detection_scores']
      return raw_class_outputs, raw_box_outputs


def export_tflite_without_nms_module(pipeline_config_path, trained_checkpoint_dir, output_dir):
  """Exports tflite model without nms using tf.Module"""
  model = NoNMSModel(pipeline_config_path, trained_checkpoint_dir)
  tf.saved_model.save(model, output_dir,
                        signatures={"serving_default": model.inference.get_concrete_function()})

  converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
  tflite_model = converter.convert()
  with open(output_dir + "/model_without_nms.tflite", 'wb') as f:
      f.write(tflite_model)
  print(f"TFLite model saved to: {output_dir}/model_without_nms.tflite")

if __name__ == '__main__':
  pipeline_config_path = 'path/to/your/pipeline.config' # replace with actual config
  trained_checkpoint_dir = 'path/to/your/checkpoint' # replace with actual checkpoint dir
  output_dir = 'path/to/output_directory' # replace with desired output
  export_tflite_without_nms_module(pipeline_config_path, trained_checkpoint_dir, output_dir)
```

This example structures the process slightly differently, using a `tf.Module` which could be beneficial if you need more intricate logic within the model, though for our purpose, the function-based approach is often sufficient. This also provides better encapsulation of the model and logic, promoting more reusable code.

**Example 3: Checking the output tensors**

```python
import tensorflow as tf

def check_tflite_model_outputs(tflite_model_path):
    """Prints names and shapes of input/output tensors of a tflite model"""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input tensors:")
    for detail in input_details:
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}")

    print("\nOutput tensors:")
    for detail in output_details:
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}")


if __name__ == '__main__':
    tflite_model_path = 'path/to/output_directory/model_without_nms.tflite'  #Replace with the actual path
    check_tflite_model_outputs(tflite_model_path)
```

This snippet demonstrates how to inspect the output tensors of your exported TFLite model. It will be incredibly helpful to confirm that the outputs match what you expect, such as `raw_detection_boxes` and `raw_detection_scores`.

As for resources, I'd suggest a deep dive into the official TensorFlow Object Detection API documentation itself. It covers the structure of the model outputs quite well and the details on pipeline configurations. For a deeper dive into the saved model format and `tf.function` behavior, the TensorFlow guide on saved models and autograph would be beneficial. Finally, understand the underlying math and implementation of SSD networks. Consider the original SSD paper by Liu et al. (SSD: Single Shot MultiBox Detector). This is not just a theory problem, understanding how it fundamentally works helps with debugging and efficient integration into your projects.

The key takeaway here is to understand where in the TensorFlow Object Detection pipeline the nms is being applied and to alter the export process to grab those intermediate outputs. This requires some manual configuration but is achievable. Good luck with your project, and feel free to ask more questions if needed.
