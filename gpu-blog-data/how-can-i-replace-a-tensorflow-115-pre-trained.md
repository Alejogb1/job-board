---
title: "How can I replace a TensorFlow 1.15 pre-trained object detection model with an existing MS-COCO model in TensorFlow 2.7?"
date: "2025-01-30"
id: "how-can-i-replace-a-tensorflow-115-pre-trained"
---
Migrating from a TensorFlow 1.x object detection model to a TensorFlow 2.x equivalent, especially when involving pre-trained weights from different sources like the TensorFlow Detection Model Zoo and MS-COCO models, requires a careful approach due to significant API changes and internal representation differences. I’ve performed this type of migration across several projects and found the key lies in understanding the specific model architecture and the TensorFlow Object Detection API's reliance on configuration files.

The core challenge stems from the shift in TensorFlow's architecture and the evolution of the Object Detection API itself. TensorFlow 1.x relied heavily on static graphs and sessions, while TensorFlow 2.x adopted an eager execution paradigm and introduced `tf.function` for performance optimization. This impacted how models are defined, loaded, and executed. Furthermore, the pre-trained models in the TensorFlow 1.x zoo were often exported as frozen graphs, while models in TensorFlow 2.x are typically saved as SavedModels or H5 files, each having specific loading procedures.

The TensorFlow Object Detection API utilizes configuration files (typically `.config` files) to define the model architecture, training parameters, and input pipeline. To swap a TensorFlow 1.x model with an MS-COCO based TensorFlow 2.x model, you need to modify this configuration file to point to the new model’s checkpoint and also adjust any model-specific parameters.  It’s not a direct swap of weight files.

Here's a conceptual walkthrough of the process:

1. **Identify the desired TF2 model:** Determine the specific TF2 object detection model from the TensorFlow Hub or other sources that aligns with the performance characteristics you require. Many pre-trained models are available trained on the MS-COCO dataset, making it suitable for this migration.

2. **Download the new model:** Download the SavedModel or checkpoint files for the identified TF2 model. Note the directory structure where the checkpoint files or SavedModel resides, as this is crucial for setting up the configuration file.

3. **Obtain a suitable TF2 configuration:** Obtain a compatible configuration file (.config) for the selected TF2 model. The TensorFlow Object Detection API repository provides several sample configuration files for different model architectures. Select one similar to your intended model (e.g., `ssd_mobilenet_v2_fpn_640x640_coco17_tpu-8.config` for a MobileNet SSD with Feature Pyramid Network).

4. **Modify the configuration:** Edit the configuration file to specify the correct checkpoint or SavedModel paths. This will include modifying specific fields within the `model` section, `train_config` and `eval_config` sections depending on if the model is used for training or evaluation.

   *   **`model.ssd.preprocessor.normalize_image`:** When using models with a built in preprocessor, this needs to be set to false to prevent double normalization of the image if the preprocessor is used.
   *  **`model.ssd.image_resizer.fixed_shape_resizer.height`** and **`model.ssd.image_resizer.fixed_shape_resizer.width`**: Update these values to correspond to the input size of the new model (most models accept square images).
   * **`model.ssd.num_classes`**: Make sure the correct number of classes are being used by the new model.
   *  **`train_config.fine_tune_checkpoint`**: This is the most crucial field; it should point to the base checkpoint (not the training checkpoint) for the new model. For SavedModels, this often involves removing the `checkpoint` suffix and instead pointing to the SavedModel directory.
   *  **`train_config.fine_tune_checkpoint_type`**: If using a model checkpoint, this must be set to ‘detection’ .
   * **`train_config.use_bfloat16`**: Depending on the machine being used, this may need to be set to false.
   *  **`train_input_reader.tf_record_input_reader.input_path`** and **`eval_input_reader.tf_record_input_reader.input_path`**: Ensure the input paths point to the training/evaluation tfrecords.
   * **`eval_config.metrics_set`**: Make sure the correct evaluation metrics are being used by the new model.

5. **Training/Evaluation**: Using the modified configuration file, either resume training on your data or evaluate the model. Make sure to update your training script to be compatible with the new model.

**Code Examples with Commentary:**

These examples are conceptual. The exact implementation depends on the chosen models and directory structure.

**Example 1: Editing a configuration file with Python**

This example showcases how to load and modify a configuration file, changing the checkpoint path to point to a new model. Assume the old model was SSD ResNet50 and new model is a SavedModel export of a MobileNetV2.

```python
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

def update_config(config_path, new_checkpoint_path, new_num_classes):

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
      text_format.Merge(f.read(), config)

    # Assuming we're swapping to a SavedModel
    config.model.ssd.preprocessor.normalize_image = False
    config.model.ssd.image_resizer.fixed_shape_resizer.height = 300
    config.model.ssd.image_resizer.fixed_shape_resizer.width = 300
    config.model.ssd.num_classes = new_num_classes

    config.train_config.fine_tune_checkpoint = new_checkpoint_path
    config.train_config.fine_tune_checkpoint_type = 'detection' #or "classification" for classification models

    with open(config_path, 'wb') as f:
      f.write(text_format.MessageToString(config).encode())
    print(f"Configuration file updated. {config_path}")

# Example Usage
old_config_path = 'path/to/ssd_resnet50_old.config'
new_model_path = 'path/to/mobilenet_v2_saved_model'
new_number_of_classes = 90 # Number of classes the new model is trained on
update_config(old_config_path, new_model_path, new_number_of_classes)
```

*   The `update_config` function takes the configuration file path, the path to the new model, and the number of classes as input.
*   It loads the configuration using the TensorFlow Object Detection API's protobuf library.
*   It modifies relevant sections of the configuration to align with the characteristics of the new model, such as the `fine_tune_checkpoint`, image resizer size and the number of classes.
*   It saves the updated configuration to disk.

**Example 2: Loading a SavedModel and performing inference (Conceptual)**

This example demonstrates conceptually how you'd load a SavedModel and perform inference.

```python
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model(model_path):
  model = tf.saved_model.load(model_path)
  return model

def inference(model, image_path):
  img = Image.open(image_path).convert("RGB")
  img_array = np.array(img)
  input_tensor = tf.convert_to_tensor(img_array, dtype = tf.uint8)

  input_tensor = tf.expand_dims(input_tensor, axis=0)
  # The model may take a float tensor as input, convert to float before resizing
  input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)
  
  # The SavedModel expects a batch of images, hence the expansion of dimensions.
  detections = model(input_tensor) #Model specific arguments may be required

  print("Detection Boxes:",detections['detection_boxes'])
  print("Detection Scores:",detections['detection_scores'])

# Example Usage
saved_model_path = 'path/to/mobilenet_v2_saved_model'
image_path = 'path/to/your/image.jpg'
model = load_model(saved_model_path)
inference(model, image_path)
```
*   The `load_model` function uses `tf.saved_model.load` to load a model given the specified directory.
*   The `inference` function loads an image, converts it to a tensor, then uses the loaded model to perform inference. The output format depends on the specific model being used (e.g., bounding boxes, scores).

**Example 3: Updating training loop**

A small example of a training loop update. This demonstrates that any variables are trainable and optimizer updates can be made.

```python
import tensorflow as tf
def train_step(images, labels, model, optimizer, loss_function):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Example Usage

# Assume model is a loaded tf.saved_model object from the previous example
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy()

# Assuming images and labels are tf.data objects
# for images, labels in train_dataset:
#    loss = train_step(images, labels, model, optimizer, loss_function)
```

*   The `train_step` function uses a gradient tape to calculate the gradients
*   The optimizer `optimizer` is used to apply those gradients to the model trainable variables.
*   This will vary greatly based on the specific training procedure for a model.

**Resource Recommendations:**

*   **TensorFlow Object Detection API Documentation:** The official documentation is the primary source for understanding the configuration file format, training scripts, and model architectures. Review the tutorials and sample configurations provided in the repository.

*   **TensorFlow Model Garden:** Explore the TensorFlow Model Garden for pre-trained model checkpoints and SavedModels. Pay attention to the supported TensorFlow version and any specific instructions for each model.

*   **TensorFlow Hub:** TensorFlow Hub is another excellent resource for exploring pre-trained models, many of which are based on popular architectures and trained on large datasets such as MS-COCO.

In summary, swapping a TensorFlow 1.x object detection model with a TensorFlow 2.x model involves a process of configuration file modification and careful attention to input/output formats. While the initial migration may require effort, the benefits of moving to the newer TensorFlow ecosystem are substantial, including performance improvements, ease of use, and access to an ever-growing library of models and tools. I recommend starting with the TensorFlow Object Detection API documentation, finding a model and config, and then methodically modifying the config to point to the downloaded weights.
