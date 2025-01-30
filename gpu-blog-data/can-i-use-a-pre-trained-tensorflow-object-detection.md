---
title: "Can I use a pre-trained TensorFlow object detection model on a different computer without the API?"
date: "2025-01-30"
id: "can-i-use-a-pre-trained-tensorflow-object-detection"
---
Transferring and utilizing a pre-trained TensorFlow object detection model on a different computer, independent of the full TensorFlow Object Detection API, is entirely feasible and a common practice. The key lies in understanding that the model's architecture (e.g., a specific variant of Faster R-CNN or SSD) and its trained weights are distinct components, and it is these components, not the entire API, which are crucial for inference. The Object Detection API acts primarily as a convenience layer for training, model management, and evaluation, but a deployed model requires only the model's computational graph and trained parameters, typically stored in a saved model format or checkpoint files. My experience deploying edge devices has involved numerous similar scenarios, and I routinely extract and utilize these models without needing the full API.

To implement object detection without the API, one needs to load the pre-trained model's graph and corresponding parameters into a runtime environment like TensorFlow or TensorFlow Lite, coupled with an input pipeline and post-processing logic. The first step generally involves exporting the trained model in a format suitable for consumption, commonly a SavedModel or, for optimized deployment, a TensorFlow Lite format. SavedModels preserve the computational graph and the parameters, while TensorFlow Lite models are often quantized and optimized for resource-constrained environments.

Once the model is exported, you can load it in a Python environment using TensorFlow's core libraries or, for more resource-efficient deployments, utilize TensorFlow Lite. Regardless of the format, several key procedures are always required: graph loading, parameter loading, input preparation, and output processing. This process can also bypass the more complex infrastructure and dependencies that the Object Detection API introduces. For instance, when dealing with real-time embedded vision applications where computational efficiency is paramount, direct API use becomes problematic, thus, using a simplified inference approach without the API is frequently necessary.

Here are examples demonstrating the core procedure using Python and TensorFlow 2.x:

**Example 1: Using a SavedModel with TensorFlow**

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
saved_model_path = 'path/to/your/exported_saved_model' # Replace with actual path
loaded_model = tf.saved_model.load(saved_model_path)
infer = loaded_model.signatures['serving_default']

# Preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640)) # Resize to model's input size (adjust accordingly)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return tf.constant(image)

# Run inference
def run_inference(image_path):
    input_tensor = preprocess_image(image_path)
    detections = infer(input_tensor)
    return detections


# Post process detections to extract bounding boxes and scores
def postprocess_detections(detections):
  boxes = detections['detection_boxes'][0].numpy()
  scores = detections['detection_scores'][0].numpy()
  classes = detections['detection_classes'][0].numpy().astype(np.int32)
  num_detections = int(detections['num_detections'][0].numpy())


  valid_detections = []
  for i in range(num_detections):
        if scores[i] > 0.5: # Threshold for confidence
            valid_detections.append(
                {
                    "box":boxes[i],
                    "score":scores[i],
                    "class":classes[i]
                }
                )
  return valid_detections

# Example usage:
image_file = 'path/to/your/image.jpg' # Replace with actual path
output = run_inference(image_file)
valid_detections = postprocess_detections(output)
print(valid_detections)

```

This example demonstrates loading a SavedModel, preprocessing an image, running the model, and retrieving results. The `saved_model_path` should be the path to your exported SavedModel directory which usually contains the `saved_model.pb` file, the `variables` folder and potentially the `assets` folder. The `preprocess_image` function resizes and normalizes the input image to be compatible with the model’s expected input format. Importantly, the `infer` object is obtained by accessing the `serving_default` signature of the loaded model. The  `postprocess_detections` function extracts valid detections based on a confidence score threshold. This implementation is tailored for a single input image at a time. For batched inference one would have to adjust the input pipeline for batches. This entire procedure is self-contained, requiring only core TensorFlow functions and not the complete Object Detection API.

**Example 2: Using a TensorFlow Lite Model**

```python
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the TensorFlow Lite model
model_path = 'path/to/your/model.tflite' # Replace with actual path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image function, same as in Example 1
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640)) # Resize to model's input size (adjust accordingly)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image


# Run inference with TFLite
def run_inference_tflite(image_path):
    input_data = preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = {}
    for out_detail in output_details:
      output_data[out_detail['name']] = interpreter.get_tensor(out_detail['index'])

    return output_data


# Post process detections to extract bounding boxes and scores
def postprocess_detections_tflite(detections):
    boxes = detections['TFLite_Detection_PostProcess'][0].numpy()
    scores = detections['TFLite_Detection_PostProcess:1'][0].numpy()
    classes = detections['TFLite_Detection_PostProcess:2'][0].numpy().astype(np.int32)
    num_detections = int(detections['TFLite_Detection_PostProcess:3'][0].numpy())
    
    valid_detections = []
    for i in range(num_detections):
        if scores[i] > 0.5: # Threshold for confidence
            valid_detections.append(
                {
                    "box":boxes[i],
                    "score":scores[i],
                    "class":classes[i]
                }
                )
    return valid_detections

# Example usage:
image_file = 'path/to/your/image.jpg' # Replace with actual path
output = run_inference_tflite(image_file)
valid_detections = postprocess_detections_tflite(output)
print(valid_detections)

```

This example shows inference using a TensorFlow Lite model. The core steps include loading the model, allocating tensors, preparing the input using the same `preprocess_image` function as before, invoking inference, extracting output tensors, and interpreting detections. The crucial difference here is the use of the `tf.lite.Interpreter` and the explicit handling of input and output tensors through their respective indices. Note, that output tensor names are specific to TFLite, and are extracted from the `output_details`. The `postprocess_detections_tflite` is slightly modified to accommodate these output tensor names. The benefit of TFLite here is that it is more lightweight making it suitable for constrained hardware.

**Example 3: Extracting the Graph and Parameters from Checkpoints**

```python
import tensorflow as tf
import numpy as np
from PIL import Image


# Define the model architecture (must match the trained model)
def build_model(): # replace with actual model architcture 
  input_tensor = tf.keras.layers.Input(shape=(640,640,3))
  # Replace the example model with the architecture used to train the weights
  x = tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu')(input_tensor)
  x = tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)
  x = tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
  x = tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(100)(x)

  model = tf.keras.Model(inputs = input_tensor, outputs = x)
  return model



# Load weights from checkpoint file
checkpoint_path = 'path/to/your/model.ckpt-XXX' # Replace with actual path
model = build_model()
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()



# Preprocess image function, same as in Example 1
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 640)) # Resize to model's input size (adjust accordingly)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return tf.constant(image)


# Run inference with the loaded model
def run_inference_checkpoint(image_path):
    input_tensor = preprocess_image(image_path)
    detections = model(input_tensor)
    return detections


#Example usage
image_file = 'path/to/your/image.jpg' # Replace with actual path
detections = run_inference_checkpoint(image_file)
print(detections)
```

This example illustrates how to load a model’s trained weights directly from a checkpoint file. This approach demands a definition of the model architecture within the program using Keras layers or another model construction method. This method allows direct loading of weights, provided the architecture defined matches the model used for training. The `tf.train.Checkpoint` mechanism loads the checkpoint data into model layers, which are subsequently used for inference in the `run_inference_checkpoint` function. This method is particularly applicable if direct control over model architecture is needed and when no `SavedModel` or TFLite versions of the model are available. It also allows more flexibility if a custom model class was used, rather than the pre-defined architectures within the Object Detection API. Note that the `build_model` function here must reflect the architecture used in the pre-trained model.

For further exploration of the topics, I recommend consulting the TensorFlow documentation, which provides comprehensive details on SavedModels, TensorFlow Lite, and checkpoint handling. Specifically, review tutorials and guides on SavedModel exporting and importing, TensorFlow Lite conversion, and checkpoint management. Furthermore, research examples on implementing custom input pipelines and post-processing logic in TensorFlow and TensorFlow Lite. Technical reports and open-source projects that focus on deploying models without the full API can also be valuable learning resources. The TensorFlow model garden also contains code samples that show models can be used outside the API. This should provide a robust foundation for implementing object detection without relying on the Object Detection API.
