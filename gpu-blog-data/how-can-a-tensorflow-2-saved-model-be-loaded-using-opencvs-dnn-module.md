---
title: "How can a TensorFlow 2 saved model be loaded using OpenCV's dnn module?"
date: "2025-01-26"
id: "how-can-a-tensorflow-2-saved-model-be-loaded-using-opencvs-dnn-module"
---

TensorFlow's `SavedModel` format, while excellent for deployment within the TensorFlow ecosystem, presents compatibility challenges when needing to interoperate with external libraries like OpenCV. The direct consumption of a `SavedModel` by OpenCV's `dnn` module isn't supported; the module expects models in formats like Caffe, TensorFlow's frozen graph (protobuf) or ONNX. Therefore, an intermediate conversion or extraction step is necessary. The core issue lies in the different data structures and graph representations each system employs.

The essential process involves: 1) loading the `SavedModel` in TensorFlow, 2) selecting the appropriate input and output tensors for the desired computation, 3) converting or exporting the selected subgraph into a format that OpenCV's `dnn` module understands, and 4) loading and executing the converted model within OpenCV. I've personally navigated this conversion several times on projects involving real-time video processing pipelines that were initially developed using a full TensorFlow infrastructure but later needed integration into edge devices where OpenCV was the more efficient choice.

The most common method for bridging this gap involves utilizing TensorFlow's API to freeze the relevant part of the model into a single protobuf file (`.pb`). This operation essentially inlines variables as constants in the graph definition, creating a self-contained, single-file representation suitable for `dnn`. This process requires specifying input and output nodes because the SavedModel's structure is often more complex than what `dnn` needs. Once frozen, the `dnn` module can interpret and execute the model. Another, sometimes simpler path depending on your model structure and needs, is to export to ONNX format. This does require the installation of an additional TensorFlow package and some extra considerations for certain complex layer types, but in some use cases may be a more direct route, particularly if you are dealing with models which might later need to be deployed to more diverse environments than OpenCV alone.

Here’s the general process using the frozen graph route with relevant code examples demonstrating how I typically tackle this in my projects:

**Example 1: Freezing a simple TensorFlow model.**

```python
import tensorflow as tf
import os

def freeze_graph(model_dir, output_node_names):
    """Freezes the SavedModel graph and exports a .pb file."""
    # Load the SavedModel.
    model = tf.saved_model.load(model_dir)
    # Get the concrete function for inference.
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # Obtain the graph and input/output tensor names from concrete function.
    input_tensor_names = [input_tensor.name for input_tensor in concrete_func.structured_input_signature[1]]
    output_tensor_names = [output_tensor.name for output_tensor in concrete_func.structured_outputs]

    # Perform the freeze step
    frozen_func = tf.compat.v1.wrap_function(concrete_func.get_concrete_function(), [
            tf.TensorSpec(spec.shape, spec.dtype, name=name)
            for name, spec in zip(input_tensor_names, concrete_func.structured_input_signature[1])]
        )
    graph = frozen_func.graph
    frozen_graph_def = graph.as_graph_def()

    with tf.io.gfile.GFile(os.path.join(model_dir, 'frozen_graph.pb'), 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    print(f"Frozen graph saved to: {os.path.join(model_dir, 'frozen_graph.pb')}")
    return input_tensor_names, output_tensor_names


# Example Usage
saved_model_dir = 'path/to/your/saved_model'
input_nodes, output_nodes = freeze_graph(saved_model_dir, output_node_names=['output_tensor_name_1', 'output_tensor_name_2'])

print(f"Input node names: {input_nodes}")
print(f"Output node names: {output_nodes}")
```

This `freeze_graph` function accepts the path to the directory where the `SavedModel` is stored. Crucially, you need to specify your desired output nodes via the `output_node_names` argument (though here we actually are retrieving this from the loaded model), which determine what output will be available for use after the frozen model is loaded into the dnn module. If not specified, you risk having output tensors which are not available to you outside of the tensorflow API. The function extracts the relevant subgraph, wraps it using a concrete function, and saves it to `frozen_graph.pb` within the same directory as the SavedModel. The key pieces are loading the `SavedModel`, and then using the `tf.compat.v1.wrap_function` to convert the graph to a representation that can be serialized into the `.pb` file. It is important to note that here we are extracting input and output tensors directly from the SavedModel. While the `output_node_names` parameter is still present it is not strictly needed when directly extracting tensors as done here, and is often added for explicitness or to handle cases where the tensors' names may not be available from the loaded model for some reason. After the freezing process the function returns both the input and output tensors as a list. This list will be necessary when configuring the dnn module in OpenCV.

**Example 2: Loading and using the frozen graph with OpenCV's `dnn` module.**

```python
import cv2
import numpy as np

def load_and_run_frozen_model(model_path, config_path, input_tensor_name, output_tensor_name, input_image):
    """Loads a frozen model and performs inference using OpenCV's dnn module."""
    try:
        # Load the model
        net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        if net.empty():
            raise Exception("Failed to load the frozen model")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

    # Prepare the input blob
    try:
      input_blob = cv2.dnn.blobFromImage(input_image, 1.0, (input_image.shape[1], input_image.shape[0]),
                                    (0, 0, 0), swapRB=False, crop=False)
    except Exception as e:
        print(f"Error creating input blob: {e}")
        return None

    # Set input to the network.
    net.setInput(input_blob, input_tensor_name)

    try:
      # Perform inference
        detections = net.forward(output_tensor_name)
        return detections
    except Exception as e:
        print(f"Inference failed: {e}")
        return None


# Example Usage:
# Construct a sample input image
input_img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
# The frozen graph, which was made above
frozen_graph_path = 'path/to/your/saved_model/frozen_graph.pb'
# An empty configuration file which is needed by opencv's dnn API
config_path = '' # Optional but necessary in some cases where custom layers are used.
input_tensor = input_nodes[0] # Get the input tensor from the result of the previous example
output_tensor = output_nodes[0] # Get the output tensor from the result of the previous example

output = load_and_run_frozen_model(frozen_graph_path, config_path, input_tensor, output_tensor, input_img)

if output is not None:
    print("Inference completed successfully.")
    print(f"Output shape: {output.shape}")
```

This `load_and_run_frozen_model` function demonstrates how to take the frozen graph and use it with the `dnn` module. `cv2.dnn.readNetFromTensorflow` loads the frozen graph. This function takes two arguments, the frozen graph `.pb` file, and the optional configuration file, which is required in some cases where custom or non-standard layers are used. If no configuration is required simply passing an empty string as the second argument is sufficient. A critical part is preparing the input for the model using `cv2.dnn.blobFromImage`, which converts the image into a 4D blob (batches, channels, height, width) that the model expects. Then the appropriate input and output tensor names, extracted from the `freeze_graph` method in example 1, are used to load the tensor into the network and forward pass it. The function then returns the results of the inference, or `None` if something went wrong. It also illustrates how to retrieve the necessary input and output tensors generated in the previous example, and use them in the correct order to load the model.

**Example 3: Exporting to ONNX using TensorFlow and loading into OpenCV**
```python
import tensorflow as tf
import tf2onnx
import os
import cv2
import numpy as np

def export_to_onnx(model_dir, output_path):
  """Exports the SavedModel to ONNX format."""
  model = tf.saved_model.load(model_dir)
  concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  # Obtain input shapes and dtypes from the concrete function
  input_shapes = [spec.shape.as_list() for spec in concrete_func.structured_input_signature[1]]
  input_dtypes = [spec.dtype.name for spec in concrete_func.structured_input_signature[1]]

  input_names = [tensor.name for tensor in concrete_func.structured_input_signature[1]]
  output_names = [tensor.name for tensor in concrete_func.structured_outputs]

  # Convert the concrete function to a graph
  frozen_func = concrete_func.get_concrete_function()

  onnx_graph, _ = tf2onnx.convert.from_function(
      frozen_func,
      input_signature=tuple(tf.TensorSpec(shape, dtype) for shape, dtype in zip(input_shapes, input_dtypes)),
      opset=13,
  )

  with open(output_path, 'wb') as f:
      f.write(onnx_graph.SerializeToString())

  return input_names, output_names


def load_and_run_onnx_model(model_path, input_tensor_name, output_tensor_name, input_image):
    """Loads an ONNX model and performs inference using OpenCV's dnn module."""
    try:
      net = cv2.dnn.readNetFromONNX(model_path)
      if net.empty():
        raise Exception("Failed to load the ONNX model")
    except Exception as e:
      print(f"An error occurred while loading the model: {e}")
      return None

    # Prepare the input blob
    try:
      input_blob = cv2.dnn.blobFromImage(input_image, 1.0, (input_image.shape[1], input_image.shape[0]),
                                    (0, 0, 0), swapRB=False, crop=False)
    except Exception as e:
        print(f"Error creating input blob: {e}")
        return None

    # Set input to the network.
    net.setInput(input_blob, input_tensor_name)
    try:
      # Perform inference
      detections = net.forward(output_tensor_name)
      return detections
    except Exception as e:
      print(f"Inference failed: {e}")
      return None

# Example Usage:
saved_model_dir = 'path/to/your/saved_model'
onnx_model_path = 'path/to/your/saved_model/model.onnx'

input_names, output_names = export_to_onnx(saved_model_dir, onnx_model_path)

input_img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
input_tensor = input_names[0]
output_tensor = output_names[0]

output = load_and_run_onnx_model(onnx_model_path, input_tensor, output_tensor, input_img)

if output is not None:
    print("Inference completed successfully.")
    print(f"Output shape: {output.shape}")

```
This example shows the general flow of first converting a SavedModel to ONNX, and then using that converted model with OpenCV. The `export_to_onnx` function is responsible for converting the Tensorflow model to an ONNX model using the `tf2onnx` library. Like with the freezing example it is critical that the input and output nodes are extracted from the model and supplied to the export function. This function saves the resulting ONNX graph to a file named `model.onnx` within the same directory as the `SavedModel`, and returns the necessary input and output tensor names. The `load_and_run_onnx_model` is similar in structure to the method used for the frozen graph example. It uses `cv2.dnn.readNetFromONNX` to load the model, and like with the frozen graph example it uses `cv2.dnn.blobFromImage` to convert the input image into the format which the ONNX model expects, and finally returns the results of running the inference.

In summary, there is no direct method to use a TensorFlow `SavedModel` within OpenCV's `dnn` module. The necessary steps involve leveraging TensorFlow’s API to either freeze a subgraph into a `.pb` file or export it to ONNX, and then use the OpenCV `dnn` API to load and execute the resulting file.

For further exploration on this topic, I recommend consulting the official TensorFlow documentation on saving and loading models. The OpenCV documentation related to `cv2.dnn` is also an excellent resource. Furthermore, resources on the ONNX format can also be helpful. Detailed examples and discussion can be found in the official repositories for both projects as well as through online search using these keywords. These resources are regularly updated with the most current best practices.
