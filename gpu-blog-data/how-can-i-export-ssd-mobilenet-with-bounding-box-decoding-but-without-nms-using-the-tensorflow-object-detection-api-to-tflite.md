---
title: "How can I export SSD-MobileNet with bounding box decoding but without NMS using the TensorFlow Object Detection API to TFLite?"
date: "2025-01-26"
id: "how-can-i-export-ssd-mobilenet-with-bounding-box-decoding-but-without-nms-using-the-tensorflow-object-detection-api-to-tflite"
---

Transferring an SSD-MobileNet model, trained using the TensorFlow Object Detection API, to TFLite while retaining bounding box decoding but excluding Non-Maximum Suppression (NMS) requires careful manipulation of the model graph and understanding of the TFLite conversion process. The standard `TFLiteConverter` pipeline will include NMS by default when dealing with object detection models. This incorporation stems from the fact that the Object Detection API usually relies on NMS to clean up the raw predictions coming from the model before they become usable bounding boxes. My experience optimizing embedded vision pipelines has driven me to address this precise challenge numerous times, because in many situations, performing NMS at the application level after inference affords more flexibility and control over the suppression algorithm or allows to integrate that step into a more complex post-processing pipeline.

The core of the issue lies in identifying and isolating the portion of the graph containing the bounding box decoding logic prior to the NMS operation. The Object Detection API’s meta-architecture often embeds NMS within a more extensive post-processing sequence. Therefore, directly extracting the pre-NMS output necessitates a focused approach to graph traversal and manipulation. Crucially, the bounding box decoder’s output comprises two key parts: the raw predicted locations (usually represented as deltas relative to anchor boxes) and the corresponding classification scores. The objective is to freeze the model’s graph at this point. This means ensuring that the saved model contains only the operations up to bounding box decoding.

To accomplish this, we need to modify the export configuration of the Object Detection API model. The standard export script `export_tflite_graph_tf2.py` included in the Object Detection API library (and in general when working with this type of model) usually freezes the full graph including NMS. This can be circumvented by directly loading the saved model into TensorFlow and then using the `tf.compat.v1.graph_util.extract_sub_graph` function to extract the sub-graph representing just the raw predictions. This extracted sub-graph can subsequently be used to create a TFLite model, which lacks NMS.

The first step is loading the trained model's saved model into TensorFlow. This can be achieved through:

```python
import tensorflow as tf

saved_model_dir = "/path/to/saved_model"  # Replace with your actual path
model = tf.saved_model.load(saved_model_dir)

input_tensor_name = "serving_default_input_tensor"
output_tensor_name = "StatefulPartitionedCall:1"  # Often this corresponds to the output before NMS. Inspect the graph with TensorBoard to confirm.

frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess=tf.compat.v1.Session(),
    input_graph_def=model.graph.as_graph_def(),
    output_node_names=[output_tensor_name]
)

with tf.io.gfile.GFile("frozen_graph.pb", "wb") as f:
    f.write(frozen_graph_def.SerializeToString())
```

In this snippet, the first key detail is replacing `/path/to/saved_model` with the correct directory where the exported saved model is located.  Next, we load this saved model using `tf.saved_model.load()`. Identifying the output tensor name prior to NMS can be done by visualizing the saved model with Tensorboard or `netron`. The line containing `output_tensor_name = "StatefulPartitionedCall:1"` is a common, albeit not universal, identifier for the last layer before NMS in the Object Detection API. It's essential to visually inspect the graph to confirm this name, as it can vary based on specific configuration details and the framework version. Crucially, the line `output_node_names = [output_tensor_name]` determines the output we are keeping in the sub-graph. We then convert the graph definition to a frozen graph using `tf.compat.v1.graph_util.convert_variables_to_constants` ensuring all the variables are converted to constants, as required by TFLite, before saving it.

Following the freezing of the graph, we proceed to convert the frozen graph to TFLite using the `TFLiteConverter` as shown in this code:

```python
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file="frozen_graph.pb",
    input_arrays=[input_tensor_name],
    input_shapes={input_tensor_name: [1, 300, 300, 3]},  # Adjust the input shape
    output_arrays=[output_tensor_name],
)

tflite_model = converter.convert()
with open("model_without_nms.tflite", "wb") as f:
  f.write(tflite_model)
```

Here the crucial parameter is `input_shapes`. The input shape provided, in this case `{input_tensor_name: [1, 300, 300, 3]}` corresponds to a common shape for SSD-MobileNet, assuming a 300x300 RGB image and a batch size of one. This needs to be adapted to match the exact requirements of the specific model. If the exported saved model is a single-batch model, you can check the shape via the `model.signatures['serving_default'].inputs` in the first python code example. The output array name should match the output tensor name you specified before when freezing the graph, thus guaranteeing that the TFLite model outputs the pre-NMS raw predictions. Finally, the converted model is saved as `model_without_nms.tflite`.

When loading the model in TFLite, it's also helpful to verify the output shape, especially after manipulating the graph. The following code demonstrates this check:

```python
interpreter = tf.lite.Interpreter(model_path="model_without_nms.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
```

This code snippet initializes a TFLite interpreter using the `model_without_nms.tflite` file that was just created. Then it prints the input and output details, including their shapes and data types. By examining the `output_details`, one can verify that the TFLite model's output shape matches expectations. Specifically, the output shape will typically correspond to the size of the feature map before NMS is applied, along with some channels for the bounding box deltas and classification scores. If you expect that you have multiple output tensors, due to the network head having multiple layers (for classification and regression for example), you should double-check the `output_tensor_name` and ensure that it's the correct one and that it's a list of multiple output names.

For further understanding of graph manipulation, I highly recommend studying the official TensorFlow documentation on freezing graphs (`tf.compat.v1.graph_util`) and TFLite conversion (`tf.lite.TFLiteConverter`).  Additionally, examining the source code of the TensorFlow Object Detection API export scripts can offer insights into how the graphs are originally constructed and how they can be modified. Also, documentation about `tensorflow.saved_model` will help when it comes to inspecting and understanding exported models. Lastly, studying some example projects and code repositories utilizing the TensorFlow Object Detection API, especially those focusing on embedded deployments, offers great insights into the intricacies of exporting such models and manipulating their graphs.
