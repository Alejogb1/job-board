---
title: "How do I export a frozen inference graph from a TensorFlow 2.x object detection model?"
date: "2025-01-30"
id: "how-do-i-export-a-frozen-inference-graph"
---
The critical aspect to understand when exporting a frozen inference graph from a TensorFlow 2.x object detection model lies in the distinction between the training-oriented SavedModel and the inference-optimized frozen graph.  My experience working on large-scale object detection projects for autonomous vehicle applications highlighted the performance discrepancies; training models often include unnecessary nodes and operations detrimental to inference speed and resource consumption.  Exporting a frozen graph directly addresses this issue.

**1. Clear Explanation:**

TensorFlow's SavedModel format, while excellent for model versioning and serving, often retains elements from the training process, such as optimizer states and training-specific variables.  These are superfluous during inference. A frozen graph, conversely, represents a single computational graph with all variables and weights embedded directly into the graph definition, eliminating the need for separate variable loading and significantly improving execution efficiency.  The process involves converting the SavedModel, which contains a graph definition and variables, into a single Protocol Buffer file (.pb) containing the optimized graph.  This involves using the `tf.compat.v1.graph_util.convert_variables_to_constants` function, a critical component from the TensorFlow 1.x era, still relevant and powerful in TensorFlow 2.x for this specific task.  The resulting `.pb` file can be deployed on various platforms, including embedded systems, with minimal dependencies.

**2. Code Examples with Commentary:**

**Example 1: Basic Frozen Graph Export**

This example demonstrates the fundamental steps for exporting a frozen graph, assuming you have a SavedModel already generated.  I encountered this scenario frequently when integrating pre-trained models into different projects.

```python
import tensorflow as tf

def freeze_graph(saved_model_dir, output_node_names, output_graph_path):
    """Freezes a TensorFlow SavedModel into a frozen graph.

    Args:
        saved_model_dir: Path to the SavedModel directory.
        output_node_names: Comma-separated string of output node names.
        output_graph_path: Path to save the frozen graph.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], saved_model_dir)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names.split(',')
        )
        with tf.io.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

# Example usage:
saved_model_directory = "path/to/your/saved_model" # Replace with your SavedModel path
output_nodes = "detection_boxes,detection_scores,detection_classes,num_detections" #  Adjust based on your model's output tensors.
frozen_graph_path = "frozen_inference_graph.pb"
freeze_graph(saved_model_directory, output_nodes, frozen_graph_path)
```

This function loads the SavedModel, converts variables into constants within the graph, and serializes the resulting graph to a `.pb` file.  Correctly identifying `output_node_names` is crucial; these are the names of the TensorFlow operations providing the final detection results (bounding boxes, scores, classes, and counts).  Incorrect specification will lead to an incomplete or unusable graph.  I've learned this the hard way through numerous debugging sessions!


**Example 2: Handling Multiple Output Tensors**

Many advanced object detection models produce multiple outputs, often across different detection layers.  This example illustrates how to handle this complexity.

```python
import tensorflow as tf

def freeze_graph_multiple_outputs(saved_model_dir, output_node_names, output_graph_path):
    """Freezes a SavedModel with multiple outputs into a frozen graph."""
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], saved_model_dir)
        output_node_names_list = [node.strip() for node in output_node_names.split(',')]
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names_list
        )
        with tf.io.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

# Example usage:
saved_model_directory = "path/to/your/saved_model"
output_nodes = "detection_boxes,detection_scores,detection_classes,num_detections,feature_maps" #Example with additional output
frozen_graph_path = "frozen_inference_graph_multiple.pb"
freeze_graph_multiple_outputs(saved_model_directory, output_nodes, frozen_graph_path)

```

This is nearly identical to Example 1, but it explicitly handles a comma-separated string of output node names, making it more flexible for models with multiple output tensors.  This is important as many modern architectures use intermediate feature maps for various tasks.


**Example 3:  Optimizing for Specific Input Shapes**

To maximize inference speed, specifying the input tensor shape during the freezing process can improve performance.  This is especially relevant for embedded systems with limited memory.

```python
import tensorflow as tf

def freeze_graph_with_input_shape(saved_model_dir, output_node_names, output_graph_path, input_shape):
    """Freezes a SavedModel, specifying the input tensor shape."""
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], saved_model_dir)
        input_tensor_name = "image_tensor:0" # Adjust based on your model's input tensor name.
        input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        # Reshape the input tensor to the specified shape.  Requires knowing input tensor name
        input_tensor_reshaped = tf.reshape(input_tensor, input_shape)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names.split(','), input_tensor_reshaped
        )
        with tf.io.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

# Example usage:
saved_model_directory = "path/to/your/saved_model"
output_nodes = "detection_boxes,detection_scores,detection_classes,num_detections"
frozen_graph_path = "frozen_inference_graph_optimized.pb"
input_shape = [1, 640, 640, 3] # Example input shape: batch size, height, width, channels
freeze_graph_with_input_shape(saved_model_directory, output_nodes, frozen_graph_path, input_shape)

```

This example showcases the addition of input shape optimization. Note that finding the correct `input_tensor_name` is crucial and depends on your specific model.  Determining and utilizing the correct input shape significantly improved inference latency in my past projects, particularly on resource-constrained hardware.


**3. Resource Recommendations:**

The official TensorFlow documentation remains the most authoritative source for detailed explanations and advanced usage examples.  Consider consulting textbooks on deep learning and computer vision for a broader theoretical understanding.  Finally, exploring relevant research papers on optimized inference for object detection models offers valuable insights into cutting-edge techniques.  Thorough understanding of TensorFlow's graph manipulation tools is also essential.
