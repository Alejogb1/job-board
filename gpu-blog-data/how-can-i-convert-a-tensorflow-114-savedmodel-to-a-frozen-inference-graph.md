---
title: "How can I convert a TensorFlow 1.14 SavedModel to a frozen inference graph?"
date: "2025-01-26"
id: "how-can-i-convert-a-tensorflow-114-savedmodel-to-a-frozen-inference-graph"
---

The discrepancy between training and deployment environments often necessitates converting TensorFlow models to formats suitable for efficient inference. Specifically, TensorFlow 1.x's SavedModel format, designed for comprehensive model saving, needs to be transformed into a frozen inference graph for optimized deployment. This process typically involves stripping the graph of training-specific operations and variables, yielding a streamlined format for execution. Having worked extensively with TensorFlow models in embedded systems, I've encountered this conversion challenge repeatedly and developed a robust workflow.

The fundamental issue arises from the fact that a SavedModel encompasses more than just the computation graph required for inference; it includes training artifacts, checkpoint data, and associated metadata. A frozen inference graph, on the other hand, consolidates all variable values into constants within the graph definition itself. This simplification removes the overhead of variable management during inference, improving performance, especially in resource-constrained environments.

The conversion process involves several key steps. First, the SavedModel must be loaded into a TensorFlow session. Then, using the session, the variable values must be extracted. Finally, these variable values are replaced with constants in the graph definition, and a new GraphDef containing the inference graph is saved.

Below I’ll illustrate this conversion with Python code examples. Keep in mind the context is specific to TensorFlow 1.14.

**Code Example 1: Loading and Inspecting the SavedModel**

The initial step involves loading the SavedModel and identifying the input and output tensor names. These tensor names are critical for subsequently feeding data and retrieving predictions from the inference graph.

```python
import tensorflow as tf
import os

def inspect_saved_model(saved_model_dir):
    """Inspects a SavedModel to identify input and output tensor names."""
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        graph = tf.get_default_graph()
        print("Available operations:")
        for op in graph.get_operations():
             print(op.name)

        input_tensor_names = [input.name for input in graph.get_operations() if 'input' in input.name and input.type == 'Placeholder'] # Specific to the names I used for input layers

        output_tensor_names = [output.name for output in graph.get_operations() if 'output' in output.name and output.type == 'BiasAdd'] # Specific to the output nodes of the neural nets I used

        print("\nInput tensor names:", input_tensor_names)
        print("Output tensor names:", output_tensor_names)
    return input_tensor_names, output_tensor_names

if __name__ == '__main__':
    saved_model_directory = './saved_model' # Replace this path to the directory where your SavedModel is located
    if os.path.exists(saved_model_directory) == False:
        print('saved_model folder missing')
    else:
        input_names, output_names = inspect_saved_model(saved_model_directory)
```
In this code, `tf.saved_model.loader.load` loads the SavedModel into a session. The `graph.get_operations()` call helps enumerate and filter operations within the graph, allowing us to identify input placeholders and output nodes. The `input_tensor_names` and `output_tensor_names` variables collect the full names of these tensors, which include the operation name and output index. These are printed to the console, and also returned as a tuple. These tensors will be used in the subsequent step. This function also prints all operations, which could help debugging when the input and output node names are not clear. The paths and tensor name filtering is specific to my experience but the approach is generally applicable.

**Code Example 2: Freezing the Graph**

Having loaded the SavedModel and identified the relevant tensor names, the next step involves freezing the graph. This process creates a GraphDef proto with constant variables.

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

def freeze_graph(saved_model_dir, output_node_names):
    """Freezes a SavedModel and saves it as a frozen graph."""
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        graph = tf.get_default_graph()

        output_nodes = [graph.get_tensor_by_name(name) for name in output_node_names] # Retrieve output tensors

        frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

        with open('frozen_inference_graph.pb', 'wb') as f: # write to a binary format protobuf file
            f.write(frozen_graph.SerializeToString())
    print('Graph frozen successfully.')

if __name__ == '__main__':
    saved_model_directory = './saved_model'  # Replace this path to your SavedModel
    if os.path.exists(saved_model_directory) == False:
        print('saved_model folder missing')
    else:
        input_names, output_names = inspect_saved_model(saved_model_directory)
        freeze_graph(saved_model_directory, output_names) # Use the identified output nodes
```
In this function, we again load the SavedModel into a session. Then, we obtain the output tensors by using the names retrieved in the previous step, calling `graph.get_tensor_by_name`. The core logic resides in `graph_util.convert_variables_to_constants`. This utility function transforms variables in the session's graph definition into constant nodes, using their current values in the session. The `output_node_names` parameter specifies which nodes should be preserved in the frozen graph, effectively trimming away parts of the graph irrelevant for inference.  Finally, the resulting frozen graph definition is saved to `frozen_inference_graph.pb`.  I have personally found this function to be extremely useful when debugging and profiling.

**Code Example 3: Verifying the Frozen Graph**

To ensure the frozen graph was generated correctly, it's helpful to load the graph and verify basic inference capability.

```python
import tensorflow as tf
import numpy as np
import os

def verify_frozen_graph(frozen_graph_path, input_node_names, output_node_names):
    """Loads and tests a frozen graph with a dummy input."""
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    with tf.Session(graph=graph) as sess:
        input_tensors = [graph.get_tensor_by_name(name) for name in input_node_names]
        output_tensors = [graph.get_tensor_by_name(name) for name in output_node_names]


        # Create dummy input data with correct shapes as indicated by the placeholders.
        input_data = []
        for t in input_tensors:
            input_data.append(np.random.rand(*t.shape.as_list()).astype(np.float32))



        feed_dict = dict(zip(input_tensors,input_data))
        predictions = sess.run(output_tensors, feed_dict=feed_dict)
        print("Output of the frozen graph", predictions)


if __name__ == '__main__':
    frozen_graph_filepath = 'frozen_inference_graph.pb' # Replace this path to the frozen graph file
    if os.path.exists(frozen_graph_filepath) == False:
        print('frozen_graph_file missing')
    else:
        saved_model_directory = './saved_model' # Replace this path to your SavedModel
        if os.path.exists(saved_model_directory) == False:
            print('saved_model folder missing')
        else:
            input_names, output_names = inspect_saved_model(saved_model_directory)
            verify_frozen_graph(frozen_graph_filepath, input_names, output_names)

```

Here, we read the frozen graph definition from the `.pb` file into a `GraphDef` object, then import this graph definition into a new TensorFlow graph.  A dummy input data using the same input tensor shapes as the placeholders is created and passed to the graph.  Finally, the output tensors are run using the dummy input, resulting in the print out of the prediction from the model, confirming basic inference works and all placeholders are correctly linked.

**Resource Recommendations**

For a deeper understanding of TensorFlow's graph manipulation, review the official TensorFlow documentation (version 1.x, as the process is different for 2.x) on SavedModel and graph freezing. The `tensorflow.python.tools.freeze_graph` tool, although used behind the scenes here, is also valuable to understand. Additionally, examination of the source code for `graph_util.convert_variables_to_constants` can provide insights into the inner workings of the freezing process. Furthermore, examples of frozen graphs can be found in many public machine learning code repositories. This function is usually the main component of converting a SavedModel to a frozen graph for deployment.
By integrating these resources with the practical examples provided, users can develop a comprehensive understanding of SavedModel conversion and ensure their models are primed for optimal inference performance.
