---
title: "How can I load a TensorFlow v1 saved model using Keras or TensorFlow v2?"
date: "2025-01-30"
id: "how-can-i-load-a-tensorflow-v1-saved"
---
TensorFlow v1's SavedModel format, while foundational, presents specific compatibility challenges when working with TensorFlow v2 or Keras, particularly concerning graph structures and session handling. Directly loading a v1 SavedModel into a v2 environment, without careful intermediary steps, typically leads to errors related to graph definitions or deprecated APIs. My experience migrating several legacy machine learning pipelines from v1 to v2 involved extensive troubleshooting in this exact area.

The key issue stems from TensorFlow v1's reliance on the concept of a computational graph defined within a session. When you saved a model in v1, it serialized not just the model's weights, but also the entire graph structure required for its operations. TensorFlow v2, however, moved towards a more eager execution model, where operations are executed immediately, and uses a different internal representation for computations. Keras, built on top of TensorFlow, adopts this v2 style of execution and model definition. Therefore, directly loading a v1 SavedModel into the Keras API or using the standard TensorFlow v2 SavedModel loading functionality will often fail.

The necessary bridge involves using the TensorFlow v1 compatibility module within v2, specifically accessing the functions found in `tf.compat.v1`. This module retains the functionality and graph-based concepts from v1, allowing us to first load the model within a v1-compatible context and then extract useful components, such as the computation graph or the model's signature, for integration with a v2 environment. The core process can be segmented into three main steps: loading the model within a v1 session, obtaining input/output tensors and their names, and then converting the model to be usable within the v2 or Keras execution framework, if necessary. Note, there is no direct one-to-one automatic conversion; manual interaction is required.

First, let's illustrate the process of loading a v1 SavedModel using `tf.compat.v1`. I've experienced issues where failing to explicitly create a graph and session leads to cryptic errors, so meticulous setup is crucial.

```python
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior() # Ensure v1 behavior

def load_v1_saved_model(model_dir):
    """Loads a v1 SavedModel and returns session, input & output tensor information."""
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        input_tensor_name = sess.graph.get_tensor_by_name("input_node:0") # Placeholder for your input node name
        output_tensor_name = sess.graph.get_tensor_by_name("output_node:0") # Placeholder for output node name
        return sess, input_tensor_name, output_tensor_name

# Example usage
model_directory = "./path/to/your/v1/saved_model"
sess, input_tensor, output_tensor = load_v1_saved_model(model_directory)

# Test inference - use your actual input data dimensions
input_data = np.random.rand(1, 100).astype(np.float32)
output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})
print("Output from v1 Model:", output_data)
sess.close()
```

Here, `tf.disable_v2_behavior()` ensures that the code is run within the v1 compatibility environment. We explicitly create a graph and a session. The `tf.saved_model.loader.load` loads the model from the specified directory.  I've highlighted the "input_node:0" and "output_node:0" string placeholders which *must* be replaced by actual names of your input and output tensor operations. These names can be inspected using `saved_model_cli` (refer to resource suggestions). This example shows how to load and perform a test inference using the loaded v1 session. You must identify the names of your input and output tensors beforehand.

The second code example focuses on integrating a Keras model *around* this loaded v1 graph. This technique avoids having to rewrite the v1 computation from scratch, but instead embeds it.

```python
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np
from tensorflow import keras

tf.disable_v2_behavior() # Ensure v1 behavior

def create_keras_wrapper(model_dir, input_tensor_name, output_tensor_name):
    """Creates a Keras model that wraps the v1 graph."""
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session(graph=graph)
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)

      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
      output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

      input_shape = input_tensor.shape.as_list()[1:] # Discard batch dimension
      output_shape = output_tensor.shape.as_list()[1:] # Discard batch dimension

    class V1Wrapper(keras.Model):
       def __init__(self, sess, input_tensor, output_tensor, **kwargs):
           super().__init__(**kwargs)
           self.sess = sess
           self.input_tensor = input_tensor
           self.output_tensor = output_tensor
       def call(self, inputs):
            output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: inputs})
            return output

    return V1Wrapper(sess, input_tensor, output_tensor)

# Example Usage
model_directory = "./path/to/your/v1/saved_model"
input_tensor_name = "input_node:0"
output_tensor_name = "output_node:0"

wrapped_model = create_keras_wrapper(model_directory, input_tensor_name, output_tensor_name)

# Test inference, input dimensions should be consistent
test_input = np.random.rand(1, 100).astype(np.float32)
v2_output = wrapped_model(test_input)
print("Output from V1 Model Wrapped in Keras: ", v2_output.numpy()) # Ensure .numpy() is used.

```

This code creates a `V1Wrapper` class, a Keras model that accepts input data and performs inference using the v1 session. This encapsulates the v1 graph’s inference logic within a Keras-compatible structure. The Keras `call` method executes the graph using the v1 session within the context of the Keras model. Crucially, note `test_input` must still be correctly formatted as v1 expects. The `input_shape` and `output_shape` are extracted and could be used for input type validations in further integrations if necessary. The `.numpy()` is essential to unwrap the tensor in the final output.

Finally, it's worth acknowledging that some complex v1 SavedModels might include custom operations or functionalities that do not translate easily. I encountered situations where certain custom layers in a v1 graph were defined within the python codebase associated with that model. In such cases, you need to ensure that these specific components are also loaded in the environment. While a full conversion to pure TensorFlow v2 is generally preferred, this solution lets you work *with* your old model in new environments.

```python
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

def extract_variables(model_dir):
    """Extracts variable values from a v1 SavedModel."""
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session(graph=graph)
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)

      all_variables = tf.global_variables() #  Get all variables
      var_values = sess.run(all_variables)

      var_dict = {} # Dictionary to store the variables and their values
      for var, value in zip(all_variables, var_values):
          var_dict[var.name] = value
      return var_dict

#Example usage
model_directory = "./path/to/your/v1/saved_model"
extracted_variables = extract_variables(model_directory)
print(f"Extracted {len(extracted_variables)} variables")
for name, value in extracted_variables.items():
    print(f"{name}: shape {value.shape}") # To understand data types and dimensions.

```

This example focuses solely on extracting the model's learned parameters. It retrieves all global variables and their associated values from the v1 SavedModel and organizes them into a dictionary keyed by variable names.  This is primarily useful when you do wish to rebuild the model in v2 natively but still need to initialize the weights from the old v1 model's learned state. This output can then be used in v2/Keras to initialize new layers with pretrained weights by iterating the dictionary.

For further understanding, consulting the TensorFlow documentation, specifically the `tf.compat.v1` module documentation, is highly recommended. The SavedModel documentation and the `saved_model_cli` command-line tool will prove invaluable. Resources dedicated to model version migration, especially those covering the transition from TensorFlow v1 to v2 are pertinent. Numerous blog posts detail individual migration strategies which may help, however, understanding the underlying principles as outlined here will be the most beneficial.
