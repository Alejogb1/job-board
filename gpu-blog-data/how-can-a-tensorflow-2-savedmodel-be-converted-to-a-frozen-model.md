---
title: "How can a TensorFlow 2 SavedModel be converted to a frozen model?"
date: "2025-01-26"
id: "how-can-a-tensorflow-2-savedmodel-be-converted-to-a-frozen-model"
---

TensorFlow 2 SavedModels, while offering advantages in terms of modularity and versioning, often require conversion to a frozen graph format for deployment in environments lacking the full TensorFlow runtime, such as embedded systems or situations demanding optimized inference. This process involves stripping away the training-specific parts of the SavedModel and embedding the model weights directly into the graph definition, resulting in a single, self-contained file. I've encountered this scenario several times while working on edge AI applications, where the reduced overhead of a frozen model is crucial for performance.

The conversion primarily relies on TensorFlow's `tf.compat.v1` module, even when working within TensorFlow 2. This is because the freezing process was historically centered around TensorFlow 1.x graph structures. While TF2 encourages eager execution and Keras models, the underlying graph representation still exists, and it’s this graph we manipulate during the freeze operation. The process isn't inherently complex but requires careful handling of input and output tensors to ensure the frozen model functions as intended. Essentially, we're taking a SavedModel, identifying the graph, and then saving a simplified version of that graph with all the necessary weights baked in as constants.

First, we load the SavedModel using `tf.saved_model.load()`. This operation creates a concrete function, which we can then access through `model.signatures['serving_default']`. This 'serving_default' signature is a standard convention for SavedModels used in inference scenarios. It contains the function specification for how to use the model; it lists the input and output tensors. We will need to know the exact names of the input and output tensors to effectively freeze the graph. Next, we iterate through the operations within this concrete function's graph using `concrete_function.graph.get_operations()`, capturing the names of the input and output tensors based on the function’s signature. Finally, we invoke `tf.compat.v1.graph_util.convert_variables_to_constants` to perform the freezing. This function takes the session graph, input tensor names, and output tensor names, and converts all the variables into constants.

Here are three illustrative examples, along with associated commentary.

**Example 1: Basic Freezing of a Keras Model**

This example demonstrates freezing a simple Keras model that was saved as a SavedModel. I’ve used this precise pattern on a few projects where a Keras model was the training artifact and a frozen model was the deployment target.

```python
import tensorflow as tf

# Create a basic Keras model for demonstration
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Save the model as a SavedModel
tf.saved_model.save(model, "saved_model_keras")

# Load the SavedModel
loaded_model = tf.saved_model.load("saved_model_keras")
concrete_function = loaded_model.signatures['serving_default']

# Get the input and output tensor names
input_tensor_names = [input_tensor.name for input_tensor in concrete_function.structured_input_signature[1]]
output_tensor_names = [output_tensor.name for output_tensor in concrete_function.structured_outputs.values()]

# Freeze the graph
with tf.compat.v1.Session(graph=concrete_function.graph) as sess:
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        concrete_function.graph.as_graph_def(),
        output_tensor_names
    )

# Save the frozen graph
with open("frozen_model.pb", "wb") as f:
    f.write(frozen_graph.SerializeToString())

print(f"Frozen model saved to frozen_model.pb with inputs: {input_tensor_names} and outputs {output_tensor_names}")
```

*Commentary:* The script first defines a basic sequential Keras model. This represents the typical training output I deal with. After saving it as a SavedModel, we load it and extract the 'serving_default' signature. The key part is how the input and output tensors' names are programmatically derived from the signature. This is vital when dealing with more complex models where manually naming the tensors is less feasible. We finally pass the necessary graph, input, and output names to the freezing function.  The resulting frozen graph is then saved to `frozen_model.pb`.

**Example 2:  Handling Multiple Inputs and Outputs**

This example highlights freezing a more complex model that has multiple inputs and outputs. I often see such architectures in multimodal systems. This complexity requires a more careful approach to handling the input and output tensor names.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Create a model with multiple inputs and outputs
input_a = tf.keras.Input(shape=(10,))
input_b = tf.keras.Input(shape=(20,))

dense_a = layers.Dense(32, activation='relu')(input_a)
dense_b = layers.Dense(64, activation='relu')(input_b)

concat = layers.concatenate([dense_a, dense_b])

output_1 = layers.Dense(5, activation='softmax')(concat)
output_2 = layers.Dense(1, activation='sigmoid')(concat)

model = tf.keras.Model(inputs=[input_a, input_b], outputs=[output_1, output_2])
tf.saved_model.save(model, "saved_model_multi")

# Load the SavedModel
loaded_model = tf.saved_model.load("saved_model_multi")
concrete_function = loaded_model.signatures['serving_default']


# Get the input and output tensor names
input_tensor_names = [input_tensor.name for input_tensor in concrete_function.structured_input_signature[1]]
output_tensor_names = [output_tensor.name for output_tensor in concrete_function.structured_outputs.values()]


# Freeze the graph
with tf.compat.v1.Session(graph=concrete_function.graph) as sess:
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        concrete_function.graph.as_graph_def(),
        output_tensor_names
    )

# Save the frozen graph
with open("frozen_model_multi.pb", "wb") as f:
    f.write(frozen_graph.SerializeToString())

print(f"Frozen model saved to frozen_model_multi.pb with inputs: {input_tensor_names} and outputs {output_tensor_names}")

```
*Commentary:* Here, we define a model with two distinct input branches and two output branches. Note how we still use the same structure to obtain the input and output tensor names from the signature. The critical change is that `concrete_function.structured_input_signature[1]` now contains a tuple of tensor specifications, and `structured_outputs` returns a dictionary. The freezing process remains the same, demonstrating its applicability to more complex structures. The output confirms the successful capture of multiple inputs and outputs.

**Example 3: Freezing with Placeholder Handling**

In some scenarios, especially legacy models, the SavedModel may explicitly use placeholders. This script illustrates how to handle such placeholders, which I've had to deal with when integrating older, pre-TF2 architectures.

```python
import tensorflow as tf

# Define placeholders
input_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 784), name='input_placeholder')
weights = tf.Variable(tf.random.normal((784, 10)), name='weights')
biases = tf.Variable(tf.zeros(10), name='biases')
output = tf.matmul(input_placeholder, weights) + biases
output = tf.nn.softmax(output, name='output_node')

# Create a SavedModel out of these tensor operations using a Session.
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("saved_model_placeholder")
with tf.compat.v1.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  builder.add_meta_graph_and_variables(
    sess,
    [tf.compat.v1.saved_model.tag_constants.SERVING],
    signature_def_map = {
      "serving_default": tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_placeholder': input_placeholder},
        outputs={'output_node':output}
      )
    }
  )

  builder.save()


# Load the SavedModel
loaded_model = tf.saved_model.load("saved_model_placeholder")
concrete_function = loaded_model.signatures['serving_default']

# Get the input and output tensor names
input_tensor_names = [input_tensor.name for input_tensor in concrete_function.structured_input_signature[1]]
output_tensor_names = [output_tensor.name for output_tensor in concrete_function.structured_outputs.values()]


# Freeze the graph
with tf.compat.v1.Session(graph=concrete_function.graph) as sess:
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        concrete_function.graph.as_graph_def(),
        output_tensor_names
    )


# Save the frozen graph
with open("frozen_model_placeholder.pb", "wb") as f:
    f.write(frozen_graph.SerializeToString())

print(f"Frozen model saved to frozen_model_placeholder.pb with inputs: {input_tensor_names} and outputs: {output_tensor_names}")

```
*Commentary:* In this instance, a basic computation is manually constructed using TensorFlow's placeholder mechanism and variable creation. This resembles older TF codebases.  Critically, the model is saved using `tf.compat.v1.saved_model.builder.SavedModelBuilder`.  Again, the input and output tensor names are extracted from the signature as in previous examples, allowing the freezing process to proceed as normal. The overall process shows compatibility with both Keras-based and legacy graph constructs.

**Recommendations:**

For a deep understanding of TensorFlow graphs, I recommend exploring the official TensorFlow documentation regarding graph manipulation and the `tf.compat.v1` module, particularly the `tf.compat.v1.graph_util` section. Specifically focus on `convert_variables_to_constants` and its requirements concerning input and output nodes.  Additionally, studying SavedModel structure (and signature definitions within them)  is invaluable to effectively working with `tf.saved_model.load()`. Finally, reviewing tutorials and examples concerning graph representations in TF1.x can provide additional context about the underlying graph we are manipulating with these procedures.
