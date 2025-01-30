---
title: "How can I update a TensorFlow SavedModel's signature?"
date: "2025-01-30"
id: "how-can-i-update-a-tensorflow-savedmodels-signature"
---
TensorFlow SavedModels, once created, possess a defined signature that governs how inputs and outputs are structured. Modifying this signature directly after model creation is not a straightforward operation. The process involves essentially re-exporting the model with a new definition, often requiring careful navigation of the Graph and Metagraph structures. I've encountered this challenge multiple times in production deployments, particularly when evolving model input preprocessing or output post-processing logic. It’s not a simple “edit-in-place” procedure.

The core issue stems from the SavedModel's underlying representation. A SavedModel consists of a protobuf file (`saved_model.pb` or `saved_model.pbtxt`) along with other assets, including variables and checkpoints. The protobuf file defines the computational graph, containing the operations and tensors that form the model. The signature is a critical part of this graph's metadata. It acts as an interface defining what the model expects as input and what it produces as output, including names, data types, and tensor shapes. To update this signature, one does not directly modify the serialized protobuf; instead, one reconstructs the computational graph with the desired new signature and then exports it, overwriting the previous SavedModel or creating a new one.

The process typically involves loading the original SavedModel, identifying the target function for signature modification, and then using TensorFlow's low-level APIs to rebuild the graph while changing how that function is represented in the signature. This is more involved than one might expect, as you're essentially rewriting the model's interface, not just patching a few values. It's essential to understand that the SavedModel's internals operate through the concept of Concrete Functions (instances of `tf.function`). These functions are the units that are actually invoked during inference and are the actual components to which a signature is attached. Therefore, signature manipulation involves directly interacting with these functions.

Here is an illustration of the process using the TensorFlow Python API:

**Example 1: Modifying Input Shape**

```python
import tensorflow as tf

# Load the existing SavedModel
model_path = "path/to/original_model"
loaded_model = tf.saved_model.load(model_path)

# Access the Concrete Function for inference (replace 'serving_default' with your specific function name)
infer_fn = loaded_model.signatures['serving_default']

# Get the signature inputs and their spec
input_signatures = infer_fn.structured_input_signature

# Here we assume the input of interest is named 'input_tensor' with a batch size of 1 and spatial dimension of 10
# We want to change the spatial dimension to 15
input_signature_spec = input_signatures[0]['input_tensor'] # The input signature is a tuple of (args, kwargs) so use index 0 for args
new_input_shape = tf.TensorSpec(shape=(1, 15), dtype=tf.float32)
new_input_signature_spec = (new_input_shape,)

# Create the new concrete function
@tf.function(input_signature=new_input_signature_spec)
def new_infer_fn(input_tensor):
  # Convert to the correct original input dimensions, it will be re-cast upon signature re-export to reflect the new shape.
  old_input_tensor = tf.reshape(input_tensor, (1, 10))

  # Retrieve graph operation from loaded model for the rest of the flow.
  output_tensor = loaded_model.signatures['serving_default'](old_input_tensor)
  return output_tensor

# Construct the new saved model builder
builder = tf.saved_model.builder.SavedModelBuilder("path/to/new_model")

# Create the method definition for the new saved model
new_concrete_function = new_infer_fn.get_concrete_function()
method_definition = tf.saved_model.signature_def_utils.build_signature_def(
      new_concrete_function.structured_input_signature,
      new_concrete_function.structured_outputs,
      'new_serving_default' # Set the name of the new method signature
  )

# Add the new method definition to the graph
with builder as builder:
    builder.add_meta_graph_and_variables(
      loaded_model._v1_graph,
      [tf.compat.v1.saved_model.tag_constants.SERVING],
      {'new_serving_default': method_definition}, # Pass in the new signature
  )

# Save the new SavedModel
builder.save()
```

In this example, I have loaded an existing SavedModel, retrieved the concrete function responsible for inference using the 'serving_default' key, and extracted its input signature. I am then modifying the shape of one of the input tensors and defining a new concrete function for inference. The new concrete function, `new_infer_fn`, takes the re-shaped input and produces the original model's output.  Finally, I create a new SavedModel Builder and export the graph with a modified signature. Crucially, note that the original operations are reused from the old model, the input shape is re-defined for the new signature, and the graph is rewritten with the new function and signature.

**Example 2: Renaming an Input Tensor**

```python
import tensorflow as tf

model_path = "path/to/original_model"
loaded_model = tf.saved_model.load(model_path)

infer_fn = loaded_model.signatures['serving_default']
input_signatures = infer_fn.structured_input_signature

# Assume the input is named 'old_input_name' and we want to rename it to 'new_input_name'
old_input_spec = input_signatures[0]['old_input_name']
new_input_spec = tf.TensorSpec(shape=old_input_spec.shape, dtype=old_input_spec.dtype, name='new_input_name')
new_input_signature_spec = {'new_input_name': new_input_spec}

@tf.function(input_signature=[new_input_signature_spec])
def new_infer_fn(**kwargs):
  # Retrieve the old input tensor from the key, and pass it to the original model with the old key.
  old_input_tensor = kwargs['new_input_name']
  output_tensor = loaded_model.signatures['serving_default'](old_input_tensor)
  return output_tensor

builder = tf.saved_model.builder.SavedModelBuilder("path/to/new_model")
new_concrete_function = new_infer_fn.get_concrete_function()
method_definition = tf.saved_model.signature_def_utils.build_signature_def(
      new_concrete_function.structured_input_signature,
      new_concrete_function.structured_outputs,
      'new_serving_default'
  )

with builder as builder:
    builder.add_meta_graph_and_variables(
      loaded_model._v1_graph,
      [tf.compat.v1.saved_model.tag_constants.SERVING],
      {'new_serving_default': method_definition},
  )

builder.save()
```

In this case, we are not changing shapes, but rather the name of the input tensor itself.  The process remains similar, where we construct a new concrete function that takes the renamed input tensor, passes it to the original function using the old name internally, and then returns the output. This example highlights that even seemingly simple changes require reconstructing and re-exporting the model’s graph to register the signature modification.

**Example 3: Changing Output Types**

```python
import tensorflow as tf

model_path = "path/to/original_model"
loaded_model = tf.saved_model.load(model_path)

infer_fn = loaded_model.signatures['serving_default']
output_signatures = infer_fn.structured_outputs

# Assume the output is a float32 and we want to cast it to int64.
# Assume the output is keyed 'output_tensor'
old_output_spec = output_signatures['output_tensor']
new_output_spec = tf.TensorSpec(shape=old_output_spec.shape, dtype=tf.int64)
new_output_signature_spec = {'output_tensor': new_output_spec}


@tf.function
def new_infer_fn(input_tensor):
  # Retrieve the output from the original model, and cast to int64.
  output_tensor = loaded_model.signatures['serving_default'](input_tensor)['output_tensor']
  casted_output = tf.cast(output_tensor, tf.int64)
  return {'output_tensor': casted_output}


builder = tf.saved_model.builder.SavedModelBuilder("path/to/new_model")
new_concrete_function = new_infer_fn.get_concrete_function(tf.TensorSpec(shape=(1,10),dtype=tf.float32)) # We need a dummy input here to compute the new output_signatures
method_definition = tf.saved_model.signature_def_utils.build_signature_def(
      new_concrete_function.structured_input_signature,
      new_concrete_function.structured_outputs,
      'new_serving_default'
  )

with builder as builder:
    builder.add_meta_graph_and_variables(
      loaded_model._v1_graph,
      [tf.compat.v1.saved_model.tag_constants.SERVING],
      {'new_serving_default': method_definition},
  )

builder.save()
```

In this final example, I demonstrate modifying the output type. I retrieved the initial output tensor's spec, defined a new spec with the desired `int64` data type and constructed a new function to perform the cast from the original `float32` type.  Similar to the previous examples, this change requires creating a new function and re-exporting the model with the updated signature.

**Resource Recommendations**

For further understanding of this process, I recommend reviewing the following resources. First, the TensorFlow documentation on SavedModel, particularly the sections covering `tf.saved_model.load`, `tf.saved_model.builder.SavedModelBuilder` and `tf.function`. Additionally, studying examples provided on the TensorFlow GitHub repository, specifically concerning SavedModel manipulation and function signatures, proves invaluable. Lastly, examination of the `tensorflow.python.saved_model` namespace will reveal internal mechanics which may deepen your grasp on the topic. It's crucial to familiarize yourself with the underlying graph structure and the concept of concrete functions when attempting these operations. Understanding how SavedModels are stored and loaded is pivotal for effective manipulation.
