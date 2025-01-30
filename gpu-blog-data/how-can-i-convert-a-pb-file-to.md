---
title: "How can I convert a .pb file to a TFlight model without encountering the AttributeError: 'AutoTrackable' object has no attribute 'inputs'?"
date: "2025-01-30"
id: "how-can-i-convert-a-pb-file-to"
---
The 'AttributeError: 'AutoTrackable' object has no attribute 'inputs'' when attempting to convert a `.pb` (TensorFlow Protocol Buffer) file to a TFLite model often stems from the discrepancy between the graph structure expected by the TensorFlow Lite converter and the actual structure encoded within the provided `.pb` file. This typically indicates that the `.pb` file represents a SavedModel format or a model that uses TensorFlow's object tracking capabilities, which are not directly compatible with a straightforward GraphDef-based conversion to TFLite. The `AutoTrackable` object is a key component in SavedModels, responsible for managing dependencies and variables, but it lacks a direct `inputs` attribute as would be present in a simpler GraphDef structure. I've experienced this issue numerous times during various model optimization phases, and the solution invariably requires adjusting how the graph is loaded and processed before conversion.

The root cause is that the TensorFlow Lite converter, when directly reading a `.pb` file, assumes it's a raw GraphDef. However, if the `.pb` file contains a SavedModel, it encapsulates more than just the graph definition; it includes variable checkpoints, function definitions, and other metadata that the TFLite converter cannot interpret using the simple GraphDef approach. The `AutoTrackable` object is central to SavedModels, and its methods and properties handle this broader range of information. Attempting to access `inputs` directly on an `AutoTrackable` object results in the reported error.

To resolve this, we must employ the correct mechanisms for loading a SavedModel and extracting its computational graph in a format consumable by the TFLite converter. This usually means loading the SavedModel with `tf.saved_model.load`, extracting the concrete functions representing the graph's operations, and then using a `tf.compat.v1.wrap_function` to create a legacy-compatible function compatible with the TFLite conversion process.

Here's a breakdown of the process, accompanied by code examples and commentary, illustrating a successful conversion:

**Example 1: Basic SavedModel Loading and Concrete Function Extraction**

```python
import tensorflow as tf

# Assume 'path/to/saved_model' is the location of your .pb file
saved_model_path = 'path/to/saved_model'

# Load the SavedModel
try:
    model = tf.saved_model.load(saved_model_path)
except Exception as e:
    print(f"Error loading SavedModel: {e}")
    exit()


# Get the available concrete function signatures from the loaded model
signatures = model.signatures

# Print available signatures to inspect
print("Available signatures:", list(signatures.keys()))

# Choose the relevant signature to convert (replace 'serving_default' if needed)
if 'serving_default' in signatures:
    concrete_function = signatures['serving_default']
else:
    print("No 'serving_default' signature found. Check available signatures and select one.")
    exit()


# Print input and output details of the concrete function
print(f"Inputs: {concrete_function.structured_input_signature}")
print(f"Outputs: {concrete_function.structured_outputs}")


# At this point, concrete_function is a tf.ConcreteFunction object, representing a callable
# that can be converted.
```

*Commentary*: This example demonstrates the fundamental steps of loading a SavedModel using `tf.saved_model.load` and identifying the appropriate concrete function using signatures. The error you encountered occurs because we are not dealing with the graph def directly, but the SavedModel containing an `AutoTrackable` object that cannot be directly used in the conversion. This step is crucial for SavedModel loaded models. By inspecting the available signatures, you identify the callable functions within the saved model. The print statements help with understanding what signatures exist and help identify the relevant function used for model inference (usually called 'serving_default'). The concrete function represents a specific, callable instance of your graph, which can be converted.

**Example 2: Wrapping the Concrete Function for TFLite Conversion**

```python
# The previous code for loading a SavedModel is assumed to have executed

# Assuming we have a concrete function called `concrete_function` from the previous example.
# Wrap the concrete function
wrapped_function = tf.compat.v1.wrap_function(concrete_function.get_concrete_function(),
                                               concrete_function.structured_input_signature)


# Create the converter
converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapped_function])


# Perform the conversion
tflite_model = converter.convert()

# Optional save the model to a file.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Conversion to TFLite successful.")

```
*Commentary*:  This example extends the previous code by wrapping the concrete function with `tf.compat.v1.wrap_function`. This step creates a compatible format for `TFLiteConverter`. The `from_concrete_functions` method is used, passing a list of these wrapped functions. If `concrete_function` is a specific implementation, then providing the input shapes is not required. However, if you encounter an error, you may have to provide the input shapes explicitly by replacing `concrete_function.structured_input_signature` with something like `[tf.TensorSpec([1, 256, 256, 3], tf.float32)]`. The shape will need to match the expected input size of the graph. The `convert` method converts the wrapped function to TFLite format, ready for deployment. The resulting TFLite model is saved to a binary `.tflite` file for persistence.

**Example 3: Specifying Input Shapes for Dynamic Input Models**

```python

# The previous code for loading a SavedModel is assumed to have executed

# Assuming we have a concrete function called `concrete_function` from the previous example.

# If the concrete function signature has a tf.TensorSpec for specific input shapes, they can be used.
# If tf.TensorSpec is unavailable, you will need to define it manually. Example below:

#Example of explicitly defining input shapes for the graph.
input_shapes = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name='input')]

wrapped_function = tf.compat.v1.wrap_function(concrete_function.get_concrete_function(),
                                            input_shapes)


converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapped_function])

# Perform the conversion.
tflite_model = converter.convert()


# Save model to disk.
with open('model_dynamic.tflite', 'wb') as f:
  f.write(tflite_model)

print("Conversion to TFLite successful using specified input shapes.")

```
*Commentary*: This example demonstrates how to specify input shapes manually if the concrete function does not have a pre-defined signature or if you have to make sure the input shape for conversion is as expected. This scenario frequently occurs when dealing with models that have dynamic input shapes. Specifying `tf.TensorSpec` objects explicitly during the `wrap_function` call ensures the converter knows the expected tensor structure, which will remove ambiguity when generating the TFLite model. It is crucial to align these input shapes with the actual requirements of the loaded model, otherwise, errors will occur when using the generated TFLite model with different input shapes.

**Resource Recommendations:**

*   **TensorFlow Documentation on SavedModel:** Thoroughly reviewing the TensorFlow documentation on working with SavedModels is vital. Understanding the SavedModel format, the concept of `AutoTrackable` objects, and how concrete functions are structured is crucial.
*   **TensorFlow Lite Conversion Guide:** Consult the official guide on TFLite conversion. It provides in-depth information about using `TFLiteConverter` and how different input formats are processed. Specific pages on concrete function conversion will be highly beneficial.
*   **TensorFlow API Reference:** Regularly check the TensorFlow API reference for precise documentation on functions like `tf.saved_model.load`, `tf.compat.v1.wrap_function`, and `tf.lite.TFLiteConverter`. Keeping the versions in sync will also help with compatibility issues.

By carefully loading SavedModels and extracting the concrete functions as described, one avoids directly encountering the problematic `AutoTrackable` object and instead manipulates the underlying callable structure, making it compatible with the TensorFlow Lite converter. Applying the methodology outlined in these code examples should resolve the `'AttributeError: 'AutoTrackable' object has no attribute 'inputs'` encountered during TFLite conversion.
