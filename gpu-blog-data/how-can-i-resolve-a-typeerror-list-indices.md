---
title: "How can I resolve a 'TypeError: list indices must be integers or slices, not str' error when converting data using ONNX?"
date: "2025-01-30"
id: "how-can-i-resolve-a-typeerror-list-indices"
---
The "TypeError: list indices must be integers or slices, not str" error during ONNX data conversion commonly stems from misinterpreting the structure of the data being processed by the ONNX model's input. Typically, when converting data for an ONNX model, we're dealing with numerical tensors, which are multi-dimensional arrays. This error arises when I attempt to access a list or an array element using a string as an index, where an integer representing the element's position, or a slice to access a portion of the array, was expected. I've encountered this numerous times, especially when dealing with nested data structures that haven’t been properly flattened before feeding them into the ONNX runtime session.

The root cause usually lies within the preprocessing or postprocessing steps surrounding the ONNX model's invocation, not in the model itself. When a model is exported to ONNX, it explicitly defines the expected input shape and data types. If my code provides data with a structure different from what the ONNX model expects, Python throws this `TypeError` since it tries to use my input (incorrectly formatted string) as if it were an integer index when accessing the actual tensor data in NumPy, PyTorch or other similar libraries. The primary issue is either passing an array or nested structure when a flat array was expected, or accidentally using string keys to access the elements of the tensor instead of numerical indexes.

To resolve this effectively, I need to meticulously examine the ONNX model's input shape and data type, and then ensure my preprocessing step transforms the data to exactly match this definition. This involves not only confirming the correct dimensions, but also making sure the data is flattened, if that is what the model expects, and using numerical indices to interact with the data. I've found that debugging this often requires a careful comparison of the model's input shape metadata against the actual data shapes during the preprocessing, and also carefully observing how intermediate data structures are accessed in the code before being fed into the model.

Here's how I approach the problem in a practical context, demonstrating a few common scenarios:

**Example 1: Incorrect Access Method**

Assume the ONNX model expects an input tensor of shape `(1, 3, 224, 224)` (a single image with 3 channels and height and width of 224 pixels). My initial data is a nested dictionary, which is incorrect:

```python
import numpy as np
import onnxruntime

# Incorrect Input Data Structure
input_data = {
    "image": np.random.rand(3, 224, 224)
}

ort_session = onnxruntime.InferenceSession("model.onnx")
input_name = ort_session.get_inputs()[0].name

try:
    ort_inputs = {input_name: input_data}
    ort_session.run(None, ort_inputs)

except TypeError as e:
    print(f"Error: {e}") # This will print the error due to using the string key 'image'
    
# Corrected Code
try:
    input_data_corrected = np.expand_dims(input_data["image"],axis=0).astype(np.float32)
    ort_inputs_corrected = {input_name: input_data_corrected}
    ort_session.run(None, ort_inputs_corrected)
    print("Inference Success")
except TypeError as e:
    print(f"Error: {e}")
```

**Commentary:**

In this example, I initially attempted to use a dictionary, with "image" as a string key, as the input to the ONNX session. This immediately triggered the `TypeError`. The ONNX runtime expects a NumPy array of the specific shape, where elements are accessed using integer indexes to address specific dimensions.

The corrected code first accesses the NumPy array from the dictionary, and then uses `np.expand_dims` to add the batch dimension to fit the expected `(1, 3, 224, 224)` input shape required by the ONNX model. I'm also explicitly casting the input to `np.float32` in order to ensure the data types are consistent.

**Example 2: Missing Batch Dimension and Incorrect Data Shape**

Let's say I receive a single image as a NumPy array but it lacks the batch dimension. The model expects an input tensor `(1, 3, 128, 128)`, and I have a NumPy array of shape `(3, 128, 128)`:

```python
import numpy as np
import onnxruntime

# Incorrect Input Data Shape
input_data = np.random.rand(3, 128, 128).astype(np.float32)

ort_session = onnxruntime.InferenceSession("model.onnx")
input_name = ort_session.get_inputs()[0].name

try:
    ort_inputs = {input_name: input_data} # This will cause an error
    ort_session.run(None, ort_inputs)

except TypeError as e:
    print(f"Error: {e}")

# Corrected code
try:
    input_data_corrected = np.expand_dims(input_data, axis=0).astype(np.float32)
    ort_inputs_corrected = {input_name: input_data_corrected}
    ort_session.run(None, ort_inputs_corrected)
    print("Inference Success")
except TypeError as e:
    print(f"Error: {e}")
```

**Commentary:**

Here, while the input is a NumPy array, it misses the batch dimension. The ONNX runtime fails because it attempts to process the data as if the first dimension was the batch dimension and therefore the rest of the dimensions are incorrect. `np.expand_dims(input_data, axis=0)` adds a dimension of size 1 at position 0 making the shape of the input `(1, 3, 128, 128)`. This aligns the provided data with the expected tensor structure, and allows the program to successfully compute the inference

**Example 3: Data Not Flattened**

This situation arises with data structures like feature maps that need to be flattened before entering the ONNX model. Assume my input is a tensor of shape `(20, 5)` while the ONNX model expects a flattened tensor of `(100)`.

```python
import numpy as np
import onnxruntime

# Incorrect Input Data Shape
input_data = np.random.rand(20, 5).astype(np.float32)

ort_session = onnxruntime.InferenceSession("model.onnx")
input_name = ort_session.get_inputs()[0].name

try:
    ort_inputs = {input_name: input_data}
    ort_session.run(None, ort_inputs)

except TypeError as e:
    print(f"Error: {e}")


# Corrected Code
try:
    input_data_corrected = input_data.flatten().astype(np.float32)
    ort_inputs_corrected = {input_name: input_data_corrected}
    ort_session.run(None, ort_inputs_corrected)
    print("Inference Success")
except TypeError as e:
    print(f"Error: {e}")

```

**Commentary:**

Here, the input is a 2D array when the model expects a 1D array (flattened input). The `flatten()` method transforms the 2D array into a 1D array before it is passed as input to the ONNX session. The resulting flattened tensor now matches the expected shape and can be processed correctly.

In summary, to address the “TypeError: list indices must be integers or slices, not str” I need to:

1.  **Inspect the ONNX Model:** Use `ort_session.get_inputs()` to understand the input structure, including names, shape, and data types. This often reveals the shape that the ONNX model was expecting.
2.  **Correct Data Structure:** Transform input data into NumPy arrays with the correct shapes and data type before passing them to `ort_session.run()`. This usually involves reshaping or adding dimensions using `np.reshape()` or `np.expand_dims()`. Also ensure to properly flatten multidimensional arrays if required.
3.  **Data Type Consistency:** Explicitly cast input arrays to match the ONNX model’s expected data type (e.g., `np.float32`).
4.  **Debugging Iteration:** I often use `print(input_array.shape)` to inspect array dimensions as I pre-process the data, confirming that it matches the model’s expectations.

When working with ONNX, I typically refer to resources that provide a solid foundation in tensor operations and ONNX fundamentals. For a more in-depth study of NumPy's array manipulations, I consult the library’s official documentation. Similarly, the ONNX official documentation provides detailed information about model input and output requirements, and is valuable to understand the inner workings of the runtime. Also, examining the documentation for `onnxruntime` is essential for usage information of the inference session. These resources, collectively, aid in troubleshooting these issues effectively and efficiently.
