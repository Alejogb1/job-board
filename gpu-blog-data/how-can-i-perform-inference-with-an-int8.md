---
title: "How can I perform inference with an INT8 quantized TensorFlow Lite model in Python?"
date: "2025-01-30"
id: "how-can-i-perform-inference-with-an-int8"
---
Quantization, particularly INT8 quantization, dramatically reduces the size and computational cost of deep learning models, making them viable for resource-constrained environments. However, working directly with INT8 quantized TensorFlow Lite models requires a careful understanding of how the data needs to be processed before inference, as well as how the results need to be interpreted. Having implemented embedded vision solutions utilizing edge TPUs extensively, I've found a methodical approach is crucial for successful deployment.

**Data Preparation for INT8 Inference**

Unlike float32 models, INT8 models operate on integer values, specifically within the range of -128 to 127. This necessitates that the input data, typically normalized floating-point values derived from images or other sensor data, be converted to the appropriate integer representation. This conversion is achieved through a quantization process involving a scale factor and a zero-point. These values are typically embedded within the TensorFlow Lite model metadata. It is essential to extract these from the model during the loading procedure.

The scale factor represents the size of each "step" in the integer space, while the zero-point indicates the integer value representing the floating point value of 0. The mathematical relationship between a floating-point value, `float_value`, and its corresponding integer representation, `int_value`, is as follows:

`int_value = round(float_value / scale + zero_point)`

and

`float_value = (int_value - zero_point) * scale`

Prior to inference, each element of your input tensor needs to be quantized according to the per-tensor or per-axis scale and zero-point available in the model. Similarly, the output tensor will be populated with INT8 values, which must be dequantized to retrieve the original floating point values during inference. Failure to properly quantize and dequantize data will result in inaccurate or nonsensical inference outputs. This operation is sometimes referred to as affinization/deaffinization.

**Code Examples**

The following code examples demonstrates how to load, preprocess, perform inference, and post-process data with an INT8 quantized TensorFlow Lite model:

**Example 1: Basic INT8 Inference with Single Input**

This example details the essential steps for inferring with a model taking a single image as input.

```python
import tensorflow as tf
import numpy as np

def perform_int8_inference(model_path, input_data):
    """Performs inference with an INT8 quantized TensorFlow Lite model.

    Args:
        model_path: Path to the .tflite model file.
        input_data: A numpy array representing the input.

    Returns:
        A numpy array representing the dequantized output.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    # Quantize input data
    input_tensor = np.round(input_data / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()

    output_tensor = interpreter.get_tensor(output_details['index'])

    # Dequantize output data
    output_data = (output_tensor - output_zero_point) * output_scale

    return output_data


if __name__ == '__main__':
   # Generate some fake normalized input data between 0 and 1 for demonstration purposes.
    fake_input_shape = (1, 224, 224, 3)  # Typical image input
    fake_input_data = np.random.rand(*fake_input_shape).astype(np.float32)

    # Path to your quantized tflite model
    model_path = "quantized_model.tflite"
    
    # Assuming the model has compatible inputs.
    output = perform_int8_inference(model_path, fake_input_data)
    print("Inference Output:", output)
```
In this code, the `perform_int8_inference` function loads the TensorFlow Lite model and obtains the input and output details. It then quantizes the input data, sets the input tensor in the interpreter, invokes the inference, and retrieves the quantized output tensor. The output is then dequantized to its float32 representation and returned. Note that a placeholder model name `quantized_model.tflite` is used. A real model needs to be provided for this example to function correctly.

**Example 2: Batch Inference and Handling Multiple Outputs**

This example builds upon Example 1 to demonstrate batch inference with models that produce multiple outputs.

```python
import tensorflow as tf
import numpy as np

def perform_batch_int8_inference(model_path, input_batch):
    """Performs batch inference with an INT8 quantized TensorFlow Lite model.
    Handles the possibility of multiple outputs.

    Args:
        model_path: Path to the .tflite model file.
        input_batch: A numpy array representing a batch of input data.

    Returns:
        A list of numpy arrays, each representing a dequantized output.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details['quantization']

    # Quantize input data
    input_tensor = np.round(input_batch / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()

    output_list = []
    for output_detail in output_details:
      output_scale, output_zero_point = output_detail['quantization']
      output_tensor = interpreter.get_tensor(output_detail['index'])
      output_data = (output_tensor - output_zero_point) * output_scale
      output_list.append(output_data)

    return output_list


if __name__ == '__main__':
    # Generate a batch of fake normalized input data between 0 and 1 for demonstration purposes.
    batch_size = 4
    fake_input_shape = (batch_size, 224, 224, 3)
    fake_input_data = np.random.rand(*fake_input_shape).astype(np.float32)

    # Path to your quantized tflite model
    model_path = "quantized_multi_output_model.tflite"

    # Assuming the model has compatible inputs.
    outputs = perform_batch_int8_inference(model_path, fake_input_data)
    for i, output in enumerate(outputs):
      print(f"Inference Output {i}:", output.shape)
```
The key difference in this example is that the function processes the input in batches and iterates through the output details, dequantizing and returning a list of output tensors. This approach handles models that produce not one but many output tensors, for instance, in an object detection scenario where bounding boxes and class probabilities are produced in separate output tensors. As with the previous example, `quantized_multi_output_model.tflite` is a placeholder for the model.

**Example 3: Handling Per-Axis Quantization**

Some models use per-axis quantization, where the scale and zero-point vary along a specific axis of the tensor. This example illustrates how to deal with this scenario.

```python
import tensorflow as tf
import numpy as np

def perform_per_axis_int8_inference(model_path, input_data):
    """Performs inference with an INT8 quantized model using per-axis quantization.

    Args:
        model_path: Path to the .tflite model file.
        input_data: A numpy array representing the input data.

    Returns:
        A numpy array representing the dequantized output.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]


    input_quant = input_details['quantization_parameters']
    output_quant = output_details['quantization_parameters']


    input_scale = input_quant.get('scales')
    input_zero_point = input_quant.get('zero_points')
    input_axis = input_quant.get('quantized_dimension')

    output_scale = output_quant.get('scales')
    output_zero_point = output_quant.get('zero_points')
    output_axis = output_quant.get('quantized_dimension')

    # Quantize input data per-axis
    if input_axis is not None and input_scale is not None and input_zero_point is not None:

       input_tensor = np.round(input_data/input_scale[np.newaxis,:,np.newaxis, np.newaxis] + input_zero_point[np.newaxis,:,np.newaxis, np.newaxis]).astype(np.int8)

    else:
       input_tensor = np.round(input_data / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()

    output_tensor = interpreter.get_tensor(output_details['index'])

    # Dequantize output data per-axis
    if output_axis is not None and output_scale is not None and output_zero_point is not None:
      output_data = (output_tensor - output_zero_point[np.newaxis,:,np.newaxis,np.newaxis]) * output_scale[np.newaxis,:,np.newaxis,np.newaxis]
    else:
        output_data = (output_tensor - output_zero_point) * output_scale


    return output_data



if __name__ == '__main__':

    # Generate some fake normalized input data between 0 and 1 for demonstration purposes.
    fake_input_shape = (1, 3, 224, 224)  # Example with a channel-based quantization axis
    fake_input_data = np.random.rand(*fake_input_shape).astype(np.float32)

    # Path to your quantized tflite model
    model_path = "quantized_per_axis_model.tflite"

    # Assuming the model has compatible inputs.
    output = perform_per_axis_int8_inference(model_path, fake_input_data)
    print("Inference Output:", output)
```
This example checks if the model is quantized per-axis. If it is, then it broadcasts the scales and zero points along the specified axes before quantizing and dequantizing. Again, `quantized_per_axis_model.tflite` is a placeholder name, a functioning model is required for actual use. This is a common pattern in models, with the channel axis often being the one on which quantization varies.

**Resource Recommendations**

To further refine understanding and implementation of INT8 inference with TensorFlow Lite, I recommend consulting the official TensorFlow documentation and API references. Specifically, search the TensorFlow website for sections on quantization, TensorFlow Lite, and the `tf.lite.Interpreter` class. Additional resources can be found in papers published on model quantization. Look for papers that specifically discuss INT8 and integer quantization techniques.

Working with quantized models requires careful data handling, but these benefits of smaller footprint and higher performance justify the effort.
