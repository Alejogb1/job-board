---
title: "How can I obtain predictions from an ONNX model in Python?"
date: "2024-12-23"
id: "how-can-i-obtain-predictions-from-an-onnx-model-in-python"
---

Alright, let’s dive into the process of getting predictions from an ONNX model using Python. It's a task I've handled countless times, particularly during my stint building real-time object detection pipelines for autonomous vehicles, so I'm quite familiar with the nuances involved. We will steer clear of unnecessary jargon and get straight to the core of it.

The Open Neural Network Exchange (ONNX) format is designed for interoperability, allowing models trained in different frameworks (like PyTorch or TensorFlow) to be deployed using various runtime environments. This makes it extremely versatile, but you still need a reliable mechanism to execute these models and obtain predictions in Python. This is where libraries like `onnxruntime` come into play.

The core idea is straightforward: you load the ONNX model, prepare your input data to match the model's expected input structure, feed it into the inference session, and then process the output. Let’s break this down into manageable steps, backed by some code examples.

First, you'll need `onnxruntime` installed. If you haven't already, use pip: `pip install onnxruntime`. I'd strongly suggest using the `onnxruntime-gpu` package if you have access to a compatible nvidia gpu, as this can drastically speed up inference, especially with larger models.

Now, let's consider a typical scenario. Let's imagine you have an image classification model, exported to ONNX. This model expects a tensor as input and gives you probability scores for different classes as the output.

Here’s the first code snippet illustrating the general process:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

def predict_with_onnx(onnx_model_path, image_path):
    # 1. Load the ONNX model.
    ort_session = ort.InferenceSession(onnx_model_path)

    # 2. Get model input details
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape

    # 3. Preprocess the image
    img = Image.open(image_path).resize((input_shape[2], input_shape[3]))
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, (2, 0, 1)) # Channels first
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

    # 4. Run inference
    ort_inputs = {input_name: img_np}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 5. Get model output details
    output_name = ort_session.get_outputs()[0].name

    # 6. Process the output
    predictions = ort_outputs[0]

    return predictions

if __name__ == '__main__':
    # Example usage, assuming you have 'model.onnx' and 'test_image.jpg'
    model_path = 'model.onnx' # Replace with actual path
    image_path = 'test_image.jpg' # Replace with actual path
    predictions = predict_with_onnx(model_path, image_path)

    # Now you can interpret the predictions (e.g., argmax for the highest probability class)
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")
    print("Raw predictions:")
    print(predictions)

```

This snippet covers the basic flow. You load the model, prepare the input (resizing, normalizing if necessary, and adding a batch dimension), perform the inference, and then interpret the results. Note the importance of correctly transposing the image and ensuring the data type matches the model’s expectation. The `get_inputs()` and `get_outputs()` methods of the inference session are crucial for dynamically understanding what shapes and names are expected.

Now, let’s address a slightly more complex scenario, which I've encountered frequently: handling models with multiple inputs or outputs. Imagine you're dealing with a semantic segmentation model, which might take both an image and additional metadata, and then output a segmentation mask and a confidence map. Here’s how you might handle that:

```python
import onnxruntime as ort
import numpy as np

def predict_multimodal_onnx(onnx_model_path, image_data, metadata):
    # 1. Load the ONNX model.
    ort_session = ort.InferenceSession(onnx_model_path)

    # 2. Identify input names and shapes
    input_names = [inp.name for inp in ort_session.get_inputs()]
    input_shapes = [inp.shape for inp in ort_session.get_inputs()]

    # 3. Prepare inputs (assuming image and metadata input).
    #  Note: You'd need to adapt this based on your actual inputs.
    # Here, I am creating a random tensor for image data and metadata.
    # This should be replaced with actual preprocessed data.
    input_data_1 = np.array(image_data).astype(np.float32)
    input_data_2 = np.array(metadata).astype(np.float32)

    # Reshape the input data
    input_data_1 = input_data_1.reshape(input_shapes[0])
    input_data_2 = input_data_2.reshape(input_shapes[1])

    # 4. Run inference using a dictionary to map input names to data
    ort_inputs = {input_names[0]: input_data_1,
                  input_names[1]: input_data_2}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 5. Process the outputs
    output_names = [output.name for output in ort_session.get_outputs()]

    #6. Return a dictionary mapping output names to output data.
    output_dict = dict(zip(output_names, ort_outputs))

    return output_dict

if __name__ == '__main__':
    # Example usage
    model_path = 'multimodal_model.onnx'
    image_data = np.random.rand(1, 3, 256, 256)
    metadata = np.random.rand(1, 10)
    output_dict = predict_multimodal_onnx(model_path, image_data, metadata)

    # You can now access outputs by their names
    print(f"Output 1: {output_dict.get('output_mask')}")
    print(f"Output 2: {output_dict.get('confidence_map')}")
```

In this snippet, we retrieve input names and create a dictionary to pass the input data. This pattern is essential for handling models that have more than one input. The outputs are also handled similarly using their names to make accessing individual outputs easier.

Finally, consider a scenario where your input data is dynamic – for example, a time series. This means the input shape to your onnx model can vary. This is particularly relevant when dealing with models that process sequence data or audio data.

```python
import onnxruntime as ort
import numpy as np

def predict_with_variable_input(onnx_model_path, variable_length_data):
    # 1. Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

     # 2. Get input details
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    
    # 3. Prepare input data.
    # Assume that our data is 2D, but could have varied sequence length.
    input_data = np.array(variable_length_data).astype(np.float32)

    # 4. Infer the batch size if needed
    input_data = np.expand_dims(input_data, axis=0)

    #Check the shape for debugging.
    print(f"Input data shape:{input_data.shape}")


    # 5. Run Inference
    ort_inputs = {input_name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    #6. Process output
    predictions = ort_outputs[0]
    return predictions


if __name__ == '__main__':
    # Example usage.
    model_path = 'variable_length_model.onnx'  # Replace with your model path
    variable_data_1 = np.random.rand(50,10) # variable length sequence data.
    variable_data_2 = np.random.rand(100,10)  # another variable length sequence data.

    predictions_1 = predict_with_variable_input(model_path, variable_data_1)
    predictions_2 = predict_with_variable_input(model_path, variable_data_2)

    print(f"Predictions for sequence 1 : {predictions_1.shape}")
    print(f"Predictions for sequence 2 : {predictions_2.shape}")
```

Here, the core is that the batch dimension is inferred and added. Most ONNX models will take a fixed input size. However, using the approach described above can help with sequence data. The critical part is to understand whether the model is designed for such variability.

For a deeper understanding of ONNX, I would recommend the official ONNX documentation, particularly the section on runtime environment and usage. Also, the book *Programming PyTorch for Deep Learning* by Ian Pointer is a great resource for understanding tensors, which are fundamental when working with model inputs and outputs. Finally, the *Tensorflow Guide* for model exportation also helps in understanding how data needs to be reshaped in order to properly work with models.

These three examples offer a decent starting point. The details, as you will often find with real-world projects, often come down to the specific ONNX model you're working with. The crucial part is to always carefully inspect input and output shapes and names using `onnxruntime`'s methods before attempting inference. Remember, precision is paramount when dealing with machine learning models, and a thorough understanding of your data flow will help avoid common pitfalls.
