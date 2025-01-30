---
title: "Why am I getting a 'need more than 1 value to unpack' error when using CoreMLtools with Keras?"
date: "2025-01-30"
id: "why-am-i-getting-a-need-more-than"
---
The "ValueError: need more than 1 value to unpack" error, when encountered while converting a Keras model to Core ML using coremltools, almost invariably points to an inconsistency between the model's output shape and the expectation of the conversion process or a subsequent usage pattern of the model within Core ML. This usually manifests itself when a Keras model has multiple output layers, and you're attempting to treat the conversion process or the model's output as if it only had one. My experience, troubleshooting issues during the deployment of an object detection model, confirms this is often a subtle error related to incorrect indexing after model conversion or a mismatch of output tensor shapes.

The crux of the issue lies in how Keras models can be structured to return multiple values – for instance, multiple detection boxes, object class probabilities, or a combination of other features. Core ML, on the other hand, expects a specific input/output signature. If your Keras model generates multiple outputs (each a separate tensor), the conversion process must explicitly recognize these and map them to a corresponding structure in the Core ML model definition. If, after converting, you attempt to interact with the Core ML output as though it were a single value or array, rather than a tuple (or dictionary) of outputs, you'll trigger the unpacking error. Similarly, if the Core ML prediction output is unpacked incorrectly in your Swift or iOS code, you will encounter this error. This incorrect "unpacking" is a failure to acknowledge and correctly handle multiple returned tensors.

Let’s break down the specific causes and how to address them. In Keras, it’s possible to have a model with multiple output layers defined using the Keras functional API. For example, if you are building a model to predict both bounding box coordinates and object classes, you would likely have distinct output layers for each type of prediction. Core ML needs to be aware of each of these output layers. The converter doesn't automatically merge them into a single output.

Here are three practical examples and commentary to illuminate this:

**Example 1: Model With Multiple Outputs (Correct Conversion)**

Consider a Keras model that outputs both a bounding box and a class probability:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import coremltools as ct
import numpy as np

# Define the input layer
input_layer = keras.Input(shape=(28, 28, 3))

# Convolution layers (simplified)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_layer)
x = layers.MaxPool2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPool2D(2, padding='same')(x)
x = layers.Flatten()(x)

# Output 1: Bounding box (4 coordinates)
bounding_box_output = layers.Dense(4, name='bbox_output')(x)

# Output 2: Class probability (e.g. 10 classes)
class_output = layers.Dense(10, activation='softmax', name='class_output')(x)


# Build the model with multiple outputs
model = keras.Model(inputs=input_layer, outputs=[bounding_box_output, class_output])


# Dummy input
dummy_input = np.random.rand(1,28,28,3).astype(np.float32)

# Convert the model to CoreML
coreml_model = ct.convert(model,
                         inputs=[ct.ImageType(name="image", shape=dummy_input.shape)],
                         outputs=[ct.TensorType(name='bbox_output'), ct.TensorType(name='class_output')]
                       )

# Save the converted model
coreml_model.save('multi_output.mlmodel')

# Verify the prediction (Illustrative - might need more pre/post processing)
# mlmodel_loaded = ct.models.MLModel('multi_output.mlmodel')
# prediction = mlmodel_loaded.predict({'image': dummy_input})

```

*   **Explanation:** This example demonstrates the crucial step of explicitly defining outputs in the `ct.convert` function. I create distinct `ct.TensorType` objects with the same names as the respective output layers of the Keras model. Failing to do so causes Core ML to misinterpret the structure, leading to the unpacking error at the output. When the coreml model is used, the prediction result should be treated as a dictionary whose keys correspond to the 'name' attribute given to the `TensorType` objects during conversion (in this case: `bbox_output` and `class_output`).
*   **Commentary:** Notice how each output has a `name` attribute, which I used in the conversion. This naming is vital for mapping between Keras output and coreml's prediction dictionaries. Without these, the conversion either fails, or more frequently, results in an incorrectly formed model.

**Example 2: Incorrect Unpacking (Core ML Output)**

This example illustrates the *incorrect* usage of a model with two outputs in Swift and the resulting error you might see at runtime in your application. The following code snippet is incorrect and exemplifies the unpacking error:

```swift
// Incorrect handling of CoreML output:
// This would cause a need more than 1 value to unpack
// Assumes you have the MLModel loaded in 'coreMLModel' variable.

// import CoreML

// func processImage(image: CVPixelBuffer){
    // let predictionOutput = try? coreMLModel.prediction(input: image)
    // guard let bbox = predictionOutput as? MLMultiArray else{
    //     print("Problem unpacking")
    //   return
    // }
    // print("bbox: \(bbox)")
// }

```

*   **Explanation:** This illustrative Swift code attempts to cast `predictionOutput` directly into an `MLMultiArray`. However, if the Core ML model has multiple outputs (like in example 1), `predictionOutput` would be a dictionary or tuple. This direct cast and assumed single value cause the unpack error when used.
*   **Commentary:** The key error here is that I assume a single output, and attempt to force the prediction result into a single type (`MLMultiArray`) when the prediction is actually a `Dictionary`, whose keys are the tensor names (i.e. `bbox_output`, `class_output` from example 1). This results in an attempt to unpack a single value when there isn't one, thereby generating the "need more than 1 value to unpack" error in Swift.

**Example 3: Incorrect Conversion (No Output Names)**

The following is an example of a *faulty* CoreML conversion which can also lead to this error:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import coremltools as ct
import numpy as np

# Define the input layer
input_layer = keras.Input(shape=(28, 28, 3))

# Convolution layers (simplified)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_layer)
x = layers.MaxPool2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPool2D(2, padding='same')(x)
x = layers.Flatten()(x)

# Output 1: Bounding box (4 coordinates)
bounding_box_output = layers.Dense(4)(x)

# Output 2: Class probability (e.g. 10 classes)
class_output = layers.Dense(10, activation='softmax')(x)


# Build the model with multiple outputs
model = keras.Model(inputs=input_layer, outputs=[bounding_box_output, class_output])


# Dummy input
dummy_input = np.random.rand(1,28,28,3).astype(np.float32)

# Convert the model to CoreML - FAULTY CONVERSION
coreml_model = ct.convert(model,
                         inputs=[ct.ImageType(name="image", shape=dummy_input.shape)]) # NO OUTPUT NAMES

# Save the converted model
coreml_model.save('multi_output_faulty.mlmodel')

# The model will still convert, but it may not be usable as desired
# or it may result in the unpacking error down the pipeline

```

*   **Explanation:** In this faulty conversion, I intentionally omitted the `outputs` argument in the `ct.convert` function. This omission will still complete the conversion process but will not correctly map Keras outputs to names that can be accessed later. The `prediction` of this model will likely cause the unpacking error.
*   **Commentary:** The `ct.convert` attempts to automatically discern the output structure; however, this is error-prone. The prediction dictionary keys are not mapped to named outputs and may not have the names you'd expect, which is especially important for correct usage downstream. Also, the order of the outputs is not guaranteed. You should **always** define output tensors explicitly to avoid misinterpretation of the model's output structure.

In summary, the "need more than 1 value to unpack" error signifies an attempt to treat multiple outputs as a single output. CoreML models with multiple output layers require meticulous handling in conversion and prediction processing. When converting, define each output tensor explicitly with a name, and when using the prediction in swift, unpack the prediction dictionary using the names defined in the conversion.

For further learning, I recommend consulting resources that cover best practices for Keras to Core ML conversion, including coremltools documentation and any documentation covering the specific use of multi-output models. Be sure to focus on sections that explain data input/output for Core ML, and pay specific attention to naming conventions. Documentation focused on using coreml models in Swift, especially around the `MLModel` class will also be beneficial.
