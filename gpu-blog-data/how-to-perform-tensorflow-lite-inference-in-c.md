---
title: "How to perform TensorFlow Lite inference in C++?"
date: "2025-01-30"
id: "how-to-perform-tensorflow-lite-inference-in-c"
---
TensorFlow Lite (TFLite) enables machine learning model inference on resource-constrained devices. My experience deploying various models on embedded systems has shown the critical importance of optimizing the inference process, and C++ offers fine-grained control for this. This response focuses on how to load a TFLite model, execute inference, and handle input/output data within a C++ environment.

**Core Process and Explanation**

At its heart, TFLite inference in C++ involves a straightforward sequence: loading the model, allocating tensors, preparing input data, running the inference, and processing output data. The TensorFlow Lite C++ API provides the necessary tools, primarily within the `tensorflow/lite.h` header file. This header exposes key classes like `Interpreter`, `FlatBufferModel`, and `Tensor`, which serve as the core components of the inference process.

The first step, model loading, typically involves creating a `FlatBufferModel` instance, which reads the `.tflite` model file from disk. The `FlatBufferModel` encapsulates the model's data structure. This model data is then passed to the `InterpreterBuilder`, which generates an `Interpreter` object. The `Interpreter` is the primary execution engine. Before inference, tensors must be allocated. This entails calling the `AllocateTensors()` method on the `Interpreter`, which allocates the memory required for all model inputs, outputs, and intermediate computations.

Preparing the input involves accessing the input `Tensor` object via the `Interpreter` using its index. Data is copied into the tensor's underlying memory buffer, respecting the tensor's data type (e.g., float, int8). The `TfLiteTensorType` enum provides type information. With the input data populated, inference is initiated by calling the `Invoke()` method on the `Interpreter`. After the `Invoke()` call, the output data can be retrieved from the output `Tensor` object, again using the tensor index. This retrieved data can then be processed according to the specific application.

**Code Examples and Commentary**

The following examples illustrate key aspects of the TFLite inference process in C++.

**Example 1: Basic Model Loading and Tensor Access**

```c++
#include "tensorflow/lite.h"
#include <iostream>

int main() {
    // 1. Load Model
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile("model.tflite");
    if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    // 2. Create Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return 1;
    }

    // 3. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // 4. Access Input and Output Tensors
    int input_tensor_index = interpreter->inputs()[0];
    int output_tensor_index = interpreter->outputs()[0];

    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);

    std::cout << "Input Tensor Type: " << input_tensor->type << std::endl;
    std::cout << "Output Tensor Type: " << output_tensor->type << std::endl;

    return 0;
}
```

This initial example focuses on model loading, interpreter creation, tensor allocation, and basic tensor information retrieval. The `FlatBufferModel::BuildFromFile()` loads the model, and an `InterpreterBuilder` constructs the `Interpreter` using a default `BuiltinOpResolver`, which handles standard TFLite operations. Error handling is included to ensure proper operation. Once created, the input and output tensor objects are accessed via their respective indices, which can be verified by examining the model metadata using tools such as the TFLite model analyzer. The example prints the data types of the accessed input and output tensors.

**Example 2: Input Data Population and Inference**

```c++
#include "tensorflow/lite.h"
#include <iostream>
#include <vector>

int main() {
    // Model Loading and Interpreter Setup (same as in Example 1)
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
        if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return 1;
    }

     if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // 1. Input Tensor Access and Preparation
    int input_tensor_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);

    // Assuming the input is a float array of shape {1, 224, 224, 3}
    int input_size = 1 * 224 * 224 * 3;
    std::vector<float> input_data(input_size, 0.5f); // Initialize with placeholder data

    // Copy Input Data to Tensor
    float* input_buffer = interpreter->typed_tensor<float>(input_tensor_index);
    std::memcpy(input_buffer, input_data.data(), input_size * sizeof(float));


    // 2. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Failed to invoke interpreter." << std::endl;
      return 1;
    }


    // 3. Output Data Access (not processed in this example)
    int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);


    std::cout << "Inference complete." << std::endl;

    return 0;
}
```

This example builds upon the previous one by populating input data and executing inference. The code first accesses the input `Tensor`.  It assumes a 4D float input tensor with dimensions `{1, 224, 224, 3}` (a common image input size).  It creates a `std::vector` of floats, initializes it, and then copies this data to the tensor's memory buffer using `std::memcpy`. The type of the tensor is explicitly used by  `typed_tensor`. The `Invoke()` method executes the inference after the input has been provided. Although not explicitly processing it, the code retrieves the output tensor for future usage. Error checking is included at each significant step.

**Example 3: Output Data Processing**

```c++
#include "tensorflow/lite.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Model Loading, Interpreter Setup, Input Population, and Inference (same as in Example 2)
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
     if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return 1;
    }

     if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    int input_tensor_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
     int input_size = 1 * 224 * 224 * 3;
    std::vector<float> input_data(input_size, 0.5f);

    float* input_buffer = interpreter->typed_tensor<float>(input_tensor_index);
    std::memcpy(input_buffer, input_data.data(), input_size * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Failed to invoke interpreter." << std::endl;
      return 1;
    }

    // 1. Output Tensor Access and Processing
    int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);

    // Assuming the output is a float array of shape {1, 1000}
    int output_size = 1000;
    float* output_buffer = interpreter->typed_tensor<float>(output_tensor_index);
    std::vector<float> output_data(output_buffer, output_buffer + output_size);


    // Find Index of Max Value
    int max_index = std::distance(output_data.begin(),
                                 std::max_element(output_data.begin(), output_data.end()));
    std::cout << "Predicted Class Index: " << max_index << std::endl;
    return 0;

}
```

This final example shows how to access and process the output from the inference.  The output tensor is accessed, assuming it is a 1D float array of size 1000, representing class probabilities for a classification task. The output data is copied to a `std::vector`, and the index of the largest value, representing the predicted class, is then identified using `std::max_element` and `std::distance`.

**Resource Recommendations**

For a comprehensive understanding of TFLite, refer to the official TensorFlow Lite documentation. The API reference for C++ is invaluable. Additionally, exploring examples within the TensorFlow repository can provide practical usage patterns. Benchmarking TFLite models using the available tools is also beneficial for performance analysis and optimization. The code examples provided here represent a foundational understanding, but further research will improve proficiency. Investigating memory management practices within the TFLite environment is essential for deployments on devices with constrained resources. Lastly, consider specialized hardware acceleration libraries or delegates to further reduce the inference latency and power consumption.
