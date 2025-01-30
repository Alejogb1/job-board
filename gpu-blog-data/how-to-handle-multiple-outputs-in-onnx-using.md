---
title: "How to handle multiple outputs in ONNX using C++?"
date: "2025-01-30"
id: "how-to-handle-multiple-outputs-in-onnx-using"
---
The core challenge in handling multiple outputs from an ONNX model using C++ lies in understanding the underlying structure of the ONNX runtime's output tensor collection and appropriately accessing each individual tensor.  My experience debugging inference pipelines within a large-scale image processing system highlighted the importance of meticulous tensor indexing and type checking to avoid runtime errors.  Failure to correctly handle this often resulted in segmentation faults or incorrect inference results.

**1. Explanation:**

The ONNX Runtime, when executing an ONNX model with multiple outputs, returns a vector (or similar collection) of `OrtValue` objects. Each `OrtValue` encapsulates a tensor.  The critical step is identifying the output names defined within the ONNX model itself and then using these names to index into the returned vector.  Crucially, this indexing isn't simply numerical (0, 1, 2…); it relies on a mapping between the output name in the ONNX graph and the position in the `OrtValue` vector.  Therefore, prior knowledge of the ONNX model's output node names is absolutely necessary. This information is typically available through the model's graph definition, readily accessible via tools like Netron.

Furthermore, the type of each output tensor must be explicitly checked. ONNX supports a variety of data types (float32, int64, etc.), and attempting to access a tensor using an incorrect type will lead to errors.  The `OrtValue` object provides methods to query its type and access the underlying data, but these methods must be used carefully and according to the ONNX specification.  Ignoring this step frequently results in unexpected behavior, especially when dealing with less common data types like boolean tensors or string tensors.  Finally, memory management is crucial; ensure the tensors are appropriately handled to prevent memory leaks.  The ONNX Runtime handles the memory of the `OrtValue` objects, but the data *within* these objects must be handled according to the underlying data type.

**2. Code Examples with Commentary:**

**Example 1: Basic Multiple Output Handling**

This example demonstrates the fundamental process of retrieving multiple outputs by name and checking their types.

```cpp
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>

int main() {
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session(env, "path/to/model.onnx", sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> output_names = {"output1", "output2"}; // Names from ONNX model
    std::vector<Ort::Value> outputs;

    auto input_name = session.GetInputName(0); // Assuming one input
    auto input_shape = session.GetInputInfo(0)->GetShape();
    // ... Prepare input data ...
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size(), input_data.data());


    session.Run(Ort::RunOptions{nullptr}, {input_name, input_tensor}, output_names, &outputs);

    if (outputs[0].IsTensor() && outputs[1].IsTensor()) {
        auto output1_tensor = outputs[0].GetTensorMutableData<float>();
        auto output2_tensor = outputs[1].GetTensorMutableData<int64_t>();  //Check and cast correctly!

        auto output1_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto output2_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

        //Process output1_tensor and output2_tensor based on their shapes
        std::cout << "Output 1 shape: ";
        for (int i = 0; i < output1_shape.size(); ++i) {
            std::cout << output1_shape[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Output 2 shape: ";
        for (int i = 0; i < output2_shape.size(); ++i) {
            std::cout << output2_shape[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error: Output is not a tensor." << std::endl;
        return 1;
    }
    return 0;
}
```


**Example 2: Error Handling and Type Checking**

This example adds robust error handling and explicit type checking to prevent unexpected crashes.

```cpp
// ... (Includes and initializations as in Example 1) ...

try {
    session.Run(Ort::RunOptions{nullptr}, {input_name, input_tensor}, output_names, &outputs);

    if (outputs.size() != 2) {
        throw std::runtime_error("Unexpected number of outputs.");
    }

    if (!outputs[0].IsTensor() || !outputs[1].IsTensor()) {
        throw std::runtime_error("Output is not a tensor.");
    }

    auto type_info_1 = outputs[0].GetTensorTypeAndShapeInfo();
    auto type_info_2 = outputs[1].GetTensorTypeAndShapeInfo();

    if (type_info_1.GetElementType() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Output 1 is not a float tensor.");
    }

    if (type_info_2.GetElementType() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("Output 2 is not an int64 tensor.");
    }

    // ... (Access tensor data as in Example 1) ...

} catch (const std::runtime_error& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
}
```


**Example 3: Handling Variable Number of Outputs**

This demonstrates how to handle a scenario where the number of outputs is not known beforehand.

```cpp
// ... (Includes and initializations as in Example 1) ...

std::vector<std::string> output_names;
for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    output_names.push_back(session.GetOutputName(i));
}

std::vector<Ort::Value> outputs;
try {
    session.Run(Ort::RunOptions{nullptr}, {input_name, input_tensor}, output_names, &outputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a tensor." << std::endl;
            continue;  //Skip non-tensor outputs
        }
        auto type_info = outputs[i].GetTensorTypeAndShapeInfo();
        std::cout << "Output " << i << " - Type: " << type_info.GetElementType() << ", Shape: ";
        auto shape = type_info.GetShape();
        for (int dim : shape) std::cout << dim << " ";
        std::cout << std::endl;
        // Process the tensor based on its type.
    }
} catch (const Ort::Exception& e) {
  std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
  return 1;
}
```

**3. Resource Recommendations:**

ONNX Runtime documentation, specifically the C++ API reference.  A good understanding of linear algebra and tensor operations is also crucial.  Familiarity with C++ exception handling and memory management best practices is essential for robust code. Finally, a debugging tool capable of inspecting memory and variables during runtime is invaluable.
