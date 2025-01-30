---
title: "How can BERT models be fully delegated to Mali GPUs?"
date: "2025-01-30"
id: "how-can-bert-models-be-fully-delegated-to"
---
The primary impediment to delegating full BERT model inference to Mali GPUs lies in the inherent incompatibility between the commonly used TensorFlow and PyTorch frameworks, which primarily target desktop-class GPUs, and the specialized architecture of mobile GPUs like Mali. These frameworks rely on CUDA or vendor-specific APIs like Apple's Metal, neither of which Mali directly supports. I've spent considerable time optimizing model deployment on edge devices, and the challenge of effectively using Mali for complex workloads like BERT is significant.

The issue breaks down into two core problems: the software stack and the memory architecture. First, the typical workflow involves training a BERT model using TensorFlow or PyTorch, producing a model representation that assumes an NVIDIA CUDA environment. Mali, being an embedded GPU, does not support CUDA. Furthermore, the APIs provided by ARM, the designers of Mali GPUs, are not drop-in replacements for these deep learning framework APIs. This necessitates significant transformations to the computational graph and the data movement between CPU and GPU. Second, the memory architecture of Mali differs significantly from desktop GPUs. Mali GPUs often have limited and fragmented memory, which needs to be managed explicitly. The large memory footprint of BERT and its intermediate tensors presents an additional hurdle. We can't simply copy the entire model and all activation tensors onto the Mali GPU, we must strategically load and unload tensors during the inference process.

To effectively delegate BERT inference, a multi-stage approach is often required. The initial step revolves around converting the model to a hardware-agnostic format. Then, we must utilize a runtime that can leverage the ARM Compute Library (ACL), or similar libraries. I've found that exporting the BERT model to an ONNX format is beneficial because it provides a standardized representation that is independent of the original framework. ONNX allows us to then use tools like the ONNX Runtime, or specific ARM inference engines, which can then interpret and execute the graph on Mali GPUs. This step requires careful optimization, not just translation. We need to be aware of the limitations of the ARM compute library, the data transfer between CPU and GPU, and manage the memory allocations carefully.

Here is a representative code example showing how I typically perform ONNX conversion with PyTorch:

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# Load a pre-trained BERT model and config
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', config=config)
model.eval()

# Dummy input for ONNX tracing
input_ids = torch.randint(0, config.vocab_size, (1, 128), dtype=torch.long)
attention_mask = torch.ones((1, 128), dtype=torch.long)

# Export the model to ONNX format
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "bert.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state', 'pooler_output'],
    dynamic_axes={'input_ids': {1: 'sequence_length'}, 'attention_mask': {1: 'sequence_length'}}
)

print("ONNX model exported successfully.")

```

This code first loads a pre-trained BERT model and sets it to evaluation mode. It creates a sample input tensor and an attention mask for tracing the computation graph. The `torch.onnx.export` method converts the PyTorch model to an ONNX format. The `export_params=True` option ensures model weights are included. Specifying the `opset_version` helps with compatibility across ONNX runtime implementations. Naming inputs and outputs helps when loading the ONNX model. Finally, `dynamic_axes` tells the export that the sequence length is variable. Once the ONNX model is exported, it can be further processed by the specific inference engine for the Mali GPU. This is a pivotal step for framework independence, a necessary prerequisite for hardware-specific optimizations.

The next phase concerns the implementation on the target device itself. The most common approach involves utilizing the ARM Compute Library. This often requires code that is tightly coupled to the ARM hardware. Here is a conceptualized example showing the loading and executing of an ONNX model, using hypothetical functions provided by a third-party inference runtime (this will vary greatly depending on the used library):

```c++
#include <iostream>
#include <vector>
#include "arm_runtime.h" // Placeholder for a specific ARM inference library

int main() {
    // 1. Load the ONNX Model
    ModelHandle model_handle;
    Status status = arm_runtime_load_model("bert.onnx", &model_handle);
    if (status != STATUS_SUCCESS) {
        std::cerr << "Error loading model." << std::endl;
        return 1;
    }

    // 2. Prepare Input Data
    std::vector<int64_t> input_ids = { /*... example integer input data...*/};
    std::vector<int64_t> attention_mask = { /*... example integer mask data ...*/};
    InputTensor input1 = {input_ids.data(), {1, 128}}; // Shape: [1, sequence_length]
    InputTensor input2 = {attention_mask.data(), {1, 128}};

    std::vector<InputTensor> inputs = {input1, input2};


    // 3. Run Inference
    std::vector<OutputTensor> outputs;
    status = arm_runtime_run_inference(model_handle, inputs, outputs);
     if (status != STATUS_SUCCESS) {
        std::cerr << "Error running inference." << std::endl;
        arm_runtime_unload_model(model_handle);
        return 1;
    }

    // 4. Process output data from outputs[0] and outputs[1]

    // Unload the model
    arm_runtime_unload_model(model_handle);

    std::cout << "Inference completed successfully." << std::endl;
    return 0;
}

```

This C++ snippet demonstrates a hypothetical API usage for loading and running inference on an ONNX model using a runtime on an ARM device. It’s a very simplified version of what's needed on a real implementation. In reality, the `arm_runtime` library interface will differ. The code initializes the inference by loading the model. The example then sets up the input tensors and uses a function to invoke the inference engine. The final step is to process the generated outputs and free the memory of the allocated model. Proper error handling is crucial for stable execution. Note that the input vector creation requires converting int64_t tensors, for example from Python based scripts output, to a C++ friendly data structure.

Finally, dealing with the inherent limitations of Mali memory requires careful memory management strategies. We can use techniques like tiling or loop fusion to minimize memory requirements and reduce memory read/write operations. The key here is to load a subset of the model parameters and input data into memory at any given time, perform computations with that subset, and then load the next subset. This requires partitioning the computation graph and orchestrating data movement. This level of optimization is typically performed within the inference runtime itself, but understanding this memory constraint is imperative for designing the runtime configuration.

```python
import onnxruntime as ort
import numpy as np

# Load the ONNX model
ort_session = ort.InferenceSession("bert.onnx")

# Dummy Input Data
input_ids = np.random.randint(0, 30000, (1, 128), dtype=np.int64)
attention_mask = np.ones((1, 128), dtype=np.int64)

# Run inference
input_feed = {'input_ids': input_ids, 'attention_mask': attention_mask}
outputs = ort_session.run(None, input_feed)


print("Output shapes:", [out.shape for out in outputs])


# Example usage of memory mapped output

output_shape = (1,128, 768) # last_hidden_state shape for example

# Assume that output is memmapped to a file "output.bin", in the previous c++ code,
# it could be memmapped in the same way the model and input data is managed.
memmapped_output = np.memmap("output.bin", dtype=np.float32, mode='r', shape=output_shape)

print("Output from memmapped file")
print (memmapped_output[0,0,:10]) # Print the first 10 entries of the first token

```

The Python code illustrates a basic usage of the ONNX runtime library, loading a converted ONNX BERT model.  The dummy input data is created with the correct shape and type. Crucially, it demonstrates how output can be memory mapped to a file. This is beneficial to prevent loading the whole output tensor into main memory, which can cause issues on embedded devices. The output can be then processed directly, minimizing data movement. This example demonstrates the flexibility of ONNX runtime and how it complements the C++ implementation by processing the results from memory mapping on the host CPU.

Resource recommendations for mastering this topic include the official ARM Compute Library documentation, which details the available primitives and optimization strategies. The ONNX documentation is an invaluable asset for understanding the standardized model format. Additionally, academic literature concerning embedded machine learning inference and model optimization is recommended. Look for research papers focused on resource-constrained deployment of large models, which often address similar challenges as those encountered with Mali GPUs. Understanding hardware specific limitations and choosing right tooling is often as important as mastering theory. These are the areas where I have focused my efforts for effective deployments.
