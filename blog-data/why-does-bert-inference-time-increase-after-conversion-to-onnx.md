---
title: "Why does BERT inference time increase after conversion to ONNX?"
date: "2024-12-23"
id: "why-does-bert-inference-time-increase-after-conversion-to-onnx"
---

Alright, let’s talk about the seemingly paradoxical issue of BERT inference slowing down after an ONNX conversion. I’ve personally encountered this several times, particularly back when we were initially deploying some large-scale NLP models for a client project involving semantic analysis. We were hoping to streamline the inference process, expecting a speed boost from ONNX, but instead, we were met with increased latency. This experience led me down a bit of a rabbit hole, and I can share some insights that might be useful.

The core issue isn’t that ONNX is inherently slower; far from it. The problem arises from how ONNX optimizes (or sometimes, doesn't) a given model for specific hardware and execution environments. When we convert a model like BERT from, say, a PyTorch or TensorFlow representation to ONNX, we're essentially transforming its computation graph into a portable, intermediate format. This format, at its core, is designed to enable optimizations during runtime on various backends (like CPU, GPU, etc.). However, these optimizations aren’t magic; they need to be explicitly configured and, sometimes, we see bottlenecks instead.

Here's the breakdown of the most frequent culprits:

1.  **Graph Optimization Issues:** During the conversion to ONNX, the structure of the computation graph is transformed. Sometimes, the ONNX graph optimizers don’t always create the optimal execution plan for the targeted backend. In some cases, specific operations that might have been optimized in the original framework (e.g., fused layers in PyTorch) are unfolded or broken down into less efficient implementations within the ONNX graph. This can lead to increased overhead from more intermediate computations that wouldn't occur in the original framework.

2.  **Backend Inefficiencies:** Different ONNX runtime backends (e.g., ONNX Runtime, TensorRT, OpenVINO) vary greatly in their capabilities and optimizations. If the runtime used for inference isn't well-tuned to the target hardware or isn't fully optimized for the types of operations present in the BERT model, you might see poor performance. For example, a CPU-based backend might struggle with the sheer scale of matrix multiplications inherent in a BERT model. Even on GPUs, the degree to which the backend uses fused kernels, optimized memory access patterns, or tensor core capabilities can significantly affect performance.

3. **Data Type and Layout:** The data type used in the model and the data layout can dramatically affect performance. By default, many ONNX converters utilize the same datatype present in the training model. Often, models are trained using float32, and the same type is maintained after conversion. Moving to float16 precision (if hardware supports it) or even quantized int8 implementations can greatly accelerate things. However, this requires adjustments on both the ONNX model and the runtime. Layout, for instance, is the order in which dimensions are arranged, and how the backend prefers to process that arrangement also plays a big factor. An incorrect layout can result in data transposes that add overhead.

To illustrate this, consider the following scenarios and code examples:

**Example 1: Inefficient Layer Unfolding:** Suppose we have a fused layer in PyTorch that gets decomposed upon ONNX export.

```python
# Fictional PyTorch code snippet for a fused layer (for illustrative purposes)
import torch
import torch.nn as nn

class FusedLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Hypothetical, inefficient ONNX conversion result: separate operations
# Hypothetical pseudocode (not directly runnable)
# x = linear_layer_1(x)
# x = relu(x)
# x = linear_layer_2(x)

# The 'fused' operation is more efficient during computation within the pytorch framework
```

Here, the *fused* operation would be performed as a single kernel for increased efficiency during PyTorch computation, while the separate operations post ONNX conversion adds extra overhead.

**Example 2: Backend Optimization and Data Type:** Imagine converting BERT and running it on two different ONNX runtimes.

```python
# Python code (using ONNX Runtime) demonstrating data type sensitivity
import onnxruntime as ort
import numpy as np

# Assuming 'bert.onnx' and 'input_data' are predefined
# Initialize ONNX Runtime Session
session_fp32 = ort.InferenceSession('bert.onnx', providers=['CPUExecutionProvider'])
session_fp16 = ort.InferenceSession('bert.onnx', providers=['CPUExecutionProvider'],
                                  session_options=ort.SessionOptions(),
                                  preferred_execution_mode = ort.ExecutionMode.PARALLEL)

# Sample input data
input_data = np.random.rand(1, 512, 768).astype(np.float32) # Shape (batch, seq_len, hidden_dim)

# Inference with float32
input_name_fp32 = session_fp32.get_inputs()[0].name
output_name_fp32 = [output.name for output in session_fp32.get_outputs()]
output_fp32 = session_fp32.run(output_name_fp32, {input_name_fp32: input_data})

# Attempt to switch to float16 (implementation might vary per ONNX backend)
# Note, typically the ONNX model itself is converted to float16 or uses FP16 specific operations
input_data_fp16 = input_data.astype(np.float16)
input_name_fp16 = session_fp16.get_inputs()[0].name
output_name_fp16 = [output.name for output in session_fp16.get_outputs()]

#If the ONNX model isn't converted to float16 or uses FP16 specific layers, this run might fail
#Or it could automatically cast it back to FP32
output_fp16 = session_fp16.run(output_name_fp16, {input_name_fp16: input_data_fp16})

print(f"Output fp32: {output_fp32[0].shape}")
print(f"Output fp16: {output_fp16[0].shape}")

# In a real-world comparison, float16 can be significantly faster given proper backend and hardware support,
# even with the data type conversion included.
```

In this second example, we illustrate that by switching to a lower precision like float16 using session options, we can achieve considerable speedup if the ONNX Runtime and the underlying hardware support this. The actual implementation might involve model conversion to FP16. Additionally, the use of *execution mode* influences how ONNXRuntime processes the computations.

**Example 3: Layout Incompatibility:** Assume the model is laid out in channels-first format within a backend, but the inputs are provided in a channels-last order.

```python
# Hypothetical scenario showing impact of incorrect layout
import numpy as np

# Imagine the optimized backend expects shape like (batch_size, channels, height, width)
# but we provide the data as (batch_size, height, width, channels)
input_data_channels_last = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Without the appropriate backend knowledge, the following operations would
# occur under the hood, and would be sub-optimal:

input_data_channels_first = np.transpose(input_data_channels_last, (0, 3, 1, 2)) # transpose
# The transpose op can be expensive as the memory layout needs rearrangement
# In optimal scenario, the data should be provided in the layout the backend expects
```

In this third example, we’re highlighting the hidden costs of data reshaping. The need to transpose data to match an expected layout adds overhead, especially for large tensors.

In conclusion, the slower inference with ONNX after conversion isn’t an indictment of ONNX itself. Rather, it’s usually an indication of suboptimal graph optimizations, mismatches between the ONNX graph and the selected backend capabilities, and often, lack of proper consideration for data types and layouts. Addressing these points involves experimentation with different ONNX runtime backends, careful consideration of data type conversions (like using float16 or INT8 where supported), and an understanding of the backend's expected data layout conventions. For deep dives into these topics, I highly recommend consulting the documentation of ONNX runtime, and also reading books like "Deep Learning with Python" by Francois Chollet for understanding general concepts and "High-Performance Deep Learning" by Jason Brownlee for practical optimizations. You might also find value in research papers published by teams working on ONNX Runtime and other model serving solutions, as these papers often delve into specific areas of performance optimization. Remember to approach it not as a single issue but rather as an optimization problem, where identifying bottlenecks and trying different tuning approaches is the key to obtaining optimal inference performance.
