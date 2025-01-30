---
title: "Why does BERT inference take longer after ONNX conversion?"
date: "2025-01-30"
id: "why-does-bert-inference-take-longer-after-onnx"
---
The discrepancy in inference times between native PyTorch BERT models and their ONNX-converted counterparts, particularly the increased latency observed post-conversion, stems primarily from a shift in execution paradigms and optimization opportunities. Specifically, while PyTorch leverages dynamic computational graphs and optimized kernels tailored to its framework, ONNX operates within a more constrained, static graph environment often requiring runtime-specific adaptations that may not always yield superior performance without careful configuration and platform-specific optimization.

My experience, accumulated across several large-scale natural language processing (NLP) deployments, has consistently shown that the supposed performance gains of ONNX – portability and potential for hardware acceleration – often don't materialize without a thorough understanding of the underlying mechanisms and limitations. In essence, the conversion itself introduces new overheads, which, if not addressed, can nullify any performance benefits.

The core of this issue lies in several factors. Firstly, the inherent nature of ONNX graphs is static; that is, the graph structure must be defined completely at the time of export. PyTorch, on the other hand, utilizes dynamic graphs, allowing for greater flexibility in computations based on input. This dynamic behavior allows PyTorch to execute only the necessary operations. When a PyTorch model is exported to ONNX, its dynamic graph must be flattened into a static one. This flattening can lead to the inclusion of optional branches that might not always be needed, which ultimately contribute to longer inference times.

Secondly, the ONNX runtime often handles operations differently than PyTorch's native implementation. For example, while PyTorch might use custom CUDA kernels optimized for specific hardware (like NVIDIA GPUs), the ONNX runtime may rely on more generic implementations. This difference can become a bottleneck, particularly for complex operations involved in BERT’s architecture, such as attention mechanisms and matrix multiplications. While the ONNX ecosystem attempts to bridge this gap with hardware-specific execution providers (e.g., TensorRT for NVIDIA, OpenVINO for Intel), these often need explicit configuration and do not always match the performance of highly tuned, framework-specific kernels. The process of selecting and enabling the appropriate execution provider for specific hardware is crucial, but if this step is missed, the inference time will be greatly affected.

Thirdly, graph optimizations within the ONNX runtime are not always as robust as those present in PyTorch. PyTorch's Just-In-Time (JIT) compilation, combined with its well-defined and optimized operators, often yields very efficient code execution. ONNX relies on its own optimization passes, which might not recognize or exploit all of the opportunities for performance enhancement available within a specific model architecture. This can lead to redundant computations and suboptimal execution patterns.

Furthermore, the model export process itself can introduce complications. PyTorch models frequently rely on operations or custom functions that do not have direct ONNX equivalents. These are often exported as subgraphs or custom nodes, which the ONNX runtime needs to interpret. This introduces an additional parsing and execution overhead. Furthermore, the process of converting tensor data types in PyTorch to ONNX may add another computational cost if the ONNX backend must handle these conversions at runtime.

To illustrate, consider three simplified examples:

**Example 1: A simple matrix multiplication within a BERT layer**

```python
# PyTorch (simplified)
import torch
import torch.nn as nn

class SimpleMatMul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.matmul(x, torch.randn(x.shape[-1], 128))

model = SimpleMatMul()
input_tensor = torch.randn(1, 256) # Batch size of 1
# PyTorch inference (assume this takes 't' time)
with torch.no_grad():
    torch_output = model(input_tensor)


# ONNX export and inference (simplified)
import onnxruntime
torch.onnx.export(model, input_tensor, "matmul.onnx")
ort_session = onnxruntime.InferenceSession("matmul.onnx")
onnx_inputs = {'input.1': input_tensor.numpy()}
onnx_output = ort_session.run(None, onnx_inputs)[0] # Assume this takes 't + delta' time

```

*Commentary:* This example highlights the basic translation of an operation from a PyTorch module to ONNX. Even for a simple matrix multiplication, the ONNX runtime must manage the execution, potentially taking slightly longer ('t + delta') than the PyTorch equivalent. This is because PyTorch often has specifically optimised kernels for this calculation, particularly for GPUs. This difference becomes more pronounced with larger models and more complex computations.

**Example 2: A conditional layer (representing a simplification of BERT's masked attention)**

```python
# PyTorch
import torch
import torch.nn as nn

class ConditionalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(256, 128)
    def forward(self, x, condition):
         if condition:
             return self.linear1(x)
         else:
             return self.linear2(x)


model = ConditionalLayer()
input_tensor = torch.randn(1, 256)
condition_true = torch.tensor(True)
condition_false = torch.tensor(False)
# PyTorch inference (dynamic branch)
with torch.no_grad():
    output1 = model(input_tensor, condition_true)
    output2 = model(input_tensor, condition_false)


# ONNX export
torch.onnx.export(model, (input_tensor, condition_true), "conditional.onnx") # Note condition has to be tensor not python variable.
ort_session = onnxruntime.InferenceSession("conditional.onnx")

# ONNX Inference (static graph; two runs are forced.)
onnx_inputs_true = {'input.1': input_tensor.numpy(), 'input.2': condition_true.numpy() }
onnx_output_true = ort_session.run(None, onnx_inputs_true)[0]
onnx_inputs_false = {'input.1': input_tensor.numpy(),'input.2': condition_false.numpy()}
onnx_output_false = ort_session.run(None, onnx_inputs_false)[0]
```

*Commentary:* This example demonstrates how a conditional flow, simple in PyTorch, can become less efficient in ONNX. The ONNX export process needs to have a specified boolean tensor (not python boolean) at the time of export and it will flatten the computation of all possible conditions, in this case creating a graph that executes both the `linear1` and `linear2` calculations in both runs of ONNX inference. Depending on the ONNX optimisation passes, these computations may or may not be merged. The condition tensor, `input.2` is no longer controlling the flow of the graph, but rather dictates the input data.

**Example 3: Graph optimizations (simplified)**

```python
#PyTorch:
import torch
import torch.nn as nn
class Suboptimal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(100,200)
        self.lin2 = nn.Linear(200, 100)

    def forward(self, x):
        tmp = self.lin1(x)
        return self.lin2(tmp)

model = Suboptimal()
input_tensor = torch.randn(1,100)

with torch.no_grad():
    pytorch_out = model(input_tensor)


#Onnx
torch.onnx.export(model,input_tensor, "suboptimal.onnx")
ort_session = onnxruntime.InferenceSession("suboptimal.onnx")
onnx_inputs = {'input.1': input_tensor.numpy()}
onnx_output = ort_session.run(None,onnx_inputs)[0]
```

*Commentary*: This illustrates how certain model structure patterns can affect runtime. Depending on the runtime and the optimisations available, the ONNX backend may or may not combine linear layers if it is more performant to do so. The PyTorch just-in-time compiler may have different optimisations to handle this kind of operation, resulting in different inference time.

To improve inference performance with ONNX, I recommend investigating and implementing the following strategies: Firstly, identify and eliminate unnecessary operations within the graph during export (e.g., avoiding redundant casts or reshaping). Secondly, when available use appropriate hardware-accelerated execution providers, ensuring they are correctly configured and up-to-date, such as TensorRT. Third, consider graph optimization techniques offered by the ONNX runtime or its ecosystem. Finally, profile both PyTorch and ONNX inferences to pinpoint bottlenecks and specific areas requiring optimization. It is also worthwhile exploring model-pruning or quantization techniques on the base PyTorch model before ONNX conversion, to reduce model size and complexity.

Resources such as the official ONNX documentation, the PyTorch documentation, and community forums for model deployment can also provide additional insights and best practices to help mitigate the increased inference time that can be seen post-ONNX conversion.
