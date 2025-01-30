---
title: "How can I resolve the 'Missing input example' error when converting a PyTorch model to ONNX?"
date: "2025-01-30"
id: "how-can-i-resolve-the-missing-input-example"
---
The "Missing input example" error during PyTorch to ONNX conversion stems from the exporter's inability to infer the input tensor shapes and data types without explicit guidance.  My experience troubleshooting this within large-scale production deployments at a previous firm highlighted the crucial role of providing a representative input sample to the `export()` function.  This isn't merely a matter of convenience; the ONNX runtime requires concrete shape information for efficient execution.  Failure to provide this results in the error, preventing successful model deployment.  The solution lies in constructing a dummy input tensor mirroring the expected input to your model during inference.  This tensor needs to have the correct dimensions and data type.

**1. Clear Explanation:**

The ONNX exporter employs static analysis to determine the model's input and output characteristics.  Unlike PyTorch's dynamic computation graph, ONNX requires a static graph representation.  This static representation necessitates knowledge of the input tensor's shape and data type at the time of export. Without this information, the exporter cannot construct the complete graph.  The error message "Missing input example" directly indicates this missing piece of information.  Therefore, the primary solution involves providing a sample input tensor to the `torch.onnx.export` function using the `example_inputs` parameter.  This tensor should represent a realistic input your model will encounter during inference; its dimensions and data type will determine the corresponding attributes in the exported ONNX model.  Ignoring this requirement leads to an incomplete and unusable ONNX model.  Furthermore, using incorrect data types in the example input can also cause export failure, even if the shape is correct.  Thus, rigorous attention must be paid to both shape and type consistency.

**2. Code Examples with Commentary:**

**Example 1: Simple Linear Model**

```python
import torch
import torch.onnx

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
dummy_input = torch.randn(1, 10, dtype=torch.float32) # Crucial: Define dummy input with correct shape and type

torch.onnx.export(model, dummy_input, "linear_model.onnx", input_names=['input'], output_names=['output'])
```

*Commentary:* This example showcases a straightforward linear model.  The key is the creation of `dummy_input`, a tensor of shape (1, 10) and `torch.float32` type. This mirrors the expected input to the model, allowing for successful export.  The `input_names` and `output_names` arguments provide semantic clarity to the ONNX graph, enhancing readability and facilitating downstream integration.


**Example 2: Convolutional Neural Network (CNN)**

```python
import torch
import torch.onnx

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(16 * 16 * 16, 10) # Assuming 32x32 input image

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32) # Correct shape and type for a 32x32 image

torch.onnx.export(model, dummy_input, "cnn_model.onnx", input_names=['input_image'], output_names=['output'], opset_version=11)
```

*Commentary:*  This illustrates exporting a CNN.  The `dummy_input` now represents a single 32x32 RGB image (hence the shape (1, 3, 32, 32)).  The `opset_version` is explicitly set to ensure compatibility with the target ONNX runtime.  Incorrect specification here can cause unexpected behavior or export failures.  Careful consideration of the input shape is paramount, as CNNs are highly sensitive to input dimensions.


**Example 3: Model with Variable-Length Input Sequences**

```python
import torch
import torch.onnx

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) # Output from the last timestep
        return out

input_size = 10
hidden_size = 20
output_size = 5
seq_length = 25 # Example sequence length
model = RNNModel(input_size, hidden_size, output_size)
dummy_input = torch.randn(1, seq_length, input_size, dtype=torch.float32) # Batch size 1, variable sequence length

torch.onnx.export(model, dummy_input, "rnn_model.onnx", input_names=['input_sequence'], output_names=['output'], dynamic_axes={'input_sequence': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}})
```

*Commentary:*  This example demonstrates handling variable-length sequences in RNNs.  The crucial aspect here is the `dynamic_axes` argument.  It informs the exporter that the sequence length (`seq_len` in this example) is dynamic. The batch size is also marked as dynamic.  This allows the exported ONNX model to handle varying sequence lengths at inference time.  Without `dynamic_axes`, the exporter would assume a fixed sequence length, rendering the model inflexible.  The choice of `seq_length` is arbitrary; its value primarily influences the model's internal representation, not its general functionality.

**3. Resource Recommendations:**

The official PyTorch documentation on ONNX export.  A comprehensive guide on ONNX and its various operators.  A reference detailing best practices for deploying PyTorch models using ONNX.  These resources provide in-depth explanations of the export process, common issues, and advanced techniques for optimization and debugging.  Thoroughly reviewing these resources proved invaluable during my previous role, enabling successful large-scale deployment of complex models.
