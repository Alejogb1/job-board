---
title: "How can I obtain the shape and type of layers in an ONNX or PyTorch model?"
date: "2025-01-30"
id: "how-can-i-obtain-the-shape-and-type"
---
The ability to inspect the architecture of machine learning models, particularly the shapes and types of individual layers, is crucial for tasks such as model analysis, debugging, and efficient deployment. This information, while not always directly available, can be obtained through the frameworks' respective APIs. I've found this capability essential when optimizing model performance on custom hardware and diagnosing issues with intermediate tensors.

Let's delve into how this process works with ONNX and PyTorch models.

**ONNX Model Inspection**

The Open Neural Network Exchange (ONNX) format is designed to be an interoperable representation of machine learning models. This explicit structure allows for programmatic access to model components. The `onnx` library in Python provides the necessary tools for parsing and inspecting ONNX models.

The core concept relies on iterating through the graph structure of the ONNX model. Specifically, we examine the *nodes* which represent operations, the *inputs* to those nodes, and the *outputs* they produce. Each output has an associated value information object that holds the tensor's shape and data type. The graph is represented as a collection of these nodes, inputs, and outputs, which all have corresponding properties that can be accessed.

The process typically involves:

1. **Loading the ONNX Model:** The model, usually stored in a `.onnx` file, is loaded using `onnx.load`. This operation creates a ModelProto object.
2. **Accessing the Graph:** The `ModelProto` object contains a `graph` attribute, representing the computational graph.
3. **Iterating Through Nodes:** The `graph.node` attribute is an iterable list of NodeProto objects, each representing a layer.
4. **Inspecting Inputs and Outputs:** Each NodeProto object has an `input` and an `output` attribute. These are lists of strings, referring to the names of tensors.
5. **Resolving Value Information:** To get the shape and type, you have to refer to information contained in the `graph.value_info`, `graph.input`, or `graph.output` fields. These are lists containing the appropriate type information based on the name of an input or output of a node.

```python
import onnx
import numpy as np # used for type conversions

def inspect_onnx_model(onnx_path):
    """Inspects an ONNX model and prints layer shapes and types."""
    model = onnx.load(onnx_path)
    graph = model.graph
    value_info_map = {vi.name: vi for vi in graph.value_info + list(graph.input) + list(graph.output)}

    for node in graph.node:
        print(f"Layer: {node.name} (OpType: {node.op_type})")
        for input_name in node.input:
            if input_name in value_info_map:
                value_info = value_info_map[input_name]
                if value_info.type.tensor_type.HasField("shape"):
                  shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                  data_type = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[value_info.type.tensor_type.elem_type])
                  print(f"  Input: {input_name}, Shape: {shape}, Type: {data_type}")
                else:
                  print(f"  Input: {input_name}, Shape: Unknown, Type: Unknown")
            else:
                  print(f"  Input: {input_name}, Shape: Unknown, Type: Unknown")

        for output_name in node.output:
          if output_name in value_info_map:
                value_info = value_info_map[output_name]
                if value_info.type.tensor_type.HasField("shape"):
                  shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                  data_type = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[value_info.type.tensor_type.elem_type])
                  print(f"  Output: {output_name}, Shape: {shape}, Type: {data_type}")
                else:
                    print(f"  Output: {output_name}, Shape: Unknown, Type: Unknown")
          else:
                  print(f"  Output: {output_name}, Shape: Unknown, Type: Unknown")


# Example usage:
# Assuming model.onnx is in the same directory.
# inspect_onnx_model("model.onnx")
```
In the provided code, `inspect_onnx_model` takes the path of the ONNX file as its input, loading the model using `onnx.load`. It extracts the graph and iterates over each node, accessing input and output names. Value info is resolved using the `value_info_map`. A check is done to confirm the shape and type are present. A key detail here is that ONNX does not enforce that all tensors have dimensions assigned, so care is required to handle the edge case. Finally, the code extracts the tensor shapes and datatypes if available and prints all information to the console. If no shape information can be determined for a given tensor, this is indicated.

**PyTorch Model Inspection**

In PyTorch, inspecting model layer information is more dynamic as the model is actively constructed during runtime, unlike ONNX, which represents a static graph. We can inspect the model's structure using its methods and properties, particularly by iterating through its `named_modules()` or `named_children()` to discover its components.

1. **Model Definition:** This assumes we've constructed a PyTorch model using `torch.nn.Module`.
2. **Iterating Through Modules:**  The `named_modules()` method returns an iterator yielding module name strings and module objects.
3. **Accessing Parameter Shapes:** For layers with learnable parameters, we can access the `parameters()` or `named_parameters()` methods. We get an iterator for all the parameters contained in a module.
4. **Determining Input/Output Shapes Dynamically:**  Direct access to layer input/output shapes is not a standard property. To get these, a common method involves using a dummy tensor input. The input shape can be assumed to be the expected input shape of the model during inference, and the dummy input should have the same shape. By passing the input through the model, we can then inspect the output shape, as well as the intermediate outputs during debugging using a specific PyTorch hooks.
5. **Inspecting Layer Types:** The type of the module can be obtained with the `type()` function on the module object.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 16 * 16, 10) # output size to match dummy input

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # flatten the features for the linear layer
        x = self.fc(x)
        return x


def inspect_pytorch_model(model, input_shape):
    """Inspects a PyTorch model and prints layer shapes and types."""
    dummy_input = torch.randn(input_shape)
    
    with torch.no_grad():
        model_output = model(dummy_input)

    print("Model Summary:")
    for name, module in model.named_modules():
        if name == '':
            continue
        print(f"Layer: {name}, Type: {type(module).__name__}")

        # Print parameter shapes for layers with learnable params
        for param_name, param in module.named_parameters():
              print(f"  Parameter: {param_name}, Shape: {param.shape}, Type: {param.dtype}")

    print(f"Input Shape: {tuple(dummy_input.shape)}, Type: {dummy_input.dtype}")
    print(f"Output Shape: {tuple(model_output.shape)}, Type: {model_output.dtype}")
    
# Example Usage
model = SimpleModel()
input_shape = (1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 input
inspect_pytorch_model(model, input_shape)
```

In this PyTorch example, `inspect_pytorch_model` is defined to iterate through each layer, printing the name and the type of each layer. The output shape is determined by creating a dummy input tensor with the specified input shape and passing it through the network to determine the resulting shape. The parameter shapes and data types are found from each module, where applicable. Note: that a forward pass is required, so the input shape to the function must be suitable for the model.

**Specific Layer Shape Example**

Often, one needs to delve deeper into the shape of an intermediate layer, a common need when modifying or visualizing parts of a network. This can be achieved by a combination of the previous methods.

```python
import torch
import torch.nn as nn

class MultiBranchModel(nn.Module):
    def __init__(self):
        super(MultiBranchModel, self).__init__()
        self.branch1_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.branch2_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16*16*16 + 32*16*16, 10)

    def forward(self, x):
        branch1 = self.maxpool(self.relu(self.branch1_conv(x)))
        branch2 = self.maxpool(self.relu(self.branch2_conv(x)))
        branch1 = branch1.view(branch1.size(0), -1)
        branch2 = branch2.view(branch2.size(0), -1)
        merged = torch.cat((branch1,branch2),dim=1)
        output = self.fc(merged)
        return output

def inspect_intermediate_shape(model, input_shape, layer_name):
    """ Inspects shape of intermediate layer."""
    dummy_input = torch.randn(input_shape)

    # Hook to capture output of the specified layer.
    intermediate_output = None
    def hook(module, input, output):
        nonlocal intermediate_output
        intermediate_output = output

    hook_ref = None
    for name, module in model.named_modules():
      if name == layer_name:
        hook_ref = module.register_forward_hook(hook)
        break;

    if hook_ref == None:
      print(f"Layer {layer_name} not found.")
      return

    with torch.no_grad():
      model_output = model(dummy_input)

    hook_ref.remove()

    print(f"Layer: {layer_name}, Output Shape: {tuple(intermediate_output.shape)}, Type: {intermediate_output.dtype}")
    
# Example usage
model = MultiBranchModel()
input_shape = (1, 3, 32, 32)
inspect_intermediate_shape(model, input_shape, 'maxpool') # check pooled output
```

Here, we inspect the shape of `maxpool` layer in a model with two branches. The code uses PyTorch's forward hook mechanism to capture the output of a specific layer and then prints the shape. This demonstrates how to get a specific intermediate shape dynamically within a model. A key point here is to remove the forward hook after usage, otherwise it could cause memory problems.

**Resource Recommendations**

For a deeper understanding, I recommend exploring these resources:

* **ONNX Official Documentation:** The official ONNX documentation provides comprehensive details on the format's specification and API. Understanding the protocol buffer structure is beneficial for more advanced manipulations.
* **PyTorch Documentation:** The official PyTorch documentation provides exhaustive descriptions of `nn.Module` methods, including `named_modules()`, `parameters()`, and hooks for layer inspection.
* **Machine Learning and Deep Learning Textbooks:** These can provide theoretical context on model structure, tensor shapes, and the role of different layers in a network. The textbooks often include a section on implementation details within deep learning frameworks.

These resources, combined with hands-on experimentation, will enhance your proficiency in inspecting model architectures and leveraging this information for more effective model management.
