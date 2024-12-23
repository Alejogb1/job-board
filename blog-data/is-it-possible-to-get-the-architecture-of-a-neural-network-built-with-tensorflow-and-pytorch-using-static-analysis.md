---
title: "Is it possible to get the architecture of a neural network built with Tensorflow and Pytorch using static analysis?"
date: "2024-12-23"
id: "is-it-possible-to-get-the-architecture-of-a-neural-network-built-with-tensorflow-and-pytorch-using-static-analysis"
---

Let's unpack this question. The idea of extracting neural network architectures via static analysis from TensorFlow and PyTorch models is not a straightforward task, yet it's one that I’ve encountered a few times in my career, particularly when dealing with complex, legacy systems where the original model definition code has, shall we say, "gone missing". It’s a common pain point, and while a perfect, 100% reliable solution might be akin to chasing a unicorn, we can definitely get some very useful information.

First, let's clarify what we mean by "static analysis." We're talking about examining the code (or in this case, the serialized model representation) without actually executing it. This contrasts with dynamic analysis, which would involve running the model and observing its behavior, which is often easier for capturing architecture but less desirable in this scenario. Why not run the model? Well, there are multiple valid reasons; for instance, you might only have access to a saved model file, not the full source code used to generate it, or you might be working in a constrained environment that prohibits execution.

Now, the challenge here arises from how these frameworks store and represent the model architecture. TensorFlow and PyTorch both typically use a graph-like structure internally. While PyTorch tends to be more dynamically defined (i.e., the graph structure is inherently tied to how the code is executed), TensorFlow often uses static graphs, especially when utilizing GraphDef or SavedModel formats. However, even within the static nature of TensorFlow’s SavedModel, extracting an interpretable representation without execution isn't trivial. Neither framework stores architecture in a plainly human-readable format; they both optimize for efficient computation, not static inspection.

So, can we do it? Yes, but with caveats. We can glean a reasonable approximation by leveraging framework-specific tooling and libraries. It won’t always give you the *exact* structure as originally programmed, particularly when frameworks perform under-the-hood optimizations. However, we can extract key details like the layers, their types, and the connectivity between them.

Here are three approaches, demonstrated with code snippets, that I've found effective in past scenarios.

**Approach 1: TensorFlow’s `SavedModel` Inspection**

For TensorFlow models saved using the `SavedModel` format, the `tensorflow.saved_model.load` and related methods can help. This approach relies on the structure that tensorflow explicitly provides.

```python
import tensorflow as tf

def inspect_tf_saved_model(saved_model_path):
    """Inspects a TensorFlow SavedModel to extract layer information."""
    model = tf.saved_model.load(saved_model_path)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    for op in concrete_func.graph.get_operations():
        if op.type in ['MatMul', 'Conv2D', 'BiasAdd', 'Relu', 'MaxPool', 'AvgPool']: # Most common types
          print(f"Operation: {op.type},  Name: {op.name}, Inputs: {[input.name for input in op.inputs]}, Outputs: {[output.name for output in op.outputs]}")

# Example usage (assuming 'my_saved_model' is your SavedModel directory)
saved_model_path = 'my_saved_model'
inspect_tf_saved_model(saved_model_path)
```

This code snippet loads the `SavedModel`, accesses its concrete function representing the inference graph, and then iterates over all operations present in it, which are the equivalent to layers. This provides a basic but crucial view of the architecture, including the layer types and connectivity based on input and output tensors. This does not, however, provide parameter counts or detailed information beyond operation type. You would need to examine the `op.node_def` for details such as kernel size, strides etc.

**Approach 2: PyTorch's `torch.onnx.export` and ONNX**

PyTorch, although less inclined toward static representations in its default usage, can leverage the ONNX (Open Neural Network Exchange) format. Exporting a PyTorch model to ONNX allows us to view it in a graph format. While it requires a dummy input, it’s still essentially static analysis once the conversion is done.

```python
import torch
import torch.nn as nn
import torch.onnx

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*26*26, 10) # Assuming input 3x32x32 for demo

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x


def inspect_pytorch_model_onnx(model, dummy_input, onnx_path):
    """Exports a PyTorch model to ONNX and prints node information."""
    torch.onnx.export(model, dummy_input, onnx_path)

    import onnx
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
       print(f"Op Type: {node.op_type}, Name: {node.name}, Inputs: {[i for i in node.input]}, Outputs: {[o for o in node.output]}")


# Example usage
model = SimpleNet()
dummy_input = torch.randn(1, 3, 32, 32)  # Assuming 3x32x32 input
onnx_path = "my_model.onnx"
inspect_pytorch_model_onnx(model, dummy_input, onnx_path)
```

In this snippet, we create a simple `SimpleNet` model, export it to an ONNX format, and then parse that ONNX representation. By iterating through the ONNX nodes, which represent the layers and operations, we can get insights into the model architecture. Again, this is high-level and would require further inspection of node attributes for complete layer information. Note that the input dimensions given are based on the dummy input given and must match the model's expected input dimensions for a successful export to ONNX.

**Approach 3: Custom Model Class Inspection in PyTorch**

A third, more manual, approach (but one which I have used to good effect) is based on inspecting the model directly in pytorch. This technique requires that the model's code exists, at least the class definition; you cannot reconstruct the model without it.

```python
import torch
import torch.nn as nn

class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(128*8*8, 10)  # Assumed spatial dimensions after max pool
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def inspect_pytorch_model_class(model):
    """Inspects a PyTorch model class to extract layer information."""
    for name, module in model.named_modules():
        if name: # Skip the top-level module itself
          print(f"Module: {name}, Type: {type(module).__name__}")


# Example Usage
model = ComplexNet()
inspect_pytorch_model_class(model)
```

This method leverages `named_modules` to traverse the model's structure and identify layers within the model. This method is incredibly useful in combination with the others for verification. Since `named_modules` gives you access to each individual layer and allows you to directly query the object itself, it gives you more direct and detailed information.

**Resources and Further Reading**

For delving deeper into these topics, I recommend checking out the following resources:

*   **"Deep Learning with Python" by François Chollet:** While focused on Keras, the book's discussions of model architecture and graph structures are highly applicable to TensorFlow in general. It also offers a good foundation for understanding fundamental deep learning concepts.
*   **The official TensorFlow documentation:** Specifically, focus on the guides for `SavedModel` and working with computation graphs. The documentation provides precise information on how models are serialized and structured within TensorFlow, invaluable for this type of analysis.
*   **The official PyTorch documentation, particularly on `torch.onnx`:** Thoroughly reading the PyTorch docs surrounding ONNX export is crucial for understanding the limitations and capabilities of this specific technique.
*   **The ONNX documentation**: This is essential when wanting to deep dive into the properties exposed within the model once exported.

In conclusion, obtaining the architecture of a neural network from serialized formats via static analysis is a feasible endeavor, although it has limitations and is not always a perfect reconstruction. By combining techniques using both framework-specific tools and standard interchange formats like ONNX, one can gather a good approximation of the architecture. In my past experiences, these methods have been vital in understanding and sometimes even recovering lost model architectures. Remember to always be critical of your findings, cross-check results with multiple methods, and document any assumptions made along the way.
