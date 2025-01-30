---
title: "How can I display input/output sizes for modules in a PyTorch Lightning model summary when using nn.ModuleList?"
date: "2025-01-30"
id: "how-can-i-display-inputoutput-sizes-for-modules"
---
The challenge of visualizing input/output shapes within `nn.ModuleList` containers in PyTorch Lightning model summaries stems from the dynamic nature of the list's contents and the limitations of standard summary tools.  My experience working on large-scale NLP models highlighted this precisely;  `nn.ModuleList` offered crucial flexibility for managing varying numbers of sub-modules, but obtaining detailed I/O shape information during model construction required a more nuanced approach than simply relying on `torchsummary` or similar libraries.  The key lies in leveraging the `register_forward_hook` functionality combined with custom shape-tracking logic within the modules themselves.

**1. Clear Explanation**

Standard model summary tools often struggle with `nn.ModuleList` because they typically perform a forward pass with dummy input tensors.  This approach fails when the number of modules or their input expectations vary dynamically, as is common with `nn.ModuleList`.  The solution involves instrumenting each module within the `nn.ModuleList` to explicitly log its input and output tensor shapes. We achieve this by registering a forward hook to each module.  This hook intercepts the forward pass, captures the tensor shapes, and stores them in a designated container. This information can then be accessed and displayed alongside the model summary, providing a complete picture of I/O sizes for each sub-module.

This process requires a structured approach. First, we need a custom module that not only performs the core computation but also handles shape logging.  Then, this custom module is populated into the `nn.ModuleList`.  Finally, a mechanism to aggregate and display this logged information is necessary. This is typically integrated with the PyTorch Lightning `summary()` functionality.  We cannot directly modify the summary functionality itself, so we create a wrapper function or modify the model's `forward` method to display our aggregated information.

**2. Code Examples with Commentary**

**Example 1: Basic Shape Logging Module**

```python
import torch
import torch.nn as nn

class ShapeLoggingModule(nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.module = base_module
        self.input_shapes = []
        self.output_shapes = []
        self.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.input_shapes.append(tuple(input[0].shape)) #Assuming single input
        self.output_shapes.append(tuple(output.shape))

    def forward(self, x):
        return self.module(x)


#Example Usage:
linear_layer = ShapeLoggingModule(nn.Linear(10, 5))
input_tensor = torch.randn(1,10)
output_tensor = linear_layer(input_tensor)
print(f"Input Shapes: {linear_layer.input_shapes}")
print(f"Output Shapes: {linear_layer.output_shapes}")
```

This example demonstrates a base `ShapeLoggingModule`.  It wraps an arbitrary module (`base_module`), registers a forward hook that appends input and output shapes to internal lists, and then executes the wrapped module's forward pass. Note the assumption of a single input tensor.  Multiple inputs require modification to handle the tuple structure of `input`.

**Example 2: Integrating with nn.ModuleList in a PyTorch Lightning Module**

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from example_1 import ShapeLoggingModule #Importing from Example 1


class MyLightningModule(pl.LightningModule):
    def __init__(self, num_layers=3, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.layers = nn.ModuleList([ShapeLoggingModule(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)) for i in range(num_layers)])
        self.final_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


    def training_step(self, batch, batch_idx):
        # ... training logic ...
        pass


    def configure_optimizers(self):
        # ... optimizer configuration ...
        pass

#Example usage:
model = MyLightningModule()
trainer = pl.Trainer(logger=False) # For simplicity, no logging
trainer.predict(model, datamodule=None, ckpt_path=None) #Predict step calls forward
# Access shapes from each layer in self.layers
for i, layer in enumerate(model.layers):
    print(f"Layer {i+1}: Input Shapes - {layer.input_shapes}, Output Shapes - {layer.output_shapes}")
```

This example integrates `ShapeLoggingModule` within a PyTorch Lightning module.  The `nn.ModuleList` now contains instances of our shape-logging wrapper. The critical part here is the absence of explicit shape logging in the `forward` method. Instead, the shape information is captured internally within each `ShapeLoggingModule`. Note that the `predict` step is used here since we are only interested in shape information. A `forward` call during training will update the shapes, but it won't display them.


**Example 3: Enhanced Shape Logging and Summary Display**

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from example_1 import ShapeLoggingModule


class MyEnhancedLightningModule(pl.LightningModule):
    # ... (init and other methods as before) ...

    def on_predict_epoch_end(self, outputs):
        summary_string = "Layer | Input Shape | Output Shape\n"
        summary_string += "-------|------------|-------------\n"
        for i, layer in enumerate(self.layers):
            summary_string += f"{i+1}     | {layer.input_shapes[0]}       | {layer.output_shapes[0]}\n" # assuming single forward pass
        print(summary_string)

# Example usage (similar to Example 2, but using on_predict_epoch_end)
```

This final example improves on the previous one by providing a more structured display of the logged shape information.  This utilizes PyTorch Lightning's `on_predict_epoch_end` callback to produce a formatted summary at the end of prediction.  This avoids interrupting the standard PyTorch Lightning training/evaluation workflow and presents the information in a user-friendly format.

**3. Resource Recommendations**

The PyTorch documentation on hooks,  the PyTorch Lightning documentation on callbacks and model customization, and a comprehensive textbook on deep learning (covering topics such as tensor operations and model architecture) are invaluable resources for understanding the underlying mechanisms and mastering more advanced techniques.  Furthermore, the documentation for any chosen visualization library will provide details on its capabilities and limitations.  Studying examples of custom PyTorch modules and Lightning callbacks will be beneficial for extending these approaches to diverse model architectures.
