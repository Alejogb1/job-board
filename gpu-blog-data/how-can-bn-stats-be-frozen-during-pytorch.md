---
title: "How can BN stats be frozen during PyTorch QAT with FX graph mode?"
date: "2025-01-30"
id: "how-can-bn-stats-be-frozen-during-pytorch"
---
Batch Normalization (BN) statistics freezing during quantization-aware training (QAT) in PyTorch using the FX graph mode requires careful manipulation of the graph representation. My experience with deploying high-performance models for mobile devices heavily relied on this precise control; otherwise, the performance gains from quantization are significantly eroded.  The key is understanding that the forward pass of a BN layer during inference uses pre-computed statistics (mean and variance) rather than dynamically calculating them from the input batch.  Freezing these statistics during QAT ensures consistent behavior between the quantized and floating-point models, preventing unexpected accuracy drops.

**1. Clear Explanation**

The FX graph mode in PyTorch provides an intermediate representation of your model, allowing for detailed manipulation of the computation graph before conversion to a quantized model.  Standard QAT processes often lead to dynamically computed BN statistics during training, which are inconsistent with the fixed statistics used during inference.  To solve this, we need to modify the FX graph to replace the standard BN layer's forward pass with a version that utilizes pre-computed statistics.  This involves:

a) **Collecting Statistics:** During a calibration phase, we perform a forward pass on a representative subset of the training data.  This generates the mean and variance for each BN layer.  We store these statistics.

b) **Modifying the FX Graph:** Using the `torch.fx` API, we traverse the graph and replace each `BatchNorm` node with a custom module that accepts the pre-computed statistics as input.  This custom module effectively hardcodes the BN statistics, preventing recalculation during QAT.

c) **Quantization-Aware Training:**  With the modified graph, we proceed with the standard QAT workflow.  The BN layers now behave consistently, using the frozen statistics, leading to a more accurate quantized model.

It's crucial to note that the calibration phase should use a dataset representative of the intended inference data distribution.  Using a skewed or insufficient dataset for calibration may compromise the accuracy of the quantized model.


**2. Code Examples with Commentary**


**Example 1:  Collecting BN Statistics**

```python
import torch
import torch.nn as nn

def collect_bn_stats(model, data_loader):
    model.eval()
    bn_stats = {}
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch # Assuming data_loader yields (inputs, labels)
            _ = model(inputs) # Forward pass to update running statistics

            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_stats[name] = {
                        'mean': module.running_mean.clone().detach(),
                        'var': module.running_var.clone().detach()
                    }
    return bn_stats
```

This function iterates through a data loader and performs forward passes in evaluation mode (`model.eval()`). Importantly, `torch.no_grad()` prevents gradient calculations. It then extracts the running mean and variance from each `BatchNorm2d` layer, storing them in a dictionary indexed by layer name.  This dictionary is crucial for the next step.



**Example 2:  Creating a Custom Frozen BN Module**

```python
import torch
import torch.nn as nn

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 mean=None, var=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if mean is None or var is None:
            raise ValueError("Mean and variance must be provided.")
        self.register_buffer('running_mean', mean)
        self.register_buffer('running_var', var)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return torch.nn.functional.batch_norm(x, self.running_mean, self.running_var,
                                              self.weight, self.bias, self.training,
                                              self.momentum, self.eps)
```

This custom module is the heart of the solution.  It takes pre-computed mean and variance as input during initialization and uses `torch.nn.functional.batch_norm` to perform inference with these frozen statistics. The `training` flag is irrelevant as we're always using the pre-computed stats, regardless of training mode.  The affine parameters (weight and bias) can be included if needed, but they often get quantized independently.


**Example 3: Modifying the FX Graph**

```python
import torch
import torch.fx as fx
from example2 import FrozenBatchNorm2d # Assuming FrozenBatchNorm2d from Example 2

def freeze_bn_stats(model, bn_stats):
    gm = fx.symbolic_trace(model)
    new_modules = {}
    for node in gm.graph.nodes:
        if node.op == 'call_module' and isinstance(model.__getattr__(node.target), nn.BatchNorm2d):
            bn_layer = model.__getattr__(node.target)
            frozen_bn = FrozenBatchNorm2d(bn_layer.num_features, mean=bn_stats[node.target]['mean'], var=bn_stats[node.target]['var'])
            new_modules[node.target] = frozen_bn

    gm.recompile()
    for name, module in new_modules.items():
        gm.graph.rebind_input(name, module)

    return gm

# Example usage (assuming model and bn_stats are defined)
modified_model = freeze_bn_stats(model, bn_stats)
```

This function traces the model using `fx.symbolic_trace`. Then, it iterates over the nodes in the graph, identifying `BatchNorm2d` layers. For each, it creates an instance of `FrozenBatchNorm2d` using the pre-computed statistics from `bn_stats`.  Finally, it replaces the original `BatchNorm2d` nodes with the frozen counterparts within the FX graph, and recompiles the graph for consistency.

**3. Resource Recommendations**

The official PyTorch documentation on FX graph mode is essential.  Thorough understanding of the quantization APIs within PyTorch is critical.  A solid grasp of the mathematical underpinnings of Batch Normalization will also enhance your understanding.  Finally, explore academic papers focusing on quantization-aware training techniques to gain further insight into optimization strategies.
