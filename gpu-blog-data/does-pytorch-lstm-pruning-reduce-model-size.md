---
title: "Does PyTorch LSTM pruning reduce model size?"
date: "2025-01-30"
id: "does-pytorch-lstm-pruning-reduce-model-size"
---
PyTorch Long Short-Term Memory (LSTM) model pruning, specifically weight pruning, does indeed reduce the size of the model in terms of storage requirements and, subsequently, often leads to faster inference times. This reduction stems from setting a substantial portion of the weight parameters within the LSTM layers to zero, thereby representing sparse matrices rather than dense ones. This sparsification doesn't necessarily translate to a proportional reduction in the number of parameters reported during model construction; instead, the memory footprint and actual computational burden are lowered. Over my career, I've found the performance improvements compelling enough to make pruning a routine post-training step in various sequence modeling projects.

The core mechanism of pruning involves identifying parameters deemed less critical to the network's function and zeroing them out. Different pruning strategies exist, including magnitude-based pruning (which targets weights with the smallest absolute values), gradient-based pruning (which considers the importance of weights based on gradient information), and more sophisticated algorithms. Regardless of the specific method, the end result is a sparse weight tensor. In PyTorch, these tensors remain the same data type and shape, but many elements are numerically zero, which allows for optimizations when performing computation, especially when employing sparse tensor libraries.

Let me illustrate with code examples. First, I'll construct a simple LSTM model to provide context.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example Usage:
input_size = 10
hidden_size = 20
num_layers = 2
num_classes = 5
seq_length = 30
batch_size = 32

model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
input_data = torch.randn(batch_size, seq_length, input_size)
output = model(input_data)

print(f"Initial model size: {sum(p.numel() for p in model.parameters())} parameters.")

```

In this basic setup, the `SimpleLSTM` class constructs a standard LSTM network with an output linear layer. The number of trainable parameters are printed to establish a baseline. Notice the creation of initial hidden and cell states set to zero. I've intentionally made the states trainable, which is not strictly necessary for many common use cases, but this enables a more complex model for demonstration purposes. The input and output shapes are also established.

Now, consider how to prune the LSTM layers using the `prune` module from PyTorch. Below I will demonstrate weight magnitude pruning to 50% sparsity.

```python
parameters_to_prune = (
    (model.lstm, 'weight_ih_l0'),
    (model.lstm, 'weight_hh_l0'),
    (model.lstm, 'weight_ih_l1'),
    (model.lstm, 'weight_hh_l1'),
    (model.fc, 'weight'),
)

for module, name in parameters_to_prune:
    prune.l1_unstructured(module, name=name, amount=0.5)

print(f"Model size after pruning: {sum(p.numel() for p in model.parameters())} parameters")

for module, name in parameters_to_prune:
    print(f"Sparsity of {name}: {torch.sum(getattr(module, name) == 0) / getattr(module, name).numel() * 100:.2f}%")
```

Here, we target the weight matrices within the LSTM layer (`weight_ih` for input-hidden and `weight_hh` for hidden-hidden) for both layers of our two-layer LSTM along with the weights in the linear layer. The `l1_unstructured` pruning method zeros out 50% of weights with the lowest absolute magnitudes. We then print the overall model parameter count, which remains the same, and print the sparsity of each pruned matrix. Even though the overall parameter count is the same, many values are now zero. The model's storage footprint is smaller due to the sparse representation of the matrices, and often can run faster because zeroed parameters do not contribute to the calculations.

Let's consider how to work with these pruned models in practice, specifically how to access the masks. This also illustrates the fact that pruned weights are actually stored using a mask, which is applied during the forward pass.

```python
# Accessing masks after pruning:

for module, name in parameters_to_prune:
    mask_name = name + '_mask'
    mask = getattr(module, mask_name)
    print(f"Mask shape for {name}: {mask.shape}")

    # Apply mask before moving model to a device. If on CPU:
    getattr(module, name).data.mul_(mask)

    # If using CUDA, move model after pruning and mask application:
    # module.to('cuda')


    # Test with a forward pass.
    if torch.cuda.is_available():
      input_data = input_data.to('cuda')
      model.to('cuda')
    with torch.no_grad():
        _ = model(input_data)

```

This code segment accesses the mask tensors. Each pruned parameter has a corresponding mask associated with it. A "1" in the mask indicates the corresponding parameter is kept, whereas a "0" indicates the parameter was pruned. It's necessary to either apply the mask manually if running on CPU or to explicitly move the model to the desired device, in which case the masks are applied when the model itself is moved to that device. Subsequently, a dummy forward pass is performed to verify that the pruned model does not produce errors.

Importantly, pruning is not a one-time operation. In practice, it is often performed iteratively, with retraining in between pruning steps to recover any performance losses. Also, different pruning methods have tradeoffs regarding the sparsity that can be achieved while maintaining an acceptable performance level.

For deeper understanding of model pruning in PyTorch, the following resources are highly beneficial: The official PyTorch documentation on the `torch.nn.utils.prune` module provides exhaustive information on implemented pruning methods and their API. Also, various research publications explore advanced pruning methods, including structured pruning, and provide insights into effective pruning strategies in different neural network architectures. Numerous blog posts and tutorials from the PyTorch community address various aspects of pruning and often provide code examples applied to common scenarios.

In closing, PyTorch LSTM pruning effectively reduces model size through the creation of sparse tensors by zeroing out less significant parameters, even though the parameter count often appears unchanged. Optimizations associated with sparse calculations frequently translate to faster inference, making pruning a useful tool for optimizing LSTM models.
