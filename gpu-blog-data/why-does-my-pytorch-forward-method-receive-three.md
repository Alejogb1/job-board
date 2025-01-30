---
title: "Why does my PyTorch `forward()` method receive three arguments when it expects two?"
date: "2025-01-30"
id: "why-does-my-pytorch-forward-method-receive-three"
---
When a PyTorch `nn.Module` subclass's `forward()` method unexpectedly receives three arguments instead of the anticipated two, it’s invariably due to an attempt to invoke the model directly as a function rather than utilizing its call mechanism. This subtle, yet crucial difference, exposes the hidden `*args, **kwargs` architecture of the Python function call and its application to `nn.Module` objects.

My experience building a deep learning-based image segmentation model highlighted this exact issue. I was initially perplexed when my `forward()` method, defined to accept an input tensor and a boolean flag, was breaking down with the error that it received an additional argument. The problem wasn't in the definition of my method, but in how I was unintentionally calling the model.

The `nn.Module` class in PyTorch overloads the `__call__` method. This crucial step allows instances of custom models to be invoked as if they are functions. However, this isn't a straightforward mapping to the `forward()` method. When you call a model instance (e.g., `model(input_tensor, flag)`), PyTorch internally invokes the `__call__` method which, in turn, pre-processes the input before dispatching it to the `forward()` method. Specifically, `__call__` prepares and passes the input through PyTorch's internal hooks. This includes an implicit first argument: the `self` reference to the model instance. Thus, the model's `forward()` receives the model instance, followed by the explicit arguments you intended to pass. Calling a model *directly* as `model.forward(input_tensor, flag)` bypasses this mechanism, resulting in the expected two arguments being passed, as intended.

Consider the following code example demonstrating this behavior:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, input_tensor, flag):
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Flag Value: {flag}")
        print(f"Self type: {type(self)}") #added to explicitly show the class instance
        output = self.linear(input_tensor)
        return output

model = MyModel()
input_data = torch.randn(1, 10)
use_flag = True
```

Now, if we incorrectly call the model’s `forward()` method directly, observe what happens:

```python
# Incorrect usage: calling forward directly
try:
    model.forward(input_data, use_flag)
except Exception as e:
     print(f"Error when using .forward() call: {e}")

# Correct Usage
output = model(input_data, use_flag)
print(f"Output Shape: {output.shape}")
```

This first `try...except` block will capture the error because the `forward()` method now receives three arguments: the model itself (the `self` pointer), the `input_data`, and the `use_flag`. This will generate a `TypeError`, specifically stating the issue of the method expecting two arguments but receiving three. The second call, however, correctly invokes the model through `__call__` and, thus, works as expected, because this call only passes the `input_data` and `use_flag` to the intended argument positions within the `forward` method, whereas the `self` argument is implicitly handled by PyTorch internally.

Let's further illustrate the effect with a modified `forward()` method that explicitly prints the identity of each argument:

```python
class MyModelWithArgs(nn.Module):
    def __init__(self):
        super(MyModelWithArgs, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, arg1, arg2):
        print(f"Argument 1 Type: {type(arg1)}")
        print(f"Argument 2 Type: {type(arg2)}")
        output = self.linear(arg1)
        return output


model_args = MyModelWithArgs()
input_data_2 = torch.randn(1, 10)
use_flag_2 = False

# Incorrect usage: calling forward directly
model_args.forward(input_data_2, use_flag_2)
print("-----------------------")
# Correct Usage
model_args(input_data_2, use_flag_2)
```

The output from this example will show the `arg1` in the incorrect usage as type of the `MyModelWithArgs` model class itself, where the correct usage will print the argument type of the tensor. This makes it obvious what is causing the discrepancy in argument number. I have encountered variations of this during complex model implementations, such as in research projects requiring non-standard data handling.

The final code example will be a scenario in a custom PyTorch-Lightning system where the `forward()` method is invoked during `training_step`. The `forward()` method here is part of the lightning module as well as of any internal model in the module:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class InnerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SimpleLightningModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.inner_model = InnerModel(input_size, hidden_size, output_size)

    def forward(self, x):
        # Correct invocation of the inner model.
        return self.inner_model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs) # this correctly invokes forward
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Generate random data for demonstration
input_size = 5
hidden_size = 10
output_size = 3

input_data = torch.randn(100, input_size)
target_data = torch.randn(100, output_size)
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=32)


model = SimpleLightningModule(input_size, hidden_size, output_size)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, dataloader)
```

In this PyTorch Lightning example, you can observe that the forward pass happens in the `training_step`, and the module is called using `self(inputs)`. This will correctly pass the `inputs` into the `forward` method defined in `SimpleLightningModule`, which in turn passes the `inputs` into the `forward` method of the inner model. You can see here that the Lightning Module's `forward` method is meant to behave as a callable function in the same way as a basic `nn.Module`. If I had instead tried to call `self.forward(inputs)` in training_step, I would encounter the same error observed in the previous examples.

In summary, the unexpected appearance of a third argument in a `forward()` method stems from a misunderstanding of how PyTorch’s `nn.Module` invokes its forward pass through the overloaded `__call__` method. Direct calls circumvent this mechanism and pass the module instance itself as an extra argument. The correct invocation always happens through the model’s callable instance syntax.

To further enhance understanding, I would suggest exploring the official PyTorch documentation for `nn.Module`, with specific attention paid to the `__call__` method. I would also recommend analyzing the source code directly, as it provides a granular view of internal processes. Additionally, working through different implementations of custom modules and models is invaluable for developing an intuitive grasp of PyTorch’s architecture. Reading PyTorch blogs discussing `nn.Module` behaviour can also aid in practical understanding.
