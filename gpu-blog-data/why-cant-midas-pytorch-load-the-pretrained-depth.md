---
title: "Why can't MIDAS PyTorch load the pretrained depth estimation model?"
date: "2025-01-30"
id: "why-cant-midas-pytorch-load-the-pretrained-depth"
---
The inability to load a pretrained MiDaS PyTorch depth estimation model commonly stems from discrepancies between the model architecture and the checkpoint's structural expectations, usually manifested during the `torch.load()` operation.  Specifically, the root cause is often an architectural mismatch at the layer level—particularly variations in the number of input channels or the specific type of layers involved, including convolutional, normalization, and attention blocks. These subtle divergences, even if seemingly minor, cause mismatches with the weights saved in the checkpoint, preventing a successful load.

I’ve repeatedly encountered this issue when porting research code or adapting pre-trained models for custom datasets. The first step, invariably, is ensuring that the model definition being loaded is absolutely identical to the model that generated the checkpoint. A crucial factor to remember here is that MiDaS models, developed by Intel and others, have gone through several architectural revisions. The exact version used to generate the publicly available weights is not always explicitly documented with the pre-trained models themselves. This implies that blindly attempting to load these weights onto what might appear to be a 'MiDaS model' will often fail.

A direct mismatch is typically observed as an error message originating from `torch.load`, reporting something like a key mismatch—`KeyError: 'model.module.layer_name.weight'` or a similar variant. These indicate that a layer in the model you’re trying to load either doesn’t exist, has a different name, or has a different shape compared to what's stored in the checkpoint. This goes beyond simply differing variable names; it refers to genuine disparities in the architecture's layout.

Furthermore, if the model was trained using DataParallel, the saved checkpoint often includes `module.` prefixes in all layer names. If one attempts to load this checkpoint into a model *not* wrapped with DataParallel, the key names in the checkpoint will not match those in the model, leading to loading failures, or incorrect loads if layer shape is the same but names differ, which can manifest as drastically wrong outputs.

Let's look at a few situations and the remedies:

**Code Example 1: Simple Mismatch Scenario**

```python
import torch
import torch.nn as nn
from collections import OrderedDict

class MismatchModel(nn.Module):
    def __init__(self):
        super(MismatchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3) # Expected 3 input channels, might be more
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(18432, 10) # Example specific output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Example checkpoint, assume this was created elsewhere with a slightly different model
checkpoint_path = "mismatch_model_checkpoint.pth"
checkpoint_data = {
    "state_dict": OrderedDict([
        ("conv1.weight", torch.randn(16, 4, 3, 3)), # 4 input channels - mismatch
        ("conv1.bias", torch.randn(16)),
        ("conv2.weight", torch.randn(32, 16, 3, 3)),
        ("conv2.bias", torch.randn(32)),
        ("fc.weight", torch.randn(10, 18432)),
        ("fc.bias", torch.randn(10))
    ])
}
torch.save(checkpoint_data, checkpoint_path)



try:
    model = MismatchModel()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
except Exception as e:
    print(f"Error: {e}")
```

In this example, the `MismatchModel` expects 3 input channels in its initial convolutional layer, `conv1`.  However, the saved checkpoint's corresponding `conv1.weight` has the shape `(16, 4, 3, 3)`,  which implies that the model that was used to generate the checkpoint actually accepted 4 input channels. This direct architectural discrepancy causes the `load_state_dict` to fail.  The `OrderedDict` is used here to ensure the weights load in order, which is important if keys are not explicitly specified with `strict=False`.

**Code Example 2: DataParallel Prefix Issue**

```python
import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(18432, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


checkpoint_path = "dataparallel_checkpoint.pth"

# Simulate training with DataParallel:
model_dp = nn.DataParallel(SimpleModel())
state_dict_dp = model_dp.state_dict()
torch.save({"state_dict": state_dict_dp}, checkpoint_path)

try:
    model = SimpleModel()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
except Exception as e:
    print(f"Error: {e}")

try:
    model = SimpleModel()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Clean the state dict:
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

except Exception as e:
        print(f"Error after cleaning : {e}")
```
In this instance, the checkpoint is created using `DataParallel`.  The saved state dictionary therefore prefixes all the layer names with `module.`. When a model *not* wrapped in `DataParallel` attempts to load this checkpoint, the `load_state_dict` fails because the layer names like `conv.weight` do not match the checkpoint's `module.conv.weight`. Setting `strict=False` during loading might avoid the exception for key mismatches, but it won't actually load the weights correctly since it's just ignoring the 'module.' prefix. The state dictionary has to be cleaned by removing the `module.` prefixes explicitly before loading the weights with `strict=True`.

**Code Example 3: Layer Type Mismatch**

```python
import torch
import torch.nn as nn
from collections import OrderedDict

class LayerMismatchModel(nn.Module):
    def __init__(self):
        super(LayerMismatchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.norm = nn.BatchNorm2d(16)  # Batchnorm here
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        return x

class DifferentLayerModel(nn.Module):
    def __init__(self):
        super(DifferentLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.norm = nn.GroupNorm(4, 16) # Groupnorm mismatch here
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        return x

checkpoint_path = "layer_mismatch_checkpoint.pth"
model = LayerMismatchModel()
checkpoint = { "state_dict": model.state_dict()}
torch.save(checkpoint, checkpoint_path)


try:
    model = DifferentLayerModel()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
except Exception as e:
    print(f"Error: {e}")

```

This scenario shows an instance of a different normalization layer being used during training versus loading. `LayerMismatchModel` utilizes batch normalization `nn.BatchNorm2d`, while `DifferentLayerModel` has group normalization `nn.GroupNorm`. Even though the layer names (`self.norm`) are the same and the expected number of channels is also the same (16) , the specific layer type is different and will generate a mismatch when trying to load weights from a checkpoint. The error will result from the shape of the weights and biases of the layer being different depending on the specific kind of normalization operation being applied.

For deeper investigation into MiDaS-specific issues,  I recommend exploring publications that detailed the original architecture and related variations. Further, study the specific version of the MiDaS model the weights belong to. Specifically, consult resources from the Intel Open Source Technology Center, or repositories where MiDaS models have been adapted or released, such as those hosted on GitHub. Inspecting and comparing source code for slight differences in model layer definitions will often identify the cause of loading failures. Official documentation and source code provide the most accurate information regarding model structures and pre-training.
