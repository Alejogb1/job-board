---
title: "How do I export a Detectron2 model to TorchScript without encountering the '_ScaleGradient' error?"
date: "2025-01-30"
id: "how-do-i-export-a-detectron2-model-to"
---
The core impediment to exporting a Detectron2 model to TorchScript frequently arises from custom operations not natively supported by TorchScript’s tracing mechanism, specifically the "_ScaleGradient" operation used internally during training by some Detectron2 components. This gradient scaling, while crucial for training stability, is a non-standard operation that TorchScript cannot directly represent within its static computational graph. I've personally encountered this issue on several projects involving object detection and instance segmentation with Detectron2 and found that addressing it requires a careful combination of model manipulation and export strategy.

The first step in resolving this is understanding that TorchScript operates by either tracing or scripting. Tracing executes the model once with sample inputs and records the operations. Scripting involves explicitly annotating the model with type hints. Detectron2 models, particularly those employing complex backbones or custom heads, often rely on operations that are not easily captured through tracing, leading to the "_ScaleGradient" error when it encounters an unsupported internal operation. We must modify or bypass such problematic operations during the export process.

Let's examine specific approaches, accompanied by code examples, to demonstrate how this can be achieved. Consider a situation where the problematic gradient scaling is occurring within the box prediction layer.

**Example 1: Disabling Gradient Scaling Through Custom Wrapping**

The initial and perhaps simplest approach is to identify where gradient scaling is applied and disable it during model export. This is achieved by wrapping the target module, overriding the forward method to bypass the scaling logic. Here's how this might be implemented:

```python
import torch
import torch.nn as nn
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class WrapperModule(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module

    def forward(self, x):
        # Manually retrieve box_pred to avoid _ScaleGradient
        if isinstance(self.original_module, FastRCNNOutputLayers):
           pred_class_logits = self.original_module.cls_score(x)
           pred_proposal_deltas = self.original_module.bbox_pred(x)
           return pred_class_logits, pred_proposal_deltas
        else:
           return self.original_module(x)

def export_model(model):
    cfg = get_cfg()
    # Setup configurations or load model weights as needed.
    # For brevity, assume cfg is already setup
    
    # Locate the modules to be wrapped
    for name, module in model.named_modules():
      if isinstance(module, FastRCNNOutputLayers):
          print(f"wrapping {name} for export")
          setattr(model, name, WrapperModule(module))

    # Create dummy input for tracing, must match input structure for Detectron2
    dummy_input = [
        {"image": torch.rand(3, 800, 800),
         "instances": None}
    ]


    traced_model = torch.jit.trace(model, (dummy_input,), check_trace=False)
    return traced_model

if __name__ == "__main__":
   cfg = get_cfg()
   cfg.merge_from_file("path/to/your/config.yaml") # Use your config here
   cfg.MODEL.WEIGHTS = "path/to/your/weights.pth" # use your weights here
   model = build_model(cfg)
   model.eval()
   traced_model = export_model(model)
   torch.jit.save(traced_model, "exported_model.pt")

```

In this example, we identify the `FastRCNNOutputLayers` module. We wrap it in our `WrapperModule`, which intercepts the call to the underlying layers. Instead of letting `FastRCNNOutputLayers` call `_ScaleGradient` during backpropagation, we manually call `cls_score` and `bbox_pred` on the output, circumventing the problematic operation.  We then use `torch.jit.trace` to create a TorchScript version of the model using a dummy input. This approach directly addresses the core issue, however requires a careful inspection of the model architecture.

**Example 2: Overriding Forward Pass with Conditional Logic**

Another strategy involves modifying the forward pass of the module directly, adding a conditional check to disable the gradient scaling related operations. This is particularly useful for models using a custom component that invokes `_ScaleGradient` based on training/evaluation mode:

```python
import torch
import torch.nn as nn
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class ModifiedFastRCNNOutputLayers(FastRCNNOutputLayers):

   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

   def forward(self, x):

      if not self.training:
        pred_class_logits = self.cls_score(x)
        pred_proposal_deltas = self.bbox_pred(x)
        return pred_class_logits, pred_proposal_deltas
      else:
        return super().forward(x)

def export_model(model):
   cfg = get_cfg()
   # Setup configurations or load model weights as needed.
   # For brevity, assume cfg is already setup

   # Locate the modules to be modified
   for name, module in model.named_modules():
     if isinstance(module, FastRCNNOutputLayers):
        print(f"modifying {name} for export")
        setattr(model, name, ModifiedFastRCNNOutputLayers(
            module.in_features, module.num_classes, module.cls_score.weight.shape[0],
            module.bbox_pred.weight.shape[0], module.cls_score.bias.shape[0],
            module.bbox_pred.bias.shape[0],  module.box2box_transform, module.smooth_l1_beta
         ))


   # Create dummy input for tracing, must match input structure for Detectron2
   dummy_input = [
       {"image": torch.rand(3, 800, 800),
        "instances": None}
   ]

   traced_model = torch.jit.trace(model, (dummy_input,), check_trace=False)
   return traced_model

if __name__ == "__main__":
   cfg = get_cfg()
   cfg.merge_from_file("path/to/your/config.yaml") # Use your config here
   cfg.MODEL.WEIGHTS = "path/to/your/weights.pth" # use your weights here
   model = build_model(cfg)
   model.eval()
   traced_model = export_model(model)
   torch.jit.save(traced_model, "exported_model.pt")

```

In this scenario, we create `ModifiedFastRCNNOutputLayers`, inheriting from the original class. In this version we use the `training` flag to conditionally control the forward pass. Specifically, if the model is not in training mode (which will be the case during export), we directly call the relevant sub-modules without invoking any _ScaleGradient.  This approach relies on Detectron2 properly setting `training` flag to false. This modification can be more elegant and maintainable than wrapping module in some situations.

**Example 3: Using TorchScript Scripting**

Finally, for more intricate models, explicitly scripting may provide a more comprehensive solution. This entails type hinting and annotating the relevant sections of the model:

```python
import torch
import torch.nn as nn
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class ScriptedFastRCNNOutputLayers(nn.Module):
   def __init__(self, module: FastRCNNOutputLayers):
      super().__init__()
      self.cls_score = module.cls_score
      self.bbox_pred = module.bbox_pred

   @torch.jit.script_method
   def forward(self, x : torch.Tensor):
      #type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
      pred_class_logits = self.cls_score(x)
      pred_proposal_deltas = self.bbox_pred(x)
      return pred_class_logits, pred_proposal_deltas

def export_model(model):
   cfg = get_cfg()
   # Setup configurations or load model weights as needed.
   # For brevity, assume cfg is already setup
   # Locate the modules to be modified
   for name, module in model.named_modules():
     if isinstance(module, FastRCNNOutputLayers):
         print(f"scripting {name} for export")
         setattr(model, name, ScriptedFastRCNNOutputLayers(module))


   # Create dummy input for tracing, must match input structure for Detectron2
   dummy_input = [
       {"image": torch.rand(3, 800, 800),
        "instances": None}
   ]

   scripted_model = torch.jit.script(model)
   return scripted_model

if __name__ == "__main__":
   cfg = get_cfg()
   cfg.merge_from_file("path/to/your/config.yaml") # Use your config here
   cfg.MODEL.WEIGHTS = "path/to/your/weights.pth" # use your weights here
   model = build_model(cfg)
   model.eval()
   scripted_model = export_model(model)
   torch.jit.save(scripted_model, "exported_model.pt")

```

Here, the `ScriptedFastRCNNOutputLayers` explicitly defines the forward pass and types through `torch.jit.script_method`. We then use `torch.jit.script` to convert the modified model. This method offers the greatest level of control and allows for more complex model transformations. Although it can be more verbose to implement, it can be more resilient to changes in the internal details of Detectron2.

These three techniques are not mutually exclusive.  You might find yourself combining them depending on the structure of the Detectron2 model you are trying to export.

In summary, successful export of a Detectron2 model to TorchScript while avoiding "_ScaleGradient" errors requires meticulous analysis of the model architecture and strategic intervention. This may involve wrapping modules, overriding forward passes, or utilizing TorchScripting. Each approach has its benefits and drawbacks, and the selection of the best method typically depends on the model's complexity and the location of the problematic operations. The key is to isolate the parts causing the issue and devise a method to either bypass the operation or re-implement it in a TorchScript-compatible manner.

For further study on the topic, consult the official PyTorch documentation on TorchScript, focusing on tracing and scripting. Additionally, explore Detectron2's source code to better understand how custom operations are utilized in various model components. Finally, reviewing relevant discussions on platforms like StackOverflow or GitHub issues related to Detectron2 and TorchScript can provide valuable insights from the community.
