---
title: "Where to find the YOLOv5 feature extractor?"
date: "2024-12-15"
id: "where-to-find-the-yolov5-feature-extractor"
---

hey there,

so, you're hunting for the yolo v5 feature extractor, huh? i get it. i’ve been down that rabbit hole myself more than once. it's not always as straightforward as just pulling a single file, and it’s easy to get turned around in the codebase if you're not super familiar. been there, done that, got the t-shirt.

let me break it down from my perspective and how i usually tackle this kinda thing. we’re not looking for a magical “feature_extractor.py” file, because it’s more integrated than that. think of yolo v5 as a network composed of different blocks and layers, not a singular pipeline with obvious compartments. the "feature extraction" isn’t done by a lone component. it's a process that occurs over the first chunk of the network, before the head does its bounding box predictions and classifications.

the specific part where feature extraction happens is primarily within the backbone of the model. in yolo v5 that backbone is usually cspdarknet or its variations. it’s the layers before the 'head', which do all the prediction stuff. we should focus on that section of the model.

let's talk implementation details, and this is where things can get a bit tricky depending on what exactly you want to achieve.

if you're simply looking to visualize features, or use them for other tasks, it’s handy to pull intermediate outputs from the model at specific points. here's how you'd typically do it, using pytorch which is what yolo v5 is built on:

```python
import torch
import torch.nn as nn
from models.yolo import Model

def get_feature_maps(model, input_tensor, layer_names):
    """
    extracts feature maps from the model at specified layers.
    """
    outputs = {}
    def hook(module, input, output):
      outputs[name] = output.detach()
    handles = []
    for name, layer in model.named_modules():
      if name in layer_names:
          handles.append(layer.register_forward_hook(hook))
    model(input_tensor)
    for handle in handles:
      handle.remove()
    return outputs


if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Model(cfg='yolov5s.yaml').to(device)
  model.load_state_dict(torch.load('yolov5s.pt', map_location=device)['model'])
  input_tensor = torch.randn(1, 3, 640, 640).to(device)

  # specify layers to get from model
  selected_layers = ["model.2", "model.4", "model.6", "model.9", "model.10"] # Example: specific CSP modules
  feature_maps = get_feature_maps(model, input_tensor, selected_layers)

  for layer_name, feature_map in feature_maps.items():
    print(f"Feature map from layer {layer_name}: {feature_map.shape}")

```

in that code we're using forward hooks to grab intermediate activations from layers we specify. i added the example of layer names that start with `model.`, it’s how it is structured within yolo v5, and it gives a useful output that contains the shape of each extracted feature map. if you want, you can inspect all names of the layers, and choose the one which correspond to the end of the feature extraction section of the model. it’s a very handy way to explore the layers, and you'll probably see layers named like 'model.6' or 'model.9' which are good candidates to extract features. you can print model and check if those layers exist.

sometimes, it is not sufficient and you need to get an output that corresponds to the end of the backbone. this is where you might need to modify the model code to return the output from the last layer of the backbone directly.

here's an example showing how you could modify `models/yolo.py` file, and how to use it to extract backbone features:

```python
import torch
import torch.nn as nn
from models.common import * # Import common modules from yolo model code

class ModifiedYOLO(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):
        super().__init__()
        import yaml  # Import here
        with open(cfg, errors='ignore') as f:
            self.yaml = yaml.safe_load(f)  # Load YAML
        self.model, self.save = parse_model(self.yaml, ch=[3])

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None) # Save layer outputs in y
        return x # Return last output of the model (backbone feature)

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ModifiedYOLO(cfg='yolov5s.yaml').to(device)
  model.load_state_dict(torch.load('yolov5s.pt', map_location=device)['model'])

  input_tensor = torch.randn(1, 3, 640, 640).to(device)
  backbone_features = model(input_tensor)
  print(f"shape of features of the backbone: {backbone_features.shape}")
```

this one is a modified class of yolo where the output of the `forward` function is the feature map, instead of the three feature maps that goes to the head of yolo. this works in the same way as the original yolo model, but now its `forward` method returns the backbone feature maps instead of the usual model prediction. this is convenient when you need the output of the backbone. you can then use it for other tasks, without having to reimplement the feature extractor.

now, this might be more complex than what you were expecting at first, but that's how yolo v5 is structured. if you need a clean feature extractor for other uses this last snippet is the most useful one. it's good to understand that it is not just a method but a network, and you can output data in any part of the model. you can even modify it to output multiple intermediate layers if needed.

a note to be careful about that modified version, it's not compatible with the original usage of yolo, meaning that it does not have the prediction layers as the output. make sure that you are using it for feature extraction only.

also, something to be careful about is the loading of the model. you'll notice i'm loading the weights from a `.pt` file. it's essential to load the pre-trained weights unless you have your own trained version, as it saves a lot of time, and is trained with very good data. if you're training from scratch and want to extract features, your results might be bad. this happened to me once. i trained a model with no data and wondered why the results were so bad. (it seems funny now, but back then i was pulling my hair out).

lastly if your goal is to use this features in another network or task, it's generally a good practice to freeze the yolo backbone weights to retain the information learned in image detection.

here’s another simple example that you can use, with a layer that’s more close to the end of the backbone:

```python
import torch
import torch.nn as nn
from models.yolo import Model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(cfg='yolov5s.yaml').to(device)
    model.load_state_dict(torch.load('yolov5s.pt', map_location=device)['model'])
    input_tensor = torch.randn(1, 3, 640, 640).to(device)

    # Get feature map from layer "model.21" this is a good example of an end of backbone layer
    feature_map = model.model[21](model.model[20](model.model[19](input_tensor)))

    print(f"shape of features of the backbone layer model.21: {feature_map.shape}")
```

this code will extract the feature from the layer `model.21`. this is useful if you want to extract a layer without modifying the model class and it is another option. as an example the layer `model.21` in the `yolov5s` model, it’s a convolution that comes after the last csp block.

remember that layer numbering and type can vary between models (like yolov5s, yolov5m, yolov5l, etc). that's why it's helpful to inspect the model first, with the first snippet, or printing it before extracting the feature.

when it comes to resources i would highly recommend reading the original yolo v5 paper, it goes deep into the architecture and will help understand how it was built, which is fundamental when you work with it. i can also recommend "deep learning with pytorch" by elias stevens, this book has a solid breakdown on pytorch, which can help understand the code snippets i mentioned. it also contains an example with image classification using pretrained models, which can give some ideas.

i hope this long explanation helps, i tried to detail my experience with this problem, and the usual workarounds i use. i'm more than glad to answer any other question you might have if something is still unclear.
