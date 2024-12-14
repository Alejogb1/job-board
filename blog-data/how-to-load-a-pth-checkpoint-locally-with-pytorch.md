---
title: "How to load a *.pth checkpoint locally with pytorch?"
date: "2024-12-14"
id: "how-to-load-a-pth-checkpoint-locally-with-pytorch"
---

hey there, i've been messing with pytorch for quite a while now, and loading those .pth checkpoints locally is something i've done a gazillion times. it's pretty straightforward once you get the hang of it, but i can see where some folks might trip up a bit at first, particularly with some of the subtle details.

first off, let's assume you've got your model all defined. this means you've created a class that inherits from `torch.nn.module`, and it specifies your network architecture. let’s say for example you are loading something like a resnet model, then you'd have something like this initial skeleton that we'll use to load our checkpoint:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    # create an instance of the model
    model = MyModel(num_classes=10)  # adjust num_classes accordingly
    # we'll load the checkpoint later here
    print("model initialised")

```

now, the key thing when loading a checkpoint is that the state dictionary within the `.pth` file needs to match your model’s layers. that’s what we call the `state_dict`. it's basically a python dictionary that maps each layer's name to the corresponding tensor parameters that are learnable during training, weights and bias. this is the trickiest part and many newbies fall in this, because sometimes the names are different between what is saved and what is created. trust me, i have spent countless hours debugging these name mismatches. so to see the `state_dict` structure i usually do this:

```python
import torch

# let's assume you have saved a .pth file like this:
# torch.save(model.state_dict(), "my_checkpoint.pth")
# and your model is named like the one above

# load the state_dict from the .pth file
try:
    checkpoint = torch.load("my_checkpoint.pth")
    print("checkpoint loaded")

    # print the keys to understand the layer names
    print(checkpoint.keys())
except FileNotFoundError:
    print("checkpoint file not found, remember to save the model first!")
except RuntimeError as e:
    print(f"Error loading checkpoint: {e}")
```

this snippet helps you inspect what keys are present in the state dictionary that you are trying to load. once we get an idea about this, we'll need to load them into the actual model we have. to do this, we use the `load_state_dict` function. you usually will have a loop to move the tensors to the proper device where we want to compute (gpu/cpu) before you load them.

so the next step will be to load the parameters in the model, like so:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    # create an instance of the model
    model = MyModel(num_classes=10)  # adjust num_classes accordingly

    # let's assume you have saved a .pth file like this:
    # torch.save(model.state_dict(), "my_checkpoint.pth")

    # load the state_dict from the .pth file
    try:
        checkpoint = torch.load("my_checkpoint.pth")
        print("checkpoint loaded")

        # move the tensors to the appropriate device before loading
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")
        model.to(device)  # Move the model to the device too

        # load the state_dict into the model
        model.load_state_dict(checkpoint)
        print("model parameters loaded from checkpoint")

        # set the model in eval mode if you are not going to train it
        model.eval()
        print("model in eval mode")

        # now you can do inference. lets create random data to see if it works.
        # remember to move your input data to the same device too.
        example_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            example_output = model(example_input)
        print(f"output tensor shape: {example_output.shape}")
        print("model prediction ok")
    except FileNotFoundError:
        print("checkpoint file not found, remember to save the model first!")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")

```

a couple of points to take note, firstly, if you have trained your model in gpu, but trying to run it in cpu, or viceversa, you'll have to either to move the tensor data to that device before loading them, or, use a slightly different method to load, which is this: `torch.load('my_checkpoint.pth', map_location=torch.device('cpu'))`. the `map_location` argument handles the device issue in the load step, automatically moving the loaded tensors to the specified device, this can be handy in some situations, but be aware this could be less efficient memory wise if you have a very large model.

another really common error that happens is when you get a key mismatch error, this happens when you train in one environment and try to load the checkpoint in another with a different definition. sometimes it's due to you having a `dataparallel` model and trying to load a model that is not `dataparallel`. dataparallel adds the prefix `module.` to all the layers. sometimes it's because you are trying to load a checkpoint from a different architecture, there is no way to solve this unless the saved model is exactly the same as the loaded one. if you want to do transfer learning you should try to load the weights of the layers that you have the same, and leave the rest without initialisation.

it took me a few weeks to understand this and i almost quit, but i'm glad i didn't because now i'm a pytorch wizard.

there are also some common variations to loading checkpoints, sometimes, you might only want to load some of the parameters from the checkpoint, maybe for transfer learning or fine-tuning. in these cases, you can filter the state dict that is inside the `.pth` file before loading. here's a simple example of filtering which layers to load:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    # create an instance of the model
    model = MyModel(num_classes=10)  # adjust num_classes accordingly

    # let's assume you have saved a .pth file like this:
    # torch.save(model.state_dict(), "my_checkpoint.pth")

    # load the state_dict from the .pth file
    try:
        checkpoint = torch.load("my_checkpoint.pth")
        print("checkpoint loaded")

        # filter which parameters to load (example: load only resnet layers)
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('resnet.')}

        # load the filtered state_dict into the model
        model_state_dict = model.state_dict()
        # update only the resnet layers
        model_state_dict.update(filtered_checkpoint)
        model.load_state_dict(model_state_dict)
        print("model parameters loaded from filtered checkpoint")

        # set the model in eval mode if you are not going to train it
        model.eval()
        print("model in eval mode")

        # now you can do inference. lets create random data to see if it works.
        # remember to move your input data to the same device too.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")
        model.to(device)  # Move the model to the device too
        example_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            example_output = model(example_input)
        print(f"output tensor shape: {example_output.shape}")
        print("model prediction ok")
    except FileNotFoundError:
        print("checkpoint file not found, remember to save the model first!")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
```

this will load the layers named starting with `resnet.`, that's the base resnet layers. as you can see in this last snippet, you can also use the `.update()` method of the dictionary to merge the parameters into the model.

for more details i would recommend reading the pytorch documentation of course which is always up to date. specifically, the `torch.load()` and `torch.nn.module.load_state_dict()` documentation will save you a lot of time. also, there are great research papers online for example you can read the original resnet paper if you want to understand more about it, because sometimes you need to know how the model is implemented to actually load it correctly.

hope this clears things up, let me know if you have more questions or if i confused you with too much detail and code!
