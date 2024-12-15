---
title: "Why is my CNN returning tokens instead or readable labels?"
date: "2024-12-15"
id: "why-is-my-cnn-returning-tokens-instead-or-readable-labels"
---

alright, so you've got a convolutional neural network (cnn) that's spitting out numerical tokens instead of, well, actual labels. i’ve been there, trust me. it’s a common gotcha, and it usually boils down to how the output layer is set up and how the model's final activation is handled. let’s walk through this.

first off, let me tell you about the time i was working on a project that involved classifying satellite images. it was supposed to identify different types of land cover like forests, urban areas, and water bodies. i trained my cnn, feeling pretty good about my data preprocessing and architecture. then the output was just a bunch of numbers, like `[3, 1, 2, 0, 3]` – no labels, nothing meaningful. i spent a whole evening thinking i had messed up the data feeding process, and i had, but not exactly where i thought i had. that’s when i realized i had fallen into this very same trap of the output layer and the interpretation of the results.

the cnn itself, at its core, is all about learning features. the convolutional layers learn spatial patterns, pooling layers reduce dimensionality, and so on. but ultimately, the final layer before the activation (or even the activation itself) determines what the network is trying to predict. a typical cnn setup, or at least, the one that i had initially set up was something like this:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes) # Example image size
        # No softmax here by default
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Dummy input
dummy_input = torch.randn(1, 3, 28, 28)
num_classes = 4
model = SimpleCNN(num_classes)
output = model(dummy_input)
print(output) # this prints the logits (or the raw un-normalized predictions)
```

notice that the last layer `fc1` has a number of output neurons equal to `num_classes`. that is perfectly fine, and it is the standard in many cases. however, notice how the code does not provide any activation function. this means the output is raw scores, or "logits," and not probabilities. these are precisely the numerical tokens you are getting. these tokens are not your final labels. they represent how confident the network is about each class.

here’s what usually happens: during training you pass these logits through a loss function, like `crossentropyloss`, which expects raw logits. this loss function implicitly applies a `softmax` to these logits internally. the problem happens when you're testing or making predictions and you don't perform the `softmax` or use the proper argmax yourself.

so, how do we fix it? it's simple really, we have multiple options, the first one is during inference, add an activation to the output of the model, and the second is to add an activation to your model itself. let's see how it works in practice:

**option 1: performing the activation in the inference step:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Dummy input
dummy_input = torch.randn(1, 3, 28, 28)
num_classes = 4
model = SimpleCNN(num_classes)
output = model(dummy_input)
probabilities = F.softmax(output, dim=1) # this one is the key step, it converts the logits to probabilities
predicted_classes = torch.argmax(probabilities, dim=1) # get index with the highest probability
print(predicted_classes) # prints the index corresponding to the predicted label
```

here, we’ve kept the model untouched, and applied a `softmax` function and an `argmax` to the outputs, directly before extracting the label, the code now works like you expect, and you get an index corresponding to a specific label.

**option 2: performing the activation in the model:**

sometimes you just want to apply the final activation in the model itself, instead of the inference step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# Dummy input
dummy_input = torch.randn(1, 3, 28, 28)
num_classes = 4
model = SimpleCNN(num_classes)
output = model(dummy_input)
predicted_classes = torch.argmax(output, dim=1) # get index with the highest probability
print(predicted_classes) # prints the index corresponding to the predicted label
```
notice how we added a softmax layer as part of the model and the output is now a list of probabilities. we still need to extract the actual label by using an `argmax` operation.

so, to sum it up, a cnn gives you numerical scores, which are the logits. to get probabilities you apply a `softmax`. once you have probabilities you should get the label that corresponds to the highest probability using an `argmax`. most likely, this is the step you are missing.

another thing i’ve seen people trip up on is, even if you get the probabilities using softmax (or logits using the model itself), you might still get numerical indices as the output. and the reason for that might be that you haven’t mapped your numerical index to an actual label. this usually involves having an array of string labels that matches your classes. that was the case when i got the satellite images classified, i forgot that the model was outputting an array of indexes not the labels itself, duh.

for a more in-depth look into convolutional neural networks, i’d recommend "deep learning" by ian goodfellow, yoshua bengio, and aaron courville. it’s a bible for all things deep learning. also, for the specifics of pytorch, the official pytorch documentation is a great resource. there’s also a very nice introductory book called “programming pytorch for deep learning” by ian hicks, which might help a lot with the details of building your model.

one last thing, don't beat yourself up about it. we've all been there, and sometimes a small mistake can have a big impact on the output, just keep at it, and remember to always test all the layers and be sure that all the operations being performed are what you expect, that's one thing that i learned the hard way. we've all been there. it's funny how many times i've spent hours looking for an error only to find i was using the wrong function all along.
