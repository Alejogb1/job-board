---
title: "How to implement a Residual Unit with ShakeDrop Depth regularization?"
date: "2024-12-14"
id: "how-to-implement-a-residual-unit-with-shakedrop-depth-regularization"
---

alright, so you’re asking about implementing a residual unit with shake-drop regularization. this is a pretty interesting topic, and it touches on a couple of cool ideas from deep learning. i’ve tinkered with this kind of stuff quite a bit, so let me share my experience.

basically, what we're talking about here is combining the power of residual connections with a regularization technique called shake-drop. if you're already familiar with resnets, you'll know they use residual connections to allow very deep networks to learn more effectively. and shake-drop, it’s a way of randomly dropping connections and scaling the output during the training, which reduces overfitting and improves robustness. it’s a bit more nuanced than simple dropout, because it involves a forward and backward pass modification.

i remember when i first encountered residual connections, i was amazed at how well they worked. i was working on an image classification problem at the time, and i had hit a wall. my models weren’t improving beyond a certain depth. i was pulling my hair out, until i saw a paper on resnets. it felt like seeing a light in the dark tunnel, those skip connections were a game changer. then, later on when i learned about shake-drop, it was like a second boost. these regularization tricks, man, they make a big difference.

so, how do we implement a residual unit with shake-drop? let’s break it down.

first, we need to define a standard residual unit. typically, this involves a series of convolutional layers, batch normalization layers, and activation functions, usually relu. then, we add the skip connection, which simply adds the input to the output of the convolutional branch before applying the final activation.

here’s a basic python code using pytorch for a simple resunit, no shake-drop yet:

```python
import torch
import torch.nn as nn

class BasicResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
           self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
```

that's a pretty standard setup for a basic residual unit. you’ll see the two convolution layers, batch norm after each, the relu activation, and then the shortcut to add the original input to the convolutional output. the shortcut also has a conv and a batchnorm, which makes sure the dimensions of both tensors match in case we need to resize or change channel numbers in the process.

now, let's add in the shake-drop part. this is where it gets a bit more interesting. shake-drop modifies how we perform the forward pass, and it needs a tiny adjustment during the backpropagation as well, it is not just about setting a dropout probability. for that, we will need a parameter ‘shake_prob’. during the forward pass, we compute the output of the conv branch, then we scale it by a random value and also the identity connection by a second random value. these random values are sampled in the [0,1] range using a specific distribution. the sum of both scaling values should be equal to 1 so the identity or shortcut connection and the feature transformations both participate in the sum.

here’s the modified unit using shake-drop:

```python
import torch
import torch.nn as nn
import torch.nn.functional as f
import random

class ShakeDropResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shake_prob=0.5):
        super(ShakeDropResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shake_prob = shake_prob

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
           self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.training:
           # shake-drop
            alpha = random.random() * (1 - self.shake_prob)
            beta = 1 - alpha
            out = (out * alpha + (self.shortcut(residual) * beta )) #the scaling is applied before the relu
        else:
            out += self.shortcut(residual)

        out = self.relu(out)
        return out

```

notice the changes? we added a `shake_prob` parameter to the init function. then, in the forward pass we add the conditional statement based on training mode. during training we generate a random alpha value based on a uniform distribution, and we scale both branches of the sum. during the evaluation stage we bypass this step to ensure we use the whole model.

i once made a silly error with shake-drop, scaling both branches without the proper sum to 1, and i spent a whole afternoon figuring it out, the learning curve can be like that sometimes.

finally let's show how you can instantiate a sequential model made of these shake-drop resunits.

```python
import torch
import torch.nn as nn

class ShakeDropModel(nn.Module):
  def __init__(self, num_classes=10):
        super(ShakeDropModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res_block1 = self._make_res_block(64, 64, 3)
        self.res_block2 = self._make_res_block(64, 128, 4, stride=2)
        self.res_block3 = self._make_res_block(128, 256, 6, stride=2)
        self.res_block4 = self._make_res_block(256, 512, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

  def _make_res_block(self, in_channels, out_channels, num_units, stride=1):
        layers = []
        layers.append(ShakeDropResUnit(in_channels, out_channels, stride))
        for _ in range(1, num_units):
            layers.append(ShakeDropResUnit(out_channels, out_channels))
        return nn.Sequential(*layers)

  def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    #create a simple input batch
    input_tensor = torch.rand(1, 3, 32, 32)
    model = ShakeDropModel(num_classes=10)
    output = model(input_tensor)
    print("output shape: ", output.shape)
```

this shows how to create a simple network that uses the previously described shake-drop resunits. notice how the function `_make_res_block` helps building a group of units one after the other. a cool trick i’ve learned is to play with the number of units per block and the number of filters per unit as it usually affects dramatically the final performance.

if you're looking to dive deeper, i recommend checking out the original shake-drop paper; "shake-shake regularization". there are also a few good survey papers on regularization techniques that are worth reading. those will give you a broader view of the different tricks used in deep learning. or, if you want a more hands on experience i would recommend reading "deep learning with pytorch" it is a great way to learn about this topic with practical examples.

implementing a residual unit with shake-drop regularization is straightforward once you understand the basic concepts behind both the resnets and the shake-drop approach. it's all about understanding how the network uses skip connections and how to modify the forward pass during training to enhance the learning capabilities of your model, good luck.
