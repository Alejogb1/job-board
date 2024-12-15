---
title: "How to do a ResNet for 32x32 images?"
date: "2024-12-15"
id: "how-to-do-a-resnet-for-32x32-images"
---

alright, so you're tackling resnets with 32x32 images. been there, done that. it's a bit different from the usual imagenet stuff, but totally doable. let's break it down.

first off, the typical resnet architectures, like resnet-50 or resnet-101, are designed for larger images, say, 224x224. these have a pretty deep series of convolutional layers, downsampling and pooling that bring the resolution down. when you try to apply these directly to tiny 32x32 images, you run into issues. you end up with a feature map that's way too small way too soon, and you kinda kill a lot of potentially useful information.

so, the key is to scale down the resnet architecture to suit your input size. you can't just blindly apply the usual thing. instead you need to adjust parameters and layers to fit smaller image sizes. let's think about how you can do this by modifying the usual resnet blocks. usually a resnet starts with a conv layer with larger kernel (eg 7x7). then you usually have max pooling with a kernel of 3x3. this is a common pattern. but if you have 32x32 images, that will rapidly downsample to something like 14x14 after the first operation, and then again to something like 7x7 with the max pooling. it's a bit too much downsampling.

so, what we do is reduce the kernel size in the first conv layer from 7x7 to say 3x3 or even 1x1 and get rid of that pooling layer after the initial conv. this helps preserve spatial resolution longer.

i remember when i first tried to get this working. i was using a resnet-34 architecture with the standard weights. after a few layers i was getting 1x1 feature maps, and i thought i had a bug. turns out the architecture was just too big for my data. it was a dataset of tiny handwritten number digits i was playing around. i had to basically rebuild it from scratch.

here’s how i would define my initial convolution layer for 32x32 inputs. it’s a pytorch snippet, but the idea applies to most frameworks:

```python
import torch.nn as nn

class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
```

another important thing when working with smaller images is to reduce the number of layers and the number of feature maps in each layer. deep resnets like 50, 101 and 152 are overkill. something simpler will work better. think resnet-18 or even a custom one with fewer residual blocks. the idea is to not overparameterize with a tiny image.

also, you want to adjust the stride in the convolutional layers to control downsampling. in larger resnets, you usually have a stride of 2 in some conv layers. for 32x32 images, you may only have stride 1 for most of the time and only use stride 2 in a few parts of the network or even eliminate stride 2 entirely. this will keep the feature maps larger.

the residual blocks themselves remain pretty much the same structure, but the number of feature maps (the channel dimensions) will be less. it's a trick we learned the hard way. it's like trying to fit a ship in a bathtub. doesn't quite work well. reduce it to a smaller boat, then you're all good. it's less data, so the models require fewer parameters.

here’s an example of a basic residual block:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
```

the key thing here is the 'stride' parameter in the first conv layer of the block, and also the shortcut connection to handle the possible change in the number of channels or spatial resolution.

then putting all together a very simplified resnet-like architecture you might use would look like this. in this simple example i keep the number of channels at 64. but feel free to experiment with that.

```python
class ResNet32(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet32, self).__init__()
        self.initial_conv = InitialConv(in_channels, 64)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 64, num_blocks=2, stride=2)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_conv(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)


        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```
in this example we are just using three residual blocks at each stage. and we use a stride of 2 after each stage to downsample the spatial dimensions of the feature maps.

last thing, about resources: if you want to deep dive, you should definitely read the original resnet paper. it's called "deep residual learning for image recognition." it will help you understand the core idea of the blocks. there is also the torchvision library on pytorch which is a good source of information and models but you may not find much for 32x32 images. but you can look into the code to see how the blocks are implemented. for more general stuff on convolutional neural networks and related concepts, i always recommend the deep learning book by goodfellow et al. it's a massive book, but the chapter on convolutional networks is very informative. just don’t expect that it will have code ready for you, but it's the right source for the theory. if it does, it might be outdated. i also think you need to get familiar with the pytorch tutorials for building up neural networks. they explain all the core concepts very well.

i’ve been working with neural networks for almost a decade and resnets are pretty cool. once you get the basics down, you'll be able to build and adapt them to most of the computer vision tasks. it is less scary than it looks. i almost gave up when i used a very deep network for a small image back when i first started.

finally, remember to normalize your input data. this will help a lot with the training stability. you would be surprised how many problems that solves. training models on very small datasets is harder than bigger ones. good luck with it, and remember to keep your batch size reasonably high if you can. a general advice i can give you is "if it works don't touch it", but also "if you do not measure, you do not know". so test your architecture with different combinations of hyperparameters to see which one suits your application.

by the way... did you hear about the programmer who quit his job? he just didn't get arrays!
