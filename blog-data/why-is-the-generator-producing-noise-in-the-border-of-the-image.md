---
title: "Why is the generator producing noise in the border of the image?"
date: "2024-12-15"
id: "why-is-the-generator-producing-noise-in-the-border-of-the-image"
---

well, i’ve seen this one a bunch of times, and yeah, it's usually a pain, especially when you're trying to get clean outputs. image generation, huh? the dream, until the edges look like static tv. here's what's most likely going on, and some stuff i've tried that usually helps.

first, let’s talk about the root cause: it’s all about how the generator “sees” the edge of your image. most generators, especially those based on convolutional neural networks (cnns) are built to process tiles or patches. basically, a smaller section of your image at a time, not the whole thing at once. when they are processing a part of the image that is not at the edge everything is fine, but when the process reach the edge of the image the network often faces boundary issues.

a convolution operation, think of it as a little window that moves across the image, has a kernel, or the weight matrix used to transform an input window to an output, when that kernel hits the edge, you are missing values to perform the operation, this effect can result in the network learning the boundary as an special place, because the values at the border are usually incomplete, if you have an input of size 200 by 200 and a window of 5 by 5 when computing the convolution the edge will be of size 196 by 196. this creates this border effect because the values are treated differently. sometimes is the padding that is used that create that, sometimes it is the missing data itself.

the most common reason for this noise, is how boundary conditions are handled during the training phase. networks don’t like sudden changes, they prefer smooth transitions. if your image borders were handled in a non-smooth fashion, the generator learned that discontinuity, and that it is indeed something that needs to generate. think about zero padding: you add a lot of black pixels to the edge of the image, in the convolution process, and the network will learn to see these black border. because those are usually the cases the network will learn to generate noisy artifacts around that area to replicate the padding.

another factor is the receptive field of the generator's network. it's like how far the network "sees" when analyzing an image. if the receptive field of a convolutional layer is large, it might be more susceptible to edge effects because the information required to generate a coherent edge is outside of its field, or some missing information creates the effect. it will hallucinate some data because it thinks that is missing values or information.

now, i've gone through this personally. back in my early days, trying to train a style transfer network, the output looked amazing in the center but had those horrible borders. i tried different kinds of padding strategies, and finally, one of the methods works better than others. let me show you the code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as f

class mygen(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='replicate')
        self.bn = nn.batchnorm2d(out_channels)
        self.relu = nn.relu()

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x
```

this is using ‘replicate’ padding in pytorch. it basically takes the pixel value from the edge and extends it outwards. it’s pretty basic, but surprisingly effective for reducing border artifacts, i mean, for me it was a game changer. i was using zero padding before and oh boy, that was ugly. the point is that each padding will have an effect in the borders so you should try all of them until you find the one that works for you. you can see that i’m also using batch normalization and relu, which are good practices in deep learning architectures.

another option is using reflection padding, similar to replicate but it flips the pixels at the border:
```python
import torch
import torch.nn as nn
import torch.nn.functional as f

class mygen_reflection(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', padding_mode='reflect')
        self.bn = nn.batchnorm2d(out_channels)
        self.relu = nn.relu()

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

```
this is another trick i used in the past when the replicate was not enough, reflection padding often works when the network has trouble smoothing the edge.

also, some generators work with overlapping tiles. this can reduce border issues, because the edges of one tile are not the edges of the whole image, so they don’t represent an actual border. i think i read about this in some paper about image inpainting long time ago, something about ‘context encoders’. it's when the network predicts missing parts of an image. that same idea can be applied in generator networks too.

finally, sometimes it’s not just the padding; it’s the lack of sufficient training data or poorly defined generator. ensure your training set has a wide variety of images and that your generator architecture can learn complex patterns. i remember i was generating faces and using just one dataset, when i added more faces the borders became more smooth. another thing that i tried, and helped me, was adding a loss function that penalizes differences at the border, this ensures that borders are not treated different than the center. that was a lot of code, i'm not going to show an example of this last one, but it might be something useful that you can try on your own.

and the best advice i could give is this: experiment with different padding strategies, and try using a network with better receptive field, it could be an upsampling layer that uses strided transposed convolution. also, data augmentation helps, but it needs to be applied carefully. sometimes you introduce artifacts in the data and that is not good.

oh, and one time, i spent hours debugging, only to realize i was accidentally scaling the images to a different resolution before passing it to the generator, creating the issues. it turned out i had a bug, who would have guessed, right?

for more information, i highly suggest you read some papers, one is ‘image inpainting for irregular holes using context priors’, and for a general overview of cnn i recommend the book “deep learning” by goodfellow, bengio, and courville. it goes over this topics very clearly.

hope that helps, and good luck debugging. this border stuff is common, so you're not alone, happy coding!
