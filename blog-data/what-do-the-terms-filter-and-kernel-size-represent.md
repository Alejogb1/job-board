---
title: "What do the terms filter and kernel size represent?"
date: "2024-12-15"
id: "what-do-the-terms-filter-and-kernel-size-represent"
---

alright, let's break down filters and kernel size. i’ve messed around with these enough to feel like i can explain it plainly, without any of that academic fluff.

so, when we talk about filters and kernel size, we’re usually knee-deep in the world of convolutional neural networks, or cnns. these things are basically the workhorses of image processing, and a bunch of other stuff too. forget fancy math for a second, at the core of a cnn lies the convolution operation. this is where filters and kernel size come into play.

think of a filter as a little matrix, a grid of numbers, a bunch of parameters if you will. the kernel size? that's just the dimensions of that filter. so, a 3x3 kernel means a filter that’s three rows tall and three columns wide. a 5x5 kernel is five by five and so on. simple enough.

now, what do these filters actually *do*? well, they’re designed to detect specific patterns in the input data. if we're talking images, these could be edges, corners, textures, or even more complex features. the numbers in the filter act as weights, and these weights are trained during the learning process. as the neural network sees more and more data, it adjusts these weights to better pick out those specific patterns that help it achieve the target objective of the network.

to make this a little clearer, imagine you're trying to find vertical edges in a grayscale image. a typical filter to do so could be:

```python
import numpy as np

vertical_edge_filter = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])

print(vertical_edge_filter)

```

this 3x3 filter has negative values on the left, and positive on the right. when this filter is applied to an image, areas with vertical changes in intensity will cause a strong response. that response is the result of the convolution calculation. this process involves sliding this filter across the image, and at each location, performing an element-wise multiplication between the filter's numbers and the corresponding numbers in that region of the image. the results are then added together to produce one single value representing the intensity of the feature at that location in the image.

the sliding thing, that’s called convolution or 'convolving' the filter. every time the filter goes to a different place on the input image, the same process is repeated and a single number is output to another array that is usually called 'feature map' or 'activation map'. these feature maps are an abstraction representation of the input where the original image gets transformed by the convolution and filters and hopefully useful information can be used to perform the target objective of the neural network.

i remember spending days working on an image recognition model back in the day. i kept getting very low accuracy scores. i thought my network architecture was the issue and was adding layers and doing all sorts of crazy stuff. then i went back to check my very first convolutional layer and boom, i had the wrong kernel sizes. it took me some time to understand that it wasn’t just the quantity of the filters that mattered, but the size too. i had a few very large filters and they were mostly just picking random things. in general the larger the size, the more the filter is receptive to the overall shape and bigger context.

now, let's say you want to detect a small pattern, you know, like a tiny little circle inside your input image. using a gigantic filter wouldn't make a lot of sense would it? well, it can actually work, but you'd be wasting a lot of computations and the training will take longer and the results will probably be bad. the filter would be looking at too much at a single go, not focusing on the specific small patterns of interest. you'd be better off using a small kernel size, maybe a 3x3 or even a 2x2, this way your network will be focusing on finer details instead of the overall big picture which at that stage would be not necessary. it's like using a magnifying glass versus a wide-angle lens.

now the number of filters matter too. each filter is capable of picking up different type of patterns so the more filters the better right? not really. the more filters the slower your network will be to train. you have to balance between the filter size, number of filters, and size of the input. here is an example of how you could define the convolutional filters in a popular deep learning framework such as tensorflow.

```python
import tensorflow as tf

# example: defining a conv2d layer in tensorflow
filters_count = 32 #number of filters
kernel_size = (3, 3) #filter dimensions
conv_layer = tf.keras.layers.Conv2D(filters=filters_count, kernel_size=kernel_size, activation='relu')

# you can also do this in pytorch
import torch
import torch.nn as nn
filters_count = 32 #number of filters
kernel_size = 3 #filter dimensions
conv_layer = nn.Conv2d(in_channels=3, out_channels=filters_count, kernel_size=kernel_size, stride=1, padding=1)

```

it’s not just about edge detection though. with enough filters, you can detect complex patterns like eyes, nose, mouths, wheels, and so on. you can even detect concepts such as 'cat-like' or 'car-like'. basically, you stack layers of filters one after the other in your network and these different filters at different layers can pick up different level of abstraction, from the more primitive patterns to more conceptual ones. it is a complex subject and it can be confusing, you should definitely check out some of the classic books about deep learning if you are really interested.

sometimes i feel like a network is just a big pile of filters all doing their own thing, and then a magic box does something with it and output a nice result, it’s kind of cool, and funny if you think about it.

another important aspect that a lot of people get confused about is when you have a color image, this means your image will have 3 channels, red, green and blue, and that will affect the way you define your filters. for grayscale images we generally have just one filter channel, but for color images you must define the number of channels in the filter to match your input data. the filter’s depth needs to match that of the input.

so, if your input is an rgb color image, your filters will have to have 3 channels too. each of these channels will be multiplied by the respective channel of the image data. the final output will sum the results of all the channels to produce one output feature map. the following example shows this better:

```python
import torch
import torch.nn as nn

#example of 3 channel filter
num_filters = 64
kernel_size = 5
# defining a convolution layer with 3 input channels
conv_layer = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=2)

# example of a greyscale image
conv_layer_greyscale = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=2)

# creating dummy input data for the color image
dummy_rgb_input = torch.randn(1, 3, 256, 256) #batch size of 1, 3 channels, 256x256 image

# creating dummy input data for the greyscale image
dummy_greyscale_input = torch.randn(1, 1, 256, 256) #batch size of 1, 1 channels, 256x256 image

# applying the convolution
output_color = conv_layer(dummy_rgb_input)
output_greyscale = conv_layer_greyscale(dummy_greyscale_input)

print(f"output shape for color image: {output_color.shape}") #should be torch.Size([1, 64, 256, 256])
print(f"output shape for greyscale image: {output_greyscale.shape}") #should be torch.Size([1, 64, 256, 256])
```

there are more things that matter to do with convolutions such as the stride and the padding, but that is a different subject for another day.

if you are seriously diving deep into this topic and want to go beyond my simple explanation and the practical code examples i just provided, i definitely recommend checking out deep learning books such as “deep learning with python” by françois chollet which is pretty straightforward, and also the “deep learning” book by goodfellow, bengio, and courville which is more theoretical and extensive if you are looking for in depth mathematics.
anyway, that’s pretty much it for filters and kernel sizes. it’s all about detecting patterns at different scales, and as always, experimentation is key. hope it helps.
