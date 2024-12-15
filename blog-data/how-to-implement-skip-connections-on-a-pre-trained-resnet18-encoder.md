---
title: "How to implement Skip Connections on a Pre-trained Resnet18 Encoder?"
date: "2024-12-15"
id: "how-to-implement-skip-connections-on-a-pre-trained-resnet18-encoder"
---

alright, let's talk skip connections on a pretrained resnet18 encoder. i've been down this rabbit hole before, and it's a common enough issue, so i'll try to break it down like i would for a fellow developer banging their head against the wall.

first, let's get the basics straight. skip connections, or residual connections, are about adding the original input of a block to its output. this helps gradients flow more easily through the network, particularly in deep networks like resnet architectures. it's why resnets can be so deep without exploding or vanishing gradients. resnet18, even though it's not the deepest resnet, still benefits greatly from these. we want to preserve this when using it as an encoder, but you want to add these skip connections to your own decoding part not the resnet part.

the key here is that you’re not *modifying* the pretrained resnet itself. you're going to treat that resnet as a feature extractor, and then build your decoding/downstream network *around* it, incorporating the skip connections from the resnet outputs into your new decoder. imagine resnet18 being a black box producing multiple stages of output, and you are going to "grab" the outputs from some layers and use them as an input to a decoder network.

i've personally run into trouble when i tried to directly modify the resnet weights. this is definitely not needed and it just makes things unnecessarily complicated. it's better to work *with* the output stages of the resnet rather than attempting to change the existing structure that’s already well-trained. you'll probably want the intermediate feature maps of the resnet instead of just its last layer output (classification) because they have valuable spatial information useful for a lot of decoding tasks like image segmentation.

here's how i usually approach it, with some python code snippets to show you how it works. i will be using pytorch but the concept should be adaptable to other frameworks.

first, you load the pretrained resnet model. let's make sure we get rid of the final fully connected layer (classifier) so we use it just as an encoder:

```python
import torch
import torchvision.models as models

def get_resnet18_encoder():
    resnet = models.resnet18(pretrained=True)
    # remove the last fully connected layer
    modules = list(resnet.children())[:-2] # we will want the features after the avgpool
    encoder = torch.nn.Sequential(*modules)
    return encoder

encoder = get_resnet18_encoder()
```
the key is that we slice off the last two layers (avgpool and fc). the average pooling layer reduces all the spatial information and our last fc layer is not what we need. now we have an encoder module that outputs intermediate features.

now, let’s think about how to access those intermediate features. since the resnet is a sequential model it's tricky to get the outputs at each stage, so one usual approach is to define a forward hook to register the features of each stage we need. here's how i normally do this:

```python
import torch.nn as nn

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_resnet18_encoder()
        self.outputs = {}
        self.features = []

        def get_features(name):
          def hook(model, input, output):
            self.outputs[name] = output
            self.features.append(output)
          return hook

        self.encoder[0].register_forward_hook(get_features('layer0')) # conv1
        self.encoder[4].register_forward_hook(get_features('layer1')) # layer1
        self.encoder[5].register_forward_hook(get_features('layer2')) # layer2
        self.encoder[6].register_forward_hook(get_features('layer3')) # layer3
        self.encoder[7].register_forward_hook(get_features('layer4')) # layer4


    def forward(self, x):
        _ = self.encoder(x) # we use the forward pass for hooks to execute and save the features
        return self.features

feature_extractor = ResNet18FeatureExtractor()

# dummy input
dummy_input = torch.randn(1, 3, 256, 256)
features = feature_extractor(dummy_input)

for f in features:
    print(f.shape)
```
notice how we register a forward hook to every layer we need. now every time we do a forward pass with a dummy input, we will extract the feature maps. the output for this dummy input will print the shapes of all extracted features. the first layer will be (1,64,128,128), the next will be (1,64,64,64), (1,128,32,32), (1,256,16,16), (1,512,8,8). now that you have your encoder outputs with their corresponding scales (downsampled).

finally, you'll build your decoder, making sure to incorporate the extracted features from the encoder using the skip connections. for a simple decoding part, we can build convolutional layers to do upsampling. here's how it could look:

```python
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # upsamples last feature of resnet
        self.conv1 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1) # concatenates skip
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # upsamples conv1
        self.conv2 = nn.Conv2d(128+128, 128, kernel_size=3, padding=1) # concatenates skip
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # upsamples conv2
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1) # concatenates skip
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # upsamples conv3
        self.conv4 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1) # concatenates skip
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features):
        # features: [layer0, layer1, layer2, layer3, layer4]
        layer4 = features[-1] # 512 x 8 x 8
        layer3 = features[-2] # 256 x 16 x 16
        layer2 = features[-3] # 128 x 32 x 32
        layer1 = features[-4] # 64 x 64 x 64
        layer0 = features[-5] # 64 x 128 x 128


        x = self.up1(layer4)  # 256 x 16 x 16
        x = torch.cat((x, layer3), dim=1) # 256+256 x 16 x 16
        x = self.conv1(x) # 256 x 16 x 16
        x = self.up2(x) # 128 x 32 x 32
        x = torch.cat((x, layer2), dim=1) # 128 + 128 x 32 x 32
        x = self.conv2(x) # 128 x 32 x 32
        x = self.up3(x) # 64 x 64 x 64
        x = torch.cat((x, layer1), dim=1) # 64 + 64 x 64 x 64
        x = self.conv3(x) # 64 x 64 x 64
        x = self.up4(x) # 64 x 128 x 128
        x = torch.cat((x, layer0), dim=1) # 64 + 64 x 128 x 128
        x = self.conv4(x) # 64 x 128 x 128
        x = self.final_conv(x) # num_classes x 128 x 128
        return x

decoder = Decoder()

output = decoder(features)
print(output.shape)
```
the output shape should be (1, num\_classes, 128, 128). this example shows a simple decoding with convtranspose upsampling but you could add more layers.

so, the gist is: use the resnet as a black box feature extractor, get the intermediate feature maps through forward hooks, and then use those features to build your custom decoder. you're not modifying the resnet but extracting the intermediate layers and adding the skip connections in the decoder network.

that's it. you've got a resnet18 encoder, and are using its intermediate outputs as skip connections for your own decoder. this structure is quite common for image segmentation or image to image translation tasks but it is adaptable to many problems.

now for a quick joke: why did the convolutional neural network break up with the fully connected layer? because they felt they weren't seeing eye to eye, it was too much of a flat relationship.

as for resources: instead of providing links, i'd highly recommend you to look into the original paper that presented resnets, *deep residual learning for image recognition*. its a must read if you are using resnets. also a book that is good for this and many other deep learning tasks is *hands-on machine learning with scikit-learn, keras and tensorflow* by aurélien géron. this book has a good section that goes in depth about resnets and it is a good introduction to how to use convolutional neural networks. another good book that is also very helpful is *deep learning with python* by françois chollet which has lots of great examples and explanations about convolutional architectures, specially with pytorch.

i hope this explanation helps, let me know if you have more questions.
