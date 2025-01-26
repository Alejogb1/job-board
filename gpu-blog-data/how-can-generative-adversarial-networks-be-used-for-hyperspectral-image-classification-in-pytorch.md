---
title: "How can generative adversarial networks be used for hyperspectral image classification in PyTorch?"
date: "2025-01-26"
id: "how-can-generative-adversarial-networks-be-used-for-hyperspectral-image-classification-in-pytorch"
---

Hyperspectral image classification presents a unique challenge due to the high dimensionality and complex spectral signatures of each pixel. Standard convolutional neural networks often struggle with this data, particularly when labeled data is limited. Generative Adversarial Networks (GANs), typically used for image generation, offer a pathway to address this issue by augmenting training data and potentially improving feature representation. My experience adapting GANs for hyperspectral data highlights the intricacies involved, particularly regarding architecture selection and training stability.

The core idea revolves around training a generator to create synthetic hyperspectral samples that mimic the real data distribution, and a discriminator to distinguish between real and synthetic data. The generator learns to create plausible hyperspectral images, ideally capturing the inherent spectral characteristics of different classes. This augmented dataset, when combined with real labeled data, can improve the performance of classification models. The classification model, then, typically uses the extracted features from real and generated images.

Specifically for hyperspectral data, a few modifications to the standard GAN architecture are often necessary. Firstly, due to the high dimensionality of the spectral data, the generator and discriminator networks need to accommodate input tensors with many channels, corresponding to the spectral bands. In the context of PyTorch, this requires careful adjustment of convolutional layer parameters and input shapes. Secondly, the spectral nature of the data necessitates a model architecture that can capture both spatial and spectral correlations.

The most effective approach I have found is to use a 3D convolutional architecture, wherein filters process both spatial and spectral dimensions simultaneously. This allows the network to capture correlations between neighboring pixels both spatially and spectrally. Further, the choice of activation functions is crucial. Leaky ReLU activations, instead of standard ReLU, typically reduce the risk of vanishing gradients during the backpropagation through deep networks. Batch Normalization layers also prove beneficial for training stability.

Here are three examples illustrating different architectural choices and considerations in PyTorch:

**Example 1: Basic 3D Convolutional GAN for Hyperspectral Generation**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Generator3D(nn.Module):
    def __init__(self, latent_dim, num_channels, output_shape):
        super(Generator3D, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.output_shape = output_shape
        self.main = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(128, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(64, num_channels, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator3D(nn.Module):
    def __init__(self, num_channels, input_shape):
        super(Discriminator3D, self).__init__()
        self.num_channels = num_channels
        self.input_shape = input_shape
        self.main = nn.Sequential(
            nn.Conv3d(num_channels, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

# Assume data_shape to be (num_channels, height, width).
data_shape = (100, 32, 32)  # Example: 100 spectral bands, 32x32 spatial dimensions
latent_dim = 100
generator = Generator3D(latent_dim, data_shape[0], data_shape)
discriminator = Discriminator3D(data_shape[0], data_shape)

# Example dummy input
batch_size = 32
z = torch.randn(batch_size, latent_dim, 1, 1, 1)

# Example optimizer and loss function
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Example training loop (simplified)
real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)

for _ in range(10): # simplified training loop
    # Train Discriminator
    optimizer_D.zero_grad()
    real_data = torch.rand(batch_size, *data_shape)  # Example real batch
    output_real = discriminator(real_data)
    loss_real = criterion(output_real, real_labels)
    loss_real.backward()

    fake_data = generator(z)
    output_fake = discriminator(fake_data.detach())
    loss_fake = criterion(output_fake, fake_labels)
    loss_fake.backward()
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    output_fake = discriminator(fake_data)
    loss_generator = criterion(output_fake, real_labels)
    loss_generator.backward()
    optimizer_G.step()
```

This first example showcases a basic implementation of a 3D convolutional GAN. The generator upsamples from a latent vector to generate hyperspectral data, while the discriminator downsamples and outputs a probability indicating whether the input is real or generated. Note the use of `ConvTranspose3d` in the generator for upsampling and `Conv3d` in the discriminator for downsampling. Batch normalization is essential for stable training. The output of the generator uses the `tanh` function to ensure the generated samples have values in the range of -1 to 1. A simplified training loop is included for demonstration purposes; actual training typically requires significantly more iterations.

**Example 2: Incorporating Spectral Attention**

```python
import torch
import torch.nn as nn

class SpectralAttention(nn.Module):
    def __init__(self, num_channels):
        super(SpectralAttention, self).__init__()
        self.conv = nn.Conv3d(num_channels, num_channels // 16, kernel_size=1)
        self.fc = nn.Linear(num_channels // 16, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w, d = x.shape
        x_reduced = self.conv(x)
        x_reduced_pooled = torch.mean(x_reduced, dim=(2,3,4))
        weights = self.fc(x_reduced_pooled).view(b, c, 1, 1, 1)
        return self.sigmoid(weights) * x

class GeneratorAttn(nn.Module):
    def __init__(self, latent_dim, num_channels, output_shape):
        super(GeneratorAttn, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.output_shape = output_shape
        self.main = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralAttention(128), # Spectral attention added here
            nn.ConvTranspose3d(128, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
             SpectralAttention(64), # Spectral attention added here
            nn.ConvTranspose3d(64, num_channels, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.main(z)
        return x

class DiscriminatorAttn(nn.Module):
    def __init__(self, num_channels, input_shape):
         super(DiscriminatorAttn, self).__init__()
         self.num_channels = num_channels
         self.input_shape = input_shape
         self.main = nn.Sequential(
             nn.Conv3d(num_channels, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.LeakyReLU(0.2, inplace=True),
             SpectralAttention(64), # Spectral attention added here
             nn.Conv3d(64, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.LeakyReLU(0.2, inplace=True),
             SpectralAttention(128), # Spectral attention added here
             nn.Conv3d(128, 1, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1)


# Assume data_shape to be (num_channels, height, width)
data_shape = (100, 32, 32)
latent_dim = 100
generator_attn = GeneratorAttn(latent_dim, data_shape[0], data_shape)
discriminator_attn = DiscriminatorAttn(data_shape[0], data_shape)

# Training would follow a similar structure to Example 1.
```

This example introduces spectral attention modules within the generator and discriminator. The attention modules focus on emphasizing the most important spectral channels, dynamically reweighting channels based on their information content. The `SpectralAttention` module first reduces the channel dimensionality using a 1x1 convolution, pools across spatial dimensions, applies a fully connected layer to learn channel weights, and then scales the input with these learned weights. This allows the network to better capture informative spectral features during both generation and discrimination.

**Example 3: Utilizing Conditioned GANs for Class-Specific Generation**

```python
import torch
import torch.nn as nn

class GeneratorConditional(nn.Module):
   def __init__(self, latent_dim, num_channels, num_classes, output_shape):
        super(GeneratorConditional, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.main = nn.Sequential(
             nn.ConvTranspose3d(latent_dim*2, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.BatchNorm3d(128),
             nn.LeakyReLU(0.2, inplace=True),
             nn.ConvTranspose3d(128, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.BatchNorm3d(64),
             nn.LeakyReLU(0.2, inplace=True),
             nn.ConvTranspose3d(64, num_channels, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
             nn.Tanh()
         )


   def forward(self, z, labels):
         embedded_labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3).unsqueeze(4)
         z = torch.cat((z, embedded_labels), dim=1)
         return self.main(z)

class DiscriminatorConditional(nn.Module):
    def __init__(self, num_channels, num_classes, input_shape):
        super(DiscriminatorConditional, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.label_embedding = nn.Embedding(num_classes, input_shape[0] * input_shape[1] * input_shape[2])

        self.main = nn.Sequential(
            nn.Conv3d(num_channels+1, 64, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        embedded_labels = self.label_embedding(labels).view(-1,1, *self.input_shape)
        x = torch.cat((x,embedded_labels), dim=1)
        return self.main(x).view(-1, 1)

# Assume data_shape to be (num_channels, height, width)
data_shape = (100, 32, 32)
latent_dim = 100
num_classes = 10 # Number of classes for the hyperspectral dataset

generator_conditional = GeneratorConditional(latent_dim, data_shape[0], num_classes, data_shape)
discriminator_conditional = DiscriminatorConditional(data_shape[0], num_classes, data_shape)

# Example training will require label inputs for both generator and discriminator
```

This example demonstrates a conditional GAN (cGAN), where both the generator and discriminator are conditioned on class labels. The generator takes a latent vector and a class label as input, allowing it to generate class-specific samples. The labels are embedded via an embedding layer. The label embedding is then concatenated with the latent input of the generator before processing by the convolutional layers. The discriminator also receives both the hyperspectral image data and a label; similar label embedding and concatenation strategies are applied. The use of cGAN allows for more fine-grained control over the generated data and is beneficial when seeking to augment specific underrepresented classes.

For further study, I recommend exploring academic literature on hyperspectral image processing, focusing on deep learning applications. Research papers on GAN architectures are also crucial. Additionally, familiarize yourself with the PyTorch documentation for 3D convolutions and related functions and the various normalization and activation functions. Open-source repositories containing pre-trained models can be useful for benchmarking purposes. A thorough understanding of these resources will allow for the development of effective GANs for hyperspectral image classification.
