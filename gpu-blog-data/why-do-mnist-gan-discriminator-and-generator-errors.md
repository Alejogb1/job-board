---
title: "Why do MNIST GAN discriminator and generator errors drop near zero after the first iteration?"
date: "2025-01-30"
id: "why-do-mnist-gan-discriminator-and-generator-errors"
---
The precipitous drop in discriminator and generator error to near zero after the first iteration in a MNIST GAN training run almost always indicates a mismatch in the scaling of the output of the generator and the input expectation of the discriminator.  This isn't inherently a problem with the GAN architecture itself, but rather a subtle implementation detail often overlooked by beginners.  I've encountered this issue numerous times during my work on generative models for handwritten digit classification, and consistent debugging revealed a root cause in the data normalization and the activation functions within the discriminator.

My experience points towards two primary culprits: a discrepancy between the generator's output range and the discriminator's input normalization, and the use of inappropriate activation functions that lead to saturated outputs.  Let's delve into the explanation and accompanying code examples to illustrate this.

**1. Data Normalization and Output Scaling:**

The MNIST dataset typically has pixel values ranging from 0 to 255.  Many implementations normalize the input images to the range [0, 1] or [-1, 1].  Crucially, the generator must output images in the *same* range.  Failure to do so will cause the discriminator to receive inputs significantly different from what it's trained on. If the generator, for example, outputs values in the range [0, 255] while the discriminator expects [0, 1], the initial discriminator will likely classify all generated images as highly improbable (resulting in near-zero generator loss). Conversely, if the discriminator is consistently receiving images close to its training data's mean and variance, it will quickly achieve near-perfect accuracy.

Similarly, the discriminator's final activation function should match the range of the target variable. If binary cross-entropy is used (as is typical), the output should be a probability between 0 and 1.   An unscaled output from the discriminator's final layer will lead to unstable training.


**2. Activation Function Saturation:**

Inappropriate activation functions in either the generator or the discriminator can lead to early convergence at zero loss.  ReLU and its variants, while popular, can suffer from a "dying ReLU" problem where neurons become inactive due to negative inputs.  This can severely restrict the model's expressive power, resulting in a generator that produces simplistic outputs which the discriminator easily classifies. The discriminator, seeing consistently "easy" samples, also converges rapidly to a state of near-perfect accuracy. Sigmoid functions, while commonly used for binary classification, suffer from gradient vanishing issues when saturated, further hampering training.

**3. Code Examples & Commentary:**

Let's examine three scenarios highlighting these issues and their solutions:

**Example 1:  Incorrect Output Scaling:**

```python
import torch
import torch.nn as nn

# Generator (incorrect scaling)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # ... layers ...
        self.final = nn.Linear(128, 784) # Outputs in range [-inf, inf]

    def forward(self, x):
        # ... layers ...
        x = self.final(x)
        return x # Incorrect: No scaling to [0, 1]

# Discriminator (expects [0, 1])
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # ... layers ...
        self.final = nn.Sigmoid()

    def forward(self, x):
        # ... layers ...
        x = self.final(x)
        return x

# Training loop (snippet):
# ...
fake_images = generator(noise)
# No scaling applied to fake_images
# ...
```
This example shows a generator that doesn't scale its output to the [0,1] range expected by the discriminator which uses a sigmoid activation function.  This directly leads to the problem described above. The solution is to add a scaling function like `torch.sigmoid` or `torch.tanh`  to the generator's final layer depending on the desired range.


**Example 2:  LeakyReLU in Generator causing limited expressivity:**

```python
import torch
import torch.nn as nn

# Generator (LeakyReLU might limit expressive power)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # ... layers ...
        self.final = nn.Linear(128, 784)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # ... layers ...
        x = self.final(x)
        x = self.activation(x) # Leaky ReLU might not be ideal here
        x = torch.sigmoid(x)
        return x

# Discriminator (...)
# ...
```
Here, the generator uses LeakyReLU.  While often beneficial, its use in the final layer of the generator might restrict the output range and limit its capacity to generate diverse images.  Consider using tanh instead, which ensures an output within the [-1, 1] range, consistent with common MNIST normalization.


**Example 3:  Correct Implementation:**

```python
import torch
import torch.nn as nn

# Generator (correct scaling)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # ... layers ...
        self.final = nn.Linear(128, 784)

    def forward(self, x):
        # ... layers ...
        x = self.final(x)
        x = torch.tanh(x) # Scale to [-1, 1]
        return x

# Discriminator (correct handling of probabilities)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # ... layers ...
        self.final = nn.Sigmoid()

    def forward(self, x):
        # ... layers ...
        x = self.final(x) # Output is in [0,1]
        return x

# Training loop (snippet):
# ...
fake_images = generator(noise)
#Correct scaling ensures discriminator receives inputs in the expected range
# ...
```

This example demonstrates a more robust implementation with appropriate scaling. The generator's output is scaled to [-1, 1] using `torch.tanh`, assuming the discriminator expects this range after normalization of the MNIST data.  The discriminator appropriately uses `nn.Sigmoid` to output probabilities between 0 and 1.


**4. Resource Recommendations:**

For a deeper understanding of GAN architectures and training, I highly recommend consulting research papers on GAN training stability and exploring the source code of established GAN implementations for MNIST.  Textbooks on deep learning provide solid foundational knowledge.  Furthermore, examining the documentation for the deep learning framework you are using (e.g., PyTorch, TensorFlow) is invaluable.  Careful attention to details in data preprocessing and model architecture is key to successful GAN training.  Analyzing the loss curves and generated images during training is crucial for debugging.
