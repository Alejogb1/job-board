---
title: "Why is SuperResolution on MNIST datasets failing?"
date: "2025-01-30"
id: "why-is-superresolution-on-mnist-datasets-failing"
---
Super-resolution (SR) applied to the MNIST dataset, while seemingly straightforward given the low resolution and simple structure of the digits, often yields unsatisfactory results, primarily due to an inherent conflict between the training data characteristics and the operational assumptions of many SR algorithms. Specifically, MNIST digits are, by design, binary and low in information content. SR algorithms, particularly those based on deep learning, typically learn to reconstruct high-frequency details, such as texture or subtle edges, that are largely absent in downsampled MNIST images. This fundamental mismatch is the root cause of observed failures.

The challenge stems from how SR models are conventionally trained. These networks are typically optimized to upsample images that have undergone a process of resolution reduction that preserves, though at a lower resolution, the critical high-frequency components. In these cases, the low-resolution inputs often retain sufficient information for the network to "infer" the higher-resolution details. However, with MNIST, the common downsampling methods (e.g., bicubic, bilinear) used to generate low-resolution training examples significantly blur and simplify the digit structure, effectively discarding the very information that a super-resolution model is intended to recover. This results in a training regime where the network is learning to generate an image that resembles the high-resolution target, not reconstruct the lost high-frequency detail from the low-resolution input.

To illustrate this, consider that a bicubic downsampling of an MNIST digit often yields a blurred version that can appear almost identical to another, slightly different, downsampled digit. The network struggles to differentiate the subtle nuances between the low-resolution images, therefore producing very similar, but generally incorrect, high-resolution versions. During the training process, the loss function minimizes the per-pixel difference between the predicted high-resolution image and the ground truth. This optimization, while useful in datasets with rich high-frequency information, effectively guides the model to generate a general form of the digit, rather than reconstructing it from its degraded representation.

The model often ends up producing a “smoothed” average of possible upscaled versions. Because the loss function penalizes any variation from the ground truth high-resolution image, the model learns to output the most common representation which will minimize loss across all possible inputs within a given class. The subtle information that distinguishes different instances of, for example, the digit "3", is lost in the downsampling stage, and therefore cannot be reconstructed by an SR network, since it was never learned in the low-resolution input.

Let me demonstrate this with some examples and code snippets that simulate the process. I am using PyTorch and its `torchvision` library here, as it is a commonly used framework for image manipulation.

**Example 1: Basic Bicubic Downsampling and Upsampling**

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image

# Load a single MNIST image for demonstration
mnist_dataset = MNIST(root='./data', download=True, train=False)
image, label = mnist_dataset[0]  # Take the first image

# Convert PIL image to tensor and normalize to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
image_tensor = transform(image).unsqueeze(0) # Add batch dimension

# Bicubic downsampling
downsample_transform = transforms.Resize(size=14, interpolation=transforms.InterpolationMode.BICUBIC)
downsampled_tensor = downsample_transform(image_tensor)

# Upsampling back to original size
upsample_transform = transforms.Resize(size=28, interpolation=transforms.InterpolationMode.BICUBIC)
upsampled_tensor = upsample_transform(downsampled_tensor)

# Convert tensors back to images (for visualization only)
def tensor_to_image(tensor):
    tensor = tensor.squeeze().cpu() # remove batch dim and move to CPU
    tensor = tensor * 0.5 + 0.5 # denormalize to [0,1] range
    image = transforms.ToPILImage()(tensor)
    return image

original_image = tensor_to_image(image_tensor)
downsampled_image = tensor_to_image(downsampled_tensor)
upsampled_image = tensor_to_image(upsampled_tensor)

original_image.save("original.png")
downsampled_image.save("downsampled.png")
upsampled_image.save("upsampled.png")
```

This code snippet first loads an MNIST image and converts it into a PyTorch tensor. It then simulates the downsampling process via a bicubic resize operation, halving the linear dimension of the image. Afterwards, it upsamples the image back to its original size, using bicubic interpolation once more. Running this code will output three images; the original, the downsampled, and the upsampled versions. The upsampled version will not be identical to the original; instead it will appear more blurred and less crisp. This basic interpolation demonstrates that just reverting the initial downscaling operation is not a super-resolution problem, but the upsampled image is still degraded and represents the failure mode of simple upsampling methods.

**Example 2: Simple SR Network**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import random
# Define a basic SR network
class SimpleSR(nn.Module):
    def __init__(self):
        super(SimpleSR, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

# Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = MNIST(root='./data', download=True, train=True, transform=transform)

#Create data loader for batching
batch_size=32
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)


# Downsampling Transform
downsample_transform = transforms.Resize(size=14, interpolation=transforms.InterpolationMode.BICUBIC)

#Initialize Model and Optimizer
model = SimpleSR()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs=5

for epoch in range(num_epochs):
  for batch_idx, (data, target) in enumerate(data_loader):
    optimizer.zero_grad()
    downsampled_data = downsample_transform(data)
    output = model(downsampled_data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    print(f'Epoch:{epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
```

This snippet defines a minimal convolutional neural network to act as a super-resolution model. The network takes low-resolution images and attempts to generate a high-resolution version using a few convolutional layers and ReLU activations. This is trained using the downsampled MNIST images and the original MNIST images as the targets. While it might reduce the loss value during training, it demonstrates that even with training the upsampled results are not satisfactory. The network learns to generate a smoothed approximation of each digit, not accurately restore the lost high frequency detail.

**Example 3: Visualizing Model Output**
```python
import matplotlib.pyplot as plt

# Load trained model and sample data
model.eval() #Set the model in evaluation mode
with torch.no_grad():
    test_image, _ = mnist_dataset[random.randint(0, len(mnist_dataset))]
    test_image = test_image.unsqueeze(0)
    downsampled_test_image = downsample_transform(test_image)
    predicted_image = model(downsampled_test_image)


original_image_plt = tensor_to_image(test_image)
downsampled_image_plt = tensor_to_image(downsampled_test_image)
predicted_image_plt = tensor_to_image(predicted_image)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(original_image_plt, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(downsampled_image_plt, cmap='gray')
plt.title('Downsampled')
plt.subplot(1,3,3)
plt.imshow(predicted_image_plt, cmap='gray')
plt.title('Predicted')
plt.show()

```

This code snippet loads a single sample image from the MNIST dataset and visualizes its original version, the downsampled version and the predicted, upscaled version after feeding it through the trained model from Example 2. The visualization will show a smoothed digit in place of the original high-resolution digit, showcasing the model's limitation. The output highlights how the predicted digit retains a low-resolution appearance and does not recover the lost information from the original.

To summarize, the fundamental reason for the poor performance of SR on MNIST lies in the conflict between the training methodology, designed for preserving details, and the low-frequency characteristics of MNIST. Standard downsampling techniques destroy the high-frequency information, and SR networks learn to approximate the overall shape of digits rather than recovering those missing details.

Recommendations for additional resources include textbooks covering topics on image processing and computer vision, such as those focusing on deep learning for image super-resolution techniques. Specific research papers that examine the limitations of classical SR techniques in the presence of significant information loss are also invaluable. Exploring alternative training data augmentation strategies is also a worthwhile avenue for investigation, though it should be noted that a key inherent limitation of MNIST still applies.
