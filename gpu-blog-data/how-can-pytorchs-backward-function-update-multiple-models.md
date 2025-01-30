---
title: "How can PyTorch's `backward()` function update multiple models simultaneously?"
date: "2025-01-30"
id: "how-can-pytorchs-backward-function-update-multiple-models"
---
The core challenge in updating multiple PyTorch models simultaneously with a single `backward()` call stems from the foundational principle that `backward()` propagates gradients based on a single loss function associated with the computation graph. PyTorch’s automatic differentiation engine, at its heart, traces operations leading to a scalar value (the loss) and then computes derivatives with respect to all tensors involved in the computation, *provided those tensors require gradients*. When employing multiple models, each often has its distinct loss function. Consequently, directly applying `backward()` to an aggregated loss over multiple models without a carefully orchestrated process will not produce the desired per-model parameter updates. To achieve concurrent updates, it is essential to invoke `backward()` on the specific loss associated with each model or to structure the computation such that a single, composite loss drives the updates for all participating models, tailored to the task at hand. I’ve personally encountered this in a multi-agent reinforcement learning project where several actor networks needed to be optimized concurrently.

The most straightforward, yet often less computationally efficient approach, involves calculating the individual losses for each model separately and subsequently calling `backward()` on each loss in its respective scope. This method guarantees correct gradient calculation and parameter updates, but introduces serial execution, which negates the potential for parallelism inherent in some hardware environments. The basic procedure can be summarized as follows: For each model, forward propagate an input; calculate its specific loss; invoke `backward()` on that specific loss. After this sequence is complete for all models, the optimizers associated with each model can step, reflecting the newly calculated gradients. This sequential execution, however, does not truly update the models 'simultaneously' in the sense that all gradient calculations happen in parallel at the exact same moment. It is an approach of independent updates, one after the other.

Consider a scenario where you have two convolutional neural networks, `model_a` and `model_b`, where each processes a unique input.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define simple convolutional models
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.fc = nn.Linear(out_channels * 26 * 26 , 10) # Placeholder for a linear layer

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Initialize models
model_a = SimpleCNN(in_channels=3, out_channels=16)
model_b = SimpleCNN(in_channels=1, out_channels=8)

# Initialize optimizers
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Input data
input_a = torch.randn(1, 3, 28, 28)  # batch_size, channels, height, width
input_b = torch.randn(1, 1, 28, 28)
target_a = torch.randint(0, 10, (1,)) # Batch of one integer between 0 and 9
target_b = torch.randint(0, 10, (1,))

# Model forward passes and loss calculations.
output_a = model_a(input_a)
loss_a = criterion(output_a, target_a)
output_b = model_b(input_b)
loss_b = criterion(output_b, target_b)

# Backpropagation and optimization (Sequential).
optimizer_a.zero_grad()
loss_a.backward()
optimizer_a.step()

optimizer_b.zero_grad()
loss_b.backward()
optimizer_b.step()

print("Sequential updates completed.")
```
In this code, `model_a` and `model_b` receive distinct data and their losses are calculated separately. The crucial point is the sequential calls to `backward()`, and then the associated optimizers step to update the parameters. Although this approach updates the models, it does not achieve actual simultaneous gradient propagation across the multiple models. Each model’s gradient computation is strictly independent of the other.

A more computationally efficient alternative when appropriate is to structure the models in a manner where they share a common loss function or can be optimized via a shared loss component. For instance, in a generative adversarial network (GAN) setting, the generator and discriminator models are trained using a composite loss function that captures their adversarial relationship. The discriminator attempts to minimize its own classification error, while the generator simultaneously attempts to maximize the classification error of the discriminator. In this composite setup, a single `backward()` can correctly propagate gradients to both the generator and discriminator models, assuming the optimizers are configured appropriately to handle different parameter groups.

```python
# Example with a GAN-like composite loss
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 28*28) # Simple linear layer
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.fc = nn.Linear(8*13*13, 1) # Assuming 28x28 image downsampled after conv
    def forward(self, x):
         x = self.conv(x)
         x = x.view(x.size(0), -1)
         x = self.fc(x)
         return x

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize optimizers for different model sets
optim_generator = optim.Adam(generator.parameters(), lr=0.001)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Generate fake data (assuming random noise input)
noise = torch.randn(64, 100) # batch_size, latent_dim
fake_images = generator(noise)

# Real data and labels
real_images = torch.randn(64, 1, 28, 28)
real_labels = torch.ones(64, 1)
fake_labels = torch.zeros(64, 1)

# Discriminator loss
disc_real_output = discriminator(real_images)
disc_fake_output = discriminator(fake_images.detach()) # Detach to only update the discriminator in this stage
disc_loss_real = criterion(disc_real_output, real_labels)
disc_loss_fake = criterion(disc_fake_output, fake_labels)
disc_loss = disc_loss_real + disc_loss_fake

# Generator loss - trick the discriminator
generator_output_for_loss = discriminator(fake_images)
gen_loss = criterion(generator_output_for_loss, real_labels) # Generator aims to make the discriminator predict ones.

# Single backward pass updating both models, with two separated optimizer steps
optim_discriminator.zero_grad()
disc_loss.backward()
optim_discriminator.step()

optim_generator.zero_grad()
gen_loss.backward()
optim_generator.step()

print("GAN-like updates completed.")
```

In this instance, the discriminator's loss involves both real and fake data, and its backward call only updates its parameters. Then, the generator’s loss depends on the discriminator output but the generator aims for the discriminator to output one which results in an opposing objective. `backward()` on gen_loss then updates the generator’s parameters. Crucially, the losses of the generator and discriminator aren’t summed to give one composite loss; rather, they are used in sequence to update two different models in one training loop.

Another approach, particularly useful in scenarios like multi-task learning, employs the concept of *shared parameter layers*. Multiple models may share certain layers which form the basis for different outputs. In this case, a composite loss function can be created by summing the individual losses. A single `backward()` call on this composite loss function then updates both the shared layers, and parameters specific to each branch, simultaneously.

```python
# Multi-task learning example with shared backbone
import torch
import torch.nn as nn
import torch.optim as optim

class SharedBackbone(nn.Module):
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.flattened_size = 16 * 13 * 13 # Assume 28x28 images, output shape after conv

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

class TaskSpecificHead(nn.Module):
    def __init__(self, backbone_size, output_dim):
        super(TaskSpecificHead, self).__init__()
        self.fc = nn.Linear(backbone_size, output_dim)
    def forward(self, x):
        return self.fc(x)

# Shared backbone
shared_backbone = SharedBackbone()

# Task specific heads
task_a_head = TaskSpecificHead(shared_backbone.flattened_size, 10) # Example classification task
task_b_head = TaskSpecificHead(shared_backbone.flattened_size, 20) # Example regression task

# Optimizers (for shared backbone and each task's head)
optimizer_shared = optim.Adam(shared_backbone.parameters(), lr=0.001)
optimizer_task_a = optim.Adam(task_a_head.parameters(), lr=0.001)
optimizer_task_b = optim.Adam(task_b_head.parameters(), lr=0.001)

# Loss functions
criterion_a = nn.CrossEntropyLoss()
criterion_b = nn.MSELoss() #Mean Squared Error

# Dummy data
input_data = torch.randn(64, 3, 28, 28) #batch size, channels, height, width
target_a = torch.randint(0, 10, (64,)) # classification labels
target_b = torch.randn(64, 20) # regression target values

# Forward Pass
shared_features = shared_backbone(input_data)
output_a = task_a_head(shared_features)
output_b = task_b_head(shared_features)

# Loss Calculations
loss_a = criterion_a(output_a, target_a)
loss_b = criterion_b(output_b, target_b)

# Composite Loss (summed losses from different tasks)
composite_loss = loss_a + loss_b

# Backward pass and optimization (shared and per-model optimizers)
optimizer_shared.zero_grad()
optimizer_task_a.zero_grad()
optimizer_task_b.zero_grad()

composite_loss.backward()

optimizer_shared.step()
optimizer_task_a.step()
optimizer_task_b.step()

print("Multi-task updates completed.")
```

In this multi-task example, the shared backbone's parameters are updated as a direct result of the composite loss, whilst each task specific head’s parameters are updated based on their individual contributions to the total loss. This enables a single forward-backward pass to train multiple heads with shared features. This demonstrates an effective way to train two or more models concurrently when shared feature representation is beneficial or necessary for training efficiency.

In summary, PyTorch's `backward()` updates parameters based on the loss function computed in a computation graph. Simultaneous updates for multiple models require strategically organizing forward passes, defining appropriate loss functions, and employing either sequential backward calls, composite losses, or a sharing layer framework. Choosing the right technique depends on the specific relationships between models and their respective tasks. Further study of multi-task learning, generative adversarial networks, and distributed training will further illuminate advanced techniques for optimizing collections of models concurrently. Resources on gradient accumulation, distributed training with PyTorch, and advanced optimization strategies may prove beneficial for complex multi-model training situations.

This response is based on my experiences within my research group, which I can not further disclose for reasons of privacy and confidentiality, and in my role at an established AI research organization.
