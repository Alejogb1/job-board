---
title: "Why does PyTorch Lightning's complex-valued CNN produce NaN outputs after one training batch?"
date: "2025-01-30"
id: "why-does-pytorch-lightnings-complex-valued-cnn-produce-nan"
---
Complex-valued convolutional neural networks, particularly those implemented within PyTorch Lightning, can exhibit NaN (Not a Number) outputs after only a single training batch, a phenomenon I've encountered and debugged extensively. This early divergence, often surprising given the relatively small computation involved, typically points to numerical instability during the forward and, more critically, the backward propagation stages. The root cause almost always lies in the interaction between the complex-number arithmetic and specific activation functions or gradient calculations, rather than an inherent flaw within PyTorch Lightning itself.

The fundamental issue stems from how gradient updates are computed for complex-valued weights. In a standard real-valued neural network, gradients represent the direction of steepest ascent or descent of the loss function with respect to the weights. Complex numbers, however, introduce a second dimension: the imaginary component. The gradients for complex-valued weights must, therefore, reflect how the loss changes with respect to both the real and imaginary parts of the weights. This dual-dimensional sensitivity can lead to substantial magnitude increases when improperly handled.

The combination of complex activation functions and their corresponding derivative calculations, compounded by common loss functions, are often the catalyst for these numerical explosions.  Standard activation functions like ReLU, designed for real-valued data, do not behave well with complex inputs. When a complex number enters ReLU, only the real part is considered and if it is negative, it is set to zero. The imaginary part is then effectively lost and becomes problematic during backpropagation as the gradient is zero for the negative regions with respect to the input. Furthermore, common loss functions like mean squared error (MSE) are designed for real outputs. Applying them to complex numbers requires either breaking down complex numbers into real and imaginary parts or using custom loss functions tailored for complex outputs. Improperly applying real-valued loss functions may lead to unintended gradient behaviors.

Additionally, if the initial weights or the input data contain very large magnitudes, especially along the imaginary dimension, the gradient may increase exponentially leading to overflow during backpropagation, eventually resulting in NaN. A small number of complex multiplications with large magnitudes could trigger this problem. These issues are not unique to PyTorch Lightning, but are exacerbated because of the framework's automatic handling of gradient backpropagation. If the user doesn’t anticipate the numerical issues inherent in working with complex numbers, using automatic differentiation may not be straightforward.

Let’s examine specific examples with corresponding code to illustrate the common causes.

**Example 1: The ReLU Trap**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class ComplexCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, dtype=torch.complex128)
        self.relu = nn.ReLU() # ReLU is problematic
        self.fc = nn.Linear(16*32*32, 10, dtype=torch.complex128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.randn(64, 3, 32, 32, dtype=torch.complex128), batch_size=16)

class ComplexClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ComplexCNN()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)
        loss = torch.mean(torch.square(y_hat.real) + torch.square(y_hat.imag)) # Inappropriate real valued loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    dm = SimpleDataModule()
    model = ComplexClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, dm)
```

This example creates a simple complex-valued CNN with a ReLU activation after the convolution. The training data is generated from random complex numbers.  The loss function is simply the sum of squares of the real and imaginary part, and it is not appropriate for the complex numbers here.  As a result, after running this, one will observe the training loss quickly turning to `NaN`. The problem stems from ReLU's disregard for the imaginary parts, coupled with a simple mean square loss that does not account for complex space. Backpropagating through this creates highly unstable gradients.

**Example 2: Unbounded Weights & Data:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, dtype=torch.complex128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, dtype=torch.complex128)
        self.fc = nn.Linear(32 * 32* 32, 10, dtype=torch.complex128)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1) # Improved activation function but still not optimal
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.randn(64, 3, 32, 32, dtype=torch.complex128) * 10000, batch_size=16) # large numbers

class ComplexClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ComplexCNN()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)
        loss = torch.mean(torch.square(y_hat.real) + torch.square(y_hat.imag))  # Still an issue
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    dm = SimpleDataModule()
    model = ComplexClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, dm)

```
In this version, the ReLU activation is replaced with Leaky ReLU (which allows gradients to flow more freely), but this does not fix the NaN problem, as it is just a partial improvement. The data is also created with a large magnitude. The combination of poorly conditioned initial weights along with large input magnitudes can cause an explosion in the gradient magnitudes. This will also result in NaN appearing in training loss after one training step. 

**Example 3: Using a Proper Complex Loss Function and Activation**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, dtype=torch.complex128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, dtype=torch.complex128)
        self.fc = nn.Linear(32 * 32* 32, 10, dtype=torch.complex128)

    def forward(self, x):
        x = F.complex_relu(self.conv1(x)) # Proper complex activation function
        x = F.complex_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.randn(64, 3, 32, 32, dtype=torch.complex128) , batch_size=16) # Use default data with small magnitude.

class ComplexClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ComplexCNN()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)

        target = torch.randn_like(y_hat) # dummy target
        loss = F.mse_loss(y_hat, target) # Use torch's default mse
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

if __name__ == '__main__':
    dm = SimpleDataModule()
    model = ComplexClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, dm)
```

Here, the `F.complex_relu` (which must be defined) is used to handle complex numbers properly, while the data magnitude is kept small, and the default mean squared error is used for the loss function.  This is the most important fix, as activation functions designed specifically for complex numbers do a much better job in handling the gradient update. However, even with these changes, it is not a complete solution and further investigation into complex number operations, data scaling, normalization and a good choice of complex-valued loss function is necessary for a robust solution. Note that `F.complex_relu` does not exist in Pytorch directly. It must be defined using complex number operations.

In conclusion, encountering NaN outputs in complex-valued CNNs within PyTorch Lightning after a single training batch is usually indicative of numerical instability due to inappropriate activation functions, insufficient complex number handling during gradient updates, or large magnitudes in either weights or data. While PyTorch Lightning provides convenience in terms of structure and automatic differentiation, it is crucial to understand these underlying issues to ensure proper training when dealing with complex numbers. For further study, one should investigate complex valued activation functions such as CReLU, and consider literature on complex-valued neural network design and stable training methods. A thorough study of complex-valued loss functions, weight initializers for complex networks, and numerical stability in backpropagation will further improve understanding.
