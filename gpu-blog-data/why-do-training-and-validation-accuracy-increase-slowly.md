---
title: "Why do training and validation accuracy increase slowly and oscillate?"
date: "2025-01-30"
id: "why-do-training-and-validation-accuracy-increase-slowly"
---
Training a neural network is rarely a smooth, monotonically improving process. I've spent considerable time debugging sluggish and oscillating accuracy curves, and while the specific causes can be multifaceted, a few key areas consistently contribute to these frustrating patterns. Understanding these underlying dynamics is crucial for optimizing training and achieving desired performance.

Firstly, the learning process in a neural network is essentially a search through a high-dimensional, non-convex loss landscape. This landscape is rarely smooth; it's peppered with local minima, saddle points, and flat regions. Initial conditions, such as the random initialization of network weights, can place the optimization process in a part of this landscape that isn't conducive to rapid improvement. If the initial weights are poor, they might place the network in a steep gradient area that oscillates rapidly but doesn't lead to significant progress towards a global minimum. Moreover, gradients can become vanishingly small in regions with flat loss, resulting in stagnation.

Secondly, a critical factor affecting training and validation accuracy is the optimization algorithm itself and its hyperparameter settings. Gradient descent-based optimizers, such as Stochastic Gradient Descent (SGD), Adam, or RMSprop, rely on an estimated gradient calculated from a mini-batch of training samples. If the batch size is too small, the gradient estimate is noisy and can lead to oscillations during training. This noisy gradient causes the optimization process to move erratically, increasing loss in one epoch but decreasing it in the next. If learning rates are too large, we might also observe similar erratic behavior because the optimizer may jump past optimal points, causing the loss to spike and then decrease again. Conversely, a learning rate that is too small leads to excessively slow learning, which can look like a long period of minimal increase in accuracy. Further complexities arise with momentum, which helps the optimizer accelerate through flat or shallow gradient regions, or from adaptive learning rate methods which change the learning rate according to the update history. If these methods are not well-tuned for a specific problem, they can contribute to oscillations.

Thirdly, dataset characteristics also impact these trends. Insufficient data, especially relative to the model's complexity, frequently leads to overfitting. Overfitting manifests when the network begins to memorize the training data rather than generalizing well to unseen data. Training accuracy will increase, but the validation accuracy will saturate or start to decrease. The model becomes attuned to the specifics of the training data, including its noise, which explains why it often cannot be generalized to new samples. Similarly, highly imbalanced datasets can pose a significant challenge for the optimizer. The model might learn to prioritize predicting the majority class, exhibiting high accuracy on the majority class but low accuracy on the minority class. This bias results in slow progress and oscillations on the overall validation dataset.

Finally, consider the influence of architecture. Deep networks, especially with many layers, are prone to the vanishing or exploding gradient problem. As gradients are backpropagated through multiple layers, they can either become exceedingly small or excessively large, which hinders learning. Specialized architectures, like those with residual connections, try to mitigate this problem, but they are not a universal cure. If chosen inappropriately for the task, or poorly parameterized, they can also contribute to slower training and oscillation of accuracy metrics.

Let’s examine this using a concrete example. Suppose I’ve trained a convolutional neural network to classify images. I’ve observed the following trends: the training accuracy increases slowly and displays a notable oscillation pattern across several epochs.

Here's a snippet of how we might define the training loop in Python with PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Example: Creating Dummy Data
X_train = torch.randn(1000, 3, 32, 32)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 3, 32, 32)
y_val = torch.randint(0,10,(200,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Example: Creating a Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    return train_loss / total_samples, correct_predictions / total_samples

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    return val_loss / total_samples, correct_predictions / total_samples

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

epochs = 20

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

In this basic implementation, I've generated dummy data, instantiated a basic CNN, and set up the training loop. Running this may reveal fluctuations in training and validation accuracy. The oscillations in this simplified example stem from a combination of reasons such as the random initialization of weights and small data sample size.

To demonstrate the effect of large learning rates causing oscillations, the above code can be modified as follows. Here I increased the learning rate from 0.001 to 0.1.

```python
optimizer = optim.Adam(model.parameters(), lr=0.1)
```

This will cause a much more oscillatory pattern in the graphs of train and validation loss. The validation loss may be more volatile, spiking high and then dropping back, or even worse, increasing constantly after an initial dip.

Let's add a modification to demonstrate the effect of a small batch size. Again, modifying the initial code, I can change the DataLoader instantiation to use a smaller batch size like 16:

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
```

A smaller batch size tends to amplify the noise in the gradient estimate, resulting in more oscillations in both the training and validation metrics, compared to the initial example which had a batch size of 64.

To summarize, these effects of slow improvement and oscillations of accuracy during training are caused by a combination of aspects of the model, the training procedure, and the data. To address these, one must methodically adjust the hyperparameters of the optimization process including the learning rate, the batch size, the choice of optimizer, and other data preprocessing or regularization techniques that can help reduce overfitting.

For further learning, the reader should consult books covering deep learning theory and practice. Specifically, books that cover model optimization, regularization, and data augmentation will be particularly useful. Reviewing literature on specific optimizers, such as Adam, or methods for addressing imbalanced datasets will also offer useful insight.
