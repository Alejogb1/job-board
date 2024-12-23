---
title: "How can GANs be used for anomaly detection?"
date: "2024-12-23"
id: "how-can-gans-be-used-for-anomaly-detection"
---

Let's dive into the fascinating, and sometimes frustrating, realm of using generative adversarial networks (gans) for anomaly detection. It’s a topic I’ve spent a considerable amount of time on, especially during my tenure at a previous firm where we were tasked with identifying potential security breaches in network traffic—an area, as you can imagine, where even the smallest abnormality can signal significant trouble.

The core principle, and the beauty of it, lies in the gan’s ability to learn the underlying distribution of the "normal" data. Think about it: instead of trying to explicitly define what constitutes an anomaly, which is a near impossible task in most real-world scenarios, we train the gan on normal examples. The generator then attempts to create data that follows the same patterns, and the discriminator tries to differentiate between real and generated data. Once the gan achieves reasonable convergence, the key insight is that it becomes really good at reproducing normal data, and really bad at generating anomalies.

I recall the first time we implemented this; we were using a relatively straightforward convolutional gan (cgan) on network packet captures. The idea was to feed the gan normal traffic patterns. The expectation was that when anomalous traffic, never seen during training, was fed into the generator, the generator wouldn't be able to reproduce it accurately. This discrepancy, as you might anticipate, was what we’d leverage to flag an anomaly.

The process generally goes like this: during the training phase, we provide the gan with only 'normal' instances, essentially training it on what 'normal' looks like. After training, when a new data point is presented, we assess how well the generator can reproduce it. The discriminator is sometimes used as an additional check by assessing the plausibility of the reproduced output. Anomalies will typically result in high reconstruction error or high discriminator uncertainty (depending on how you formulate the anomaly score).

Let's look at a simplified example using PyTorch. This illustration uses a basic autoencoder architecture to emulate the core mechanics of a gan for demonstration purposes. In a true gan setup, you'd have a separate generator and discriminator with adversarial training, but the essence of reconstruction error-based anomaly detection remains similar.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

# Generate some dummy data for demonstration
def generate_data(num_samples, input_size, anomaly_fraction=0.1):
    normal_data = torch.randn(int(num_samples * (1 - anomaly_fraction)), input_size)
    anomaly_data = torch.randn(int(num_samples * anomaly_fraction), input_size) * 5 # Anomaly values are more pronounced
    data = torch.cat((normal_data, anomaly_data), dim=0)
    labels = torch.cat((torch.zeros(len(normal_data)), torch.ones(len(anomaly_data))), dim=0)
    return data, labels

# Setup parameters
input_size = 10
hidden_size = 5
num_samples = 1000
learning_rate = 0.001
epochs = 100

# Create the autoencoder and optimizer
autoencoder = Autoencoder(input_size, hidden_size)
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Prepare data
data, labels = generate_data(num_samples, input_size)
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch_data, _ in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(batch_data)
        loss = criterion(outputs, batch_data)
        loss.backward()
        optimizer.step()

# Anomaly detection (using reconstruction error)
reconstructions = autoencoder(data).detach()
reconstruction_errors = torch.mean((data - reconstructions)**2, dim=1)
threshold = torch.quantile(reconstruction_errors, 0.9)  # Using 90th percentile for demo

# Classify based on threshold
predicted_anomalies = (reconstruction_errors > threshold).int()

# Evaluate
correctly_detected_anomalies = (predicted_anomalies == labels).sum()
accuracy = correctly_detected_anomalies / len(labels)
print(f"Accuracy: {accuracy.item():.4f}")
```

This code trains a simplified autoencoder and uses its reconstruction error as an anomaly score. Notice how the 'anomaly' data is created to be more pronounced, leading to higher reconstruction errors.

Now, let's consider a variation that uses the gan structure more faithfully. This snippet assumes that we have a trained generator. We then calculate reconstruction errors using the generator.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a trained generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 128)
        self.linear2 = nn.Linear(128, output_size)

    def forward(self, z):
        h = torch.relu(self.linear1(z))
        return self.linear2(h)

# Parameters
latent_dim = 10
input_size = 10
num_samples = 100

# Create generator and dummy data
generator = Generator(latent_dim, input_size)
# We would ideally load a pre-trained generator model
# For demonstration, we make a dummy forward pass
def generate_latent_space(num_samples, latent_dim):
  return torch.randn(num_samples, latent_dim)

def generate_data(num_samples, input_size, anomaly_fraction=0.1):
    normal_data = torch.randn(int(num_samples * (1 - anomaly_fraction)), input_size)
    anomaly_data = torch.randn(int(num_samples * anomaly_fraction), input_size) * 5 # Anomaly values are more pronounced
    data = torch.cat((normal_data, anomaly_data), dim=0)
    labels = torch.cat((torch.zeros(len(normal_data)), torch.ones(len(anomaly_data))), dim=0)
    return data, labels

data, labels = generate_data(num_samples, input_size)
latent_vector = generate_latent_space(num_samples, latent_dim)

# Anomaly detection using generator
generated_data = generator(latent_vector).detach()
reconstruction_errors = torch.mean((data - generated_data)**2, dim=1)
threshold = torch.quantile(reconstruction_errors, 0.9)  # Using 90th percentile for demo

# Classify based on threshold
predicted_anomalies = (reconstruction_errors > threshold).int()

# Evaluate
correctly_detected_anomalies = (predicted_anomalies == labels).sum()
accuracy = correctly_detected_anomalies / len(labels)
print(f"Accuracy: {accuracy.item():.4f}")
```

In this example, the reconstruction is not done by the autoencoder itself, but rather by using a generator mapping from a latent space. This more closely resembles how GANs work.

Finally, here is an approach that emphasizes the discriminator as an anomaly detector based on its classification confidence:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a trained discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        return self.sigmoid(self.linear2(h))

# Parameters
input_size = 10
num_samples = 100

# Create discriminator and dummy data
discriminator = Discriminator(input_size)
# We would ideally load a pre-trained discriminator model

def generate_data(num_samples, input_size, anomaly_fraction=0.1):
    normal_data = torch.randn(int(num_samples * (1 - anomaly_fraction)), input_size)
    anomaly_data = torch.randn(int(num_samples * anomaly_fraction), input_size) * 5 # Anomaly values are more pronounced
    data = torch.cat((normal_data, anomaly_data), dim=0)
    labels = torch.cat((torch.zeros(len(normal_data)), torch.ones(len(anomaly_data))), dim=0)
    return data, labels

data, labels = generate_data(num_samples, input_size)

# Anomaly detection using discriminator output
discriminator_output = discriminator(data).detach()
anomaly_scores = 1 - discriminator_output  # Low probability implies higher anomaly
threshold = torch.quantile(anomaly_scores, 0.9) # Using 90th percentile for demo

# Classify based on threshold
predicted_anomalies = (anomaly_scores > threshold).int()

# Evaluate
correctly_detected_anomalies = (predicted_anomalies == labels).sum()
accuracy = correctly_detected_anomalies / len(labels)
print(f"Accuracy: {accuracy.item():.4f}")
```

Here, the discriminator’s confidence that a data point is real is used as a measure of ‘normality’. Anomalies are expected to receive a lower confidence score from the discriminator.

These examples are simplified for illustration, of course. In practical applications, you’ll encounter various challenges including the need for more sophisticated architectures such as improved conditional gans (cgan), robust training techniques to avoid mode collapse, and careful hyperparameter tuning. The "An Introduction to Statistical Learning" by Gareth James, et al, serves as a great foundation for understanding these statistical learning principles. For a more focused view on deep learning techniques, "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville is an essential read. As for specific research papers, look into the original gan paper by Goodfellow et al. and those investigating anomaly detection with gans, such as 'Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Placement'.

Based on my experience, carefully evaluating both the reconstruction error from the generator, and the discriminator's output confidence provides a robust anomaly detection system using gans. The particular approach used depends on the nature of your data and application. The power of gans lies in their ability to implicitly learn the intricacies of normal data, allowing them to effectively discern what is truly anomalous.
