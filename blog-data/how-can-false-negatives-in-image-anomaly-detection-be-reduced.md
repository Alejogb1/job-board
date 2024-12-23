---
title: "How can false negatives in image anomaly detection be reduced?"
date: "2024-12-23"
id: "how-can-false-negatives-in-image-anomaly-detection-be-reduced"
---

Alright, let’s tackle this one. False negatives in image anomaly detection – I've seen my fair share of those over the years, especially during that particularly challenging project involving automated quality control for semiconductor wafers. Dealing with the subtle imperfections that often passed unnoticed by initial models definitely forced me to delve deeper into the intricacies of this problem. It’s not just a theoretical concern; failing to detect a genuine anomaly can have significant downstream implications, ranging from production losses to potential safety hazards.

The core issue with false negatives, of course, is that they represent anomalies that the detection system incorrectly classifies as normal. To effectively reduce them, we need to address multiple contributing factors. Fundamentally, it often boils down to the sensitivity and representational power of the detection model, the quality and diversity of the training data, and how well the model's parameters are tuned.

Firstly, let’s consider the model itself. A common approach for anomaly detection is using autoencoders. An autoencoder learns a compressed representation of the normal input data and reconstructs the input from this compressed representation. Anomalies, ideally, would not be reconstructed as effectively, thus resulting in a higher reconstruction error. However, if the model isn't deep or complex enough, or if the latent space isn't sufficiently detailed, it may not capture the subtle nuances of 'normality' well enough and might actually learn to reconstruct anomalies to some degree. This blurs the distinction between normal and anomalous samples and hence, contributes to false negatives. One solution is to increase the network's complexity, perhaps by adding more layers or using a more intricate architecture such as a variational autoencoder (vae). However, blindly increasing complexity can lead to overfitting, so careful attention is needed.

Another important aspect is the training data. In my experience, the most frequent cause of false negatives was not the model itself, but rather training the model with inadequate examples of normal conditions. If the training dataset does not sufficiently cover the inherent variability of what we consider normal, the model might then wrongly classify anomalies within this unseen variability as normal. It’s crucial, therefore, to ensure that your training data is both comprehensive and representative. To this effect, techniques such as data augmentation can help, especially if real-world data is limited. Furthermore, considering more robust feature representations is key. Sometimes, relying solely on pixel data can be insufficient. Features such as edges, textures or more complex representations derived using techniques like convolutional layers, might provide a more discriminant descriptor of normal conditions.

Now, let’s delve into practical implementations. Let’s say we are using a standard autoencoder. Here’s a simple example using python and the pytorch library:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Sample data (replace with your actual data)
normal_data = np.random.rand(1000, 28 * 28).astype(np.float32) # 1000 images, 28x28 pixels
anomalous_data = np.random.rand(100, 28 * 28).astype(np.float32) * 2  # add some "anomaly"
all_data = np.concatenate((normal_data, anomalous_data), axis=0)
labels = np.concatenate((np.zeros(1000), np.ones(100)), axis=0)
all_data = torch.tensor(all_data)
labels = torch.tensor(labels).long()

dataset = TensorDataset(all_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
epochs = 10
for epoch in range(epochs):
    for images, _ in dataloader:  # We ignore the labels during training
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print(f'Epoch:{epoch+1}/{epochs}, loss: {loss.item():.4f}')

# Example evaluation (simplified)
model.eval()
with torch.no_grad():
    reconstructions = model(all_data)
    losses = torch.mean((reconstructions - all_data)**2, dim=1)
    threshold = torch.quantile(losses[labels==0], 0.95)  # 95th percentile of normal loss
    predictions = (losses > threshold).int()
    false_negatives = (predictions == 0) & (labels == 1)

print(f"number of false negatives: {torch.sum(false_negatives)}")
```

This is a very simplistic autoencoder example. Notice how we set the threshold on the reconstruction error based on normal samples’ losses. If we find that many anomalies are slipping by (high false negative), there's a few things we might tweak. One would be to tune the threshold value more precisely (experiment with different percentiles and validation data), perhaps using a more robust method such as receiver operating characteristic (roc) curves. If that does not work, moving onto a more complex model like a convolutional autoencoder might be necessary, like in the following example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assuming 28x28 grayscale images
normal_data = np.random.rand(1000, 1, 28, 28).astype(np.float32) # 1000 images, 1 channel
anomalous_data = np.random.rand(100, 1, 28, 28).astype(np.float32) * 2
all_data = np.concatenate((normal_data, anomalous_data), axis=0)
labels = np.concatenate((np.zeros(1000), np.ones(100)), axis=0)
all_data = torch.tensor(all_data)
labels = torch.tensor(labels).long()

dataset = TensorDataset(all_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(7*7*32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7*7*32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32,7,7)),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
  for images, _ in dataloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, images)
    loss.backward()
    optimizer.step()
  print(f'Epoch:{epoch+1}/{epochs}, loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
  reconstructions = model(all_data)
  losses = torch.mean((reconstructions - all_data)**2, dim=(1,2,3))
  threshold = torch.quantile(losses[labels==0], 0.95)
  predictions = (losses > threshold).int()
  false_negatives = (predictions == 0) & (labels == 1)


print(f"number of false negatives: {torch.sum(false_negatives)}")
```

This convolutional model will learn spatial hierarchies, which should allow for more precise reconstruction of normal data. If that still doesn't get the false negative rate low enough, using more robust techniques such as generative adversarial networks (gans) specifically trained for anomaly detection (e.g., using encoder-decoder networks in the discriminator), might prove more suitable.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assuming 28x28 grayscale images
normal_data = np.random.rand(1000, 1, 28, 28).astype(np.float32)
anomalous_data = np.random.rand(100, 1, 28, 28).astype(np.float32) * 2
all_data = np.concatenate((normal_data, anomalous_data), axis=0)
labels = np.concatenate((np.zeros(1000), np.ones(100)), axis=0)
all_data = torch.tensor(all_data)
labels = torch.tensor(labels).long()

dataset = TensorDataset(all_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Discriminator (simplified)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(7*7*64, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
      return self.model(img)


# Generator (simplified)
class Generator(nn.Module):
  def __init__(self, latent_dim=100):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(latent_dim, 7*7*128),
        nn.ReLU(),
        nn.Unflatten(dim=1, unflattened_size=(128,7,7)),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )

  def forward(self, z):
    return self.model(z)

latent_dim = 100
discriminator = Discriminator()
generator = Generator(latent_dim)

# loss functions
bce_loss = nn.BCELoss()

# Optimizers
optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizer_gen = optim.Adam(generator.parameters(), lr=0.0002)

epochs = 10
for epoch in range(epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.shape[0]

        ## Train Discriminator
        optimizer_disc.zero_grad()
        real_labels = torch.ones(batch_size,1)
        fake_labels = torch.zeros(batch_size,1)

        output_real = discriminator(real_images)
        loss_real = bce_loss(output_real, real_labels)

        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        output_fake = discriminator(fake_images)
        loss_fake = bce_loss(output_fake, fake_labels)

        loss_disc = loss_real + loss_fake
        loss_disc.backward()
        optimizer_disc.step()


        ## Train Generator
        optimizer_gen.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        output_fake = discriminator(fake_images)
        loss_gen = bce_loss(output_fake, real_labels)
        loss_gen.backward()
        optimizer_gen.step()

    print(f'Epoch: {epoch+1}/{epochs}, Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}')


# Anomaly detection (simplified)
generator.eval()
discriminator.eval()
with torch.no_grad():
    real_preds = discriminator(all_data)
    z_latent = torch.randn(all_data.shape[0], latent_dim)
    reconstructions = generator(z_latent)

    losses = torch.mean((reconstructions - all_data)**2, dim=(1,2,3))
    threshold = torch.quantile(losses[labels==0], 0.95)
    predictions = (losses > threshold).int()
    false_negatives = (predictions == 0) & (labels == 1)


print(f"number of false negatives: {torch.sum(false_negatives)}")

```
This is again simplified for demonstration. For a full implementation, you would need more sophisticated approaches, including optimizing the training procedure for stable gan training. A critical aspect of using GANs for anomaly detection is selecting a suitable anomaly score, which in our example is the reconstruction loss. More complex metrics, such as feature matching within the discriminator, may improve results.

Finally, the performance evaluation is as important as the model itself. I recommend looking into statistical performance evaluation metrics. Instead of focusing solely on the overall accuracy, pay close attention to precision, recall, and f1-scores, because these metrics explicitly differentiate the performance on each class. Additionally, use roc curves to better select the threshold value.

For a deeper understanding of autoencoders, variational autoencoders, and gan, I highly recommend “deep learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. For more on statistical methods, the book "elements of statistical learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman will be quite useful. The paper "a review of deep learning techniques for image anomaly detection" by Pang, et al., gives a great overview of recent advancements in the field.

Reducing false negatives is a multi-faceted endeavor, not a silver bullet solution. It often requires a blend of architectural improvements, carefully constructed datasets and a deep understanding of the underlying mechanisms. Experience has taught me that a data-centric approach, combined with rigorous evaluation, forms the backbone of an effective solution.
