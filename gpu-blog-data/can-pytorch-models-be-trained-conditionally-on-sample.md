---
title: "Can PyTorch models be trained conditionally on sample labels?"
date: "2025-01-30"
id: "can-pytorch-models-be-trained-conditionally-on-sample"
---
Training PyTorch models conditionally on sample labels is a core capability, enabling the construction of discriminative and generative models where output is influenced by input class information. I've utilized this extensively across varied projects, from multi-modal image synthesis to structured data classification, and in my experience, properly handling conditional training is crucial for effective model learning. The primary mechanisms for this are leveraging label information during forward passes and modifying loss functions to incorporate these labels.

The foundation of conditional training in PyTorch lies in using labels as additional inputs, alongside the typical feature inputs. This is achieved by augmenting the model's architecture to process labels and then use these processed label representations to influence the intermediate feature maps. One can achieve this through different techniques, but fundamentally, the model needs to learn a function that maps feature and label representations to the desired output space. For classification problems, this might manifest as a modulation of feature maps prior to the final classification layer, while for generative models, labels often condition the noise or intermediate latent variables to control the generation process.

The key concept here is that the model's decision-making is no longer solely reliant on the input data alone. Instead, the model learns a mapping that is a function of both the input data **and** the associated label. This allows a trained model to learn distinct decision boundaries for each class, even if there is overlap in the feature space. In generative scenarios, the same mechanism enables the model to generate different types of outputs depending on the supplied label, thereby facilitating more controllable generation processes.

The most common practical method for integrating labels into training involves encoding the labels, usually as one-hot vectors or learned embeddings, and then concatenating them with feature maps, injecting them as inputs into specific model layers, or using them in specialized conditioning layers. This approach can be summarized as follows: (1) Encode labels to a numerical representation, (2) Incorporate the encoded representation into the model’s forward pass using concatenation, element-wise additions, multiplication or learnable affine transforms, (3) compute the model’s output based on both data and label information and (4) Calculate the loss, comparing model outputs to the target data, potentially modifying the loss function to incorporate class weights.

Here are three practical code examples demonstrating these concepts, along with detailed commentary:

**Example 1: Conditional Classification using Embedding and Concatenation**

```python
import torch
import torch.nn as nn

class ConditionalClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, embedding_dim, hidden_dim):
        super(ConditionalClassifier, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc1 = nn.Linear(input_dim + embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, labels):
        # 1. Embed the labels
        label_embed = self.embedding(labels)

        # 2. Concatenate embeddings with input data
        x = torch.cat((x, label_embed), dim=1)

        # 3. Process the combined input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example Usage
num_classes = 10
input_dim = 20
embedding_dim = 10
hidden_dim = 50
batch_size = 32

model = ConditionalClassifier(num_classes, input_dim, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Sample training data
x = torch.randn(batch_size, input_dim)
labels = torch.randint(0, num_classes, (batch_size,))

# Forward pass, loss calculation, and backpropagation
outputs = model(x, labels)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```
This first example illustrates the simplest implementation. We use an `nn.Embedding` layer to learn a vector representation for each class label.  These embeddings are then concatenated with the input features. The combined input is then fed through a fully connected network to produce class predictions. Critically, the labels are not used only in the loss but are incorporated directly in the forward pass of the model, allowing the model to learn a function conditioned on the class information. This particular example is suitable for smaller datasets where the number of classes is limited, but the concept is directly applicable to complex models when embedding sizes are tuned.

**Example 2: Conditional Generative Adversarial Network (cGAN)**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, embedding_dim, output_dim):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc1 = nn.Linear(latent_dim + embedding_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z, labels):
        label_embed = self.embedding(labels)
        z = torch.cat((z, label_embed), dim=1)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes, embedding_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc1 = nn.Linear(input_dim + embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, labels):
        label_embed = self.embedding(labels)
        x = torch.cat((x, label_embed), dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.sigmoid(self.fc3(x))
        return x

# Example Usage
latent_dim = 100
num_classes = 10
embedding_dim = 20
output_dim = 784
batch_size = 64

generator = Generator(latent_dim, num_classes, embedding_dim, output_dim)
discriminator = Discriminator(output_dim, num_classes, embedding_dim)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()


# Dummy training data
z = torch.randn(batch_size, latent_dim)
labels = torch.randint(0, num_classes, (batch_size,))
real_images = torch.randn(batch_size, output_dim)

# Discriminator training
optimizer_d.zero_grad()
real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)

# Real Image Discriminator Loss
d_real_outputs = discriminator(real_images, labels)
d_real_loss = criterion(d_real_outputs, real_labels)

# Fake Image Discriminator Loss
fake_images = generator(z, labels)
d_fake_outputs = discriminator(fake_images.detach(), labels)
d_fake_loss = criterion(d_fake_outputs, fake_labels)

d_loss = d_real_loss + d_fake_loss
d_loss.backward()
optimizer_d.step()

# Generator Training
optimizer_g.zero_grad()
g_outputs = discriminator(fake_images, labels)
g_loss = criterion(g_outputs, real_labels)
g_loss.backward()
optimizer_g.step()

print(f"Discriminator loss: {d_loss.item():.4f}, Generator loss: {g_loss.item():.4f}")

```
This second example demonstrates a conditional Generative Adversarial Network (cGAN), where both the generator and discriminator are conditioned on class labels.  Here the label information is used in a slightly more involved fashion. Specifically, the same embedding mechanism from example 1 is used, with these embeddings being concatenated with latent input for the generator, and image input for the discriminator.  This allows the generator to generate different outputs according to the provided label, and forces the discriminator to learn to discriminate between real and fake images based on the input label. Such a construction is used extensively in various tasks including image and text generation tasks.

**Example 3: Conditional Normalization Layer**
```python
import torch
import torch.nn as nn

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim):
        super(ConditionalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gamma_fc = nn.Linear(embedding_dim, num_features)
        self.beta_fc = nn.Linear(embedding_dim, num_features)

        # Initialize weights to 1 and biases to 0 for gamma/beta
        self.gamma_fc.weight.data.fill_(0)
        self.beta_fc.weight.data.fill_(0)

    def forward(self, x, labels):
        x = self.bn(x)
        label_embed = self.embedding(labels)
        gamma = self.gamma_fc(label_embed)
        beta = self.beta_fc(label_embed)

        # Reshape for channel-wise scaling and shifting
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        return gamma * x + beta

class ConditionalModel(nn.Module):
    def __init__(self, input_dim, num_features, num_classes, embedding_dim):
        super(ConditionalModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_features)
        self.cbn = ConditionalBatchNorm(num_features, num_classes, embedding_dim)

    def forward(self, x, labels):
        x = self.fc(x)
        x = self.cbn(x, labels)
        return x

# Example Usage
num_features = 128
num_classes = 10
embedding_dim = 32
input_dim = 64
batch_size = 32

model = ConditionalModel(input_dim, num_features, num_classes, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() # Example loss function, not necessarily tied to labels directly

# Sample training data
x = torch.randn(batch_size, input_dim)
labels = torch.randint(0, num_classes, (batch_size,))
target = torch.randn(batch_size, num_features)

# Forward pass, loss calculation, and backpropagation
outputs = model(x, labels)
loss = criterion(outputs, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```
This final example demonstrates a different technique for injecting label information: a conditional Batch Normalization layer (CBN). Here, the label embeddings are used to scale (gamma) and shift (beta) the output of the batch normalization layer, effectively modulating the feature representations according to label information. By changing the per-channel mean and variance, we can tailor the activations to suit the input labels. This strategy is very popular for many conditional generative model, because it adds very little overhead while allowing different modes of variance/mean adjustments.

To deepen understanding, several resources are valuable. For theoretical grounding and a broad overview, 'Deep Learning' by Ian Goodfellow et al. provides a comprehensive examination of relevant concepts.  For a practical PyTorch focused perspective, the official PyTorch documentation is absolutely essential and a constant companion. Research papers on conditional image generation and conditional sequence models offer in-depth exploration into how these mechanisms are applied in specific domains. Exploring implementations of various conditional models on sites like GitHub can also illuminate specific approaches.  Finally, experimentation with different approaches, particularly varying embedding sizes, positions of concatenation, and model architectures, will offer an intuition for what works best.
