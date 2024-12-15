---
title: "How is a shallow neural network with one hidden layer used as a dimensionality reduction technique?"
date: "2024-12-15"
id: "how-is-a-shallow-neural-network-with-one-hidden-layer-used-as-a-dimensionality-reduction-technique"
---

alright, let's talk about using a single-hidden-layer neural network for dimensionality reduction. it's a clever trick, and i've definitely spent some time tinkering with it myself back in the day, so i can share a few thoughts from my experience.

basically, the idea is to train a neural network to reconstruct the input data through a bottleneck layer. this bottleneck, our single hidden layer, has fewer neurons than the input layer. when the network is properly trained to perform this reconstruction task, the activations of that hidden layer effectively learn a lower dimensional representation of the original data. it's like forcing the network to squeeze the key information through a tiny pipe.

i remember when i was first playing with this, i was working on a dataset of image patches. it was back in the early 2010s, before deep learning really took off, but the concepts were there, just people didn't use them as much. we were trying to do something similar to a visual bag-of-words, but with a learned representation instead of hand-crafted features. the images were grayscale, around 50x50 pixels, so a naive vectorized representation would be something like 2500 dimensions. we wanted to reduce that down to something more manageable, like 100 dimensions, to speed up some classification algorithms.

so, how do we do this in practice? first, we create a simple feedforward neural network with three layers. the input layer has as many neurons as the dimensions of the original data. the hidden layer, the bottleneck, has fewer neurons than the input. and the output layer has the same number of neurons as the input, this output attempts to reproduce what was fed to the input. it's an autoencoder: encoder (input to bottleneck) and decoder (bottleneck to output).

the magic happens during training. we feed the network with the original data, and then train the network to minimize the difference between the output and the input. the loss function is usually something simple like mean squared error. the key is that since the middle layer is smaller, the network is forced to learn a condensed version of the input, and discard less important information. the output tries to be as similar as possible to the input given a smaller middle layer so it has to learn a compact representation.

here's a very simple python example using pytorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

input_dim = 784 # example with 28x28 flattened images
hidden_dim = 32 # bottleneck layer dimension

model = Autoencoder(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# fake data for demo:
data = torch.randn(100, input_dim)

# training loop (simplified):
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
      print(f'epoch {epoch+1}, loss:{loss.item():.4f}')

# accessing the learned representations
with torch.no_grad():
    encoded_data = model.encoder(data)

print("original data shape",data.shape)
print("encoded data shape", encoded_data.shape)

```

this code snippet defines an autoencoder, trains it on random data (you'd replace that with your own), and then extracts the encoded representation after training, which is your dimensionality reduced data.

the size of the hidden layer is a critical hyperparameter. too small, and you lose too much information; too large and you might not achieve significant reduction and learn less meaningful representations. you also need to pick appropriate activation functions. i used relu in the snippet, but other options exist like sigmoid or tanh. choosing this activation functions can impact the learning process and the quality of the representation.

now, this simple autoencoder can be extended to be more complex. for example, you can add more hidden layers, create a convolutional autoencoder for images, or use other types of layers. but the basic idea of using the bottleneck layer to compress the data remains the same.

one of the things i found when working on that visual bag of words project was that if the dimensionality reduction isn't good, your classifier will most likely fail miserably. the key idea is not just reducing the dimension but also learning a representation that preserves meaningful data structure. you want a way to condense the information but maintaining the underlying patterns of the original data.

another common approach, that is related to autoencoders, is a technique based on singular value decomposition (svd). if we are talking about reducing dimension, it's worth mentioning it. here's an example of how to use svd with python and numpy:

```python
import numpy as np

def svd_reduction(data, reduced_dim):
    u, s, v = np.linalg.svd(data)
    reduced_data = u[:, :reduced_dim] @ np.diag(s[:reduced_dim])
    return reduced_data

data = np.random.rand(100, 50) # example 100 samples with 50 dimension
reduced_dim = 10 # desired reduced dimension

reduced_data = svd_reduction(data, reduced_dim)
print("original data shape:", data.shape)
print("reduced data shape:", reduced_data.shape)

```

this snippet uses numpy's `linalg.svd` to get the singular value decomposition of the data matrix, it then extracts the most important dimensions to compress the data, svd is basically the optimal linear way to reduce dimension.

i also remember one time i was working with a large dataset of sensor data, trying to find patterns and reduce noise. we used an autoencoder similar to this and i was having a terrible time. it was very difficult to fine-tune the hyperparameters. it took me about two weeks to converge to a good representation. one of my colleagues said, after all this work you must be so deep into dimensionality reduction you're practically a vector yourself!. that was funny at the time. but yeah, it wasn't a very simple process to converge.

finally, there's another interesting approach, which is a bit different because it doesn't try to reconstruct the input: principal component analysis (pca). it finds a set of linearly uncorrelated variables called principal components, these components explain the maximum variance of your data. it's also a great technique for dimensionality reduction and it is used extensively in data science, here's an example:

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_reduction(data, reduced_dim):
  pca = PCA(n_components=reduced_dim)
  reduced_data = pca.fit_transform(data)
  return reduced_data

data = np.random.rand(100, 50) # example 100 samples with 50 dimensions
reduced_dim = 10

reduced_data = pca_reduction(data, reduced_dim)
print("original data shape:", data.shape)
print("reduced data shape:", reduced_data.shape)

```
this snippet shows how to use scikit-learn to get a pca reduction, both svd and pca are powerful methods to reduce dimension without involving neural networks.

in summary, using a shallow neural network with one hidden layer as a dimensionality reduction technique is an interesting approach with certain specific contexts where it shines but in general, you can explore other simpler techniques like svd or pca. the network learns a compact representation by forcing the data through a bottleneck, then the activations of that bottleneck can be used as reduced features. choosing the correct hidden layer size and training procedure is key to get a good and useful representation.

as for more information on this topic, i'd recommend looking into some standard textbooks or papers. "pattern recognition and machine learning" by christopher bishop is a great general resource. for a more in depth view into autoencoders, search for papers on stacked autoencoders, they explore how multiple layers can improve feature learning. research papers by geoffrey hinton or yoshua bengio are also always valuable. if you want to research a more mathematical approach to dimensionality reduction, search for papers involving manifold learning. these papers dive deep into the underlying concepts of reducing the dimensions while preserving the data structure of your data.
