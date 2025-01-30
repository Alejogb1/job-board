---
title: "How can MLPs be used for efficient feature extraction and dimensionality reduction?"
date: "2025-01-30"
id: "how-can-mlps-be-used-for-efficient-feature"
---
Multilayer Perceptrons (MLPs), traditionally associated with classification and regression tasks, offer a surprisingly effective, and sometimes overlooked, approach to feature extraction and dimensionality reduction. My experience over several years has shown that careful architectural design, along with appropriate training methodologies, allows an MLP to learn a compressed, yet information-rich representation of the input data, enabling efficient downstream tasks.

The fundamental principle behind leveraging MLPs for this purpose rests on their ability to learn complex non-linear mappings. Unlike methods like Principal Component Analysis (PCA), which operates linearly, an MLP can capture non-linear relationships within the data. This is particularly crucial when dealing with real-world datasets where features are often correlated in intricate ways. The hidden layers of an MLP can be seen as learning a new set of features, transforming the original input space into a potentially lower-dimensional representation that retains salient information. The activation functions employed within the hidden layers are integral to this non-linearity. Rectified Linear Unit (ReLU) or its variants, Sigmoid, or hyperbolic tangent (tanh) activations all contribute to the MLP’s ability to represent complex input-output relationships. Critically, the dimensionality of these hidden layers, specifically those preceeding the final layer used for classification or regression in a typical network, acts as the mechanism for dimensionality reduction. If such a final layer is removed, and the penultimate layer is used as the output, then the MLP will act as a dimensionality reduction mechanism.

To achieve effective feature extraction, the design and training of the MLP must be considered. The architecture should typically consist of an input layer corresponding to the original feature space, followed by one or more hidden layers, each progressively decreasing in size, and then optionally a final layer for a task such as classification or regression. This structure encourages the MLP to learn increasingly compressed representations within each hidden layer. This progressive compression mirrors Autoencoder network structures. The final layer before any task specific output can be extracted and used for downstream applications. Furthermore, regularization techniques, such as L1 or L2 regularization, and dropout can be integrated into training to prevent overfitting and to encourage the MLP to learn a generalizable representation that is robust to noise. Furthermore, learning rates, batch sizes, optimizers, etc. all play a role in training the MLP well and therefore must be tuned. The activation functions of hidden layers often also play a role in the final quality of the extracted feature map.

Let's look at several code examples. The first will showcase how we would define an MLP suitable for extracting a compressed feature map using a PyTorch implementation:

```python
import torch
import torch.nn as nn

class FeatureExtractorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(FeatureExtractorMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim)) #Output Layer used for extracting feature map
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

input_dim = 784 # Example: flattened image of 28x28 pixels
hidden_dims = [256, 128] # Example: hidden layers decrease in size
output_dim = 64 # Desired dimensionality of the extracted features
activation=nn.ReLU()

model = FeatureExtractorMLP(input_dim, hidden_dims, output_dim, activation)
print(model)
```

In this example, `FeatureExtractorMLP` is a PyTorch module that defines an MLP with multiple hidden layers. The `input_dim` would correspond to the original dimensionality of the input data, such as the flattened version of a 28x28 pixel image. `hidden_dims` is a list specifying the size of each hidden layer, progressively decreasing to the `output_dim` parameter which defines the final dimensionality of our extracted feature map. The last linear layer effectively provides a compressed embedding. This network could be trained with backpropagation on a task where the final `output_dim` layer is directly used. This provides an output in the form of a vector, which may be used for downstream tasks. The `activation` parameter allows for user choice in activation functions.

The second example illustrates how this could be incorporated into a larger training loop. A key detail here is the task performed by the network. In this case it is a simple classification, although this could be any desired task given suitable modification to the output layer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Assume we have some data and labels
input_data = torch.rand((1000, 784)) # 1000 samples of 784 features
labels = torch.randint(0, 10, (1000,)).long()  # 10 class classification

dataset = TensorDataset(input_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# define our model, identical to the previous example, with the addition of a final layer
class FeatureExtractorMLPWithClassification(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_classes, activation=nn.ReLU()):
        super(FeatureExtractorMLPWithClassification, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim)) #Extraction layer
        self.feature_extractor = nn.Sequential(*layers) # feature extractor part of model

        self.classifier = nn.Linear(output_dim, num_classes) #Classification layer
    def forward(self, x):
        features = self.feature_extractor(x) #Pass input through feature extractor
        return self.classifier(features)  # Pass features through classifier

input_dim = 784
hidden_dims = [256, 128]
output_dim = 64
num_classes = 10 # number of classification classes
activation = nn.ReLU()

model = FeatureExtractorMLPWithClassification(input_dim, hidden_dims, output_dim, num_classes, activation)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# Extraction of Feature Map after training. Use the feature extractor layer only.
model.eval() # Set the model to evaluation mode for inference
with torch.no_grad():
  extracted_features = model.feature_extractor(input_data)

print(extracted_features.shape)

```

This code introduces a classification task following the feature extraction layers. The network is trained using standard backpropagation for classification. Once trained, the trained `feature_extractor` layers alone are used to extract the new, compressed feature map. In the print statement, the resulting feature map has a dimensionality of `torch.Size([1000, 64])`, as we trained on a sample of 1000 data points, and the final output dimension was 64. This feature map can then be used for any downstream task.

Finally, the third example illustrates how we can use Scikit-learn for a simpler implementation using their `MLPRegressor`, although the logic remains the same, and a classifier could be used instead. Here we demonstrate the use of the regressor which is useful when the output feature map represents an underlying variable and is not just a reduced dimension representation.

```python
from sklearn.neural_network import MLPRegressor
import numpy as np

#Generate random input data
X = np.random.rand(1000, 784)
y = np.random.rand(1000, 64)  # Example: 64-dimensional regression target

#Define MLPRegressor with specified architecture
mlp = MLPRegressor(hidden_layer_sizes=(256, 128),
                      activation='relu',
                      solver='adam',
                      max_iter=300,
                      random_state=42)

# Train the MLP
mlp.fit(X, y)

# Extract features.
extracted_features = mlp.predict(X)
print(extracted_features.shape)

```

Here we can see how the Scikit-learn library makes the implementation even simpler. We simply instantiate an MLPRegressor, specify the size of the hidden layers, train the model on the original input data and output map, and predict on the original data again to get our extracted feature map. Again, the output is of size 1000 x 64.

For further exploration, resources on deep learning frameworks such as PyTorch and TensorFlow are essential. Specifically, focus on the documentation related to `nn.Module` in PyTorch or `tf.keras.Model` in TensorFlow, as well as optimization algorithms like Adam or SGD. Additionally, understanding the theoretical underpinnings of backpropagation, non-linear activation functions, and regularization techniques will deepen your understanding.  Materials on dimensionality reduction techniques, such as PCA and t-SNE, will allow you to contrast the performance of MLPs with classical methods. Finally, numerous papers detail the application of MLPs to different feature extraction problems which would prove valuable to the interested reader.
