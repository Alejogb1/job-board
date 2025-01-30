---
title: "How can PyTorch Lightning be used for transfer learning with stacked LSTMs?"
date: "2025-01-30"
id: "how-can-pytorch-lightning-be-used-for-transfer"
---
Transfer learning, often the most efficient approach to training models with limited labeled data, is particularly effective with sequential data when paired with recurrent neural networks like LSTMs. I’ve personally seen its impact across time-series forecasting projects, where pre-training on a large dataset and then fine-tuning on a smaller target dataset consistently yields significant performance improvements. PyTorch Lightning streamlines this process, providing a robust and organized framework for managing complex training workflows, including those involving stacked LSTMs. The core advantage lies in abstracting away boilerplate code, allowing a focus on the architectural design and the specifics of the transfer learning methodology itself.

Firstly, let's define what stacked LSTMs are in this context. They're essentially multiple LSTM layers stacked on top of each other, where the output sequence of one layer becomes the input sequence of the next. This enables the network to learn more intricate temporal dependencies by creating hierarchical representations of the sequence data. The number of layers, along with the hidden state size within each LSTM, are key hyper-parameters that influence the model's capacity.

Now, let’s examine how PyTorch Lightning facilitates transfer learning with such a model. The first step is usually defining a custom `LightningModule` which encapsulates the model architecture, the forward pass, the loss calculation, and the optimization steps. This encapsulation is crucial for clean code and for enabling PyTorch Lightning's powerful training loops.

In the case of transfer learning, we'd typically start with a pre-trained model. The key is to ensure the input/output shapes of the layers we intend to freeze or modify align with the target dataset. Here's an example demonstrating a minimal approach:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) #take only last time step
        return out

class LightningLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-3):
        super().__init__()
        self.model = StackedLSTM(input_size, hidden_size, num_layers, output_size)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Dummy data
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
seq_len = 30
batch_size = 32
num_batches = 10
X_train = torch.randn(num_batches * batch_size, seq_len, input_size)
y_train = torch.randn(num_batches * batch_size, output_size)
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size)

# Instantiate Lightning Module
lightning_model = LightningLSTM(input_size, hidden_size, num_layers, output_size)

# Create Trainer
trainer = pl.Trainer(max_epochs=10)

# Train
trainer.fit(lightning_model, train_loader)

```

In this initial example, I've defined `StackedLSTM` as a regular PyTorch module and `LightningLSTM` as its Lightning counterpart. The `LightningLSTM` module handles the training logic, defining the `training_step` and `configure_optimizers` methods, allowing PyTorch Lightning to manage the training loop. We initialize with dummy data and a simple MSE loss. This is a basic setup; it does not yet include transfer learning, and is simply an example of how to structure the code for PyTorch Lightning.

To illustrate transfer learning with PyTorch Lightning, consider the scenario where we have a model pre-trained on a larger dataset with a different output size. We then want to fine-tune this model on a new dataset. Here's an example focusing on layer freezing and modifying the final layer.

```python
# Pre-trained model initialization (assume this part is loaded from disk)
pretrained_model = StackedLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=10)

# Assume pretrained_model is loaded with a previous weight state using torch.load

class TransferLearningLSTM(pl.LightningModule):
    def __init__(self, pretrained_model, output_size, learning_rate=1e-3):
        super().__init__()
        self.lstm_layers = pretrained_model.lstm  # Access the LSTM layers
        for param in self.lstm_layers.parameters():
             param.requires_grad = False # Freeze LSTM layers

        self.fc = nn.Linear(pretrained_model.fc.in_features, output_size)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm_layers(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Assume we have new training data and the target output size has changed
output_size = 5
X_train_new = torch.randn(num_batches * batch_size, seq_len, input_size)
y_train_new = torch.randn(num_batches * batch_size, output_size)
new_train_data = TensorDataset(X_train_new, y_train_new)
new_train_loader = DataLoader(new_train_data, batch_size=batch_size)


transfer_model = TransferLearningLSTM(pretrained_model, output_size)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(transfer_model, new_train_loader)
```

In this modification, `TransferLearningLSTM` initializes with the `pretrained_model`. We access the LSTM layers via `pretrained_model.lstm` and freeze their weights using `param.requires_grad = False`. We then replace the original fully connected layer (`pretrained_model.fc`) with a new one compatible with the new output size, keeping the hidden feature dimension the same. Now the `TransferLearningLSTM` is only optimizing the weights for the fully connected output layer and will not modify the internal LSTM weights.

Finally, consider a more complex scenario where we not only want to replace the final layer but also fine-tune specific layers in the LSTM network. We achieve this by carefully setting `requires_grad` on certain layers only.

```python
class SelectiveFineTuneLSTM(pl.LightningModule):
    def __init__(self, pretrained_model, output_size, layers_to_finetune = 1, learning_rate=1e-3):
        super().__init__()
        self.lstm_layers = pretrained_model.lstm
        #Freeze all LSTM layers
        for param in self.lstm_layers.parameters():
             param.requires_grad = False

        #Unfreeze last n LSTM layers
        for layer in self.lstm_layers.lstm.children(): # access layers directly
            if isinstance(layer, nn.LSTM):
                num_lstm_layers = len(list(layer.children()))
                layers_to_unfreeze = min(layers_to_finetune, num_lstm_layers)
                for i, param in enumerate(layer.parameters()):
                   if i < layers_to_unfreeze*2: # each lstm layer has two types of weight parameters and bias parameters to tune
                       param.requires_grad = True # only unfreeze last n layers
        self.fc = nn.Linear(pretrained_model.fc.in_features, output_size)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()


    def forward(self, x):
       out, _ = self.lstm_layers(x)
       out = self.fc(out[:, -1, :])
       return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
         return optim.Adam(self.parameters(), lr=self.learning_rate)


# Assume we still have new training data and the target output size has changed
layers_to_finetune=1 # select last one LSTM layer to fine tune
finetune_model = SelectiveFineTuneLSTM(pretrained_model, output_size, layers_to_finetune )

trainer = pl.Trainer(max_epochs=10)
trainer.fit(finetune_model, new_train_loader)
```

Here, `SelectiveFineTuneLSTM` introduces an additional parameter `layers_to_finetune`.  This lets us decide how many of the *final* LSTM layers should be unfrozen. The nested loop iterates through the LSTM layers and selectively sets `requires_grad` based on the requested `layers_to_finetune` parameter, allowing fine-grained control over which layers are updated during training.

For further exploration, I would recommend consulting the official PyTorch Lightning documentation, as it provides comprehensive explanations of core concepts like `LightningModule`, trainers, and optimizers. The book "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann also offers a detailed perspective on both PyTorch and neural networks in general. Additionally, papers on recurrent neural networks, transfer learning, and fine-tuning would greatly expand understanding of the underlying principles and best practices.
