---
title: "Can sequential models be used with SMILES strings for deep learning applications?"
date: "2025-01-30"
id: "can-sequential-models-be-used-with-smiles-strings"
---
The intrinsic sequential nature of Simplified Molecular Input Line Entry System (SMILES) strings, representing chemical structures as sequences of characters, makes them fundamentally suitable for sequential deep learning models. Having spent the past several years developing cheminformatics tools, I've found that recurrent neural networks (RNNs), particularly Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), as well as transformer architectures, are effective in processing these strings for diverse tasks. Their capacity to capture dependencies between characters, a direct proxy for atom-to-atom relationships in a molecule's structural formula, is key to this success.

The efficacy stems from how sequential models handle data. RNNs, LSTMs, and GRUs process input elements one at a time, maintaining an internal state, or memory, that is updated at each step. For SMILES, this translates to reading the string character-by-character, with the network implicitly learning the rules of SMILES syntax and the chemical significance of character groupings. For example, the sequence "CC(=O)O" represents acetic acid. A sequential model, trained with a sufficient quantity of similar SMILES strings, can learn that "CC" commonly appears as an ethyl group, "(=O)" represents a carbonyl group, and "O" an oxygen atom. The order in which these symbols appear dictates the overall structure. Transformers, on the other hand, use attention mechanisms to relate all characters in a sequence to each other, often achieving better performance, albeit with greater computational cost.

Here's a practical demonstration of how sequential models can be used with SMILES. These examples illustrate tasks like sequence generation, property prediction, and molecular representation learning. I'm using Python and PyTorch here, common tools in this space.

**Example 1: SMILES String Generation using an LSTM**

This demonstrates a simplified text generation task, training an LSTM to learn the syntax of SMILES and generate new, albeit potentially invalid, molecules.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Simplified Dataset class for SMILES strings
class SMILESDataset(Dataset):
    def __init__(self, smiles_strings, char_to_idx, seq_length=128):
        self.smiles_strings = smiles_strings
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length

    def __len__(self):
       return len(self.smiles_strings)

    def __getitem__(self, idx):
        smiles = self.smiles_strings[idx]
        encoded_seq = [self.char_to_idx[char] for char in smiles]
        encoded_seq = encoded_seq[:self.seq_length] # Truncate longer seqs

        # Pad shorter sequences
        padded_seq = np.pad(encoded_seq, (0, self.seq_length - len(encoded_seq)), 'constant', constant_values=0)
        
        input_seq = padded_seq[:-1]
        target_seq = padded_seq[1:]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


# Basic LSTM model
class SMILESGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SMILESGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

# Example SMILES strings and vocabulary
smiles_strings = ["CC(=O)O", "c1ccccc1", "CCO", "CC(C)CC", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C[C@H]1[C@@H]([C@H](C[C@H](O1)O)O)O"] # Some random SMILES
chars = sorted(list(set("".join(smiles_strings))))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
vocab_size = len(chars)
seq_length = 64 # Set max sequence length

dataset = SMILESDataset(smiles_strings, char_to_idx, seq_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

embedding_dim = 64
hidden_dim = 128
model = SMILESGenerator(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        hidden = (torch.zeros(1, inputs.size(0), hidden_dim),
                  torch.zeros(1, inputs.size(0), hidden_dim))
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Generation function
def generate_smiles(model, char_to_idx, seq_length, start_char = 'C', num_chars=100):
    model.eval()
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    start_idx = char_to_idx[start_char]
    input_seq = torch.tensor([[start_idx]], dtype=torch.long)
    generated_smiles = start_char
    hidden = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))


    for _ in range(num_chars):
        output, hidden = model(input_seq, hidden)
        probs = torch.softmax(output[0, -1], dim=0).detach().numpy()
        next_idx = np.random.choice(vocab_size, p=probs) # Sample
        next_char = idx_to_char[next_idx]
        generated_smiles += next_char
        input_seq = torch.tensor([[next_idx]], dtype=torch.long)
    return generated_smiles

# Generate sample SMILES
generated_smiles = generate_smiles(model, char_to_idx, seq_length)
print("Generated SMILES:", generated_smiles)
```
This snippet defines a rudimentary LSTM model and trains it on a small dataset of SMILES strings. The model's objective is to predict the next character in the sequence. I utilize a custom dataset class to convert SMILES strings into numerical sequences and a generation function for sampling. This is a highly simplified version and requires more extensive data for meaningful output.

**Example 2: SMILES-Based Property Prediction using a GRU**

Here, the focus shifts to regression. I train a GRU network to predict a fictional property value (e.g., boiling point, reactivity score) based on a given SMILES string.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset with associated property values
class PropertyDataset(Dataset):
    def __init__(self, smiles_strings, property_values, char_to_idx, seq_length=128):
        self.smiles_strings = smiles_strings
        self.property_values = property_values
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length


    def __len__(self):
       return len(self.smiles_strings)


    def __getitem__(self, idx):
        smiles = self.smiles_strings[idx]
        encoded_seq = [self.char_to_idx[char] for char in smiles]
        encoded_seq = encoded_seq[:self.seq_length]

        padded_seq = np.pad(encoded_seq, (0, self.seq_length - len(encoded_seq)), 'constant', constant_values=0)
        
        property_value = self.property_values[idx]
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(property_value, dtype=torch.float)

# GRU model for property prediction
class PropertyPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PropertyPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) # Single output for regression


    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden[-1])
        return output

# Example data
smiles_strings = ["CC(=O)O", "c1ccccc1", "CCO", "CC(C)CC", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C[C@H]1[C@@H]([C@H](C[C@H](O1)O)O)O"]
property_values = [118.0, 80.1, 78.3, 27.8, 290.0, 200.0] # Fictional values

chars = sorted(list(set("".join(smiles_strings))))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
vocab_size = len(chars)
seq_length = 64

dataset = PropertyDataset(smiles_strings, property_values, char_to_idx, seq_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

embedding_dim = 64
hidden_dim = 128
model = PropertyPredictor(vocab_size, embedding_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

#Prediction
with torch.no_grad():
    smiles_test = "CC(=O)O"
    test_input = torch.tensor([char_to_idx[char] for char in smiles_test], dtype=torch.long).unsqueeze(0)
    padded_test_input = nn.functional.pad(test_input, (0,seq_length - test_input.shape[1]), mode='constant', value=0)
    prediction = model(padded_test_input).item()
    print(f"Predicted property value for {smiles_test}: {prediction}")
```

This second example trains a GRU to predict a numerical property. The key difference is the GRU layer and the linear layer at the end to regress to a single value. Additionally, the loss function switches to Mean Squared Error (MSE). I again used a fictional dataset for demonstration.

**Example 3:  SMILES-Based Molecular Representation using a Transformer Encoder**

Lastly, I'll showcase a simplified transformer encoder for learning molecular representations. This demonstrates how more complex architectures can process SMILES and extract meaningful latent vector representations. These vectors can be used in downstream tasks like clustering, similarity searches or other prediction models.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset for representation learning (no associated property)
class RepresentationDataset(Dataset):
    def __init__(self, smiles_strings, char_to_idx, seq_length=128):
        self.smiles_strings = smiles_strings
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length


    def __len__(self):
       return len(self.smiles_strings)


    def __getitem__(self, idx):
        smiles = self.smiles_strings[idx]
        encoded_seq = [self.char_to_idx[char] for char in smiles]
        encoded_seq = encoded_seq[:self.seq_length]

        padded_seq = np.pad(encoded_seq, (0, self.seq_length - len(encoded_seq)), 'constant', constant_values=0)

        return torch.tensor(padded_seq, dtype=torch.long)

# Simplified Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim), num_layers)
        self.fc = nn.Linear(embedding_dim, hidden_dim)


    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(0,1)
        output = self.transformer(embedded) # [seq_len, batch_size, embedding_dim]
        output = output.transpose(0,1) # [batch_size, seq_len, embedding_dim]
        pooled_output = torch.mean(output, dim=1) # Pool across sequence dimension to get a vector per sequence
        output = self.fc(pooled_output)
        return output

# Example data
smiles_strings = ["CC(=O)O", "c1ccccc1", "CCO", "CC(C)CC", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C[C@H]1[C@@H]([C@H](C[C@H](O1)O)O)O"]

chars = sorted(list(set("".join(smiles_strings))))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
vocab_size = len(chars)
seq_length = 64

dataset = RepresentationDataset(smiles_strings, char_to_idx, seq_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


embedding_dim = 64
hidden_dim = 128
num_heads = 2
num_layers = 2

model = TransformerEncoder(vocab_size, embedding_dim, hidden_dim, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    for inputs in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.mean(torch.sum(outputs**2, dim=1)) # Arbitrary loss for demonstration
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Get Representation
with torch.no_grad():
    smiles_test = "CC(=O)O"
    test_input = torch.tensor([char_to_idx[char] for char in smiles_test], dtype=torch.long).unsqueeze(0)
    padded_test_input = nn.functional.pad(test_input, (0,seq_length - test_input.shape[1]), mode='constant', value=0)
    representation = model(padded_test_input)
    print(f"Representation vector for {smiles_test}: {representation.squeeze().detach().numpy()}")
```
In the last example, I’ve built a transformer encoder.  The key point is generating a vector that represents the molecule as a whole. In real projects, this would typically be used as a feature vector for downstream tasks. I also implemented mean pooling to obtain the representation vector.

To further deepen understanding, I recommend focusing on materials that discuss the theoretical basis of recurrent neural networks and transformers. The original papers describing these architectures, as well as textbooks on natural language processing or deep learning with a focus on sequence data provide necessary fundamental background. Tutorials focusing on cheminformatics and deep learning, particularly with the use of PyTorch or TensorFlow, are excellent for understanding the practical aspects and the implementation techniques. There are several excellent online courses that cover these topics as well. Exploring examples on open-source repositories such as GitHub is also invaluable for seeing real-world implementations.
