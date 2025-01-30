---
title: "How can word2vec embeddings be used to train a text classifier?"
date: "2025-01-30"
id: "how-can-word2vec-embeddings-be-used-to-train"
---
Word2vec embeddings, by transforming words into dense numerical vectors capturing semantic relationships, provide a crucial foundation for training effective text classifiers. These embeddings move beyond simple one-hot encoding, enabling models to generalize from unseen data by understanding the underlying meaning and context of words. Instead of treating words as discrete, unrelated tokens, word2vec allows classifiers to leverage the pre-existing structure of language.

The core idea is that, rather than directly feeding raw text into a classifier, we first represent each word in a text using its corresponding word2vec embedding. These embeddings are typically pre-trained on large text corpora, such as Wikipedia or Google News, and thus already encode significant linguistic information. This pre-training is essential because creating robust word embeddings from scratch usually requires an immense amount of data, exceeding the typical text classification dataset size.

A text classifier, in this context, learns to map sequences of word embeddings (representing a sentence or document) to a pre-defined set of categories or labels. The process generally involves these steps: First, tokenize the text into individual words. Then, for each word, look up its corresponding vector from the pre-trained word2vec model. If a word is not present in the vocabulary of the pre-trained model (an out-of-vocabulary or OOV word), it is handled either by ignoring it, mapping it to a special OOV vector (often a zero vector), or in some instances, employing more complex sub-word techniques. Next, given the set of word embeddings for a document, these embeddings are combined into a single representation of the whole text. This aggregation process is a critical step. Simple approaches involve averaging all word embeddings in the text. However, this approach loses information on the order of the words, and more advanced techniques, as I have found in practice with sentiment analysis classification, use sequence modeling architectures such as Recurrent Neural Networks (RNNs), specifically LSTMs or GRUs. These sequence models consider the word order, allowing the model to understand context. After the sequence has been processed into a representation, a dense layer followed by an activation function (e.g., softmax) for multi-class classification is applied. Finally, the network is trained to predict the correct class based on the label for the training instances.

The success of this approach lies in the quality of pre-trained embeddings and the architecture of the classification model. When I worked on a project classifying product reviews for online retailers, I found that using pre-trained embeddings dramatically improved results compared to hand-crafted features. Let's now explore some practical examples.

**Example 1: Averaging Word Embeddings for a Basic Classifier**

This example demonstrates a simple approach using Python with NumPy for word vector averaging and scikit-learn for classification. While rudimentary, it demonstrates the concept. Note that I am assuming the existence of a dictionary 'word_embeddings' that maps words to their word2vec vectors (likely loaded from a pre-trained model file), and the presence of training data `X_train` with tokenized sentences (each a list of words), and a list `y_train` with the corresponding class labels.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def document_vector(doc):
    # Computes the average vector for a given document
    vectors = [word_embeddings[word] for word in doc if word in word_embeddings]
    if len(vectors) == 0:
        return np.zeros(word_embeddings["example"].shape)  # Return a zero vector if all words are OOV
    return np.mean(vectors, axis=0)

X_train_vec = np.array([document_vector(doc) for doc in X_train]) # Convert list of documents into matrix of document vectors
X_test_vec = np.array([document_vector(doc) for doc in X_test]) # Convert list of documents into matrix of document vectors


# Train the Logistic Regression classifier
clf = LogisticRegression(random_state=0, solver='liblinear').fit(X_train_vec, y_train)

# Make Predictions
y_pred = clf.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

```

In this example, the `document_vector` function computes an average of word vectors for a given document, and it handles OOV words by returning a zero vector which will lead to a 'no signal' for this word and overall document, if the document is purely composed of OOV words. This simple averaging is fast but as mentioned previously, loses contextual information.  Logistic Regression, a linear classifier, is then trained on these averaged vectors. The classifier works reasonably well with this basic averaging of the word embeddings, but performance could be notably improved with more complex methods such as the subsequent example with sequence models.

**Example 2: Using an LSTM for Sequence-Based Classification**

This example showcases how we can utilize a Long Short-Term Memory (LSTM) network to capture sequential relationships in the text. I'll use PyTorch for this demonstration. Again, this assumes you have pre-loaded word embeddings.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Assuming X_train, X_test are lists of lists of word indices, padded to max length, and y_train, y_test are numerical labels
# Assuming word_embeddings_tensor is a tensor containing the embeddings with index of each word
# num_classes is the number of output classes

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(word_embeddings_tensor) #Initialize with our embeddings
        self.embedding.weight.requires_grad = False #Freeze pre-trained embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded) # Output is of shape [batch_size, seq_len, hidden_dim]
        output = output[:, -1, :] # Use only the output of the last time step
        output = self.fc(output)
        return output

#Hyper parameters
embedding_dim = word_embeddings_tensor.shape[1]
hidden_dim = 128
num_classes = len(set(y_train))
vocab_size = len(word_embeddings)
batch_size = 32
learning_rate = 0.001
epochs = 10

# Initialize tensors, dataloaders and model
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTMClassifier(embedding_dim, hidden_dim, num_classes, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%')


```
This example implements an LSTM-based classifier. The `nn.Embedding` layer is initialized with pre-trained embeddings, with the `.requires_grad = False` freezing these embeddings, which is good practice in most classification problems. The LSTM processes the sequence of word embeddings and the last output is then fed into a fully connected layer for classification. I found that for long form texts, this method consistently outperformed the averaged vector approach in my previous project for long form reviews and blog posts.

**Example 3: Using Convolutional Neural Networks (CNNs) for Text Classification**

Another approach involves using Convolutional Neural Networks (CNNs) for text classification. Though primarily designed for images, they can effectively extract local features from sequences of word embeddings. Here’s an example using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, num_classes, vocab_size):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(word_embeddings_tensor)
        self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2) # Convert to (batch, embedding_dim, seq_len)
        conved = [torch.relu(conv(embedded)) for conv in self.convs] # Convolution on each kernel size
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        concatenated = torch.cat(pooled, dim=1)
        output = self.fc(concatenated)
        return output

# Hyperparameters
embedding_dim = word_embeddings_tensor.shape[1]
num_filters = 100
kernel_sizes = [3, 4, 5]
num_classes = len(set(y_train))
vocab_size = len(word_embeddings)
batch_size = 32
learning_rate = 0.001
epochs = 10

# Initializing model
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = CNNClassifier(embedding_dim, num_filters, kernel_sizes, num_classes, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%')


```

In this example, the input embeddings are passed through multiple 1D convolutional layers with varying kernel sizes, which acts to detect local patterns in words. The results are then max-pooled and concatenated before being passed into a fully connected layer for classification. This can work well for certain types of text classifications, such as identifying important keywords within documents.

For resources to deepen your understanding, I recommend exploring the documentation for the libraries I've used: scikit-learn, PyTorch, and NumPy. Furthermore, research on techniques used in sequence modelling, like LSTMs and GRUs, is beneficial. A deep dive into papers on natural language processing that use word embeddings for text classification will also be invaluable. Finally, exploring online tutorials or blog posts that showcase practical examples of this type of classification can help solidify your understanding and introduce other useful libraries like gensim which provide pre-trained word embeddings. Using a combination of these resources, one can gain a better understanding of this widely employed technique.
