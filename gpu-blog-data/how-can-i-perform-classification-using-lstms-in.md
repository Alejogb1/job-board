---
title: "How can I perform classification using LSTMs in PyTorch?"
date: "2025-01-30"
id: "how-can-i-perform-classification-using-lstms-in"
---
The core challenge in utilizing Long Short-Term Memory (LSTM) networks for classification lies in adapting their sequential processing nature to the discrete categories inherent in classification problems. I've observed that developers often struggle with converting the time-series output of an LSTM into a fixed-length vector suitable for a classifier, which is what I want to address here. The crux of a successful implementation involves choosing the appropriate output from the LSTM layer and then passing that output to one or more feed-forward layers, ultimately culminating in a softmax layer to generate probabilities for each class.

Firstly, understanding how LSTMs output data is crucial. An LSTM layer, when given a sequence of inputs, produces an output sequence of hidden states and optionally a final hidden state and cell state. For classification tasks, the final hidden state is typically the most informative summary of the entire input sequence, as it's the culmination of all the sequential processing the LSTM has performed. This final hidden state encapsulates the relevant information about the input sequence as encoded by the LSTM. However, be aware that depending on your specific problem and data, using the entire sequence output *can* sometimes yield better results, especially when attention mechanisms are implemented but that complexity is beyond our present scope. For this reason, I've noticed a tendency to focus on the final hidden state during initial experimentation.

Next, this final hidden state is then used as input to a fully-connected linear layer, or a series of them. Each linear layer transforms the data, and typically a non-linear activation function follows each layer except for the final one. Common choices are ReLU or tanh. The final linear layer needs to output a vector whose length matches the number of classes you are classifying into. A softmax activation applied to this final output produces the probability distribution across all classes. The class with the highest probability is predicted as the output.

The objective function during training is usually a cross-entropy loss, which quantifies the difference between the predicted probabilities and the actual class labels. Using a backpropagation algorithm the network learns to associate input sequences to the correct class. This loss function assumes that one input sequence is associated to one label, and this label is a one-hot encoded vector or, more generally, an integer number that represent a class.

Let's examine concrete code examples, highlighting these elements. The first example showcases a straightforward binary classification problem:

```python
import torch
import torch.nn as nn

class BinaryClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BinaryClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, (hidden_state, _) = self.lstm(x)
        # hidden_state shape: (num_layers, batch_size, hidden_size)

        # Use the final hidden state
        output = self.fc(hidden_state[-1, :, :])
        output = self.sigmoid(output)
        return output

# Example Usage
input_size = 10
hidden_size = 32
num_layers = 2
batch_size = 64
seq_len = 20

model = BinaryClassifierLSTM(input_size, hidden_size, num_layers)
input_data = torch.randn(batch_size, seq_len, input_size)
output = model(input_data)

print(output.shape) # Should print: torch.Size([64, 1])
```

In this example, the `forward` method extracts the final hidden state from the LSTM, and then transforms it using the `fc` layer to a single output. This layer produces a single real value that is then transformed using a sigmoid to generate a probability between 0 and 1. This probability is interpreted as the probability of the input belonging to class 1 (the other implicit class is 0), which is common in binary problems. Key to understand in this example is the indexing of the hidden state. As previously stated, the lstm outputs hidden and cell states for every layer and every time point in the sequence. Here we are specifically focusing on the final hidden states of the last layer. That is why we index using `[-1, :, :]` which can be read as: "take the last layer, all the examples in the batch and all of the hidden states in the layer".

Now, let's move on to multi-class classification. The primary difference is that the output layer must match the number of classes, and a softmax function is needed.

```python
import torch
import torch.nn as nn

class MultiClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MultiClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1) # Softmax over classes

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, (hidden_state, _) = self.lstm(x)
        # hidden_state shape: (num_layers, batch_size, hidden_size)

        # Use the final hidden state
        output = self.fc(hidden_state[-1, :, :])
        output = self.softmax(output)
        return output

# Example Usage
input_size = 10
hidden_size = 32
num_layers = 2
num_classes = 4
batch_size = 64
seq_len = 20

model = MultiClassifierLSTM(input_size, hidden_size, num_layers, num_classes)
input_data = torch.randn(batch_size, seq_len, input_size)
output = model(input_data)

print(output.shape) # Should print: torch.Size([64, 4])
```

Here, the `fc` layer maps the final hidden state to `num_classes` output units. The `softmax` function then normalizes this output into a probability distribution over the classes. The `dim=1` argument specifies that the softmax should be applied across the class dimension rather than the batch dimension. The main difference with the binary case, is that, besides changing the dimension of the output layer and the activation function, you need to change the loss function. Since you have a multi-class output the cross entropy loss function should be used.

Lastly, consider a case with multiple fully-connected layers after the LSTM. This can improve the model's ability to learn non-linear relationships:

```python
import torch
import torch.nn as nn

class ComplexClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ComplexClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, (hidden_state, _) = self.lstm(x)
        # hidden_state shape: (num_layers, batch_size, hidden_size)

        # Use the final hidden state
        output = self.fc1(hidden_state[-1, :, :])
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output


# Example Usage
input_size = 10
hidden_size = 32
num_layers = 2
num_classes = 4
batch_size = 64
seq_len = 20

model = ComplexClassifierLSTM(input_size, hidden_size, num_layers, num_classes)
input_data = torch.randn(batch_size, seq_len, input_size)
output = model(input_data)

print(output.shape) # Should print: torch.Size([64, 4])
```

Here, the final hidden state is passed through two linear layers, with ReLU activation between. The rationale is to increase the non-linearity of the network and enable the learning of more complex patterns.

In all the examples, the `batch_first=True` argument in the LSTM definition specifies that the input tensor will have the shape `(batch_size, seq_len, input_size)`, which is usually convenient. The output shape of each model should reflect the expected output from each classification tasks, which is either a probability (binary classification) or a vector of probabilities over all classes (multi class classification).

When developing these models, I've often found the following resources to be indispensable. For core PyTorch mechanics, the official documentation provides detailed explanations of all functionalities, including LSTMs, linear layers, and various activation functions. Standard textbooks covering deep learning and sequence modelling theory provide a firm mathematical foundation. In addition, numerous blog posts and tutorials explore the specific application of LSTMs to classification problems, which provide a practical perspective that complements the theoretical understanding gained through textbooks. In practice, experimentation with diverse datasets will show you what works best for your specific data.

To summarize, performing classification with LSTMs in PyTorch involves careful consideration of how the sequential output of the LSTM is adapted into a fixed-length representation suitable for classification. The final hidden state is the most common choice. These methods, when combined with appropriate linear layers and activation functions, and with the support of adequate learning resources, permit the generation of very powerful classifiers.
