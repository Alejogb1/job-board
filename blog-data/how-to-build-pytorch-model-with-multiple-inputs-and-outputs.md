---
title: "How to build PyTorch model with multiple inputs and outputs?"
date: "2024-12-16"
id: "how-to-build-pytorch-model-with-multiple-inputs-and-outputs"
---

Okay, let's tackle this one. I recall working on a fairly intricate multimodal machine learning project a few years back, involving both image and text inputs, and requiring, ultimately, not one but several distinct predictions. It's a classic case, really, and figuring out how to structure such a model in PyTorch required some careful planning. Building models with multiple inputs and outputs in PyTorch isn't inherently complex, but it does call for a methodical approach to ensure everything aligns properly both in the forward pass and during the backpropagation stage.

Fundamentally, the core idea hinges on leveraging PyTorch's modularity. We treat each input stream as a branch, processing them individually (or in concert), and then we combine these processed representations as needed to produce the different outputs. It's akin to a network of interconnected paths, where each path handles a distinct type of information.

First, let’s discuss the input handling. We need to accept multiple inputs simultaneously. PyTorch’s `torch.nn.Module` class makes this quite straightforward. We define our model's `forward()` method to receive these multiple inputs, each of which could be of a different shape or type. There is no 'one-size-fits-all' solution here; the specifics depend entirely on the nature of your data. For example, we could be dealing with image tensors (batch, channels, height, width) and text tensors (batch, sequence_length, embedding_dimension).

Then we have to consider the output side. Multiple outputs mean multiple loss functions potentially. And the outputs may even have different interpretations. For instance, in that old project I mentioned, we had one output branch predicting a classification label, another predicting a numerical value, and yet a third output representing a probability distribution. Each of these outputs was paired with a corresponding loss function appropriate for its type of prediction. The key is to keep everything explicitly defined and organized. This approach makes the model easy to debug, understand and maintain.

Now let me provide you with some concrete examples. We'll start with a simple case, then move to something a bit more involved.

**Example 1: Basic Image and Text Input, Two Outputs**

This example demonstrates a very basic model that takes an image (flattened) and text embedding as inputs and outputs a classification prediction and a regression value.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultimodalModel(nn.Module):
    def __init__(self, image_input_size, text_embedding_size, hidden_size=128):
        super().__init__()
        self.image_fc = nn.Linear(image_input_size, hidden_size)
        self.text_fc = nn.Linear(text_embedding_size, hidden_size)
        self.combined_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.classification_output = nn.Linear(hidden_size, 2)  # Binary classification
        self.regression_output = nn.Linear(hidden_size, 1)   # Regression

    def forward(self, image_input, text_input):
        image_processed = F.relu(self.image_fc(image_input))
        text_processed = F.relu(self.text_fc(text_input))
        combined = torch.cat((image_processed, text_processed), dim=1)
        combined_processed = F.relu(self.combined_fc(combined))

        classification = self.classification_output(combined_processed)
        regression = self.regression_output(combined_processed)

        return classification, regression

# Dummy input
image_input_size = 28 * 28  # Example flattened image
text_embedding_size = 100
batch_size = 32
dummy_image_input = torch.randn(batch_size, image_input_size)
dummy_text_input = torch.randn(batch_size, text_embedding_size)

model = SimpleMultimodalModel(image_input_size, text_embedding_size)
classification_output, regression_output = model(dummy_image_input, dummy_text_input)

print("Classification Output shape:", classification_output.shape)
print("Regression Output shape:", regression_output.shape)

```

In this example, we have separate linear layers to initially process image and text data independently, before concatenating, further processing, and branching out to generate our classification and regression outputs.

**Example 2: Using Convolutional Neural Network (CNN) for Image Processing**

Let's enhance this a little by using a simple CNN for image processing. This is a more realistic scenario.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMultimodalModel(nn.Module):
    def __init__(self, text_embedding_size, hidden_size=128):
        super().__init__()
        # Image CNN part
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Assuming 3 channels for RGB
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc_image = nn.Linear(32 * 7 * 7, hidden_size) # Assuming initial image size is 28x28

        # Text processing part
        self.text_fc = nn.Linear(text_embedding_size, hidden_size)
        self.combined_fc = nn.Linear(hidden_size * 2, hidden_size)

        # Output layers
        self.classification_output = nn.Linear(hidden_size, 5)  # 5-class classification
        self.probability_output = nn.Linear(hidden_size, 10)  # Probability distribution over 10 classes

    def forward(self, image_input, text_input):
        # Image processing
        x = self.pool(F.relu(self.conv1(image_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten for FC
        image_processed = F.relu(self.fc_image(x))

        # Text processing
        text_processed = F.relu(self.text_fc(text_input))

        # Combining the processing
        combined = torch.cat((image_processed, text_processed), dim=1)
        combined_processed = F.relu(self.combined_fc(combined))

        # Output branches
        classification = self.classification_output(combined_processed)
        probability = F.softmax(self.probability_output(combined_processed), dim=1)


        return classification, probability

# Dummy input
image_input_size = (3, 28, 28)
text_embedding_size = 100
batch_size = 32
dummy_image_input = torch.randn(batch_size, *image_input_size)
dummy_text_input = torch.randn(batch_size, text_embedding_size)


model = CNNMultimodalModel(text_embedding_size)
classification_output, probability_output  = model(dummy_image_input, dummy_text_input)

print("Classification Output shape:", classification_output.shape)
print("Probability Output shape:", probability_output.shape)
```

Here, a CNN extracts features from the image, which are then combined with the textual input. The crucial addition here is the probability output and a softmax layer to produce a proper probability distribution.

**Example 3: Different Loss Functions**

Finally, let's add a little complexity with different loss functions, each paired with its respective output. This is vital because different output types might require different approaches during training.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiLossModel(nn.Module):
    def __init__(self, image_input_size, text_embedding_size, hidden_size=128):
        super().__init__()
        self.image_fc = nn.Linear(image_input_size, hidden_size)
        self.text_fc = nn.Linear(text_embedding_size, hidden_size)
        self.combined_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.classification_output = nn.Linear(hidden_size, 3) # 3 class classification
        self.regression_output = nn.Linear(hidden_size, 1)
        self.probability_output = nn.Linear(hidden_size, 5) # 5 class distribution

    def forward(self, image_input, text_input):
        image_processed = F.relu(self.image_fc(image_input))
        text_processed = F.relu(self.text_fc(text_input))
        combined = torch.cat((image_processed, text_processed), dim=1)
        combined_processed = F.relu(self.combined_fc(combined))

        classification = self.classification_output(combined_processed)
        regression = self.regression_output(combined_processed)
        probability = F.softmax(self.probability_output(combined_processed), dim=1)
        return classification, regression, probability


# Dummy input
image_input_size = 28 * 28
text_embedding_size = 100
batch_size = 32
dummy_image_input = torch.randn(batch_size, image_input_size)
dummy_text_input = torch.randn(batch_size, text_embedding_size)


model = MultiLossModel(image_input_size, text_embedding_size)
classification_output, regression_output, probability_output = model(dummy_image_input, dummy_text_input)


# Dummy labels
classification_labels = torch.randint(0, 3, (batch_size,))
regression_labels = torch.randn(batch_size, 1)
probability_labels = torch.rand(batch_size, 5)


# Loss functions and Optimizer
classification_loss_fn = nn.CrossEntropyLoss()
regression_loss_fn = nn.MSELoss()
probability_loss_fn = nn.KLDivLoss(reduction='batchmean') # Using KL divergence

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Compute Losses
classification_loss = classification_loss_fn(classification_output, classification_labels)
regression_loss = regression_loss_fn(regression_output, regression_labels)
probability_loss = probability_loss_fn(probability_output.log(), probability_labels)

# Combine losses
total_loss = classification_loss + regression_loss + probability_loss

# Optimization
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("Total Loss:", total_loss.item())
```

In this final example, we explicitly showcase how multiple loss functions can be applied and combined for backpropagation. You’ll notice the separate loss calculation for each output branch, and their subsequent summation before backpropagating through the network. Note the probability branch used a KL divergence loss, often used with probability outputs, as opposed to CrossEntropyLoss typically used for classification labels, or MSE loss used with continuous regression outputs.

For further reading, I highly recommend “Deep Learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. It's a comprehensive resource and provides a solid understanding of the underlying principles. Additionally, research papers on multi-modal learning from venues like NeurIPS or ICML can be very insightful. Specifically, look into work concerning multimodal fusion and representation learning.

The core takeaway here is to think of your model as a series of modular components that are stitched together using PyTorch’s expressive framework, and treat each output with the appropriate loss function during training. With thoughtful design and clear coding practices, building sophisticated models with multiple inputs and outputs in PyTorch is very achievable.
