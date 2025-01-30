---
title: "How can a feed-forward network be retrained with backpropagation and output adjustments?"
date: "2025-01-30"
id: "how-can-a-feed-forward-network-be-retrained-with"
---
The process of retraining a feed-forward neural network using backpropagation, particularly when needing to adjust its output layer’s behavior, necessitates a nuanced understanding of gradient descent and its interaction with network architecture. I've frequently encountered situations where initial model training yielded satisfactory performance overall but required specific adaptations in the output behavior, often to refine class probabilities or target particular regression scales. This adjustment goes beyond simply adding new training data; it involves modifying the network's final layer and carefully managing the backpropagation process to prevent catastrophic forgetting of previously learned representations.

Fundamentally, the backpropagation algorithm iteratively adjusts the weights within the network, beginning at the output layer and propagating backward to hidden layers, to minimize the difference between the network's predictions and the true target values, as defined by a loss function. When retraining and output adjustments are required, we’re not simply starting the training process anew. We're leveraging the existing learned representations, often by manipulating the final layer or modifying the loss function to emphasize specific output behaviors while preserving core features learned earlier. This can involve freezing early layers and only training the modified final layer and, perhaps, the last few layers.

Let's break down this process through practical examples. Assume we have a pre-trained network that classifies images into ten categories, but we now need to fine-tune it to improve the separation of two similar, but distinct, sub-classes within category 5.  The initial output layer is a fully connected layer with 10 neurons, each activated by a softmax function, representing a categorical probability distribution.

**Example 1: Fine-tuning with Modified Output**

Here we alter the output representation for classes five and six (the sub-classes), leaving other classes untouched. Instead of a single node representing the entire category five, we introduce two output neurons for the sub-classes within the fifth class (say, 5a and 5b), and remap the expected training data. This requires a change in the target vector and, therefore, the associated loss calculation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Assume 'pretrained_model' is the pre-trained network
# and 'train_loader' is a DataLoader for the training data, with adjusted targets
# for example, if the original target was '5', now one hot vector could be [0,0,0,0,1,0,0,0,0,0]
#and we have it converted to [0,0,0,0, 1, 0,0,0,0,0,0,0] and [0,0,0,0,0,1, 0,0,0,0,0,0]

# Reconfigure output layer, example:
num_classes_old = 10
num_classes_new = 12 # 10 original + 2 sub-classes
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes_new)
pretrained_model.final_activation = nn.Softmax(dim=1)

# Define loss function (cross entropy, adjusted for 12 classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

# Example training loop, using only the last few layers, freezing the earlier ones:
for name, param in pretrained_model.named_parameters():
    if 'fc' not in name: # 'fc' for the final, fully connected layer. Replace with correct layer if not final
        param.requires_grad = False # Freezing parameters

for epoch in range(5): # Run for a few epochs to fine tune
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = pretrained_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss {loss.item()}")
```

In this example, I am replacing the final fully connected layer (`pretrained_model.fc`), expanding it to handle the new sub-classes, and freezing the previous layers with `param.requires_grad = False` to primarily fine-tune only the adjusted portion of the network. The loss function and optimizer are instantiated with the network’s updated parameters. This avoids retraining layers that were already fine-tuned and effectively isolates our adjustments.

**Example 2: Bias towards specific outputs**

Often, when retraining, we don't need to radically change the structure but rather modify the loss function to penalize specific outputs more harshly or favor certain outcomes. Suppose we have a regression model that outputs a price and, during retraining, we find that it systematically underestimates the prices, in particular, prices that are above 100.  We can then adjust the loss function to pay more attention to the points where we see higher error. This does not need a change to the output representation but instead uses a modified loss function.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Assume 'pretrained_regressor' is the pre-trained network and 'train_loader' is a DataLoader
# with regression outputs
# and 'targets' is a 1D vector

# Define a custom loss function to emphasize larger errors, particularly with higher targets.
def custom_loss(outputs, targets):
    errors = (outputs - targets).abs()
    weights = torch.where(targets>100, (targets/100), torch.tensor(1.0)) # Increase weights for targets above 100
    weighted_errors = errors * weights
    return weighted_errors.mean() # Mean squared error using custom weights


optimizer = optim.Adam(pretrained_regressor.parameters(), lr=0.001)

for epoch in range(5):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = pretrained_regressor(inputs)
        loss = custom_loss(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss {loss.item()}")

```

Here, I defined a custom loss function that weighs errors based on the target value. Higher target values lead to larger errors being penalized more. The use of `torch.where` creates a weighting function that can be arbitrarily complex and enables the use of pre-existing networks where the architecture may be difficult to alter. The use of `outputs.squeeze()` in the custom loss removes the batch dimension from the model’s output to be used in tandem with the one-dimensional labels.

**Example 3: Output Layer Adaptation with Knowledge Distillation**

In this scenario, instead of completely overhauling the output, we can use knowledge distillation to transfer knowledge from the original pre-trained network to a modified network. This strategy is useful for transferring a complex probability distribution while adjusting output size. Consider that a pre-trained classification model gives probabilities for 1000 classes. We need to modify this to 20 categories. We can keep the pre-trained model output probabilities and use it as the target for the smaller model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Assume 'pretrained_classifier' is the teacher model with 1000 outputs,
# and student_model has 20
# Assume train_loader outputs only inputs

student_model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 20),
    nn.Softmax(dim=1)
)

def distillation_loss(student_logits, teacher_logits, temperature=5.0):
    student_probabilities = student_logits
    teacher_probabilities = (teacher_logits/ temperature).softmax(dim=1) # Soften teacher’s probabilities
    loss = nn.KLDivLoss(reduction="batchmean")(torch.log(student_probabilities), teacher_probabilities)
    return loss


optimizer = optim.Adam(student_model.parameters(), lr=0.001)

for epoch in range(5):
    for inputs in train_loader:
      with torch.no_grad():
          teacher_logits = pretrained_classifier(inputs)

      student_logits = student_model(inputs)
      loss = distillation_loss(student_logits, teacher_logits)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print(f"Epoch {epoch+1}, Loss {loss.item()}")

```

In this example, the `pretrained_classifier`’s logits serve as a target for the distillation loss, using the Kullback-Leibler divergence. The softmax over the teacher output has an increased temperature to soften the probability distribution and hence provide more guidance for the student model during training. The student model’s loss is then backpropagated, retraining the smaller model by incorporating the knowledge from the teacher. This demonstrates an alternative approach where the output layer is effectively modified through knowledge transfer rather than direct layer manipulation.

When retraining with backpropagation, it is essential to consider not only the code implementation but also proper hyperparameter tuning, such as learning rate, to avoid destabilizing already learned weights. Additionally, careful monitoring of both training and validation losses is necessary to prevent overfitting or underfitting the refined task.

For further exploration of these concepts, I recommend focusing on resources that discuss: *Transfer learning* and *fine-tuning* in the context of deep learning. Textbooks or tutorials that cover *loss function design* in detail are also beneficial. Lastly, resources detailing *knowledge distillation* techniques will offer additional insight into how to refine pre-trained networks by transferring knowledge between models. Understanding the nuances of these topics will greatly aid in retraining feed-forward networks with specific adjustments to output behavior.
