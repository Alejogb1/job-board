---
title: "What is the right way of mini-batching the validation set while training?"
date: "2024-12-14"
id: "what-is-the-right-way-of-mini-batching-the-validation-set-while-training"
---

alright, so you're asking about mini-batching the validation set during training, eh? it's a common gotcha, and i've definitely tripped over this myself a few times. i remember back when i was first messing around with convolutional neural nets for image recognition, i was naively just feeding the entire validation set in one go. it wasn't pretty. memory errors everywhere and the whole training process was sluggish, like a dial-up modem in 2024.

the problem, as i see it, is this: your validation set, while usually smaller than your training data, can still be pretty large. if you try to compute metrics and gradients on it all at once, you're going to run into memory issues, slow down your training loop, and frankly, it's just not how it's supposed to be done. validation is more about evaluating the current state of model not training the model. we need to get metrics to track model progress, not contribute on the training.

so, the right way is to break it down into mini-batches, just like you do with your training data. the key is to not backpropagate errors on the validation set though, we just calculate the metric. we want an independent estimation of progress, without contributing to model weights modification.

i've seen some folks get confused and start thinking that the validation batch size needs to be the same as the training batch size, or that it has to be small. it doesn't. your validation batch size can be whatever works well for your memory limitations. a common mistake is to use a batch size which is too big and cause memory overflow. usually i try to choose a size that allows efficient computation, but without overwhelming the gpu (if you are using one). it's a trade-off between processing speed and memory usage.

here's a basic example using pseudocode in python-like format:

```python
def validate_model(model, validation_data, batch_size, loss_function, metric_function, device):
    model.eval() # crucial: switch to evaluation mode, this disables dropout, batchnorm, etc.
    total_loss = 0
    total_metric = 0
    num_batches = 0

    with torch.no_grad(): # also crucial: no gradient calculation needed
        for batch_index in range(0, len(validation_data), batch_size):
            batch = validation_data[batch_index:batch_index+batch_size] # a simple way of batching
            inputs, targets = batch # assuming the batch contains both input and target tensors

            inputs = inputs.to(device) # move to the device
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            metric = metric_function(outputs, targets)

            total_loss += loss.item()
            total_metric += metric.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches

    return avg_loss, avg_metric
```
this is pretty straightforward. notice the `model.eval()` and the `torch.no_grad()`. these are vital. if you leave the model in training mode or allow gradient calculations, your validation results are not going to be true. it also makes a great difference in training time to not calculate the backpropagation on validation. also, using .item() on loss is important to retrieve only the value of the scalar in that moment. not doing it would lead to unexpected memory issues because the loss tensor retains the computational graph history.

that function is nice, but how do we call this? how do we retrieve the validation dataset to begin with? here's a small example of how to do it using a python list or similar, along with a simple example of the training loop and use of the validation routine:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

# sample data (replace with your actual data loading)
train_data = [torch.randn(1, 10) for _ in range(1000)]  # 1000 samples, each of size 10
train_labels = [torch.randint(0, 2, (1,)) for _ in range(1000)]
train_dataset = list(zip(train_data, train_labels))

val_data = [torch.randn(1, 10) for _ in range(200)] # 200 samples, each of size 10
val_labels = [torch.randint(0, 2, (1,)) for _ in range(200)]
val_dataset = list(zip(val_data, val_labels))


# simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# loss function and metric
def loss_function(outputs, targets):
    return f.cross_entropy(outputs, targets.squeeze())

def accuracy_metric(outputs, targets):
    predicted_classes = torch.argmax(outputs, dim=1)
    correct = (predicted_classes == targets.squeeze()).sum().item()
    return correct / len(targets)

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = optim.adam(model.parameters(), lr=0.001)

# parameters
num_epochs = 10
batch_size = 32
val_batch_size = 64  # validation batch size can differ

for epoch in range(num_epochs):
    model.train() # put the model in training mode
    for batch_index in range(0, len(train_dataset), batch_size):
        batch = train_dataset[batch_index:batch_index + batch_size]
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs).to(device) # stack the list of tensors in the batch
        targets = torch.stack(targets).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    val_loss, val_acc = validate_model(model, val_dataset, val_batch_size, loss_function, accuracy_metric, device)
    print(f"epoch {epoch+1} validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

print("training finished")
```
here, we set up a dummy training set and a dummy validation set. the important part is how we load these as batches in our training loop and validation loop. notice how the batch extraction from the dataset uses slices. and also, how the training and validation batches are not the same size, this highlights the fact that it is ok to do so. we also use `torch.stack` to convert the list of tensors that represent the batch into a single tensor ready to be used in the model.

regarding best practices i have a couple of thoughts, the most important one is to shuffle your validation set, even if less important than shuffling the train data. for each epoch, shuffle the validation data so that the validation process isn’t biased towards the way data is presented. we must remember the purpose of validation is to see if the generalization error on the model is acceptable. it must see different samples every epoch. you are not going to see good results if you keep feeding the same samples in the same order every epoch, even if not training on it, your model may become biased on those. i have seen it happen. it is not fun to debug issues that appear by only neglecting a simple shuffle function.

here's a modified version of the previous validation function that does the validation dataset shuffling, and shows the complete code including the training function as well:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import random

# sample data (replace with your actual data loading)
train_data = [torch.randn(1, 10) for _ in range(1000)]  # 1000 samples, each of size 10
train_labels = [torch.randint(0, 2, (1,)) for _ in range(1000)]
train_dataset = list(zip(train_data, train_labels))

val_data = [torch.randn(1, 10) for _ in range(200)] # 200 samples, each of size 10
val_labels = [torch.randint(0, 2, (1,)) for _ in range(200)]
val_dataset = list(zip(val_data, val_labels))


# simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# loss function and metric
def loss_function(outputs, targets):
    return f.cross_entropy(outputs, targets.squeeze())

def accuracy_metric(outputs, targets):
    predicted_classes = torch.argmax(outputs, dim=1)
    correct = (predicted_classes == targets.squeeze()).sum().item()
    return correct / len(targets)

def validate_model(model, validation_data, batch_size, loss_function, metric_function, device):
    model.eval()
    total_loss = 0
    total_metric = 0
    num_batches = 0

    with torch.no_grad():
        # shuffling validation set
        shuffled_validation_data = list(validation_data)
        random.shuffle(shuffled_validation_data)

        for batch_index in range(0, len(shuffled_validation_data), batch_size):
            batch = shuffled_validation_data[batch_index:batch_index+batch_size]
            inputs, targets = zip(*batch)
            inputs = torch.stack(inputs).to(device)
            targets = torch.stack(targets).to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            metric = metric_function(outputs, targets)

            total_loss += loss.item()
            total_metric += metric.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches
    return avg_loss, avg_metric

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = optim.adam(model.parameters(), lr=0.001)

# parameters
num_epochs = 10
batch_size = 32
val_batch_size = 64

for epoch in range(num_epochs):
    model.train()
    for batch_index in range(0, len(train_dataset), batch_size):
        batch = train_dataset[batch_index:batch_index + batch_size]
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    val_loss, val_acc = validate_model(model, val_dataset, val_batch_size, loss_function, accuracy_metric, device)
    print(f"epoch {epoch+1} validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

print("training finished")
```

it's a subtle change, but it makes a big difference, we now shuffle the validation set before doing the batches. also, it’s a good idea to track metrics on your validation set beyond just loss. accuracy, f1-score, precision, recall. whatever helps you understand how your model is doing. that's how to make sure your model works like a charm. also, try to be consistent in your evaluation methods and use a validation function in all experiments you do. it helps to track progress properly.

as for resources, i would avoid general tutorials and go for something more specific. the "deep learning" book by goodfellow, bengio and courville has a very good chapter on general model training and evaluation. also, you can find good papers on the subject on scientific databases like ieee xplore or similar, just make sure that you get papers on the last 5 years since this field is constantly updated. you can search by keywords related to training and deep learning validation methods.

oh, and a little joke i heard the other day while debugging: why did the neural network cross the road? to get to the other side... of the loss function. (i know, its terrible, sorry for that)

so yeah, mini-batch your validation data. keep your training loop clean, track your metrics, and you should be good to go. if you have more questions just ask, i'm usually around. happy coding!.
