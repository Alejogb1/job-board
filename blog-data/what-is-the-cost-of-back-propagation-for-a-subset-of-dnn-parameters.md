---
title: "What is the Cost of back-propagation for a subset of DNN parameters?"
date: "2024-12-14"
id: "what-is-the-cost-of-back-propagation-for-a-subset-of-dnn-parameters"
---

alright, let's tackle this. the cost of backpropagation for a subset of dnn parameters, huh? that's a question that brings back memories of some late nights debugging. i've definitely been there, staring at a loss curve that just refused to cooperate.

basically, what we're talking about is how much computation it takes to update only a *portion* of the weights in a deep neural network, instead of all of them. it’s a common trick, especially if you have a large model and only want to tweak a small part of it – think fine-tuning a pre-trained model, or maybe you're experimenting with a new layer and don’t want to touch the rest.

so, the usual backpropagation process, you probably already have it in your mind. gradients are calculated layer by layer, from output back to the input. these gradients tell us how to adjust the weights to minimize the loss function. the magic here is the chain rule. the total cost, if i may say, in terms of computation, involves calculating gradients for *every* single parameter.

now, when we talk about a subset, things get interesting. it's not like we magically skip calculations. backpropagation inherently relies on computing gradients from layer to layer. even if we're only going to update a few parameters, the underlying machinery still needs to push the gradients back through the full network – until we get to that specific layer we intend to update.

imagine this: your network has, let's say, ten layers. but you only want to update the weights of layer number five, just an example. backpropagation will still compute gradients for layers ten, nine, eight, seven and six. it's only at layer five that we actually apply those updates. gradients for layers one, two, three and four will be computed, even if we discard them.

the cost breakdown is something like this: the forward pass is obviously the same, independent on the number of parameters. the difference is all in the backward pass. if we were updating all the parameters, we'd compute all the partial derivatives, multiply them with the learning rate and make the weight updates using an optimization algorithm. if we are only updating a subset of parameters, we have to compute all the partial derivatives until we get to the desired parameters, which we use, and, at the end, we don't use others. it’s not a huge win, computing all and updating only a subset. the big advantage is that the parameters that we did not update, remain with the previous parameters.

i recall one time, working on a sentiment analysis model, i had this massive transformer network trained on a huge dataset. i wanted to adapt it for a different domain, like reviews for electronics. i didn’t have time to retrain the whole thing. retraining takes a lot of time, hardware and patience. the approach i took was to freeze almost all the parameters, except the last layer which was specific to the classification. this meant only updating the weights in that specific layer.

another time, while working with a custom cnn for image segmentation, i tried to debug a network that was not performing well. initially i thought it was a bug somewhere, so i wrote several checks to see the shapes and values of the intermediate tensors. to check if it was not a problem of the input data, i did checks with more and more data and still could not find the issue. later, i realized that my network was too deep for my limited dataset. i tried to fine-tune only the parameters of the last layers, instead of updating all of them, to avoid overfitting.

here's the thing: backpropagation's cost is essentially proportional to the *number of operations* needed during both the forward and backward pass, not necessarily to the *number of parameters* that we will eventually update. calculating gradients is what's computationally intensive, especially with modern deep nets.

now, let's get to some code examples, i think these examples make it clear what i'm talking about:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# a toy example neural network
class toy_network(nn.Module):
    def __init__(self):
        super(toy_network, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = toy_network()

# say we only want to update layer 2 parameters
optimizer = optim.Adam([
    {'params': model.layer2.parameters()}
    ], lr=0.01)


criterion = nn.CrossEntropyLoss()
#random input and target, just for an example
input_data = torch.randn(1,10)
target = torch.randint(0, 5, (1,))
#forward pass
output = model(input_data)
#loss calculation
loss = criterion(output, target)
#backpropagation
optimizer.zero_grad()
loss.backward()
#optimization step
optimizer.step()

print("gradients for layer 1 are zero: ", model.layer1.weight.grad is None) # True
print("gradients for layer 2 are not zero: ", model.layer2.weight.grad is not None) #True
print("gradients for layer 3 are zero: ", model.layer3.weight.grad is None) # True

```

this code snippet shows a very simple network. we’re only updating the parameters of layer2. even though the optimizer only updates layer2, we still compute gradients for all the layers.

here’s a slightly more detailed example using pytorch, this time with more layers and different optimizers:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class complex_network(nn.Module):
    def __init__(self):
        super(complex_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128) #assuming input is 28x28 after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = complex_network()
#only update the last two layers parameters
optimizer = optim.Adam([
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()}
    ], lr=0.001)


criterion = nn.CrossEntropyLoss()
#random input and target, just for an example
input_data = torch.randn(1,3, 28, 28)
target = torch.randint(0, 10, (1,))
#forward pass
output = model(input_data)
#loss calculation
loss = criterion(output, target)
#backpropagation
optimizer.zero_grad()
loss.backward()
#optimization step
optimizer.step()


print("gradients for conv1 are zero: ", model.conv1.weight.grad is None)
print("gradients for conv2 are zero: ", model.conv2.weight.grad is None)
print("gradients for fc1 are not zero: ", model.fc1.weight.grad is not None)
print("gradients for fc2 are not zero: ", model.fc2.weight.grad is not None)

```

in this second example, we’ve got a cnn architecture, and we are updating only the parameters of the fully connected layers, the last two layers. still, we compute gradients for all layers but update only a subset.

a final example, with only one line changed on the previous one, but the implications are different.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class complex_network_2(nn.Module):
    def __init__(self):
        super(complex_network_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128) #assuming input is 28x28 after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = complex_network_2()
#this line changed from the example above, now we update all the parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
#random input and target, just for an example
input_data = torch.randn(1,3, 28, 28)
target = torch.randint(0, 10, (1,))
#forward pass
output = model(input_data)
#loss calculation
loss = criterion(output, target)
#backpropagation
optimizer.zero_grad()
loss.backward()
#optimization step
optimizer.step()

print("gradients for conv1 are not zero: ", model.conv1.weight.grad is not None)
print("gradients for conv2 are not zero: ", model.conv2.weight.grad is not None)
print("gradients for fc1 are not zero: ", model.fc1.weight.grad is not None)
print("gradients for fc2 are not zero: ", model.fc2.weight.grad is not None)

```

in this example, instead of updating a subset, we update all the parameters. just for a side-by-side comparison with the other snippet, so you can see the difference.

some resources you may find helpful in further understanding this topic:

*   *deep learning* by ian goodfellow, yoshua bengio, and aaron courville. a classic text that covers backpropagation in great detail, including its computational aspects. i think it’s the go to book for a solid foundation.
*   *neural networks and deep learning* by michael nielsen, it’s available online. it explains the concepts in a very clear way, and is a great source for learning the foundations of backpropagation.
*   for more advanced stuff, look into papers on topics like *efficient backpropagation algorithms*, you may find a lot of recent advances in this area, for instance related to memory management during the backpropagation. you can use the search engines like semantic scholar or google scholar to look for them.

to wrap it up, while updating only a subset of parameters doesn’t save you from computing all gradients, it helps you fine-tune specific parts of the network without affecting the rest. and, sometimes, that’s the only thing that will save your sanity while debugging. if not, you might need to go for a long walk, or maybe a second coffee, or a third... don't judge me.
