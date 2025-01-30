---
title: "Can PyTorch initialize two sub-modules with identical weights?"
date: "2025-01-30"
id: "can-pytorch-initialize-two-sub-modules-with-identical-weights"
---
Yes, PyTorch allows for the creation of two sub-modules within a larger model that share identical weight tensors, a behavior distinct from creating independent modules with the same initial *values*. This capability stems from PyTorch's object-oriented structure and the way it manages tensors as references rather than copies within `nn.Module` instances. Understanding this distinction is crucial for implementing certain model architectures, such as those used in Siamese networks or for weight tying. My experience has often led me to use this feature deliberately, although one must proceed with caution due to its implications.

The key difference lies in whether you're copying the tensor values to new tensors or directly sharing the same tensor object between module parameters. When two modules are initialized by simply applying the same initialization operation (e.g., normal distribution sampling), each module creates its own set of parameter tensors stored independently in memory. However, by assigning the same tensor object to multiple module parameter attributes, they are in fact linked. Modifications to the tensor in one module will simultaneously reflect in the linked tensor of another module because both are references to the same underlying data in memory. Backpropagation also impacts the shared tensor simultaneously, meaning gradients during training will accumulate for both modules together, effectively training them as a single unit with the same parameters. This can be used to construct efficient architectures that leverage parameter sharing.

Here's the first code example illustrating the behavior of independently initialized modules:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
      out1 = self.fc1(x)
      out2 = self.fc2(x)
      return out1, out2


net = SimpleNet(10, 5)

# Initial weights of fc1
print("fc1 weights before:", net.fc1.weight)

# Initial weights of fc2
print("fc2 weights before:", net.fc2.weight)


#Modify one weight
with torch.no_grad():
  net.fc1.weight[0,0] = 100

print("fc1 weights after:", net.fc1.weight)

print("fc2 weights after:", net.fc2.weight)
```

In this example, `SimpleNet` contains two fully connected layers, `fc1` and `fc2`. After instantiation, each layer is initialized with its own independent set of weights, as evidenced by the output of their weights before modification. Modifications applied to `fc1.weight` do not affect `fc2.weight`, demonstrating they are distinct tensors residing at different memory locations. This is the typical scenario where each module’s parameters are its own distinct instance.

Now consider the following example showcasing how we can achieve weight sharing:

```python
import torch
import torch.nn as nn

class SharedWeightNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc2.weight = self.fc1.weight
        self.fc2.bias = self.fc1.bias


    def forward(self, x):
      out1 = self.fc1(x)
      out2 = self.fc2(x)
      return out1, out2


shared_net = SharedWeightNet(10, 5)


print("fc1 weights before:", shared_net.fc1.weight)


print("fc2 weights before:", shared_net.fc2.weight)


with torch.no_grad():
  shared_net.fc1.weight[0,0] = 100


print("fc1 weights after:", shared_net.fc1.weight)


print("fc2 weights after:", shared_net.fc2.weight)
```

In `SharedWeightNet`, instead of having `fc2` initialize its own weight and bias tensors during the `nn.Linear` initialization, it is explicitly set to reference the same tensors as `fc1`. Consequently, both `fc1.weight` and `fc2.weight` now point to the same underlying tensor data. This is not a copy but rather an assignment that shares the tensor object. Modification of the weight tensor of `fc1` then immediately impacts the `fc2` module because they use the same tensor object.  This is the behavior you need for specific types of modules such as Siamese networks where one wants the encoders to have the same weights.

A third illustrative example that expands upon this with regards to backpropagation behavior should be considered:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SharedWeightNetBackprop(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedWeightNetBackprop, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc2.weight = self.fc1.weight
        self.fc2.bias = self.fc1.bias

    def forward(self, x):
      out1 = self.fc1(x)
      out2 = self.fc2(x)
      return out1, out2

input_size = 10
hidden_size = 5
shared_backprop_net = SharedWeightNetBackprop(input_size, hidden_size)

optimizer = optim.SGD(shared_backprop_net.parameters(), lr=0.01)
criterion = nn.MSELoss()


inputs = torch.randn(1, input_size)
targets1 = torch.randn(1, hidden_size)
targets2 = torch.randn(1, hidden_size)



optimizer.zero_grad()


outputs1, outputs2 = shared_backprop_net(inputs)

loss1 = criterion(outputs1, targets1)
loss2 = criterion(outputs2, targets2)

loss = loss1 + loss2
loss.backward()
optimizer.step()



print("fc1 weights after backprop:", shared_backprop_net.fc1.weight)
print("fc2 weights after backprop:", shared_backprop_net.fc2.weight)
```

In `SharedWeightNetBackprop`, both `fc1` and `fc2` again share the same weight and bias. The key difference here is backpropagation. When gradients are calculated with `.backward()`, both losses (`loss1` and `loss2`) contribute to the gradients for the same underlying shared weight tensor. Consequently, the optimizer step (`optimizer.step()`) simultaneously updates these shared tensors. The output will verify that these updated weights are the same in both `fc1` and `fc2`. If separate parameters were present, each would have been updated independently and not shared. This demonstrates that training updates will be reflected across all modules sharing the same tensors.

When implementing this shared-weight technique, several factors must be considered. First, care must be taken in the initialization. Attempting to initialize `fc1` and `fc2` independently using `nn.Linear` then force-assigning the tensors will overwrite any previously independent weight values on `fc2`. Hence in the `SharedWeightNet` and `SharedWeightNetBackprop` classes, `fc2`’s weights are immediately overwritten with those of `fc1`, ensuring a proper copy. Second, while this approach reduces memory usage by not storing separate copies, it has implications for parallelization strategies. Due to shared memory access, the concurrent updates on shared parameters may present issues if not handled properly for distributed processing. Further, the gradient accumulation becomes complex as loss from different modules which share the same tensor are added, potentially leading to undesired behavior if not meticulously managed.

For further exploration of similar concepts, I recommend reviewing sections on tensor manipulation, parameter management, and backpropagation within PyTorch’s official documentation. Further resources can also be found in papers describing the implementation and architecture of Siamese networks. Additionally, several tutorials on building advanced neural network models often cover parameter sharing as a design pattern. Pay special attention to the object model of pytorch, its use of tensors, and how updates are computed with respect to different parameters. Understanding how pytorch manages the underlying tensor representation will be critical to understanding how this works and how you can take advantage of it for specialized architectures.
