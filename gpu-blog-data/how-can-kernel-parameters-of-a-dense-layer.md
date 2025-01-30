---
title: "How can kernel parameters of a dense layer be made non-trainable?"
date: "2025-01-30"
id: "how-can-kernel-parameters-of-a-dense-layer"
---
A common requirement when fine-tuning pre-trained deep learning models, or when implementing specific architectural constraints, is to freeze the kernel parameters of a dense layer, preventing them from being updated during backpropagation. This control over parameter training is achievable by manipulating the `trainable` attribute of the kernel variable within a deep learning framework, such as TensorFlow or PyTorch. This attribute, when set to `False`, effectively excludes a parameter from the optimization process.

The primary mechanism for achieving this non-trainability lies in accessing the specific kernel variable associated with the dense layer and modifying its `trainable` attribute. This approach works because the optimizers within these frameworks, such as Adam or SGD, only update variables marked as trainable. Any parameter that has been explicitly set to non-trainable is simply ignored during the gradient descent steps of the training process. This means that the kernel values will remain at their initialized or loaded values throughout the training cycle.

In TensorFlow, the approach involves directly manipulating the `kernel` attribute of a `tf.keras.layers.Dense` layer. Upon initialization, a dense layer's kernel is a `tf.Variable` object. This variable carries the weight matrix of the layer and typically starts trainable. To freeze these weights, we access the `kernel` attribute of the layer and set its `trainable` property to `False`. This must be performed after the layer has been created or loaded from a model. Once `trainable` is set to `False`, any training loop involving this layer will exclude the kernel parameters from gradient calculations, causing them to remain static.

Consider this TensorFlow code example:

```python
import tensorflow as tf

# Create a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Freeze the kernel of the first dense layer
model.layers[0].kernel.trainable = False

# Verify the trainable status
print("Layer 1 kernel trainable:", model.layers[0].kernel.trainable) # Expected output: False
print("Layer 2 kernel trainable:", model.layers[1].kernel.trainable) # Expected output: True


# Example of using the modified model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

import numpy as np
X_train = np.random.rand(1000, 100)
y_train = np.random.randint(0, 10, size=(1000,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

model.fit(X_train, y_train, epochs=2)

```
Here, a sequential model is created, consisting of two dense layers. We freeze the kernel of the first dense layer. After modification, printing `model.layers[0].kernel.trainable` confirms that the attribute is indeed `False`. The subsequent training process using dummy data demonstrates that while the second layer’s weights will be updated, the kernel in the first layer will remain fixed. Although the layer is still active during forward propagation, its weights won't change during backpropagation. This mechanism allows specific layers to retain their previously acquired characteristics. This is valuable in scenarios such as transfer learning, where fine-tuning a pre-trained model with a portion of the network’s weights fixed can result in enhanced accuracy with the new task.

PyTorch employs a similar mechanism, but with the `requires_grad` attribute of its weight tensors, stored as the `weight` attribute in a `torch.nn.Linear` layer. By setting `requires_grad` to `False` on this tensor, the optimizer is prevented from updating the corresponding weights. This, similar to TensorFlow, is performed post initialization or loading. We navigate the model's structure to access the specific linear layer and its weight tensor for this modification.

Consider the following PyTorch example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = SimpleModel()

# Freeze the weights of the first fully connected layer
model.fc1.weight.requires_grad = False

# Verify that the parameter is not trainable
print("Layer 1 weight requires_grad:", model.fc1.weight.requires_grad) #Expected output: False
print("Layer 2 weight requires_grad:", model.fc2.weight.requires_grad) #Expected output: True

# Example of using the modified model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

X_train = torch.tensor(np.random.rand(1000, 100), dtype=torch.float32)
y_train = torch.tensor(np.random.randint(0, 10, size=(1000,)), dtype=torch.long)

for epoch in range(2):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```
In this code, the first fully connected layer's weights are made non-trainable using `requires_grad = False`. Note that the optimizer, when initialized, uses the filter to only consider the trainable parameters from `model.parameters()`. Similar to the TensorFlow example, this ensures only the second linear layer's weights are updated during training, while the first remains static. This approach can be applied to any linear or dense layer within the PyTorch architecture.

Another important use case is when loading pre-trained weights. It is often necessary to retain the original parameters of certain pre-trained layers while fine-tuning the rest of the network with new data. The process of freezing parameters described is the most efficient approach to achieve this. For example, a ResNet’s earlier convolutional layers might remain unchanged, while the final fully-connected layer adapts to a new task with a different output dimension.

A more complex scenario might involve dynamically enabling or disabling trainability based on specific training criteria. Imagine a system with two training phases. Initially, only the final dense layers are updated, allowing them to adapt to the new task before fine-tuning the remaining network. Once the accuracy of these new layers reaches a certain threshold, the earlier layers may be thawed by setting their parameters' `trainable` attributes to `True`. This progressive training can lead to more stable and effective learning. This can be achieved in the training loop using conditional logic to change the `trainable` attributes of the layers dynamically.

Here is a code example of how to freeze or unfreeze a specific layer during a training process:
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

import numpy as np
X_train = np.random.rand(1000, 100)
y_train = np.random.randint(0, 10, size=(1000,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Pretraining only the last layer
for epoch in range(10):
    model.layers[0].kernel.trainable = False # Freezing the first layer
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, verbose=0)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f'Epoch {epoch+1}, Loss {loss:.4f}, Accuracy {accuracy:.4f}, Layer 1 Trainable: {model.layers[0].kernel.trainable}')

# Fine-tuning the whole model
model.layers[0].kernel.trainable = True # Unfreezing the first layer
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
for epoch in range(10):
    model.fit(X_train, y_train, epochs=1, verbose=0)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f'Epoch {epoch+1}, Loss {loss:.4f}, Accuracy {accuracy:.4f}, Layer 1 Trainable: {model.layers[0].kernel.trainable}')

```
This code demonstrates a two-phase training strategy. Initially, only the second layer is updated for ten epochs while keeping the first layer frozen. After ten epochs, the first layer is unfrozen to fine-tune the whole network for another ten epochs. Such a strategy allows gradual and targeted adaptation of different parts of the model. This ability to selectively freeze or unfreeze layers adds another dimension to fine tuning pre-trained models.

For a more thorough understanding, refer to the official documentation of TensorFlow and PyTorch for detailed explanations of layers, optimizers, and variable attributes. Additionally, consult deep learning textbooks and research papers covering transfer learning and fine-tuning strategies, where freezing network weights is a commonly used technique. Exploring implementations of established architectures, such as ResNets or VGG networks, will give more practical insight into how freezing is typically applied within existing models. These resources offer the comprehensive context necessary for leveraging the fine-grained control over trainable parameters.
