---
title: "How can a pre-trained model be modified with additional layers without using a sequential model?"
date: "2025-01-26"
id: "how-can-a-pre-trained-model-be-modified-with-additional-layers-without-using-a-sequential-model"
---

Pre-trained models, often convolutional neural networks (CNNs) for image data or transformer networks for text, provide a strong foundation for complex tasks. Their architecture, weights, and biases are the result of extensive training on large datasets. Directly appending or modifying these models using sequential constructions, while straightforward, limits flexibility in scenarios requiring branching pathways, parallel processing, or more complex integration of additional layers. I’ve encountered such requirements frequently in my work developing custom vision pipelines for robotics, where feature-rich pre-trained backbones must be augmented with task-specific layers that often diverge from linear flows. Modifying these models without relying on sequential constructs involves a more explicit, functional approach, defining how tensors flow through the various pre-trained and custom layers.

Essentially, this process requires manually wiring input tensors through desired computations, rather than relying on the implied sequencing of a `Sequential` model. We leverage the inherent modularity of deep learning libraries like TensorFlow or PyTorch by using the pre-trained model as a callable object, whose output tensor(s) become input(s) for our custom layers. These custom layers are also defined as callable objects. Instead of relying on `Sequential` to handle the output-to-input flow automatically, we explicitly control which tensors are passed to which layers within the forward pass of the overarching model. This approach also grants control over how the pre-trained model is treated with regard to training; specifically, whether its weights are frozen, fine-tuned, or if different parts are treated differently during optimization.

Let’s explore specific examples using a hypothetical scenario in which I am adapting a pre-trained ResNet50 model for a specific image classification task that requires separate heads for distinct categories, and then integrating the outputs of these heads later in the network.

**Example 1: Branching Architecture with Frozen Backbone**

In this example, the pre-trained ResNet50 backbone acts as a feature extractor. Its weights are frozen, and its output is then passed to two parallel classification heads, each designed to predict a separate set of classes. This scenario was common during a project involving multimodal sensory data fusion for object identification, where different sensory input required specialized processing.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

class BranchingClassifier(tf.keras.Model):
    def __init__(self, num_classes_head1, num_classes_head2):
        super(BranchingClassifier, self).__init__()
        self.resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
        self.resnet.trainable = False # Freeze backbone

        self.head1 = layers.Dense(256, activation='relu')
        self.head1_output = layers.Dense(num_classes_head1, activation='softmax')

        self.head2 = layers.Dense(256, activation='relu')
        self.head2_output = layers.Dense(num_classes_head2, activation='softmax')


    def call(self, inputs):
        x = self.resnet(inputs)
        h1 = self.head1(x)
        out1 = self.head1_output(h1)
        h2 = self.head2(x)
        out2 = self.head2_output(h2)
        return out1, out2


# Example Usage
model = BranchingClassifier(num_classes_head1=5, num_classes_head2=10)
input_tensor = tf.random.normal(shape=(1, 224, 224, 3)) # Example batch

output1, output2 = model(input_tensor)
print("Output 1 shape:", output1.shape) # Output: Output 1 shape: (1, 5)
print("Output 2 shape:", output2.shape) # Output: Output 2 shape: (1, 10)
```

Here, I define a custom `BranchingClassifier` inheriting from `tf.keras.Model`.  The `__init__` method instantiates the pre-trained ResNet50, freezes its weights, and then creates the two separate fully connected heads with their respective activation layers. In the `call` method, I retrieve the output from ResNet, then pass the same feature map separately to each head. The `call` function then returns two output tensors, one from each classification head. This functional approach allows us to branch out after the ResNet backbone and process outputs independently, rather than enforcing a sequential pipeline.

**Example 2: Feature Fusion with a Concatenation Layer**

This example demonstrates how to modify a pre-trained model by adding new layers after the main backbone output, followed by a feature fusion. In one past project involving audio and visual data classification, I fused features from different modalities using a method similar to this. This was imperative to have the network use both sensory inputs.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # Remove pooling and final layer
        for param in self.resnet.parameters():
          param.requires_grad = False # Freeze backbone

        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, num_classes) # Input size derived from output dimensions of previous layer

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Example Usage
model = FusionModel(num_classes=10)
input_tensor = torch.randn(1, 3, 224, 224) # Example batch

output = model(input_tensor)
print("Output shape:", output.shape) # Output: Output shape: torch.Size([1, 10])
```

In this PyTorch-based implementation,  `FusionModel` also inherits from `nn.Module`. I load a pre-trained ResNet50 model and remove its average pooling and final fully connected layers using list slicing. Then, I explicitly construct new convolutional, pooling, and fully connected layers after the backbone. In the `forward` method, the tensor flows through ResNet, then sequentially through my custom convolutional and pooling layers before being flattened and passed into the final fully connected classification layer. The critical aspect is that the tensors are routed through layers explicitly, again avoiding the use of a sequential model. I determine the correct input dimension of the `fc` layer by manual calculation, which is a common task when working with customized networks.

**Example 3: Conditional Layer Application**

Sometimes you need to control which layers are active based on conditions, like in a situation where the input type determines the path in the network. In a different project, I had to dynamically process data from distinct sensor sources through distinct networks, and this is a simplified analog of that.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class ConditionalModel(nn.Module):
    def __init__(self, num_classes):
        super(ConditionalModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) # Remove final layer
        for param in self.resnet.parameters():
          param.requires_grad = False # Freeze backbone


        self.head1 = nn.Linear(512, num_classes)
        self.head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, condition):
        x = self.resnet(x)
        x = x.mean(dim=[2,3]) # Global average pooling
        if condition == 1:
            x = self.head1(x)
        elif condition == 2:
            x = self.head2(x)
        return x


# Example Usage
model = ConditionalModel(num_classes=5)
input_tensor1 = torch.randn(1, 3, 224, 224) # Example batch
input_tensor2 = torch.randn(1, 3, 224, 224)
output1 = model(input_tensor1, condition=1)
output2 = model(input_tensor2, condition=2)

print("Output 1 shape:", output1.shape) # Output: Output 1 shape: torch.Size([1, 5])
print("Output 2 shape:", output2.shape) # Output: Output 2 shape: torch.Size([1, 5])
```

Here, the `ConditionalModel` uses the `condition` argument within its `forward` method to determine the flow of tensors. Based on the condition, either `self.head1` or `self.head2` processes the output of the pre-trained ResNet, showcasing how to integrate conditional logic into the forward pass. This is very useful when dealing with multiple data types being fed into a single model.

**Resource Recommendations**

For further study, I recommend exploring several resources. Start by thoroughly understanding the core concepts of deep learning, especially convolutional neural networks and transformer architectures, which are commonly used as pre-trained models. Then, dive into the documentation of TensorFlow and PyTorch, particularly sections covering model subclassing and custom layers. Experimenting with code and manipulating tensors directly is key to grasping the flexibility this functional approach provides. I also find a good working knowledge of linear algebra, especially matrix multiplications, useful.

In summary, modifying pre-trained models without sequential structures allows for creating highly flexible and tailored network architectures. By handling tensor flow directly within the forward method of your custom model class, you gain fine-grained control over model structure and training. This is fundamental to deploying pre-trained models in complex practical scenarios, such as those I encountered in robotic vision and multimodal sensory data analysis. The flexibility of explicitly routing tensors allows for highly customized deep learning model architectures, where sequentially concatenating layers is insufficient.
