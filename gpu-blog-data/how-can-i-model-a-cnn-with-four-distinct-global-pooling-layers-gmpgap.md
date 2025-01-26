---
title: "How can I model a CNN with four distinct global pooling layers (GMP/GAP)?"
date: "2025-01-26"
id: "how-can-i-model-a-cnn-with-four-distinct-global-pooling-layers-gmpgap"
---

Implementing multiple global pooling layers within a Convolutional Neural Network (CNN), while not standard practice, can offer a pathway to capture varied feature representations across the convolutional feature maps. My experience building custom image analysis pipelines has demonstrated that while a single global pooling layer effectively reduces spatial dimensions, employing several can reveal subtle hierarchical feature information often lost in a single pass. This approach necessitates a careful understanding of how these pooled outputs interact and their subsequent routing into the fully connected layers.

The core challenge when employing multiple Global Max Pooling (GMP) or Global Average Pooling (GAP) layers stems from the desire to preserve spatial information without retaining spatial dimensionality, thus, global pooling operates by transforming a feature map tensor into a vector by either selecting the maximum activation value or computing the average activation value, respectively across the entire spatial dimension. If each pooling layer operates on the entire spatial feature map without spatial re-processing, all pooling layers will produce the exact same output vector. To achieve distinct pooling outputs, the output of the convolutional layers must be processed differently *before* being passed to each respective pooling layer. This can be accomplished via different branches of convolutional layer sets before each global pooling layer is applied. Each branch will act as a separate feature extractor before passing the outputs on to the global pooling layers. It's crucial to ensure the filter count for these branches eventually matches the expected input of the final dense layer(s).

Consider a hypothetical scenario where we aim to classify images using a CNN architecture. Instead of a single GMP layer, we design four separate convolutional paths, each leading to a distinct global pooling layer before feeding the flattened vector into fully connected layers for classification. This allows for a more diverse extraction of features at various scales or perspectives. The design also necessitates a specific strategy to concatenate the four different pooled outputs into a single vector before the dense layer(s) as they will not have similar channel depths. Below, I illustrate an example using PyTorch, which would generalize similarly to TensorFlow or other deep learning frameworks.

**Example 1: Basic Framework**

This snippet shows how the convolutional layers are split into four branches before the pooling layers.

```python
import torch
import torch.nn as nn

class MultiGlobalPoolCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiGlobalPoolCNN, self).__init__()
        # Shared initial convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Branch 1 with a few extra convolutional layers
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Branch 2, 3, and 4 each branch is a simple single convolution
        self.branch2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.branch3_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.branch4_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)


        # Global pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Branch 1 forward pass
        x1 = self.branch1_conv(x)
        x1 = self.pool1(x1)
        x1 = x1.view(x1.size(0), -1) # Flatten
        # Branch 2 forward pass
        x2 = self.branch2_conv(x)
        x2 = self.pool2(x2)
        x2 = x2.view(x2.size(0), -1) # Flatten
        # Branch 3 forward pass
        x3 = self.branch3_conv(x)
        x3 = self.pool3(x3)
        x3 = x3.view(x3.size(0), -1) # Flatten
         # Branch 4 forward pass
        x4 = self.branch4_conv(x)
        x4 = self.pool4(x4)
        x4 = x4.view(x4.size(0), -1) # Flatten

        # Concatenate
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Fully connected layers
        x = self.fc(x)
        return x

# Example instantiation
model = MultiGlobalPoolCNN(num_classes=10)
```

This foundational example illustrates a CNN with four separate branches leading to four distinct global pooling layers; note that the use of `AdaptiveMaxPool2d` and `AdaptiveAvgPool2d` as they guarantee the output size will be (1,1) regardless of input dimensions. The output from each global pooling layer is then flattened into a vector using the view function. The key is the concatenation step: the four flattened feature vectors, each with 128 features in this case, are combined using `torch.cat` along dimension 1, leading to a total vector length of 512 (128 * 4). The result is then used as the input to the fully connected layers.

**Example 2: Different Branch Architectures**

This example demonstrates a further development to the previous design by allowing different architectures within each branch, while maintaining the four distinct paths and overall structure. This is accomplished by defining separate sequential layers for each branch.

```python
import torch
import torch.nn as nn

class MultiGlobalPoolCNN_AdvancedBranches(nn.Module):
    def __init__(self, num_classes):
        super(MultiGlobalPoolCNN_AdvancedBranches, self).__init__()
        # Shared initial convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Branch 1: Two Conv layers
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Branch 2: Dilated Conv
        self.branch2_conv = nn.Sequential(
             nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
             nn.ReLU(),
        )

        # Branch 3: Only ReLU
        self.branch3_conv = nn.Sequential(
             nn.Conv2d(64, 128, kernel_size=1, padding=0),
             nn.ReLU()
        )
        # Branch 4: 2 layers
        self.branch4_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
            nn.ReLU()
        )

        # Global pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))


        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Branch 1
        x1 = self.branch1_conv(x)
        x1 = self.pool1(x1)
        x1 = x1.view(x1.size(0), -1) # Flatten
        # Branch 2
        x2 = self.branch2_conv(x)
        x2 = self.pool2(x2)
        x2 = x2.view(x2.size(0), -1) # Flatten
        # Branch 3
        x3 = self.branch3_conv(x)
        x3 = self.pool3(x3)
        x3 = x3.view(x3.size(0), -1) # Flatten
         # Branch 4
        x4 = self.branch4_conv(x)
        x4 = self.pool4(x4)
        x4 = x4.view(x4.size(0), -1) # Flatten

        # Concatenate
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Fully connected layers
        x = self.fc(x)
        return x

# Example instantiation
model = MultiGlobalPoolCNN_AdvancedBranches(num_classes=10)
```

This example shows the versatility possible with different branches. Notably, one branch uses dilated convolution to expand the receptive field. Another branch employs only a ReLU function, enabling the global pooling to act on a modified but minimally altered feature space. The key takeaway is that each branch is a separate module which can be independently designed according to the features desired to extract.

**Example 3: Branching with different filter counts**

This snippet demonstrates another approach where the branches have varied filter counts. In this example, each branch still ultimately outputs 128 feature channels, but the channel counts vary within each branch which results in different levels of feature processing at each point before the final global pooling. The pooling layers are still used to reduce spatial dimensions to (1,1), and then flattened and concatenated as before.

```python
import torch
import torch.nn as nn

class MultiGlobalPoolCNN_VariableFilters(nn.Module):
    def __init__(self, num_classes):
        super(MultiGlobalPoolCNN_VariableFilters, self).__init__()
        # Shared initial convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Branch 1: Increased channel count
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Branch 2: Decreased channel count
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
             nn.Conv2d(32, 128, kernel_size=3, padding=1),
             nn.ReLU()
        )
        # Branch 3: Single conv
        self.branch3_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Branch 4: Identity mapping
        self.branch4_conv = nn.Conv2d(64, 128, kernel_size=1, padding=0)



        # Global pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))


        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Branch 1
        x1 = self.branch1_conv(x)
        x1 = self.pool1(x1)
        x1 = x1.view(x1.size(0), -1) # Flatten
        # Branch 2
        x2 = self.branch2_conv(x)
        x2 = self.pool2(x2)
        x2 = x2.view(x2.size(0), -1) # Flatten
        # Branch 3
        x3 = self.branch3_conv(x)
        x3 = self.pool3(x3)
        x3 = x3.view(x3.size(0), -1) # Flatten
         # Branch 4
        x4 = self.branch4_conv(x)
        x4 = self.pool4(x4)
        x4 = x4.view(x4.size(0), -1) # Flatten

        # Concatenate
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Fully connected layers
        x = self.fc(x)
        return x


# Example instantiation
model = MultiGlobalPoolCNN_VariableFilters(num_classes=10)
```

This architecture highlights the flexibility in how each branch processes the feature maps. One branch expands the channel count before reducing it, thereby potentially capturing a more complex set of features before the global pooling operation. Another branch dramatically decreases the channel count before expanding it, which will provide a different feature representation to the other branches. As long as the output of each branch results in the same shape, this strategy can be employed.

**Resource Recommendations:**

To gain a deeper understanding of convolutional neural networks and their applications, consult foundational texts on deep learning. Books detailing practical implementations of CNN architectures, with examples in TensorFlow and PyTorch, provide valuable hands-on learning experiences. Additionally, research papers on novel CNN architectures, particularly those exploring variations on pooling layers, offer important theoretical perspectives. Online resources such as those offered by deep learning libraries themselves often contain the most up to date information regarding the implementation of models, including how to structure a model with different feature branches. A variety of academic journals such as IEEE transactions on pattern analysis and machine intelligence are useful for more cutting edge research in the field. Exploring examples within well-known model repositories can also be helpful to grasp different architectural patterns.
