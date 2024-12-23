---
title: "How can I adjust the number of classes in a pre-trained model?"
date: "2024-12-23"
id: "how-can-i-adjust-the-number-of-classes-in-a-pre-trained-model"
---

Okay, let's tackle this. It’s a common challenge, and one I remember facing vividly a few years back when we were transitioning our image recognition pipeline from a very general pre-trained model to something tailored for a specific product category. The need to adjust the number of classes in a pre-trained model, especially in deep learning, usually arises when you want to fine-tune the model for a dataset that has a different number of output categories than the original model was trained on. Directly using a pre-trained model 'as is' often leads to poor performance because its final classification layer, which provides the output probabilities for each category, is not aligned with your specific task.

The fundamental approach involves replacing or modifying the final classification layer of the pre-trained model, which is typically a fully connected layer with a softmax activation. The key here is understanding that the pre-trained layers, especially the convolutional layers in convolutional neural networks (CNNs), have learned hierarchical feature representations from the original dataset which are still beneficial for your new task. We want to leverage those learned features, not throw them away.

There are a few common strategies I've found effective, each with their own nuances:

**1. Removing and Replacing the Classification Layer:**

This method is the most straightforward. You literally chop off the last fully connected layer (and any associated activation functions or dropout layers directly after it) which output the number of classes the pre-trained model was originally trained for. Then, you add a new fully connected layer that outputs the number of classes you need, initialized with random weights. This new layer is then typically paired with a suitable activation function, like softmax for multi-class classification, or a sigmoid activation for multi-label classification.

The pre-trained part of the network is often 'frozen' or 'locked' for the first few epochs of training, meaning their weights are not updated. This prevents the large initial gradients from the newly added random layer from corrupting the pre-trained, well-honed feature representations. Once the new layer has settled into a somewhat sensible place, you can then start to unfreeze the pre-trained layers and fine-tune the entire model at a much lower learning rate.

Here’s an example using PyTorch, which i've found very accessible for these kinds of tasks:

```python
import torch
import torch.nn as nn
import torchvision.models as models

def adjust_model_classes(model_name, num_classes, pretrained=True):
  """
    Adjusts the final classification layer of a pre-trained model.

    Args:
      model_name (str): The name of the pre-trained model (e.g., 'resnet18').
      num_classes (int): The desired number of output classes.
      pretrained (bool): Whether to load pre-trained weights.

    Returns:
      torch.nn.Module: The modified model.
    """
  if model_name == 'resnet18':
      model = models.resnet18(pretrained=pretrained)
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs, num_classes)  # Replace the final layer
  elif model_name == 'vgg16':
      model = models.vgg16(pretrained=pretrained)
      num_ftrs = model.classifier[6].in_features
      model.classifier[6] = nn.Linear(num_ftrs, num_classes)
  else:
      raise ValueError(f"Model {model_name} not supported.")
  return model

# Example usage:
model = adjust_model_classes('resnet18', num_classes=10) # Adjust ResNet18 for 10 classes
# Then you would pass the 'model' instance to your training loop
print(model)
```

This snippet shows how, depending on the specific pre-trained model selected, the access of the linear layer can vary, and this should be carefully inspected when dealing with different pre-trained architectures.

**2. Adding an Adapter or Bottleneck Layer:**

Sometimes, directly replacing the final layer might not be optimal, especially if the original output size was significantly different from the new target. Adding an adapter or bottleneck layer before the final classification layer allows you to transition gradually from the high-dimensional features extracted by the pre-trained layers to the lower or higher number of classes you need. A bottleneck layer typically reduces the dimension of the feature space before outputting to your intended class size, which reduces the number of parameters for the new layer, making it easier to train and less susceptible to overfitting.

This strategy can often provide a performance boost, particularly on small datasets where you want to minimize the number of trainable parameters within the new layers. Here’s an example using a simple bottleneck layer:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class BottleneckModel(nn.Module):
    def __init__(self, model_name, num_classes, bottleneck_dim=512, pretrained=True):
        super(BottleneckModel, self).__init__()
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential() # Remove the original fc layer
        elif model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=pretrained)
            num_ftrs = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Sequential() #Remove the original fc layer
        else:
            raise ValueError(f"Model {model_name} not supported.")
        self.bottleneck = nn.Linear(num_ftrs, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
      x = self.base_model(x)
      x = self.relu(self.bottleneck(x))
      x = self.classifier(x)
      return x

# Example usage:
model_bottleneck = BottleneckModel('resnet18', num_classes=10, bottleneck_dim=256)
print(model_bottleneck)

```
This code shows how a 'BottleneckModel' class can be implemented, which involves passing the output of the base model to an extra linear layer, a relu, and then finally to the classification layer. This pattern can be extended to use more complex bottlenecks, like transformers or other kinds of dense layers.

**3. Using a Custom Layer or Head:**

For specialized tasks or when the output needs to be structured differently, you can construct a custom layer that sits on top of the pre-trained feature extractor. This could be a more complex, multi-layered network designed for a particular task. This is especially common when dealing with tasks beyond simple classification, such as object detection, segmentation, or image captioning. The key is ensuring that the input shape for your custom head matches the output shape of the feature extraction portion of the pre-trained model, and usually, freezing the feature extraction part is a good strategy at the beginning of training.

Here is an example of a custom classification layer:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class CustomClassificationHead(nn.Module):
  def __init__(self, input_dim, num_classes, hidden_dims=[1024, 512]):
    super(CustomClassificationHead, self).__init__()
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(prev_dim, hidden_dim))
      layers.append(nn.ReLU())
      prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, num_classes))
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
     return self.layers(x)

class CustomHeadModel(nn.Module):
  def __init__(self, model_name, num_classes, hidden_dims=[1024, 512], pretrained = True):
    super(CustomHeadModel, self).__init__()
    if model_name == 'resnet18':
        self.feature_extractor = models.resnet18(pretrained=pretrained)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential()
    elif model_name == 'vgg16':
        self.feature_extractor = models.vgg16(pretrained=pretrained)
        num_ftrs = self.feature_extractor.classifier[6].in_features
        self.feature_extractor.classifier[6] = nn.Sequential()
    else:
        raise ValueError(f"Model {model_name} not supported.")

    self.classifier = CustomClassificationHead(num_ftrs, num_classes, hidden_dims)


  def forward(self, x):
    features = self.feature_extractor(x)
    output = self.classifier(features)
    return output

model_custom_head = CustomHeadModel('resnet18', num_classes=10, hidden_dims=[2048, 1024])
print(model_custom_head)

```

In this case, we create a separate class ‘CustomClassificationHead’ which takes the number of features, number of classes and hidden dimensions as an input, and generates a multilayered neural net, which can then be used as the classification layer of the new model, which is defined in the class ‘CustomHeadModel’.

**Important Considerations:**

*   **Freezing Layers:** Start by freezing most of the pre-trained layers and train only the new classification layer. Gradually unfreeze and fine-tune the pre-trained layers with a low learning rate. This helps avoid catastrophic forgetting.

*   **Learning Rate:** Employ a lower learning rate for the pre-trained layers than the new layer to prevent overfitting.

*   **Dataset Size:** If you have a small dataset, the more parameters you introduce in your custom layer, the more chance you will be to overfit. In this situation, it can be a good idea to reduce the dimensions of the custom layer and use more regularisation methods such as dropout.
* **Choice of Pre-Trained Model:** The pre-trained model should be selected in a way that the dataset that you are training on is similar to the one the model was trained on. For example, using a classification model trained on imagenet to classify medical images could be not appropriate, unless extensive feature extraction and custom training is performed.

For deeper understanding, I’d highly recommend delving into the following:

*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** A comprehensive textbook that provides a solid theoretical foundation in deep learning concepts and practices.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** A practical guide with examples for implementing deep learning models with the most popular libraries in Python.
*   **Papers on transfer learning and fine-tuning techniques:** Many researchers publish their cutting-edge methods in places like *NeurIPS, ICML, ICLR*. Exploring those can be very beneficial.

In summary, adjusting the number of classes in a pre-trained model is a standard procedure, but it requires a delicate balance between re-purposing the knowledge captured in pre-trained layers, and fine-tuning to your custom task. The strategies above, when implemented thoughtfully, can effectively navigate this challenge.
