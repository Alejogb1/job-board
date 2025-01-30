---
title: "How do I visualize intermediate CNN activations?"
date: "2025-01-30"
id: "how-do-i-visualize-intermediate-cnn-activations"
---
Visualizing intermediate convolutional neural network (CNN) activations provides critical insight into how these models learn and process information, far beyond just observing final classification results. My experience debugging complex image classification models has repeatedly shown that these visualizations are invaluable for diagnosing issues like vanishing gradients, feature map saturation, and inappropriate filter responses. These internal representations reveal the specific image features each layer is attending to, offering a window into the model’s decision-making process.

The fundamental process involves extracting the output of a selected layer after feeding an input image through the network. These activations are typically multi-dimensional arrays (tensors), where the depth dimension represents the feature maps learned by the convolution filters. To visualize them effectively, I generally reduce the dimensionality of the activations for each filter and then display each filter map as a greyscale or heatmap image. The depth of the layer usually determines the complexity of the features captured, with earlier layers capturing lower-level features like edges and corners, and later layers responding to increasingly complex, abstract patterns.

Here's a breakdown of the methodology, followed by some code examples:

1. **Model Definition and Layer Selection:** First, you need to have a trained CNN model. You must then select the specific layer whose activations you want to inspect. This could be any convolutional layer within your network. It’s common to examine both early and later layers to observe the hierarchy of feature extraction.
2. **Input Preparation:** Choose an input image that represents a good test case for your model. The image needs to be preprocessed in the same way the model expects its input – typically resizing, normalization, and potentially conversion to a tensor format.
3. **Forward Pass:** Using a framework like PyTorch or TensorFlow, you perform a forward pass through the network up to the selected layer. You instruct the framework to capture the output of this specific layer. This results in a tensor of intermediate activations.
4. **Activation Visualization:** The activations tensor contains feature maps. You’ll usually visualize each of these maps independently. To accomplish this, I use the following general approach:
    *   **Dimensionality Reduction:** Each feature map from a given convolution layer is typically a 2D array with spatial dimensions and a number of channels. It’s often most useful to extract each individual channel's 2D spatial activation map.
    *   **Visualization:** The extracted 2D maps are then visualized as greyscale images or heatmaps. In the latter, higher activation values can correspond to warmer colors.

The following examples will use PyTorch to demonstrate this process.

**Code Example 1: Basic Feature Map Visualization**

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval() # Set to evaluation mode

# Define the specific layer to visualize (e.g., the first conv layer)
target_layer = model.conv1

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an example image (replace with your own)
image_path = 'example.jpg'
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

# Hook to capture the activations
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach())

hook = target_layer.register_forward_hook(hook_fn)

# Perform a forward pass
with torch.no_grad():
    model(input_tensor)

hook.remove() # Remove the hook

# Visualize the first few feature maps
feature_maps = activations[0].squeeze()
num_maps_to_visualize = min(feature_maps.shape[0], 9)  #Limit to 9 for display
plt.figure(figsize=(10, 10))
for i in range(num_maps_to_visualize):
    plt.subplot(3, 3, i+1)
    plt.imshow(feature_maps[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**Commentary:**
This example demonstrates the core concepts. We use a pre-trained ResNet18 model and select the first convolutional layer (`model.conv1`). The image is loaded, preprocessed, and fed into the network. A forward hook is used to capture the layer's output. Finally, the first few feature maps are displayed as greyscale images. The hook mechanism allows you to intercept and store the intermediate outputs. You’ll see these maps tend to highlight basic patterns like edges and color variations in the image.

**Code Example 2: Visualizing a Deeper Layer**

```python
# Use the same setup from above (model, preprocess, etc.)

target_layer = model.layer2[1].conv2  # Access a deeper convolutional layer

# Hook to capture the activations
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach())

hook = target_layer.register_forward_hook(hook_fn)


# Perform a forward pass using same image
with torch.no_grad():
    model(input_tensor)

hook.remove()

# Visualize the first few feature maps
feature_maps = activations[0].squeeze()
num_maps_to_visualize = min(feature_maps.shape[0], 9)
plt.figure(figsize=(10, 10))
for i in range(num_maps_to_visualize):
    plt.subplot(3, 3, i+1)
    plt.imshow(feature_maps[i], cmap='viridis') #Changed to viridis
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**Commentary:**

This example demonstrates the visualization for a deeper layer. I’ve targeted the second convolution layer (`conv2`) of the second block in `model.layer2`. You'll notice that these feature maps are visually more complex and tend to highlight more abstract features compared to the first example, reflecting the model's hierarchical learning process. Also, I changed the colormap to 'viridis', which can sometimes reveal better details than greyscale. This flexibility to try different colormaps is useful during exploration.

**Code Example 3: Aggregated Feature Map Visualization**

```python
# Use the same setup from above (model, preprocess, etc.)

target_layer = model.layer3[0].conv1 # Another layer
# Hook to capture the activations
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach())
hook = target_layer.register_forward_hook(hook_fn)


# Perform a forward pass using same image
with torch.no_grad():
    model(input_tensor)
hook.remove()

# Visualize the average activation map
feature_maps = activations[0].squeeze()
average_map = torch.mean(feature_maps, dim=0)

plt.figure(figsize=(5, 5))
plt.imshow(average_map, cmap='plasma')
plt.axis('off')
plt.title("Average Activation Map")
plt.tight_layout()
plt.show()
```

**Commentary:**

This example demonstrates aggregating the feature maps by calculating the average activation map across all filters in a given layer. This method provides a generalized view of what that layer is most strongly responding to. Using the average helps reduce noise in the visualization, especially for layers with a large number of feature maps. Here, the `plasma` colormap is chosen.

**Resource Recommendations:**

For further study, I suggest exploring materials related to:

*   **Convolutional Neural Networks:** Understanding the architecture and training of CNNs is crucial. Look for resources on topics such as convolutional layers, pooling, and backpropagation.
*   **PyTorch and TensorFlow documentation:** Comprehensive documentation provided by the frameworks is the best source of information on creating, training, and extracting activations from models. Focus on the specifics of layer hooks.
*   **Feature visualization techniques:** Beyond basic activation maps, learning about methods like saliency maps, CAM (Class Activation Mapping), and deconvolution networks will significantly improve your understanding.
*   **Image processing and computer vision:** A strong grasp of fundamental concepts in image processing can give context to the features detected by each convolutional layer.

Visualizing intermediate CNN activations is a powerful method I consistently employ to diagnose and improve models. The key takeaways are the ability to use forward hooks to extract intermediate representations, the conversion of multi-dimensional feature maps into visualizable 2D images, and the systematic exploration of different layers to gain a holistic understanding of model behavior.
