---
title: "How can ResNet-like CNNs be combined with LSTMs to process multiple images per data sample?"
date: "2025-01-30"
id: "how-can-resnet-like-cnns-be-combined-with-lstms"
---
The core challenge in processing sequential image data with convolutional neural networks (CNNs) and recurrent neural networks (RNNs), specifically LSTMs, lies in bridging the gap between the spatial feature extraction capabilities of CNNs and the temporal dependency modeling of LSTMs. I've encountered this exact issue several times while working on video understanding and time-lapse analysis projects. A naive approach of simply feeding each image from a sequence directly into an LSTM typically fails because the LSTM inputs are not structured to capture complex spatial relationships within each image, necessitating a pre-processing step.

Here's how I've successfully combined ResNet-like CNNs with LSTMs to process multi-image sequences, and the rationale behind it:

**1. Feature Extraction with CNNs:**

The first stage is leveraging a CNN, in this case a ResNet (or a ResNet variant) pre-trained on a large image dataset (such as ImageNet), to transform each image into a compact feature vector. The architecture of ResNets is particularly advantageous here due to its skip connections, mitigating the vanishing gradient problem and enabling training of very deep networks. This capacity is vital for complex feature extraction from individual images. The last convolutional layer of the ResNet model, just prior to the fully connected layer normally associated with classification, is used. Instead of the classification layer output, we extract the feature map at this stage, and often perform average pooling or global max pooling to collapse the spatial dimensions of the map into a single vector representation. Each image within the data sample goes through this processing individually. The resulting vector serves as the input to the LSTM. This ensures the LSTM operates on high-level feature representations rather than raw pixel data.

**2. Temporal Modeling with LSTMs:**

The second stage involves feeding the sequence of feature vectors (one vector per input image of the sequence) into an LSTM. LSTMs are well-suited for handling time-series data due to their internal memory cells that allow them to retain information about past inputs, making them capable of capturing dependencies within the image sequence. Each feature vector from the preceding CNN processing is treated as a time step in this recurrent process. The LSTM output then represents the processed information, reflecting both the extracted spatial features from each image, and their temporal relationships across the sequence. This final output can be utilized for various downstream tasks, including, but not limited to, classification, regression, or further processing.

**3. Code Example 1: Feature Extraction Function**

The following function illustrates how to extract features from a sequence of images using a pre-trained ResNet model:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

def extract_features(image_paths, resnet_model, device):
  """Extracts ResNet features from a list of images.
    Args:
      image_paths: list of paths to images.
      resnet_model: pre-trained ResNet model.
      device: device to run the model on.
    Returns:
      A torch tensor of shape (num_images, feature_dimension) containing the feature vectors.
  """
  resnet_model.eval() # Set model to evaluation mode.

  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  feature_vectors = []
  with torch.no_grad():
    for image_path in image_paths:
      image = Image.open(image_path).convert('RGB')
      image_tensor = preprocess(image).unsqueeze(0).to(device)
      features = resnet_model(image_tensor)
      features = torch.mean(features, dim=[2, 3]) # average pooling to get a vector
      feature_vectors.append(features.squeeze(0))

  return torch.stack(feature_vectors)

# Example Usage:
if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  resnet_model = models.resnet18(pretrained=True) # Load pretrained ResNet18.
  resnet_model = nn.Sequential(*list(resnet_model.children())[:-2]) # Remove final layers for feature extraction
  resnet_model = resnet_model.to(device)

  image_sequence = ['img1.jpg', 'img2.jpg', 'img3.jpg'] # Replace with your list of images.
  feature_sequence = extract_features(image_sequence, resnet_model, device)
  print("Extracted Feature Shape:", feature_sequence.shape) # Will be (3, 512)
```

The code demonstrates the procedure for loading a pre-trained ResNet, removing its classifier head, and applying pre-processing transforms.  It then loops through a list of image paths, extracts the feature map by passing the pre-processed image through the ResNet, performs average pooling over the spatial dimensions to get a single vector, and returns the sequence of resulting feature vectors, stacked into a single tensor.

**4. Code Example 2: LSTM Network**

The next code example shows how to set up an LSTM layer and pass the feature vectors extracted in the previous step through it:

```python
import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the LSTM sequence model."""
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # Fully connected layer after LSTM

    def forward(self, x):
        """Forward pass through the LSTM layer."""
        _, (h_n, _) = self.lstm(x) # h_n is output of lstm for each sequence
        out = self.fc(h_n[-1]) # Use the final hidden state for classification or other downstream
        return out

# Example Usage:
if __name__ == '__main__':
    input_size = 512 # Matches dimension of extracted feature vectors
    hidden_size = 128 #  Size of the hidden states of the LSTM.
    output_size = 2 # Number of output classes for, say, a classification task

    lstm_model = SequenceModel(input_size, hidden_size, output_size)

    # Assuming 'feature_sequence' is from the previous example and contains 3 image features.
    # Reshape so batch dimension is 1, then add to device
    feature_sequence = feature_sequence.unsqueeze(0) # Reshape to (1, 3, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_sequence = feature_sequence.to(device)
    lstm_model = lstm_model.to(device)

    output = lstm_model(feature_sequence)
    print("LSTM Output Shape:", output.shape) # Should be (1, 2)
```

This example presents a class that encapsulates the LSTM model and a final fully connected layer, which maps from the LSTM's hidden state to a desired output size.  The forward method takes the sequence of features as an input, processes it through the LSTM, and then passes the last hidden state to the final fully connected layer. The example shows how to pass the features extracted from the earlier example through the LSTM and how to utilize the resulting output for a downstream task.

**5. Code Example 3: End-to-End Training Example (Illustrative)**

While the details of an end-to-end training routine are specific to the task at hand, this example provides a basic structure using a hypothetical classification task:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assume feature extraction function and SequenceModel are from previous examples.

def train_model(resnet_model, lstm_model, dataloader, num_epochs, device):
    """
    Train the ResNet+LSTM model for a multi-image sequence processing task.
    """
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001) # Optimizer
    
    for epoch in range(num_epochs):
      for image_paths, labels in dataloader:
          feature_sequence = extract_features(image_paths, resnet_model, device)
          feature_sequence = feature_sequence.unsqueeze(0).to(device)
          labels = labels.to(device)
          
          optimizer.zero_grad() # Reset gradient
          outputs = lstm_model(feature_sequence)
          loss = criterion(outputs, labels) # Compute the loss
          loss.backward() # Compute gradients
          optimizer.step() # Optimize weights

      print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

#Example Usage:
if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  resnet_model = models.resnet18(pretrained=True) # Load pretrained ResNet18.
  resnet_model = nn.Sequential(*list(resnet_model.children())[:-2]) # Remove final layers for feature extraction
  resnet_model = resnet_model.to(device)

  input_size = 512 # Matches dimension of extracted feature vectors
  hidden_size = 128 # Size of the hidden states of the LSTM.
  output_size = 2 # Number of output classes for, say, a classification task
  lstm_model = SequenceModel(input_size, hidden_size, output_size).to(device)

  # Simulate Data Loading:
  # Create dummy image paths and corresponding labels.
  num_samples = 20
  num_images_per_sample = 3
  image_paths_list = [['img1.jpg', 'img2.jpg', 'img3.jpg'] for _ in range(num_samples)]
  labels = torch.randint(0, 2, (num_samples,))
  dataset = TensorDataset(image_paths_list, labels)
  dataloader = DataLoader(dataset, batch_size=1) # for simplicity, batch size 1.

  num_epochs = 5
  train_model(resnet_model, lstm_model, dataloader, num_epochs, device)
```

This code outlines the basic training loop where image paths and corresponding labels are loaded from the dataloader, the feature extraction function is used to generate a feature vector sequence from the images, the sequence is then passed to the LSTM module, which then outputs the classification result. This output is used to calculate the loss and perform back propagation, while the optimizer adjusts the weights of the LSTM module, with the CNN (ResNet) usually remaining fixed or with a very small learning rate.

**6. Further Resources**

For a deeper understanding of the components involved in this solution, I recommend exploring the following:

*   **Deep Learning with PyTorch:** This will provide a thorough introduction to building neural networks with the PyTorch framework, which was used in the code examples provided.
*   **Understanding Convolutional Neural Networks:** Researching materials on the architecture, theory, and capabilities of CNNs, especially ResNets, will help deepen understanding of the feature extraction stage.
*   **Long Short-Term Memory Networks:** A good grasp on LSTMs, including their internal mechanics and their use in sequence processing, will be invaluable in constructing the temporal processing component of the network.

This approach to combining CNNs and LSTMs allows one to leverage the strengths of both architectures, enabling the effective processing of sequential image data for a wide range of applications. Remember that the specifics of a successful model often depend on fine-tuning and task-specific architectural modifications.
