---
title: "How can CNNs be used for segmenting portrait images in Python?"
date: "2025-01-30"
id: "how-can-cnns-be-used-for-segmenting-portrait"
---
Convolutional Neural Networks (CNNs), typically associated with image classification, can be powerfully adapted for pixel-level semantic segmentation, including the specific task of segmenting portrait images. My experience, specifically with a project aimed at generating portrait-based digital avatars, demonstrated that CNNs can learn to distinguish between the human figure and the background with remarkable precision, given sufficient training data and appropriate network architecture. The core principle here is shifting from assigning a single label to an entire image to assigning a label to every pixel, effectively creating a mask that delineates the portrait from its surroundings.

The architecture of CNNs for segmentation differs from classification models. While classification networks often compress information through pooling layers towards a single output label, segmentation networks need to retain spatial information, mapping each pixel of the input image to a corresponding pixel in the output segmentation mask. This is achieved through a combination of encoder and decoder structures. The encoder, typically composed of convolutional and pooling layers, extracts features from the input image, progressively reducing its spatial resolution but increasing the number of channels (feature maps). The decoder, conversely, upsamples the feature maps, gradually recovering spatial resolution while maintaining the context acquired in the encoder. This upsampling often employs transposed convolutions or interpolation techniques, which are crucial for precise pixel-wise segmentation.

To effectively train a CNN for portrait segmentation, a suitable dataset of portrait images with corresponding segmentation masks is required. These masks are binary images where pixel values indicate the class label (foreground or background). Data augmentation techniques, such as rotation, scaling, and flipping, can significantly improve the model’s generalization capabilities by exposing it to variations in portrait pose, lighting, and background. Furthermore, the choice of loss function also impacts model performance. For binary segmentation, binary cross-entropy is a common choice, although variations like Dice loss might be preferred for imbalanced datasets where the foreground (portrait) occupies a smaller portion of the image.

Now, let's examine several code examples using Python with PyTorch. I will demonstrate how to adapt a relatively simple U-Net architecture for segmentation. U-Net, known for its skip connections, effectively merges context information from the encoder to the decoder, which improves fine-grained details in the generated segmentation mask.

**Code Example 1: Defining the U-Net Architecture**

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Crop x2 to match the dimensions of x1 after upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = x2[:, :, diffY // 2:x2.size()[2] - diffY // 2, diffX // 2:x2.size()[3] - diffX // 2]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
```

This code defines the U-Net architecture using PyTorch. `DoubleConv` represents two convolutional layers with ReLU activations. `Down` encapsulates a max pooling layer followed by a `DoubleConv`, while `Up` incorporates an upsampling and concatenates the features with skip connections from the encoder, ultimately outputting a `DoubleConv`. The `UNet` class creates the full network, specifying the number of input channels and classes.  The `forward` method defines the data flow through the U-Net. The cropping in the `Up` class is essential to address the size differences after upsampling, ensuring the encoder feature maps are aligned correctly with the decoder maps.

**Code Example 2: Training Loop (Simplified)**

```python
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assume train_images and train_masks are numpy arrays of shape (num_images, height, width, channels) and (num_images, height, width, 1) respectively
# Convert to torch tensors
train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float)
train_masks_tensor = torch.tensor(train_masks.transpose(0, 3, 1, 2), dtype=torch.float)


train_dataset = TensorDataset(train_images_tensor, train_masks_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = UNet(n_channels=3, n_classes=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 10
for epoch in range(num_epochs):
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```
This simplified code snippet demonstrates the basic training process. The input data, `train_images` and `train_masks`, are assumed to be numpy arrays and are converted into PyTorch tensors before being loaded into a `DataLoader`. The optimizer is an Adam optimizer, and the loss function is `BCEWithLogitsLoss` which combines sigmoid activation and binary cross-entropy loss for better numerical stability. The `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` calls perform a single gradient descent update for each batch. While basic, this illustrates the general pattern of training a segmentation model using a loss function to guide its learning.

**Code Example 3: Segmentation Inference**
```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

def predict_mask(image_path, model, device):

  image = Image.open(image_path).convert("RGB")
  transform = transforms.Compose([
      transforms.Resize((256,256)), # Ensure images have same dimensions during inference
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension


  with torch.no_grad(): # Disable gradient calculation during inference
      output_tensor = model(input_tensor)

  mask_prediction = torch.sigmoid(output_tensor).cpu().numpy() # Apply sigmoid to get probabilities
  mask_prediction = (mask_prediction > 0.5).astype(np.uint8).squeeze() # Threshold probabilities
  
  return mask_prediction

# Assuming a trained model and the path to an image
if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = UNet(n_channels=3, n_classes=1).to(device)

  #load the trained model, replace "your_model.pth" with your saved model
  model.load_state_dict(torch.load("your_model.pth", map_location=device))
  model.eval() # set the model to evaluation mode


  image_path = "your_image.jpg"
  segmented_mask = predict_mask(image_path,model,device)

  # Display or save the resulting mask, which is a numpy array representing the segmentation
  # Here, the result is converted to PIL image before saving to disk.
  seg_image = Image.fromarray(segmented_mask*255)
  seg_image.save("segmented_image.png")
  print ("Segmentation output saved to segmented_image.png")
```

This code demonstrates how to perform inference on a previously trained model. The `predict_mask` function applies necessary transforms to the input image, feeds it through the model, and obtains the segmentation mask. It's crucial to apply a sigmoid function to convert the raw output into probabilities, then threshold these probabilities to get a binary mask. The `with torch.no_grad()` context manager prevents PyTorch from calculating gradients during inference, improving efficiency. This resulting mask, represented by a NumPy array, can then be further processed or visualized.  The `model.eval()` call sets the model to evaluation mode, which can change the behaviour of some layers (e.g., batch norm or dropout).

For further study, I recommend delving into advanced segmentation architectures like DeepLab and Mask R-CNN. The implementations of these are typically more complex but offer further improvements in performance. Exploring techniques for handling imbalanced data, such as class-weighted loss functions, is crucial for real-world portrait segmentation where the portrait occupies a relatively small area. Furthermore, investigating different data augmentation methods and loss functions would also prove beneficial. Finally, the torchvision library in PyTorch provides pre-trained models, datasets, and transforms that can be very helpful when working with images.
