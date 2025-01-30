---
title: "How can I train a YOLOv5 model on a dataset with 100 classes?"
date: "2025-01-30"
id: "how-can-i-train-a-yolov5-model-on"
---
Training a YOLOv5 model on a dataset encompassing 100 distinct classes presents a significant challenge primarily due to the increased model complexity and computational demands associated with a higher number of categories. I've tackled similar projects requiring precise object detection across dozens of classes, and the key lies in strategic configuration and resource management to prevent overfitting and ensure efficient training.

Firstly, the primary adjustment compared to a lower-class dataset is the modification of the model’s output layer, a crucial area. In YOLOv5, the final convolutional layer’s output channels directly correspond to the number of classes plus bounding box parameters (4) and an objectness score. For 100 classes, this means your final convolution will output 105 channels, per anchor and feature grid. Therefore, the model configuration file, typically `yolov5s.yaml`, `yolov5m.yaml`, etc., needs modification to reflect this change. Ignoring this can lead to training failure as the model tries to process an incompatible output shape.

Beyond the model architecture change, data augmentation becomes increasingly important. With 100 classes, the likelihood of encountering imbalanced classes and intra-class variance is high. Employing a wider range of augmentations such as random scaling, rotation, mosaic augmentations, and color jittering can make the model more robust and less prone to memorizing specific instances. Conversely, excessive augmentation can obscure crucial features, so a careful, iterative adjustment is needed.

Also, hyperparameter tuning takes on heightened importance. The learning rate, batch size, and weight decay parameters, which often work well for smaller class counts, may need significant adjustment for 100 classes. Specifically, increasing the batch size may improve gradient stability with the added model complexity; however, this requires careful attention to available GPU memory. Similarly, the learning rate may require a more gradual decay schedule to allow convergence to an optimal solution within the parameter space.

Here's how this translates into practical application, incorporating modifications to the model configuration file and training script based on my experience:

**Code Example 1: Modifying the Model Configuration File**

I will illustrate changes to the `yolov5s.yaml` configuration file. In my experience, I’ve often copied the default configuration and created custom config to allow for parallel experiments. The crucial changes are at the output layer within the `model` section, specifically in the `head` block:

```yaml
# Parameters
nc: 100  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # model width multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
  [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
  [-1, 3, C3, [128]],
  [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
  [-1, 6, C3, [256]],
  [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
  [-1, 9, C3, [512]],
  [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
  [-1, 3, C3, [1024]],
  [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  [[-1, 6], 1, Concat, [1]],  # cat backbone P4
  [-1, 3, C3, [512, False]],  # 13
  [-1, 1, Conv, [256, 1, 1]],
  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  [[-1, 4], 1, Concat, [1]],  # cat backbone P3
  [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
  [-1, 1, Conv, [256, 3, 1]],
  [[-1, 16], 1, Concat, [1]],  # cat head P4
  [-1, 3, C3, [256, False]],  # 20 (P4/16-medium)
  [-1, 1, Conv, [512, 3, 1]],
  [[-1, 13], 1, Concat, [1]],  # cat head P5
  [-1, 3, C3, [512, False]],  # 23 (P5/32-large)
  [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```
**Commentary:**

The key modification is `nc: 100`. This single line instructs the model to expect 100 distinct object classes in the dataset. The final `Detect` layer uses this parameter to output the correct number of channels for each anchor scale and feature grid. Without this change, the model would be configured for the default number of classes, usually 80.

**Code Example 2: Adjusting Hyperparameters in the Training Script**

While specific values will vary depending on your hardware and dataset, demonstrating adjustments are beneficial. Typically, I modify `train.py` directly or create a custom training script. Here's a snippet focusing on hyperparameter changes:

```python
import torch
import torch.optim as optim
from yolov5.utils.torch_utils import select_device
from yolov5.models.yolo import Model
from yolov5.utils.general import create_dataloader
from yolov5.utils.loss import ComputeLoss
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(opt, model, data, device):
    #Initialize SummaryWriter
    writer = SummaryWriter(log_dir='runs/train')

    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=opt.lr0, weight_decay=opt.weight_decay)
    criterion = ComputeLoss(model)
    
    #DataLoader initialization
    train_loader, val_loader = data
    
    epochs = opt.epochs
    
    for epoch in range(epochs):
      model.train()
      progress_bar = tqdm(train_loader, desc = f'Epoch {epoch + 1}/{epochs}')
      for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss, loss_components = criterion(predictions, targets)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()
        
        #Logging
        writer.add_scalar('Train/total_loss', loss.item(), epoch * len(train_loader) + progress_bar.n)
        for i, component in enumerate(criterion.names):
            writer.add_scalar(f'Train/{component}',loss_components[i], epoch * len(train_loader) + progress_bar.n)

      
      #Validation
      model.eval()
      with torch.no_grad():
        total_val_loss = 0
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)

            val_predictions = model(val_images)
            val_loss, _ = criterion(val_predictions, val_targets)
            total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss/len(val_loader)
        writer.add_scalar('Validation/total_loss', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")
    writer.close()

def main(opt):
    # Initialize device
    device = select_device(opt.device)
    
    #Load the model from the modified yaml file
    with open(opt.model_config) as f:
        model_config = yaml.safe_load(f)
        model = Model(model_config, ch = 3, nc = model_config['nc']).to(device)

    #Load Data
    train_loader, val_loader = create_dataloader(
            path=opt.dataset_path,
            img_size = opt.img_size,
            batch_size = opt.batch_size
            )
    data = (train_loader, val_loader)
    
    train(opt, model, data, device)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str, default='yolov5s.yaml', help='model configuration file path')
  parser.add_argument('--dataset_path', type = str, default='dataset.yaml', help='path to dataset yaml')
  parser.add_argument('--img_size', type=int, default=640, help='image size')
  parser.add_argument('--batch_size', type = int, default=32, help='batch size')
  parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
  parser.add_argument('--lr0', type = float, default = 0.001, help = 'initial learning rate')
  parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--device', default='cuda:0', help='device')
  opt = parser.parse_args()
  main(opt)
```
**Commentary:**

This is an abbreviated training script designed to highlight important hyperparameter changes. I’ve adjusted `lr0` (initial learning rate) to 0.001, a starting point which I often adjust from. The `weight_decay` parameter (L2 regularization) is set to 0.0005. This combination tends to work for me for moderately complex datasets initially. The batch_size argument allows for dynamic adjustment to fit hardware constraints and dataset nuances. I’ve included the command line argument to select a device and to parse from a custom dataset. Note the incorporation of tensorboard for tracking training progress and losses. These hyperparameters frequently require iterative adjustment based on training metrics such as the validation loss curve and precision recall curves.

**Code Example 3: Incorporating Data Augmentations**

Augmentation is typically configured within the dataset loader functionality in the `utils/datasets.py` file or by directly modifying the `create_dataloader` function, but I will illustrate a standalone augmentation pipeline:

```python
import torch
from torchvision import transforms
from PIL import Image
import random

class AugmentationPipeline:
    def __init__(self, img_size):
        self.img_size = img_size
        self.augmentations = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # Crop and resize
            transforms.RandomHorizontalFlip(p=0.5), # Horizontal flip
            transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #Color Jitter
            transforms.ToTensor(), # Convert to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization
        ])


    def apply(self, image):
        """
        Applies augmentation pipeline to a single image.
        Args:
            image (PIL.Image): image to augment
        Returns:
            torch.Tensor: augmented image
        """
        augmented_image = self.augmentations(image)
        return augmented_image
    
if __name__ == '__main__':
    #Sample Usage
    img = Image.open('test.jpg').convert('RGB')  #Load an example image
    img_size = 640
    
    aug_pipeline = AugmentationPipeline(img_size)
    
    num_augmentations = 5
    for i in range(num_augmentations):
      augmented_image = aug_pipeline.apply(img)
      print(f"Augmented image {i+1} shape: {augmented_image.shape}")
      #If you want to display the images, you can convert the augmented tensors to an image again
```

**Commentary:**

This is a simplified, standalone data augmentation pipeline class. I include common transformations, such as RandomResizedCrop, HorizontalFlip, Rotation, and ColorJitter, which I have found effective in my experience. This class would typically integrate within a custom `Dataset` object or function. The critical aspect is that data is randomly transformed to mitigate overfitting; I avoid extreme transforms as they can degrade the signal. This pipeline transforms PIL images, standard for computer vision, to PyTorch tensors.

For additional learning about training models, I suggest exploring resources such as the official PyTorch documentation, which covers the fundamentals of deep learning, and various tutorials that focus on implementing object detection workflows. Many repositories on platforms such as GitHub offer implementations of object detection models and can serve as a basis for further learning. Also, papers published by researchers describing novel implementations often detail theoretical principles and practical applications which can expand your understanding.
