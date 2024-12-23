---
title: "How can I use two images as a training dataset in PyTorch?"
date: "2024-12-23"
id: "how-can-i-use-two-images-as-a-training-dataset-in-pytorch"
---

Okay, let's tackle this. I've certainly seen my share of, shall we say, 'creative' attempts to wrangle image data into usable training sets. You're aiming to use *two* images for training in pytorch, which on the face of it, seems…limited, but it's actually a great starting point to understand fundamental concepts. It’s the sort of thing I encountered early on when experimenting with generative models, or trying to debug some odd data loader pipeline. What you are really asking is how to represent and handle this data, and how to feed that into your model.

The core issue here isn't that you have two images, it's that you have a very small dataset. Usually, deep learning thrives on large, varied datasets to learn robust features and avoid overfitting. Two images won't let you train a robust image classifier or detector, but they're perfect for learning how to build a pytorch dataset and data loader correctly. Essentially, what we need to do is create a custom dataset class, and then use a data loader to iterate over it. I'm going to walk through how I would handle this situation, including a basic example.

First, let's discuss the dataset class. Pytorch’s dataset class, which you extend from `torch.utils.data.Dataset`, essentially serves as an interface to retrieve your data. It needs to define two core methods: `__len__` which returns the total number of items in the dataset, and `__getitem__(idx)` which returns a specific item at the given index `idx`. This allows pytorch to access your data systematically. When dealing with image data, the `PIL` library is your friend for image loading and basic manipulation. It can handle various image formats without issues. Also, the torchvision library can provide a few transformations.

So, here's the code snippet for constructing a custom pytorch dataset for this two-image scenario:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

class TwoImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') #ensure RGB to be consistent
        if self.transform:
            image = self.transform(image)
        # if labels are relevant for your task, you may want to return them too
        return image, 0 # example. 0 can be your class, in this case, every image can be a zero class


#Example usage
if __name__ == '__main__':
    #ensure you have these files in the script directory.
    img1_path = 'image1.png' #example paths
    img2_path = 'image2.jpg'

    #create dummy image files if they don't exist.
    if not os.path.exists(img1_path):
        Image.new('RGB', (100, 100), color = 'red').save(img1_path)
    if not os.path.exists(img2_path):
        Image.new('RGB', (100, 100), color = 'blue').save(img2_path)

    image_paths = [img1_path, img2_path]

    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TwoImageDataset(image_paths, transform=transform)
    print(f"Number of data items: {len(dataset)}")
    # retrieve one image for demonstration purposes
    first_image, first_label = dataset[0]

    print(f"Shape of the first image tensor: {first_image.shape}")
    print(f"Label of the first image: {first_label}")


```

In this code, `TwoImageDataset` loads each image using `PIL`, applies specified transforms (resizing, converting to tensor, normalizing) if any, and returns the transformed image as a tensor along with a label, here simply zero for demonstration purposes. This means, that each time you retrieve an item with `dataset[index]`, it is preprocessed via transform function. If you were training a classification network, then you’d need to adapt the `__getitem__` function to return appropriate labels for each image. Also note that I've added a dummy image generation routine that creates a red square for `image1.png` and a blue square for `image2.jpg` if those images don't already exist. This makes it easy for you to try running this code.

Now, we still have to put the images to use. A pytorch data loader is responsible for efficiently batching data from the dataset and optionally shuffling it. To actually load your images, use it like this:

```python
from torch.utils.data import DataLoader

#assuming you've created dataset using the code above
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Shape of image batch: {images.shape}")
    print(f"Labels of batch : {labels}")

```

In this example, we set `batch_size` to 1 because, again, with only two images, we do not need more. You can set shuffle to true during training for larger datasets, but this makes no sense for two images. The dataloader iterates through the dataset, yielding batches of images which will be of shape `(batch_size, channels, height, width)`. With only two images, I typically won't be using a dataloader with shuffle. It’s mostly relevant when your dataset is large and there is some order to the data within the dataset. It is helpful to start with no shuffling to check that the data and labels match what you expect.

Finally, here is an example of how to use a basic network with this data:

```python
import torch.nn as nn
import torch.optim as optim

# Assuming you have a dataset and a dataloader defined as in previous examples

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 128 * 128, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Instantiate the model, loss, and optimizer
model = SimpleCNN()
criterion = nn.MSELoss() #or nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop

num_epochs = 5

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        #zero gradient buffer
        optimizer.zero_grad()
        outputs = model(images)
        # labels need to be converted to appropriate shape
        loss = criterion(outputs, labels.float().unsqueeze(1))  #make sure label shape matches the output shape
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, loss = {loss.item()}")
print("Training completed.")

```

Here, `SimpleCNN` takes your input images, processes them through convolutional and fully connected layers, and outputs the predicted value. `MSELoss()` is an example of a loss function you might use if your task is regression type of problem, and you can substitute that with `BCELoss()` if it's binary classification. Be aware of shapes of the labels you feed into the loss function, making sure it matches the shape of the output. The loop simply iterates through the data, does forward and backward pass, and updates the network's parameters. It's a basic training loop and is meant to show the integration of a dataloader with a network.

For deeper understanding of data loading pipelines, I recommend examining chapters on data loading and manipulation in books like "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann, which provide detailed practical advice. Additionally, the documentation for `torch.utils.data` and `torchvision.transforms` is invaluable for understanding data processing. These resources helped me immensely early on in my learning journey and continue to be very helpful. Also, the original papers that introduced PyTorch data loading are worth looking for, if you are interested in how it was designed.

Remember, deep learning training on two images is a starting point, a way to comprehend the mechanism. To build meaningful systems, you need large varied datasets, but hopefully, this gives you a clear understanding of how you can set up your data using Pytorch.
