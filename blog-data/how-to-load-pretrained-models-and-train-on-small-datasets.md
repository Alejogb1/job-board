---
title: "How to load pretrained models and train on small datasets?"
date: "2024-12-16"
id: "how-to-load-pretrained-models-and-train-on-small-datasets"
---

Right then, let’s talk about leveraging pre-trained models when your data resources are… less than abundant. I’ve bumped into this particular challenge more times than I care to count, and it's a common hurdle, especially when venturing beyond established datasets or tackling niche problems. The crux of the matter is, training deep learning models from scratch requires a substantial amount of labeled data, and that’s often not a luxury we have. The pre-trained model comes to the rescue, offering a powerful starting point. But it's not as simple as just plugging it in and hoping for the best. We need to strategize.

The core idea is transfer learning, where the knowledge gained by a model on a massive dataset (like ImageNet for images or a corpus of text for language tasks) is transferred to a new, typically smaller, dataset. This approach allows us to benefit from the learned features, reducing the amount of data and computational resources needed for our target task. However, the execution details matter. There are different techniques for doing this effectively.

Firstly, we can treat the pre-trained model as a feature extractor. In this case, we freeze the weights of the pre-trained layers, and append new layers to handle our specific task. Then, we train only the new layers using our smaller dataset. The intuition here is that the initial layers of a deep network often learn general features useful for a variety of tasks, while the final layers learn features specific to the data they were trained on.

Here's a basic python example using pytorch to illustrate:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

def train_feature_extractor(num_classes, learning_rate=0.001, num_epochs=10):
    # load a pre-trained resnet18 model
    resnet = models.resnet18(pretrained=True)

    # freeze all the model's parameters
    for param in resnet.parameters():
        param.requires_grad = False

    # replace the last layer to match our classification problem
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)


    # define loss function and optimizer (only for the trainable last layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=learning_rate)

    # simulate a basic dataloader for demonstration
    # (in practice, use a dataloader class to handle the real data)
    def data_generator(batch_size=32):
        for _ in range(100): #arbitrary 100 batches
            inputs = torch.randn(batch_size, 3, 224, 224) # random inputs in correct shape
            labels = torch.randint(0, num_classes, (batch_size,)) # random labels
            yield inputs, labels

    # Training Loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_generator()):
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/100], Loss: {loss.item():.4f}')
    return resnet
if __name__ == '__main__':
    trained_model = train_feature_extractor(num_classes=5)
    print("Training finished, model ready for inference.")
```
In this snippet, we load the pre-trained ResNet18, freeze its layers, and replace its final fully connected layer with a new one suitable for our number of classes. The training process then only updates weights in this final layer, allowing us to leverage the feature extraction abilities of the pre-trained model while learning to classify our data.

Secondly, we can perform fine-tuning. Here we unfreeze some, or all, of the layers of the pre-trained model. This allows our target data to further adjust the weights of the model. This can yield higher accuracy than feature extraction, but it also has risks. If the target data is very different from the data the model was pre-trained on, or if the target dataset is very small, we can overfit and lose the pre-trained model's good performance. We should normally use a lower learning rate when doing fine-tuning compared to training from scratch to avoid sudden or dramatic changes in the model's parameters.

Here’s a modified version of the previous example, this time fine-tuning the later layers of resnet:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

def fine_tune_model(num_classes, learning_rate=0.0001, num_epochs=10):
    # load a pre-trained resnet18 model
    resnet = models.resnet18(pretrained=True)

    # unfreeze the last few layers for fine-tuning
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.fc.parameters(): # include the fc layer
        param.requires_grad = True

    # replace the last layer to match our classification problem
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)


    # define loss function and optimizer (now for the trainable layers)
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, resnet.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    # simulate a basic dataloader for demonstration
    def data_generator(batch_size=32):
        for _ in range(100):
            inputs = torch.randn(batch_size, 3, 224, 224)
            labels = torch.randint(0, num_classes, (batch_size,))
            yield inputs, labels
    # Training Loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_generator()):
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/100], Loss: {loss.item():.4f}')
    return resnet

if __name__ == '__main__':
    trained_model = fine_tune_model(num_classes=5)
    print("Fine-tuning finished, model ready for inference.")
```

Here, instead of freezing the entire model, we selectively allow the later layers (layer4, the final convolutional block, and the fully connected layer) to be trained, while keeping the earlier layers fixed. The learning rate has been lowered to facilitate a more gradual adjustment of parameters in the fine-tuning process. The optimizer now only considers the trainable parameters, which are those that have `requires_grad=True`.

Finally, a third approach that I've found particularly useful when dealing with very small datasets is data augmentation. While not directly related to the model itself, it artificially expands our training data by applying transformations like rotations, flips, zooms, and shifts to our existing training images. This, in effect, can dramatically increase the diversity of our training data without needing more real-world examples.

Here's a quick example of how you might incorporate data augmentation using pytorch and torchvision:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image # for demonstration purpose

def augment_and_train(num_classes, learning_rate=0.0001, num_epochs=10):
    # load a pre-trained resnet18 model
    resnet = models.resnet18(pretrained=True)

    # freeze all the model's parameters except the last layer
    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)

    # define data transformations including augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224), # random cropping and resizing
        transforms.RandomHorizontalFlip(), # random horizontal flipping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalization
    ])

    # define loss function and optimizer (only for the last layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=learning_rate)

    # simulate a basic dataloader with augmentation
    def data_generator(batch_size=32, data_size=100): # only data_size actual inputs
        for _ in range(data_size // batch_size):
            images = []
            for _ in range(batch_size):
                # load dummy image as example, replace with real image loading
                dummy_image = Image.new('RGB', (256, 256), color = (255, 255, 255)) # white image
                images.append(transform(dummy_image))
            inputs = torch.stack(images)
            labels = torch.randint(0, num_classes, (batch_size,))
            yield inputs, labels
    # Training Loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_generator(data_size=3200)): # larger data size
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/100], Loss: {loss.item():.4f}')

    return resnet

if __name__ == '__main__':
    trained_model = augment_and_train(num_classes=5)
    print("Training finished, model ready for inference.")

```

Here, we add random resizing and cropping and random horizontal flips during the preprocessing phase. The key here is that data augmentation introduces variations to training samples, thereby reducing the overfitting that can happen with smaller datasets.

When it comes to resources, I highly recommend diving into the classic text "Deep Learning" by Goodfellow, Bengio, and Courville. That's a comprehensive overview of everything deep learning. Specifically for transfer learning, papers like "How transferable are features in deep neural networks?" by Yosinski et al. provide key insights. For more practical, hands-on guidance on pytorch, I often refer back to the official pytorch documentation and tutorials.

In practice, combining these techniques -- using a pre-trained model as a feature extractor, then trying fine-tuning some of the later layers, and simultaneously applying data augmentation -- generally delivers the best results for small datasets. Remember to monitor for overfitting and adjust accordingly – early stopping, adding dropout regularization, and carefully tuning hyperparameters are all very worthwhile adjustments to fine-tune the performance of your models. Start with the simpler feature extraction approach before moving to more complex fine-tuning and always keep an eye on the metrics. It’s about finding the right balance for your specific case.
