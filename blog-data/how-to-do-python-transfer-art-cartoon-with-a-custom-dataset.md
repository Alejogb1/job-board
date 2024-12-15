---
title: "How to do python transfer art cartoon with a custom dataset?"
date: "2024-12-15"
id: "how-to-do-python-transfer-art-cartoon-with-a-custom-dataset"
---

alright, so you're aiming to create this python-powered art transfer thing, specifically cartoon-style, using your own dataset. that's a cool project. i've spent more than a few late nights wrestling with similar image manipulation stuff, so let me share my take on it, and some of the potholes i’ve stumbled into so you can avoid them.

first, breaking down the challenge. we’re essentially talking about style transfer here. there are a bunch of approaches for that, but for cartoon-ish outcomes, and with a custom dataset, we need to focus on techniques that can learn and adapt well from your specific data. generic pre-trained models might give generic results. we want that unique flair.

so, my go-to setup for this has been a combination of convolutional neural networks (cnns) and some clever loss functions. don’t freak out, i'll keep it practical. let’s approach this in stages: data prep, model selection, and then training.

**data, data, and more data.**

this is a big one. your results are going to be as good as your data. it’s not just about quantity; it's about quality. your custom dataset should ideally contain images that represent the cartoon style you want to replicate. think about the variations: different characters, different backgrounds, different levels of detail, maybe even some different colour palettes. the more diverse and well-labelled, the better the model will generalise.

i remember once trying to train a style transfer model on a set of poorly lit cartoon screengrabs. it was a disaster. the model was trying to compensate for the bad lighting rather than learning the art style. lessons were learned. so, clean, well-exposed, and varied data will save you hours of debugging. think of it like this: a model is like a kid. if you feed it only junk, it will only make junk. the model will never be the mona lisa if it trains with garbage data.

now that you have your data. resize them to some reasonable size. i’ve found that something like 256x256 works well for most cases. also, normalize the pixel values to be between 0 and 1. this helps the training process converge faster. here is a bit of python code for this pre-processing that you should run before the model can see it. it will turn all your images into tensors.

```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomCartoonDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image

#usage example
if __name__ == '__main__':
    image_folder = 'path/to/your/cartoon_images'
    dataset = CustomCartoonDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch.shape) # expected output: torch.Size([32, 3, 256, 256])
        break

```

**model architecture**

for the actual neural network, i’ve had a good amount of success using a cnn based architecture. you could use something simple like vgg16 without the final classification layer. the features this network extracts are good enough to work with. the actual structure of the model is less important than the training process. this means the important part comes when you try to make the model learn the features you care about.

once you have a cnn you can build a style transfer network as follows: take your vgg16 network as the base of your model, and then add some convolutional layers that use transpose convolutions instead of regular ones. this will allow you to increase the spatial dimensions, or put another way, it will allow you to create images instead of classifying them.

you’ll need to create a loss function, that will tell the network how well is doing. here’s where the ‘art’ (pun intended) of deep learning meets. i suggest that you explore using a combination of these losses:

*   **content loss:** this makes sure that the structure of the input image is preserved in the generated output. we compare the feature maps from intermediate layers of the cnn between the input image and the generated output. we basically compare how similar the learned features are between both images.
*   **style loss:** this ensures that the resulting image has the same stylistic characteristics as your cartoon training set. we compare the gram matrices (statistical relations between features) of both the generated image and your cartoon images. this forces the model to learn the style.
*   **total variation loss:** this helps to reduce noise and artifacts in the output. it acts as a regularizer, smoothing out small unnecessary details in your picture.

and you need an optimiser, that helps the network learn. i've found that using adam works well in these cases. here is the relevant snippet code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        # load vgg16, but we do not want the fully connected part of the network
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:23])
        for param in self.features.parameters():
          param.requires_grad = False
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        features = self.features(x)
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

def gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (b * c * h * w)
def content_loss(features_gen, features_input):
    return torch.mean((features_gen - features_input)**2)
def style_loss(gram_gen, gram_style):
    return torch.mean((gram_gen - gram_style)**2)
def tv_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_transfer_model = StyleTransferNet().to(device)
    optimizer = optim.Adam(style_transfer_model.parameters(), lr=0.001)
    content_weight = 1
    style_weight = 50
    tv_weight = 0.01
    input_image = torch.rand((1, 3, 256, 256)).to(device)
    style_image = torch.rand((1, 3, 256, 256)).to(device)
    #load dataset as shown in the previous code, we assume that you have a dataloader called 'dataloader'
    for batch in dataloader:
      batch = batch.to(device)
      optimizer.zero_grad()
      generated_image = style_transfer_model(batch)

      input_features = style_transfer_model.features(batch)
      gen_features = style_transfer_model.features(generated_image)
      style_features = style_transfer_model.features(style_image)
      # compute the content loss, from the generated image and the input image
      c_loss = content_loss(gen_features, input_features)*content_weight
      # compute the style loss, from the generated image and a random style image from the dataset
      gram_gen = gram_matrix(gen_features)
      gram_style = gram_matrix(style_features)
      s_loss = style_loss(gram_gen, gram_style)*style_weight
      #compute the tv loss, for regularization purposes.
      tv = tv_loss(generated_image)*tv_weight
      #compute the final loss by summing the individual loses
      total_loss = c_loss + s_loss + tv
      total_loss.backward()
      optimizer.step()
      print(f"loss:{total_loss}")
      break # we only do one loop to showcase that the code runs

```

**training process**

train the model on your custom dataset. monitor the losses. it is normal to find your model produce noisy images at first. this is the most important step, and i can’t stress enough that you have to pay attention to the training process. in my experience, it can be frustrating to see a model converge to a noisy output and not to what you would like to get.

during training, keep track of the losses and evaluate them on a validation set, not the same set you trained with. if your losses increase on the validation set it means you are overfitting. meaning that your model is not learning general features, it is only learning the training images, and will be useless for any new image. if this happens reduce the complexity of the model. if it doesn’t learn the style, play with the loss function parameters, and be patient. learning takes time. you may need to tweak the learning rate as well, if the model converges too slowly or too quickly.

**resources**

there’s a lot to this, and it can be quite a deep dive, but some texts that i found particularly useful, when i was starting out and still find useful are:

*   "image style transfer using convolutional neural networks" by gatys et al. this is where it all started. a good read if you want to know the specifics of style transfer and the theory behind it.
*   "deep learning with python" by francois chollet. a good general resource, but with particular focus on deep learning and neural networks.
*   the pytorch official documentation. especially the tutorials section. this helped me a ton when i was trying to get a handle on the basics.

one last thing: when you present the results to other people be ready for all sort of opinions about the quality of the generated results. it is like trying to make your own food. some people will like it, some will not, and some will just be jerks about it. don't let it discourage you. the most important thing is that *you* like it.

i know this was a long answer, but there's a fair bit of detail to this. hope it helps you make your python based cartoon-inator. let me know if you have any more specific questions.
