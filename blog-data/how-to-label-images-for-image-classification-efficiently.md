---
title: "how to label images for image classification efficiently?"
date: "2024-12-13"
id: "how-to-label-images-for-image-classification-efficiently"
---

Alright so you're asking about efficient image labeling for image classification yeah I get it Been there done that got the t-shirt And by t-shirt I mean a thousand labeled images and some seriously tired fingers It’s like a right of passage in this field isn’t it I guess if you're serious about training a decent image classifier you know you gotta label your data properly and fast Now fast is relative right but I’m talking about avoiding the soul-crushing task of manually clicking through millions of images one by one That’s not living that’s just existing so let’s get to it

First off let’s talk basics you're gonna need to get organized like seriously organized You're not doing this haphazardly I tried that once ended up with a confusing mess of folders and labels that no one could understand not even me I swear I thought I was good at organizing things but I was very very wrong Lesson learned the hard way

Before you even think about labeling get your file structure straight I'm talking dedicated folders per class This is ground zero for any decent labeling workflow. Imagine trying to find a specific sock in a huge laundry pile. That's how you're going to feel when trying to find that "cat" image among 10.000 unlabeled images So like this:

```
data/
    class_a/
        image_1.jpg
        image_2.png
        ...
    class_b/
        image_3.jpg
        image_4.jpeg
        ...
   class_c/
        image_5.png
        ...
```
See how simple it is That’s like the most important first step no joke I mean seriously it’s crucial

Now you've got your folders sorted that's like half the battle If I'm being honest and I usually am labeling at its core is tedious I mean yeah we can talk about AI assisted labeling but you'll still need to oversee its labeling so it doesn't go crazy and label a dog as a car I mean we are not there yet Now for your labeling tools there are many out there I've had some success using LabelImg it’s not the fanciest thing out there but it's a good basic annotation tool that works on Linux Windows and Mac and that's all you need you don't need the latest cutting edge AI powered tool that costs an arm and a leg trust me I've used the super expensive ones and for the most part they are just hype for labeling a small image classification dataset you just don't need them And if you are into more web-based solutions I've tried CVAT it can be a bit more complicated to set up but it's robust especially if you are collaborating with others

Now the actual labeling part if you're working with bounding boxes that is a bit more time consuming for every image you need to draw a box around the thing you’re trying to classify. If you are working with classification with no bounding box required then you just choose the folder of the object to place the image in that is it easy no complications simple and effective.
Here is a quick and very simple python code if you are trying to do basic image classification with no bounding box required. This code is for when you are done with the image labeling and you are ready to load your dataset

```python
import os
from PIL import Image
import numpy as np

def load_images(data_dir):
    images = []
    labels = []
    label_map = {}
    label_count = 0
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in label_map:
          label_map[class_name] = label_count
          label_count +=1

        for image_name in os.listdir(class_dir):
          if not image_name.lower().endswith(('.png','.jpg','.jpeg')):
            continue
          image_path = os.path.join(class_dir,image_name)
          try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            images.append(image)
            labels.append(label_map[class_name])
          except Exception as e:
            print(f"Could not load image {image_name} error: {e}")
            continue
    return np.array(images), np.array(labels), label_map

data_dir = "data/"
images, labels , label_map = load_images(data_dir)
print (f"images shape : {images.shape}")
print (f"labels shape : {labels.shape}")
print (f"label map {label_map}")

```
That script is very simple but it will give you the gist of how data loading should be done with minimal code and dependencies This approach is scalable even if you have thousands of image classes

Here's the deal even with good tools labeling can be repetitive. That's where active learning comes in Active learning? Its when you train a classifier with small amounts of labeled data then you give it unlabeled data and it will predict the class Then you inspect the predictions and only label the data that the algorithm was unsure about or it predicted wrong you save so much time doing it this way trust me it’s amazing! This helps focus your energy on the most ambiguous images rather than labeling everything which is frankly a waste of resources and time especially if you have a large dataset I've been using this technique in many of my projects and it has saved me an insane amount of time. I'm not going to lie that stuff can be complicated so I am going to focus on basic manual labeling

One trick I've used is to create a "maybe" folder I don't label it right away just put images that I am unsure about there later on after a big break I can look at them with fresh eyes It’s a little hack that has saved my sanity I remember staring at a picture of a dog that kinda looked like a wolf thinking is that a dog or a wolf? It was a stressful time believe me the "maybe" folder will save your sanity.

Now let's say you want to automate a bit more than just the active learning part. There are some really cool techniques to explore. You might want to look into weakly supervised learning. The idea is that instead of labeling every single pixel or object you label the data at a lower resolution for example instead of bounding box you label the image as belonging to a specific class and then some algorithms can learn the features based on the image class and find the object This is a powerful technique for speeding up data labeling but it will require a little bit more work. Also remember if you have multiple people working on the data labeling always have a style guide a very basic one for consistency this is important believe me inconsistent data labeling will ruin your model.

Now for another example this is for when you are done with labeling and you want to split your dataset to training testing and validation. It is important that the images are split evenly this ensures that your models perform better in the real world than in testing here is an example:

```python
import os
import random
import shutil

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if (train_ratio + val_ratio + test_ratio) != 1:
      raise ValueError("Train val and test ratio must sum up to 1")
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = [image_name for image_name in os.listdir(class_dir) if image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        num_images = len(images)
        train_size = int(num_images * train_ratio)
        val_size = int(num_images * val_ratio)
        test_size = num_images - train_size - val_size

        train_images = images[:train_size]
        val_images = images[train_size : train_size+val_size]
        test_images = images[train_size+val_size:]

        os.makedirs(os.path.join(data_dir,'train',class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir,'val',class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir,'test',class_name), exist_ok=True)


        for image_name in train_images:
          shutil.copy(os.path.join(class_dir,image_name), os.path.join(data_dir,'train',class_name,image_name))
        for image_name in val_images:
          shutil.copy(os.path.join(class_dir,image_name), os.path.join(data_dir,'val',class_name,image_name))
        for image_name in test_images:
          shutil.copy(os.path.join(class_dir,image_name), os.path.join(data_dir,'test',class_name,image_name))


data_dir = "data/"
split_dataset(data_dir)
print ("dataset splitting is done")

```
That script creates the folder "train" "val" and "test" and it moves the images based on the ratios provided by you

If you want to dive deeper into image classification and stuff like active learning or weakly supervised learning check out "Deep Learning with Python" by François Chollet and the "Pattern Recognition and Machine Learning" book by Christopher Bishop those are the textbooks that have been my bible for the last few years those books will give you a very solid ground on the field. Also if you are into more advanced topics like active learning research papers are your best friend no really there is some amazing stuff out there.

Alright that's about it I think I've covered the main points on labeling images for classification efficiently remember organization is key automation can help and if you get tired just take a break its important you don’t want to label a dog as a cat or worse a cat as a dog. Data labeling can be a very tedious task but it is a necessary one if you want to train a decent machine learning algorithm
