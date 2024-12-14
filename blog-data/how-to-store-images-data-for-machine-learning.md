---
title: "How to store images data for machine learning?"
date: "2024-12-14"
id: "how-to-store-images-data-for-machine-learning"
---

so, you're asking about image data storage for machine learning, huh? i've been around the block a few times with this one, trust me. it's not as straightforward as it might seem at first glance. i’ve had my share of headaches trying to get it just *so*, and i can definitely share some of my scars… err, experience.

first off, forget about just dumping raw pixel data into a database. that's a recipe for disaster, and i've learned that the hard way. back in my early days, i was working on a project classifying types of, well, let's call them "widgets." i was naive and stored the raw images in a relational database as blobs. query performance was… atrocious. it took forever to load even a small batch of images for training, and i had to redesign the whole pipeline. i was practically rewriting my entire work when i realize i messed it up. lesson learned: databases aren't optimized for this sort of thing if you are not using specialised one.

so, what's the better approach? well, it depends on a few things, mainly your scale and your budget and what you actually need from your data. but let me break down some of the common strategies i've used and seen others use, and why they work. i'll include some code snippets in python, since it's what i usually use for machine learning stuff.

**1. simple file system storage (with some structure)**

this is the most common and, honestly, my go-to for most projects. you basically store your images as individual files on your file system, organizing them into folders based on their classes or some other meaningful category. it’s simple to setup, cheap, and fairly performant for most cases. here’s a simplified example of a folder structure:

```
images/
    cats/
        cat1.jpg
        cat2.png
        ...
    dogs/
        dog1.jpeg
        dog2.bmp
        ...
    birds/
        bird1.gif
        bird2.tiff
        ...
```

now, loading these images into your machine learning pipeline is straightforward. you can use libraries like `opencv` or `pillow` (the python imaging library) in python to easily load them. here is how you can do it with `pillow`:

```python
from PIL import Image
import os

def load_images_from_directory(directory_path):
    images = []
    labels = []
    for label_name in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label_name)
        if not os.path.isdir(label_path):
            continue
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            try:
                img = Image.open(image_path)
                # you could do some resizing or preprocessing here.
                images.append(img)
                labels.append(label_name)
            except Exception as e:
                print(f"error loading image: {image_path}, error:{e}")
    return images, labels


# using the function
image_dir = "images"  # change this to the path of your folder.
images, labels = load_images_from_directory(image_dir)

print(f"loaded {len(images)} images.")
```

this way, you can easily load your images in batches and process them, feeding them into your model. the key thing to remember here is that the file system is *designed* for this – it’s good at storing and retrieving files efficiently. no surprises with it.

**2. using a specific format for fast loading: hdf5**

when you start dealing with large datasets, the io overhead of loading individual image files can add up, really fast. i remember working on a project with thousands of high-resolution satellite images. it was painfully slow loading the data. that's where hdf5 comes in.

hdf5 (hierarchical data format version 5) is a binary file format designed for storing large amounts of numerical data. it's particularly suitable for storing multi-dimensional arrays, which, as you guessed it, images essentially are.

you can convert your images into numpy arrays and save them in an hdf5 file, along with their labels. that way, all of your image data is neatly packed into one single file. here’s a snippet of how you can save a set of images and labels into an hdf5 file. you will need to install numpy and h5py: `pip install numpy h5py`:

```python
import h5py
import numpy as np
from PIL import Image
import os

def create_hdf5_from_images(directory_path, hdf5_file_path):
    images = []
    labels = []
    for label_name in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label_name)
        if not os.path.isdir(label_path):
            continue
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            try:
                img = Image.open(image_path)
                img_array = np.array(img) #converting the image to a numpy array.
                images.append(img_array)
                labels.append(label_name)
            except Exception as e:
                print(f"error loading image: {image_path}, error:{e}")
    
    images = np.array(images)
    labels = np.array(labels, dtype='S') #string dtype for storing labels in hdf5
    
    with h5py.File(hdf5_file_path, 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=labels)
        
        
image_dir = "images" # change this to the path of your folder.
hdf5_file = "image_data.hdf5"

create_hdf5_from_images(image_dir, hdf5_file)
print(f"hdf5 file created successfully on: {hdf5_file}")
```

loading from an hdf5 is much faster, especially if you do it in chunks rather than loading the entire dataset at once, which you should avoid if possible. here is a snippet of how to load the images in chunks:

```python
import h5py
import numpy as np

def load_hdf5_in_batches(hdf5_file_path, batch_size):
    with h5py.File(hdf5_file_path, 'r') as hf:
        images = hf['images']
        labels = hf['labels']
        
        num_samples = images.shape[0]
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            yield batch_images, batch_labels

# usage
hdf5_file = "image_data.hdf5"
batch_size = 32 #you can change the batch size

for images, labels in load_hdf5_in_batches(hdf5_file, batch_size):
    print(f"loaded a batch of {images.shape[0]} images, labels are {labels}")
    # do stuff with this batch.

```

that code reads the image in batches avoiding memory issues. you might need more than one hdf5 file if your dataset is huge.

**3. cloud storage and object stores**

if you’re working at scale, or if you’re on a team that needs to share data easily, cloud storage solutions like aws s3, google cloud storage, or azure blob storage are good options. these services are highly scalable, reliable, and are designed to handle large amounts of data.

the data is stored as objects, and you can use the cloud provider’s libraries to read the images directly into your machine learning pipeline. it essentially works the same as a regular filesystem in terms of organization but with extra goodies like version control and permissions. for example, you might use boto3 to access s3 in aws or google-cloud-storage to access google cloud storage in python.

it’s a bit more involved to set up, but it is really useful when collaborating. also most cloud machine learning platforms are really well integrated with their cloud storage solutions, that's why if you are using their machine learning service that is good to have the data there. i always recommend to use this approach when working in teams or when your dataset is large.

**important considerations**

no matter what storage method you pick, keep these things in mind:

*   **preprocessing:** always pre-process your data consistently. resizing, normalization, and other transformations should be done at some point of your data pipeline. it’s better to do it when loading data as part of your loading process or when transforming your data to hdf5, depending on the scenario.
*   **data versioning:** when working with lots of images over a project timeline, tracking changes is important, so you know which version of data was used for training each model.
*  **data augmentation:** some data augmentation should be done on the fly during training. libraries like tensorflow or pytorch have tools to do that. that avoid pre-generating more data and keep your storage cleaner.
*   **data validation:** always validate your images, make sure you are loading what you expect, and handle errors when loading corrupt images. it happens a lot more than what people think!

**more resources**

if you want to really delve deeper into this, i'd recommend checking out:

*   "deep learning with python" by francois chollet for a detailed look at data preprocessing.
*   the official documentation of libraries like `h5py` and your cloud provider's storage apis for the specifics on storing and loading large datasets. also, check out the documentation of `opencv` and `pillow`, i know i have been talking about how to use them.

and that’s about it, i've had my share of problems with this, and i think these guidelines will help you avoid a lot of issues with how you store your images. just remember to keep it simple at first, and scale your solution as needed. oh, and by the way, why did the image get a promotion? because it was outstanding in its field… a bad joke i know, but i had to add one to fulfill your requirements. anyway, hope this helps you, feel free to ask any more questions if you have them.
