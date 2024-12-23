---
title: "How can I create a suitable dataset for a convolutional neural network?"
date: "2024-12-23"
id: "how-can-i-create-a-suitable-dataset-for-a-convolutional-neural-network"
---

Let's tackle this one. I remember back in '18, when I was working on an aerial image classification project, the dataset preparation consumed more time than the actual model training. It’s a classic issue, and it's important to get it right. The effectiveness of any convolutional neural network (CNN) hinges significantly on the quality and suitability of the dataset it’s trained on. It's not just about volume; it’s about the information contained within and how well it’s organized.

First off, let's acknowledge that "suitable" is a context-dependent term. What works well for satellite imagery might be completely inadequate for, say, medical imaging or natural language processing (though technically, you *can* use a CNN for NLP, that’s another topic). For the sake of this discussion, we'll assume you're dealing with a fairly standard image classification or object detection task, where CNNs are most commonly applied.

Building a strong dataset isn't just about collecting random images; it's a methodical process involving several key steps: data acquisition, cleaning, annotation, augmentation, and finally, organization. Each step is crucial, and overlooking any of them can degrade the model’s performance.

Data acquisition is the initial hurdle. Where do you source your images? Sometimes, you might have an existing archive, other times you need to scrape the web (carefully and ethically, of course), or even set up cameras to collect the data yourself. The method affects the initial quality, so plan carefully. For example, in my previous project, the aerial imagery was captured by different sensors and altitudes, meaning a substantial pre-processing effort was necessary to normalize the data before it could even be labelled.

Cleaning is the next essential task, focusing on removing irrelevant images or corrupt entries. For instance, if you’re training a model to recognize cats, images containing only text or extremely low-quality photos need to be removed. This step also involves ensuring consistency in formats, like converting all images to a uniform size, colour space (like RGB), and file format (PNG or JPG for example). Having varied formats can cause issues later during training when batching images from diverse sources.

Then comes annotation, perhaps the most laborious, yet fundamental stage, particularly if you require supervised learning. This is where you label each image according to the classes or bounding boxes you intend for your CNN to learn. The label quality matters as much as the data; inaccurate or inconsistent annotation will severely affect the model's training accuracy. I recall working on a facial recognition project and we had to use multiple annotators and detailed guidelines to minimize label discrepancies and inconsistencies between the images.

Following the annotation, we come to data augmentation. A technique to increase dataset diversity artificially, augmentation involves modifying existing images through processes like rotation, zooming, cropping, flipping, color adjustments, adding noise, and more. It’s not only a technique to create a bigger dataset when the initial collection is limited; it also helps to improve the model’s generalization capabilities, meaning the model will perform better on data it has not seen before.

Finally, all this preprocessed data must be organized. A typical setup uses separate directories for training, validation, and testing. A good split ensures that the model isn’t trained and evaluated on the same data points. Typical splits can be 70/20/10 or 80/10/10 but this can vary based on the data size and the problem context. It is also important to maintain a balance between classes; if one class has significantly more examples than others, your model is likely to exhibit bias towards the more represented class.

Let's look at some code to exemplify these steps.

**Example 1: Image Resizing and Conversion**

This snippet showcases basic image pre-processing— resizing and converting to a specified format, in this case, all to the same target size and saved as a PNG:

```python
import os
from PIL import Image

def preprocess_images(image_dir, target_size=(256, 256)):
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path)
                img = img.resize(target_size, Image.LANCZOS) # using lanczos method, for a high quality resize.
                img = img.convert("RGB") # ensuring consistent colorspace
                new_filename = os.path.splitext(filename)[0] + ".png"
                img.save(os.path.join(image_dir, new_filename)) # Save the processed image
                if filename != new_filename: # avoid erasing the original if same extension.
                    os.remove(img_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
image_dir = "path/to/your/images"  # Replace with your directory
preprocess_images(image_dir)
```
This python code iterates through all the specified image directory, resizing every image to a target size and ensuring all the images are in RGB, saving them in PNG format. Error handling is included to capture issues during processing.

**Example 2: Data Augmentation using Keras**

Here is a snippet that uses `ImageDataGenerator` from keras to perform image augmentation.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_augmented_dataset(image_dir, target_dir, augmentation_parameters):
    datagen = ImageDataGenerator(**augmentation_parameters)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(image_dir, filename)
                img = tf.keras.utils.load_img(img_path)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                i = 0
                for batch in datagen.flow(img_array, batch_size=1, save_to_dir=target_dir, save_prefix=filename.split('.')[0], save_format='png'):
                     i += 1
                     if i > 5:  # Generate 5 augmented images per input image
                        break
            except Exception as e:
                 print(f"Error processing {filename}: {e}")

# Example parameters
augmentation_params = {
    'rotation_range': 40,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

image_dir = "path/to/your/original/images" # Replace with the path to your original images
target_dir = "path/to/your/augmented/images" # Replace with the desired directory for augmented images

create_augmented_dataset(image_dir, target_dir, augmentation_params)
```

This code takes the original image and uses parameters for rotation, shifting, shearing, zooming, flipping and filling mode to create augmented data in a specified target directory.

**Example 3: Data Splitting into Training, Validation, and Testing Sets**
This code organises the images into train, validation and test directories.
```python
import os
import shutil
import random

def split_dataset(image_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    if not os.path.exists(train_dir):
       os.makedirs(train_dir)
    if not os.path.exists(val_dir):
       os.makedirs(val_dir)
    if not os.path.exists(test_dir):
       os.makedirs(test_dir)

    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)
    train_size = int(len(all_images) * split_ratio[0])
    val_size = int(len(all_images) * split_ratio[1])
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size+val_size:]


    for filename in train_images:
        shutil.copy(os.path.join(image_dir, filename), os.path.join(train_dir,filename))
    for filename in val_images:
        shutil.copy(os.path.join(image_dir,filename), os.path.join(val_dir,filename))
    for filename in test_images:
        shutil.copy(os.path.join(image_dir, filename), os.path.join(test_dir, filename))

# Define your path's
image_dir = "path/to/your/preprocessed/images" # Replace with your directory
train_dir = "path/to/your/train/directory" # Replace with the path where training data should reside
val_dir = "path/to/your/val/directory" # Replace with the path where validation data should reside
test_dir = "path/to/your/test/directory" # Replace with the path where testing data should reside

split_dataset(image_dir, train_dir, val_dir, test_dir)
```

This code randomly shuffles all the available images and splits them into training, validation and test directories based on the splitting ratio. The function makes sure to create these directories if they do not exist.

For more in-depth knowledge, I recommend exploring these resources:
*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** This book is an essential textbook on deep learning, covering the theoretical foundations and practical aspects. The chapters on data preprocessing are very helpful.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This practical book provides a hands-on guide to implementing machine learning projects, including detailed sections on data preparation and augmentation using Keras.
*  **Research papers on data augmentation techniques:** Look for papers that explore advanced augmentation methods like Mixup, CutMix, and others, particularly those that are effective for convolutional neural networks. They can help you understand the latest best practices and techniques to incorporate into your data preparation pipeline.

Remember, a robust dataset is the cornerstone of a successful CNN model. A data-centric approach, focusing on meticulously crafted datasets, will often result in more significant improvements than fine-tuning the model’s architecture. The details of dataset creation are often overlooked, and these details make all the difference in practice.
