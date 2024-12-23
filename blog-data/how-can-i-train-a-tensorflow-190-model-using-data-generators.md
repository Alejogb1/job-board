---
title: "How can I train a TensorFlow 1.9.0 model using data generators?"
date: "2024-12-23"
id: "how-can-i-train-a-tensorflow-190-model-using-data-generators"
---

Ah, data generators. A topic near and dear to my heart, having spent a good chunk of time optimizing training pipelines in, let's say, 2018, when tensorflow 1.9.0 was the tool of the day. Back then, before tf.data was quite as robust as it is now, data generators were often a crucial component for managing large datasets that couldn't realistically fit into memory. Let’s dive into how we can achieve this, focusing specifically on that older tensorflow version, which does present some unique considerations.

The core issue is feeding data into your model efficiently. Traditional methods, where you’d load all your data into a numpy array and feed it in one go, quickly become impractical when dealing with significant datasets. This is where generators step in. They act as iterators, producing batches of data on demand, which eliminates the need to load the entire dataset into memory. It's all about conserving resources and enabling you to train on much larger datasets than your ram might initially suggest.

Using a data generator in tensorflow 1.9.0 involves defining a python function that yields batches of data, and then hooking that into your model's training process. This function could perform any necessary preprocessing, like data augmentation or normalization, on the fly. We'll explore different types of generators, from basic ones to ones that incorporate custom logic, to cover several scenarios.

Let’s illustrate with a basic example. Imagine we're working with images for a classification task. We'll start with a simple generator that loads images from a directory. Note that in tensorflow 1.9.0, you typically had to use placeholders to pass the batch data.

```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image  # pip install Pillow if needed


def image_data_generator(image_dir, batch_size, image_size=(128, 128)):
    """
    A simple generator for loading images from a directory.

    Args:
        image_dir (str): Path to the directory containing image files.
        batch_size (int): Number of images per batch.
        image_size (tuple): Desired image dimensions (height, width).

    Yields:
        tuple: A tuple containing image batch and corresponding label batch
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_paths)
    labels = np.zeros(num_images) #Placeholder labels for the example, make it realistic
    num_classes = 2
    
    
    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).resize(image_size)
                img_array = np.array(img).astype(np.float32) / 255.0 #Normalization added
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue #Skip corrupt images

        #One hot encoding for placeholder labels 
        batch_labels = labels[i:i + batch_size]
        one_hot_labels = np.eye(num_classes)[batch_labels.astype(int)] #Convert to one-hot if needed
        
        if batch_images: #yield batch if we managed to load at least one image.
            yield np.array(batch_images), one_hot_labels



# Example of how to use this generator in the training process.
if __name__ == "__main__":
    # Create a dummy image directory and some dummy images for testing.
    dummy_dir = "dummy_images"
    os.makedirs(dummy_dir, exist_ok=True)
    for i in range(5):  # creating 5 test images
      dummy_image = Image.new('RGB', (200, 200), color = (i * 50, i*40, i*30)) #Creates different colored dummy images
      dummy_image.save(f"{dummy_dir}/dummy_image_{i}.jpg")

    batch_size = 2
    image_size = (128, 128)

    #Setup Placeholders
    X = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3], name="input")
    Y = tf.placeholder(tf.float32, [None, 2], name="labels")

    # Basic Model (using random values for weights and biases for simplicity).
    W = tf.Variable(tf.random_normal([image_size[0] * image_size[1] * 3, 2], stddev=0.01))
    b = tf.Variable(tf.zeros([2]))
    
    flat_X = tf.reshape(X, [-1, image_size[0] * image_size[1] * 3])
    pred = tf.nn.softmax(tf.matmul(flat_X, W) + b) #Simple linear classification
    
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        generator = image_data_generator(dummy_dir, batch_size, image_size)
        epochs = 2
        for epoch in range(epochs):
            for batch_images, batch_labels in generator:
               _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_images, Y: batch_labels})
               print(f"Epoch {epoch}, Loss: {loss_val}")
    
    #Clean up directory
    for filename in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, filename))
    os.rmdir(dummy_dir) #removing dummy dir created
```

This first code demonstrates the basic principles. It defines a generator that yields batches of images loaded from a given directory. Each batch is preprocessed (scaled to the range 0-1) before being yielded. I've added a rudimentary dummy model to demonstrate how you can use the generator to feed data into the training process via `feed_dict`. I included a basic error handling within the image load process, a practice I highly suggest implementing. For real projects you would need more robust label loading, which could involve reading from json or csv files.

Now let's add some complexity. Suppose your dataset includes not just images but also corresponding numerical features and you want to use both for your model. We need a more sophisticated generator that combines these data sources and performs any necessary transforms.
```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image  # pip install Pillow if needed
import csv

def combined_data_generator(image_dir, feature_file, batch_size, image_size=(128, 128)):
    """
    A generator that combines images with numerical features.
    
    Args:
        image_dir (str): Path to the image directory
        feature_file (str): Path to the CSV file containing features and labels
        batch_size (int): Number of samples per batch
        image_size (tuple): Desired image dimensions (height, width)

    Yields:
        tuple: A tuple containing (image batch, numerical feature batch, label batch)
    """
    
    #Load features and labels from the CSV
    data_dict = {} # key is image name, value is [features, label]
    with open(feature_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader) #Skip header
      for row in reader:
        image_name = row[0]
        features = np.array(row[1:-1],dtype=np.float32)
        label = int(row[-1]) #Assuming last is label, make sure this matches actual format
        data_dict[image_name] = [features, label]

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_paths)
    num_classes = 2

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_features = []
        batch_labels = []
        
        for path in batch_paths:
            try:
                img_name = os.path.basename(path)
                if img_name not in data_dict:
                  print(f"Missing feature or label information for {img_name}")
                  continue
                  
                img = Image.open(path).resize(image_size)
                img_array = np.array(img).astype(np.float32) / 255.0
                features, label = data_dict[img_name]
                batch_images.append(img_array)
                batch_features.append(features)
                batch_labels.append(label)
            except Exception as e:
                print(f"Error loading image or feature for {path}: {e}")
                continue
        if batch_images:
           batch_labels_encoded = np.eye(num_classes)[np.array(batch_labels)] # one-hot
           yield np.array(batch_images), np.array(batch_features), batch_labels_encoded


if __name__ == "__main__":
    # Create a dummy image directory and some dummy images for testing.
    dummy_dir = "dummy_images_2"
    os.makedirs(dummy_dir, exist_ok=True)
    for i in range(5):  # creating 5 test images
        dummy_image = Image.new('RGB', (200, 200), color = (i * 50, i*40, i*30))
        dummy_image.save(f"{dummy_dir}/dummy_image_{i}.jpg")
    
    #Create dummy CSV file
    dummy_feature_file = "dummy_features.csv"
    with open(dummy_feature_file, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['image_name','feature1','feature2', 'label']) #Header
      for i in range(5):
          writer.writerow([f'dummy_image_{i}.jpg', i, i*2, i % 2]) #Dummy features and labels

    batch_size = 2
    image_size = (128, 128)


    # Setup Placeholders
    X = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3], name="input_image")
    F = tf.placeholder(tf.float32, [None, 2], name="input_features") #Dummy Features
    Y = tf.placeholder(tf.float32, [None, 2], name="labels")

    # Basic Model (Illustrative): Using concatenation of image and features
    flat_X = tf.reshape(X, [-1, image_size[0] * image_size[1] * 3])
    combined_input = tf.concat([flat_X, F], axis=1)

    W = tf.Variable(tf.random_normal([image_size[0] * image_size[1] * 3 + 2, 2], stddev=0.01)) #Adjusted for concatenated size
    b = tf.Variable(tf.zeros([2]))
    pred = tf.nn.softmax(tf.matmul(combined_input, W) + b) 
    
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        generator = combined_data_generator(dummy_dir, dummy_feature_file, batch_size, image_size)
        epochs = 2
        for epoch in range(epochs):
            for batch_images, batch_features, batch_labels in generator:
                _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_images, F: batch_features, Y: batch_labels})
                print(f"Epoch {epoch}, Loss: {loss_val}")
                
    # Clean up directory
    for filename in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, filename))
    os.rmdir(dummy_dir) #removing dummy dir created
    os.remove(dummy_feature_file)
```
Here we load numerical feature and label data from a csv file, linking them to the corresponding image file based on the filename. These additional features and labels are loaded alongside the images, and the generator yields a tuple of images, features and labels. The training loop is updated to use all three data elements. This demonstrates a generator designed to handle more complex and varied data sources.

Finally, consider augmentation. You often want to introduce variability into your training data to improve generalization. The next example extends our previous one to include basic image augmentations:

```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageEnhance # pip install Pillow if needed
import random
import csv


def augment_image(img):
    """Randomly apply augmentations to an image"""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))

    return img
    
def augmented_combined_data_generator(image_dir, feature_file, batch_size, image_size=(128, 128)):
    """
    A generator that combines images with numerical features and applies augmentations.

    Args:
        image_dir (str): Path to the image directory
        feature_file (str): Path to the CSV file containing features and labels
        batch_size (int): Number of samples per batch
        image_size (tuple): Desired image dimensions (height, width)

    Yields:
        tuple: A tuple containing (image batch, numerical feature batch, label batch)
    """
    #Load features and labels from the CSV
    data_dict = {} # key is image name, value is [features, label]
    with open(feature_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      next(reader) #Skip header
      for row in reader:
        image_name = row[0]
        features = np.array(row[1:-1],dtype=np.float32)
        label = int(row[-1])
        data_dict[image_name] = [features, label]

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_paths)
    num_classes = 2

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_features = []
        batch_labels = []
        
        for path in batch_paths:
           try:
                img_name = os.path.basename(path)
                if img_name not in data_dict:
                  print(f"Missing feature or label information for {img_name}")
                  continue
                img = Image.open(path).resize(image_size)
                img = augment_image(img) #Augmentation applied here
                img_array = np.array(img).astype(np.float32) / 255.0
                features, label = data_dict[img_name]
                batch_images.append(img_array)
                batch_features.append(features)
                batch_labels.append(label)
           except Exception as e:
                print(f"Error loading image or feature for {path}: {e}")
                continue
        if batch_images:
            batch_labels_encoded = np.eye(num_classes)[np.array(batch_labels)] # one-hot
            yield np.array(batch_images), np.array(batch_features), batch_labels_encoded

if __name__ == "__main__":
    # Create a dummy image directory and some dummy images for testing.
    dummy_dir = "dummy_images_3"
    os.makedirs(dummy_dir, exist_ok=True)
    for i in range(5):  # creating 5 test images
      dummy_image = Image.new('RGB', (200, 200), color = (i * 50, i*40, i*30))
      dummy_image.save(f"{dummy_dir}/dummy_image_{i}.jpg")
    
    #Create dummy CSV file
    dummy_feature_file = "dummy_features.csv"
    with open(dummy_feature_file, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['image_name','feature1','feature2', 'label']) #Header
      for i in range(5):
          writer.writerow([f'dummy_image_{i}.jpg', i, i*2, i % 2]) #Dummy features and labels

    batch_size = 2
    image_size = (128, 128)

    # Setup Placeholders
    X = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3], name="input_image")
    F = tf.placeholder(tf.float32, [None, 2], name="input_features") #Dummy Features
    Y = tf.placeholder(tf.float32, [None, 2], name="labels")

    # Basic Model (Illustrative): Using concatenation of image and features
    flat_X = tf.reshape(X, [-1, image_size[0] * image_size[1] * 3])
    combined_input = tf.concat([flat_X, F], axis=1)

    W = tf.Variable(tf.random_normal([image_size[0] * image_size[1] * 3 + 2, 2], stddev=0.01)) #Adjusted for concatenated size
    b = tf.Variable(tf.zeros([2]))
    pred = tf.nn.softmax(tf.matmul(combined_input, W) + b) 

    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        generator = augmented_combined_data_generator(dummy_dir, dummy_feature_file, batch_size, image_size)
        epochs = 2
        for epoch in range(epochs):
           for batch_images, batch_features, batch_labels in generator:
               _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_images, F: batch_features, Y: batch_labels})
               print(f"Epoch {epoch}, Loss: {loss_val}")
                
    # Clean up directory
    for filename in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, filename))
    os.rmdir(dummy_dir) #removing dummy dir created
    os.remove(dummy_feature_file)
```
This final code builds on the last two and incorporates simple image augmentation techniques – flipping and slight brightness/contrast adjustments – using pillow. These are performed within the generator itself, meaning each epoch sees different variations of the original training data, which can help improve your model’s generalization performance.

For more on efficient data pipelines, especially with tensorflow, I’d recommend looking into the details of the `tf.data` API, even if it’s from versions newer than 1.9.0. The underlying principles of data loading and preprocessing are the same, and the performance improvements that tf.data brings are noteworthy, even if you adapt the concepts for generators. The book *Deep Learning with Python* by François Chollet is also extremely useful in understanding these concepts at a high level and shows how to implement data generators using keras in a clear and concise way. Another great resource is the original tensorflow documentation, even for previous versions, as it contains information regarding best practices for specific versions. Finally, academic publications focused on data loading, augmentation, and their impact on neural network training can offer deeper insight into the methods and rationale behind efficient data pipelines.

Working with tensorflow 1.9.0 and data generators, it's clear that careful management of data input is crucial. Although the technology has moved on, these underlying principles, and understanding of how generators can keep training efficient, are valuable for anyone working with machine learning, no matter the framework.
