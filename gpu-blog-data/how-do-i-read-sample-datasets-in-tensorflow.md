---
title: "How do I read sample datasets in TensorFlow?"
date: "2025-01-30"
id: "how-do-i-read-sample-datasets-in-tensorflow"
---
Working extensively with deep learning models requires a solid foundation in data input pipelines. TensorFlow, being a framework that prioritizes efficient computation, necessitates a specific approach to loading and managing datasets. I’ve found that directly working with large datasets by loading them entirely into memory is often impractical and computationally limiting, especially as I progressed in projects like image recognition and natural language processing where the datasets often exceed system RAM. Therefore, TensorFlow's `tf.data` API is crucial for efficient data ingestion, which enables streamed reading, shuffling, and pre-processing that doesn’t bog down system resources.

The core concept is using the `tf.data.Dataset` object, which represents a sequence of elements. These elements are typically tensors, but could also be more complex structures. Instead of loading everything at once, `tf.data.Dataset` allows you to create an iterator that yields data in batches, essentially a pipeline for your data. This significantly improves performance by enabling parallel loading and processing, and prevents out-of-memory errors when dealing with large files.

To create a `tf.data.Dataset`, you generally start with a source, such as file paths, or numpy arrays, and then apply transformations like batching, shuffling, and mapping functions. Let's break down specific examples.

**Example 1: Reading from NumPy arrays**

Consider a simplified scenario where my experimental setup involves training a basic classifier using synthetically generated numerical data. I have my input features and target labels readily available as NumPy arrays. Here is how I would efficiently set this up as a TensorFlow dataset:

```python
import tensorflow as tf
import numpy as np

# Simulate features (1000 samples, 5 features) and labels (1000 samples)
num_samples = 1000
num_features = 5
features = np.random.rand(num_samples, num_features).astype(np.float32)
labels = np.random.randint(0, 2, num_samples).astype(np.int32)

# Create a tf.data.Dataset from the numpy arrays.
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Batch the dataset for training
batch_size = 32
batched_dataset = dataset.batch(batch_size)

# Iterate through batches to verify.
for batch_features, batch_labels in batched_dataset.take(2):
  print("Features shape:", batch_features.shape)
  print("Labels shape:", batch_labels.shape)

```

The key here is the use of `tf.data.Dataset.from_tensor_slices`. This function converts our numpy arrays into a `Dataset` object, where each element corresponds to a single training sample (feature set and its associated label). I then batch the data using `.batch(batch_size)`, where 32 is the batch size I prefer for this example. This prepares the data for the iterative learning process in my model. The `.take(2)` method is included to only display information from the first two batches for demonstration purposes. This example illustrates how easy it is to ingest in-memory data for small datasets, and provides the base framework to start using the `tf.data` pipeline.

**Example 2: Reading from image files**

In a more realistic setting, I've often needed to work directly with image files, particularly for my image classification projects.  Consider a directory where all image files are stored with their corresponding labels specified in a separate text file or an easily accessible directory structure. Below is a strategy to ingest these images into a `tf.data.Dataset` for a model.

```python
import tensorflow as tf
import os

# Sample directory structure for demo purposes
image_dir = "sample_images" # Ensure the folder is created and contains sample images
labels_file = "sample_labels.txt"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
# Create dummy image files and labels file for demonstration.
# In reality, these would be real image and label files
for i in range(5):
    open(os.path.join(image_dir, f'img_{i}.jpg'),'a').close()

with open(labels_file, "w") as f:
    f.write(f"img_0.jpg 0\nimg_1.jpg 1\nimg_2.jpg 0\nimg_3.jpg 1\nimg_4.jpg 0\n")
    

# Function to parse image and label
def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [128, 128]) # Resize to a suitable input shape
    return image_resized, label

# Read image file paths and labels
filenames = []
labels = []
with open(labels_file, "r") as f:
  for line in f:
    filename, label = line.strip().split()
    filenames.append(os.path.join(image_dir, filename))
    labels.append(int(label))


# Create dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(parse_function) # Preprocess images on-the-fly
dataset = dataset.batch(3)

# Verify results.
for image_batch, label_batch in dataset.take(1):
  print("Image Batch shape:", image_batch.shape)
  print("Label batch shape:", label_batch.shape)

```

The `parse_function` is particularly crucial here. It takes a filename and its associated label, reads the file, decodes it as a JPEG, resizes it to a fixed size (128x128 in this case), and returns both the processed image tensor and the label. I used `tf.io.read_file` to load the image file, followed by `tf.io.decode_jpeg` to convert it into a tensor, and finally, `tf.image.resize` to ensure all images are of the same size. This is a vital step because most deep learning models expect fixed-size inputs. The key aspect is `dataset.map(parse_function)` which applies this preprocessing function to each item in the dataset during the iteration. This is significantly more efficient than loading and processing all images beforehand. This streamlined approach to data preprocessing enhances both memory usage and model training time.

**Example 3: Reading from a Text File**

Finally, I frequently work with text data, whether in the form of sequences or documents, for my NLP related tasks. Here is an example on reading textual data from a text file. This dataset can consist of sequences of text, where each sequence is associated with a corresponding label.

```python
import tensorflow as tf
import os
# Simulate a text file for this example.
text_file = "sample_text.txt"

with open(text_file, "w") as f:
  f.write("This is the first sentence.\t0\nThis is the second one.\t1\nAnd another.\t0\nMore text here.\t1\n")


def parse_text_function(line):
    line = tf.strings.split(line, sep='\t') # Split the line into text and label
    text = line[0]
    label = tf.strings.to_number(line[1], out_type=tf.int32) # Convert label to integer
    
    return text, label

dataset = tf.data.TextLineDataset(text_file)
dataset = dataset.map(parse_text_function)
dataset = dataset.batch(2)

for text_batch, label_batch in dataset.take(1):
    print("Text Batch:", text_batch.numpy())
    print("Label Batch:", label_batch.numpy())
```

Here, I use `tf.data.TextLineDataset`, which specifically reads the file line by line, treating each line as a separate example. The `parse_text_function` further processes each line by separating the text and the corresponding label (delimited by a tab). I use `tf.strings.split` to split the line into parts based on the tab delimiter and `tf.strings.to_number` to convert string label to an integer. Similar to the image dataset example, `map` allows me to transform the data on-the-fly. This workflow is particularly helpful when dealing with larger text files that would be inconvenient or unfeasible to load into memory completely.

In conclusion, the `tf.data` API is fundamental for creating efficient data input pipelines within TensorFlow. It allows you to process data in batches, perform on-the-fly preprocessing, and manage data of any size without overwhelming your system’s memory. This provides a flexible and highly efficient way of handling diverse types of data, whether numerical, image or textual. Utilizing these concepts, particularly with the `map` function, allows for a robust and scalable data pipeline for deep learning projects.

For deeper understanding, I recommend referring to the official TensorFlow documentation, focusing specifically on the `tf.data` module. There are also numerous online courses and tutorials that detail the use of `tf.data` which can be very helpful. Look for books and articles which delve into advanced usage including caching, prefetching and optimization strategies. These resources will help develop a complete mastery in data handling with TensorFlow and enable you to build scalable deep learning applications.
